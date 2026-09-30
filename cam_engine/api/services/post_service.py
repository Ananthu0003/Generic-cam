"""Post-processing and G-code export service.

Contains domain logic for G-code generation, per-setup downloads, and
the full CAM-package ZIP export.  Pure Python — no FastAPI coupling.
"""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING

from ...post import get_post_processor
from ..schemas import PostProcessRequest, StockConfig
from .context_service import build_context_and_setups
from .toolpath_service import generate_toolpaths_for_session

if TYPE_CHECKING:
    from ..state import SessionState


def generate_gcode_for_session(
    sess: "SessionState",
    req: PostProcessRequest,
) -> dict:
    """Generate combined (or per-setup) G-code and return a result dict.

    Automatically triggers toolpath generation if not yet done.
    """
    if not sess.toolpaths:
        generate_toolpaths_for_session(sess, sess.stock_config)

    # Collect ordered unique setup IDs from toolpaths
    unique_setup_ids: list[str] = []
    for tp in sess.toolpaths:
        sid = tp.metadata.get("setup_id", "setup_001")
        if sid not in unique_setup_ids:
            unique_setup_ids.append(sid)

    # Per-setup separate files
    if req.separate_files:
        return _generate_per_setup(sess, req, unique_setup_ids)

    # Combined G-code
    return _generate_combined(sess, req, unique_setup_ids)


def _generate_per_setup(
    sess: "SessionState",
    req: PostProcessRequest,
    unique_setup_ids: list[str],
) -> dict:
    setups_data: dict[str, dict] = {}
    total_time = 0.0

    for i, sid in enumerate(unique_setup_ids):
        setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id", "setup_001") == sid]
        if not setup_tps:
            continue
        matching_setup = next((s for s in sess.setups if s.id == sid), None)
        wo = matching_setup.work_offset if matching_setup else req.work_offset
        sname = matching_setup.name if matching_setup else f"Setup {i + 1}"
        post = get_post_processor(controller=req.controller, work_offset=wo)
        result = post.post_process(setup_tps, program_name=f"{req.program_name}_{sid}")
        setups_data[sid] = {
            "gcode": result.gcode,
            "line_count": result.line_count,
            "cycle_time_seconds": result.total_time_seconds,
            "work_offset": wo,
            "name": sname,
            "tool_changes": result.tool_changes,
            "rapid_dist_mm": result.total_rapid_dist_mm,
            "cut_dist_mm": result.total_cut_dist_mm,
        }
        total_time += result.total_time_seconds

    return {
        "separate_files": True,
        "controller": req.controller,
        "setups": setups_data,
        "total_cycle_time_seconds": total_time,
    }


def _generate_combined(
    sess: "SessionState",
    req: PostProcessRequest,
    unique_setup_ids: list[str],
) -> dict:
    gcode_blocks: list[str] = []
    total_time = rapid_time = cut_time = total_rapid_dist = total_cut_dist = 0.0
    all_tools: list[str] = []
    setups_processed: list[str] = []
    per_setup_data: dict[str, dict] = {}

    for i, sid in enumerate(unique_setup_ids):
        setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id", "setup_001") == sid]
        if not setup_tps:
            continue

        matching_setup = next((s for s in sess.setups if s.id == sid), None)
        wo = matching_setup.work_offset if matching_setup else req.work_offset
        sname = matching_setup.name if matching_setup else f"Setup {i + 1}"
        setups_processed.append(f"{sname} ({wo})")

        post = get_post_processor(controller=req.controller, work_offset=wo)
        result = post.post_process(setup_tps, program_name=f"{req.program_name}_{sid}")

        per_setup_data[sid] = {
            "gcode": result.gcode,
            "line_count": result.line_count,
            "cycle_time_seconds": result.total_time_seconds,
            "work_offset": wo,
            "name": sname,
            "tool_changes": result.tool_changes,
            "rapid_dist_mm": result.total_rapid_dist_mm,
            "cut_dist_mm": result.total_cut_dist_mm,
        }

        if i > 0:
            retract_cmd = "G53 G00 Z0." if req.controller in ("fanuc", "haas") else "G28 G91 Z0.\nG90"
            msg_cmd = (
                f"(MSG, FLIP PART - LOAD {sname.upper()} [{wo}])"
                if req.controller == "haas"
                else f"({sname.upper()} - FLIP / RE-FIXTURE PART)"
            )
            gcode_blocks.append(
                f"\n; ==========================================\n"
                f"; (--- {sname.upper()} - FLIP / RE-FIXTURE PART ---)\n"
                f"; ==========================================\n"
                f"M05 (Spindle Stop)\nM09 (Coolant Off)\n"
                f"{retract_cmd}\n{msg_cmd}\n"
                f"M00 (Program Stop - Operator Flip/Reclamp)\n"
                f"{wo} (Activate Setup Work Offset)\n"
            )

        gcode_blocks.append(result.gcode)
        total_time += result.total_time_seconds
        rapid_time += result.rapid_time_seconds
        cut_time += result.cut_time_seconds
        total_rapid_dist += result.total_rapid_dist_mm
        total_cut_dist += result.total_cut_dist_mm
        for t in result.tool_changes:
            if t not in all_tools:
                all_tools.append(t)

    combined_gcode = "\n".join(gcode_blocks)

    return {
        "separate_files": False,
        "controller": req.controller,
        "gcode": combined_gcode,
        "line_count": len(combined_gcode.splitlines()),
        "cycle_time_seconds": total_time,
        "rapid_time_seconds": rapid_time,
        "cut_time_seconds": cut_time,
        "total_rapid_dist_mm": total_rapid_dist,
        "total_cut_dist_mm": total_cut_dist,
        "tool_changes": all_tools,
        "work_offset": req.work_offset,
        "setups": setups_processed,
        "per_setup": per_setup_data,
    }


def export_cam_package_for_session(
    sess: "SessionState",
    controller: str = "haas",
    program_name: str | None = None,
) -> bytes:
    """Build and return a ZIP archive containing NC programs, setup sheets, and routing sheet."""
    from ...setup_sheet import (
        generate_setup_sheet,
        format_setup_sheet,
        generate_master_routing_sheet,
        format_master_routing_sheet,
    )

    if not sess.toolpaths:
        generate_toolpaths_for_session(sess, sess.stock_config)

    ctx, setups = build_context_and_setups(sess, sess.stock_config)
    base_name = program_name or sess.model_name.replace(" ", "_")

    zip_buffer = io.BytesIO()
    per_setup_results: dict[str, dict] = {}

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, s in enumerate(setups):
            setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id") == s.id]
            post = get_post_processor(controller=controller, work_offset=s.work_offset.value)
            s_prog_name = f"{base_name}_OP{(idx + 1) * 10}_{s.work_offset.value}"
            res = post.post_process(setup_tps, program_name=s_prog_name)
            per_setup_results[s.id] = {"cycle_time_seconds": res.total_time_seconds, "gcode": res.gcode}

            zf.writestr(f"NC_Programs/{s_prog_name}.nc", res.gcode)

            s_ops = [op for op in sess.planned_ops if op.notes.get("setup_id") == s.id]
            s_feats = [f for f in sess.features if f.notes.get("setup_id") == s.id]
            sheet = generate_setup_sheet(
                program_name=s_prog_name,
                part_name=sess.model_name,
                planning_ctx=ctx,
                setup=s,
                features=s_feats,
                operations=s_ops,
                toolpaths=setup_tps,
                post_result=res,
            )
            zf.writestr(
                f"Setup_Sheets/SETUP_SHEET_{s.id}_{s.work_offset.value}.txt",
                format_setup_sheet(sheet),
            )

        # Master combined NC
        fake_req = PostProcessRequest(controller=controller, program_name=base_name, separate_files=False)
        combined_result = generate_gcode_for_session(sess, fake_req)
        zf.writestr(f"NC_Programs/{base_name}_MASTER_ALL_SETUPS.nc", combined_result["gcode"])

        # Master routing sheet
        master_sheet = generate_master_routing_sheet(
            program_name=f"{base_name}_MASTER",
            part_name=sess.model_name,
            planning_ctx=ctx,
            setups=setups,
            all_features=sess.features,
            all_operations=sess.planned_ops,
            all_toolpaths=sess.toolpaths,
            per_setup_results=per_setup_results,
        )
        zf.writestr("MASTER_ROUTING_SHEET.txt", format_master_routing_sheet(master_sheet))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
