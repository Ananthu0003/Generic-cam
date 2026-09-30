"""Post-processing router — thin HTTP controller."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response

from ..dependencies import get_session
from ..schemas import (
    GenerateToolpathsRequest,
    PostProcessPerSetupResponse,
    PostProcessRequest,
    PostProcessResponse,
)
from ..services.post_service import generate_gcode_for_session, export_cam_package_for_session
from ..services.toolpath_service import generate_toolpaths_for_session
from ...errors import CamError

router = APIRouter()


@router.post("/generate-gcode", response_model=PostProcessResponse)
def generate_gcode(request: Request, req: PostProcessRequest):
    """Post-process generated toolpaths into controller-specific CNC G-code with multi-setup support."""
    sess = get_session(request)
    try:
        result = generate_gcode_for_session(sess, req)

        if result["separate_files"]:
            return PostProcessPerSetupResponse(
                controller=result["controller"],
                setups=result["setups"],
                total_cycle_time_seconds=result["total_cycle_time_seconds"],
            )

        return PostProcessResponse(
            controller=result["controller"],
            gcode=result["gcode"],
            line_count=result["line_count"],
            cycle_time_seconds=result["cycle_time_seconds"],
            rapid_time_seconds=result["rapid_time_seconds"],
            cut_time_seconds=result["cut_time_seconds"],
            total_rapid_dist_mm=result["total_rapid_dist_mm"],
            total_cut_dist_mm=result["total_cut_dist_mm"],
            tool_changes=result["tool_changes"],
            work_offset=result["work_offset"],
            setups=result["setups"],
            per_setup=result["per_setup"],
        )
    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")


@router.get("/gcode/download/{setup_id}")
def download_setup_gcode(
    request: Request,
    setup_id: str,
    controller: str = "haas",
    program_name: Optional[str] = None,
):
    """Generate and download the G-code file for a specific setup."""
    sess = get_session(request)

    if not sess.toolpaths:
        generate_toolpaths_for_session(sess, sess.stock_config)

    target_setup = next((s for s in sess.setups if s.id == setup_id), None)
    if not target_setup:
        raise HTTPException(status_code=404, detail=f"Setup {setup_id} not found")

    from ...post import get_post_processor
    setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id", "setup_001") == setup_id]
    prog_name = program_name or f"{sess.model_name.replace(' ', '_')}_{target_setup.name.replace(' ', '_')}"
    post = get_post_processor(controller=controller, work_offset=target_setup.work_offset)
    result = post.post_process(setup_tps, program_name=prog_name)

    safe_name = (
        f"{sess.model_name.replace(' ', '_')}"
        f"_{target_setup.name.replace(' ', '_')}"
        f"_{target_setup.work_offset}.nc"
    )
    return Response(
        content=result.gcode,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@router.get("/export-package")
def export_cam_package(
    request: Request,
    controller: str = "haas",
    program_name: Optional[str] = None,
):
    """Generate a complete multi-setup production package (ZIP)."""
    sess = get_session(request)
    zip_bytes = export_cam_package_for_session(sess, controller=controller, program_name=program_name)
    base_name = (program_name or sess.model_name.replace(" ", "_"))
    zip_name = f"{base_name}_CAM_Package.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )
