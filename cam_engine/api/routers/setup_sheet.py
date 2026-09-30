"""Setup-sheet router — thin HTTP controller."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..dependencies import get_session
from ..exceptions import bad_request, not_found
from ..schemas import PlanOpsRequest, GenerateToolpathsRequest
from ..services.context_service import build_context_and_setups
from ..services.operation_service import plan_operations_for_session
from ..services.toolpath_service import generate_toolpaths_for_session
from ...post import get_post_processor
from ...setup_sheet import (
    generate_setup_sheet,
    format_setup_sheet,
    generate_master_routing_sheet,
    format_master_routing_sheet,
)

router = APIRouter()


def _ensure_computed(sess, request):
    """Lazily trigger planning and toolpath generation if not already done."""
    if not sess.planned_ops:
        plan_operations_for_session(sess, sess.stock_config)
    if not sess.toolpaths:
        generate_toolpaths_for_session(sess, sess.stock_config)


@router.get("/setup-sheet-master")
def get_master_setup_sheet(request: Request):
    """Generate the consolidated master routing sheet across all setups."""
    sess = get_session(request)
    if not sess.raw_mesh:
        raise bad_request("No model loaded")

    _ensure_computed(sess, request)

    ctx, setups = build_context_and_setups(sess, sess.stock_config)
    base_name = sess.model_name.replace(" ", "_")

    per_setup_results = {}
    for s in setups:
        setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id") == s.id]
        post = get_post_processor(controller=sess.machine.controller or "haas", work_offset=s.work_offset.value)
        res = post.post_process(setup_tps, program_name=f"{base_name}_{s.id}")
        per_setup_results[s.id] = {"cycle_time_seconds": res.total_time_seconds}

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

    return {
        "text": format_master_routing_sheet(master_sheet),
        "sheet": {
            "program_name": master_sheet.program_name,
            "part_name": master_sheet.part_name,
            "material": master_sheet.material,
            "machine": master_sheet.machine,
            "total_cycle_time": master_sheet.total_cycle_time,
            "setup_summaries": master_sheet.setup_summaries,
            "consolidated_tools": master_sheet.consolidated_tools,
            "notes": master_sheet.notes,
        },
    }


@router.get("/setup-sheet/{setup_id}")
def get_setup_sheet_for_setup(request: Request, setup_id: str):
    """Generate a setup sheet for a specific setup."""
    sess = get_session(request)
    if not sess.raw_mesh:
        raise bad_request("No model loaded")

    _ensure_computed(sess, request)

    ctx, setups = build_context_and_setups(sess, sess.stock_config)
    setup = next((s for s in setups if s.id == setup_id), None)
    if not setup:
        raise not_found("Setup", setup_id)

    setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id") == setup_id]
    s_ops = [op for op in sess.planned_ops if op.notes.get("setup_id") == setup_id]
    s_feats = [f for f in sess.features if f.notes.get("setup_id") == setup_id]

    post = get_post_processor(controller=sess.machine.controller or "haas", work_offset=setup.work_offset.value)
    post_result = post.post_process(
        setup_tps, program_name=f"{sess.model_name.replace(' ', '_')}_{setup.id}"
    )

    sheet = generate_setup_sheet(
        program_name=f"{sess.model_name.upper().replace(' ', '_')}_{setup.id}",
        part_name=sess.model_name,
        planning_ctx=ctx,
        setup=setup,
        features=s_feats,
        operations=s_ops,
        toolpaths=setup_tps,
        post_result=post_result,
    )

    return {
        "text": format_setup_sheet(sheet),
        "sheet": {
            "program_name": sheet.program_name,
            "part_name": sheet.part_name,
            "material": sheet.material,
            "machine": sheet.machine,
            "setup_name": sheet.setup_name,
            "work_offset": sheet.work_offset,
            "stock_dimensions": sheet.stock_dimensions,
            "tool_count": len(sheet.tool_list),
            "operation_count": len(sheet.operations),
            "estimated_cycle_time": sheet.estimated_cycle_time,
            "notes": sheet.notes,
        },
    }


@router.get("/setup-sheet")
def get_setup_sheet(request: Request):
    """Generate a setup sheet for shop floor documentation (primary setup)."""
    sess = get_session(request)
    if not sess.raw_mesh:
        raise bad_request("No model loaded")

    _ensure_computed(sess, request)

    ctx, setups = build_context_and_setups(sess, sess.stock_config)
    setup = setups[0]
    post = get_post_processor(controller="grbl", work_offset="G54")
    post_result = post.post_process(sess.toolpaths, program_name="SETUP_SHEET")

    sheet = generate_setup_sheet(
        program_name=sess.model_name.upper().replace(" ", "_"),
        part_name=sess.model_name,
        planning_ctx=ctx,
        setup=setup,
        features=sess.features,
        operations=sess.planned_ops,
        toolpaths=sess.toolpaths,
        post_result=post_result,
    )

    return {
        "text": format_setup_sheet(sheet),
        "sheet": {
            "program_name": sheet.program_name,
            "part_name": sheet.part_name,
            "material": sheet.material,
            "machine": sheet.machine,
            "stock_dimensions": sheet.stock_dimensions,
            "tool_count": len(sheet.tool_list),
            "operation_count": len(sheet.operations),
            "estimated_cycle_time": sheet.estimated_cycle_time,
            "notes": sheet.notes,
        },
    }
