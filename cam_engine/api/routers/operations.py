"""Operations router — thin HTTP controller."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..dependencies import get_session
from ..schemas import (
    BoundingBox,
    PlannedOpItem,
    PlanOpsRequest,
    PlanOpsResponse,
)
from ..services.operation_service import plan_operations_for_session
from ..services.geometry_service import tool_name
from ...errors import CamError

router = APIRouter()


@router.post("/plan-operations", response_model=PlanOpsResponse)
def plan_operations(request: Request, req: PlanOpsRequest):
    """Plan ordered machining operations based on recognised features and tooling across setups."""
    sess = get_session(request)

    try:
        plan_operations_for_session(sess, req.stock, req.selected_features)

        op_items = [
            PlannedOpItem(
                id=op.id,
                purpose=op.purpose.value if hasattr(op.purpose, "value") else str(op.purpose),
                feature_id=op.feature.id,
                tool_id=op.tool.id,
                tool_name=tool_name(op.tool),
                feed_rate=op.params.feed_rate,
                spindle_rpm=op.params.spindle_rpm,
                stepover_mm=op.params.stepover_mm,
                stepdown_mm=op.params.depth_of_cut_mm,
                setup_id=op.notes.get("setup_id"),
                notes=op.notes,
            )
            for op in sess.planned_ops
        ]

        # Stock bounds from the first setup
        from ..services.context_service import build_context_and_setups
        _, setups = build_context_and_setups(sess, req.stock)
        s0 = setups[0]
        stock_min = [float(x) for x in s0.stock.bounds_min]
        stock_max = [float(x) for x in s0.stock.bounds_max]
        stock_size = [stock_max[i] - stock_min[i] for i in range(3)]

        return PlanOpsResponse(
            operations=op_items,
            stock_bounds=BoundingBox(min=stock_min, max=stock_max, size=stock_size),
            setup_id=sess.active_setup_id,
            warnings=[],
            unmachined_features=[],
        )

    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")
