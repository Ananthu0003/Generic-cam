"""Operation planning service.

Contains domain logic for planning machining operations across all session
setups.  Pure Python — no FastAPI coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...operations import OperationPlanner
from ..schemas import StockConfig
from .context_service import build_context_and_setups
from .feature_service import recognize_features_for_session

if TYPE_CHECKING:
    from ..state import SessionState


def plan_operations_for_session(
    sess: "SessionState",
    stock: StockConfig,
    selected_features: Optional[list[str]] = None,
) -> None:
    """Plan machining operations for all setups and write results into ``sess``.

    Calls feature recognition first if features are not yet computed.
    """
    if not sess.features:
        recognize_features_for_session(sess)

    sess.stock_config = stock
    ctx, setups = build_context_and_setups(sess, stock)

    all_ops = []
    for s in setups:
        setup_features = [
            f for f in sess.features
            if f.notes.get("setup_id", sess.setups[0].id) == s.id
            and (not selected_features or f.id in selected_features)
        ]
        if not setup_features:
            continue

        planner = OperationPlanner(ctx, s, setup_features)
        ops = planner.plan()

        for op in ops:
            op.notes["setup_id"] = s.id
            op.notes["setup_name"] = s.name
            op.notes["work_offset"] = s.work_offset.value
        all_ops.extend(ops)

        for sc in sess.setups:
            if sc.id == s.id:
                sc.operations_count = len(ops)

    sess.planned_ops = all_ops
