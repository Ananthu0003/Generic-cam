"""Toolpath generation service.

Contains domain logic for generating semantic toolpaths.
Pure Python — no FastAPI coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...toolpaths.strategies import StrategyEngine, StrategyContext
from ...toolpaths.semantic import Toolpath
from ...geometry.mesh import TriangleMesh
from ...geometry.voxel import VoxelStock
from ...demo_models import create_prismatic_bracket_mesh
from ..schemas import StockConfig
from .context_service import build_context_and_setups
from .operation_service import plan_operations_for_session

if TYPE_CHECKING:
    from ..state import SessionState


def generate_toolpaths_for_session(
    sess: "SessionState",
    stock: StockConfig,
) -> dict:
    """Generate toolpaths for all setups and write them into ``sess.toolpaths``.

    Returns a diagnostics dict (may be empty).
    Calls operation planning if not already done.
    """
    if not sess.planned_ops:
        plan_operations_for_session(sess, stock)

    ctx, setups = build_context_and_setups(sess, stock)

    all_toolpaths: list[Toolpath] = []
    total_cut_len = 0.0
    total_rapid_len = 0.0
    prev_remaining_stock = None
    prev_setup = None
    all_diagnostics: dict = {}

    for s in setups:
        setup_ops = [op for op in sess.planned_ops if op.notes.get("setup_id") == s.id]
        if not setup_ops:
            continue

        # Build triangle mesh for the current setup orientation
        if sess.imported_model:
            setup_shape = ctx.model_shape_in_setup(s)
            mesh = TriangleMesh.from_faces(setup_shape.faces())
        else:
            demo = sess.raw_mesh or create_prismatic_bracket_mesh()
            tri_pts = np.array(demo.vertices).reshape(-1, 3)
            indices = np.array(demo.indices).reshape(-1, 3)
            tris = tri_pts[indices]
            mesh = TriangleMesh(triangles=tris)

        dims = s.stock.bounds_max - s.stock.bounds_min
        resolution = max(0.5, min(dims) / 50.0)

        if prev_remaining_stock is not None and prev_setup is not None:
            M_curr = s.model_to_setup
            M_prev = prev_setup.model_to_setup
            try:
                T_prev_to_curr = M_curr @ np.linalg.inv(M_prev)
            except np.linalg.LinAlgError:
                T_prev_to_curr = np.eye(4)
            stock_voxels = prev_remaining_stock.transformed(
                T_prev_to_curr, s.stock.bounds_min, s.stock.bounds_max, resolution
            )
        else:
            stock_voxels = VoxelStock.from_bounds(s.stock.bounds_min, s.stock.bounds_max, resolution)

        clearance_z = float(s.stock.bounds_max[2]) + sess.machine.safe_retract_height

        strat_ctx = StrategyContext(
            context=ctx,
            setup=s,
            mesh=mesh,
            stock_voxels=stock_voxels,
            clearance_z=clearance_z,
            finish_allowance_wall=0.3,
            finish_allowance_floor=0.3,
        )
        engine = StrategyEngine(strat_ctx)

        for op in setup_ops:
            tp = engine.generate(op)
            tp.metadata["setup_id"] = s.id
            tp.metadata["work_offset"] = s.work_offset.value
            all_toolpaths.append(tp)
            total_cut_len += tp.cutting_length()
            total_rapid_len += tp.rapid_length()

        prev_remaining_stock = stock_voxels
        prev_setup = s

        if strat_ctx.diagnostics:
            all_diagnostics.update(strat_ctx.diagnostics)

    sess.toolpaths = all_toolpaths
    return all_diagnostics
