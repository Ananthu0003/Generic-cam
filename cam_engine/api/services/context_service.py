"""Planning context and setup builder service.

Pure functions — no FastAPI coupling.  Extracted from the old ``utils.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ...context import (
    PlanningContext,
    Setup,
    WorkOffset,
    DEFAULT_MATERIALS,
)
from ..schemas import StockConfig
from .geometry_service import euler_to_matrix
from .stock_service import create_stock_from_config, build_fixtures

if TYPE_CHECKING:
    from ..state import SessionState

_WORK_OFFSET_MAP: dict[str, WorkOffset] = {
    "G54": WorkOffset.G54,
    "G55": WorkOffset.G55,
    "G56": WorkOffset.G56,
    "G57": WorkOffset.G57,
    "G58": WorkOffset.G58,
    "G59": WorkOffset.G59,
}


def build_context_and_setups(
    sess: "SessionState",
    stock_cfg: StockConfig,
) -> tuple[PlanningContext, list[Setup]]:
    """Build a ``PlanningContext`` and a list of ``Setup`` objects from session state."""
    from .demo_service import make_dummy_imported

    imported = sess.imported_model or make_dummy_imported(sess)
    material = DEFAULT_MATERIALS.get(stock_cfg.material, DEFAULT_MATERIALS["aluminum_6061"])

    setups_list: list[Setup] = []

    for sc in sess.setups:
        rot_mat = euler_to_matrix(*sc.rotation_deg)

        if sess.imported_model:
            shape_in_setup = sess.imported_model.shape.transformed(rot_mat)
            bmin, bmax = shape_in_setup.bounding_box()
        else:
            bmin = sess.raw_mesh.bounds_min if sess.raw_mesh else [0, 0, 0]
            bmax = sess.raw_mesh.bounds_max if sess.raw_mesh else [100, 60, 25]

        cfg = sc.stock_cfg
        stock = create_stock_from_config(cfg, bmin, bmax)
        fixtures = build_fixtures(sc, stock, cfg)
        wo = _WORK_OFFSET_MAP.get(sc.work_offset, WorkOffset.G54)

        setups_list.append(
            Setup(
                id=sc.id,
                name=sc.name,
                model_to_setup=rot_mat,
                work_offset=wo,
                stock=stock,
                fixtures=fixtures,
            )
        )

    ctx = PlanningContext(
        model=imported,
        setups=setups_list,
        machine=sess.machine,
        tools=sess.tools,
        material=material,
    )
    return ctx, setups_list
