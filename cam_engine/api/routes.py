"""FastAPI REST API routes for VexCAM.

This file only wires routers together.  Business logic lives in
``cam_engine.api.services``.  Schemas live in ``cam_engine.api.schemas``.
"""

from fastapi import APIRouter

from .routers.model import router as model_router
from .routers.setups import router as setups_router
from .routers.tools import router as tools_router
from .routers.features import router as features_router
from .routers.operations import router as operations_router
from .routers.toolpaths import router as toolpaths_router
from .routers.post import router as post_router
from .routers.setup_sheet import router as setup_sheet_router

router = APIRouter(prefix="/api", tags=["cam"])

router.include_router(model_router)
router.include_router(setups_router)
router.include_router(tools_router)
router.include_router(features_router)
router.include_router(operations_router)
router.include_router(toolpaths_router)
router.include_router(post_router)
router.include_router(setup_sheet_router)

# ---------------------------------------------------------------------------
# Backwards-compatibility re-exports for existing test code
# ---------------------------------------------------------------------------
from .services.stock_service import create_stock_from_config as _create_stock_from_config  # noqa: E402,F401

