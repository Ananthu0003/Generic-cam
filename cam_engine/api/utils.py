"""Backwards-compatibility shim.

All utility functions have been moved to ``cam_engine.api.services``.
This module re-exports the public helpers so any code that still imports
from ``cam_engine.api.utils`` continues to work without modification.

New code should import directly from the appropriate service module.
"""

from .services.geometry_service import (  # noqa: F401
    euler_to_matrix as _euler_to_matrix,
    shape_to_demo_cad_model as _shape_to_demo_cad_model,
    mesh_to_response as _mesh_to_response,
    tool_name as _tool_name,
)
from .services.stock_service import create_stock_from_config as _create_stock_from_config  # noqa: F401
from .services.context_service import build_context_and_setups as _build_context_and_setups  # noqa: F401
from .services.demo_service import (  # noqa: F401
    populate_demo_features as _populate_demo_features,
    make_dummy_imported as _make_dummy_imported,
)
