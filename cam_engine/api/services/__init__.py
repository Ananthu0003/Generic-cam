"""Services package for VexCAM API.

Each sub-module encapsulates the business logic for one domain.
Services are plain Python — they have no FastAPI coupling (no Request,
no HTTPException, no Depends) so they are easy to test in isolation.
"""

from .geometry_service import (
    euler_to_matrix,
    shape_to_demo_cad_model,
    mesh_to_response,
    tool_name,
)
from .stock_service import create_stock_from_config
from .context_service import build_context_and_setups
from .demo_service import populate_demo_features, make_dummy_imported
from .feature_service import recognize_features_for_session
from .operation_service import plan_operations_for_session
from .toolpath_service import generate_toolpaths_for_session
from .post_service import generate_gcode_for_session, export_cam_package_for_session
from .setup_service import (
    build_setups_response,
    auto_generate_setups_for_session,
)

__all__ = [
    "euler_to_matrix",
    "shape_to_demo_cad_model",
    "mesh_to_response",
    "tool_name",
    "create_stock_from_config",
    "build_context_and_setups",
    "populate_demo_features",
    "make_dummy_imported",
    "recognize_features_for_session",
    "plan_operations_for_session",
    "generate_toolpaths_for_session",
    "generate_gcode_for_session",
    "export_cam_package_for_session",
    "build_setups_response",
    "auto_generate_setups_for_session",
]
