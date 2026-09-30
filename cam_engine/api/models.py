"""Backwards-compatibility shim.

All Pydantic schemas have been moved to ``cam_engine.api.schemas``.
This module re-exports everything so any code that still imports from
``cam_engine.api.models`` continues to work without modification.

New code should import directly from ``cam_engine.api.schemas``.
"""

from .schemas import (  # noqa: F401
    BoundingBox,
    MeshData,
    StockConfig,
    ModelInfoResponse,
    OrientModelRequest,
    FeatureItem,
    RecognizeFeaturesResponse,
    ToolItem,
    AddToolRequest,
    AutoSuggestToolRequest,
    MachineItem,
    SetupItem,
    AddSetupRequest,
    UpdateSetupRequest,
    SetupsListResponse,
    PlanOpsRequest,
    PlannedOpItem,
    PlanOpsResponse,
    MotionSegmentItem,
    ToolpathItem,
    GenerateToolpathsRequest,
    GenerateToolpathsResponse,
    MultiSetupToolpathsResponse,
    PostProcessRequest,
    PostProcessResponse,
    MultiSetupPostProcessResponse,
    PostProcessPerSetupResponse,
)
