"""Schemas package for VexCAM API.

Re-exports every schema so existing code that imported from ``..models``
(the old monolithic file) can switch to ``..schemas`` without hunt-and-replace.
Individual domain modules under this package contain the actual class definitions.
"""

from .common import BoundingBox, MeshData, StockConfig
from .model import ModelInfoResponse, OrientModelRequest
from .features import FeatureItem, RecognizeFeaturesResponse
from .tools import ToolItem, AddToolRequest, AutoSuggestToolRequest, MachineItem
from .setups import SetupItem, AddSetupRequest, UpdateSetupRequest, SetupsListResponse
from .operations import PlanOpsRequest, PlannedOpItem, PlanOpsResponse
from .toolpaths import (
    MotionSegmentItem,
    ToolpathItem,
    GenerateToolpathsRequest,
    GenerateToolpathsResponse,
    MultiSetupToolpathsResponse,
)
from .post import (
    PostProcessRequest,
    PostProcessResponse,
    MultiSetupPostProcessResponse,
    PostProcessPerSetupResponse,
)

__all__ = [
    # common
    "BoundingBox",
    "MeshData",
    "StockConfig",
    # model
    "ModelInfoResponse",
    "OrientModelRequest",
    # features
    "FeatureItem",
    "RecognizeFeaturesResponse",
    # tools
    "ToolItem",
    "AddToolRequest",
    "AutoSuggestToolRequest",
    "MachineItem",
    # setups
    "SetupItem",
    "AddSetupRequest",
    "UpdateSetupRequest",
    "SetupsListResponse",
    # operations
    "PlanOpsRequest",
    "PlannedOpItem",
    "PlanOpsResponse",
    # toolpaths
    "MotionSegmentItem",
    "ToolpathItem",
    "GenerateToolpathsRequest",
    "GenerateToolpathsResponse",
    "MultiSetupToolpathsResponse",
    # post
    "PostProcessRequest",
    "PostProcessResponse",
    "MultiSetupPostProcessResponse",
    "PostProcessPerSetupResponse",
]
