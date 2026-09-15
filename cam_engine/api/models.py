"""Pydantic request and response schemas for Generic-CAM API."""

from __future__ import annotations

from typing import Optional, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    min: list[float]
    max: list[float]
    size: list[float]


class MeshData(BaseModel):
    vertices: list[float]
    normals: list[float]
    indices: list[int]
    triangle_count: int
    face_count: int
    bounding_box: BoundingBox


class ModelInfoResponse(BaseModel):
    name: str
    units: str
    mesh: MeshData
    features_count: Optional[int] = 0


class FeatureItem(BaseModel):
    id: str
    type: str
    bounds_min: list[float]
    bounds_max: list[float]
    depth: float
    top_z: float
    floor_z: float
    diameter: Optional[float] = None
    radius: Optional[float] = None
    accessibility: str
    is_concave: bool
    face_indices: list[int]
    notes: dict[str, Any] = Field(default_factory=dict)


class RecognizeFeaturesResponse(BaseModel):
    features: list[FeatureItem]
    count: int


class ToolItem(BaseModel):
    id: str
    tool_number: int = 1
    name: str = ""
    tool_type: str
    diameter: float
    corner_radius: float = 0.0
    flute_length: float
    overall_length: float
    flutes: int = 2
    max_rpm: float = 12000.0
    max_feed: float = 3000.0


class AddToolRequest(BaseModel):
    tool_number: Optional[int] = None
    name: Optional[str] = None
    tool_type: str = "drill"          # drill, flat_endmill, ball_endmill, bullnose_endmill, chamfer_mill, countersink_tool, boring_bar, reamer, tap
    diameter: float
    corner_radius: float = 0.0
    flute_length: float = 15.0
    overall_length: float = 50.0
    flutes: int = 2
    material: str = "carbide"


class AutoSuggestToolRequest(BaseModel):
    feature_id: str
    tool_type: Optional[str] = None
    diameter: Optional[float] = None



class MachineItem(BaseModel):
    id: str
    name: str
    machine_type: str
    controller: str
    spindle_max_rpm: float
    max_spindle_power_kw: float
    axes: dict[str, Any]


class StockConfig(BaseModel):
    margin_x: float = 5.0
    margin_y: float = 5.0
    margin_z_top: float = 1.0
    margin_z_bottom: float = 3.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    material: str = "aluminum_6061"
    work_offset: str = "G54"
    thread_pitch: float = 1.0
    groove_width: float = 3.0
    clamp_type: str = "vise_jaws"     # "vise_jaws", "toe_clamps", "fixture_plate", "none"
    clamp_height: float = 3.0         # clamping hold allowance height from stock bottom in mm
    clamp_width: float = 12.0         # clamp/vise jaw thickness in mm


class OrientModelRequest(BaseModel):
    axis: str = "X"              # "X", "Y", "Z"
    angle_deg: float = 90.0
    align_face_id: Optional[int] = None


class SetupItem(BaseModel):
    id: str
    name: str
    work_offset: str = "G54"
    tool_axis: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    rotation_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    stock: StockConfig = Field(default_factory=StockConfig)
    feature_ids: list[str] = Field(default_factory=list)
    operations_count: int = 0
    is_active: bool = False


class AddSetupRequest(BaseModel):
    name: str
    work_offset: str = "G55"
    preset: Optional[str] = "flip_x_180"  # "top", "flip_x_180", "flip_y_180", "side_x_90", "side_y_90"
    rotation_deg: Optional[list[float]] = None
    stock: Optional[StockConfig] = None


class SetupsListResponse(BaseModel):
    setups: list[SetupItem]
    active_setup_id: str


class MultiSetupToolpathsResponse(BaseModel):
    setups_toolpaths: dict[str, list[ToolpathItem]]
    total_cutting_length_mm: float
    total_rapid_length_mm: float
    estimated_time_seconds: float


class MultiSetupPostProcessResponse(BaseModel):
    controller: str
    gcode: str
    line_count: int
    cycle_time_seconds: float
    rapid_time_seconds: float
    cut_time_seconds: float
    total_rapid_dist_mm: float
    total_cut_dist_mm: float
    tool_changes: list[str]
    setups: list[str]


class PlanOpsRequest(BaseModel):
    stock: StockConfig = Field(default_factory=StockConfig)
    selected_features: Optional[list[str]] = None
    work_offset: str = "G54"
    setup_id: Optional[str] = None


class PlannedOpItem(BaseModel):
    id: str
    purpose: str
    feature_id: str
    tool_id: str
    tool_name: str
    feed_rate: float
    spindle_rpm: float
    stepover_mm: float
    stepdown_mm: float
    setup_id: Optional[str] = None
    enabled: bool = True
    notes: dict[str, Any] = Field(default_factory=dict)


class PlanOpsResponse(BaseModel):
    operations: list[PlannedOpItem]
    stock_bounds: BoundingBox
    setup_id: Optional[str] = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    unmachined_features: list[dict[str, Any]] = Field(default_factory=list)



class MotionSegmentItem(BaseModel):
    motion_type: str
    start: list[float]
    end: list[float]
    feed: Optional[float] = None
    spindle: Optional[float] = None
    arc_center: Optional[list[float]] = None
    arc_ccw: bool = True
    tool_id: Optional[str] = None
    operation_id: Optional[str] = None


class ToolpathItem(BaseModel):
    operation_id: str
    purpose: str
    tool_id: str
    setup_id: Optional[str] = None
    segment_count: int
    cutting_length_mm: float
    rapid_length_mm: float
    segments: list[MotionSegmentItem]


class GenerateToolpathsRequest(BaseModel):
    operations: Optional[list[PlannedOpItem]] = None
    stock: StockConfig = Field(default_factory=StockConfig)
    work_offset: str = "G54"
    setup_id: Optional[str] = None
    all_setups: bool = False


class GenerateToolpathsResponse(BaseModel):
    toolpaths: list[ToolpathItem]
    total_cutting_length_mm: float
    total_rapid_length_mm: float
    estimated_time_seconds: float
    setup_id: Optional[str] = None


class PostProcessRequest(BaseModel):
    controller: str = "grbl"
    work_offset: str = "G54"
    program_name: str = "PART_01"
    all_setups: bool = True


class PostProcessResponse(BaseModel):
    controller: str
    gcode: str
    line_count: int
    cycle_time_seconds: float
    rapid_time_seconds: float
    cut_time_seconds: float
    total_rapid_dist_mm: float
    total_cut_dist_mm: float
    tool_changes: list[str]
    work_offset: str
    setups: Optional[list[str]] = None
