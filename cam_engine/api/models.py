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
    margin_z_bottom: float = 2.0
    material: str = "aluminum_6061"


class PlanOpsRequest(BaseModel):
    stock: StockConfig = Field(default_factory=StockConfig)
    selected_features: Optional[list[str]] = None


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
    enabled: bool = True
    notes: dict[str, Any] = Field(default_factory=dict)


class PlanOpsResponse(BaseModel):
    operations: list[PlannedOpItem]
    stock_bounds: BoundingBox


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
    segment_count: int
    cutting_length_mm: float
    rapid_length_mm: float
    segments: list[MotionSegmentItem]


class GenerateToolpathsRequest(BaseModel):
    operations: Optional[list[PlannedOpItem]] = None
    stock: StockConfig = Field(default_factory=StockConfig)


class GenerateToolpathsResponse(BaseModel):
    toolpaths: list[ToolpathItem]
    total_cutting_length_mm: float
    total_rapid_length_mm: float
    estimated_time_seconds: float


class PostProcessRequest(BaseModel):
    controller: str = "grbl"
    work_offset: str = "G54"
    program_name: str = "PART_01"


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
