"""Common/shared Pydantic schemas used across multiple domains."""

from __future__ import annotations
from typing import Optional
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


class StockConfig(BaseModel):
    stock_mode: str = "relative_box"   # "relative_box", "fixed_box", "cylinder"
    margin_x: float = 5.0
    margin_y: float = 5.0
    margin_z_top: float = 1.0
    margin_z_bottom: float = 3.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    # Fixed size billet parameters
    fixed_size_x: Optional[float] = None
    fixed_size_y: Optional[float] = None
    fixed_size_z: Optional[float] = None
    # Cylindrical stock parameters
    cylinder_diameter: Optional[float] = None
    cylinder_length: Optional[float] = None
    cylinder_axis: str = "Z"           # "Z", "X", "Y"
    cylinder_margin_radial: float = 2.0
    cylinder_margin_axial_top: float = 1.0
    cylinder_margin_axial_bot: float = 3.0
    # Material and clamping
    material: str = "aluminum_6061"
    work_offset: str = "G54"
    thread_pitch: float = 1.0
    groove_width: float = 3.0
    clamp_type: str = "vise_jaws"     # "vise_jaws", "toe_clamps", "fixture_plate", "none"
    clamp_height: float = 3.0         # clamping hold allowance height from stock bottom in mm
    clamp_width: float = 12.0         # clamp/vise jaw thickness in mm
