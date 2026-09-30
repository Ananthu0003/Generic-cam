"""Schemas for machining setup management endpoints."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from .common import BoundingBox, StockConfig


class SetupItem(BaseModel):
    id: str
    name: str
    work_offset: str = "G54"
    tool_axis: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    rotation_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    stock: StockConfig = Field(default_factory=StockConfig)
    stock_bounds: Optional[BoundingBox] = None
    feature_ids: list[str] = Field(default_factory=list)
    operations_count: int = 0
    is_active: bool = False


class AddSetupRequest(BaseModel):
    name: str
    work_offset: str = "G55"
    preset: Optional[str] = "flip_x_180"  # "top", "flip_x_180", "flip_y_180", "side_x_90", "side_y_90"
    rotation_deg: Optional[list[float]] = None
    stock: Optional[StockConfig] = None


class UpdateSetupRequest(BaseModel):
    name: Optional[str] = None
    work_offset: Optional[str] = None
    rotation_deg: Optional[list[float]] = None
    stock: Optional[StockConfig] = None


class SetupsListResponse(BaseModel):
    setups: list[SetupItem]
    active_setup_id: str
