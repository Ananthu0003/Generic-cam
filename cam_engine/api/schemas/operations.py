"""Schemas for operation planning endpoints."""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field

from .common import BoundingBox, StockConfig


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
