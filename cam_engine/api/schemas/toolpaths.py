"""Schemas for toolpath generation endpoints."""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field

from .common import StockConfig
from .operations import PlannedOpItem


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
    diagnostics: Optional[dict[str, Any]] = None


class MultiSetupToolpathsResponse(BaseModel):
    setups_toolpaths: dict[str, list[ToolpathItem]]
    total_cutting_length_mm: float
    total_rapid_length_mm: float
    estimated_time_seconds: float
