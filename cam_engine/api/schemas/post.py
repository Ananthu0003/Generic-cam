"""Schemas for G-code post-processing and export endpoints."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

from .toolpaths import ToolpathItem


class PostProcessRequest(BaseModel):
    controller: str = "grbl"
    work_offset: str = "G54"
    program_name: str = "PART_01"
    all_setups: bool = True
    separate_files: bool = False  # If true, return separate G-code per setup


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
    per_setup: Optional[dict[str, dict]] = None


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


class PostProcessPerSetupResponse(BaseModel):
    """Separate G-code file per setup."""
    controller: str
    setups: dict[str, dict]  # setup_id -> {gcode, line_count, cycle_time, work_offset, name}
    total_cycle_time_seconds: float
