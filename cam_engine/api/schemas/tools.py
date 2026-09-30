"""Schemas for tool and machine management endpoints."""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel


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
