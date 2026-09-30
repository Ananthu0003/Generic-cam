"""Schemas for machining feature recognition."""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field


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
    setup_id: Optional[str] = None
    notes: dict[str, Any] = Field(default_factory=dict)


class RecognizeFeaturesResponse(BaseModel):
    features: list[FeatureItem]
    count: int
