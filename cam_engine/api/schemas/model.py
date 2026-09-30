"""Schemas for CAD model upload and orientation endpoints."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

from .common import MeshData


class ModelInfoResponse(BaseModel):
    name: str
    units: str
    mesh: MeshData
    features_count: Optional[int] = 0


class OrientModelRequest(BaseModel):
    axis: str = "X"              # "X", "Y", "Z"
    angle_deg: float = 90.0
    align_face_id: Optional[int] = None
