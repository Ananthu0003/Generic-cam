"""Coordinate-system model: explicit transforms, never assumed.

Spaces:
    MODEL   - the raw CAD/B-Rep coordinate system
    SETUP   - the machining setup coordinate system (stock-relative, Z up +Z)
    WORK    - the work-offset coordinate system programmed in G-code (G54 etc.)
    MACHINE - the machine physical coordinate system
    TOOL    - tool-local (orientation only, used by collision checks)

Every transform is a rigid 4x4 (rotation + translation, no scale). The chain
MODEL -> SETUP -> WORK -> MACHINE must be validated before any toolpath exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .errors import INVALID_COORDINATE_SYSTEM, INVALID_SETUP_TRANSFORM, CamError


class CoordSpace(str, Enum):
    MODEL = "model"
    SETUP = "setup"
    WORK = "work"
    MACHINE = "machine"
    TOOL = "tool"


def identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def make_transform(rotation: Optional[np.ndarray] = None,
                   translation: Optional[np.ndarray] = None) -> np.ndarray:
    t = identity()
    if rotation is not None:
        r = np.asarray(rotation, dtype=float)
        if r.shape != (3, 3):
            raise CamError(INVALID_COORDINATE_SYSTEM, "rotation must be 3x3", stage="coords")
        t[:3, :3] = r
    if translation is not None:
        v = np.asarray(translation, dtype=float).reshape(3)
        t[:3, 3] = v
    return t


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise CamError(INVALID_COORDINATE_SYSTEM, "zero-length rotation axis", stage="coords")
    a = a / n
    x, y, z = a
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def validate_rigid(m: np.ndarray, what: str = "transform") -> np.ndarray:
    m = np.asarray(m, dtype=float)
    if m.shape != (4, 4):
        raise CamError(INVALID_SETUP_TRANSFORM, f"{what}: transform must be 4x4", stage="coords")
    r = m[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6):
        raise CamError(INVALID_SETUP_TRANSFORM, f"{what}: rotation not orthonormal", stage="coords")
    if not np.isclose(np.linalg.det(r), 1.0, atol=1e-6):
        raise CamError(INVALID_SETUP_TRANSFORM, f"{what}: rotation determinant != 1 (mirrored?)", stage="coords")
    if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise CamError(INVALID_SETUP_TRANSFORM, f"{what}: bottom row must be [0,0,0,1]", stage="coords")
    return m


@dataclass
class CoordinateSystemChain:
    """Explicit chain of transforms between named spaces."""

    model_to_setup: np.ndarray = field(default_factory=identity)
    setup_to_work: np.ndarray = field(default_factory=identity)
    work_to_machine: np.ndarray = field(default_factory=identity)

    def __post_init__(self) -> None:
        validate_rigid(self.model_to_setup, "model->setup")
        validate_rigid(self.setup_to_work, "setup->work")
        validate_rigid(self.work_to_machine, "work->machine")

    def transform(self, src: CoordSpace, dst: CoordSpace) -> np.ndarray:
        """Compose the transform mapping points from space src to space dst."""
        m2s, s2w, w2m = self.model_to_setup, self.setup_to_work, self.work_to_machine
        fwd = {
            "model": identity(),
            "setup": m2s,
            "work": s2w @ m2s,
            "machine": w2m @ s2w @ m2s,
            "tool": s2w @ m2s,  # tool space ≈ work space for 3-axis (tool tip at origin)
        }
        if src == dst:
            return identity()
        if src == CoordSpace.MODEL:
            return fwd[dst.value]
        if dst == CoordSpace.MODEL:
            return _invert(fwd[src.value])
        # general: go via model
        to_model = _invert(fwd[src.value])
        return fwd[dst.value] @ to_model

    def validate_chain(self) -> None:
        # each leg validated at construction; validate composition consistency
        m2s, s2w, w2m = self.model_to_setup, self.setup_to_work, self.work_to_machine
        composed = w2m @ s2w @ m2s
        validate_rigid(composed, "model->machine composition")


def _invert(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    out = identity()
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t[:3, 3]
    return out


def dataclass_field_default() -> np.ndarray:
    return identity()


def transform_points(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to an (N,3) array of points."""
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return p.reshape(0, 3)
    r, tr = t[:3, :3], t[:3, 3]
    return p @ r.T + tr
