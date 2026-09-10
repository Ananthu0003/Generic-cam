"""Semantic toolpath representation: controller-independent machining intent.

Segments carry full provenance (operation, feature, tool, coordinate space) so
post-processing, validation, simulation, and the UI can all trace back.
No G-code strings ever appear here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from ..coords import CoordSpace


class MotionType(str, Enum):
    RAPID = "rapid"
    CUT = "cut"
    PLUNGE = "plunge"
    RAMP = "ramp"
    HELIX = "helix"
    ARC_CW = "arc_cw"
    ARC_CCW = "arc_ccw"
    ENTRY = "entry"
    EXIT = "exit"
    RETRACT = "retract"
    LINK = "link"
    DWELL = "dwell"
    TOOL_CHANGE = "tool_change"
    SPINDLE_START = "spindle_start"
    SPINDLE_STOP = "spindle_stop"
    COOLANT_ON = "coolant_on"
    COOLANT_OFF = "coolant_off"


@dataclass
class MotionSegment:
    """One semantic motion segment in a defined coordinate space."""

    motion_type: MotionType
    start: np.ndarray                  # (3,) in `space`
    end: np.ndarray                    # (3,)
    space: CoordSpace = CoordSpace.SETUP
    feed: Optional[float] = None       # mm/min (cut/plunge/ramp/helix)
    spindle: Optional[float] = None    # rpm
    tool_id: Optional[str] = None
    operation_id: Optional[str] = None
    feature_id: Optional[str] = None
    arc_center: Optional[np.ndarray] = None   # arc center (space coords) for arcs/helix
    arc_ccw: bool = True
    dwell_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def length(self) -> float:
        if self.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW):
            return self._arc_length()
        return float(np.linalg.norm(self.end - self.start))

    def _arc_length(self) -> float:
        c = self.arc_center
        if c is None:
            return float(np.linalg.norm(self.end - self.start))
        r0 = np.linalg.norm(self.start[:2] - c[:2])
        r1 = np.linalg.norm(self.end[:2] - c[:2])
        r = 0.5 * (r0 + r1)
        a0 = np.arctan2(self.start[1] - c[1], self.start[0] - c[0])
        a1 = np.arctan2(self.end[1] - c[1], self.end[0] - c[0])
        if self.arc_ccw:
            sweep = (a1 - a0) % (2 * np.pi)
        else:
            sweep = (a0 - a1) % (2 * np.pi)
        # full circle when start==end
        if np.allclose(self.start, self.end, atol=1e-9):
            sweep = 2 * np.pi
        return float(r * sweep)


@dataclass
class Toolpath:
    """A validated semantic toolpath for one operation."""

    operation_id: str
    feature_id: str
    tool_id: str
    purpose: str
    segments: list[MotionSegment] = field(default_factory=list)
    space: CoordSpace = CoordSpace.SETUP
    metadata: dict = field(default_factory=dict)

    def add(self, seg: MotionSegment) -> None:
        self.segments.append(seg)

    def cutting_length(self) -> float:
        cutting = {MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
                   MotionType.HELIX, MotionType.ENTRY, MotionType.EXIT,
                   MotionType.ARC_CW, MotionType.ARC_CCW}
        return sum(s.length() for s in self.segments if s.motion_type in cutting)

    def rapid_length(self) -> float:
        return sum(s.length() for s in self.segments
                   if s.motion_type in (MotionType.RAPID, MotionType.LINK, MotionType.RETRACT))

    def segment_count(self) -> int:
        return len(self.segments)
