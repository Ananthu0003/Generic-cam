"""Base context and foundation strategy classes."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

import numpy as np

from ...context import PlanningContext, Setup
from ...coords import CoordSpace
from ...operations import PlannedOperation
from ..semantic import MotionSegment, MotionType, Toolpath


@dataclass
class StrategyContext:
    """Everything a strategy needs, drawn from the planning context."""

    context: PlanningContext
    setup: Setup
    mesh: object                     # TriangleMesh of model in setup space
    stock_voxels: object             # VoxelStock: current remaining stock state
    clearance_z: float               # setup-space Z for safe rapids (derived)
    finish_allowance_wall: float     # from FinishRequirement or config default
    finish_allowance_floor: float
    ramp_angle_deg: float = 15.0     # ramp entry angle (deg); lower for harder materials
    debug_stage: Optional[str] = field(default_factory=lambda: os.environ.get("DEBUG_STAGE"))
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    """Base class holding StrategyContext and common primitive toolpath generation utilities."""

    def __init__(self, strat: StrategyContext):
        self.s = strat

    def _toolpath_skeleton(self, op: PlannedOperation) -> Toolpath:
        return Toolpath(operation_id=op.id, feature_id=op.feature.id,
                        tool_id=op.tool.id, purpose=op.purpose.value,
                        space=CoordSpace.SETUP)

    def _start_op(self, tp: Toolpath, op: PlannedOperation, start_pt: Optional[np.ndarray] = None) -> None:
        """Spindle/coolant state + approach from clearance."""
        tool = op.tool
        pos = start_pt if start_pt is not None else self._v3(0, 0, self.s.clearance_z)
        tp.add(MotionSegment(
            motion_type=MotionType.TOOL_CHANGE, start=pos, end=pos,
            tool_id=tool.id, operation_id=op.id, feature_id=op.feature.id,
            metadata={"tool_number": tool.tool_number,
                      "length_offset_h": tool.length_offset_h}))
        tp.add(MotionSegment(
            motion_type=MotionType.SPINDLE_START, start=pos, end=pos,
            spindle=op.params.spindle_rpm, tool_id=tool.id,
            operation_id=op.id, feature_id=op.feature.id))
        if op.params.coolant:
            tp.add(MotionSegment(
                motion_type=MotionType.COOLANT_ON, start=pos, end=pos,
                tool_id=tool.id, operation_id=op.id, feature_id=op.feature.id))

    def _end_op(self, tp: Toolpath, op: PlannedOperation) -> None:
        tool = op.tool
        last = self._seg_end(tp)
        if op.params.coolant:
            tp.add(MotionSegment(
                motion_type=MotionType.COOLANT_OFF, start=last, end=last,
                tool_id=tool.id, operation_id=op.id, feature_id=op.feature.id))
        tp.add(MotionSegment(
            motion_type=MotionType.SPINDLE_STOP, start=last, end=last,
            tool_id=tool.id, operation_id=op.id, feature_id=op.feature.id))

    @staticmethod
    def _v3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
        return np.array([x, y, z], dtype=float)

    def _seg_end(self, tp: Toolpath) -> np.ndarray:
        """Return the end point of the last spatial motion segment, or a clearance-height origin."""
        spatial_types = {
            MotionType.RAPID, MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
            MotionType.HELIX, MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.ENTRY,
            MotionType.EXIT, MotionType.RETRACT, MotionType.LINK
        }
        for s in reversed(tp.segments):
            if s.motion_type in spatial_types and s.end is not None:
                return s.end.copy()
        return self._v3(0, 0, self.s.clearance_z)

    def _add_cut(self, tp: Toolpath, op: PlannedOperation, end: np.ndarray,
                 feed: Optional[float] = None) -> None:
        prev = self._seg_end(tp)
        tp.add(MotionSegment(
            motion_type=MotionType.CUT, start=prev, end=np.asarray(end, float),
            feed=feed or op.params.feed_rate, spindle=op.params.spindle_rpm,
            tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))

    def _adaptive_feed(self, base_feed: float, engagement_angle_deg: float,
                       max_engagement_deg: float = 180.0) -> float:
        """Adjust feed rate based on engagement angle for constant chip thickness.
        Higher engagement = slower feed to maintain load; lower engagement = faster feed.
        This is the core of adaptive/trochoidal roughing efficiency."""
        if max_engagement_deg <= 0:
            return base_feed
        # Scale feed inversely with engagement ratio
        ratio = engagement_angle_deg / max_engagement_deg
        # Clamp between 30% and 200% of base feed
        scale = max(0.3, min(2.0, 1.0 / max(ratio, 0.1)))
        return base_feed * scale

    def _estimate_engagement(self, tool_diameter: float, stepover_mm: float) -> float:
        """Estimate engagement angle from stepover (radial depth of cut).
        For a flat endmill: ae = D * (1 - cos(theta/2)) -> theta = 2*acos(1 - 2*ae/D)
        Returns angle in degrees."""
        ae = stepover_mm
        D = tool_diameter
        if ae >= D:
            return 180.0  # full slot
        ratio = 1.0 - 2.0 * ae / D
        ratio = max(-1.0, min(1.0, ratio))
        return math.degrees(2.0 * math.acos(ratio))

    def _rapid_to(self, tp: Toolpath, op: PlannedOperation, p: np.ndarray) -> None:
        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RAPID, start=last,
                             end=np.asarray(p, float),
                             tool_id=op.tool.id, operation_id=op.id,
                             feature_id=op.feature.id))
