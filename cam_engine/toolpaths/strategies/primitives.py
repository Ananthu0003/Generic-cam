"""Motion primitives and linking routines for toolpath generation."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ...operations import PlannedOperation
from ..semantic import MotionSegment, MotionType, Toolpath


class PrimitivesMixin:
    """Arc, helical, ramp, and linking motion primitives."""

    def _arc_to(self, tp: Toolpath, op: PlannedOperation, center: np.ndarray,
                end: np.ndarray, ccw: bool = True, feed: Optional[float] = None) -> None:
        """Add an arc segment (ARC_CW or ARC_CCW)."""
        prev = self._seg_end(tp)
        mtype = MotionType.ARC_CCW if ccw else MotionType.ARC_CW
        tp.add(MotionSegment(
            motion_type=mtype, start=prev, end=np.asarray(end, float),
            arc_center=np.asarray(center, float), arc_ccw=ccw,
            feed=feed or op.params.feed_rate, spindle=op.params.spindle_rpm,
            tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))

    def _helical_entry(self, tp: Toolpath, op: PlannedOperation,
                       center: np.ndarray, radius: float,
                       top_z: float, bottom_z: float, pitch: float) -> None:
        """Helical ramp descent into material at the given center/radius."""
        params = op.params
        center = np.asarray(center[:2], float)
        z = top_z
        n_seg_per_rev = 12
        p_start = self._v3(center[0] + radius, center[1], top_z)
        last = self._seg_end(tp)
        if np.linalg.norm(last - p_start) > 1e-4:
            if abs(last[2] - top_z) > 1e-4:
                self._rapid_to(tp, op, self._v3(p_start[0], p_start[1], max(last[2], top_z)))
            self._rapid_to(tp, op, p_start)

        while z > bottom_z + 1e-9:
            z_next = max(bottom_z, z - pitch)
            for k in range(1, n_seg_per_rev + 1):
                a0 = 2.0 * math.pi * (k - 1) / n_seg_per_rev
                a1 = 2.0 * math.pi * k / n_seg_per_rev
                z_interp0 = z + (z_next - z) * (k - 1) / n_seg_per_rev
                z_interp1 = z + (z_next - z) * k / n_seg_per_rev
                p0 = self._v3(center[0] + radius * math.cos(a0),
                              center[1] + radius * math.sin(a0), z_interp0)
                p1 = self._v3(center[0] + radius * math.cos(a1),
                              center[1] + radius * math.sin(a1), z_interp1)
                tp.add(MotionSegment(
                    motion_type=MotionType.HELIX, start=p0, end=p1,
                    arc_center=self._v3(center[0], center[1], z_interp1),
                    arc_ccw=True, feed=params.ramp_feed,
                    spindle=params.spindle_rpm, tool_id=op.tool.id,
                    operation_id=op.id, feature_id=op.feature.id))
            z = z_next

    def _link_to(self, tp: Toolpath, op: PlannedOperation, p: np.ndarray,
                 approach_offset: float = 1.0) -> None:
        """Retract to clearance, rapid to XY, rapid down to approach Z, plunge to target Z."""
        last = self._seg_end(tp)
        p_target = np.asarray(p, dtype=float)
        clearance_z = self.s.clearance_z
        approach_z = min(clearance_z, p_target[2] + approach_offset)

        # 1. Retract to clearance if not already there
        if abs(last[2] - clearance_z) > 1e-4:
            tp.add(MotionSegment(
                motion_type=MotionType.RETRACT, start=last,
                end=self._v3(last[0], last[1], clearance_z),
                tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))
            last = self._seg_end(tp)

        # 2. Rapid across in XY at clearance height
        if np.linalg.norm(last[:2] - p_target[:2]) > 1e-4:
            self._rapid_to(tp, op, self._v3(p_target[0], p_target[1], clearance_z))

        # 3. Rapid descend to approach height (e.g. target_z + 1mm)
        if abs(clearance_z - approach_z) > 1e-4:
            tp.add(MotionSegment(
                motion_type=MotionType.RAPID,
                start=self._v3(p_target[0], p_target[1], clearance_z),
                end=self._v3(p_target[0], p_target[1], approach_z),
                tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))

        # 4. Controlled plunge to target Z level
        if abs(approach_z - p_target[2]) > 1e-4:
            plunge_feed = op.params.plunge_feed if op.params.plunge_feed else 300.0
            tp.add(MotionSegment(
                motion_type=MotionType.PLUNGE,
                start=self._v3(p_target[0], p_target[1], approach_z),
                end=self._v3(p_target[0], p_target[1], p_target[2]),
                feed=plunge_feed,
                spindle=op.params.spindle_rpm,
                tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))

    def _link_stay_down(self, tp: Toolpath, op: PlannedOperation,
                        p: np.ndarray, clearance_above: float = 2.0) -> None:
        """Retract only to a small clearance above the current cut surface,
        not all the way to clearance_z. Use for moves between nearby points."""
        last = self._seg_end(tp)
        retract_z = max(last[2], p[2]) + clearance_above
        # small retract
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT, start=last,
            end=self._v3(last[0], last[1], retract_z),
            tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))
        # rapid XY at retract height
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID,
            start=self._v3(last[0], last[1], retract_z),
            end=self._v3(p[0], p[1], retract_z),
            tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))
        # descend at feed rate — stay-down uses controlled descent
        descent_feed = op.params.feed_rate * 0.5 if op.params.feed_rate else 100.0
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=self._v3(p[0], p[1], retract_z),
            end=self._v3(p[0], p[1], p[2]),
            feed=descent_feed,
            tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))

    def _arc_lead_in(self, tp: Toolpath, op: PlannedOperation,
                     target: np.ndarray, radius: float) -> None:
        """Tangential arc approach to the cut start point."""
        params = op.params
        prev = self._seg_end(tp)
        # direction from current position to target
        dx = target[0] - prev[0]
        dy = target[1] - prev[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-9:
            return
        # tangent direction (perpendicular to radial)
        tx, ty = -dy / dist, dx / dist
        # arc center is offset from target along the tangent
        cx = target[0] + tx * radius
        cy = target[1] + ty * radius
        arc_center = self._v3(cx, cy, target[2])
        # use CCW arc from current position to target
        self._arc_to(tp, op, arc_center, target, ccw=True, feed=params.ramp_feed)

    def _arc_lead_out(self, tp: Toolpath, op: PlannedOperation,
                      start_point: np.ndarray, radius: float) -> None:
        """Tangential arc departure from the cut end point."""
        params = op.params
        prev = self._seg_end(tp)
        # arc center offset perpendicular to the cut direction
        dx = prev[0] - start_point[0] if start_point is not None else 1.0
        dy = prev[1] - start_point[1] if start_point is not None else 0.0
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-9:
            dx, dy, dist = 1.0, 0.0, 1.0
        tx, ty = -dy / dist, dx / dist
        cx = prev[0] + tx * radius
        cy = prev[1] + ty * radius
        arc_end = self._v3(prev[0] + tx * radius * 0.5,
                           prev[1] + ty * radius * 0.5,
                           prev[2] + radius * 0.3)
        arc_center = self._v3(cx, cy, prev[2])
        self._arc_to(tp, op, arc_center, arc_end, ccw=False, feed=params.feed_rate)

    def _helical_ramp(self, tp: Toolpath, op: PlannedOperation,
                      target: np.ndarray, helix_radius: float,
                      pitch: float) -> None:
        """Helical descent into pocket at the target XY location."""
        feat_top = getattr(op.feature, 'top_z', target[2] + 5.0)
        top_z = min(self._seg_end(tp)[2], feat_top + 1.0)
        bottom_z = target[2]
        if top_z > bottom_z + 1e-6:
            self._helical_entry(tp, op, target, helix_radius, top_z, bottom_z, pitch)
        # finish at exact target point
        prev = self._seg_end(tp)
        if np.linalg.norm(prev - target) > 1e-4:
            self._add_cut(tp, op, target, feed=op.params.feed_rate)

    def _ramp_entry(self, tp: Toolpath, op: PlannedOperation, target: np.ndarray,
                    along: np.ndarray) -> None:
        """Ramp entry into material along `along` direction, respecting ramp feed.
        Ramp angle is taken from StrategyContext.ramp_angle_deg (default 15°)."""
        params = op.params
        end = np.asarray(target, dtype=float)
        prev = self._seg_end(tp)
        dz = end[2] - prev[2]
        if dz > -1e-9:
            tp.add(MotionSegment(motion_type=MotionType.CUT, start=prev, end=end,
                                 feed=params.feed_rate, spindle=params.spindle_rpm,
                                 tool_id=op.tool.id, operation_id=op.id,
                                 feature_id=op.feature.id))
            return

        # Localized ramp entry: ramp from approach height just above target
        # rather than all the way from clearance height
        ramp_dz = min(abs(dz), max(params.depth_of_cut_mm, 2.0))
        start_z = end[2] + ramp_dz

        horizontal = np.array([along[0], along[1]], dtype=float)
        nh = np.linalg.norm(horizontal)
        if nh < 1e-9:
            horizontal = np.array([1.0, 0.0])
            nh = 1.0
        horizontal = horizontal / nh
        ramp_angle = getattr(self.s, 'ramp_angle_deg', 15.0)
        ramp_len = ramp_dz / math.tan(math.radians(ramp_angle))

        # Keep ramp length constrained within feature bounds
        if hasattr(op.feature, 'xy_extent') and len(op.feature.xy_extent) > 0:
            max_extent = float(max(op.feature.xy_extent))
            ramp_len = min(ramp_len, max(3.0, max_extent * 0.4))

        a = end[:2] - horizontal * ramp_len
        start = self._v3(a[0], a[1], start_z)

        # Ensure tool reaches start position before ramping
        if np.linalg.norm(prev - start) > 1e-6:
            if prev[2] < start_z:
                self._rapid_to(tp, op, self._v3(prev[0], prev[1], start_z))
            self._rapid_to(tp, op, start)

        tp.add(MotionSegment(motion_type=MotionType.RAMP, start=start, end=end,
                             feed=params.ramp_feed, spindle=params.spindle_rpm,
                             tool_id=op.tool.id, operation_id=op.id,
                             feature_id=op.feature.id))
