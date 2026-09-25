"""Holemaking, boring, tapping, reaming, chamfering, and thread milling strategies."""

from __future__ import annotations

import math
import numpy as np

from ...errors import CamError, INVALID_TOOLPATH, UNSUPPORTED_FEATURE
from ...operations import PlannedOperation
from ..semantic import MotionSegment, MotionType, Toolpath


class HoleStrategyMixin:
    """Holemaking and localized feature machining strategies."""

    def _drilling(self, op: PlannedOperation) -> Toolpath:
        f = op.feature
        if op.notes.get("method") == "helical_interpolation":
            return self._helical_interpolation(op)
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z = f.top_z
        floor_z = f.floor_z
        # approach above hole
        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        depth = top_z - floor_z
        peck = params.depth_of_cut_mm
        r_plane = top_z + 1.0
        z = top_z
        # rapid down to R-plane before starting canned cycle
        prev = self._v3(center[0], center[1], r_plane)
        tp.add(MotionSegment(motion_type=MotionType.RAPID, start=self._v3(center[0], center[1], self.s.clearance_z),
                             end=prev,
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        first_plunge = True
        while z > floor_z + 1e-9:
            z_next = max(floor_z, z - peck)
            meta: dict = {}
            if first_plunge:
                # Post-processor converts this into a real G83 peck cycle;
                # explicit peck segments remain for simulation/validation.
                meta = {"canned_cycle": {
                    "type": "G83", "x": float(center[0]), "y": float(center[1]),
                    "z_r": float(r_plane), "z_depth": float(floor_z),
                    "peck": float(peck), "feed": params.plunge_feed,
                    "initial_z": float(self.s.clearance_z),
                }}
                first_plunge = False
            tp.add(MotionSegment(motion_type=MotionType.PLUNGE, start=prev,
                                 end=self._v3(center[0], center[1], z_next),
                                 feed=params.plunge_feed, spindle=params.spindle_rpm,
                                 tool_id=tool.id, operation_id=op.id, feature_id=f.id,
                                 metadata=meta))
            prev = self._v3(center[0], center[1], z_next)
            if z_next > floor_z + 1e-9:
                # retract chip-clear
                tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=prev,
                                      end=self._v3(center[0], center[1], top_z + 1.0),
                                      tool_id=tool.id, operation_id=op.id,
                                      feature_id=f.id))
                tp.add(MotionSegment(motion_type=MotionType.PLUNGE,
                                     start=self._v3(center[0], center[1], top_z + 1.0),
                                     end=self._v3(center[0], center[1], z_next + 0.5),
                                     feed=params.plunge_feed, spindle=params.spindle_rpm,
                                     tool_id=tool.id, operation_id=op.id,
                                     feature_id=f.id))
                prev = self._v3(center[0], center[1], z_next + 0.5)
            z = z_next
        # retract
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=prev,
                             end=self._v3(center[0], center[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "peck_drill"
        tp.metadata["canned_cycle_intent"] = {
            "position": center.tolist(), "depth": depth,
            "retract_height": 1.0, "peck_depth": peck,
            "feed": params.plunge_feed, "spindle": params.spindle_rpm}
        return tp

    def _helical_interpolation(self, op: PlannedOperation) -> Toolpath:
        """Open a hole or bore by continuous 3D helical milling with smooth tangential entry/exit.
        For large bores (radius > 0.45 * tool.diameter), uses concentric multi-ring helical clearing
        to evacuate 100% of material from center to finish wall."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z, floor_z = f.top_z, f.floor_z
        dia = f.diameter if (f.diameter is not None and f.diameter > 0) else (f.radius * 2.0 if (f.radius is not None and f.radius > 0) else (f.bounds_max[0] - f.bounds_min[0]))
        r_hole = dia / 2.0
        r_tool = tool.diameter / 2.0
        r_path = r_hole - r_tool
        if r_path <= 1e-4:
            raise CamError(UNSUPPORTED_FEATURE,
                           f"hole/bore {f.id}: tool diameter {tool.diameter}mm exceeds or equals hole diameter {dia:.2f}mm",
                           stage="toolpath", operation_id=op.id, feature_id=f.id)

        # 1. Approach over center at clearance Z, then rapid to top_z + 1.0
        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID,
            start=self._v3(center[0], center[1], self.s.clearance_z),
            end=self._v3(center[0], center[1], top_z + 1.0),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id
        ))

        # 2. Plunge down to top_z at plunge feed
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=self._v3(center[0], center[1], top_z + 1.0),
            end=self._v3(center[0], center[1], top_z),
            feed=params.plunge_feed, spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id
        ))

        # Multi-ring concentric passes if hole diameter significantly exceeds tool diameter
        radial_step = max(0.5, min(params.stepover_mm, tool.diameter * 0.65))
        n_rings = max(1, int(np.ceil(r_path / radial_step)))
        ring_radii = [float(r_path * (k / n_rings)) for k in range(1, n_rings + 1)]
        pitch = min(params.depth_of_cut_mm, tool.diameter * 0.5, 3.0)
        n_seg = 4  # 4 quadrant arcs per 360-deg revolution

        for ring_idx, r_curr in enumerate(ring_radii):
            entry_center = self._v3(center[0] + r_curr / 2.0, center[1], top_z if ring_idx == 0 else floor_z)
            start_pt = self._v3(center[0], center[1], top_z if ring_idx == 0 else floor_z)
            if ring_idx > 0:
                # Linear stepover to next ring at floor_z
                prev = self._seg_end(tp)
                tp.add(MotionSegment(
                    motion_type=MotionType.CUT,
                    start=prev,
                    end=self._v3(center[0] + r_curr, center[1], floor_z),
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id, operation_id=op.id, feature_id=f.id
                ))
            else:
                tp.add(MotionSegment(
                    motion_type=MotionType.ENTRY,
                    start=start_pt,
                    end=self._v3(center[0] + r_curr, center[1], top_z),
                    arc_center=entry_center,
                    arc_ccw=True,
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id, operation_id=op.id, feature_id=f.id
                ))

            # Continuous helical descent down to floor_z (only needed for first ring)
            if ring_idx == 0:
                z = top_z
                while z > floor_z + 1e-9:
                    z_next = max(floor_z, z - pitch)
                    for k in range(1, n_seg + 1):
                        a0 = 2 * np.pi * (k - 1) / n_seg
                        a1 = 2 * np.pi * k / n_seg
                        z0 = z + (z_next - z) * (k - 1) / n_seg
                        z1 = z + (z_next - z) * k / n_seg
                        p0 = self._v3(center[0] + r_curr * np.cos(a0), center[1] + r_curr * np.sin(a0), z0)
                        p1 = self._v3(center[0] + r_curr * np.cos(a1), center[1] + r_curr * np.sin(a1), z1)
                        tp.add(MotionSegment(
                            motion_type=MotionType.HELIX,
                            start=p0,
                            end=p1,
                            arc_center=self._v3(center[0], center[1], z1),
                            arc_ccw=True,
                            feed=params.feed_rate,
                            spindle=params.spindle_rpm,
                            tool_id=tool.id, operation_id=op.id, feature_id=f.id
                        ))
                    z = z_next

            # Full 360-degree flat circular pass at floor_z for current ring
            for k in range(1, n_seg + 1):
                a0 = 2 * np.pi * (k - 1) / n_seg
                a1 = 2 * np.pi * k / n_seg
                p0 = self._v3(center[0] + r_curr * np.cos(a0), center[1] + r_curr * np.sin(a0), floor_z)
                p1 = self._v3(center[0] + r_curr * np.cos(a1), center[1] + r_curr * np.sin(a1), floor_z)
                tp.add(MotionSegment(
                    motion_type=MotionType.ARC_CCW,
                    start=p0,
                    end=p1,
                    arc_center=self._v3(center[0], center[1], floor_z),
                    arc_ccw=True,
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id, operation_id=op.id, feature_id=f.id
                ))

        # Final tangential circular lead-out arc back to center
        tp.add(MotionSegment(
            motion_type=MotionType.EXIT,
            start=self._v3(center[0] + r_path, center[1], floor_z),
            end=self._v3(center[0], center[1], floor_z),
            arc_center=self._v3(center[0] + r_path / 2.0, center[1], floor_z),
            arc_ccw=True,
            feed=params.feed_rate,
            spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id
        ))

        # Retract straight up from center to clearance Z
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], floor_z),
            end=self._v3(center[0], center[1], self.s.clearance_z),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id
        ))

        self._end_op(tp, op)
        tp.metadata["strategy"] = "helical_interpolation"
        tp.metadata["concentric_rings"] = len(ring_radii)
        return tp

    def _boring(self, op: PlannedOperation) -> Toolpath:
        """Boring bar operation: single-point tool fed to depth, spindle stops at bottom,
        tool shifts off-centre and rapids out (G86 oriented boring cycle).

        Boring bars CANNOT use circular interpolation (G2/G3) — the cutting edge
        is offset from centre and the tool would gouge the wall on a circular arc.
        Instead, the strategy emits a single PLUNGE segment annotated with
        canned_cycle metadata so the post processor emits G86 (or G76 for fine boring).
        The post is responsible for emitting the correct modal cycle."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z, floor_z = f.top_z, f.floor_z
        if not f.diameter:
            raise CamError(UNSUPPORTED_FEATURE,
                           f"bore {f.id}: no diameter specified for boring",
                           stage="toolpath", operation_id=op.id, feature_id=f.id)

        r_plane = top_z + 1.0   # R-plane: clearance above bore entry
        tool_notes = getattr(tool, "notes", {}) or {}
        fine_boring = bool(op.notes.get("fine_boring") or (tool_notes.get("fine_boring") if isinstance(tool_notes, dict) else False))
        cycle_type = "G76" if fine_boring else "G86"
        shift_mm = float(op.notes.get("shift_amount_mm", tool_notes.get("shift_amount_mm", 0.2) if isinstance(tool_notes, dict) else 0.2))
        # G86: feed to depth → spindle stop → rapid out (roughing/semi boring)
        # G76: feed to depth → orient spindle → shift tip → rapid out (fine boring)

        # Position over bore at clearance
        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        # Rapid to R-plane
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID,
            start=self._seg_end(tp),
            end=self._v3(center[0], center[1], r_plane),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        # Single canned-cycle plunge to full depth
        canned_dict = {
            "type": cycle_type,
            "x": float(center[0]), "y": float(center[1]),
            "z_r": float(r_plane), "z_depth": float(floor_z),
            "feed": params.plunge_feed,
            "initial_z": float(self.s.clearance_z),
            "bore_diameter": float(f.diameter),
        }
        if fine_boring:
            canned_dict["shift"] = shift_mm
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=self._v3(center[0], center[1], r_plane),
            end=self._v3(center[0], center[1], floor_z),
            feed=params.plunge_feed, spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id,
            metadata={"canned_cycle": canned_dict}))
        # Retract after spindle stop (G86 semantics: rapid out from bore centre)
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], floor_z),
            end=self._v3(center[0], center[1], self.s.clearance_z),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "boring_canned_cycle"
        intent = {
            "cycle": cycle_type,
            "bore_diameter": float(f.diameter),
            "depth": float(top_z - floor_z),
            "retract_height": 1.0,
        }
        if fine_boring:
            intent["shift"] = shift_mm
        tp.metadata["canned_cycle_intent"] = intent
        return tp

    def _spot_drilling(self, op: PlannedOperation) -> Toolpath:
        """Spot-drill a conical center at each hole position so subsequent drills
        start true. Depth comes from the operation notes (planner-derived),
        never from a fallback constant."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy

        spot_depth = op.notes.get("spot_depth")
        if spot_depth is None:
            raise CamError(INVALID_TOOLPATH,
                           f"spot drilling {f.id}: 'spot_depth' missing from operation notes",
                           stage="toolpath", operation_id=op.id, feature_id=f.id,
                           tool_id=tool.id)
        spot_depth = float(spot_depth)
        if spot_depth <= 0:
            raise CamError(INVALID_TOOLPATH,
                           f"spot drilling {f.id}: non-positive spot depth {spot_depth:.2f}mm",
                           stage="toolpath", operation_id=op.id, feature_id=f.id,
                           tool_id=tool.id)

        top_z = f.top_z
        spot_bottom_z = top_z - spot_depth

        # approach above the hole position
        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        # rapid down to a small clearance above the surface, then feed to depth
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID, start=self._seg_end(tp),
            end=self._v3(center[0], center[1], top_z + 1.0),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=self._v3(center[0], center[1], top_z + 1.0),
            end=self._v3(center[0], center[1], spot_bottom_z),
            feed=params.plunge_feed, spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id,
            metadata={"canned_cycle": {
                "type": "G82", "x": float(center[0]), "y": float(center[1]),
                "z_r": top_z + 1.0, "z_depth": float(spot_bottom_z),
                "dwell": 0.1, "feed": params.plunge_feed,
                "initial_z": float(self.s.clearance_z),
            }}))
        # dwell at depth to establish the spot cleanly
        tp.add(MotionSegment(
            motion_type=MotionType.DWELL,
            start=self._v3(center[0], center[1], spot_bottom_z),
            end=self._v3(center[0], center[1], spot_bottom_z),
            dwell_seconds=0.1, tool_id=tool.id, operation_id=op.id,
            feature_id=f.id))
        # retract back to clearance
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], spot_bottom_z),
            end=self._v3(center[0], center[1], self.s.clearance_z),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))

        self._end_op(tp, op)
        tp.metadata["strategy"] = "spot_drill"
        tp.metadata["spot_depth_mm"] = spot_depth
        tp.metadata["canned_cycle_intent"] = {
            "position": center.tolist(), "depth": spot_depth,
            "retract_height": 1.0, "cycle": "G82"}
        return tp

    def _reaming(self, op: PlannedOperation) -> Toolpath:
        """Ream a pre-drilled hole to final size: single continuous feed down to
        depth, then feed back out at the same (or slightly reduced) feed. Reamers
        must never reverse-feed at reduced speed or dwell in the hole."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z, floor_z = f.top_z, f.floor_z

        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID, start=self._seg_end(tp),
            end=self._v3(center[0], center[1], top_z + 1.0),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))

        # feed down to full depth in one continuous plunge (no pecking)
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=self._v3(center[0], center[1], top_z + 1.0),
            end=self._v3(center[0], center[1], floor_z),
            feed=params.plunge_feed, spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id,
            metadata={"canned_cycle": {
                "type": "G85", "x": float(center[0]), "y": float(center[1]),
                "z_r": top_z + 1.0, "z_depth": float(floor_z),
                "feed": params.plunge_feed,
                "initial_z": float(self.s.clearance_z),
            }}))

        # feed back out at the same feed - reamers cut on the way out too
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], floor_z),
            end=self._v3(center[0], center[1], top_z + 1.0),
            feed=params.plunge_feed, spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))

        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], top_z + 1.0),
            end=self._v3(center[0], center[1], self.s.clearance_z),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))

        self._end_op(tp, op)
        tp.metadata["strategy"] = "reaming"
        tp.metadata["ream_diameter"] = f.diameter
        tp.metadata["canned_cycle_intent"] = {
            "position": center.tolist(), "depth": top_z - floor_z,
            "retract_height": 1.0, "cycle": "G85"}
        return tp

    def _tapping(self, op: PlannedOperation) -> Toolpath:
        """Tapping requires spindle-feed synchronization (rigid tapping G84 or
        spindle-sync feed). This strategy records the tapping intent on the
        toolpath; the post-processor emits the controller-appropriate cycle.

        Feed basis: G84 on Fanuc/Haas-class controllers runs under G94, so F
        must be mm/min = feed_per_rev * spindle_rpm. If that feed exceeds the
        machine's Z-axis feed capability, the spindle RPM is reduced until the
        required feed fits — the reduction is explicit and recorded, never
        silent."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z, floor_z = f.top_z, f.floor_z

        # tapping must be run with a feed synchronized to spindle rotation;
        # rpm-derived feed is wrong here - require the caller to supply it
        tap_feed = op.notes.get("tap_feed_mm_per_rev")
        if tap_feed is None:
            raise CamError(INVALID_TOOLPATH,
                           f"tapping {f.id}: 'tap_feed_mm_per_rev' missing from "
                           f"operation notes (tapping is feed-per-rev, not ipm)",
                           stage="toolpath", operation_id=op.id, feature_id=f.id,
                           tool_id=tool.id)
        tap_feed = float(tap_feed)
        if tap_feed <= 0:
            raise CamError(INVALID_TOOLPATH,
                           f"tapping {f.id}: non-positive tap feed",
                           stage="toolpath", operation_id=op.id, feature_id=f.id,
                           tool_id=tool.id)

        rpm = float(params.spindle_rpm)
        if rpm <= 0:
            raise CamError(INVALID_TOOLPATH,
                           f"tapping {f.id}: non-positive spindle rpm",
                           stage="toolpath", operation_id=op.id, feature_id=f.id,
                           tool_id=tool.id)

        # G94 tap feed in mm/min: feed_per_rev * rpm. Clamp against the
        # machine's Z-axis max feed by lowering RPM if needed (rigid tapping
        # keeps F = pitch * S constant, so RPM is the only free variable).
        z_axis = self.s.context.machine.axis_for("Z")
        tap_feed_mm_min = tap_feed * rpm
        if tap_feed_mm_min > z_axis.max_feed:
            rpm = z_axis.max_feed / tap_feed
            tap_feed_mm_min = tap_feed * rpm

        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID, start=self._seg_end(tp),
            end=self._v3(center[0], center[1], top_z + 1.0),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))

        # feed down to depth (post emits G84 rigid tapping from this intent)
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=self._v3(center[0], center[1], top_z + 1.0),
            end=self._v3(center[0], center[1], floor_z),
            feed=tap_feed_mm_min, spindle=rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id,
            metadata={"tap_feed_mm_per_rev": tap_feed,
                      "canned_cycle": {
                          "type": "G84", "x": float(center[0]), "y": float(center[1]),
                          "z_r": top_z + 1.0, "z_depth": float(floor_z),
                          "feed": tap_feed_mm_min,  # G94 basis: mm/min = mm/rev * rpm
                          "spindle": float(rpm),
                          "initial_z": float(self.s.clearance_z),
                      }}))

        # spindle reversal + feed out is emitted by the post from the cycle intent
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], floor_z),
            end=self._v3(center[0], center[1], top_z + 1.0),
            feed=tap_feed_mm_min, spindle=rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id,
            metadata={"tap_feed_mm_per_rev": tap_feed,
                      "spindle_reverse": True}))

        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], top_z + 1.0),
            end=self._v3(center[0], center[1], self.s.clearance_z),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))

        self._end_op(tp, op)
        tp.metadata["strategy"] = "tapping"
        tp.metadata["tap_feed_mm_per_rev"] = tap_feed
        tp.metadata["tap_spindle_rpm"] = rpm
        tp.metadata["tap_feed_mm_min"] = tap_feed_mm_min
        tp.metadata["canned_cycle_intent"] = {
            "position": center.tolist(), "depth": top_z - floor_z,
            "retract_height": 1.0, "cycle": "G84",
            "feed_mm_per_rev": tap_feed,
            "feed_mm_min": tap_feed_mm_min,
            "spindle_rpm": rpm}
        return tp

    def _chamfering(self, op: PlannedOperation) -> Toolpath:
        """Machine a chamfer feature: contour the boundary edge at the Z and XY
        offset that places the tool tip cone on the chamfer face. Depth and
        offset derive from the chamfer's actual slant geometry."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool

        # chamfer slant angle: op notes may override; the recognizer stores the
        # measured slant angle on the feature notes
        angle_deg = op.notes.get("angle_from_horizontal",
                                 f.notes.get("angle_from_horizontal"))
        if angle_deg is None:
            raise CamError(INVALID_TOOLPATH,
                           f"chamfer {f.id}: 'angle_from_horizontal' missing from notes",
                           stage="toolpath", operation_id=op.id, feature_id=f.id,
                           tool_id=tool.id)
        angle_deg = float(angle_deg)
        if not (0.0 < angle_deg < 90.0):
            raise CamError(INVALID_TOOLPATH,
                           f"chamfer {f.id}: implausible slant angle {angle_deg:.1f} deg",
                           stage="toolpath", operation_id=op.id, feature_id=f.id,
                           tool_id=tool.id)

        # horizontal leg of the chamfer cross-section (45deg chamfer: = depth)
        chamfer_depth = f.depth
        horizontal_leg = chamfer_depth / max(math.tan(math.radians(angle_deg)), 1e-9)

        # For a chamfer on top of a wall, the tool center must be offset outward
        # from the wall by (horizontal_leg + tool_radius) at the top surface,
        # descending so the cone's flank sweeps the chamfer face.
        radius = tool.diameter / 2.0
        xy_offset = horizontal_leg + radius

        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        # contour the chamfer's top boundary (bounds of the slanted face)
        cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
        hx = (bmax[0] - bmin[0]) / 2.0
        hy = (bmax[1] - bmin[1]) / 2.0
        if hx <= 1e-9 and hy <= 1e-9:
            tp.metadata["skipped"] = "degenerate chamfer bounds"
            self._end_op(tp, op)
            return tp

        top_z = f.top_z
        bottom_z = f.floor_z
        z_mid = 0.5 * (top_z + bottom_z)

        corners = [
            self._v3(cx - hx - xy_offset, cy - hy - xy_offset, z_mid),
            self._v3(cx + hx + xy_offset, cy - hy - xy_offset, z_mid),
            self._v3(cx + hx + xy_offset, cy + hy + xy_offset, z_mid),
            self._v3(cx - hx - xy_offset, cy + hy + xy_offset, z_mid),
        ]

        pt = corners[0]
        self._link_to(tp, op, pt)
        self._ramp_entry(tp, op, pt, along=np.array([1.0, 0, 0]))
        for c in corners[1:] + [corners[0]]:
            self._add_cut(tp, op, c)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "chamfer_contour"
        tp.metadata["chamfer_angle_deg"] = angle_deg
        tp.metadata["chamfer_depth_mm"] = chamfer_depth
        return tp

    def _thread_milling(self, op: PlannedOperation) -> Toolpath:
        """Thread milling: helical interpolation with thread mill tool to cut
        internal or external threads. Uses multi-pass helical motion with
        gradual radial engagement."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z = f.top_z
        floor_z = f.floor_z
        # Thread pitch from notes or default
        pitch = op.notes.get("pitch", f.diameter * 0.15 if f.diameter else 1.0)
        thread_depth = op.notes.get("thread_depth", pitch * 1.2)
        n_passes = max(1, int(math.ceil(thread_depth / (tool.diameter * 0.1))))
        radial_step = thread_depth / n_passes

        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))

        for pass_i in range(n_passes):
            r = (pass_i + 1) * radial_step  # start shallow, deepen each pass
            # Helical interpolation at this radius: Z descends linearly along the helix
            z = top_z
            while z > floor_z + 1e-9:
                z_next = max(floor_z, z - pitch)
                n_arc_pts = 12
                dz_per_pt = (z_next - z) / n_arc_pts
                for k in range(n_arc_pts):
                    angle = 2 * math.pi * k / n_arc_pts
                    x = center[0] + r * math.cos(angle)
                    y = center[1] + r * math.sin(angle)
                    pt = self._v3(x, y, z + dz_per_pt * (k + 1))
                    if pass_i == 0 and z == top_z and k == 0:
                        self._link_to(tp, op, pt)
                    else:
                        self._add_cut(tp, op, pt, feed=params.feed_rate * 0.5)
                z = z_next

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "thread_milling"
        tp.metadata["pitch"] = pitch
        return tp

    def _countersinking(self, op: PlannedOperation) -> Toolpath:
        """Countersink: single-pass plunge to chamfer depth with dwell (G82).
        A countersink is always a canned-cycle operation — no ramp, no contour.
        The tool plunges to the depth that produces the correct chamfer diameter,
        dwells briefly for a clean finish, then retracts rapidly."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z = f.top_z
        floor_z = f.floor_z
        r_plane = top_z + 1.0
        dwell_s = 0.2  # seconds at depth for clean countersink chamfer

        # Position over hole centre at clearance height
        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        # Rapid down to R-plane
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID,
            start=self._seg_end(tp),
            end=self._v3(center[0], center[1], r_plane),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        # Feed to countersink depth with G82 dwell canned cycle
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=self._v3(center[0], center[1], r_plane),
            end=self._v3(center[0], center[1], floor_z),
            feed=params.plunge_feed, spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id,
            metadata={
                "canned_cycle": {
                    "type": "G82",
                    "x": float(center[0]), "y": float(center[1]),
                    "z_r": float(r_plane), "z_depth": float(floor_z),
                    "dwell": dwell_s, "feed": params.plunge_feed,
                    "initial_z": float(self.s.clearance_z),
                }
            }))
        # Semantic dwell segment (simulation / cycle-time accounting)
        at_depth = self._v3(center[0], center[1], floor_z)
        tp.add(MotionSegment(
            motion_type=MotionType.DWELL,
            start=at_depth, end=at_depth,
            dwell_seconds=dwell_s,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        # Retract
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=at_depth,
            end=self._v3(center[0], center[1], self.s.clearance_z),
            tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "countersinking"
        tp.metadata["canned_cycle_intent"] = {
            "cycle": "G82", "position": center.tolist(),
            "depth": float(top_z - floor_z), "dwell_seconds": dwell_s,
            "countersink_angle_deg": op.notes.get("countersink_angle_deg", 45.0),
        }
        return tp

    def _grooving(self, op: PlannedOperation) -> Toolpath:
        """Grooving: plunge-cut to create a groove/slot at a specific location.
        Uses peck plunge for chip breaking, then retract."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z = f.top_z
        floor_z = f.floor_z
        groove_width = f.diameter if f.diameter else tool.diameter
        peck_depth = op.notes.get("peck_depth", tool.diameter * 0.3)

        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        self._rapid_to(tp, op, self._v3(center[0], center[1], top_z + 1.0))

        # Peck plunge
        z = top_z
        while z > floor_z + 1e-9:
            z_next = max(floor_z, z - peck_depth)
            self._add_cut(tp, op, self._v3(center[0], center[1], z_next),
                          feed=params.plunge_feed)
            # Small retract for chip breaking
            if z_next > floor_z + 1e-9:
                retract_z = z_next + min(peck_depth * 0.3, 2.0)
                self._add_cut(tp, op, self._v3(center[0], center[1], retract_z),
                              feed=params.feed_rate)
            z = z_next

        # Side-to-side groove widening if groove is wider than tool
        if groove_width > tool.diameter * 1.5:
            n_wipes = max(1, int(math.ceil((groove_width - tool.diameter) / (tool.diameter * 0.4))))
            wipe_step = (groove_width - tool.diameter) / (2 * n_wipes)
            for wi in range(n_wipes):
                offset = (wi + 1) * wipe_step
                for dx in [offset, -offset]:
                    self._add_cut(tp, op, self._v3(center[0] + dx, center[1], floor_z),
                                  feed=params.feed_rate * 0.3)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "grooving"
        return tp
