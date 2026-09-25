"""Semi-finishing and 2D/3D finishing strategies."""

from __future__ import annotations

import math
import numpy as np

from ...context import ToolType
from ...features import FeatureType
from ...operations import PlannedOperation
from ..semantic import MotionSegment, MotionType, Toolpath


class FinishingStrategyMixin:
    """Semi-finishing and finishing strategy implementations."""

    def _semi_finishing(self, op: PlannedOperation) -> Toolpath:
        """Single contour-parallel pass at mid-depth to even out wall stock before finishing.
        Uses the actual feature boundary shape (via _contour_parallel_passes) rather than
        a rectangular approximation, so the allowance is uniform on non-rectangular pockets."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0
        z_target = f.floor_z + self.s.finish_allowance_floor

        # Single contour-parallel pass at half the finish allowance
        contour_passes = self._contour_parallel_passes(
            f, radius, 0.5 * self.s.finish_allowance_wall,
            params.stepover_mm, z_target)

        if not contour_passes:
            tp.metadata["skipped"] = "feature too small for semi-finishing tool"
            self._end_op(tp, op)
            return tp

        # Take only the first (outermost) pass — semi-finishing is one boundary pass
        pass_pts = contour_passes[0]
        pt = pass_pts[0]
        self._link_to(tp, op, pt)
        self._ramp_entry(tp, op, pt, along=np.array([1.0, 0.0, 0.0]))
        for c in pass_pts[1:] + [pass_pts[0]]:
            self._add_cut(tp, op, c)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "semi_offset_pass"
        return tp

    def _finishing(self, op: PlannedOperation) -> Toolpath:
        """Dispatch to the appropriate finishing strategy based on feature type
        and operation notes."""
        f = op.feature
        method = op.notes.get("method", "")

        if f.type is FeatureType.FREEFORM_SURFACE:
            return self._finish_freeform(op)
        if method == "waterline":
            return self._finish_waterline(op)
        if method == "scallop":
            return self._finish_scallop(op)
        if method == "pencil":
            return self._finish_pencil(op)
        if method == "radial":
            return self._finish_radial(op)
        if f.type in (FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE) or (f.type == FeatureType.POCKET and f.notes.get("is_circular", False)):
            return self._helical_interpolation(op)
        if (f.type in (FeatureType.BOSS, FeatureType.CONTOUR) and (f.notes.get("is_circular", False) or (f.diameter is not None and f.diameter > 0 and f.type == FeatureType.BOSS))):
            return self._boss_circular_contour(op)
        return self._finish_wall_floor(op)

    def _boss_circular_contour(self, op: PlannedOperation) -> Toolpath:
        """Circular boss & cylindrical model finishing: full circular G2/G3 arc passes with
        tangential circular lead-in and lead-out in free air outside the workpiece,
        complete with cutter radius compensation (G41/G42)."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        r_boss = (f.diameter / 2.0) if (f.diameter is not None and f.diameter > 0) else (f.radius if (f.radius is not None and f.radius > 0) else (f.bounds_max[0] - f.bounds_min[0]) / 2.0)
        r_tool = tool.diameter / 2.0
        r_path = r_boss + r_tool
        top_z = f.top_z
        floor_z = f.floor_z
        doc = max(params.depth_of_cut_mm, 1e-3)
        bands = max(1, int(np.ceil((top_z - floor_z) / doc)))
        comp_d = tool.tool_number if tool.tool_number else 1
        lead_r = max(r_tool * 0.8, 1.0)

        for b in range(bands):
            z_band = max(floor_z, top_z - (b + 1) * doc) if bands > 1 else floor_z
            approach_pt = self._v3(center[0] + r_path + 2.0 * lead_r, center[1], z_band)
            start_cut_pt = self._v3(center[0] + r_path, center[1], z_band)
            departure_pt = self._v3(center[0] + r_path + 2.0 * lead_r, center[1], z_band)

            self._link_to(tp, op, approach_pt)

            tp.add(MotionSegment(
                motion_type=MotionType.ENTRY,
                start=approach_pt,
                end=start_cut_pt,
                arc_center=self._v3(center[0] + r_path + lead_r, center[1], z_band),
                arc_ccw=True,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata={"cutter_comp": {"side": "left", "action": "on", "d_register": comp_d}}
            ))

            quads = [
                self._v3(center[0], center[1] + r_path, z_band),
                self._v3(center[0] - r_path, center[1], z_band),
                self._v3(center[0], center[1] - r_path, z_band),
                self._v3(center[0] + r_path, center[1], z_band),
            ]
            prev = start_cut_pt
            for q in quads:
                tp.add(MotionSegment(
                    motion_type=MotionType.ARC_CCW,
                    start=prev,
                    end=q,
                    arc_center=self._v3(center[0], center[1], z_band),
                    arc_ccw=True,
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id,
                    operation_id=op.id,
                    feature_id=f.id
                ))
                prev = q

            tp.add(MotionSegment(
                motion_type=MotionType.EXIT,
                start=prev,
                end=departure_pt,
                arc_center=self._v3(center[0] + r_path + lead_r, center[1], z_band),
                arc_ccw=True,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata={"cutter_comp": {"action": "off"}}
            ))

        retract_start = self._seg_end(tp)
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=retract_start,
            end=self._v3(retract_start[0], retract_start[1], self.s.clearance_z),
            tool_id=tool.id,
            operation_id=op.id,
            feature_id=f.id
        ))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "circular_boss_contour"
        return tp

    def _finish_wall_floor(self, op: PlannedOperation) -> Toolpath:
        """Contour finishing: full-depth wall pass + floor pass, engagement-aware."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0
        model_bmin = self.s.setup.stock.bounds_min[:2]
        model_bmax = self.s.setup.stock.bounds_max[:2]
        if hasattr(self.s, "mesh") and self.s.mesh is not None and hasattr(self.s.mesh, "bounds_min"):
            model_bmin = self.s.mesh.bounds_min[:2]
            model_bmax = self.s.mesh.bounds_max[:2]

        bmin = f.bounds_min[:2]
        bmax = f.bounds_max[:2]
        is_open_feat = f.type in (FeatureType.STEP, FeatureType.OPEN_POCKET, FeatureType.OPEN_SLOT, FeatureType.SHOULDER) or f.notes.get("is_open", False)

        if f.type in (FeatureType.CONTOUR, FeatureType.BOSS):
            x_min = bmin[0] - radius
            x_max = bmax[0] + radius
            y_min = bmin[1] - radius
            y_max = bmax[1] + radius
        elif is_open_feat:
            x_min = (bmin[0] - radius) if abs(bmin[0] - model_bmin[0]) < 1.0 else (bmin[0] + radius)
            x_max = (bmax[0] + radius) if abs(bmax[0] - model_bmax[0]) < 1.0 else (bmax[0] - radius)
            y_min = (bmin[1] - radius) if abs(bmin[1] - model_bmin[1]) < 1.0 else (bmin[1] + radius)
            y_max = (bmax[1] + radius) if abs(bmax[1] - model_bmax[1]) < 1.0 else (bmax[1] - radius)
        else:
            x_min = bmin[0] + radius
            x_max = bmax[0] - radius
            y_min = bmin[1] + radius
            y_max = bmax[1] - radius

        if (x_max < x_min - 1e-9) or (y_max < y_min - 1e-9):
            tp.metadata["skipped"] = "tool does not fit for finishing"
            self._end_op(tp, op)
            return tp

        # wall finishing: full depth contour with cutter compensation (G41)
        z = f.top_z
        floor_z = f.floor_z
        corners = [self._v3(x_min, y_min, z), self._v3(x_max, y_min, z),
                   self._v3(x_max, y_max, z), self._v3(x_min, y_max, z)]
        # contour at floor level, then step up (axial spacing uses depth_of_cut,
        # not the radial stepover)
        doc = max(params.depth_of_cut_mm, 1e-3)
        bands = max(1, int(np.ceil((f.top_z - floor_z) / doc)))
        comp_d = tool.tool_number if tool.tool_number else 1
        comp_vec = radius * 0.8  # linear lead-in/out displacement vector

        is_outer = (not f.is_concave) or (f.type in (FeatureType.CONTOUR, FeatureType.BOSS))
        for b in range(bands):
            z_band = min(f.top_z, floor_z + b * doc)
            band_corners = [self._v3(c[0], c[1], z_band) for c in corners]
            c0 = band_corners[0]
            # Approach point for linear G41 engagement:
            # For outer contours: must be in open air outside the model, not inside the edge!
            if is_outer:
                approach_pt = self._v3(c0[0] - comp_vec, c0[1], z_band)
                departure_pt = self._v3(c0[0], c0[1] - comp_vec, z_band)
            else:
                approach_pt = self._v3(c0[0] + comp_vec, c0[1] + comp_vec, z_band)
                departure_pt = self._v3(c0[0] + comp_vec, c0[1] + comp_vec, z_band)

            self._link_to(tp, op, approach_pt)

            # Linear lead-in move engaging cutter compensation (G41)
            prev = self._seg_end(tp)
            tp.add(MotionSegment(
                motion_type=MotionType.ENTRY,
                start=prev,
                end=c0,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata={"cutter_comp": {"side": "left", "action": "on", "d_register": comp_d}}
            ))

            # Cut around closed contour
            for c in band_corners[1:] + [band_corners[0]]:
                self._add_cut(tp, op, c)

            # Linear departure move cancelling cutter compensation (G40)
            prev = self._seg_end(tp)
            tp.add(MotionSegment(
                motion_type=MotionType.EXIT,
                start=prev,
                end=departure_pt,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata={"cutter_comp": {"action": "off"}}
            ))

        # floor finishing: serpentine at floor level (skip for outer contours)
        if f.type is not FeatureType.CONTOUR:
            stepover = max(params.stepover_mm, 1e-3)
            is_open_feat = f.type in (FeatureType.STEP, FeatureType.OPEN_POCKET, FeatureType.OPEN_SLOT, FeatureType.SHOULDER) or f.notes.get("is_open", False)
            model_bmin = self.s.setup.stock.bounds_min[:2]
            model_bmax = self.s.setup.stock.bounds_max[:2]
            if hasattr(self.s, "mesh") and self.s.mesh is not None and hasattr(self.s.mesh, "bounds_min"):
                model_bmin = self.s.mesh.bounds_min[:2]
                model_bmax = self.s.mesh.bounds_max[:2]

            # For closed walls, inset by radius; for open edges, extend past the boundary by radius to clear corners
            x0 = (bmin[0] - radius) if (is_open_feat and abs(bmin[0] - model_bmin[0]) < 1.0) else (bmin[0] + radius)
            x1 = (bmax[0] + radius) if (is_open_feat and abs(bmax[0] - model_bmax[0]) < 1.0) else (bmax[0] - radius)
            y0 = (bmin[1] - radius) if (is_open_feat and abs(bmin[1] - model_bmin[1]) < 1.0) else (bmin[1] + radius)
            y1 = (bmax[1] + radius) if (is_open_feat and abs(bmax[1] - model_bmax[1]) < 1.0) else (bmax[1] - radius)

            if x1 >= x0 and y1 >= y0:
                n_rows = max(1, int(np.ceil((y1 - y0) / stepover)) + 1)
                for r in range(n_rows):
                    y = y0 + min(r * stepover, y1 - y0)
                    xa, xb = (x0, x1) if r % 2 == 0 else (x1, x0)
                    row_start = self._v3(xa, y, floor_z)
                    self._link_to(tp, op, row_start)
                    self._add_cut(tp, op, self._v3(xb, y, floor_z))

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "wall_floor_contour"
        tp.metadata["hsm_mode"] = "finishing"
        return tp

    def _finish_freeform(self, op: PlannedOperation) -> Toolpath:
        """Parallel (raster) finishing with drop-cutter Z from the real mesh,
        auto-detecting steep vs shallow regions for hybrid strategy."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        mesh = self.s.mesh

        scallop = 0.12
        if f.finish is not None and getattr(f.finish, "allowed_scallop_mm", None):
            scallop = f.finish.allowed_scallop_mm
        r = tool.tip_radius if tool.type is ToolType.BALL_ENDMILL else tool.diameter / 2.0
        if r <= 0:
            r = tool.diameter / 2.0
        # scallop -> stepover for ball tool on flat: s = 2*sqrt(2*R*h - h^2)
        stepover = 2.0 * np.sqrt(max(2.0 * r * scallop - scallop * scallop, 1e-6))
        stepover = float(np.clip(stepover, 0.1, tool.diameter * 0.9))

        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        margin = tool.diameter / 2.0
        x0, x1 = bmin[0] - margin, bmax[0] + margin
        y0, y1 = bmin[1] - margin, bmax[1] + margin
        n_rows = max(1, int(np.ceil((y1 - y0) / stepover)) + 1)

        first = True
        for row in range(n_rows):
            y = y0 + min(row * stepover, y1 - y0)
            n_cols = max(2, int(np.ceil((x1 - x0) / max(tool.diameter * 0.35, 0.4))) + 1)
            xs = np.linspace(x0, x1, n_cols)
            pts: list[np.ndarray] = []

            # detect steep/shallow at each sample point
            for x in xs:
                zsurf = mesh.top_height(x, y)
                if zsurf is None:
                    continue
                # estimate slope from neighboring heights
                slope = self._estimate_slope(mesh, x, y, zsurf)
                if slope > 45.0:
                    # steep region: skip parallel raster here; waterline handles it
                    continue
                if tool.type is ToolType.BALL_ENDMILL:
                    z = zsurf - r
                else:
                    z = zsurf
                pts.append(self._v3(x, y, z))

            if not pts:
                continue
            if first:
                start = self._v3(pts[0][0], pts[0][1], self.s.clearance_z)
                self._rapid_to(tp, op, start)
                first = False
            else:
                prev = self._seg_end(tp)
                retract_h = self.s.context.machine.safe_retract_height or 10.0
                link_z = max(prev[2], pts[0][2]) + retract_h
                tp.add(MotionSegment(motion_type=MotionType.LINK, start=prev,
                                     end=self._v3(pts[0][0], pts[0][1], link_z),
                                     tool_id=tool.id, operation_id=op.id,
                                     feature_id=f.id))
            prev = self._seg_end(tp)
            tp.add(MotionSegment(motion_type=MotionType.CUT, start=prev, end=pts[0],
                                 feed=params.plunge_feed, spindle=params.spindle_rpm,
                                 tool_id=tool.id, operation_id=op.id,
                                 feature_id=f.id))
            for p in pts[1:]:
                self._add_cut(tp, op, p)

        if tp.segments:
            last = self._seg_end(tp)
            tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                                 end=self._v3(last[0], last[1], self.s.clearance_z),
                                 tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "parallel_drop_cutter"
        tp.metadata["scallop_mm"] = scallop
        return tp

    def _estimate_slope(self, mesh, x: float, y: float, z_center: float,
                        delta: float = 0.5) -> float:
        """Estimate surface slope angle in degrees at (x,y) using finite differences."""
        dzdx = 0.0
        dzdy = 0.0
        h_left = mesh.top_height(x - delta, y)
        h_right = mesh.top_height(x + delta, y)
        h_back = mesh.top_height(x, y - delta)
        h_fwd = mesh.top_height(x, y + delta)
        if h_left is not None and h_right is not None:
            dzdx = (h_right - h_left) / (2.0 * delta)
        elif h_left is not None:
            dzdx = (z_center - h_left) / delta
        elif h_right is not None:
            dzdx = (h_right - z_center) / delta
        if h_back is not None and h_fwd is not None:
            dzdy = (h_fwd - h_back) / (2.0 * delta)
        elif h_back is not None:
            dzdy = (z_center - h_back) / delta
        elif h_fwd is not None:
            dzdy = (h_fwd - z_center) / delta
        return math.degrees(math.atan2(math.sqrt(dzdx * dzdx + dzdy * dzdy), 1.0))

    def _finish_waterline(self, op: PlannedOperation) -> Toolpath:
        """Constant-Z contour passes at discrete Z levels along steep walls.
        Z levels spaced by stepover to achieve target scallop height."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0

        scallop = 0.12
        if f.finish is not None and getattr(f.finish, "allowed_scallop_mm", None):
            scallop = f.finish.allowed_scallop_mm
        r = tool.tip_radius if tool.type is ToolType.BALL_ENDMILL else radius

        # Z level spacing for target scallop
        if r > 0:
            z_step = 2.0 * math.sqrt(max(2.0 * r * scallop - scallop * scallop, 1e-6))
        else:
            z_step = scallop
        z_step = float(np.clip(z_step, 0.15, params.stepover_mm))

        floor_z = f.floor_z + self.s.finish_allowance_floor
        top_z = f.top_z
        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        extents = bmax[:2] - bmin[:2]
        wall_inset = radius

        first_entry = True
        z = top_z
        passes_count = 0
        while z > floor_z + 1e-9:
            z = max(floor_z, z - z_step)
            passes_count += 1
            # contour at this Z level using mesh ray casting
            hx = extents[0] / 2.0 - wall_inset
            hy = extents[1] / 2.0 - wall_inset
            if hx < -1e-9 or hy < -1e-9:
                continue
            # sample points along rectangular contour, adjusting Z via mesh
            n_pts = max(4, int(np.ceil(max(extents) / max(tool.diameter * 0.35, 0.35))))
            contour_pts = []
            perimeter = 2 * (extents[0] + extents[1])
            for i in range(n_pts):
                t = i / n_pts
                dist = t * perimeter
                if dist < extents[0]:
                    px = bmin[0] + dist + radius
                    py = bmin[1] + radius
                elif dist < extents[0] + extents[1]:
                    px = bmax[0] - radius
                    py = bmin[1] + (dist - extents[0]) + radius
                elif dist < 2 * extents[0] + extents[1]:
                    px = bmax[0] - (dist - extents[0] - extents[1]) - radius
                    py = bmax[1] - radius
                else:
                    px = bmin[0] + radius
                    py = bmax[1] - (dist - 2 * extents[0] - extents[1]) - radius
                # use mesh to find actual Z at this XY for the wall contact
                zsurf = self.s.mesh.top_height(px, py)
                if zsurf is not None and abs(zsurf - z) < z_step * 1.5:
                    if tool.type is ToolType.BALL_ENDMILL:
                        z_tool = z - r
                    else:
                        z_tool = z
                    contour_pts.append(self._v3(px, py, z_tool))

            if not contour_pts:
                continue
            if first_entry:
                self._link_to(tp, op, contour_pts[0])
                self._ramp_entry(tp, op, contour_pts[0],
                                 along=np.array([1.0, 0, 0]))
                first_entry = False
            else:
                prev_end = self._seg_end(tp)
                dist = np.linalg.norm(contour_pts[0][:2] - prev_end[:2])
                if dist < tool.diameter:
                    self._link_stay_down(tp, op, contour_pts[0])
                else:
                    self._link_to(tp, op, contour_pts[0])
            for pt in contour_pts[1:]:
                self._add_cut(tp, op, pt)
            self._add_cut(tp, op, contour_pts[0])

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "waterline"
        tp.metadata["passes"] = passes_count
        return tp

    def _finish_scallop(self, op: PlannedOperation) -> Toolpath:
        """Follows the surface contour with constant stepover computed from target
        scallop, adapting stepover based on local surface curvature."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0
        mesh = self.s.mesh

        scallop = 0.12
        if f.finish is not None and getattr(f.finish, "allowed_scallop_mm", None):
            scallop = f.finish.allowed_scallop_mm
        r = tool.tip_radius if tool.type is ToolType.BALL_ENDMILL else radius
        if r <= 0:
            r = radius

        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        margin = tool.diameter / 2.0
        x0, x1 = bmin[0] - margin, bmax[0] + margin
        y0, y1 = bmin[1] - margin, bmax[1] + margin

        # adaptive stepover based on local curvature
        base_stepover = 2.0 * np.sqrt(max(2.0 * r * scallop - scallop * scallop, 1e-6))
        base_stepover = float(np.clip(base_stepover, 0.1, tool.diameter * 0.9))

        n_rows = max(1, int(np.ceil((y1 - y0) / base_stepover)) + 1)
        first = True
        for row in range(n_rows):
            y = y0 + min(row * base_stepover, y1 - y0)
            n_cols = max(2, int(np.ceil((x1 - x0) / max(tool.diameter * 0.35, 0.35))) + 1)
            xs = np.linspace(x0, x1, n_cols)
            pts: list[np.ndarray] = []
            for x in xs:
                zsurf = mesh.top_height(x, y)
                if zsurf is None:
                    continue
                # local curvature estimation
                curvature = self._estimate_curvature(mesh, x, y, zsurf)
                # adapt stepover: tighter in high curvature areas
                local_stepover = base_stepover / max(1.0, abs(curvature) * 10.0)
                local_stepover = max(local_stepover, base_stepover * 0.3)
                if tool.type is ToolType.BALL_ENDMILL:
                    z = zsurf - r
                else:
                    z = zsurf
                pts.append(self._v3(x, y, z))

            if not pts:
                continue
            if first:
                start = self._v3(pts[0][0], pts[0][1], self.s.clearance_z)
                self._rapid_to(tp, op, start)
                first = False
            else:
                prev = self._seg_end(tp)
                retract_h = self.s.context.machine.safe_retract_height or 10.0
                link_z = max(prev[2], pts[0][2]) + retract_h
                tp.add(MotionSegment(motion_type=MotionType.LINK, start=prev,
                                     end=self._v3(pts[0][0], pts[0][1], link_z),
                                     tool_id=tool.id, operation_id=op.id,
                                     feature_id=f.id))
            prev = self._seg_end(tp)
            tp.add(MotionSegment(motion_type=MotionType.CUT, start=prev, end=pts[0],
                                 feed=params.plunge_feed, spindle=params.spindle_rpm,
                                 tool_id=tool.id, operation_id=op.id,
                                 feature_id=f.id))
            for p in pts[1:]:
                self._add_cut(tp, op, p)

        if tp.segments:
            last = self._seg_end(tp)
            tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                                 end=self._v3(last[0], last[1], self.s.clearance_z),
                                 tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "scallop_uniform"
        tp.metadata["scallop_mm"] = scallop
        return tp

    def _estimate_curvature(self, mesh, x: float, y: float, z_center: float,
                            delta: float = 0.5) -> float:
        """Estimate surface curvature (second derivative) at (x,y)."""
        h_left = mesh.top_height(x - delta, y)
        h_right = mesh.top_height(x + delta, y)
        h_back = mesh.top_height(x, y - delta)
        h_fwd = mesh.top_height(x, y + delta)
        d2x = 0.0
        d2y = 0.0
        if h_left is not None and h_right is not None:
            d2x = (h_right - 2.0 * z_center + h_left) / (delta * delta)
        if h_back is not None and h_fwd is not None:
            d2y = (h_fwd - 2.0 * z_center + h_back) / (delta * delta)
        return math.sqrt(d2x * d2x + d2y * d2y)

    def _finish_pencil(self, op: PlannedOperation) -> Toolpath:
        """Single-pass toolpath that traces concave corners/intersections of surfaces.
        Used for corner cleanup after main finishing."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        mesh = self.s.mesh

        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        margin = tool.diameter
        x0, x1 = bmin[0] - margin, bmax[0] + margin
        y0, y1 = bmin[1] - margin, bmax[1] + margin

        # pencil trace: find concave edges by detecting sign change in curvature
        pencil_pts: list[np.ndarray] = []
        scan_step = tool.diameter * 0.15
        n_x = max(2, int(np.ceil((x1 - x0) / scan_step)) + 1)
        n_y = max(2, int(np.ceil((y1 - y0) / scan_step)) + 1)
        xs = np.linspace(x0, x1, n_x)
        ys = np.linspace(y0, y1, n_y)

        for x in xs:
            for y in ys:
                zsurf = mesh.top_height(x, y)
                if zsurf is None:
                    continue
                curvature = self._estimate_curvature(mesh, x, y, zsurf)
                if curvature > 0.5:  # concave corner threshold
                    if tool.type is ToolType.BALL_ENDMILL:
                        z = zsurf - tool.tip_radius
                    else:
                        z = zsurf
                    pencil_pts.append(self._v3(x, y, z))

        if not pencil_pts:
            tp.metadata["skipped"] = "no concave corners detected"
            self._end_op(tp, op)
            return tp

        # sort pencil points by proximity (nearest neighbor)
        ordered = self._nearest_neighbor_sort(pencil_pts)

        first = True
        for pt in ordered:
            if first:
                self._link_to(tp, op, pt)
                first = False
            else:
                prev_end = self._seg_end(tp)
                dist = np.linalg.norm(pt[:2] - prev_end[:2])
                if dist < tool.diameter:
                    self._link_stay_down(tp, op, pt)
                else:
                    self._link_to(tp, op, pt)
            self._add_cut(tp, op, pt)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "pencil_trace"
        return tp

    @staticmethod
    def _nearest_neighbor_sort(points: list[np.ndarray]) -> list[np.ndarray]:
        """Simple nearest-neighbor ordering of points."""
        if len(points) <= 1:
            return points[:]
        used = [False] * len(points)
        result = [points[0]]
        used[0] = True
        for _ in range(len(points) - 1):
            last = result[-1]
            best_dist = float('inf')
            best_idx = -1
            for i, pt in enumerate(points):
                if used[i]:
                    continue
                d = float(np.linalg.norm(pt[:2] - last[:2]))
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0:
                used[best_idx] = True
                result.append(points[best_idx])
        return result

    def _finish_radial(self, op: PlannedOperation) -> Toolpath:
        """Fan-shaped pattern from center point for circular features,
        or spiral pattern for circular pocket finishing."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0

        center = f.center_xy
        r_hole = f.diameter / 2.0 if f.diameter else float(np.min(f.xy_extent)) / 2.0
        floor_z = f.floor_z
        top_z = f.top_z

        scallop = 0.05
        if f.finish is not None and getattr(f.finish, "allowed_scallop_mm", None):
            scallop = f.finish.allowed_scallop_mm
        r_tool = tool.tip_radius if tool.type is ToolType.BALL_ENDMILL else radius
        if r_tool <= 0:
            r_tool = radius

        # radial stepover from scallop
        radial_step = 2.0 * math.sqrt(max(2.0 * r_tool * scallop - scallop * scallop, 1e-6))
        radial_step = float(np.clip(radial_step, 0.1, params.stepover_mm))

        # spiral from center outward
        n_rings = max(1, int(math.ceil((r_hole - radius) / radial_step)))
        z = top_z
        passes = 0
        while z > floor_z + 1e-9:
            z = max(floor_z, z - params.depth_of_cut_mm)
            passes += 1
            if passes == 1:
                self._link_to(tp, op, self._v3(center[0], center[1], z))
            else:
                # retract between Z-passes
                last = self._seg_end(tp)
                tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                                     end=self._v3(last[0], last[1], self.s.clearance_z),
                                     tool_id=tool.id, operation_id=op.id, feature_id=f.id))
                self._link_to(tp, op, self._v3(center[0], center[1], z))
            # spiral path
            total_angle = n_rings * 2 * math.pi
            n_pts = max(12, n_rings * 24)
            for i in range(n_pts + 1):
                t = i / n_pts
                angle = t * total_angle
                r = min(radius + t * (r_hole - radius), r_hole - radius)
                px = center[0] + r * math.cos(angle)
                py = center[1] + r * math.sin(angle)
                pt = self._v3(px, py, z)
                if i == 0 and passes == 1:
                    self._ramp_entry(tp, op, pt, along=np.array([1.0, 0, 0]))
                else:
                    self._add_cut(tp, op, pt)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "radial_spiral"
        tp.metadata["passes"] = passes
        return tp
