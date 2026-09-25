"""Facing, 2D contour-parallel roughing, adaptive/trochoidal clearing, and rest machining."""

from __future__ import annotations

import math
import numpy as np

from ...features import FeatureType
from ...operations import PlannedOperation
from ..semantic import MotionSegment, MotionType, Toolpath


class RoughingStrategyMixin:
    """Facing, roughing, adaptive/trochoidal clearing, and rest machining strategies."""

    def _facing(self, op: PlannedOperation) -> Toolpath:
        """Face the excess stock above a top face: zig-zag over the region that
        actually contains excess material (from the stock heightmap)."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params

        excess = f.notes.get("excess_height", 0.0)
        if excess <= 1e-6:
            tp.metadata["skipped"] = "no excess stock"
            self._end_op(tp, op)
            return tp

        stock_bmin = self.s.setup.stock.bounds_min[:2]
        stock_bmax = self.s.setup.stock.bounds_max[:2]
        region_min = np.maximum(f.bounds_min[:2], stock_bmin)
        region_max = np.minimum(f.bounds_max[:2], stock_bmax)
        top_z = f.top_z          # stock top
        target_z = f.floor_z     # model top face
        stepdown = params.depth_of_cut_mm
        stepover = params.stepover_mm
        radius = tool.diameter / 2.0

        # region must be extended by tool radius so the cutter fully covers edges
        r_min = region_min - radius
        r_max = region_max + radius
        span = r_max - r_min
        n_rows = max(1, int(np.ceil(span[1] / stepover)) + 1)

        z = top_z
        passes = 0
        while z > target_z + 1e-9:
            passes += 1
            z = max(target_z, top_z - passes * stepdown)
            direction = 1 if passes % 2 == 1 else -1
            for row in range(n_rows):
                y_index = row if direction == 1 else n_rows - 1 - row
                y = r_min[1] + min(y_index * stepover, span[1])
                x0, x1 = (r_min[0], r_max[0]) if row % 2 == 0 else (r_max[0], r_min[0])
                if row == 0:
                    self._link_to(tp, op, self._v3(x0, y, z))
                else:
                    self._add_cut(tp, op, self._v3(x0, y, z))
                self._add_cut(tp, op, self._v3(x1, y, z))
        # exit and retract
        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["passes"] = passes
        return tp

    def _roughing(self, op: PlannedOperation) -> Toolpath:
        """Boundary-conforming contour-parallel clearing: true polygon offsets
        from the feature boundary, with engagement-aware corner reduction."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0

        floor_z = f.floor_z + self.s.finish_allowance_floor
        top_z = f.top_z
        stepdown = params.depth_of_cut_mm
        stepover = params.stepover_mm
        finish_wall = self.s.finish_allowance_wall

        z = top_z
        first_entry = True
        passes = 0
        while z > floor_z + 1e-9:
            z = max(floor_z, z - stepdown)
            passes += 1
            contour_passes = self._contour_parallel_passes(
                f, radius, finish_wall, stepover, z)
            for pts in contour_passes:
                if not pts:
                    continue
                if first_entry:
                    self._link_to(tp, op, pts[0])
                    # Helical ramp entry is only needed for enclosed concave pockets.
                    # Outer contours, bosses, steps, and open regions enter from free air/stock margin.
                    if f.is_concave and f.type not in (FeatureType.CONTOUR, FeatureType.BOSS, FeatureType.FACING_REGION, FeatureType.STEP):
                        self._helical_ramp(tp, op, pts[0],
                                           helix_radius=min(radius * 0.8, stepover * 0.4),
                                           pitch=min(stepdown, tool.diameter * 0.3))
                    first_entry = False
                else:
                    # stay-down link for nearby contour points
                    prev_end = self._seg_end(tp)
                    dist = np.linalg.norm(pts[0][:2] - prev_end[:2])
                    if dist < tool.diameter * 2.0:
                        self._link_stay_down(tp, op, pts[0])
                    else:
                        self._link_to(tp, op, pts[0])
                for pt in pts[1:]:
                    self._add_cut(tp, op, pt)
                # close the contour
                if len(pts) > 2:
                    self._add_cut(tp, op, pts[0])
        # retract
        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["passes"] = passes
        tp.metadata["strategy"] = "contour_parallel"
        return tp

    def _adaptive_roughing(self, op: PlannedOperation) -> Toolpath:
        """Trochoidal / high-efficiency milling with constant engagement angle.
        For slotting (width < 1.5 * tool_dia): trochoidal loops.
        For pockets: HECM with reduced radial engagement and full axial depth."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0

        floor_z = f.floor_z + self.s.finish_allowance_floor
        top_z = f.top_z
        stepdown = params.depth_of_cut_mm
        finish_wall = self.s.finish_allowance_wall

        extent = f.xy_extent
        is_slot = min(extent) < 1.5 * tool.diameter

        if is_slot:
            self._trochoidal_slot(tp, op, f, tool, params, top_z, floor_z,
                                  stepdown, finish_wall)
        else:
            self._hecm_pocket(tp, op, f, tool, params, top_z, floor_z,
                              stepdown, finish_wall)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "adaptive_trochoidal"
        return tp

    def _trochoidal_slot(self, tp, op, f, tool, params, top_z, floor_z,
                         stepdown, finish_wall):
        """Trochoidal loops for slot cutting: small circles moving along slot axis."""
        radius = tool.diameter / 2.0
        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        # slot direction: longest axis
        dx = bmax[0] - bmin[0]
        dy = bmax[1] - bmin[1]
        if dx >= dy:
            slot_dir = np.array([1.0, 0.0])
            perp_dir = np.array([0.0, 1.0])
            slot_length = dx
            slot_width = dy
        else:
            slot_dir = np.array([0.0, 1.0])
            perp_dir = np.array([1.0, 0.0])
            slot_length = dy
            slot_width = dx

        loop_radius = min(params.stepover_mm * 0.5, radius * 0.8)
        loop_spacing = loop_radius * 2.0  # advance per loop
        n_loops = max(1, int(math.ceil(slot_length / loop_spacing)))
        z = top_z
        first_entry = True
        while z > floor_z + 1e-9:
            z = max(floor_z, z - stepdown)
            center_2d = np.array([0.5 * (bmin[0] + bmax[0]),
                                  0.5 * (bmin[1] + bmax[1])])
            start_2d = center_2d - 0.5 * slot_length * slot_dir
            for i in range(n_loops):
                t = (i + 0.5) / n_loops
                center = start_2d + t * slot_length * slot_dir
                # trochoidal point: circle center moves along slot, tool oscillates perpendicularly
                n_arc = 8
                for k in range(n_arc):
                    angle = 2.0 * math.pi * k / n_arc
                    tool_pt_2d = center + loop_radius * (
                        math.cos(angle) * perp_dir + math.sin(angle) * slot_dir)
                    pt = self._v3(tool_pt_2d[0], tool_pt_2d[1], z)
                    if first_entry:
                        self._link_to(tp, op, pt)
                        self._helical_ramp(tp, op, pt,
                                           helix_radius=loop_radius * 0.5,
                                           pitch=min(stepdown, tool.diameter * 0.3))
                        first_entry = False
                    else:
                        self._add_cut(tp, op, pt)
            # reconnect between depth passes
            last = self._seg_end(tp)
            tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                                 end=self._v3(last[0], last[1], self.s.clearance_z),
                                 tool_id=tool.id, operation_id=op.id,
                                 feature_id=f.id))

    def _hecm_pocket(self, tp, op, f, tool, params, top_z, floor_z,
                     stepdown, finish_wall):
        """High-efficiency cavity milling: full axial depth, reduced radial stepover,
        with trochoidal motion to maintain constant engagement."""
        radius = tool.diameter / 2.0
        finish_wall = self.s.finish_allowance_wall

        # Use actual feature boundary polygon (not bounding box)
        region = self._feature_boundary_polygon(f)
        boundary = region.outer
        if not boundary or len(boundary) < 3:
            return

        # reduced stepover for adaptive: 10-20% of tool diameter
        ae = tool.diameter * 0.15
        # engagement-aware feed: estimate engagement angle from stepover
        eng_angle = self._estimate_engagement(tool.diameter, ae)
        adapted_feed = self._adaptive_feed(params.feed_rate, eng_angle, 180.0)
        # full depth of cut per level
        z = top_z
        first_entry = True
        while z > floor_z + 1e-9:
            z_level = max(floor_z, z - stepdown)
            # contour-parallel offset paths using actual boundary polygon
            inset0 = radius + finish_wall
            # generate offset contours inward from original boundary (not accumulating)
            offset_pass = 0
            while True:
                inset = inset0 + offset_pass * ae
                offset_poly = self._offset_polygon(boundary, -inset)
                if not offset_poly or len(offset_poly) < 3:
                    break
                # check if offset polygon is too small
                areas = [abs(self._polygon_area(offset_poly))]
                if areas[0] < 1e-6:
                    break
                # generate cut path along offset polygon
                pts_3d = [self._v3(p[0], p[1], z_level) for p in offset_poly]
                if first_entry:
                    self._link_to(tp, op, pts_3d[0])
                    # Helical ramp entry is only needed for enclosed concave pockets.
                    # Outer contours, bosses, steps, and open regions enter from free air/stock margin.
                    if f.is_concave and f.type not in (FeatureType.CONTOUR, FeatureType.BOSS, FeatureType.FACING_REGION, FeatureType.STEP):
                        self._helical_ramp(tp, op, pts_3d[0],
                                           helix_radius=min(ae, radius * 0.3),
                                           pitch=min(stepdown, tool.diameter * 0.3))
                    first_entry = False
                else:
                    prev_end = self._seg_end(tp)
                    dist = np.linalg.norm(pts_3d[0][:2] - prev_end[:2])
                    if dist < tool.diameter:
                        self._link_stay_down(tp, op, pts_3d[0])
                    else:
                        self._link_to(tp, op, pts_3d[0])
                for pt in pts_3d[1:]:
                    self._add_cut(tp, op, pt, feed=adapted_feed)
                self._add_cut(tp, op, pts_3d[0], feed=adapted_feed)  # close the loop
                offset_pass += 1

    def _rest_machining(self, op: PlannedOperation) -> Toolpath:
        """Machine only regions where remaining stock stands above the target:
        driven by the current stock heightmap (from simulation of prior ops)."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        radius = tool.diameter / 2.0

        if self.s.stock_voxels is None:
            tp.metadata["skipped"] = "no stock voxel data available for rest machining"
            self._end_op(tp, op)
            return tp

        hm = self.s.stock_voxels.heightmap()  # remaining stock top per column
        target_z = f.floor_z + self.s.finish_allowance_floor
        bmin, bmax = f.bounds_min, f.bounds_max
        vs = self.s.stock_voxels.voxel_size

        idx0 = self.s.stock_voxels.world_to_index(np.array([bmin[0] + radius, bmin[1] + radius, bmin[2]]))
        idx1 = self.s.stock_voxels.world_to_index(np.array([bmax[0] - radius, bmax[1] - radius, bmax[2]]))
        i0, j0 = int(idx0[0]), int(idx0[1])
        i1, j1 = int(idx1[0]), int(idx1[1])
        i0, i1 = max(i0, 0), min(i1, hm.shape[0] - 1)
        j0, j1 = max(j0, 0), min(j1, hm.shape[1] - 1)
        if i1 < i0 or j1 < j0:
            tp.metadata["skipped"] = "no remaining stock in feature"
            self._end_op(tp, op)
            return tp

        step = max(1, int(np.ceil(tool.diameter / vs / 2)))
        cut_points: list[tuple[float, float, float]] = []
        for i in range(i0, i1 + 1, step):
            for j in range(j0, j1 + 1, step):
                h = hm[i, j]
                if np.isnan(h):
                    continue
                if h > target_z + 1e-6:
                    w = self.s.stock_voxels.index_to_world(np.array([i, j, 0]))
                    cut_points.append((w[0], w[1], min(h, f.top_z)))
        if not cut_points:
            tp.metadata["skipped"] = "rest machining not needed"
            self._end_op(tp, op)
            return tp

        # order points into serpentine rows by Y band
        cut_points.sort(key=lambda p: (round(p[1] / (tool.diameter * 0.8)), p[0]))
        first = True
        for p in cut_points:
            pt = self._v3(p[0], p[1], target_z)
            if first:
                self._link_to(tp, op, pt)
                self._ramp_entry(tp, op, pt, along=np.array([1.0, 0, 0]))
                first = False
            else:
                prev = self._seg_end(tp)
                dist = np.linalg.norm(pt[:2] - prev[:2])
                if dist > 1e-9:
                    self._add_cut(tp, op, pt)
        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "rest_heightmap"
        return tp
