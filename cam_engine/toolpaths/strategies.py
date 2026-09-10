"""Toolpath strategy implementations.

Every strategy consumes real geometry (feature bounds, mesh height fields,
remaining-stock heightmaps) plus tool geometry and cutting parameters.
Engagement is an explicit input, not an afterthought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..context import PlanningContext, Setup, Tool, ToolType
from ..coords import CoordSpace
from ..errors import CamError, INVALID_TOOLPATH, UNSUPPORTED_FEATURE
from ..features import FeatureType, MachiningFeature
from ..machinability import CuttingParameters
from ..operations import OpPurpose, PlannedOperation
from .semantic import MotionSegment, MotionType, Toolpath


@dataclass
class StrategyContext:
    """Everything a strategy needs, drawn from the planning context."""

    context: PlanningContext
    setup: Setup
    mesh: object                     # TriangleMesh of model in setup space
    stock_voxels: object             # VoxelStock: current remaining stock state
    clearance_z: float               # setup-space Z for safe rapids (derived)
    finish_allowance_wall: float     # from config, domain default
    finish_allowance_floor: float


class StrategyEngine:
    """Generates semantic toolpaths for planned operations."""

    def __init__(self, strat: StrategyContext):
        self.s = strat

    # ------------------------------------------------------------------
    def generate(self, op: PlannedOperation) -> Toolpath:
        if op.purpose is OpPurpose.FACING:
            return self._facing(op)
        if op.purpose is OpPurpose.ROUGHING:
            if op.notes.get("method") == "adaptive":
                return self._adaptive_roughing(op)
            return self._roughing(op)
        if op.purpose is OpPurpose.REST_MACHINING:
            return self._rest_machining(op)
        if op.purpose is OpPurpose.SEMI_FINISHING:
            return self._semi_finishing(op)
        if op.purpose is OpPurpose.FINISHING:
            return self._finishing(op)
        if op.purpose is OpPurpose.DRILLING:
            return self._drilling(op)
        if op.purpose is OpPurpose.BORING:
            return self._boring(op)
        if op.purpose is OpPurpose.SPOT_DRILLING:
            return self._spot_drilling(op)
        if op.purpose is OpPurpose.REAMING:
            return self._reaming(op)
        if op.purpose is OpPurpose.TAPPING:
            return self._tapping(op)
        if op.purpose is OpPurpose.CHAMFERING:
            return self._chamfering(op)
        raise CamError(UNSUPPORTED_FEATURE,
                       f"no strategy for purpose {op.purpose}", stage="toolpath",
                       operation_id=op.id)

    # ---------------------------------------------------------- helpers --
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

    def _rapid_to(self, tp: Toolpath, op: PlannedOperation, p: np.ndarray) -> None:
        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RAPID, start=last,
                             end=np.asarray(p, float),
                             tool_id=op.tool.id, operation_id=op.id,
                             feature_id=op.feature.id))

    # ---- arc / helical primitives ----
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

    # ---- linking primitives ----
    def _link_to(self, tp: Toolpath, op: PlannedOperation, p: np.ndarray) -> None:
        """Retract to clearance, rapid, descend to feed height (with safe retract)."""
        last = self._seg_end(tp)
        safe_z = p[2] + self.s.context.machine.safe_retract_height + \
            max(1.0, op.tool.diameter * 0.5)
        # retract
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT, start=last,
            end=self._v3(last[0], last[1], self.s.clearance_z),
            tool_id=op.tool.id, operation_id=op.id, feature_id=op.feature.id))
        # rapid across
        self._rapid_to(tp, op, self._v3(p[0], p[1], self.s.clearance_z))
        # descend to feed start height
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID,
            start=self._v3(p[0], p[1], self.s.clearance_z),
            end=self._v3(p[0], p[1], safe_z),
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
        # descend
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID,
            start=self._v3(p[0], p[1], retract_z),
            end=self._v3(p[0], p[1], p[2]),
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
        arc_start = self._v3(prev[0], prev[1], target[2])
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
        top_z = self._seg_end(tp)[2]
        bottom_z = target[2]
        self._helical_entry(tp, op, target, helix_radius, top_z, bottom_z, pitch)
        # finish at exact target point
        prev = self._seg_end(tp)
        self._add_cut(tp, op, target, feed=op.params.feed_rate)

    # ---- polygon offset helpers for boundary-conforming roughing ----
    @staticmethod
    def _offset_polygon(polygon: list[np.ndarray], offset: float) -> list[np.ndarray]:
        """Offset a 2D polygon inward (negative offset) or outward (positive).
        Uses Minkowski-sum clipping via vertex normal approach."""
        n = len(polygon)
        if n < 3:
            return polygon[:]
        result = []
        for i in range(n):
            p = polygon[i]
            prev_pt = polygon[(i - 1) % n]
            next_pt = polygon[(i + 1) % n]
            # edge vectors
            e1 = p - prev_pt
            e2 = next_pt - p
            # inward normals (pointing left of edge direction)
            n1 = np.array([-e1[1], e1[0]], dtype=float)
            n2 = np.array([-e2[1], e2[0]], dtype=float)
            n1_len = np.linalg.norm(n1)
            n2_len = np.linalg.norm(n2)
            if n1_len < 1e-12 or n2_len < 1e-12:
                result.append(p.copy())
                continue
            n1 /= n1_len
            n2 /= n2_len
            # bisector
            bisector = n1 + n2
            bisector_len = np.linalg.norm(bisector)
            if bisector_len < 1e-12:
                result.append(p + n1 * offset)
                continue
            bisector /= bisector_len
            # offset distance along bisector
            dot = n1 @ bisector
            if abs(dot) < 1e-12:
                result.append(p + n1 * offset)
                continue
            d = offset / dot
            result.append(p + bisector * d)
        return result

    def _feature_boundary_polygon(self, f: MachiningFeature) -> list[np.ndarray]:
        """Extract a 2D boundary polygon from the feature's wall faces.
        Falls back to bounding-box if wall face geometry is unavailable."""
        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        # default: bounding-box rectangle
        return [
            np.array([bmin[0], bmin[1]]),
            np.array([bmax[0], bmin[1]]),
            np.array([bmax[0], bmax[1]]),
            np.array([bmin[0], bmax[1]]),
        ]

    def _contour_parallel_passes(self, f: MachiningFeature, tool_radius: float,
                                 finish_wall: float, stepover: float,
                                 z: float) -> list[list[np.ndarray]]:
        """Generate offset polygon contour passes at a given Z level."""
        boundary = self._feature_boundary_polygon(f)
        inset0 = tool_radius + finish_wall
        bmin = np.min(boundary, axis=0)
        bmax = np.max(boundary, axis=0)
        extents = bmax - bmin
        min_extent = float(np.min(extents))
        n_offsets = max(1, int(np.ceil((min_extent / 2.0 - inset0) / stepover)) + 1)
        passes = []
        for oi in range(n_offsets):
            inset = inset0 + oi * stepover
            offset_poly = self._offset_polygon(boundary, -inset)
            # filter degenerate offsets
            area = self._polygon_area(offset_poly)
            if area < 1e-6:
                continue
            # convert to 3D points at z
            pts = [self._v3(p[0], p[1], z) for p in offset_poly]
            passes.append(pts)
        return passes

    @staticmethod
    def _polygon_area(poly: list[np.ndarray]) -> float:
        """Shoelace formula for signed area."""
        n = len(poly)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i][0] * poly[j][1]
            area -= poly[j][0] * poly[i][1]
        return abs(area) * 0.5

    # ----------------------------------------------------------- facing --
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
            z = max(target_z, z - stepdown)
            passes += 1
            direction = 1 if passes % 2 == 1 else -1
            for row in range(n_rows):
                y_index = row if direction == 1 else n_rows - 1 - row
                y = r_min[1] + min(y_index * stepover, span[1])
                x0, x1 = (r_min[0], r_max[0]) if row % 2 == 0 else (r_max[0], r_min[0])
                if row == 0:
                    self._link_to(tp, op, self._v3(x0, y, z))
                    self._ramp_entry(tp, op, self._v3(x0, y, z),
                                     along=np.array([1.0, 0.0, 0.0]))
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

    # -------------------------------------------------------- roughing --
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

    def _ramp_entry(self, tp: Toolpath, op: PlannedOperation, target: np.ndarray,
                    along: np.ndarray) -> None:
        """Ramp entry into material along `along` direction, respecting ramp feed."""
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
        horizontal = np.array([along[0], along[1]], dtype=float)
        nh = np.linalg.norm(horizontal)
        if nh < 1e-9:
            horizontal = np.array([1.0, 0.0])
            nh = 1.0
        horizontal = horizontal / nh
        ramp_len = abs(dz) / math.tan(math.radians(15.0))
        a = end[:2] - horizontal * ramp_len
        start = self._v3(a[0], a[1], end[2] - dz)
        tp.add(MotionSegment(motion_type=MotionType.RAMP, start=start, end=end,
                             feed=params.ramp_feed, spindle=params.spindle_rpm,
                             tool_id=op.tool.id, operation_id=op.id,
                             feature_id=op.feature.id))

    # ---- adaptive / trochoidal roughing ----
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
        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        cx = 0.5 * (bmin[0] + bmax[0])
        cy = 0.5 * (bmin[1] + bmax[1])
        extents = bmax - bmin
        finish_wall = self.s.finish_allowance_wall

        # reduced stepover for adaptive: 10-20% of tool diameter
        ae = tool.diameter * 0.15
        # full depth of cut per level
        z = top_z
        first_entry = True
        while z > floor_z + 1e-9:
            z_level = max(floor_z, z - stepdown)
            # contour-parallel offset paths with reduced ae
            inset0 = radius + finish_wall
            n_offsets = max(1, int(np.ceil((min(extents) / 2.0 - inset0) / ae)) + 1)
            for oi in range(n_offsets):
                inset = inset0 + oi * ae
                hx = extents[0] / 2.0 - inset
                hy = extents[1] / 2.0 - inset
                if hx < -1e-9 or hy < -1e-9:
                    continue
                # offset rectangle corners
                corners = [
                    self._v3(cx - hx, cy - hy, z_level),
                    self._v3(cx + hx, cy - hy, z_level),
                    self._v3(cx + hx, cy + hy, z_level),
                    self._v3(cx - hx, cy + hy, z_level),
                ]
                if hx <= 1e-9 and hy <= 1e-9:
                    corners = corners[:1]
                if first_entry and oi == 0:
                    self._link_to(tp, op, corners[0])
                    self._helical_ramp(tp, op, corners[0],
                                       helix_radius=min(hx, hy) * 0.4,
                                       pitch=min(stepdown, tool.diameter * 0.3))
                    first_entry = False
                else:
                    prev_end = self._seg_end(tp)
                    dist = np.linalg.norm(corners[0][:2] - prev_end[:2])
                    if dist < tool.diameter:
                        self._link_stay_down(tp, op, corners[0])
                    else:
                        self._link_to(tp, op, corners[0])
                for pt in corners[1:]:
                    self._add_cut(tp, op, pt)
                self._add_cut(tp, op, corners[0])

    # --------------------------------------------------- rest machining --
    def _rest_machining(self, op: PlannedOperation) -> Toolpath:
        """Machine only regions where remaining stock stands above the target:
        driven by the current stock heightmap (from simulation of prior ops)."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
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

    # -------------------------------------------------- semi-finishing --
    def _semi_finishing(self, op: PlannedOperation) -> Toolpath:
        """Single offset pass at mid-depth to even out wall stock before finishing."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0
        bmin, bmax = f.bounds_min, f.bounds_max
        extents = bmax[:2] - bmin[:2]
        z_target = f.floor_z + self.s.finish_allowance_floor
        inset = radius + 0.5 * self.s.finish_allowance_wall
        cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
        hx = extents[0] / 2.0 - inset
        hy = extents[1] / 2.0 - inset
        if hx < 0 or hy < 0:
            tp.metadata["skipped"] = "feature too small for semi-finishing tool"
            self._end_op(tp, op)
            return tp
        corners = [self._v3(cx - hx, cy - hy, z_target),
                   self._v3(cx + hx, cy - hy, z_target),
                   self._v3(cx + hx, cy + hy, z_target),
                   self._v3(cx - hx, cy + hy, z_target)]
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
        tp.metadata["strategy"] = "semi_offset_pass"
        return tp

    # -------------------------------------------------------- finishing --
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
        return self._finish_wall_floor(op)

    def _finish_wall_floor(self, op: PlannedOperation) -> Toolpath:
        """Contour finishing: full-depth wall pass + floor pass, engagement-aware."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0
        bmin, bmax = f.bounds_min, f.bounds_max
        extents = bmax[:2] - bmin[:2]

        wall_inset = radius
        cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
        hx = extents[0] / 2.0 - wall_inset
        hy = extents[1] / 2.0 - wall_inset
        if hx < -1e-9 or hy < -1e-9:
            tp.metadata["skipped"] = "tool does not fit for finishing"
            self._end_op(tp, op)
            return tp

        # wall finishing: full depth contour
        z = f.top_z
        floor_z = f.floor_z
        corners = [self._v3(cx - hx, cy - hy, z), self._v3(cx + hx, cy - hy, z),
                   self._v3(cx + hx, cy + hy, z), self._v3(cx - hx, cy + hy, z)]
        pt = corners[0]
        self._link_to(tp, op, pt)
        # helical entry down the wall
        self._helical_ramp(tp, op, self._v3(pt[0], pt[1], floor_z),
                           helix_radius=min(hx, hy) * 0.4,
                           pitch=min(params.depth_of_cut_mm, tool.diameter * 0.3))
        # contour at floor level, then step up (axial spacing uses depth_of_cut,
        # not the radial stepover)
        doc = max(params.depth_of_cut_mm, 1e-3)
        bands = max(1, int(np.ceil((f.top_z - floor_z) / doc)))
        for b in range(bands):
            z_band = min(f.top_z, floor_z + b * doc)
            band_corners = [self._v3(c[0], c[1], z_band) for c in corners]
            if b == 0:
                for c in band_corners[1:] + [band_corners[0]]:
                    self._add_cut(tp, op, c)
            else:
                self._link_to(tp, op, band_corners[0])
                for c in band_corners[1:] + [band_corners[0]]:
                    self._add_cut(tp, op, c)

        # floor finishing: serpentine at floor level
        stepover = max(params.stepover_mm, 1e-3)
        x0, x1 = bmin[0] + radius, bmax[0] - radius
        y0, y1 = bmin[1] + radius, bmax[1] - radius
        if x1 >= x0 and y1 >= y0:
            n_rows = max(1, int(np.ceil((y1 - y0) / stepover)) + 1)
            for r in range(n_rows):
                y = y0 + min(r * stepover, y1 - y0)
                xa, xb = (x0, x1) if r % 2 == 0 else (x1, x0)
                row_start = self._v3(xa, y, floor_z)
                if r == 0:
                    prev = self._seg_end(tp)
                    tp.add(MotionSegment(motion_type=MotionType.CUT, start=prev,
                                         end=row_start, feed=params.feed_rate,
                                         spindle=params.spindle_rpm, tool_id=tool.id,
                                         operation_id=op.id, feature_id=f.id))
                else:
                    self._link_to(tp, op, row_start)
                self._add_cut(tp, op, self._v3(xb, y, floor_z))

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "wall_floor_contour"
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

        scallop = 0.05
        if f.finish is not None and getattr(f.finish, "allowed_scallop_mm", None):
            scallop = f.finish.allowed_scallop_mm
        r = tool.tip_radius if tool.type is ToolType.BALL_ENDMILL else tool.diameter / 2.0
        if r <= 0:
            r = tool.diameter / 2.0
        # scallop -> stepover for ball tool on flat: s = 2*sqrt(2*R*h - h^2)
        stepover = 2.0 * np.sqrt(max(2.0 * r * scallop - scallop * scallop, 1e-6))
        stepover = float(np.clip(stepover, 0.05, tool.diameter * 0.9))

        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        margin = tool.diameter / 2.0
        x0, x1 = bmin[0] - margin, bmax[0] + margin
        y0, y1 = bmin[1] - margin, bmax[1] + margin
        n_rows = max(1, int(np.ceil((y1 - y0) / stepover)) + 1)

        first = True
        for row in range(n_rows):
            y = y0 + min(row * stepover, y1 - y0)
            n_cols = max(2, int(np.ceil((x1 - x0) / max(tool.diameter * 0.2, 0.2))) + 1)
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
                start = pts[0] + np.array([0, 0, self.s.clearance_z])
                self._rapid_to(tp, op, start)
                first = False
            else:
                prev = self._seg_end(tp)
                link_z = max(prev[2], pts[0][2]) + self.s.context.machine.safe_retract_height
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

    # ---- waterline finishing ----
    def _finish_waterline(self, op: PlannedOperation) -> Toolpath:
        """Constant-Z contour passes at discrete Z levels along steep walls.
        Z levels spaced by stepover to achieve target scallop height."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0

        scallop = 0.05
        if f.finish is not None and getattr(f.finish, "allowed_scallop_mm", None):
            scallop = f.finish.allowed_scallop_mm
        r = tool.tip_radius if tool.type is ToolType.BALL_ENDMILL else radius

        # Z level spacing for target scallop
        if r > 0:
            z_step = 2.0 * math.sqrt(max(2.0 * r * scallop - scallop * scallop, 1e-6))
        else:
            z_step = scallop
        z_step = float(np.clip(z_step, 0.1, params.stepover_mm))

        floor_z = f.floor_z + self.s.finish_allowance_floor
        top_z = f.top_z
        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
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
            n_pts = max(4, int(np.ceil(max(extents) / max(tool.diameter * 0.2, 0.2))))
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

    # ---- scallop-uniform finishing ----
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

        scallop = 0.05
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
        base_stepover = float(np.clip(base_stepover, 0.05, tool.diameter * 0.9))

        n_rows = max(1, int(np.ceil((y1 - y0) / base_stepover)) + 1)
        first = True
        for row in range(n_rows):
            y = y0 + min(row * base_stepover, y1 - y0)
            n_cols = max(2, int(np.ceil((x1 - x0) / max(tool.diameter * 0.15, 0.15))) + 1)
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
                start = pts[0] + np.array([0, 0, self.s.clearance_z])
                self._rapid_to(tp, op, start)
                first = False
            else:
                prev = self._seg_end(tp)
                link_z = max(prev[2], pts[0][2]) + self.s.context.machine.safe_retract_height
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

    # ---- pencil trace finishing ----
    def _finish_pencil(self, op: PlannedOperation) -> Toolpath:
        """Single-pass toolpath that traces concave corners/intersections of surfaces.
        Used for corner cleanup after main finishing."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0
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

    # ---- radial / spiral finishing ----
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

    # --------------------------------------------------------- drilling --
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
        tp.add(MotionSegment(motion_type=MotionType.SPINDLE_START, start=self._v3(0),
                             end=self._v3(0), spindle=params.spindle_rpm,
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        depth = top_z - floor_z
        peck = params.depth_of_cut_mm
        z = top_z
        prev = self._v3(center[0], center[1], self.s.clearance_z)
        first_plunge = True
        while z > floor_z + 1e-9:
            z_next = max(floor_z, z - peck)
            meta: dict = {}
            if first_plunge:
                # Post-processor converts this into a real G83 peck cycle;
                # explicit peck segments remain for simulation/validation.
                meta = {"canned_cycle": {
                    "type": "G83", "x": float(center[0]), "y": float(center[1]),
                    "z_r": top_z + 1.0, "z_depth": float(floor_z),
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
        """Open a hole larger than any drill by helical milling (arc segments)."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z, floor_z = f.top_z, f.floor_z
        r_hole = f.diameter / 2.0
        r_tool = tool.diameter / 2.0
        r_path = r_hole - r_tool
        if r_path <= 0:
            raise CamError(UNSUPPORTED_FEATURE,
                           f"hole {f.id}: tool too large for helical interpolation",
                           stage="toolpath", operation_id=op.id, feature_id=f.id)
        self._rapid_to(tp, op, self._v3(center[0] + r_path, center[1],
                                        self.s.clearance_z))
        prev = self._v3(center[0] + r_path, center[1], self.s.clearance_z)
        tp.add(MotionSegment(motion_type=MotionType.RAPID, start=prev,
                             end=self._v3(center[0] + r_path, center[1], top_z + 1.0),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        z = top_z + 1.0
        pitch = min(params.depth_of_cut_mm, tool.diameter * 0.5)
        while z > floor_z + 1e-9:
            z_next = max(floor_z, z - pitch)
            n_seg = 12
            for k in range(1, n_seg + 1):
                a0 = 2 * np.pi * (k - 1) / n_seg
                a1 = 2 * np.pi * k / n_seg
                p0 = self._v3(center[0] + r_path * np.cos(a0),
                              center[1] + r_path * np.sin(a0),
                              z + (z_next - z) * (k - 1) / n_seg)
                p1 = self._v3(center[0] + r_path * np.cos(a1),
                              center[1] + r_path * np.sin(a1),
                              z + (z_next - z) * k / n_seg)
                tp.add(MotionSegment(motion_type=MotionType.HELIX, start=p0, end=p1,
                                     arc_center=self._v3(center[0], center[1], p1[2]),
                                     arc_ccw=True, feed=params.feed_rate,
                                     spindle=params.spindle_rpm, tool_id=tool.id,
                                     operation_id=op.id, feature_id=f.id))
            z = z_next
        # final circular finish pass at floor
        for k in range(1, 13):
            a0 = 2 * np.pi * (k - 1) / 12
            a1 = 2 * np.pi * k / 12
            p0 = self._v3(center[0] + r_path * np.cos(a0),
                          center[1] + r_path * np.sin(a0), floor_z)
            p1 = self._v3(center[0] + r_path * np.cos(a1),
                          center[1] + r_path * np.sin(a1), floor_z)
            tp.add(MotionSegment(motion_type=MotionType.ARC_CCW, start=p0, end=p1,
                                 arc_center=self._v3(center[0], center[1], floor_z),
                                 arc_ccw=True, feed=params.feed_rate,
                                 spindle=params.spindle_rpm, tool_id=tool.id,
                                 operation_id=op.id, feature_id=f.id))
        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.CUT, start=last,
                             end=self._v3(center[0], center[1], floor_z),
                             feed=params.feed_rate, tool_id=tool.id,
                             operation_id=op.id, feature_id=f.id))
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(center[0], center[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "helical_interpolation"
        return tp

    # ----------------------------------------------------------- boring --
    def _boring(self, op: PlannedOperation) -> Toolpath:
        """Boring bar operation for precise hole sizing: single-point tool
        with circular interpolation at the target diameter."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        top_z, floor_z = f.top_z, f.floor_z
        r_bore = f.diameter / 2.0 if f.diameter else 0.0
        if r_bore <= 0:
            raise CamError(UNSUPPORTED_FEATURE,
                           f"bore {f.id}: no diameter specified for boring",
                           stage="toolpath", operation_id=op.id, feature_id=f.id)

        self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
        # rapid to bore start position
        self._rapid_to(tp, op, self._v3(center[0] + r_bore, center[1], top_z + 1.0))
        prev = self._v3(center[0] + r_bore, center[1], top_z + 1.0)
        tp.add(MotionSegment(motion_type=MotionType.RAPID, start=prev,
                             end=self._v3(center[0] + r_bore, center[1], top_z + 0.5),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))

        # peck boring cycle
        peck = params.depth_of_cut_mm
        z = top_z + 0.5
        while z > floor_z + 1e-9:
            z_next = max(floor_z, z - peck)
            # circular interpolation at bore radius
            for k in range(1, 13):
                a0 = 2 * np.pi * (k - 1) / 12
                a1 = 2 * np.pi * k / 12
                z_interp = z + (z_next - z) * k / 12
                p0 = self._v3(center[0] + r_bore * math.cos(a0),
                              center[1] + r_bore * math.sin(a0), z_interp)
                p1 = self._v3(center[0] + r_bore * math.cos(a1),
                              center[1] + r_bore * math.sin(a1), z_interp)
                tp.add(MotionSegment(motion_type=MotionType.ARC_CCW, start=p0, end=p1,
                                     arc_center=self._v3(center[0], center[1], z_interp),
                                     arc_ccw=True, feed=params.feed_rate,
                                     spindle=params.spindle_rpm, tool_id=tool.id,
                                     operation_id=op.id, feature_id=f.id))
            z = z_next
            if z > floor_z + 1e-9:
                # retract for chip clear
                prev = self._seg_end(tp)
                tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=prev,
                                     end=self._v3(center[0] + r_bore, center[1],
                                                  top_z + 1.0),
                                     tool_id=tool.id, operation_id=op.id,
                                     feature_id=f.id))
                tp.add(MotionSegment(motion_type=MotionType.RAPID, start=prev,
                                     end=self._v3(center[0] + r_bore, center[1],
                                                  z + 0.5),
                                     tool_id=tool.id, operation_id=op.id,
                                     feature_id=f.id))

        # final spring pass at full depth for size accuracy
        for k in range(1, 13):
            a0 = 2 * np.pi * (k - 1) / 12
            a1 = 2 * np.pi * k / 12
            p0 = self._v3(center[0] + r_bore * math.cos(a0),
                          center[1] + r_bore * math.sin(a0), floor_z)
            p1 = self._v3(center[0] + r_bore * math.cos(a1),
                          center[1] + r_bore * math.sin(a1), floor_z)
            tp.add(MotionSegment(motion_type=MotionType.ARC_CCW, start=p0, end=p1,
                                 arc_center=self._v3(center[0], center[1], floor_z),
                                 arc_ccw=True, feed=params.feed_rate * 0.5,
                                 spindle=params.spindle_rpm, tool_id=tool.id,
                                 operation_id=op.id, feature_id=f.id))

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(center[0], center[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "boring"
        tp.metadata["bore_diameter"] = f.diameter
        return tp

    # ----------------------------------------------------- spot drilling --
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

    # ---------------------------------------------------------- reaming --
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

    # ---------------------------------------------------------- tapping --
    def _tapping(self, op: PlannedOperation) -> Toolpath:
        """Tapping requires spindle-feed synchronization (rigid tapping G84 or
        spindle-sync feed). This strategy records the tapping intent on the
        toolpath; the post-processor emits the controller-appropriate cycle.
        Rigid tapping needs a feed-per-revolution feed rate."""
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
            feed=tap_feed, spindle=params.spindle_rpm,
            tool_id=tool.id, operation_id=op.id, feature_id=f.id,
            metadata={"tap_feed_mm_per_rev": tap_feed,
                      "canned_cycle": {
                          "type": "G84", "x": float(center[0]), "y": float(center[1]),
                          "z_r": top_z + 1.0, "z_depth": float(floor_z),
                          "feed": tap_feed,  # mm/rev: correct F basis for G84
                          "initial_z": float(self.s.clearance_z),
                      }}))

        # spindle reversal + feed out is emitted by the post from the cycle intent
        tp.add(MotionSegment(
            motion_type=MotionType.RETRACT,
            start=self._v3(center[0], center[1], floor_z),
            end=self._v3(center[0], center[1], top_z + 1.0),
            feed=tap_feed, spindle=params.spindle_rpm,
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
        tp.metadata["canned_cycle_intent"] = {
            "position": center.tolist(), "depth": top_z - floor_z,
            "retract_height": 1.0, "cycle": "G84",
            "feed_mm_per_rev": tap_feed}
        return tp

    # ------------------------------------------------------- chamfering --
    def _chamfering(self, op: PlannedOperation) -> Toolpath:
        """Machine a chamfer feature: contour the boundary edge at the Z and XY
        offset that places the tool tip cone on the chamfer face. Depth and
        offset derive from the chamfer's actual slant geometry."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params

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
