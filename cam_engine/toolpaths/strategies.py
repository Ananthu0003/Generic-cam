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

from ..context import (
    DepthReference,
    DrillDepthType,
    DrillingDepthSemantics,
    PlanningContext,
    Setup,
    Tool,
    ToolType,
    SetupClearanceInfo,
)
from ..coords import CoordSpace
from ..errors import CamError, INVALID_TOOLPATH, UNSUPPORTED_FEATURE
from ..features import FeatureType, MachiningFeature
from ..machinability import CuttingParameters
from ..operations import OpPurpose, PlannedOperation
from ..algorithms import generate_zigzag_facing, generate_spiral_facing, generate_circular_spiral_facing
from .semantic import MotionSegment, MotionType, Toolpath



def _has_self_intersection(polygon: list[np.ndarray]) -> bool:
    """Check if any non-adjacent edges of a 2D polygon intersect (ray-casting)."""
    n = len(polygon)
    if n < 4:
        return False
    for i in range(n):
        a0 = polygon[i]
        a1 = polygon[(i + 1) % n]
        for j in range(i + 2, n):
            if j == (i - 1) % n or (i == 0 and j == n - 1):
                continue
            b0 = polygon[j]
            b1 = polygon[(j + 1) % n]
            da = a1 - a0
            db = b1 - b0
            denom = da[0] * db[1] - da[1] * db[0]
            if abs(denom) < 1e-9:
                continue
            dp = a0 - b0
            t = (db[0] * dp[1] - db[1] * dp[0]) / denom
            u = (da[0] * dp[1] - da[1] * dp[0]) / denom
            if 0.0 < t < 1.0 and 0.0 < u < 1.0:
                return True
    return False


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
    clearance_info: Optional[SetupClearanceInfo] = None
    approach_margin: float = 1.0     # Safe approach standoff distance above material top
    retract_margin: float = 2.0      # Safe local retract standoff distance above cut
    target_cad_voxels: Optional[object] = None       # VoxelStock: nominal finished CAD solid
    target_removal_volume: Optional[object] = None   # VoxelStock: raw stock MINUS target CAD solid

    @property
    def approach_z(self) -> float:
        """Setup-level approach plane."""
        if self.clearance_info is not None:
            return self.clearance_info.approach_z
        stock_top = float(self.setup.stock.bounds_max[2]) if (hasattr(self.setup, "stock") and self.setup.stock is not None and hasattr(self.setup.stock, "bounds_max") and self.setup.stock.bounds_max is not None) else 0.0
        return stock_top + self.approach_margin

    @property
    def retract_z(self) -> float:
        """Setup-level retract plane."""
        if self.clearance_info is not None:
            return self.clearance_info.retract_z
        stock_top = float(self.setup.stock.bounds_max[2]) if (hasattr(self.setup, "stock") and self.setup.stock is not None and hasattr(self.setup.stock, "bounds_max") and self.setup.stock.bounds_max is not None) else 0.0
        return stock_top + self.retract_margin


class StrategyEngine:
    """Generates semantic toolpaths for planned operations."""

    def __init__(self, strat: StrategyContext):
        self.s = strat

    # ------------------------------------------------------------------
    def generate(self, op: PlannedOperation) -> Toolpath:
        """Generate one toolpath and stamp it with its validity report.

        The stamp is what blocks G-code emission later: if any critical
        operation check fails, ``assert_valid_for_post`` refuses to write code
        for this operation.
        """
        tp = self._dispatch(op)
        return self._stamp_validity(op, tp)

    def _stamp_validity(self, op: PlannedOperation, tp: Toolpath) -> Toolpath:
        from ..validity import OperationValidityChecker
        checker = OperationValidityChecker(
            context=getattr(self.s, "context", None),
            setup=getattr(self.s, "setup", None),
        )
        tp.metadata["validity"] = checker.check(op, tp).to_dict()
        return tp

    def _dispatch(self, op: PlannedOperation) -> Toolpath:
        # Automatic Tool-Radius & Envelope Feasibility Checking
        from .validation import PreToolpathGeometryValidator
        validator = PreToolpathGeometryValidator(
            context=getattr(self.s, "context", None),
            setup=getattr(self.s, "setup", None),
            shape=getattr(self.s, "shape", None),
        )
        feasibility_errors = validator.check_tool_feasibility(
            tool=op.tool,
            feature=op.feature,
            op_purpose=op.purpose,
            op_id=op.id,
        )
        if feasibility_errors:
            raise feasibility_errors[0]

        if op.purpose is OpPurpose.FACING:
            return self._facing(op)

        if op.purpose is OpPurpose.ROUGHING:
            if op.notes.get("method") == "adaptive":
                return self._adaptive_roughing(op)
            if op.notes.get("method") == "helical_interpolation" or op.feature.type in (FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE):
                return self._helical_interpolation(op)
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
        if op.purpose is OpPurpose.FILLETING:
            return self._filleting(op)
        if op.purpose is OpPurpose.COUNTERSINKING:
            return self._countersinking(op)
        if op.purpose is OpPurpose.THREAD_MILLING:
            return self._thread_milling(op)
        if op.purpose is OpPurpose.GROOVING:
            return self._grooving(op)
        raise CamError(UNSUPPORTED_FEATURE,
                       f"no strategy for purpose {op.purpose}", stage="toolpath",
                       operation_id=op.id)

    # ---------------------------------------------------------- helpers --
    def _toolpath_skeleton(self, op: PlannedOperation) -> Toolpath:
        tool = op.tool
        from .semantic import ToolpathRepresentation, CompensationMode, CutterCompensationPolicy
        
        comp_mode_str = op.notes.get("compensation_mode", getattr(tool, "compensation_mode", None))
        if comp_mode_str is None:
            comp_mode_str = "wear" if op.notes.get("cutter_comp") else "none"

        try:
            comp_mode = CompensationMode(comp_mode_str)
        except ValueError:
            comp_mode = CompensationMode.NONE

        explicit_rep = op.notes.get("toolpath_representation")
        if explicit_rep:
            try:
                geom_mode = ToolpathRepresentation(explicit_rep)
            except ValueError:
                geom_mode = ToolpathRepresentation.CENTER_PATH
        elif comp_mode == CompensationMode.GEOMETRY:
            geom_mode = ToolpathRepresentation.PART_BOUNDARY_PATH
        else:
            geom_mode = ToolpathRepresentation.CENTER_PATH

        target_surf = op.notes.get("target_surface", getattr(op.feature.type, "value", str(op.feature.type)))
        tp = Toolpath(
            operation_id=op.id,
            feature_id=op.feature.id,
            tool_id=op.tool.id,
            purpose=op.purpose.value,
            space=CoordSpace.SETUP,
            geometry_mode=geom_mode,
            source_face_ids=list(op.feature.face_indices),
            source_edge_ids=list(op.feature.source_edges),
            target_surface=target_surf,
            design_geometry=op.feature.design_geometry if hasattr(op.feature, "design_geometry") else None,
            manufacturing_geometry=op.get_manufacturing_geometry() if hasattr(op, "get_manufacturing_geometry") else None,
        )
        tp.metadata["compensation_mode"] = comp_mode.value
        tp.metadata["toolpath_representation"] = geom_mode.value
        tp.metadata["tool_info"] = {
            "id": tool.id,
            "tool_number": tool.tool_number,
            "type": tool.type.value if hasattr(tool.type, "value") else str(tool.type),
            "diameter": tool.diameter,
            "flute_length": tool.flute_length,
            "overall_length": tool.overall_length,
            "corner_radius": tool.corner_radius,
            "flutes": tool.flutes,
            "material": tool.material,
            "length_offset_h": tool.length_offset_h,
            "diameter_offset_d": tool.diameter_offset_d,
        }

        if op.params is not None:
            tp.metadata["cutting_parameters"] = {
                "spindle_rpm": op.params.spindle_rpm,
                "feed_rate": op.params.feed_rate,
                "plunge_feed": op.params.plunge_feed,
                "ramp_feed": op.params.ramp_feed,
                "feed_per_tooth_mm": op.params.feed_per_tooth_mm,
                "chip_load_mm": getattr(op.params, "feed_per_tooth_mm", 0.0),
                "depth_of_cut_mm": op.params.depth_of_cut_mm,
                "stepover_mm": op.params.stepover_mm,
                "cutting_speed_m_min": getattr(op.params, "cutting_speed_m_min", 0.0),
                "target_cutting_speed_m_min": getattr(op.params, "target_cutting_speed_m_min", getattr(op.params, "cutting_speed_m_min", 0.0)),
                "effective_cutting_speed_m_min": getattr(op.params, "effective_cutting_speed_m_min", getattr(op.params, "cutting_speed_m_min", 0.0)),
                "target_rpm": getattr(op.params, "target_rpm", op.params.spindle_rpm),
                "calculated_rpm": getattr(op.params, "calculated_rpm", getattr(op.params, "target_rpm", op.params.spindle_rpm)),
                "rpm_clamped": getattr(op.params, "rpm_clamped", False),
                "target_feed_rate": getattr(op.params, "target_feed_rate", op.params.feed_rate),
                "feed_clamped": getattr(op.params, "feed_clamped", False),
                "material_name": getattr(op.params, "material_name", getattr(self.s.context.material, "name", "") if hasattr(self.s.context, "material") and self.s.context.material else ""),
                "material_hardness_hb": getattr(op.params, "material_hardness_hb", getattr(self.s.context.material, "hardness_hb", 0.0) if hasattr(self.s.context, "material") and self.s.context.material else 0.0),
                "tool_id": tool.id,
                "tool_type": tool.type.value if hasattr(tool.type, "value") else str(tool.type),
                "tool_material": tool.material,
                "tool_diameter_mm": tool.diameter,
                "flute_count": tool.flutes,
                "machine_max_rpm": getattr(op.params, "machine_max_rpm", getattr(self.s.context.machine, "spindle_max_rpm", 0.0) if hasattr(self.s.context, "machine") and self.s.context.machine else 0.0),
                "machine_min_rpm": getattr(op.params, "machine_min_rpm", getattr(self.s.context.machine, "spindle_min_rpm", 0.0) if hasattr(self.s.context, "machine") and self.s.context.machine else 0.0),
                "machine_max_feed": getattr(op.params, "machine_max_feed", 0.0),
                "power_kw": op.params.power_kw,
                "torque_nm": op.params.torque_nm,
                "coolant": op.params.coolant,
            }
        tp.metadata["clearance_z"] = self.s.clearance_z
        tp.metadata["retract_z"] = self.s.retract_z
        tp.metadata["approach_z"] = self.s.approach_z
        tp.metadata["approach_margin"] = getattr(self.s, "approach_margin", 1.0)
        tp.metadata["operation_planes"] = {
            "clearance_z": self.s.clearance_z,
            "retract_z": self.s.retract_z,
            "approach_z": getattr(self.s, "approach_z", self.s.clearance_z),
            "start_z": getattr(op, "start_z", None),
            "final_z": getattr(op, "final_z", None),
        }
        if getattr(self.s, "clearance_info", None) is not None:
            tp.metadata["clearance_info"] = self.s.clearance_info.to_dict()
        return tp

    def _start_op(self, tp: Toolpath, op: PlannedOperation, start_pt: Optional[np.ndarray] = None) -> None:
        """Spindle/coolant state + approach from clearance."""
        tool = op.tool
        pos = start_pt if start_pt is not None else self._v3(0, 0, self.s.clearance_z)
        tool_info = tp.metadata.get("tool_info")
        tp.add(MotionSegment(
            motion_type=MotionType.TOOL_CHANGE, start=pos, end=pos,
            tool_id=tool.id, operation_id=op.id, feature_id=op.feature.id,
            metadata={"tool_number": tool.tool_number,
                      "length_offset_h": tool.length_offset_h,
                      "diameter_offset_d": tool.diameter_offset_d,
                      "overall_length": tool.overall_length,
                      "tool_info": tool_info}))
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

    def _material_top_at(self, p_xy: np.ndarray, tool_r: float = 0.0) -> float:
        """Calculate the highest obstacle/material Z level at (x, y) accounting for tool radius.
        
        Returns -inf if the point (with tool radius) is completely in free air.
        """
        top_z = float('-inf')
        p_xy = np.asarray(p_xy[:2], dtype=float)

        # 1. Stock material envelope
        stock = getattr(self.s.setup, "stock", None) if getattr(self.s, "setup", None) is not None else None
        if stock is not None:
            if hasattr(stock, "kind") and getattr(stock.kind, "value", str(stock.kind)) == "cylinder" and hasattr(stock, "radius") and stock.radius > 0:
                s_center = stock.center_xy if (hasattr(stock, "center_xy") and stock.center_xy is not None) else np.zeros(2)
                d_xy = float(np.linalg.norm(p_xy - s_center[:2]))
                s_margin = tool_r + 2.0
                if d_xy <= stock.radius + s_margin:
                    top_z = max(top_z, float(stock.bounds_max[2]))
            else:
                s_margin = tool_r + 2.0
                s_min = stock.bounds_min[:2] - s_margin
                s_max = stock.bounds_max[:2] + s_margin
                if (s_min[0] <= p_xy[0] <= s_max[0]) and (s_min[1] <= p_xy[1] <= s_max[1]):
                    top_z = max(top_z, float(stock.bounds_max[2]))

        # 2. Model mesh geometry
        mesh = getattr(self.s, "mesh", None)
        if mesh is not None and hasattr(mesh, "heights_above"):
            pts_to_check = [p_xy]
            if tool_r > 0.1:
                for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
                    pts_to_check.append(p_xy + np.array([tool_r * math.cos(a), tool_r * math.sin(a)]))
            for pt in pts_to_check:
                hs = mesh.heights_above(pt[0], pt[1])
                if hs:
                    top_z = max(top_z, max(hs))

        # 3. Clamping fixtures
        fixtures = getattr(self.s.setup, "fixtures", None) if getattr(self.s, "setup", None) is not None else None
        if fixtures:
            for fix in fixtures:
                f_min = getattr(fix, "bounds_min", None)
                f_max = getattr(fix, "bounds_max", None)
                if f_min is not None and f_max is not None:
                    if (f_min[0] - tool_r <= p_xy[0] <= f_max[0] + tool_r) and \
                       (f_min[1] - tool_r <= p_xy[1] <= f_max[1] + tool_r):
                        top_z = max(top_z, float(f_max[2]))

        return top_z

    # ---- linking primitives ----
    def _link_to(self, tp: Toolpath, op: PlannedOperation, p: np.ndarray,
                 approach_offset: Optional[float] = None) -> None:
        """Retract to clearance, rapid to XY, rapid down to safe approach Z, plunge to target Z."""
        last = self._seg_end(tp)
        p_target = np.asarray(p, dtype=float)
        clearance_z = self.s.clearance_z
        tool_r = op.tool.diameter / 2.0
        margin = approach_offset if approach_offset is not None else getattr(self.s, "approach_margin", 1.0)

        # Calculate environment material/obstacle top at target XY
        mat_top = self._material_top_at(p_target, tool_r)
        if mat_top > float('-inf'):
            # Point is above/near solid material or stock: rapid MUST stop above material top
            approach_z = min(clearance_z, max(mat_top + margin, p_target[2] + margin))
        else:
            # Free space completely outside stock/fixtures/model: safe to rapid down to target + margin
            approach_z = min(clearance_z, p_target[2] + margin)

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

        # 3. Rapid descend to safe approach height (strictly above material/obstacle)
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
        tool_r = op.tool.diameter / 2.0
        mat_top = self._material_top_at(p, tool_r)
        last = self._seg_end(tp)
        retract_z = max(last[2], p[2]) + clearance_above
        # If stay-down retract height is below material top, it's unsafe to rapid across -> use full link_to
        if mat_top > float('-inf') and retract_z < mat_top + 0.1:
            self._link_to(tp, op, p)
            return

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
        dx = float(target[0] - prev[0])
        dy = float(target[1] - prev[1])
        chord = math.hypot(dx, dy)
        if chord < 1e-9:
            return
        # Semicircle / tangent arc where center lies on perpendicular bisector
        perp_x, perp_y = -dy / chord, dx / chord
        arc_radius = max(radius, chord / 2.0)
        half_chord = chord / 2.0
        h = math.sqrt(max(0.0, arc_radius * arc_radius - half_chord * half_chord))
        cx = (prev[0] + target[0]) / 2.0 + perp_x * h
        cy = (prev[1] + target[1]) / 2.0 + perp_y * h
        arc_center = self._v3(cx, cy, target[2])
        self._arc_to(tp, op, arc_center, target, ccw=True, feed=params.ramp_feed)

    def _arc_lead_out(self, tp: Toolpath, op: PlannedOperation,
                      start_point: np.ndarray, radius: float) -> None:
        """Tangential arc departure from the cut end point."""
        params = op.params
        prev = self._seg_end(tp)
        dx = float(prev[0] - start_point[0]) if start_point is not None else 1.0
        dy = float(prev[1] - start_point[1]) if start_point is not None else 0.0
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            dx, dy, dist = 1.0, 0.0, 1.0
        tx, ty = dx / dist, dy / dist  # tangent direction
        nx, ny = -ty, tx               # normal direction
        cx = prev[0] + nx * radius
        cy = prev[1] + ny * radius
        arc_end = self._v3(cx + tx * radius, cy + ty * radius, prev[2])
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

    def _offset_polygon(self, polygon: list[np.ndarray], offset: float) -> list[np.ndarray]:
        """Offset a 2D polygon inward (negative offset) or outward (positive).
        Uses shapely when available (production-quality, handles concave/complex polygons);
        falls back to edge-parallel intersection math for environments without shapely."""
        try:
            from shapely.geometry import Polygon as _ShapelyPoly
            from shapely.validation import make_valid
            if len(polygon) < 3:
                return polygon[:]
            clean_pts: list[tuple[float, float]] = []
            for p in polygon:
                pt = (float(p[0]), float(p[1]))
                if not clean_pts or (abs(pt[0] - clean_pts[-1][0]) > 1e-5 or abs(pt[1] - clean_pts[-1][1]) > 1e-5):
                    clean_pts.append(pt)
            if len(clean_pts) < 3:
                return polygon[:]
            poly = _ShapelyPoly(clean_pts)
            if not poly.is_valid:
                valid_geom = make_valid(poly)
                if hasattr(valid_geom, 'geoms'):
                    poly = max(valid_geom.geoms, key=lambda g: getattr(g, 'area', 0.0))
                else:
                    poly = valid_geom
            result_geom = poly.buffer(offset, join_style=1, cap_style=3)
            if result_geom.is_empty:
                return []
            # MultiPolygon: take the largest piece
            if hasattr(result_geom, 'geoms'):
                result_geom = max(result_geom.geoms, key=lambda g: getattr(g, 'area', 0.0))
            if not hasattr(result_geom, 'exterior') or result_geom.exterior is None:
                return []
            coords = list(result_geom.exterior.coords)[:-1]  # exclude closing duplicate
            return [np.array([x, y], dtype=float) for x, y in coords]
        except (ImportError, Exception):
            return self._offset_polygon_fallback(polygon, offset)

    @staticmethod
    def _offset_polygon_fallback(polygon: list[np.ndarray], offset: float) -> list[np.ndarray]:
        """Fallback edge-parallel polygon offset (used when shapely is unavailable).
        Handles convex and simple concave polygons via consecutive edge intersections."""
        n = len(polygon)
        if n < 3:
            return polygon[:]
        area = 0.5 * sum(polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1] for i in range(n))
        sign = 1.0 if area >= 0 else -1.0
        offset_edges = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            edge = p1 - p0
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            normal = sign * np.array([edge[1], -edge[0]], dtype=float) / edge_len
            op0 = p0 + normal * offset
            op1 = p1 + normal * offset
            offset_edges.append((op0, op1))
        if len(offset_edges) < 3:
            return polygon[:]

        result = []
        m = len(offset_edges)
        for i in range(m):
            _, e0_end = offset_edges[i]
            e1_start, _ = offset_edges[(i + 1) % m]
            d0 = e0_end - offset_edges[i][0]
            d1 = offset_edges[(i + 1) % m][1] - e1_start
            cross = d0[0] * d1[1] - d0[1] * d1[0]
            if abs(cross) < 1e-12:
                result.append(e0_end.copy())
            else:
                w = offset_edges[i][0] - e1_start
                t = (d1[0] * w[1] - d1[1] * w[0]) / cross
                result.append(offset_edges[i][0] + t * d0)
        if len(result) < 3:
            return result
        cleaned = [result[0]]
        for pt in result[1:]:
            if np.linalg.norm(pt - cleaned[-1]) > 1e-9:
                cleaned.append(pt)
        if len(cleaned) > 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) < 1e-9:
            cleaned.pop()
        if len(cleaned) < 3:
            return result
        if _has_self_intersection(cleaned):
            return []
        return cleaned

    @staticmethod
    def _convex_hull_2d(points: list[np.ndarray]) -> list[np.ndarray]:
        """Compute 2D convex hull using Monotone Chain algorithm."""
        pts = sorted(set((float(round(p[0], 4)), float(round(p[1], 4))) for p in points))
        if len(pts) <= 2:
            return [np.array(p, dtype=float) for p in pts]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return [np.array(p, dtype=float) for p in hull]

    @staticmethod
    def _is_feature_circular(f: MachiningFeature) -> bool:
        """Check if a feature is geometrically a true cylinder/circular boss/hole."""
        if bool(f.notes.get("is_circular")):
            return True
        if f.diameter is not None and f.diameter > 0 and f.bounds_min is not None and f.bounds_max is not None:
            dx = float(f.bounds_max[0] - f.bounds_min[0])
            dy = float(f.bounds_max[1] - f.bounds_min[1])
            if abs(dx - dy) < 1e-3 and abs(dx - f.diameter) < 1e-3:
                return True
        return False

    def _feature_boundary_polygon(self, f: MachiningFeature, tool_radius: float = 0.0) -> list[np.ndarray]:
        """Extract a 2D boundary polygon from the feature's wall faces.
        For open features (Step, Shoulder, Open Pocket/Slot), extends open boundaries into open air
        by (tool_radius + 1.0mm) to prevent standing corner slivers."""
        is_open = f.type in (FeatureType.STEP, FeatureType.OPEN_POCKET, FeatureType.OPEN_SLOT, FeatureType.SHOULDER) or f.notes.get("is_open", False)
        
        if is_open and tool_radius > 0:
            extension = tool_radius * 1.2 + 1.0
            bmin, bmax = f.bounds_min[:2].copy(), f.bounds_max[:2].copy()
            model_bmin = self.s.setup.stock.bounds_min[:2]
            model_bmax = self.s.setup.stock.bounds_max[:2]
            if hasattr(self.s, "mesh") and self.s.mesh is not None and hasattr(self.s.mesh, "bounds_min"):
                model_bmin = self.s.mesh.bounds_min[:2]
                model_bmax = self.s.mesh.bounds_max[:2]
            
            # Check which boundaries are open (coinciding with or touching outer model bounds)
            if abs(bmin[0] - model_bmin[0]) < 1.0:
                bmin[0] -= extension
            if abs(bmax[0] - model_bmax[0]) < 1.0:
                bmax[0] += extension
            if abs(bmin[1] - model_bmin[1]) < 1.0:
                bmin[1] -= extension
            if abs(bmax[1] - model_bmax[1]) < 1.0:
                bmax[1] += extension
            
            return [
                np.array([bmin[0], bmin[1]]),
                np.array([bmax[0], bmin[1]]),
                np.array([bmax[0], bmax[1]]),
                np.array([bmin[0], bmax[1]]),
            ]

        # Use actual boundary geometry if the recognizer extracted it
        if f.boundary_loop and len(f.boundary_loop) >= 3:
            return [np.array([p[0], p[1]], dtype=float) for p in f.boundary_loop]

        # If feature has circular definition (diameter / axis / center), generate high-resolution circular boundary
        if self._is_feature_circular(f):
            r = f.diameter / 2.0 if (f.diameter is not None and f.diameter > 0) else (f.radius or (f.bounds_max[0] - f.bounds_min[0]) / 2.0)
            cx = (f.bounds_min[0] + f.bounds_max[0]) / 2.0
            cy = (f.bounds_min[1] + f.bounds_max[1]) / 2.0
            n_pts = 64
            angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
            return [np.array([cx + r * math.cos(a), cy + r * math.sin(a)]) for a in angles]

        # Check if we can build a polygon from mesh triangles of the feature faces
        if hasattr(self.s, "mesh") and self.s.mesh is not None and hasattr(self.s.mesh, "triangles"):
            tris = self.s.mesh.triangles
            if len(tris) > 0 and (f.floor_faces or f.face_indices):
                # Sample vertices within feature bounds
                bmin, bmax = f.bounds_min, f.bounds_max
                pts_in_box = []
                for tri in tris:
                    for v in tri:
                        if (bmin[0] - 0.1 <= v[0] <= bmax[0] + 0.1 and
                            bmin[1] - 0.1 <= v[1] <= bmax[1] + 0.1 and
                            bmin[2] - 0.5 <= v[2] <= bmax[2] + 0.5):
                            pts_in_box.append(v[:2])
                if len(pts_in_box) >= 3:
                    hull = self._convex_hull_2d(pts_in_box)
                    if len(hull) >= 3:
                        return hull

        # Fallback: bounding-box rectangle
        bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
        return [
            np.array([bmin[0], bmin[1]]),
            np.array([bmax[0], bmin[1]]),
            np.array([bmax[0], bmax[1]]),
            np.array([bmin[0], bmax[1]]),
        ]

    def _stock_outer_radius(self, center_xy: np.ndarray) -> float:
        """Compute the maximum radial distance from a given 2D center to the outer boundary of the raw stock."""
        stock = self.s.setup.stock
        if stock is None:
            return 10.0
        # If cylindrical stock
        if hasattr(stock, "kind") and getattr(stock.kind, "value", str(stock.kind)) == "cylinder" and hasattr(stock, "radius") and stock.radius > 0:
            s_center = stock.center_xy if (hasattr(stock, "center_xy") and stock.center_xy is not None) else np.zeros(2)
            d_center = float(np.linalg.norm(center_xy[:2] - s_center[:2]))
            return d_center + float(stock.radius)
        if hasattr(stock, "cylinder_radius") and stock.cylinder_radius is not None and stock.cylinder_radius > 0:
            s_center = stock.cylinder_center[:2] if (stock.cylinder_center is not None) else np.zeros(2)
            d_center = float(np.linalg.norm(center_xy[:2] - s_center[:2]))
            return d_center + float(stock.cylinder_radius)
        # Box stock: distance from center to outer bounds
        bmin = stock.bounds_min[:2]
        bmax = stock.bounds_max[:2]
        dx = max(abs(bmax[0] - center_xy[0]), abs(bmin[0] - center_xy[0]))
        dy = max(abs(bmax[1] - center_xy[1]), abs(bmin[1] - center_xy[1]))
        return max(dx, dy)

    def _contour_parallel_passes(self, f: MachiningFeature, tool_radius: float,
                                 finish_wall: float, stepover: float,
                                 z: float) -> list[list[np.ndarray]]:
        """Generate offset polygon contour passes at a given Z level.
        Handles islands (internal obstacles) by generating offset contours
        around both the outer boundary and each island."""
        boundary = self._feature_boundary_polygon(f, tool_radius)
        inset0 = tool_radius + finish_wall
        bmin = np.min(boundary, axis=0)
        bmax = np.max(boundary, axis=0)
        extents = bmax - bmin
        min_extent = float(np.min(extents))
        n_offsets = max(1, int(np.ceil((min_extent / 2.0 - inset0) / stepover)) + 1)
        passes: list[list[np.ndarray]] = []
        # For outer contours / convex bosses: offset outward from part boundary into stock margin
        if not f.is_concave or f.type in (FeatureType.CONTOUR, FeatureType.BOSS):
            stock = self.s.setup.stock
            if stock is not None:
                s_bmin = stock.bounds_min[:2]
                s_bmax = stock.bounds_max[:2]
                margin_x = max(float(s_bmax[0] - bmax[0]), float(bmin[0] - s_bmin[0]), 0.0)
                margin_y = max(float(s_bmax[1] - bmax[1]), float(bmin[1] - s_bmin[1]), 0.0)
                radial_margin = max(margin_x, margin_y)
            else:
                radial_margin = 0.0

            if radial_margin < 1e-4:
                n_offsets = 1
            else:
                n_offsets = max(1, int(np.ceil(radial_margin / stepover)) + 1)
            for oi in range(n_offsets):
                outset = inset0 + (n_offsets - 1 - oi) * (radial_margin / max(1, n_offsets - 1) if n_offsets > 1 else 0.0)
                offset_poly = self._offset_polygon(boundary, outset)
                if len(offset_poly) >= 3:
                    pts = [self._v3(p[0], p[1], z) for p in offset_poly]
                    passes.append(pts)
            return passes

        # Generate offset contours from outer boundary (inward)
        for oi in range(n_offsets):
            inset = inset0 + oi * stepover
            offset_poly = self._offset_polygon(boundary, -inset)
            area = self._polygon_area(offset_poly)
            if area < 1e-6:
                continue
            pts = [self._v3(p[0], p[1], z) for p in offset_poly]
            passes.append(pts)

        # Generate offset contours from islands (outward from island boundary)
        # Clip: only include island contours whose centroid is inside the outer boundary
        if f.island_loops:
            for island in f.island_loops:
                if not island or len(island) < 3:
                    continue
                for oi in range(n_offsets):
                    inset = inset0 + oi * stepover
                    offset_poly = self._offset_polygon(island, inset)
                    area = self._polygon_area(offset_poly)
                    if area < 1e-6:
                        continue
                    # Clip: check if island centroid is inside the outer boundary
                    centroid = np.mean(offset_poly, axis=0)
                    if not self._point_in_polygon(centroid, boundary):
                        continue
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

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: list[np.ndarray]) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        x, y = point[0], point[1]
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    # ----------------------------------------------------------- facing --
    def _facing(self, op: PlannedOperation) -> Toolpath:
        """Face the excess stock above a top face: zig-zag over the full stock envelope
        above the target Z level to remove all top excess material."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params

        excess = f.notes.get("excess_height", op.notes.get("excess_height", getattr(f, "depth", 0.0) or 0.0))
        if excess <= 1e-6:
            tp.metadata["skipped"] = "no excess stock"
            self._end_op(tp, op)
            return tp

        # Facing covers the entire stock envelope above the model top face
        stock_bmin = self.s.setup.stock.bounds_min[:2]
        stock_bmax = self.s.setup.stock.bounds_max[:2]
        region_min = stock_bmin
        region_max = stock_bmax
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

        depth_mode = op.notes.get("depth_mode", "fixed")
        facing_pattern = op.notes.get("facing_pattern", f.notes.get("facing_pattern", "zigzag"))
        depth_levels = self._compute_depth_levels(
            top_z=top_z,
            target_z=target_z,
            max_stepdown=stepdown,
            remaining_allowance=0.0,
            mode=depth_mode,
        )

        stock_is_cyl = (
            hasattr(self.s.setup.stock, "kind")
            and getattr(self.s.setup.stock.kind, "value", str(self.s.setup.stock.kind)) == "cylinder"
        )
        is_circular = (
            stock_is_cyl
            or bool(f.notes.get("is_circular"))
            or self._is_feature_circular(f)
            or facing_pattern in ("circular", "circular_spiral")
        )

        passes = 0
        if is_circular or facing_pattern in ("circular", "circular_spiral"):
            c_xy = (
                self.s.setup.stock.center_xy
                if (stock_is_cyl and self.s.setup.stock.center_xy is not None)
                else f.notes.get("center_xy", 0.5 * (region_min + region_max))
            )
            r_outer = (
                self.s.setup.stock.radius
                if (stock_is_cyl and self.s.setup.stock.radius is not None)
                else (f.diameter / 2.0 if (f.diameter and f.diameter > 0) else 0.5 * max(span[0], span[1]))
            )
            for b, z in enumerate(depth_levels):
                passes += 1
                circ_segs = generate_circular_spiral_facing(
                    center_xy=c_xy,
                    outer_radius=float(r_outer),
                    tool_diameter=tool.diameter,
                    stepover_mm=stepover,
                    z_depth=z,
                    climb=True,
                )
                for s_idx, seg in enumerate(circ_segs):
                    if s_idx == 0 and b == 0:
                        self._link_to(tp, op, seg.start)
                    elif s_idx == 0:
                        self._link_stay_down(tp, op, seg.start)

                    if seg.motion_type == "arc_cw":
                        tp.add(MotionSegment(
                            motion_type=MotionType.ARC_CW,
                            start=seg.start,
                            end=seg.end,
                            arc_center=seg.arc_center,
                            arc_ccw=False,
                            tool_id=tool.id,
                            operation_id=op.id,
                            feature_id=f.id,
                        ))
                    elif seg.motion_type == "arc_ccw":
                        tp.add(MotionSegment(
                            motion_type=MotionType.ARC_CCW,
                            start=seg.start,
                            end=seg.end,
                            arc_center=seg.arc_center,
                            arc_ccw=True,
                            tool_id=tool.id,
                            operation_id=op.id,
                            feature_id=f.id,
                        ))
                    else:
                        self._add_cut(tp, op, seg.end)
        elif facing_pattern == "spiral":
            for b, z in enumerate(depth_levels):
                passes += 1
                spiral_segs = generate_spiral_facing(
                    bounds_min=region_min,
                    bounds_max=region_max,
                    tool_diameter=tool.diameter,
                    stepover_mm=stepover,
                    z_depth=z,
                )
                for s_idx, seg in enumerate(spiral_segs):
                    if s_idx == 0 and b == 0:
                        self._link_to(tp, op, seg.start)
                    elif s_idx == 0:
                        self._link_stay_down(tp, op, seg.start)
                    self._add_cut(tp, op, seg.end)
        elif facing_pattern in ("zigzag_arc", "smooth_zigzag"):
            for b, z in enumerate(depth_levels):
                passes += 1
                zigzag_segs = generate_zigzag_facing(
                    bounds_min=region_min,
                    bounds_max=region_max,
                    tool_diameter=tool.diameter,
                    stepover_mm=stepover,
                    z_depth=z,
                    smooth_arc_links=True,
                )
                for s_idx, seg in enumerate(zigzag_segs):
                    if s_idx == 0 and b == 0:
                        self._link_to(tp, op, seg.start)
                    elif s_idx == 0:
                        self._link_stay_down(tp, op, seg.start)

                    if seg.motion_type == "arc_cw":
                        tp.add(MotionSegment(
                            motion_type=MotionType.ARC_CW,
                            start=seg.start,
                            end=seg.end,
                            arc_center=seg.arc_center,
                            tool_id=tool.id,
                            operation_id=op.id,
                            feature_id=f.id,
                        ))
                    elif seg.motion_type == "arc_ccw":
                        tp.add(MotionSegment(
                            motion_type=MotionType.ARC_CCW,
                            start=seg.start,
                            end=seg.end,
                            arc_center=seg.arc_center,
                            tool_id=tool.id,
                            operation_id=op.id,
                            feature_id=f.id,
                        ))
                    else:
                        self._add_cut(tp, op, seg.end)
        else:
            for b, z in enumerate(depth_levels):
                passes += 1
                direction = 1 if passes % 2 == 1 else -1
                for row in range(n_rows):
                    y_index = row if direction == 1 else n_rows - 1 - row
                    y = r_min[1] + min(y_index * stepover, span[1])
                    x0, x1 = (r_min[0], r_max[0]) if row % 2 == 0 else (r_max[0], r_min[0])
                    if row == 0 and b == 0:
                        self._link_to(tp, op, self._v3(x0, y, z))
                    elif row == 0:
                        self._link_stay_down(tp, op, self._v3(x0, y, z))
                    else:
                        self._add_cut(tp, op, self._v3(x0, y, z))
                    self._add_cut(tp, op, self._v3(x1, y, z))
        # exit and retract
        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)


        total_removal = max(0.0, float(top_z - target_z)) if (top_z is not None and target_z is not None) else float(excess)
        actual_stepdown = float(total_removal / passes) if passes > 0 else float(total_removal)

        tp.metadata["passes"] = passes
        tp.metadata["facing_info"] = {
            "stock_top_z": top_z,
            "target_z": target_z,
            "stock_removal_mm": total_removal,
            "actual_stepdown_mm": actual_stepdown,
            "max_doc_mm": stepdown,
            "passes": passes,
        }
        if "cutting_parameters" in tp.metadata:
            tp.metadata["cutting_parameters"]["actual_depth_of_cut_mm"] = actual_stepdown
            tp.metadata["cutting_parameters"]["stock_removal_z_mm"] = total_removal
            tp.metadata["cutting_parameters"]["max_depth_of_cut_mm"] = stepdown
            
            kc = getattr(self.s.context.material, "specific_cutting_force_kn_mm2", 0.70) if hasattr(self.s.context, "material") and self.s.context.material else 0.70
            mrr_actual = actual_stepdown * stepover * params.feed_rate / 60.0
            power_actual = (mrr_actual * kc) / 1000.0
            if hasattr(self.s.context, "machine") and self.s.context.machine:
                power_actual = min(power_actual, self.s.context.machine.max_spindle_power_kw)
            torque_actual = (power_actual * 9550.0) / max(params.spindle_rpm, 1.0)
            tp.metadata["cutting_parameters"]["power_kw"] = power_actual
            tp.metadata["cutting_parameters"]["torque_nm"] = torque_actual

        return tp

    # -------------------------------------------------------- roughing --
    def _boss_circular_roughing(self, op: PlannedOperation) -> Toolpath:
        """Roughing for circular bosses & cylindrical contours using concentric circular G2/G3 arc passes."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params
        center = f.center_xy
        r_boss = (f.diameter / 2.0) if (f.diameter is not None and f.diameter > 0) else (f.radius if (f.radius is not None and f.radius > 0) else (f.bounds_max[0] - f.bounds_min[0]) / 2.0)
        r_tool = tool.diameter / 2.0
        finish_wall = self.s.finish_allowance_wall
        r_inner = r_boss + finish_wall + r_tool

        r_stock_outer = self._stock_outer_radius(center)
        # To machine from stock outer boundary down to the boss:
        # Innermost pass tool center is at r_inner (tool cutting edge at r_boss + finish_wall).
        # Outermost pass tool center places tool outer edge at the stock outer boundary:
        # r_outer_target = max(r_inner, r_stock_outer - r_tool + min(stepover, r_tool * 0.5))
        stepover = max(params.stepover_mm, 1e-3)
        r_outer_target = max(r_inner, r_stock_outer - r_tool + min(stepover, r_tool * 0.5))
        r_outer = min(r_stock_outer, r_outer_target)
        radial_depth = max(0.0, r_outer - r_inner)
        if radial_depth < 1e-4:
            n_rings = 1
        else:
            n_rings = max(1, int(np.ceil(radial_depth / stepover)) + 1)

        doc = max(params.depth_of_cut_mm, 1e-3)
        depth_mode = op.notes.get("depth_mode", "fixed")
        top_z = op.start_z if (hasattr(op, "start_z") and op.start_z is not None) else f.top_z
        floor_z = op.final_z if (hasattr(op, "final_z") and op.final_z is not None) else f.floor_z
        depth_levels = self._compute_finishing_depth_levels(
            top_z=top_z,
            floor_z=floor_z,
            max_stepdown=doc,
            remaining_allowance=self.s.finish_allowance_floor,
            mode=depth_mode,
        )

        passes = 0
        for b, z_band in enumerate(depth_levels):
            passes += 1

            for ri in range(n_rings):
                outset = (n_rings - 1 - ri) * (radial_depth / max(1, n_rings - 1) if n_rings > 1 else 0.0)
                r_ring = r_inner + outset
                p_start = self._v3(center[0] - r_ring, center[1], z_band)
                p_mid = self._v3(center[0] + r_ring, center[1], z_band)
                c_3d = self._v3(center[0], center[1], z_band)

                # Approach to ring start
                if ri == 0 and b == 0:
                    self._link_to(tp, op, p_start)
                else:
                    prev_end = self._seg_end(tp)
                    if np.linalg.norm(p_start[:2] - prev_end[:2]) < tool.diameter * 2.0:
                        self._link_stay_down(tp, op, p_start)
                    else:
                        self._link_to(tp, op, p_start)

                # Full circle as two 180-degree arcs sharing the exact center
                tp.add(MotionSegment(
                    motion_type=MotionType.ARC_CW,
                    start=p_start,
                    end=p_mid,
                    arc_center=c_3d,
                    arc_ccw=False,
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id,
                    operation_id=op.id,
                    feature_id=f.id
                ))
                tp.add(MotionSegment(
                    motion_type=MotionType.ARC_CW,
                    start=p_mid,
                    end=p_start,
                    arc_center=c_3d,
                    arc_ccw=False,
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id,
                    operation_id=op.id,
                    feature_id=f.id
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
        tp.metadata["passes"] = passes
        tp.metadata["strategy"] = "boss_circular_roughing"
        return tp

    def _roughing(self, op: PlannedOperation) -> Toolpath:
        """Boundary-conforming contour-parallel clearing: true polygon offsets
        from the feature boundary, with engagement-aware corner reduction."""
        f = op.feature
        if (f.type in (FeatureType.BOSS, FeatureType.CONTOUR) and self._is_feature_circular(f)):
            return self._boss_circular_roughing(op)

        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0

        top_z = op.start_z if (hasattr(op, "start_z") and op.start_z is not None) else f.top_z
        target_floor_z = op.final_z if (hasattr(op, "final_z") and op.final_z is not None) else f.floor_z
        stepdown = params.depth_of_cut_mm
        stepover = params.stepover_mm
        finish_wall = self.s.finish_allowance_wall

        depth_mode = op.notes.get("depth_mode", "fixed")
        depth_levels = self._compute_depth_levels(
            top_z=top_z,
            target_z=target_floor_z,
            max_stepdown=stepdown,
            remaining_allowance=self.s.finish_allowance_floor,
            mode=depth_mode,
        )

        first_entry = True
        passes = 0
        for z in depth_levels:
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

        top_z = op.start_z if (hasattr(op, "start_z") and op.start_z is not None) else f.top_z
        target_floor_z = op.final_z if (hasattr(op, "final_z") and op.final_z is not None) else f.floor_z
        floor_z = target_floor_z + self.s.finish_allowance_floor
        stepdown = params.depth_of_cut_mm
        finish_wall = self.s.finish_allowance_wall

        extent = f.xy_extent
        is_slot = min(extent) < 1.5 * tool.diameter

        if is_slot:
            self._trochoidal_slot(tp, op, f, tool, params, top_z, target_floor_z,
                                  stepdown, finish_wall)
        else:
            self._hecm_pocket(tp, op, f, tool, params, top_z, target_floor_z,
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
        depth_mode = op.notes.get("depth_mode", "fixed")
        depth_levels = self._compute_depth_levels(
            top_z=top_z,
            target_z=floor_z,
            max_stepdown=stepdown,
            remaining_allowance=self.s.finish_allowance_floor,
            mode=depth_mode,
        )

        first_entry = True
        for z in depth_levels:
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
        boundary = self._feature_boundary_polygon(f)
        if not boundary or len(boundary) < 3:
            return

        # reduced stepover for adaptive: 10-20% of tool diameter
        ae = tool.diameter * 0.15
        # engagement-aware feed: estimate engagement angle from stepover
        eng_angle = self._estimate_engagement(tool.diameter, ae)
        adapted_feed = self._adaptive_feed(params.feed_rate, eng_angle, 180.0)
        # full depth of cut per level
        depth_mode = op.notes.get("depth_mode", "fixed")
        depth_levels = self._compute_depth_levels(
            top_z=top_z,
            target_z=floor_z,
            max_stepdown=stepdown,
            remaining_allowance=self.s.finish_allowance_floor,
            mode=depth_mode,
        )

        first_entry = True
        for z_level in depth_levels:
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
                current_boundary = offset_poly
                offset_pass += 1

    # --------------------------------------------------- rest machining --
    def _rest_machining(self, op: PlannedOperation) -> Toolpath:
        """Machine only regions where remaining stock stands above the target:
        driven by the current stock heightmap (from simulation of prior ops)."""
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        if not f.is_concave or f.type in (FeatureType.CONTOUR, FeatureType.BOSS, FeatureType.FACING_REGION, FeatureType.PLANAR_WALL):
            tp.metadata["skipped"] = "rest machining not applicable to external/convex features"
            tp.segments = []
            return tp

        tool = op.tool
        params = op.params
        radius = tool.diameter / 2.0

        stock_voxels = self.s.stock_voxels
        if stock_voxels is None:
            from ..geometry.voxel import VoxelStock
            if self.s.setup and getattr(self.s.setup, "stock", None) and self.s.setup.stock:
                stock_voxels = VoxelStock.from_bounds(self.s.setup.stock.bounds_min, self.s.setup.stock.bounds_max, 1.0)
            else:
                stock_voxels = VoxelStock.from_bounds(f.bounds_min, f.bounds_max + np.array([0, 0, 5.0]), 1.0)

        # 1. Check RemainingStock ∩ TargetRemovalVolume in feature bounds
        if hasattr(stock_voxels, "has_remaining_removal") and self.s.target_removal_volume is not None:
            if not stock_voxels.has_remaining_removal(self.s.target_removal_volume, f.bounds_min, f.bounds_max):
                tp.metadata["skipped"] = "no remaining removal volume in feature (RemainingStock ∩ TargetRemovalVolume is empty)"
                tp.segments = []
                return tp

        hm = stock_voxels.heightmap()  # remaining stock top per column
        target_z = f.floor_z + self.s.finish_allowance_floor
        bmin, bmax = f.bounds_min, f.bounds_max
        vs = stock_voxels.voxel_size
        boundary = self._feature_boundary_polygon(f, tool_radius=0.0)

        idx0 = stock_voxels.world_to_index(np.array([bmin[0] + radius, bmin[1] + radius, bmin[2]]))
        idx1 = stock_voxels.world_to_index(np.array([bmax[0] - radius, bmax[1] - radius, bmax[2]]))
        i0, j0 = int(idx0[0]), int(idx0[1])
        i1, j1 = int(idx1[0]), int(idx1[1])
        i0, i1 = max(i0, 0), min(i1, hm.shape[0] - 1)
        j0, j1 = max(j0, 0), min(j1, hm.shape[1] - 1)
        if i1 < i0 or j1 < j0:
            tp.metadata["skipped"] = "no remaining stock in feature"
            tp.segments = []
            return tp

        step = max(1, int(np.ceil(tool.diameter / vs / 2)))
        cut_points: list[tuple[float, float, float]] = []
        for i in range(i0, i1 + 1, step):
            for j in range(j0, j1 + 1, step):
                h = hm[i, j]
                if np.isnan(h):
                    continue
                if h > target_z + 1e-6:
                    w = stock_voxels.index_to_world(np.array([i, j, 0]))
                    if not self._point_in_polygon(w[:2], boundary):
                        continue
                    cut_points.append((w[0], w[1], min(h, f.top_z)))
        if not cut_points:
            tp.metadata["skipped"] = "rest machining not needed (no material above target)"
            tp.segments = []
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
        if f.type in (FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE) or (f.type == FeatureType.POCKET and f.notes.get("is_circular", False)):
            return self._helical_interpolation(op)
        if (f.type in (FeatureType.BOSS, FeatureType.CONTOUR) and self._is_feature_circular(f)):
            return self._boss_circular_contour(op)
        if f.type is FeatureType.FILLET:
            if self.s.mesh is not None and hasattr(self.s.mesh, "triangles") and len(self.s.mesh.triangles) > 0 and op.tool.type is ToolType.BALL_ENDMILL:
                return self._finish_freeform(op)
            return self._finish_wall_floor(op)
        if f.type is FeatureType.GROOVE:
            return self._grooving(op)
        return self._finish_wall_floor(op)

    def _compute_depth_levels(
        self,
        top_z: float,
        target_z: float,
        max_stepdown: float,
        remaining_allowance: float = 0.0,
        mode: str = "fixed",
    ) -> list[float]:
        """Dynamically compute descending machining depth levels from top_z down to (target_z + remaining_allowance).

        The final pass depth is GUARANTEED to be derived directly from the CAD target geometry
        (target_z + remaining_allowance), never from stepdown accumulation (previous_depth + DOC).

        Architecture:
          CAD feature boundary
                  ↓
          target Z/radial surface (target_z)
                  ↓
          roughing step-down planner
                  ↓
          multiple intermediate passes (D1, D2, D3, ...)
                  ↓
          final finishing pass exactly on target (target_z + remaining_allowance)
        """
        target_floor = float(target_z + remaining_allowance)
        safe_z = getattr(self.s.setup, "safe_machining_z_min", float('-inf')) if (hasattr(self, "s") and hasattr(self.s, "setup") and self.s.setup is not None) else float('-inf')
        if safe_z > float('-inf') and target_floor < safe_z:
            target_floor = safe_z + remaining_allowance

        total_depth = float(top_z - target_floor)

        if total_depth <= 1e-6:
            return [round(target_floor, 6)]

        step = max(float(max_stepdown), 1e-4)
        n_passes = max(1, int(math.ceil(total_depth / step - 1e-6)))

        if n_passes == 1:
            return [round(target_floor, 6)]

        if mode == "equalized":
            actual_step = total_depth / n_passes
            levels = [round(top_z - (i + 1) * actual_step, 6) for i in range(n_passes - 1)]
            levels.append(round(target_floor, 6))
            return levels
        else:  # "fixed" / "max_stepdown"
            levels = []
            for i in range(n_passes - 1):
                z_lev = top_z - (i + 1) * step
                levels.append(round(z_lev, 6))
            levels.append(round(target_floor, 6))
            return levels

    def _compute_finishing_depth_levels(
        self,
        top_z: float,
        floor_z: float,
        max_stepdown: float,
        remaining_allowance: float = 0.0,
        mode: str = "fixed",
    ) -> list[float]:
        """Backward-compatible alias for _compute_depth_levels."""
        return self._compute_depth_levels(
            top_z=top_z,
            target_z=floor_z,
            max_stepdown=max_stepdown,
            remaining_allowance=remaining_allowance,
            mode=mode,
        )

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
        top_z = op.start_z if (hasattr(op, "start_z") and op.start_z is not None) else f.top_z
        floor_z = op.final_z if (hasattr(op, "final_z") and op.final_z is not None) else f.floor_z
        doc = max(params.depth_of_cut_mm, 1e-3)
        depth_mode = op.notes.get("depth_mode", "fixed")
        depth_levels = self._compute_finishing_depth_levels(
            top_z=top_z,
            floor_z=floor_z,
            max_stepdown=doc,
            remaining_allowance=0.0,
            mode=depth_mode,
        )
        # Explicit Cutter Compensation Policy
        from .semantic import ToolpathRepresentation, CompensationMode

        comp_mode_str = op.notes.get("compensation_mode", getattr(tool, "compensation_mode", None))
        if comp_mode_str is None:
            comp_mode_str = "wear" if op.notes.get("cutter_comp") else "none"

        try:
            comp_mode = CompensationMode(comp_mode_str)
        except ValueError:
            comp_mode = CompensationMode.NONE

        if comp_mode == CompensationMode.GEOMETRY:
            geom_mode = ToolpathRepresentation.PART_BOUNDARY_PATH
            r_path = r_boss
        else:
            geom_mode = ToolpathRepresentation.CENTER_PATH
            r_path = r_boss + r_tool

        explicit_h = getattr(tool, "length_offset_h", None)
        explicit_d = getattr(tool, "diameter_offset_d", None)
        comp_d = explicit_d or getattr(tool, "tool_number", 1) or 1
        lead_r = max(r_tool * 0.8, 1.0)
        linear_comp_dist = max(r_tool * 1.2, 2.0)

        emits_cutter_comp = comp_mode in (CompensationMode.WEAR, CompensationMode.GEOMETRY, CompensationMode.CONTROL)

        tp.geometry_mode = geom_mode
        tp.metadata["path_type"] = geom_mode.value
        tp.metadata["compensation_mode"] = comp_mode.value
        tp.metadata["toolpath_representation"] = geom_mode.value
        if emits_cutter_comp:
            tp.metadata["cutter_comp_info"] = {
                "d_register": comp_d,
                "diameter_offset_d": explicit_d,
                "length_offset_h": explicit_h,
                "tool_id": tool.id,
                "tool_number": getattr(tool, "tool_number", 1),
                "side": "left",
                "mode": comp_mode.value,
                "path_type": geom_mode.value,
                "lead_len": linear_comp_dist,
                "tool_diameter": float(tool.diameter),
            }
        tp.metadata["depth_levels"] = depth_levels

        for b, z_band in enumerate(depth_levels):
            # Linear approach in free space
            start_cut_pt = self._v3(center[0] + r_path, center[1], z_band)
            arc_lead_in_start = self._v3(center[0] + r_path + lead_r, center[1] - lead_r, z_band)
            approach_pt = self._v3(center[0] + r_path + lead_r + linear_comp_dist, center[1] - lead_r, z_band)
            
            arc_lead_out_end = self._v3(center[0] + r_path + lead_r, center[1] + lead_r, z_band)
            departure_pt = self._v3(center[0] + r_path + lead_r + linear_comp_dist, center[1] + lead_r, z_band)

            self._link_to(tp, op, approach_pt)

            # 1. Linear move (engages G41 only if cutter compensation policy is active)
            entry_meta = {"is_lead": True, "tangent_lead": True}
            if emits_cutter_comp:
                entry_meta["cutter_comp"] = {
                    "side": "left",
                    "action": "on",
                    "d_register": comp_d,
                    "diameter_offset_d": explicit_d,
                }
            tp.add(MotionSegment(
                motion_type=MotionType.ENTRY,
                start=approach_pt,
                end=arc_lead_in_start,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata=entry_meta,
            ))

            # 2. Tangential 90-deg lead-in arc into circle (smooth CW transition to CCW contour)
            tp.add(MotionSegment(
                motion_type=MotionType.ARC_CW,
                start=arc_lead_in_start,
                end=start_cut_pt,
                arc_center=self._v3(center[0] + r_path + lead_r, center[1], z_band),
                arc_ccw=False,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata={"is_lead": True, "tangent_lead": True}
            ))

            # 3. Cutting passes around the cylinder: 4 quadrant arcs sharing exact geometric center
            c_3d = self._v3(center[0], center[1], z_band)
            quads = [
                (start_cut_pt, self._v3(center[0], center[1] + r_path, z_band)),
                (self._v3(center[0], center[1] + r_path, z_band), self._v3(center[0] - r_path, center[1], z_band)),
                (self._v3(center[0] - r_path, center[1], z_band), self._v3(center[0], center[1] - r_path, z_band)),
                (self._v3(center[0], center[1] - r_path, z_band), start_cut_pt),
            ]
            for q_start, q_end in quads:
                tp.add(MotionSegment(
                    motion_type=MotionType.ARC_CCW,
                    start=q_start,
                    end=q_end,
                    arc_center=c_3d,
                    arc_ccw=True,
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id,
                    operation_id=op.id,
                    feature_id=f.id,
                    metadata={"contour_id": f"boss_finish_b{b}", "is_contour": True}
                ))

            # 4. Tangential 90-deg lead-out arc away from circle (smooth CW transition from CCW contour)
            tp.add(MotionSegment(
                motion_type=MotionType.ARC_CW,
                start=start_cut_pt,
                end=arc_lead_out_end,
                arc_center=self._v3(center[0] + r_path + lead_r, center[1], z_band),
                arc_ccw=False,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata={"is_lead": True, "tangent_lead": True}
            ))

            # 5. Linear departure move (cancelling G40 only if cutter compensation was active)
            exit_meta = {"is_lead": True, "tangent_lead": True}
            if emits_cutter_comp:
                exit_meta["cutter_comp"] = {"action": "off"}
            tp.add(MotionSegment(
                motion_type=MotionType.EXIT,
                start=arc_lead_out_end,
                end=departure_pt,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata=exit_meta,
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
        is_outer = (not f.is_concave) or (f.type in (FeatureType.CONTOUR, FeatureType.BOSS))
        is_open_feat = f.type in (FeatureType.STEP, FeatureType.OPEN_POCKET, FeatureType.OPEN_SLOT, FeatureType.SHOULDER, FeatureType.PLANAR_WALL) or f.notes.get("is_open", False)

        # Check if polygon boundary loop is available
        poly_corners: Optional[list[np.ndarray]] = None
        if f.boundary_loop and len(f.boundary_loop) >= 3 and f.type not in (FeatureType.SLOT, FeatureType.PLANAR_WALL):
            boundary = self._feature_boundary_polygon(f, radius)
            offset_dist = radius if is_outer else -radius
            offset_poly = self._offset_polygon(boundary, offset_dist)
            if offset_poly and len(offset_poly) >= 3:
                poly_corners = offset_poly

        if poly_corners is None:
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

            # If tool does not fit inside box, check if it's a planar wall / narrow slot that can take centerline passes
            if (x_max < x_min - 1e-9) or (y_max < y_min - 1e-9):
                if f.type in (FeatureType.PLANAR_WALL, FeatureType.SLOT, FeatureType.OPEN_SLOT, FeatureType.GROOVE) or (bmax[0] - bmin[0] > 1e-4 or bmax[1] - bmin[1] > 1e-4):
                    # Linear centerline pass along the longer dimension
                    cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
                    if (bmax[0] - bmin[0]) >= (bmax[1] - bmin[1]):
                        corners = [np.array([bmin[0], cy]), np.array([bmax[0], cy])]
                    else:
                        corners = [np.array([cx, bmin[1]]), np.array([cx, bmax[1]])]
                else:
                    tp.metadata["skipped"] = "tool does not fit for finishing"
                    self._end_op(tp, op)
                    return tp
            else:
                corners = [np.array([x_min, y_min]), np.array([x_max, y_min]),
                           np.array([x_max, y_max]), np.array([x_min, y_max])]
        else:
            corners = poly_corners

        # wall finishing: full depth contour with cutter compensation (G41 only if requested)
        from .semantic import ToolpathRepresentation, CompensationMode

        floor_z = op.final_z if (hasattr(op, "final_z") and op.final_z is not None) else f.floor_z
        top_z = op.start_z if (hasattr(op, "start_z") and op.start_z is not None) else f.top_z
        doc = max(params.depth_of_cut_mm, 1e-3)
        depth_mode = op.notes.get("depth_mode", "fixed")
        depth_levels = self._compute_finishing_depth_levels(
            top_z=top_z,
            floor_z=floor_z,
            max_stepdown=doc,
            remaining_allowance=0.0,
            mode=depth_mode,
        )

        comp_mode_str = op.notes.get("compensation_mode", getattr(tool, "compensation_mode", None))
        if comp_mode_str is None:
            comp_mode_str = "wear" if op.notes.get("cutter_comp") else "none"

        try:
            comp_mode = CompensationMode(comp_mode_str)
        except ValueError:
            comp_mode = CompensationMode.NONE

        if comp_mode == CompensationMode.GEOMETRY:
            geom_mode = ToolpathRepresentation.PART_BOUNDARY_PATH
        else:
            geom_mode = ToolpathRepresentation.CENTER_PATH

        explicit_h = getattr(tool, "length_offset_h", None)
        explicit_d = getattr(tool, "diameter_offset_d", None)
        comp_d = explicit_d or getattr(tool, "tool_number", 1) or 1
        comp_vec = radius * 0.8  # linear lead-in/out displacement vector

        emits_cutter_comp = comp_mode in (CompensationMode.WEAR, CompensationMode.GEOMETRY, CompensationMode.CONTROL)

        tp.geometry_mode = geom_mode
        tp.metadata["path_type"] = geom_mode.value
        tp.metadata["compensation_mode"] = comp_mode.value
        tp.metadata["toolpath_representation"] = geom_mode.value
        if emits_cutter_comp:
            tp.metadata["cutter_comp_info"] = {
                "d_register": comp_d,
                "diameter_offset_d": explicit_d,
                "length_offset_h": explicit_h,
                "tool_id": tool.id,
                "tool_number": getattr(tool, "tool_number", 1),
                "side": "left",
                "mode": comp_mode.value,
                "path_type": geom_mode.value,
                "lead_len": float(comp_vec),
                "tool_diameter": float(tool.diameter),
            }
        tp.metadata["depth_levels"] = depth_levels

        # Dogbone corner relief for internal concave pockets/slots
        if op.notes.get("dogbone") and not is_outer and len(corners) >= 3:
            from ..algorithms.dressups import generate_dogbone_corners
            corners = [np.array([p[0], p[1]], dtype=float) for p in generate_dogbone_corners(corners, radius)]

        is_closed_loop = len(corners) >= 3

        # Compute 2D centroid for robust outward/inward approach vectors
        if len(corners) >= 3:
            c_arr = np.array([c[:2] for c in corners])
            centroid_2d = c_arr.mean(axis=0)
        else:
            centroid_2d = 0.5 * (bmin + bmax)

        for b, z_band in enumerate(depth_levels):
            band_corners = [self._v3(c[0], c[1], z_band) for c in corners]
            c0 = band_corners[0]
            
            # Outward normal vector from centroid to ensure lead-in/out is in open air
            v_out = c0[:2] - centroid_2d
            norm_v = np.linalg.norm(v_out)
            dir_normal = (v_out / norm_v) if norm_v > 1e-6 else np.array([1.0, 0.0])
            lead_vec = (dir_normal * comp_vec) if is_outer else (-dir_normal * comp_vec)

            approach_pt = self._v3(c0[0] + lead_vec[0], c0[1] + lead_vec[1], z_band)
            departure_pt = self._v3(c0[0] + lead_vec[0], c0[1] + lead_vec[1], z_band)

            self._link_to(tp, op, approach_pt)

            # Linear lead-in move engaging cutter compensation (G41 only if active)
            prev = self._seg_end(tp)
            entry_meta = {}
            if emits_cutter_comp:
                entry_meta["cutter_comp"] = {
                    "side": "left",
                    "action": "on",
                    "d_register": comp_d,
                    "diameter_offset_d": explicit_d,
                }
            tp.add(MotionSegment(
                motion_type=MotionType.ENTRY,
                start=prev,
                end=c0,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata=entry_meta,
            ))

            # Cut around closed contour or linear path with optional elevated holding tabs on final depth level
            cut_targets = (band_corners[1:] + [band_corners[0]]) if is_closed_loop else band_corners[1:]
            if b == len(depth_levels) - 1 and op.notes.get("holding_tabs") and is_closed_loop and is_outer:
                from ..algorithms.dressups import generate_holding_tabs
                tab_count = int(op.notes.get("tab_count", 4))
                tab_w = float(op.notes.get("tab_width", max(tool.diameter * 1.5, 4.0)))
                tab_h = float(op.notes.get("tab_height", 2.0))
                tabbed_pts = generate_holding_tabs(
                    contour_points=[c0] + cut_targets,
                    tab_count=tab_count,
                    tab_width=tab_w,
                    tab_height=tab_h,
                    floor_z=z_band,
                )
                for pt in tabbed_pts[1:]:
                    self._add_cut(tp, op, pt)
            else:
                for c in cut_targets:
                    self._add_cut(tp, op, c)

            # Linear departure move cancelling cutter compensation (G40 only if active)
            prev = self._seg_end(tp)
            exit_meta = {}
            if emits_cutter_comp:
                exit_meta["cutter_comp"] = {"action": "off"}
            tp.add(MotionSegment(
                motion_type=MotionType.EXIT,
                start=prev,
                end=departure_pt,
                feed=params.feed_rate,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
                metadata=exit_meta,
            ))


        # floor finishing: boundary-conforming contour-parallel passes at floor level
        if f.type not in (FeatureType.CONTOUR, FeatureType.PLANAR_WALL, FeatureType.BOSS):
            stepover = max(params.stepover_mm, 1e-3)
            floor_passes = self._contour_parallel_passes(
                f, radius, finish_wall=0.0, stepover=stepover, z=floor_z)
            for pts in floor_passes:
                if not pts:
                    continue
                self._link_to(tp, op, pts[0])
                for pt in pts[1:]:
                    self._add_cut(tp, op, pt)
                if len(pts) > 2:
                    self._add_cut(tp, op, pts[0])

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
        cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
        extents = bmax[:2] - bmin[:2]
        wall_inset = radius

        depth_mode = op.notes.get("depth_mode", "fixed")
        depth_levels = self._compute_depth_levels(
            top_z=top_z,
            target_z=f.floor_z,
            max_stepdown=z_step,
            remaining_allowance=self.s.finish_allowance_floor,
            mode=depth_mode,
        )

        first_entry = True
        passes_count = 0
        for z in depth_levels:
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
        depth_mode = op.notes.get("depth_mode", "fixed")
        depth_levels = self._compute_depth_levels(
            top_z=top_z,
            target_z=floor_z,
            max_stepdown=params.depth_of_cut_mm,
            remaining_allowance=0.0,
            mode=depth_mode,
        )

        passes = 0
        for z in depth_levels:
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
        
        # Tool tip geometry calculation dynamically from tool definition:
        point_angle = float(getattr(tool, "point_angle", 118.0) or 118.0)
        tip_geometry = getattr(tool, "tip_geometry", "conical")
        tip_length = tool.tip_length if hasattr(tool, "tip_length") else (
            float((tool.diameter / 2.0) / math.tan(math.radians(point_angle / 2.0))) if point_angle < 179.9 else 0.0
        )

        # Feature depth definitions & reference:
        start_plane = float(f.top_z)
        req_floor_z = float(f.floor_z)
        drawing_depth = float(f.depth) if (hasattr(f, "depth") and f.depth is not None and f.depth > 0) else (start_plane - req_floor_z)
        clearance_plane = float(self.s.clearance_z)
        r_clearance = float(op.notes.get("r_clearance", getattr(self.s, "approach_margin", 1.0)))
        retract_plane = float(start_plane + r_clearance)

        is_through = (
            f.type in (FeatureType.THROUGH_HOLE, FeatureType.THROUGH_BORE) or
            bool(f.notes.get("is_through")) or
            bool(op.notes.get("is_through"))
        )
        depth_ref_raw = op.notes.get("depth_reference", f.notes.get("depth_reference", "shoulder" if is_through else "shoulder"))
        breakthrough = float(op.notes.get("breakthrough", 0.5 if is_through else 0.0))
        rem_web = op.notes.get("remaining_web") if op.notes.get("remaining_web") is not None else op.notes.get("web_thickness")

        # Authoritative DrillDepthSemantics from operation or dynamically derived
        if getattr(op, "depth_semantics", None) is not None:
            depth_sem = op.depth_semantics
        else:
            depth_sem = tool.compute_drill_depth_semantics(
                start_z=start_plane,
                target_depth=drawing_depth,
                depth_type=depth_ref_raw,
                depth_reference=DepthReference.TOP_FACE,
                breakthrough=breakthrough,
                remaining_web=rem_web,
                is_through=is_through,
            )

        final_depth = float(depth_sem.programmed_tip_depth)
        cylindrical_depth = float(depth_sem.achieved_cylindrical_depth)
        tip_travel = float(depth_sem.total_hole_depth)
        total_depth = tip_travel
        depth_ref = "tip" if depth_sem.depth_type is DrillDepthType.TIP_DEPTH else ("shoulder" if depth_sem.depth_type is DrillDepthType.CYLINDRICAL_DEPTH else "through")

        # Tool-size aware approach standoff distance (clearance above surface or prior cut bottom)
        default_approach_clearance = min(0.20, max(0.02, round(0.10 * tool.diameter, 3)))
        approach_clearance = float(op.notes.get("approach_clearance", default_approach_clearance))

        # First peck, subsequent peck, peck reduction
        nominal_peck = float(params.depth_of_cut_mm if params.depth_of_cut_mm > 0 else tool.diameter * 0.5)
        first_peck = float(op.notes.get("first_peck", nominal_peck))
        subsequent_peck = float(op.notes.get("subsequent_peck", nominal_peck))
        peck_reduction = float(op.notes.get("peck_reduction", 0.0))
        min_peck = float(op.notes.get("min_peck", max(nominal_peck * 0.3, 0.05)))

        # Calculate discrete peck target depths:
        # surface_Z (start_plane) -> surface_Z - peck1 -> surface_Z - peck1 - peck2 -> ... -> final_depth
        peck_depths: list[float] = []
        curr_z = start_plane
        current_peck_inc = first_peck

        while curr_z > final_depth + 1e-9:
            next_z = max(final_depth, curr_z - current_peck_inc)
            peck_depths.append(next_z)
            curr_z = next_z
            if peck_reduction > 0:
                current_peck_inc = max(min_peck, current_peck_inc - peck_reduction)

        # Target drilling points (single hole or multi-point corner relief drilling)
        raw_points = op.notes.get("drill_points")
        if raw_points and len(raw_points) > 0:
            target_points = [np.asarray(p[:2], dtype=float) for p in raw_points]
        else:
            target_points = [center]

        for pt_idx, pt_center in enumerate(target_points):
            # 1. Approach above hole to clearance plane
            self._rapid_to(tp, op, self._v3(pt_center[0], pt_center[1], clearance_plane))

            # 2. Rapid to R-plane
            r_pos = self._v3(pt_center[0], pt_center[1], retract_plane)
            tp.add(MotionSegment(
                motion_type=MotionType.RAPID,
                start=self._v3(pt_center[0], pt_center[1], clearance_plane),
                end=r_pos,
                tool_id=tool.id, operation_id=op.id, feature_id=f.id
            ))

            cycle_type = str(op.notes.get("cycle_type", "G83"))
            mat_name = getattr(self.s.context.material, "name", "") if hasattr(self.s, "context") and getattr(self.s, "context", None) and hasattr(self.s.context, "material") and self.s.context.material else ""

            # Determine retract mode (G98 vs G99) based on operation safety and obstacle environment
            explicit_retract = op.notes.get("retract_mode")
            if explicit_retract in ("G98", "G99"):
                retract_mode = explicit_retract
                retract_reason = "Operation Note Override"
            else:
                has_high_fixture = False
                if hasattr(self.s.setup, "fixtures") and self.s.setup.fixtures:
                    for fix in self.s.setup.fixtures:
                        if fix.bounds_max is not None and fix.bounds_max[2] > retract_plane:
                            has_high_fixture = True
                            break

                is_multi_hole = bool(op.notes.get("is_multi_hole", False) or op.notes.get("hole_count", 1) > 1 or op.notes.get("is_pattern", False) or len(target_points) > 1)
                if has_high_fixture:
                    retract_mode = "G98"
                    retract_reason = "Fixture Clearance Above R-Plane"
                elif is_multi_hole and op.notes.get("planar_pattern", True):
                    retract_mode = "G99"
                    retract_reason = "Planar Pattern Efficient Rapid"
                else:
                    retract_mode = "G98"
                    retract_reason = "Single Hole Clearance"

            # 3. Generate structured peck sequence
            canned_cycle_meta = {
                "canned_cycle": {
                    "type": cycle_type,
                    "cycle": cycle_type,
                    "retract_mode": retract_mode,
                    "retract_reason": retract_reason,
                    "x": float(pt_center[0]),
                    "y": float(pt_center[1]),
                    "position": pt_center.tolist(),
                    "start_plane": float(start_plane),
                    "retract_plane": float(retract_plane),
                    "z_r": float(retract_plane),
                    "z_depth": float(final_depth),
                    "final_depth": float(final_depth),
                    "tip_travel": float(tip_travel),
                    "cylindrical_depth": float(cylindrical_depth),
                    "required_cylindrical_depth": float(cylindrical_depth),
                    "peck": float(first_peck),
                    "first_peck": float(first_peck),
                    "subsequent_peck": float(subsequent_peck),
                    "peck_reduction": float(peck_reduction),
                    "min_peck": float(min_peck),
                    "feed": float(params.plunge_feed),
                    "spindle": float(params.spindle_rpm),
                    "initial_z": float(clearance_plane),
                    "depth_reference": depth_ref,
                    "depth_type": depth_sem.depth_type.value,
                    "depth_ref_datum": depth_sem.depth_reference.value,
                    "target_depth": depth_sem.target_depth,
                    "reference_z": depth_sem.reference_z,
                    "programmed_tip_depth": depth_sem.programmed_tip_depth,
                    "tip_length": float(tip_length),
                    "point_angle_deg": float(point_angle),
                    "tip_geometry": tip_geometry,
                    "floor_z": float(req_floor_z),
                    "remaining_web": float(op.notes.get("remaining_web", f.notes.get("remaining_web", 0.0))),
                    "breakthrough": float(breakthrough),
                    "ld_ratio": float(op.notes.get("ld_ratio", round(cylindrical_depth / max(tool.diameter, 0.01), 2))),
                    "peck_ratio": float(op.notes.get("peck_ratio", round(first_peck / max(tool.diameter, 0.01), 2))),
                    "material_name": mat_name,
                    "derivation_reason": depth_sem.derivation_reason,
                    "depth_semantics": depth_sem.to_dict(),
                }
            }
            if pt_idx == 0:
                tp.metadata["canned_cycle_intent"] = canned_cycle_meta["canned_cycle"]
                tp.metadata["depth_semantics"] = depth_sem.to_dict()

            # Generate clean, independent peck motions
            current_depth = start_plane
            for peck_idx, target_z in enumerate(peck_depths):
                meta = canned_cycle_meta if (peck_idx == 0 and pt_idx == 0) else {}

                # Chip clearance retract (for all pecks after the first)
                if peck_idx > 0:
                    tp.add(MotionSegment(
                        motion_type=MotionType.RETRACT,
                        start=self._v3(pt_center[0], pt_center[1], current_depth),
                        end=r_pos,
                        tool_id=tool.id, operation_id=op.id, feature_id=f.id
                    ))

                # Rapid approach to standoff clearance above current_depth
                approach_plane = min(retract_plane, current_depth + approach_clearance)
                if abs(r_pos[2] - approach_plane) > 1e-6:
                    tp.add(MotionSegment(
                        motion_type=MotionType.RAPID,
                        start=r_pos,
                        end=self._v3(pt_center[0], pt_center[1], approach_plane),
                        tool_id=tool.id, operation_id=op.id, feature_id=f.id
                    ))

                # Controlled plunge feed from approach_plane to target_z
                tp.add(MotionSegment(
                    motion_type=MotionType.PLUNGE,
                    start=self._v3(pt_center[0], pt_center[1], approach_plane),
                    end=self._v3(pt_center[0], pt_center[1], target_z),
                    feed=params.plunge_feed,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id, operation_id=op.id, feature_id=f.id,
                    metadata=meta
                ))

                current_depth = target_z

            # 4. Final Retract to clearance plane
            tp.add(MotionSegment(
                motion_type=MotionType.RETRACT,
                start=self._v3(pt_center[0], pt_center[1], current_depth),
                end=self._v3(pt_center[0], pt_center[1], clearance_plane),
                tool_id=tool.id, operation_id=op.id, feature_id=f.id
            ))

        self._end_op(tp, op)
        tp.metadata["strategy"] = "corner_drilling" if op.notes.get("corner_drilling") else "peck_drill"
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

    # ----------------------------------------------------------- boring --
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

    # ------------------------------------------------------- chamfering --
    def _chamfering(self, op: PlannedOperation) -> Toolpath:
        """Machine a chamfer feature using exact B-Rep conical surface geometry.

        The toolpath is derived entirely from the analytical cone parameters:
          - axis origin and direction (from the revolved profile)
          - r_start / r_end: exact B-Rep radii at the axial extents of the cone
          - z_start / z_end: axial bounds of the conical face along the axis
          - semi_angle_deg: half-angle of the cone measured from the rotation axis

        The tool tip (or lower cutting edge, for a chamfer mill) follows a
        circular arc path in the XY plane at the correct Z level for each
        radial position on the cone, producing a helical-descent trace that
        exactly matches the 3D conical surface.  No rectangular bounding-box
        approximation is used.
        """
        import math as _math
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params

        # ---- Extract B-Rep cone geometry from notes (populated by planner) ----
        semi_angle_deg = op.notes.get("semi_angle_deg") or f.notes.get("semi_angle_deg")
        r_start = op.notes.get("r_start") or f.notes.get("r_start")   # smaller radius
        r_end   = op.notes.get("r_end")   or f.notes.get("r_end")     # larger radius
        z_brep_start = op.notes.get("z_start") or f.notes.get("z_start")
        z_brep_end   = op.notes.get("z_end")   or f.notes.get("z_end")

        # ---- Check if angle information is present ----
        angle_h_raw = op.notes.get("angle_from_horizontal") or f.notes.get("angle_from_horizontal")
        if angle_h_raw is None and semi_angle_deg is None:
            raise CamError(INVALID_TOOLPATH,
                           f"chamfer {f.id}: chamfer angle missing from notes",
                           stage="toolpath", operation_id=op.id, feature_id=f.id)

        # ---- Check if feature is a 3D conical face or a prismatic boundary chamfer ----
        is_cone = (
            f.surface_type == "cone"
            or f.notes.get("surface_kind") == "cone"
            or op.notes.get("surface_type") == "cone"
            or r_start is not None
        )

        if not is_cone:
            # Prismatic / planar chamfer on a boundary or 2D profile
            angle_h = float(angle_h_raw if angle_h_raw is not None else (90.0 - float(semi_angle_deg)))
            depth = f.depth
            horizontal_leg = depth / _math.tan(_math.radians(angle_h)) if depth > 0 else 1.0
            tool_radius = tool.diameter / 2.0
            offset = horizontal_leg + tool_radius
            z_cut = 0.5 * (f.top_z + f.floor_z)

            if f.boundary_loop and len(f.boundary_loop) >= 3:
                poly = [np.asarray(p, dtype=float)[:2] for p in f.boundary_loop]
                from shapely.geometry import Polygon as _Poly
                sp = _Poly(poly)
                off_p = sp.buffer(offset if not f.is_concave else -offset)
                if not off_p.is_empty and off_p.geom_type == "Polygon":
                    coords = list(off_p.exterior.coords)
                    first_pt = self._v3(coords[0][0], coords[0][1], z_cut)
                    self._link_to(tp, op, first_pt)
                    for cx, cy in coords[1:]:
                        self._add_cut(tp, op, self._v3(cx, cy, z_cut))
                else:
                    coords = poly + [poly[0]]
                    first_pt = self._v3(coords[0][0], coords[0][1], z_cut)
                    self._link_to(tp, op, first_pt)
                    for cx, cy in coords[1:]:
                        self._add_cut(tp, op, self._v3(cx, cy, z_cut))
            else:
                bmin_x, bmin_y = float(f.bounds_min[0]), float(f.bounds_min[1])
                bmax_x, bmax_y = float(f.bounds_max[0]), float(f.bounds_max[1])
                rect_pts = [
                    (bmin_x - offset, bmin_y - offset),
                    (bmax_x + offset, bmin_y - offset),
                    (bmax_x + offset, bmax_y + offset),
                    (bmin_x - offset, bmax_y + offset),
                    (bmin_x - offset, bmin_y - offset),
                ]
                first_pt = self._v3(rect_pts[0][0], rect_pts[0][1], z_cut)
                self._link_to(tp, op, first_pt)
                for rx, ry in rect_pts[1:]:
                    self._add_cut(tp, op, self._v3(rx, ry, z_cut))

            last = self._seg_end(tp)
            tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                                 end=self._v3(last[0], last[1], self.s.clearance_z),
                                 tool_id=tool.id, operation_id=op.id, feature_id=f.id))
            self._end_op(tp, op)
            tp.metadata["strategy"] = "chamfer_contour"
            tp.metadata["angle_from_horizontal"] = angle_h
            tp.metadata["chamfer_angle_deg"] = angle_h
            tp.metadata["surface_type"] = "plane"
            return tp

        # Fall back: derive from feature geometry if B-Rep data is absent
        if semi_angle_deg is None:
            angle_h = op.notes.get("angle_from_horizontal") or f.notes.get("angle_from_horizontal")
            if angle_h is not None:
                semi_angle_deg = 90.0 - float(angle_h)
            else:
                raise CamError(INVALID_TOOLPATH,
                               f"chamfer {f.id}: cone geometry (semi_angle_deg) missing from notes",
                               stage="toolpath", operation_id=op.id, feature_id=f.id)

        semi_angle_deg = float(semi_angle_deg)
        if not (0.0 < semi_angle_deg < 90.0):
            raise CamError(INVALID_TOOLPATH,
                           f"chamfer {f.id}: implausible semi_angle {semi_angle_deg:.1f} deg",
                           stage="toolpath", operation_id=op.id, feature_id=f.id)

        # ---- Determine axis and center of the conical face ----
        ax_raw = op.notes.get("axis") or (f.axis.tolist() if f.axis is not None else [0, 0, 1])
        ax = np.asarray(ax_raw, dtype=float)
        if np.linalg.norm(ax) < 1e-9:
            ax = np.array([0.0, 0.0, 1.0])
        ax = ax / np.linalg.norm(ax)

        # Center of the conical feature in XY
        cx, cy = float(f.center_xy[0]), float(f.center_xy[1])

        # Axial bounds of the cone
        z_top    = f.top_z
        z_bottom = f.floor_z
        if z_brep_start is not None:
            z_top    = float(max(z_brep_start, z_brep_end))
            z_bottom = float(min(z_brep_start, z_brep_end))

        # Radii from B-Rep; fall back to diameter property
        if r_start is not None and r_end is not None:
            r_small = float(min(r_start, r_end))
            r_large = float(max(r_start, r_end))
        else:
            r_large = (f.diameter / 2.0) if f.diameter else 5.0
            axial_h = abs(z_top - z_bottom)
            r_small = max(0.0, r_large - axial_h * _math.tan(_math.radians(semi_angle_deg)))

        # ---- Tool geometry compensation ----
        tool_radius = tool.diameter / 2.0
        corner_r    = getattr(tool, "corner_radius", 0.0) or 0.0
        r_offset = tool_radius - corner_r   # effective side-offset from surface

        # ---- Multi-level helical trace down the cone face ----
        axial_length = abs(z_top - z_bottom)
        if axial_length < 1e-4:
            tp.metadata["skipped"] = "degenerate cone (zero axial length)"
            self._end_op(tp, op)
            return tp

        # Number of Z levels: at least 2; finer for steep cones
        n_levels = max(2, int(_math.ceil(axial_length / max(params.depth_of_cut_mm, 0.1))))
        # Points per revolution on the circular contour (resolution ~0.5 mm chord)
        n_pts = max(36, int(_math.ceil(2 * _math.pi * r_large / 0.5)))

        first_cut = True
        for level_i in range(n_levels + 1):
            t_level = level_i / n_levels       # 0.0 at z_top, 1.0 at z_bottom
            z_level = z_top - t_level * axial_length
            # Exact nominal radius at this z on the linear cone
            r_nom = r_small + (r_large - r_small) * t_level
            # Tool center radius
            r_center = r_nom + r_offset
            if r_center < 1e-4:
                r_center = 1e-4

            # Build one full revolution at this Z, sampled as linear cuts
            for k in range(n_pts + 1):
                angle = 2.0 * _math.pi * k / n_pts
                x = cx + r_center * _math.cos(angle)
                y = cy + r_center * _math.sin(angle)
                pt = self._v3(x, y, z_level)
                if first_cut and k == 0:
                    self._link_to(tp, op, pt)
                    first_cut = False
                else:
                    self._add_cut(tp, op, pt)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "chamfer_conical_surface_follow"
        tp.metadata["cone_semi_angle_deg"] = semi_angle_deg
        tp.metadata["cone_r_small_mm"] = r_small
        tp.metadata["cone_r_large_mm"] = r_large
        tp.metadata["cone_z_top"] = z_top
        tp.metadata["cone_z_bottom"] = z_bottom
        tp.metadata["n_levels"] = n_levels
        tp.metadata["surface_type"] = "cone"
        return tp

    # ------------------------------------------------------- filleting --
    def _filleting(self, op: PlannedOperation) -> Toolpath:
        """Machine a toroidal fillet blend using exact B-Rep torus geometry.

        The fillet is a surface of revolution with:
          - major radius r_major: distance from the rotation axis to the centre
            of the circular cross-section (i.e. the nominal OD of the part at
            the fillet shoulder)
          - minor radius r_minor (= fillet_radius): radius of the circular
            arc cross-section

        The tool (ball-nose or bull-nose with corner_radius == fillet_radius)
        follows the toroidal surface by tracing one complete revolution at each
        axial (Z) level, where the XY radius of the tool center at each level
        is computed from the torus parametric equation:

            r_contact(z) = r_major + r_minor * sin(phi)   where phi ∈ [0, π/2]
            z_contact(phi) = z_shoulder + r_minor * (cos(phi) - 1)  (concave blend)

        The tool center is then offset radially by (tool_corner_radius - r_minor)
        so the tool corner stays on the surface.  A ball-nose with
        corner_radius == r_minor needs zero offset (ball tip rides the torus).

        IMPORTANT: this function generates a 3D surface-follow path.  A flat
        circular XY toolpath at a single Z does NOT represent a toroidal surface;
        such an approximation would leave a scallop error equal to r_minor.
        """
        import math as _math
        tp = self._toolpath_skeleton(op)
        self._start_op(tp, op)
        f = op.feature
        tool = op.tool
        params = op.params

        # ---- Extract B-Rep torus geometry ----
        fillet_r = op.notes.get("fillet_radius") or f.fillet_radius or f.radius or 1.0
        fillet_r = float(fillet_r)

        r_start_raw = op.notes.get("r_start") or f.notes.get("r_start")
        r_end_raw   = op.notes.get("r_end")   or f.notes.get("r_end")

        is_revolved_torus = (
            (op.notes.get("surface_type") == "torus" or f.surface_type == "torus")
            and (r_start_raw is not None or f.diameter is not None)
            and (f.axis is None or abs(abs(float(f.axis[2])) - 1.0) < 0.1)
        )

        if not is_revolved_torus:
            # Fall back to 3D surface/freeform or wall finishing for general fillets
            if self.s.mesh is not None and hasattr(self.s.mesh, "triangles") and len(self.s.mesh.triangles) > 0 and tool.type in (ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL):
                return self._finish_freeform(op)
            elif self.s.mesh is not None and hasattr(self.s.mesh, "triangles") and len(self.s.mesh.triangles) > 0:
                return self._finish_wall_floor(op)

        # Major radius: distance from axis to arc centre-line (= shoulder OD / 2 - r_minor)
        if r_start_raw is not None and r_end_raw is not None:
            r_major = float(max(r_start_raw, r_end_raw))
        elif f.diameter is not None and f.diameter > 0:
            r_major = f.diameter / 2.0
        elif f.radius is not None and f.radius > 0:
            r_major = f.radius
        elif f.bounds_max is not None and f.bounds_min is not None:
            r_major = max(float(np.linalg.norm(f.bounds_max[:2] - f.bounds_min[:2])) / 2.0, fillet_r)
        else:
            r_major = max(fillet_r, 5.0)

        # Axial position of the shoulder (bottom Z of the fillet arc)
        z_brep_start = op.notes.get("z_start") or f.notes.get("z_start")
        z_brep_end   = op.notes.get("z_end")   or f.notes.get("z_end")
        if z_brep_start is not None and z_brep_end is not None:
            z_top    = float(max(z_brep_start, z_brep_end))
            z_bottom = float(min(z_brep_start, z_brep_end))
        else:
            z_top    = f.top_z
            z_bottom = f.floor_z
        # The shoulder plane is at z_bottom (where the fillet meets the flat surface)
        z_shoulder = z_bottom

        cx, cy = float(f.center_xy[0]), float(f.center_xy[1])

        # ---- Tool geometry ----
        corner_r = getattr(tool, "corner_radius", 0.0) or 0.0
        # Offset from nominal surface r_contact to tool CENTER:
        # For a ball-nose (corner_r = tool_radius): offset = 0 (ball center on surface)
        # For a bull-nose (corner_r < tool_radius): offset = (tool_radius - corner_r)
        tool_radius = tool.diameter / 2.0
        r_side_offset = tool_radius - corner_r
        # Corner offset in Z: ball tip is below tool center by corner_r
        # We want the ball arc to ride on the torus surface.
        # The tool center height above the nominal contact point is corner_r.

        # ---- Sample the torus arc in N angular steps (phi 0→90°) ----
        # phi = 0: top of the fillet (tangent to the cylindrical OD surface)
        # phi = π/2: bottom of the fillet (tangent to the planar shoulder)
        n_phi = max(12, int(_math.ceil(_math.pi / 2.0 / (_math.radians(2)))))
        # Points per revolution
        r_max_contour = r_major + fillet_r + r_side_offset
        n_pts = max(36, int(_math.ceil(2 * _math.pi * r_max_contour / 0.5)))

        first_cut = True
        for phi_i in range(n_phi + 1):
            phi = (_math.pi / 2.0) * phi_i / n_phi   # 0 → π/2

            # Nominal contact point on the torus surface
            r_nom = r_major - fillet_r + fillet_r * _math.cos(phi)
            # Z of contact point relative to shoulder
            dz_contact = fillet_r * _math.sin(phi)    # rises above shoulder
            z_contact = z_shoulder + dz_contact

            # Tool center: offset radially by r_side_offset, vertically by corner_r
            r_center = r_nom + r_side_offset
            z_center = z_contact + corner_r * (1.0 - _math.sin(phi))  # approx vertical lift
            if r_center < 1e-4:
                r_center = 1e-4

            for k in range(n_pts + 1):
                angle = 2.0 * _math.pi * k / n_pts
                x = cx + r_center * _math.cos(angle)
                y = cy + r_center * _math.sin(angle)
                pt = self._v3(x, y, z_center)
                if first_cut and k == 0:
                    self._link_to(tp, op, pt)
                    first_cut = False
                else:
                    self._add_cut(tp, op, pt)

        last = self._seg_end(tp)
        tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=last,
                             end=self._v3(last[0], last[1], self.s.clearance_z),
                             tool_id=tool.id, operation_id=op.id, feature_id=f.id))
        self._end_op(tp, op)
        tp.metadata["strategy"] = "fillet_toroidal_surface_follow"
        tp.metadata["fillet_radius_mm"] = fillet_r
        tp.metadata["r_major_mm"] = r_major
        tp.metadata["z_shoulder"] = z_shoulder
        tp.metadata["n_phi_levels"] = n_phi
        tp.metadata["surface_type"] = "torus"
        return tp


    # ------------------------------------------------------- thread milling --
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

    # ----------------------------------------------------- countersinking --
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

    # --------------------------------------------------------- grooving --
    def _grooving(self, op: PlannedOperation) -> Toolpath:
        """Grooving: plunge-cut or circular interpolation to create a groove/slot/undercut.
        Supports annular bore grooves and linear slot grooves with chip breaking."""
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

        is_circular = bool(f.notes.get("is_circular") or (f.diameter is not None and f.diameter > tool.diameter and (f.notes.get("is_internal") or f.notes.get("is_annular") or "bore" in str(f.parent_feature_id or ""))))

        if is_circular and f.diameter and f.diameter > tool.diameter:
            # Circular groove (e.g. internal circlip groove or face groove)
            r_groove = (f.diameter - tool.diameter) / 2.0
            r_tool = tool.diameter / 2.0
            lead_r = max(r_tool * 0.5, 0.5)

            # Move over center at clearance, plunge to floor_z
            self._rapid_to(tp, op, self._v3(center[0], center[1], self.s.clearance_z))
            self._rapid_to(tp, op, self._v3(center[0], center[1], top_z + 1.0))
            self._add_cut(tp, op, self._v3(center[0], center[1], floor_z), feed=params.plunge_feed)

            # Radial feed out to groove diameter
            start_cut = self._v3(center[0] + r_groove, center[1], floor_z)
            tp.add(MotionSegment(
                motion_type=MotionType.ENTRY,
                start=self._v3(center[0], center[1], floor_z),
                end=start_cut,
                arc_center=self._v3(center[0] + r_groove / 2.0, center[1], floor_z),
                arc_ccw=True,
                feed=params.feed_rate * 0.5,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id,
            ))

            # 360-degree circular cut in 4 quadrant arcs
            quads = [
                self._v3(center[0], center[1] + r_groove, floor_z),
                self._v3(center[0] - r_groove, center[1], floor_z),
                self._v3(center[0], center[1] - r_groove, floor_z),
                self._v3(center[0] + r_groove, center[1], floor_z),
            ]
            prev = start_cut
            for q in quads:
                tp.add(MotionSegment(
                    motion_type=MotionType.ARC_CCW,
                    start=prev,
                    end=q,
                    arc_center=self._v3(center[0], center[1], floor_z),
                    arc_ccw=True,
                    feed=params.feed_rate,
                    spindle=params.spindle_rpm,
                    tool_id=tool.id,
                    operation_id=op.id,
                    feature_id=f.id
                ))
                prev = q

            # Retract back to center before Z-retract
            tp.add(MotionSegment(
                motion_type=MotionType.EXIT,
                start=prev,
                end=self._v3(center[0], center[1], floor_z),
                arc_center=self._v3(center[0] + r_groove / 2.0, center[1], floor_z),
                arc_ccw=True,
                feed=params.feed_rate * 0.5,
                spindle=params.spindle_rpm,
                tool_id=tool.id,
                operation_id=op.id,
                feature_id=f.id
            ))
        else:
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
