"""Operation planning: derive the machining sequence from geometry and
manufacturing requirements. No fixed sequence is assumed; operations exist only
when the geometry and stock state demand them.

Every candidate stage (facing, roughing, rest, semi, finish, hole work, ...)
first runs the feature-driven requirement chain from cam_engine.requirements:

    detected feature -> required surface -> stock condition -> accessibility
                    -> tool availability -> operation type

A stage emits an operation only when the chain demands it for that feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .context import (
    DepthReference,
    DrillDepthType,
    DrillingDepthSemantics,
    PlanningContext,
    Setup,
    Tool,
    ToolType,
)
from .coords import CoordSpace
from .errors import CamError, FEATURE_INACCESSIBLE, UNSUPPORTED_FEATURE
from .features import (Accessibility, FeatureRecognizer, FeatureType,
                       MachiningFeature)
from .machinability import CuttingParameters, MachinabilityAnalyzer
from .requirements import (
    BORE_TYPES,
    FINISHABLE_TYPES,
    REAMABLE_TYPES,
    ROUGHABLE_TYPES,
    FeatureRequirement,
    evaluate_feature_requirement,
    internal_corner_radius,
)


class OpPurpose(str, Enum):
    FACING = "facing"
    ROUGHING = "roughing"
    REST_MACHINING = "rest_machining"
    SEMI_FINISHING = "semi_finishing"
    FINISHING = "finishing"
    DRILLING = "drilling"
    BORING = "boring"
    SPOT_DRILLING = "spot_drilling"
    REAMING = "reaming"
    TAPPING = "tapping"
    COUNTERSINKING = "countersinking"  # countersink / spotfacing with conical tool
    CHAMFERING = "chamfering"
    FILLETING = "filleting"            # toroidal/round fillet blend
    THREAD_MILLING = "thread_milling"
    GROOVING = "grooving"


@dataclass
class PlannedOperation:
    """One planned machining operation (semantic, pre-toolpath)."""

    id: str
    purpose: OpPurpose
    feature: MachiningFeature
    tool: Tool
    params: CuttingParameters
    depends_on: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)
    depth_semantics: Optional[DrillingDepthSemantics] = None
    manufacturing_geometry: Optional[Any] = None

    def get_manufacturing_geometry(self):
        """Derive explicit manufacturing geometry for this operation using its tool and cut parameters."""
        if self.manufacturing_geometry is not None:
            return self.manufacturing_geometry
        tool_r = getattr(self.tool, "diameter", 6.0) / 2.0
        rad_allow = float(self.notes.get("finish_allowance_side", getattr(self.params, "finish_allowance_side", 0.0)) or 0.0)
        floor_allow = float(self.notes.get("finish_allowance_floor", getattr(self.params, "finish_allowance_floor", 0.0)) or 0.0)
        tool_len = float(getattr(self.tool, "flute_length", 50.0) or 50.0)

        
        mfg = self.feature.compute_manufacturing_geometry(
            tool_radius=tool_r,
            radial_allowance=rad_allow,
            axial_allowance=floor_allow,
            operation_id=self.id,
            tool_id=getattr(self.tool, "id", None),
            tool_length=tool_len,
        )
        self.manufacturing_geometry = mfg
        return mfg


    @property
    def start_z(self) -> float:
        """Nominal cut entry / start Z level for this operation."""
        if "start_z" in self.notes:
            return float(self.notes["start_z"])
        if hasattr(self.feature, "depth_start_z") and self.feature.depth_start_z is not None:
            return float(self.feature.depth_start_z)
        if hasattr(self.feature, "top_z") and self.feature.top_z is not None:
            return float(self.feature.top_z)
        return 0.0

    @property
    def final_z(self) -> float:
        """Nominal floor / bottom Z level for this operation."""
        if "final_z" in self.notes:
            return float(self.notes["final_z"])
        if hasattr(self.feature, "floor_z") and self.feature.floor_z is not None:
            return float(self.feature.floor_z)
        if hasattr(self.feature, "bottom_z") and self.feature.bottom_z is not None:
            return float(self.feature.bottom_z)
        if hasattr(self.feature, "depth") and self.feature.depth is not None:
            return float(self.start_z - self.feature.depth)
        return self.start_z

    def get_approach_z(self, approach_margin: float = 1.0) -> float:
        """Operation-specific approach plane."""
        if "approach_z" in self.notes:
            return float(self.notes["approach_z"])
        return self.start_z + approach_margin

    def get_retract_z(self, retract_margin: float = 2.0) -> float:
        """Operation-specific retract plane."""
        if "retract_z" in self.notes:
            return float(self.notes["retract_z"])
        return self.start_z + retract_margin


class OperationPlanner:
    """Plans operations per setup from recognized features + stock state."""

    def __init__(self, context: PlanningContext, setup: Setup,
                 features: list[MachiningFeature]):
        self.context = context
        self.setup = setup
        self.features = features
        self.analyzer = MachinabilityAnalyzer(context.machine, context.material)
        from .tool_selection import GeometryDrivenToolSelector
        self.tool_selector = GeometryDrivenToolSelector(context.machine, context.material, setup=setup)
        self._counter: dict[str, int] = {}
        self.warnings: list[dict] = []
        self.unmachined_features: list[dict] = []
        # feature id -> evaluated requirement chain (populated by plan())
        self.requirements: dict[str, FeatureRequirement] = {}

    def _next_id(self, prefix: str) -> str:
        n = self._counter.get(prefix, 0) + 1
        self._counter[prefix] = n
        return f"{prefix}_{n:03d}"

    def _req(self, feature: MachiningFeature) -> FeatureRequirement:
        """Requirement chain state for one feature (evaluated once)."""
        req = self.requirements.get(feature.id)
        if req is None:
            req = evaluate_feature_requirement(feature, self.setup, self.context)
            self.requirements[feature.id] = req
        return req

    def plan(self) -> list[PlannedOperation]:
        """Plan the operations this setup actually requires.

        Each detected feature is evaluated through

            feature -> required surface -> stock condition -> accessibility
                    -> tool availability -> operation type

        and every stage below emits an operation only when that evaluation
        demands it.  Operations are then ordered by their dependencies.
        """
        self.requirements = {
            f.id: evaluate_feature_requirement(f, self.setup, self.context)
            for f in self.features
        }
        ops: list[PlannedOperation] = []
        ops += self._plan_facing()
        ops += self._plan_spot_drilling()
        ops += self._plan_roughing()
        ops += self._plan_rest_machining(ops)
        ops += self._plan_semi_finishing(ops)
        ops += self._plan_finishing(ops)
        ops += self._plan_chamfering()
        ops += self._plan_filleting()
        ops += self._plan_drilling()
        ops += self._plan_boring()
        ops += self._plan_reaming()
        ops += self._plan_tapping()
        ops += self._plan_thread_milling()
        ops += self._plan_grooving()
        ops += self._plan_countersinking()
        # Topological sort: reorder by depends_on dependencies
        sorted_ops = self._topo_sort(ops)
        self._validate_feature_webs_and_walls(sorted_ops)
        return sorted_ops

    def _validate_feature_webs_and_walls(self, ops: list[PlannedOperation]) -> None:
        """Generic minimum-wall and web validation system.
        Evaluates nominal feature geometry, tool tip geometry, depth tolerances,
        and remaining material between opposing, collinear, or adjacent features.
        Emits warnings and enforces safe depth references when machining uncertainty
        threatens critical webs."""
        depth_tol = 0.05  # nominal machining/axial depth tolerance (mm)
        min_recommended_web = 0.50  # mm, threshold below which wall/web is thin

        op_by_feature = {op.feature.id: op for op in ops}

        # 1. Pairwise feature web / wall check
        checked_pairs = set()
        for i, f1 in enumerate(self.features):
            for j in range(i + 1, len(self.features)):
                f2 = self.features[j]
                pair_key = (min(f1.id, f2.id), max(f1.id, f2.id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                c1 = f1.center_xy
                c2 = f2.center_xy
                d_xy = float(np.linalg.norm(c1 - c2))

                r1 = float(f1.diameter / 2.0 if f1.diameter else min(f1.xy_extent) / 2.0)
                r2 = float(f2.diameter / 2.0 if f2.diameter else min(f2.xy_extent) / 2.0)

                # Case A: Co-axial / opposing features with overlapping footprint in XY
                if d_xy <= max(r1, r2, 0.5):
                    # Check floor-to-floor web in Z
                    if f1.floor_z is not None and f2.floor_z is not None:
                        z_high = max(f1.floor_z, f2.floor_z)
                        z_low = min(f1.floor_z, f2.floor_z)
                        t_web = float(z_high - z_low)

                        # Only analyze if neither feature is a through-hole past the other
                        is_thru1 = f1.type in (FeatureType.THROUGH_HOLE, FeatureType.THROUGH_BORE) or bool(f1.notes.get("is_through"))
                        is_thru2 = f2.type in (FeatureType.THROUGH_HOLE, FeatureType.THROUGH_BORE) or bool(f2.notes.get("is_through"))

                        if not (is_thru1 or is_thru2) and 0.0 < t_web <= 1.5:
                            op1 = op_by_feature.get(f1.id)
                            op2 = op_by_feature.get(f2.id)

                            h_tip1 = 0.0
                            if op1 and op1.tool.type in (ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL):
                                pt_angle = float(getattr(op1.tool, "point_angle", 118.0) or 118.0)
                                h_tip1 = float((op1.tool.diameter / 2.0) / np.tan(np.radians(pt_angle / 2.0)))

                            h_tip2 = 0.0
                            if op2 and op2.tool.type in (ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL):
                                pt_angle = float(getattr(op2.tool, "point_angle", 118.0) or 118.0)
                                h_tip2 = float((op2.tool.diameter / 2.0) / np.tan(np.radians(pt_angle / 2.0)))

                            # Check if shoulder depth reference would violate the web
                            if op1 and (op1.notes.get("depth_reference") in ("shoulder", "CYLINDRICAL_DEPTH") or (op1.depth_semantics and op1.depth_semantics.depth_type is DrillDepthType.CYLINDRICAL_DEPTH)) and h_tip1 >= t_web - depth_tol:
                                op1.notes["depth_reference"] = "tip"
                                target_d1 = float(f1.depth) if (hasattr(f1, "depth") and f1.depth is not None and f1.depth > 0) else float(f1.top_z - f1.floor_z)
                                op1.depth_semantics = op1.tool.compute_drill_depth_semantics(
                                    start_z=float(f1.top_z),
                                    target_depth=target_d1,
                                    depth_type=DrillDepthType.TIP_DEPTH,
                                    depth_reference=DepthReference.TOP_FACE,
                                    remaining_web=t_web,
                                )
                                op1.notes["depth_semantics"] = op1.depth_semantics.to_dict()
                                self.warnings.append({
                                    "code": "CRITICAL_WEB_THICKNESS",
                                    "feature_id": f1.id,
                                    "opposing_feature_id": f2.id,
                                    "message": f"Critical remaining web of {t_web:.2f} mm between {f1.id} and {f2.id}. Drill tip length ({h_tip1:.2f} mm) would violate the web if programmed to shoulder depth. Enforcing tip-depth programming.",
                                    "nominal_web_mm": round(t_web, 3),
                                    "tool_tip_length_mm": round(h_tip1, 3),
                                })

                            if op2 and (op2.notes.get("depth_reference") in ("shoulder", "CYLINDRICAL_DEPTH") or (op2.depth_semantics and op2.depth_semantics.depth_type is DrillDepthType.CYLINDRICAL_DEPTH)) and h_tip2 >= t_web - depth_tol:
                                op2.notes["depth_reference"] = "tip"
                                target_d2 = float(f2.depth) if (hasattr(f2, "depth") and f2.depth is not None and f2.depth > 0) else float(f2.top_z - f2.floor_z)
                                op2.depth_semantics = op2.tool.compute_drill_depth_semantics(
                                    start_z=float(f2.top_z),
                                    target_depth=target_d2,
                                    depth_type=DrillDepthType.TIP_DEPTH,
                                    depth_reference=DepthReference.TOP_FACE,
                                    remaining_web=t_web,
                                )
                                op2.notes["depth_semantics"] = op2.depth_semantics.to_dict()
                                self.warnings.append({
                                    "code": "CRITICAL_WEB_THICKNESS",
                                    "feature_id": f2.id,
                                    "opposing_feature_id": f1.id,
                                    "message": f"Critical remaining web of {t_web:.2f} mm between {f2.id} and {f1.id}. Drill tip length ({h_tip2:.2f} mm) would violate the web if programmed to shoulder depth. Enforcing tip-depth programming.",
                                    "nominal_web_mm": round(t_web, 3),
                                    "tool_tip_length_mm": round(h_tip2, 3),
                                })

                            # If nominal web is thin, emit validation warning
                            if t_web <= min_recommended_web:
                                self.warnings.append({
                                    "code": "THIN_WALL_WARNING",
                                    "feature_id": f1.id,
                                    "opposing_feature_id": f2.id,
                                    "message": f"Thin remaining web ({t_web:.2f} mm) between features {f1.id} and {f2.id}. Machining uncertainty and tool tip geometry must be tightly controlled to prevent breaching remaining material.",
                                    "details": {
                                        "nominal_web_mm": round(t_web, 3),
                                        "feature_1": {"id": f1.id, "floor_z": round(f1.floor_z, 3), "tool_tip_length": round(h_tip1, 3)},
                                        "feature_2": {"id": f2.id, "floor_z": round(f2.floor_z, 3), "tool_tip_length": round(h_tip2, 3)},
                                        "depth_tolerance_mm": depth_tol,
                                    }
                                })

                # Case B: Adjacent features (XY wall thickness)
                elif d_xy > max(r1, r2):
                    t_wall = float(d_xy - (r1 + r2))
                    if 0.0 < t_wall <= min_recommended_web:
                        z_overlap = min(f1.top_z, f2.top_z) - max(f1.floor_z, f2.floor_z)
                        if z_overlap > 1e-3:
                            self.warnings.append({
                                "code": "THIN_WALL_WARNING",
                                "feature_id": f1.id,
                                "adjacent_feature_id": f2.id,
                                "message": f"Thin remaining wall ({t_wall:.2f} mm) detected between adjacent features {f1.id} and {f2.id}.",
                                "details": {
                                    "nominal_wall_mm": round(t_wall, 3),
                                    "center_distance_mm": round(d_xy, 3),
                                }
                            })

        # 2. Check feature bottom web against stock/model floor
        stock = getattr(self.setup, "stock", None)
        if stock is not None and stock.bounds_min is not None:
            z_stock_floor = float(stock.bounds_min[2])
            for f in self.features:
                if f.floor_z is None:
                    continue
                is_thru = f.type in (FeatureType.THROUGH_HOLE, FeatureType.THROUGH_BORE) or bool(f.notes.get("is_through"))
                if is_thru:
                    continue
                t_bottom = float(f.floor_z - z_stock_floor)
                if 0.0 < t_bottom <= min_recommended_web:
                    op = op_by_feature.get(f.id)
                    h_tip = 0.0
                    if op and op.tool.type in (ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL):
                        pt_angle = float(getattr(op.tool, "point_angle", 118.0) or 118.0)
                        h_tip = float((op.tool.diameter / 2.0) / np.tan(np.radians(pt_angle / 2.0)))

                    if op and (op.notes.get("depth_reference") in ("shoulder", "CYLINDRICAL_DEPTH") or (op.depth_semantics and op.depth_semantics.depth_type is DrillDepthType.CYLINDRICAL_DEPTH)) and h_tip >= t_bottom - depth_tol:
                        op.notes["depth_reference"] = "tip"
                        target_d = float(f.depth) if (hasattr(f, "depth") and f.depth is not None and f.depth > 0) else float(f.top_z - f.floor_z)
                        op.depth_semantics = op.tool.compute_drill_depth_semantics(
                            start_z=float(f.top_z),
                            target_depth=target_d,
                            depth_type=DrillDepthType.TIP_DEPTH,
                            depth_reference=DepthReference.TOP_FACE,
                            remaining_web=t_bottom,
                        )
                        op.notes["depth_semantics"] = op.depth_semantics.to_dict()
                        self.warnings.append({
                            "code": "CRITICAL_WEB_THICKNESS",
                            "feature_id": f.id,
                            "opposing_feature_id": "stock_floor",
                            "message": f"Critical remaining bottom web of {t_bottom:.2f} mm beneath {f.id}. Drill tip length ({h_tip:.2f} mm) would violate the bottom wall if programmed to shoulder depth. Enforcing tip-depth programming.",
                            "nominal_web_mm": round(t_bottom, 3),
                            "tool_tip_length_mm": round(h_tip, 3),
                        })
                    else:
                        self.warnings.append({
                            "code": "THIN_WALL_WARNING",
                            "feature_id": f.id,
                            "message": f"Thin remaining floor web ({t_bottom:.2f} mm) below feature {f.id}.",
                            "details": {
                                "nominal_bottom_web_mm": round(t_bottom, 3),
                                "tool_tip_length_mm": round(h_tip, 3),
                                "depth_tolerance_mm": depth_tol,
                            }
                        })

        # 3. Cross-setup web validation using canonical model transform
        if hasattr(self.context, "setups") and len(self.context.setups) > 1:
            M_curr = self.setup.model_to_setup
            try:
                M_curr_inv = np.linalg.inv(M_curr)
            except np.linalg.LinAlgError:
                M_curr_inv = np.eye(4)

            # Tool axis in model coordinates for current setup
            u_curr = (M_curr_inv @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]

            for other_s in self.context.setups:
                if other_s.id == self.setup.id:
                    continue
                M_other = other_s.model_to_setup
                try:
                    M_other_inv = np.linalg.inv(M_other)
                except np.linalg.LinAlgError:
                    M_other_inv = np.eye(4)

                u_other = (M_other_inv @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]
                # Check if tool axes are opposing
                dot = float(np.dot(u_curr, u_other))
                if dot <= -0.90:  # Opposing setups (e.g. 180 flip)
                    other_shape = self.context.model_shape_in_setup(other_s)
                    if hasattr(other_s, "stock") and other_s.stock is not None:
                        stock_top = float(other_s.stock.bounds_max[2])
                    elif other_shape is not None:
                        _, other_bmax = other_shape.bounding_box()
                        stock_top = float(other_bmax[2]) + 1.0
                    else:
                        stock_top = 0.0
                    from .features import FeatureRecognizer
                    rec = FeatureRecognizer(other_shape, stock_top_z=stock_top)
                    other_feats = rec.recognize()
                    if hasattr(other_s, "features") and other_s.features:
                        other_feats.extend(other_s.features)
                    if hasattr(self.context, "all_features") and self.context.all_features:
                        other_feats.extend([f for f in self.context.all_features if f.notes.get("setup_id") == other_s.id])

                    for f1 in self.features:
                        if f1.floor_z is None:
                            continue
                        is_thru1 = f1.type in (FeatureType.THROUGH_HOLE, FeatureType.THROUGH_BORE) or bool(f1.notes.get("is_through"))
                        if is_thru1:
                            continue
                        # Transform f1 floor center to canonical model coordinates
                        p1_setup = np.array([f1.center_xy[0], f1.center_xy[1], f1.floor_z, 1.0])
                        p1_model = (M_curr_inv @ p1_setup)[:3]
                        r1 = float(f1.diameter / 2.0 if f1.diameter else min(f1.xy_extent) / 2.0)

                        for f2 in other_feats:
                            if f2.floor_z is None:
                                continue
                            is_thru2 = f2.type in (FeatureType.THROUGH_HOLE, FeatureType.THROUGH_BORE) or bool(f2.notes.get("is_through"))
                            if is_thru2:
                                continue
                            # Transform f2 floor center to canonical model coordinates
                            p2_setup = np.array([f2.center_xy[0], f2.center_xy[1], f2.floor_z, 1.0])
                            p2_model = (M_other_inv @ p2_setup)[:3]
                            r2 = float(f2.diameter / 2.0 if f2.diameter else min(f2.xy_extent) / 2.0)

                            # Distance perpendicular to tool axis in model space
                            delta = p1_model - p2_model
                            axial_dist = float(np.dot(delta, u_curr))
                            radial_vec = delta - axial_dist * u_curr
                            radial_dist = float(np.linalg.norm(radial_vec))

                            if radial_dist <= max(r1, r2, 0.5) and 0.0 < axial_dist <= 1.5:
                                t_web = axial_dist
                                op1 = op_by_feature.get(f1.id)
                                h_tip1 = 0.0
                                if op1 and op1.tool.type in (ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL):
                                    pt_angle = float(getattr(op1.tool, "point_angle", 118.0) or 118.0)
                                    h_tip1 = float((op1.tool.diameter / 2.0) / np.tan(np.radians(pt_angle / 2.0)))

                                if op1 and (op1.notes.get("depth_reference") in ("shoulder", "CYLINDRICAL_DEPTH") or (op1.depth_semantics and op1.depth_semantics.depth_type is DrillDepthType.CYLINDRICAL_DEPTH)) and h_tip1 >= t_web - depth_tol:
                                    op1.notes["depth_reference"] = "tip"
                                    target_d1 = float(f1.depth) if (hasattr(f1, "depth") and f1.depth is not None and f1.depth > 0) else float(f1.top_z - f1.floor_z)
                                    op1.depth_semantics = op1.tool.compute_drill_depth_semantics(
                                        start_z=float(f1.top_z),
                                        target_depth=target_d1,
                                        depth_type=DrillDepthType.TIP_DEPTH,
                                        depth_reference=DepthReference.TOP_FACE,
                                        remaining_web=t_web,
                                    )
                                    op1.notes["depth_semantics"] = op1.depth_semantics.to_dict()
                                    self.warnings.append({
                                        "code": "CRITICAL_WEB_THICKNESS",
                                        "feature_id": f1.id,
                                        "opposing_feature_id": f"{other_s.id}_{f2.id}",
                                        "message": f"Critical cross-setup remaining web of {t_web:.2f} mm between {f1.id} ({self.setup.name}) and opposing feature in {other_s.name}. Drill tip length ({h_tip1:.2f} mm) would violate the web if programmed to shoulder depth. Enforcing tip-depth programming.",
                                        "nominal_web_mm": round(t_web, 3),
                                        "tool_tip_length_mm": round(h_tip1, 3),
                                    })

                                if t_web <= min_recommended_web:
                                    self.warnings.append({
                                        "code": "THIN_WALL_WARNING",
                                        "feature_id": f1.id,
                                        "opposing_feature_id": f"{other_s.id}_{f2.id}",
                                        "message": f"Thin remaining web ({t_web:.2f} mm) detected across setups between {f1.id} ({self.setup.name}) and {other_s.name}. Machining uncertainty and tool tip geometry must be tightly controlled.",
                                        "details": {
                                            "nominal_web_mm": round(t_web, 3),
                                            "setup_1": self.setup.name,
                                            "setup_2": other_s.name,
                                            "tool_tip_length": round(h_tip1, 3),
                                            "depth_tolerance_mm": depth_tol,
                                        }
                                    })

    @staticmethod
    def _topo_sort(ops: list[PlannedOperation]) -> list[PlannedOperation]:
        """Topological sort of operations by depends_on relationships.
        Operations with no dependencies come first. Within the same
        dependency level, preserve the original planner ordering."""
        if not ops:
            return ops
        # Build op lookup by id and by feature id
        op_by_id = {o.id: o for o in ops}
        # Map feature_id -> list of op ids that depend on it
        dep_graph: dict[str, list[str]] = {o.id: [] for o in ops}
        in_degree: dict[str, int] = {o.id: 0 for o in ops}
        for o in ops:
            for dep_id in (o.depends_on or []):
                # dep_id could be a feature id or an operation id
                # Find the operation that produces this dependency
                producer = None
                for candidate in ops:
                    if candidate.id == dep_id or candidate.feature.id == dep_id:
                        producer = candidate.id
                        break
                if producer and producer != o.id:
                    dep_graph[producer].append(o.id)
                    in_degree[o.id] += 1
        # Kahn's algorithm
        queue = [oid for oid, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            # Stable: preserve original order within same level
            queue.sort(key=lambda oid: ops.index(op_by_id[oid]))
            oid = queue.pop(0)
            result.append(op_by_id[oid])
            for child in dep_graph[oid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        # If cycle detected, append remaining in original order
        if len(result) < len(ops):
            seen = {o.id for o in result}
            for o in ops:
                if o.id not in seen:
                    result.append(o)
        return result

    def _plan_facing(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type is not FeatureType.FACING_REGION:
                continue
            req = self._req(f)
            # Stock condition: only a planar face with standing stock is faced.
            demanded, reason = req.stock.demands_facing()
            if not demanded:
                continue
            if not req.accessible:
                continue
            tool = self._pick_facing_tool(f)
            if tool is None:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.FACING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_facing"), purpose=OpPurpose.FACING,
                feature=f, tool=tool, params=params,
                notes={
                    "excess_height": req.stock.axial_stock_mm,
                    "requirement": req.trace(OpPurpose.FACING.value, reason),
                }))
        return ops

    def _pick_facing_tool(self, f: MachiningFeature) -> Optional[Tool]:
        res = self.tool_selector.select_optimal_tool(f, "facing", self.context.tools)
        if not res.success or res.selected_tool is None:
            self.warnings.append({
                "code": "MISSING_TOOL",
                "feature_id": f.id,
                "feature_type": "facing_region",
                "message": res.diagnostic_message,
                "suggested_tool": res.suggested_tool,
            })
            return None
        return res.selected_tool

    def _is_machined_in_prior_setup(self, feature: MachiningFeature) -> bool:
        """Check if this through-feature or base contour was already planned/machined in a prior opposing setup."""
        if not hasattr(self.context, "setups") or len(self.context.setups) <= 1:
            return False

        curr_idx = next((i for i, s in enumerate(self.context.setups) if s.id == self.setup.id), -1)
        if curr_idx <= 0:
            return False  # Primary setup machines it first

        M_curr = self.setup.model_to_setup
        try:
            M_curr_inv = np.linalg.inv(M_curr)
        except np.linalg.LinAlgError:
            M_curr_inv = np.eye(4)
        u_curr = (M_curr_inv @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]

        # Case 1: Base outer contour that was already machined from the primary setup
        is_base_contour = bool(feature.notes.get("base_contour"))
        if is_base_contour:
            for prior_idx in range(curr_idx):
                prior_s = self.context.setups[prior_idx]
                M_prior = prior_s.model_to_setup
                try:
                    M_prior_inv = np.linalg.inv(M_prior)
                except np.linalg.LinAlgError:
                    M_prior_inv = np.eye(4)
                u_prior = (M_prior_inv @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]
                if float(np.dot(u_curr, u_prior)) <= -0.85:
                    return True

        # Case 2: Coaxial Holes / Bores / Through features
        is_hole_or_bore = (
            feature.type in (
                FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
                FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
            )
            or bool(feature.notes.get("is_through"))
        )
        if not is_hole_or_bore:
            return False

        c_setup = np.array([feature.center_xy[0], feature.center_xy[1], feature.top_z if feature.top_z is not None else 0.0, 1.0])
        c_model = (M_curr_inv @ c_setup)[:3]
        r1 = float(feature.diameter / 2.0 if feature.diameter else (min(feature.xy_extent) / 2.0 if len(feature.xy_extent) else 1.0))

        for prior_idx in range(curr_idx):
            prior_s = self.context.setups[prior_idx]
            M_prior = prior_s.model_to_setup
            try:
                M_prior_inv = np.linalg.inv(M_prior)
            except np.linalg.LinAlgError:
                M_prior_inv = np.eye(4)
            u_prior = (M_prior_inv @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]

            dot = float(np.dot(u_curr, u_prior))
            if dot <= -0.85:
                prior_feats = getattr(prior_s, "features", []) or []
                if hasattr(self.context, "all_features") and self.context.all_features:
                    prior_feats = [f for f in self.context.all_features if f.notes.get("setup_id") == prior_s.id]
                for f2 in prior_feats:
                    is_f2_cyl = (
                        f2.type in (
                            FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
                            FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
                        )
                        or bool(f2.notes.get("is_through"))
                    )
                    if not is_f2_cyl:
                        continue
                    p2_setup = np.array([f2.center_xy[0], f2.center_xy[1], f2.top_z if f2.top_z is not None else 0.0, 1.0])
                    p2_model = (M_prior_inv @ p2_setup)[:3]
                    delta = c_model - p2_model
                    axial_dist = float(np.dot(delta, u_curr))
                    radial_dist = float(np.linalg.norm(delta - axial_dist * u_curr))
                    if radial_dist <= max(r1, 0.5):
                        return True
        return False

    def _get_effective_z_floor(self, feature: MachiningFeature) -> float:
        """Calculate effective safe cut floor for a feature, respecting fixtures and multi-setup partitioning."""
        safe_fixture_z = getattr(self.setup, "safe_machining_z_min", float('-inf'))
        floor_z = float(feature.floor_z) if feature.floor_z is not None else 0.0
        top_z = float(feature.top_z) if feature.top_z is not None else floor_z + 10.0

        is_outer = (
            feature.type in (FeatureType.CONTOUR, FeatureType.BOSS)
            or bool(feature.notes.get("outer_contour"))
            or bool(feature.notes.get("perimeter"))
        )

        if not is_outer:
            if safe_fixture_z > float('-inf') and floor_z < safe_fixture_z:
                return safe_fixture_z + 0.2
            return floor_z

        if bool(feature.notes.get("step_boss")):
            # Step boss terminates at its explicit step floor face
            if safe_fixture_z > float('-inf') and floor_z < safe_fixture_z:
                return safe_fixture_z + 0.2
            return floor_z

        # For full-height outer contours / perimeters:
        # Check if there are opposing setups (Setup 1 Top, Setup 2 Bottom Flip)
        if hasattr(self.context, "setups") and len(self.context.setups) > 1:
            curr_idx = next((i for i, s in enumerate(self.context.setups) if s.id == self.setup.id), 0)
            M_curr = self.setup.model_to_setup
            try:
                M_curr_inv = np.linalg.inv(M_curr)
            except np.linalg.LinAlgError:
                M_curr_inv = np.eye(4)
            u_curr = (M_curr_inv @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]

            opposing_setups = []
            for s_other in self.context.setups:
                if s_other.id == self.setup.id:
                    continue
                M_other = s_other.model_to_setup
                try:
                    M_other_inv = np.linalg.inv(M_other)
                except np.linalg.LinAlgError:
                    M_other_inv = np.eye(4)
                u_other = (M_other_inv @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]
                if float(np.dot(u_curr, u_other)) <= -0.85:
                    opposing_setups.append(s_other)

            if opposing_setups:
                total_height = top_z - floor_z
                if total_height > 1.0:
                    parting_z = top_z - 0.55 * total_height
                    if safe_fixture_z > float('-inf'):
                        return max(parting_z, safe_fixture_z + 0.5)
                    return parting_z

        if safe_fixture_z > float('-inf') and floor_z < safe_fixture_z:
            return safe_fixture_z + 0.5
        return floor_z

    def _plan_roughing(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type not in ROUGHABLE_TYPES:
                continue
            if self._is_machined_in_prior_setup(f):
                continue
            req = self._req(f)
            if f.accessibility is Accessibility.INACCESSIBLE:
                self.warnings.append({
                    "code": "FEATURE_INACCESSIBLE",
                    "feature_id": f.id,
                    "feature_type": f.type.value if hasattr(f.type, "value") else str(f.type),
                    "message": f"feature {f.id} ({f.type.value}) inaccessible along tool axis",
                })
                continue
            # Stock condition: only material that is really there gets removed.
            demanded, reason = req.stock.demands_roughing()
            if not demanded:
                continue
            tool = self._pick_roughing_tool(f)
            if tool is None:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.ROUGHING.value, self.setup)
            effective_final_z = self._get_effective_z_floor(f)
            notes: dict = {
                "requirement": req.trace(OpPurpose.ROUGHING.value, reason),
                "strategy": req.strategy,
                "final_z": effective_final_z,
            }
            if req.strategy == "pocket":
                # Deep enclosed cavity -> pocket (high-efficiency) clearing
                notes["method"] = "adaptive"
            ops.append(PlannedOperation(
                id=self._next_id("op_rough"), purpose=OpPurpose.ROUGHING,
                feature=f, tool=tool, params=params, notes=notes))
        return ops

    def _pick_roughing_tool(self, f: MachiningFeature) -> Optional[Tool]:
        res = self.tool_selector.select_optimal_tool(f, "roughing", self.context.tools)
        if not res.success or res.selected_tool is None:
            self.warnings.append({
                "code": "MISSING_TOOL",
                "feature_id": f.id,
                "feature_type": f.type.value if hasattr(f.type, "value") else str(f.type),
                "message": res.diagnostic_message,
                "suggested_tool": res.suggested_tool,
            })
            self.unmachined_features.append({
                "id": f.id,
                "diameter": round(res.access_profile.max_tool_diameter, 2) if res.access_profile.max_tool_diameter < 1e6 else None,
                "type": f.type.value if hasattr(f.type, "value") else str(f.type),
                "reason": "No compatible roughing tool available",
            })
            return None
        return res.selected_tool

    def _plan_rest_machining(self, prior: list[PlannedOperation]) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        rough_ops = [o for o in prior if o.purpose is OpPurpose.ROUGHING]
        for rough in rough_ops:
            f = rough.feature
            # Rest machining only applies to concave internal cavity features with internal corners
            if not f.is_concave or f.type in (FeatureType.CONTOUR, FeatureType.BOSS, FeatureType.FACING_REGION, FeatureType.PLANAR_WALL):
                continue
            req = self._req(f)
            if not req.accessible:
                continue
            # Stock condition: rest exists only when the roughing tool really
            # left radial stock in the internal corners of this feature.
            corner_r = internal_corner_radius(f)
            demanded, reason = req.stock.demands_rest(rough.tool, corner_r)
            if not demanded:
                continue
            smaller = [t for t in self.context.tools if t.diameter < rough.tool.diameter - 1e-6]
            res = self.tool_selector.select_optimal_tool(f, "rest_machining", smaller)
            if not res.success or res.selected_tool is None:
                continue
            rest_tool = res.selected_tool
            params = self.analyzer.analyze(rest_tool, f, OpPurpose.REST_MACHINING.value,
                                           self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_rest"), purpose=OpPurpose.REST_MACHINING,
                feature=f, tool=rest_tool, params=params,
                depends_on=[rough.id],
                notes={
                    "parent_rough_tool": rough.tool.id,
                    "requirement": req.trace(OpPurpose.REST_MACHINING.value, reason),
                }))
        return ops

    def _plan_semi_finishing(self, prior: list[PlannedOperation]) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        rough_ops = [o for o in prior if o.purpose is OpPurpose.ROUGHING]
        for rough in rough_ops:
            f = rough.feature
            if not f.is_concave or f.type in (FeatureType.CONTOUR, FeatureType.BOSS, FeatureType.FACING_REGION, FeatureType.PLANAR_WALL):
                continue
            req = self._req(f)
            if not req.accessible:
                continue
            finish_tools = [t for t in self.context.tools if t.diameter < rough.tool.diameter]
            res = self.tool_selector.select_optimal_tool(f, "semi_finishing", finish_tools)
            if not res.success or res.selected_tool is None:
                continue
            # Stock condition: only stock thicker than one finishing pass
            # needs an intermediate semi-finishing pass.
            demanded, reason = req.stock.demands_semi(
                rough.tool, float(rough.params.stepover_mm), res.selected_tool.diameter)
            if not demanded:
                continue
            tool = res.selected_tool
            params = self.analyzer.analyze(tool, f, OpPurpose.SEMI_FINISHING.value,
                                           self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_semi"), purpose=OpPurpose.SEMI_FINISHING,
                feature=f, tool=tool, params=params,
                depends_on=[rough.id],
                notes={"requirement": req.trace(OpPurpose.SEMI_FINISHING.value, reason)}))
        return ops

    def _plan_finishing(self, prior: list[PlannedOperation]) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        roughed = {o.feature.id for o in prior if o.purpose is OpPurpose.ROUGHING}
        for f in self.features:
            if f.type not in FINISHABLE_TYPES:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            if self._is_machined_in_prior_setup(f):
                continue
            extent = f.xy_extent
            if f.type is FeatureType.CONTOUR and float(max(extent) if len(extent) else 0.0) < 1e-6:
                continue
            # Fillets are handled exclusively by the dedicated _plan_filleting() pass
            if f.type is FeatureType.FILLET:
                continue
            req = self._req(f)
            # Required surface + stock condition: finish only what still has
            # stock standing against it.
            demanded, reason = req.demands_finishing(f.id in roughed)
            if not demanded:
                continue
            tool = self._pick_finishing_tool(f)
            if tool is None:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.FINISHING.value,
                                           self.setup)
            effective_final_z = self._get_effective_z_floor(f)
            ops.append(PlannedOperation(
                id=self._next_id("op_finish"), purpose=OpPurpose.FINISHING,
                feature=f, tool=tool, params=params,
                notes={
                    "requirement": req.trace(OpPurpose.FINISHING.value, reason),
                    "strategy": req.strategy,
                    "final_z": effective_final_z,
                }))
        return ops

    def _pick_finishing_tool(self, f: MachiningFeature) -> Optional[Tool]:
        res = self.tool_selector.select_optimal_tool(f, "finishing", self.context.tools)
        if not res.success or res.selected_tool is None:
            self.warnings.append({
                "code": "MISSING_TOOL",
                "feature_id": f.id,
                "feature_type": f.type.value if hasattr(f.type, "value") else str(f.type),
                "message": res.diagnostic_message,
                "suggested_tool": res.suggested_tool,
            })
            self.unmachined_features.append({
                "id": f.id,
                "diameter": round(res.access_profile.max_tool_diameter, 2) if res.access_profile.max_tool_diameter < 1e6 else None,
                "type": f.type.value if hasattr(f.type, "value") else str(f.type),
                "reason": "No compatible finishing tool available",
            })
            return None
        return res.selected_tool

    def _plan_spot_drilling(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        hole_types = (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
        for f in self.features:
            if f.type not in hole_types or f.diameter is None:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            if self._is_machined_in_prior_setup(f):
                continue
            # Stock condition: a hole is only spotted when material fills it.
            req = self._req(f)
            demanded, reason = req.demands_hole_work()
            if not demanded:
                continue
            res = self.tool_selector.select_optimal_tool(f, "spot_drilling", self.context.tools)
            if not res.success or res.selected_tool is None:
                continue
            tool = res.selected_tool
            params = self.analyzer.analyze(tool, f, OpPurpose.SPOT_DRILLING.value, self.setup)

            point_angle = getattr(tool, "point_angle", 90.0) or 90.0
            half_angle_rad = np.radians(point_angle / 2.0)
            target_spot_dia = min(tool.diameter, f.diameter)
            spot_depth = (target_spot_dia / 2.0) / np.tan(half_angle_rad) if half_angle_rad > 0.1 else 0.5
            spot_depth = float(min(spot_depth, (f.depth or 5.0) * 0.2, tool.flute_length))
            spot_depth = max(spot_depth, 0.2)

            ops.append(PlannedOperation(
                id=self._next_id("op_spot"), purpose=OpPurpose.SPOT_DRILLING,
                feature=f, tool=tool, params=params,
                notes={"spot_depth": spot_depth, "point_angle_deg": point_angle,
                       "requirement": req.trace(OpPurpose.SPOT_DRILLING.value, reason)}))
        return ops

    def _plan_drilling(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        hole_types = (
            FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
            FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
            FeatureType.COUNTERBORE,
        )
        for f in self.features:
            if f.type not in hole_types or f.diameter is None:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            if self._is_machined_in_prior_setup(f):
                continue

            # Stock condition: drilling exists only when material fills the bore.
            req = self._req(f)
            demanded, reason = req.demands_hole_work()
            if not demanded:
                continue

            drills = [t for t in self.context.tools if t.type is ToolType.DRILL]
            exact = [d for d in drills if abs(d.diameter - f.diameter) <= 0.05 or abs(d.diameter - round(f.diameter, 2)) < 1e-4]
            if exact and f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                tool = exact[0]
                params = self.analyzer.analyze(tool, f, OpPurpose.DRILLING.value, self.setup)
                peck_strat = self.analyzer.calculate_drilling_peck_strategy(tool, f, coolant_active=params.coolant)
                is_through = f.type is FeatureType.THROUGH_HOLE or bool(f.notes.get("is_through"))
                depth_ref = "shoulder" if is_through else f.notes.get("depth_reference", "shoulder")
                breakthrough = 0.5 if is_through else 0.0

                target_depth = float(f.depth) if (hasattr(f, "depth") and f.depth is not None and f.depth > 0) else float(f.top_z - f.floor_z)
                depth_sem = tool.compute_drill_depth_semantics(
                    start_z=float(f.top_z),
                    target_depth=target_depth,
                    depth_type=depth_ref,
                    depth_reference=DepthReference.TOP_FACE,
                    breakthrough=breakthrough,
                    is_through=is_through,
                )

                ops.append(PlannedOperation(
                    id=self._next_id("op_drill"), purpose=OpPurpose.DRILLING,
                    feature=f, tool=tool, params=params,
                    depth_semantics=depth_sem,
                    notes={
                        "is_through": is_through,
                        "depth_reference": depth_ref,
                        "depth_semantics": depth_sem.to_dict(),
                        "breakthrough": breakthrough,
                        "first_peck": peck_strat["first_peck"],
                        "subsequent_peck": peck_strat["subsequent_peck"],
                        "peck_reduction": peck_strat["peck_reduction"],
                        "min_peck": peck_strat["min_peck"],
                        "cycle_type": peck_strat["cycle_type"],
                        "max_allowable_peck": peck_strat["max_allowable_peck"],
                        "peck_ratio": peck_strat["peck_ratio"],
                        "ld_ratio": peck_strat["ld_ratio"],
                        "requirement": req.trace(OpPurpose.DRILLING.value, reason),
                    }))
                continue

            # If no exact drill, or feature is bore/counterbore -> Plan helical milling with tool_selector
            mills = [t for t in self.context.tools if t.type in ToolType_mill() and t.diameter < f.diameter]
            res = self.tool_selector.select_optimal_tool(f, "drilling", mills)
            if not res.success or res.selected_tool is None:
                is_std_hole = f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
                sugg_type = "drill" if is_std_hole else "flat_endmill"
                if is_std_hole:
                    sugg_dia = round(f.diameter, 2)
                else:
                    sugg_dia = round(max(f.diameter * 0.7, 0.2), 2)
                    if sugg_dia >= f.diameter:
                        sugg_dia = round(f.diameter * 0.5, 2)

                self.warnings.append({
                    "code": "MISSING_TOOL",
                    "feature_id": f.id,
                    "feature_type": f.type.value if hasattr(f.type, "value") else str(f.type),
                    "feature_diameter": round(f.diameter, 2),
                    "depth": round(f.depth, 2) if f.depth else 5.0,
                    "message": f"Feature {f.id} (dia {f.diameter:.2f}mm): no {'drill or endmill' if is_std_hole else 'fitting endmill'} in active library",
                    "suggested_tool": {
                        "type": sugg_type,
                        "diameter": sugg_dia,
                        "flute_length": round(max((f.depth or 5.0) * 1.25, 5.0), 1),
                        "overall_length": round(max((f.depth or 5.0) * 2.0, 30.0), 1),
                        "flutes": 2,
                    }
                })
                self.unmachined_features.append({
                    "id": f.id,
                    "diameter": round(f.diameter, 2),
                    "type": f.type.value if hasattr(f.type, "value") else str(f.type)
                })
                continue
            tool = res.selected_tool
            params = self.analyzer.analyze(tool, f, OpPurpose.DRILLING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_drill"), purpose=OpPurpose.DRILLING,
                feature=f, tool=tool, params=params,
                notes={
                    "method": "helical_interpolation",
                    "target_diameter": f.diameter,
                    "requirement": req.trace(OpPurpose.DRILLING.value, reason),
                }))
        return ops

    def _plan_boring(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type not in BORE_TYPES or f.diameter is None:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            if self._is_machined_in_prior_setup(f):
                continue
            # Required surface: only a precision bore is bored; bulk material
            # is removed by drilling.
            req = self._req(f)
            demanded, reason = req.demands_precision_hole_finish()
            if not demanded:
                continue
            res = self.tool_selector.select_optimal_tool(f, "boring", self.context.tools)
            if not res.success or res.selected_tool is None:
                continue
            tool = res.selected_tool
            if tool.type is ToolType.BORING_BAR and abs(tool.diameter - f.diameter) > 0.05:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.BORING.value, self.setup)
            deps = [f.id]
            if f.parent_feature_id:
                deps.append(f.parent_feature_id)
            ops.append(PlannedOperation(
                id=self._next_id("op_bore"), purpose=OpPurpose.BORING,
                feature=f, tool=tool, params=params, depends_on=deps,
                notes={"requirement": req.trace(OpPurpose.BORING.value, reason)}))
        return ops

    def _plan_reaming(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type not in REAMABLE_TYPES or f.diameter is None:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            # Required surface: reaming is demanded by a precision bore only.
            req = self._req(f)
            demanded, reason = req.demands_precision_hole_finish()
            if not demanded:
                continue
            res = self.tool_selector.select_optimal_tool(f, "reaming", self.context.tools)
            if not res.success or res.selected_tool is None:
                continue
            tool = res.selected_tool
            if abs(tool.diameter - f.diameter) > 0.05:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.REAMING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_ream"), purpose=OpPurpose.REAMING,
                feature=f, tool=tool, params=params, depends_on=[f.id],
                notes={"requirement": req.trace(OpPurpose.REAMING.value, reason)}))
        return ops

    def _plan_tapping(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        _metric_pitch = {
            2.0: 0.4, 2.5: 0.45, 3.0: 0.5, 3.5: 0.6, 4.0: 0.7,
            5.0: 0.8, 6.0: 1.0, 8.0: 1.25, 10.0: 1.5, 12.0: 1.75,
            14.0: 2.0, 16.0: 2.0, 18.0: 2.5, 20.0: 2.5, 22.0: 2.5,
            24.0: 3.0, 27.0: 3.0, 30.0: 3.5, 33.0: 3.5, 36.0: 4.0,
        }
        for f in self.features:
            if f.type not in (FeatureType.TAPPED_HOLE, FeatureType.THREADED_HOLE) and not f.notes.get("tapped"):
                continue
            if f.diameter is None:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            req = self._req(f)
            demanded, reason = req.demands_hole_work()
            if not demanded:
                continue
            res = self.tool_selector.select_optimal_tool(f, "tapping", self.context.tools)
            if not res.success or res.selected_tool is None:
                continue
            tool = res.selected_tool
            if abs(tool.diameter - f.diameter) >= 0.1:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.TAPPING.value, self.setup)
            pitch = _metric_pitch.get(f.diameter, f.diameter * 0.15)
            tap_feed_mm_per_rev = pitch
            ops.append(PlannedOperation(
                id=self._next_id("op_tap"), purpose=OpPurpose.TAPPING,
                feature=f, tool=tool, params=params, depends_on=[f.id],
                notes={"tap_feed_mm_per_rev": tap_feed_mm_per_rev,
                       "requirement": req.trace(OpPurpose.TAPPING.value, reason)}))
        return ops

    def _plan_thread_milling(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type not in (FeatureType.TAPPED_HOLE, FeatureType.THREADED_HOLE) and not f.notes.get("tapped"):
                continue
            if f.diameter is None:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            req = self._req(f)
            demanded, reason = req.demands_hole_work()
            if not demanded:
                continue
            res = self.tool_selector.select_optimal_tool(f, "thread_milling", self.context.tools)
            if not res.success or res.selected_tool is None:
                continue
            tool = res.selected_tool
            params = self.analyzer.analyze(tool, f, OpPurpose.THREAD_MILLING.value, self.setup)
            _metric_pitch = {
                2.0: 0.4, 2.5: 0.45, 3.0: 0.5, 3.5: 0.6, 4.0: 0.7,
                5.0: 0.8, 6.0: 1.0, 8.0: 1.25, 10.0: 1.5, 12.0: 1.75,
                14.0: 2.0, 16.0: 2.0, 18.0: 2.5, 20.0: 2.5, 22.0: 2.5,
                24.0: 3.0, 27.0: 3.0, 30.0: 3.5, 33.0: 3.5, 36.0: 4.0,
            }
            pitch = _metric_pitch.get(f.diameter, f.diameter * 0.15)
            ops.append(PlannedOperation(
                id=self._next_id("op_threadmill"), purpose=OpPurpose.THREAD_MILLING,
                feature=f, tool=tool, params=params, depends_on=[f.id],
                notes={"pitch": pitch, "thread_depth": pitch * 1.2,
                       "requirement": req.trace(OpPurpose.THREAD_MILLING.value, reason)}))
        return ops

    def _plan_grooving(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            is_groove = (
                f.type is FeatureType.GROOVE or
                (f.type in (FeatureType.BORE, FeatureType.THROUGH_BORE) and f.notes.get("groove"))
            )
            if not is_groove:
                continue
            req = self._req(f)
            demanded, reason = req.demands_surface_pass(OpPurpose.GROOVING.value)
            if not demanded:
                continue
            res = self.tool_selector.select_optimal_tool(f, "grooving", self.context.tools)
            if not res.success or res.selected_tool is None:
                continue
            tool = res.selected_tool
            params = self.analyzer.analyze(tool, f, OpPurpose.GROOVING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_groove"), purpose=OpPurpose.GROOVING,
                feature=f, tool=tool, params=params,
                notes={"requirement": req.trace(OpPurpose.GROOVING.value, reason)}))
        return ops

    def _plan_chamfering(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        seen_chamfers = set()
        for f in self.features:
            if f.type is not FeatureType.CHAMFER:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            req = self._req(f)
            demanded, reason = req.demands_surface_pass(OpPurpose.CHAMFERING.value)
            if not demanded:
                continue
            # Deduplicate duplicate/split chamfer features at the exact same location & radius
            center_key = (
                round(float(f.bounds_min[0] + f.bounds_max[0]) / 2.0, 2),
                round(float(f.bounds_min[1] + f.bounds_max[1]) / 2.0, 2),
                round(float(f.floor_z if f.floor_z is not None else (f.bounds_min[2] if len(f.bounds_min) > 2 else 0.0)), 2),
                round(float(f.notes.get("r_start") or 0.0), 2),
                round(float(f.notes.get("r_end") or 0.0), 2),
            )
            if center_key in seen_chamfers:
                continue
            seen_chamfers.add(center_key)

            res = self.tool_selector.select_optimal_tool(f, "chamfering", self.context.tools)
            tool = res.selected_tool if res.success else self._pick_finishing_tool(f)
            if tool is None:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.CHAMFERING.value, self.setup)
            # Carry exact B-Rep cone geometry into the operation notes for the strategy
            cone_notes = {
                "surface_type": f.surface_type,            # "cone"
                "r_start": f.notes.get("r_start"),         # smaller radius of cone (mm)
                "r_end": f.notes.get("r_end"),             # larger radius of cone (mm)
                "z_start": f.notes.get("z_start"),         # axial start of cone (mm)
                "z_end": f.notes.get("z_end"),             # axial end of cone (mm)
                "semi_angle_deg": f.notes.get("semi_angle_deg"),
                # Legacy key expected by older strategy code
                "angle_from_horizontal": 90.0 - float(f.notes.get("semi_angle_deg") or 45.0),
                "axis": f.axis.tolist() if f.axis is not None else [0, 0, 1],
                "requirement": req.trace(OpPurpose.CHAMFERING.value, reason),
            }
            ops.append(PlannedOperation(
                id=self._next_id("op_chamfer"), purpose=OpPurpose.CHAMFERING,
                feature=f, tool=tool, params=params, notes=cone_notes))
        return ops

    def _fillet_tool(self, f: MachiningFeature) -> Optional[Tool]:
        """Tool able to cut the toroidal blend of a FILLET feature.

        Requires corner_radius >= fillet_radius so the tool geometry can fully
        engage the blend arc. Returns None when no such tool exists.
        Only ball-nose and bull-nose endmills are permitted (never face mills).
        """
        fillet_r = f.fillet_radius if (f.fillet_radius is not None and f.fillet_radius > 0.0) else f.radius
        if fillet_r is None or fillet_r <= 0.0:
            return None
        allowed_types = (ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL)
        candidates = [t for t in self.context.tools if t.type in allowed_types]
        if not candidates:
            return None

        # Check maximum diameter bounds if feature has an extent limit
        extent = f.xy_extent
        max_tool_dia = float(max(extent)) * 2.0 if len(extent) and float(max(extent)) > 1e-4 else 1e9

        valid_candidates = [t for t in candidates if t.diameter <= max_tool_dia]
        if not valid_candidates:
            valid_candidates = candidates

        best_tool = None
        for t in sorted(valid_candidates, key=lambda t: t.diameter):
            corner_r = getattr(t, "corner_radius", 0.0) or getattr(t, "tip_radius", 0.0) or (t.diameter / 2.0 if t.type is ToolType.BALL_ENDMILL else 0.0)
            if corner_r >= fillet_r - 1e-3:
                best_tool = t
                break
        if best_tool is None:
            # Fall back to smallest ball-nose / bull-nose available
            radius_fits = [
                t for t in valid_candidates
                if (getattr(t, "corner_radius", 0.0) or getattr(t, "tip_radius", 0.0) or t.diameter / 2.0) and (t.diameter / 2.0 >= fillet_r - 1e-3)
            ]
            if radius_fits:
                best_tool = min(radius_fits, key=lambda t: t.diameter)
        return best_tool

    def _plan_filleting(self) -> list[PlannedOperation]:
        """Plan toroidal fillet finishing operations.

        Fillet features are recognised from B-Rep torus surfaces.  The tool
        must have a corner_radius matching (or smaller than) the fillet_radius
        stored on the feature.  A ball-nose (corner_radius == tool_radius) or
        bull-nose end mill is used.  The toolpath generator will follow the
        actual 3D toroidal surface — not a flat circular XY loop.
        """
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type is not FeatureType.FILLET:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            fillet_r = f.fillet_radius if (f.fillet_radius is not None and f.fillet_radius > 0.0) else f.radius
            if fillet_r is None or fillet_r <= 0.0:
                continue
            req = self._req(f)
            demanded, reason = req.demands_surface_pass(OpPurpose.FILLETING.value)
            if not demanded:
                continue

            best_tool = self._fillet_tool(f)
            if best_tool is None:
                self.warnings.append({
                    "type": "no_fillet_tool",
                    "feature_id": f.id,
                    "fillet_radius_mm": fillet_r,
                    "message": (
                        f"No tool with corner_radius >= {fillet_r:.3f} mm "
                        f"available for fillet {f.id}; operation skipped."
                    ),
                })
                continue

            params = self.analyzer.analyze(best_tool, f, OpPurpose.FILLETING.value, self.setup)
            torus_notes = {
                "surface_type": f.surface_type,             # "torus"
                "fillet_radius": fillet_r,                  # minor (curvature) radius (mm)
                "r_start": f.notes.get("r_start"),          # major OD at arc start
                "r_end": f.notes.get("r_end"),              # major OD at arc end
                "z_start": f.notes.get("z_start"),          # axial start (mm)
                "z_end": f.notes.get("z_end"),              # axial end (mm)
                "axis": f.axis.tolist() if f.axis is not None else [0, 0, 1],
                "curvature_radius": f.notes.get("curvature_radius"),
                "requirement": req.trace(OpPurpose.FILLETING.value, reason),
            }
            ops.append(PlannedOperation(
                id=self._next_id("op_fillet"), purpose=OpPurpose.FILLETING,
                feature=f, tool=best_tool, params=params, notes=torus_notes))
        return ops

    def _plan_countersinking(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        seen_countersinks = set()
        for f in self.features:
            if f.type is not FeatureType.COUNTERSINK:
                continue
            if f.diameter is None:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                continue
            # Deduplicate concentric countersinks at same location & depth
            center_key = (
                round(float(f.bounds_min[0] + f.bounds_max[0]) / 2.0, 2),
                round(float(f.bounds_min[1] + f.bounds_max[1]) / 2.0, 2),
                round(float(f.floor_z if f.floor_z is not None else (f.bounds_min[2] if len(f.bounds_min) > 2 else 0.0)), 2),
                round(float(f.diameter or 0.0), 2),
            )
            if center_key in seen_countersinks:
                continue
            seen_countersinks.add(center_key)

            res = self.tool_selector.select_optimal_tool(f, "countersinking", self.context.tools)
            if not res.success or res.selected_tool is None:
                continue
            tool = res.selected_tool
            params = self.analyzer.analyze(tool, f, OpPurpose.COUNTERSINKING.value, self.setup)
            deps = [f.parent_feature_id] if f.parent_feature_id else []
            semi_angle_deg = f.notes.get("semi_angle_deg", 45.0)
            ops.append(PlannedOperation(
                id=self._next_id("op_csink"), purpose=OpPurpose.COUNTERSINKING,
                feature=f, tool=tool, params=params, depends_on=deps,
                notes={"countersink_angle_deg": 90.0 - float(semi_angle_deg)}))
        return ops


def ToolType_mill():
    """Milling-capable tool types."""
    from .context import ToolType
    return (ToolType.FLAT_ENDMILL, ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL, ToolType.FACE_MILL)
