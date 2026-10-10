"""Geometry-driven Feature-to-Tool Compatibility and Selection Engine.

Pipeline:
Feature
   ↓
Required access (geometry envelope, cavity bounds, corner radius, island gaps)
   ↓
Minimum / optimal tool diameter & reach bounds
   ↓
Available tools filtering (library compatibility)
   ↓
Machine compatibility (spindle RPM, power, torque, axis travel)
   ↓
Material cutting data (speeds, feeds, deflection, MRR)
   ↓
Selected tool (or actionable diagnostic)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .context import MachineConfig, Material, Setup, Tool, ToolType
from .errors import (
    CamError,
    FEATURE_INACCESSIBLE,
    NO_COMPATIBLE_TOOL,
    TOOL_REACH_INSUFFICIENT,
)
from .features import Accessibility, FeatureType, MachiningFeature
from .machinability import CuttingParameters, MachinabilityAnalyzer


@dataclass
class GeometricAccessProfile:
    """Rigorous physical access requirements extracted from nominal CAD geometry."""

    feature_id: str
    feature_type: FeatureType
    purpose: str                                # facing, roughing, finishing, drilling, etc.
    max_tool_diameter: float                    # Absolute geometric upper bound (cavity, slot, corner, island clearance)
    min_recommended_diameter: float             # Lower bound for rigidity / minimum deflection
    required_reach: float                       # Minimum required axial reach / cutting depth
    max_holder_diameter: Optional[float]        # Max allowed holder OD if reaching deep cavity
    permissible_tool_types: list[ToolType]      # Permissible cutter types
    is_internal_cavity: bool                    # True for enclosed internal voids
    requires_corner_fit: bool                   # True if cutter radius must conform to internal corner/fillet
    max_corner_radius: Optional[float] = None   # Maximum cutter corner/tip radius
    min_channel_width: Optional[float] = None   # Width of slot / channel
    min_wall_clearance: Optional[float] = None  # Clearance between islands / walls
    ideal_diameter_range: tuple[float, float] = (0.0, 0.0) # (ideal_min, ideal_max)
    notes: dict = field(default_factory=dict)


@dataclass
class ToolCompatibilityCandidate:
    """Evaluation result for one candidate tool through the geometry-driven pipeline."""

    tool: Tool
    is_compatible: bool
    rejection_reasons: list[str]
    geometric_fit_score: float                  # Sizing, rigidity, and corner conformance score
    machine_fit_score: float                    # Spindle RPM, power, torque, and travel score
    cutting_params: Optional[CuttingParameters] = None
    composite_score: float = 0.0                # Overall ranking score


@dataclass
class ToolSelectionResult:
    """Auditable result of feature-to-tool selection."""

    feature: MachiningFeature
    purpose: str
    selected_tool: Optional[Tool]
    access_profile: GeometricAccessProfile
    candidates_evaluated: list[ToolCompatibilityCandidate]
    success: bool
    diagnostic_message: str = ""
    suggested_tool: Optional[dict[str, Any]] = None


class GeometryDrivenToolSelector:
    """Automated, geometry-driven tool compatibility and selection engine."""

    def __init__(
        self,
        machine: MachineConfig,
        material: Material,
        setup: Optional[Setup] = None,
    ):
        self.machine = machine
        self.material = material
        self.setup = setup
        self.analyzer = MachinabilityAnalyzer(machine, material)

    # -------------------------------------------------------------------------
    # Step 1: Feature -> Required Access Profile
    # -------------------------------------------------------------------------
    def compute_access_profile(
        self,
        feature: MachiningFeature,
        purpose: str,
    ) -> GeometricAccessProfile:
        """Extract exact geometric access requirements from nominal CAD geometry."""
        ftype = feature.type
        purpose_lower = purpose.lower()

        # 1. Permissible Tool Types
        permissible_types: list[ToolType] = []
        if purpose_lower == "facing":
            permissible_types = [ToolType.FACE_MILL, ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL]
        elif purpose_lower == "spot_drilling":
            permissible_types = [ToolType.SPOT_DRILL, ToolType.CENTER_DRILL, ToolType.CHAMFER_MILL]
        elif purpose_lower in ("drilling", "peck_drilling"):
            permissible_types = [ToolType.DRILL, ToolType.FLAT_ENDMILL]
        elif purpose_lower == "boring":
            permissible_types = [ToolType.BORING_BAR]
        elif purpose_lower == "reaming":
            permissible_types = [ToolType.REAMER]
        elif purpose_lower == "tapping":
            permissible_types = [ToolType.TAP]
        elif purpose_lower == "thread_milling":
            permissible_types = [ToolType.THREAD_MILL]
        elif purpose_lower in ("corner_drilling", "pre_drilling"):
            permissible_types = [ToolType.DRILL, ToolType.FLAT_ENDMILL, ToolType.SPOT_DRILL]
        elif purpose_lower in ("countersinking", "chamfering"):
            permissible_types = [ToolType.COUNTERSINK_TOOL, ToolType.CHAMFER_MILL]
        elif purpose_lower == "grooving":
            permissible_types = [ToolType.GROOVE_CUTTER, ToolType.FLAT_ENDMILL]
        elif purpose_lower == "filleting":
            permissible_types = [ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL]
        elif purpose_lower in ("finishing", "semi_finishing", "rest_machining"):
            if ftype in (FeatureType.FREEFORM_SURFACE, FeatureType.FILLET):
                permissible_types = [ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL, ToolType.FLAT_ENDMILL]
            elif ftype is FeatureType.CHAMFER:
                permissible_types = [ToolType.CHAMFER_MILL, ToolType.COUNTERSINK_TOOL]
            elif ftype is FeatureType.GROOVE:
                permissible_types = [ToolType.GROOVE_CUTTER, ToolType.FLAT_ENDMILL]
            else:
                permissible_types = [ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL]
        else:  # roughing
            if ftype is FeatureType.GROOVE:
                permissible_types = [ToolType.GROOVE_CUTTER, ToolType.FLAT_ENDMILL]
            elif ftype in (FeatureType.FREEFORM_SURFACE, FeatureType.FILLET):
                permissible_types = [ToolType.BULLNOSE_ENDMILL, ToolType.BALL_ENDMILL, ToolType.FLAT_ENDMILL]
            else:
                permissible_types = [ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL]

        # 2. Enclosed Cavity & Internal Geometry Detection
        internal_cavity_types = {
            FeatureType.HOLE, FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE,
            FeatureType.COUNTERBORE, FeatureType.TAPPED_HOLE,
            FeatureType.THREADED_HOLE, FeatureType.BORE, FeatureType.BLIND_BORE,
            FeatureType.THROUGH_BORE, FeatureType.POCKET, FeatureType.OPEN_POCKET,
            FeatureType.SLOT, FeatureType.OPEN_SLOT, FeatureType.GROOVE,
        }
        is_internal = bool(feature.is_concave and (
            ftype in internal_cavity_types
            or (ftype is FeatureType.FILLET and feature.is_concave)
        ))

        # 3. Required Reach & Depth
        feature_depth = float(feature.depth if (feature.depth is not None and feature.depth > 0) else (
            feature.top_z - feature.floor_z if (feature.top_z is not None and feature.floor_z is not None) else 0.0
        ))
        if purpose_lower in ("spot_drilling", "countersinking", "chamfering") or ftype in (FeatureType.COUNTERSINK, FeatureType.CHAMFER):
            required_reach = float(feature.notes.get("spot_depth", feature.notes.get("chamfer_depth", min(feature_depth, 5.0))))
        else:
            required_reach = max(feature_depth, 0.5)

        # 4. Feature Bounds & Spans
        extent = feature.xy_extent
        min_span = float(min(extent)) if len(extent) and float(min(extent)) > 1e-4 else 1e9
        if feature.diameter is not None and feature.diameter > 1e-4:
            feat_dia = float(feature.diameter)
            if feat_dia < min_span or min_span > 1e6:
                min_span = feat_dia
        else:
            feat_dia = min_span if min_span < 1e6 else None

        # 5. Wall Clearance & Island Gaps
        min_wall_clearance = None
        if feature.boundary_loop and feature.island_loops:
            outer_arr = np.asarray(feature.boundary_loop, dtype=float)
            for isl in feature.island_loops:
                isl_arr = np.asarray(isl, dtype=float)
                if len(outer_arr) > 0 and len(isl_arr) > 0:
                    dists = [np.min(np.linalg.norm(outer_arr[:, :2] - pt[:2], axis=1)) for pt in isl_arr]
                    if dists:
                        gap = float(np.min(dists))
                        # Ignore step-shelf artifacts / concentric openings (< 0.5mm)
                        if gap >= 0.5 and (min_wall_clearance is None or gap < min_wall_clearance):
                            min_wall_clearance = gap

        # 6. Corner / Fillet Radius
        corner_r = None
        if feature.fillet_radius is not None and feature.fillet_radius > 1e-4:
            corner_r = float(feature.fillet_radius)
        elif feature.radius is not None and ftype in (FeatureType.FILLET, FeatureType.POCKET, FeatureType.OPEN_POCKET, FeatureType.SLOT, FeatureType.OPEN_SLOT):
            corner_r = float(feature.radius)
        elif feature.notes.get("min_corner_radius") is not None:
            corner_r = float(feature.notes["min_corner_radius"])
        elif feature.notes.get("corner_radius") is not None:
            corner_r = float(feature.notes["corner_radius"])
        elif feature.boundary_loop and len(feature.boundary_loop) >= 8 and purpose_lower in ("finishing", "semi_finishing"):
            pts = np.asarray(feature.boundary_loop, dtype=float)
            diffs = np.diff(pts[:, :2], axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            valid_lens = seg_lens[seg_lens > 0.05]
            if len(valid_lens) > 0:
                min_seg = float(np.min(valid_lens))
                if min_seg < 3.0:
                    corner_r = max(min_seg * 1.5, 1.0)


        requires_corner_fit = bool(
            corner_r is not None
            and purpose_lower in ("finishing", "semi_finishing", "rest_machining", "corner_drilling", "pre_drilling")
            and purpose_lower not in ("spot_drilling", "countersinking", "chamfering")
        )
        max_corner_radius = corner_r if requires_corner_fit else None


        # 7. Maximum Allowed Tool Diameter Calculation
        # Base ceiling: default to large capacity unless constrained by geometry
        max_allowed_dia = 100.0

        if purpose_lower in ("corner_drilling", "pre_drilling"):
            # Corner drilling tool must fit the smallest corner radius (diameter <= 2 * R_min)
            min_c_r = float(feature.notes.get("min_corner_radius") or corner_r or 1.0)
            max_allowed_dia = 2.0 * min_c_r + 0.05

        elif ftype in (FeatureType.HOLE, FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE, FeatureType.TAPPED_HOLE, FeatureType.THREADED_HOLE):
            if purpose_lower in ("drilling", "tapping", "reaming"):
                max_allowed_dia = feat_dia or 100.0
            elif purpose_lower in ("spot_drilling", "countersinking"):
                # Conical spot drill can have tool body >= hole diameter
                max_allowed_dia = 50.0
            else:
                max_allowed_dia = (feat_dia - 1e-4) if feat_dia else 100.0

        elif ftype in (FeatureType.BORE, FeatureType.BLIND_BORE, FeatureType.THROUGH_BORE, FeatureType.COUNTERBORE):
            if feat_dia:
                max_allowed_dia = feat_dia

        elif ftype in (FeatureType.SLOT, FeatureType.OPEN_SLOT, FeatureType.GROOVE):
            slot_w = feat_dia or min_span
            if slot_w < 1e6:
                max_allowed_dia = slot_w

        elif ftype in (FeatureType.POCKET, FeatureType.OPEN_POCKET):
            if min_span < 1e6:
                max_allowed_dia = min_span

        elif ftype in (FeatureType.CHAMFER, FeatureType.COUNTERSINK):
            # Chamfer tools need sufficient body diameter to span the chamfer width
            max_allowed_dia = 50.0

        elif ftype is FeatureType.FILLET and is_internal:
            if corner_r is not None:
                max_allowed_dia = 2.0 * corner_r

        # Apply internal corner radius restriction for finishing operations
        if requires_corner_fit and corner_r is not None and purpose_lower not in ("roughing",):
            max_allowed_dia = min(max_allowed_dia, 2.0 * corner_r)

        # Apply island / wall gap restriction
        if min_wall_clearance is not None and min_wall_clearance > 1e-4:
            max_allowed_dia = min(max_allowed_dia, min_wall_clearance)

        # External scale constraint (bosses, contours, facing)
        # Prevents selecting giant cutters on tiny features
        if not is_internal and ftype not in (FeatureType.FACING_REGION, FeatureType.CONTOUR):
            if ftype == FeatureType.BOSS:
                external_bound = max(min_span * 0.8, 1.5) if min_span < 1e6 else 50.0
            else:
                external_bound = max(min_span * 2.0, 3.0) if min_span < 1e6 else 50.0
                if self.setup and getattr(self.setup, "stock", None) is not None:
                    stk = self.setup.stock
                    if stk.bounds_min is not None and stk.bounds_max is not None:
                        stk_span = float(np.min(np.abs(stk.bounds_max[:2] - stk.bounds_min[:2])))
                        if stk_span > 1e-3:
                            external_bound = min(external_bound, max(stk_span * 1.5, 4.0))
            max_allowed_dia = min(max_allowed_dia, external_bound)



        # 8. Minimum Recommended Tool Diameter & Ideal Range
        if ftype in (FeatureType.HOLE, FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE, FeatureType.TAPPED_HOLE):
            min_rec_dia = feat_dia if feat_dia else 1.0
            ideal_range = (min_rec_dia, max_allowed_dia)
        elif is_internal and min_span < 1e6:
            # Traditional shop best practice: pocketing cutter dia 0.5 to 0.75 of cavity span
            ideal_min = max(0.40 * min_span, 0.5)
            ideal_max = min(0.75 * min_span, max_allowed_dia)
            if ideal_max < ideal_min:
                ideal_min = ideal_max * 0.8
            min_rec_dia = max(0.25 * min_span, 0.4)
            ideal_range = (ideal_min, ideal_max)
        else:
            min_rec_dia = 1.0
            ideal_range = (min_rec_dia, max_allowed_dia)

        # 9. Maximum Holder Diameter
        max_holder_dia = None
        if is_internal and min_span < 1e6:
            max_holder_dia = min_span

        return GeometricAccessProfile(
            feature_id=feature.id,
            feature_type=ftype,
            purpose=purpose_lower,
            max_tool_diameter=max_allowed_dia,
            min_recommended_diameter=min_rec_dia,
            required_reach=required_reach,
            max_holder_diameter=max_holder_dia,
            permissible_tool_types=permissible_types,
            is_internal_cavity=is_internal,
            requires_corner_fit=requires_corner_fit,
            max_corner_radius=max_corner_radius,
            min_channel_width=min_span if min_span < 1e6 else None,
            min_wall_clearance=min_wall_clearance,
            ideal_diameter_range=ideal_range,
            notes=dict(feature.notes),
        )

    # -------------------------------------------------------------------------
    # Step 2 & 3: Library Filtering & Machine Compatibility
    # -------------------------------------------------------------------------
    def evaluate_tool_compatibility(
        self,
        tool: Tool,
        feature: MachiningFeature,
        profile: GeometricAccessProfile,
    ) -> ToolCompatibilityCandidate:
        """Evaluate a tool against geometric access bounds and machine envelope."""
        rejection_reasons: list[str] = []
        tool_dia = float(getattr(tool, "diameter", 0.0))
        tool_type = getattr(tool, "type", ToolType.FLAT_ENDMILL)
        tool_flute = float(getattr(tool, "flute_length", 0.0) or 0.0)
        tool_overall = float(getattr(tool, "overall_length", 0.0) or tool_flute)
        tool_r = tool_dia / 2.0

        # 1. Tool Type Compatibility Check
        is_type_ok = tool_type in profile.permissible_tool_types
        # Special spot drilling / plunge milling exceptions
        if profile.purpose == "spot_drilling":
            is_type_ok = (
                is_type_ok
                or getattr(tool, "can_spot_drill", False)
                or (tool_type is ToolType.CHAMFER_MILL and getattr(tool, "can_plunge", False))
                or "spot" in tool.id.lower() or "center" in tool.id.lower()
            )
        elif profile.purpose == "grooving":
            is_type_ok = is_type_ok or "groove" in tool.id.lower()

        if not is_type_ok:
            rejection_reasons.append(
                f"Tool type '{tool_type.value if hasattr(tool_type, 'value') else tool_type}' "
                f"is not permissible for {profile.purpose} on {profile.feature_type.value}"
            )

        # 2. Maximum Diameter Constraint
        # Spot drilling / countersinking can have body >= hole diameter
        is_conical_entry = (
            profile.purpose in ("spot_drilling", "countersinking", "chamfering")
            or tool_type in (ToolType.SPOT_DRILL, ToolType.CENTER_DRILL, ToolType.COUNTERSINK_TOOL, ToolType.CHAMFER_MILL)
        )
        if not is_conical_entry and tool_dia > profile.max_tool_diameter + 1e-4:
            rejection_reasons.append(
                f"Tool diameter Ø{tool_dia:.2f} mm exceeds geometric access limit Ø{profile.max_tool_diameter:.2f} mm"
            )

        # 3. Internal Corner / Fillet Radius Fit (for finishing/rest)
        if profile.requires_corner_fit and profile.max_corner_radius is not None:
            if tool_r > profile.max_corner_radius + 1e-4:
                rejection_reasons.append(
                    f"Tool radius ({tool_r:.2f} mm) exceeds internal corner radius ({profile.max_corner_radius:.2f} mm)"
                )

        # 4. Reach and Flute Length Constraint
        available_reach = tool_flute if (profile.is_internal_cavity) else tool_overall
        if available_reach < profile.required_reach - 1e-4:
            rejection_reasons.append(
                f"Tool reach ({available_reach:.2f} mm) is insufficient for depth ({profile.required_reach:.2f} mm)"
            )

        # 5. Tool Holder Envelope Constraint
        if tool.holder is not None and tool_flute < profile.required_reach - 1e-4:
            holder_dia = float(getattr(tool.holder, "diameter", 40.0))
            if profile.max_holder_diameter is not None and holder_dia >= profile.max_holder_diameter:
                rejection_reasons.append(
                    f"Tool holder (Ø{holder_dia:.2f} mm) collides with cavity opening (Ø{profile.max_holder_diameter:.2f} mm)"
                )

        # 6. Machine Compatibility Check
        machine_score = 100.0
        cutting_params = None
        if len(rejection_reasons) == 0 and self.setup is not None:
            try:
                cutting_params = self.analyzer.analyze(
                    tool=tool,
                    feature=feature,
                    op_purpose=profile.purpose,
                    setup=self.setup,
                )
                # Check RPM limits
                if cutting_params.spindle_rpm > self.machine.spindle_max_rpm:
                    rejection_reasons.append(
                        f"Required spindle RPM ({cutting_params.spindle_rpm:.0f}) exceeds machine max RPM ({self.machine.spindle_max_rpm:.0f})"
                    )
                elif cutting_params.spindle_rpm < self.machine.spindle_min_rpm:
                    machine_score -= 20.0  # Clamped

                # Check machine power limits
                if cutting_params.power_kw > self.machine.max_spindle_power_kw:
                    rejection_reasons.append(
                        f"Estimated cutting power ({cutting_params.power_kw:.2f} kW) exceeds machine max power ({self.machine.max_spindle_power_kw:.2f} kW)"
                    )
            except Exception as ex:
                rejection_reasons.append(f"Cutting data evaluation failed: {ex}")

        # 7. Geometric Fit Score Calculation
        geom_score = 0.0
        if len(rejection_reasons) == 0:
            geom_score = 100.0
            ideal_min, ideal_max = profile.ideal_diameter_range

            # Stiffer tools (larger diameter) receive stiffness bonus: I ~ D^4
            stiffness_bonus = min(25.0 * (tool_dia / max(profile.max_tool_diameter, 1.0)), 25.0)
            geom_score += stiffness_bonus

            # Ideal engagement window reward
            if ideal_min <= tool_dia <= ideal_max:
                geom_score += 15.0
            elif tool_dia < ideal_min:
                geom_score -= 10.0 * (1.0 - tool_dia / max(ideal_min, 1e-3))

            # Exact tool diameter match for holes / taps / reamers
            if profile.feature_type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE, FeatureType.TAPPED_HOLE):
                if feature.diameter is not None and abs(tool_dia - feature.diameter) < 0.05:
                    geom_score += 50.0  # Exact standard size match

            # Dedicated specialized tool reward
            if profile.purpose == "facing" and tool_type is ToolType.FACE_MILL:
                geom_score += 30.0
            elif profile.purpose == "spot_drilling" and tool_type in (ToolType.SPOT_DRILL, ToolType.CENTER_DRILL):
                geom_score += 30.0
            elif profile.purpose == "countersinking" and tool_type is ToolType.COUNTERSINK_TOOL:
                geom_score += 30.0
            elif profile.purpose == "grooving" and tool_type is ToolType.GROOVE_CUTTER:
                geom_score += 30.0
            elif profile.purpose == "finishing" and profile.feature_type in (FeatureType.FILLET, FeatureType.FREEFORM_SURFACE) and tool_type in (ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL):
                geom_score += 30.0

        is_compatible = len(rejection_reasons) == 0
        composite_score = (geom_score * 0.7 + machine_score * 0.3) if is_compatible else 0.0

        return ToolCompatibilityCandidate(
            tool=tool,
            is_compatible=is_compatible,
            rejection_reasons=rejection_reasons,
            geometric_fit_score=geom_score,
            machine_fit_score=machine_score,
            cutting_params=cutting_params,
            composite_score=composite_score,
        )

    # -------------------------------------------------------------------------
    # Step 4 & 5: Tool Selection & Actionable Diagnostics
    # -------------------------------------------------------------------------
    def select_optimal_tool(
        self,
        feature: MachiningFeature,
        purpose: str,
        available_tools: list[Tool],
    ) -> ToolSelectionResult:
        """Execute complete geometry-driven tool selection pipeline for a feature."""
        # 1. Feature -> Required Access Profile
        profile = self.compute_access_profile(feature, purpose)

        # 2. Evaluate all available tools
        evaluations: list[ToolCompatibilityCandidate] = []
        for t in available_tools:
            cand = self.evaluate_tool_compatibility(t, feature, profile)
            evaluations.append(cand)

        compatible_candidates = [c for c in evaluations if c.is_compatible]

        if compatible_candidates:
            # Sort by composite score descending
            compatible_candidates.sort(key=lambda c: -c.composite_score)
            best_candidate = compatible_candidates[0]

            return ToolSelectionResult(
                feature=feature,
                purpose=purpose,
                selected_tool=best_candidate.tool,
                access_profile=profile,
                candidates_evaluated=evaluations,
                success=True,
            )

        # Construct actionable suggestion and diagnostic
        sugg_type = profile.permissible_tool_types[0].value if profile.permissible_tool_types else "flat_endmill"
        sugg_dia = round(min(profile.max_tool_diameter, max(profile.min_recommended_diameter, 0.5)), 2)
        sugg_flute = round(max(profile.required_reach * 1.25, 5.0), 1)

        suggested = {
            "type": sugg_type,
            "diameter": sugg_dia,
            "flute_length": sugg_flute,
            "overall_length": round(max(profile.required_reach * 2.0, 30.0), 1),
            "flutes": 2 if sugg_dia < 3.0 else 4,
        }

        diag = (
            f"No compatible {purpose} tool available for {feature.id} ({feature.type.value}). "
            f"Required access: max diameter <= {profile.max_tool_diameter:.2f} mm, reach >= {profile.required_reach:.2f} mm. "
            f"Recommended tool: {sugg_type} Ø{sugg_dia:.2f} mm (flute length >= {sugg_flute:.2f} mm)."
        )

        return ToolSelectionResult(
            feature=feature,
            purpose=purpose,
            selected_tool=None,
            access_profile=profile,
            candidates_evaluated=evaluations,
            success=False,
            diagnostic_message=diag,
            suggested_tool=suggested,
        )
