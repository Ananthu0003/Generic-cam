"""Feature × Setup × Orientation Accessibility Analysis and Matrix-Driven Setup Planning.

For every feature and every candidate setup/orientation:
    Feature × Setup × Orientation
calculates:
    - accessible? (line-of-sight & approach alignment)
    - tool access? (tool reach, diameter bounds, flute length)
    - holder collision? (tool holder envelope clearance)
    - fixture collision? (vise jaws, toe clamps, fixture plates)
    - machine-axis requirement? (3-axis vertical, horizontal, 4-axis, 5-axis)

Produces an auditable AccessibilityMatrix that drives setup planning from geometry,
eliminating heuristic rules like 'if model looks like X -> 2 setups'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Optional, Sequence

import numpy as np

from .context import (
    Fixture, MachineConfig, MachineType, Setup, Tool, ToolType, WorkOffset,
    get_material,
)
from .coords import CoordSpace, transform_direction, transform_points, _invert
from .errors import CamError, FEATURE_INACCESSIBLE, NO_COMPATIBLE_TOOL, COLLISION_DETECTED
from .features import Accessibility, FeatureType, MachiningFeature
from .geometry import Shape, TriangleMesh
from .demo_models import get_default_machine
from .tool_selection import GeometryDrivenToolSelector


class MachineAxisRequirement(str, Enum):
    THREE_AXIS_VERTICAL = "3axis_vertical"
    THREE_AXIS_HORIZONTAL = "3axis_horizontal"
    FOUR_AXIS_ROTARY = "4axis_rotary"
    FIVE_AXIS_INDEXED = "5axis_indexed"
    FIVE_AXIS_SIMULTANEOUS = "5axis_simultaneous"
    NONE = "none"


@dataclass
class FeatureSetupAccessibilityEntry:
    """Evaluation result for one (Feature, Setup) pair."""

    feature_id: str
    setup_id: str
    work_offset: str                           # e.g. "G54", "G55"
    orientation: np.ndarray                    # Tool approach direction vector in CAD model space
    accessible: bool                           # Overall boolean: machinable in this setup
    optical_access: bool                       # Line-of-sight and approach direction alignment
    tool_access: bool                          # Cutter available with sufficient diameter & reach
    holder_collision: bool                     # Tool holder body collides with part or cavity
    fixture_collision: bool                    # Tool or holder collides with vise/clamps
    machine_axis_requirement: MachineAxisRequirement
    rejection_reasons: list[str] = field(default_factory=list)
    selected_tool_id: Optional[str] = None
    reach_required_mm: float = 0.0
    reach_available_mm: float = 0.0
    clearance_score: float = 100.0             # Score 0..100 (higher = better clearance & orientation)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "setup_id": self.setup_id,
            "work_offset": self.work_offset,
            "orientation": [round(float(x), 4) for x in self.orientation],
            "accessible": self.accessible,
            "optical_access": self.optical_access,
            "tool_access": self.tool_access,
            "holder_collision": self.holder_collision,
            "fixture_collision": self.fixture_collision,
            "machine_axis_requirement": self.machine_axis_requirement.value,
            "rejection_reasons": list(self.rejection_reasons),
            "selected_tool_id": self.selected_tool_id,
            "reach_required_mm": round(self.reach_required_mm, 3),
            "reach_available_mm": round(self.reach_available_mm, 3),
            "clearance_score": round(self.clearance_score, 2),
            "details": self.details,
        }


@dataclass
class AccessibilityMatrix:
    """The complete Feature × Setup × Orientation accessibility evaluation matrix."""

    features: list[MachiningFeature]
    setups: list[Setup]
    entries: dict[tuple[str, str], FeatureSetupAccessibilityEntry] = field(default_factory=dict)

    def is_accessible(self, feature_id: str, setup_id: str) -> bool:
        entry = self.entries.get((feature_id, setup_id))
        return bool(entry and entry.accessible)

    def get_entry(self, feature_id: str, setup_id: str) -> Optional[FeatureSetupAccessibilityEntry]:
        return self.entries.get((feature_id, setup_id))

    def get_accessible_setups(self, feature_id: str) -> list[Setup]:
        """Return all setups capable of machining this feature."""
        return [
            s for s in self.setups
            if self.is_accessible(feature_id, s.id)
        ]

    def get_accessible_features(self, setup_id: str) -> list[MachiningFeature]:
        """Return all features machinable within the specified setup."""
        return [
            f for f in self.features
            if self.is_accessible(f.id, setup_id)
        ]

    def get_coverage_ratio(self) -> float:
        """Fraction of features accessible in at least one setup."""
        if not self.features:
            return 1.0
        covered = sum(1 for f in self.features if len(self.get_accessible_setups(f.id)) > 0)
        return float(covered / len(self.features))

    def uncovered_features(self) -> list[MachiningFeature]:
        """Return features that cannot be machined in any evaluated setup."""
        return [
            f for f in self.features
            if len(self.get_accessible_setups(f.id)) == 0
        ]

    def summary_table(self) -> str:
        """Render a clean text matrix table of Feature × Setup accessibility."""
        lines = []
        setup_headers = [f"{s.id} ({s.WCS})" for s in self.setups]
        hdr = f"{'Feature ID':<24} | " + " | ".join(f"{sh:<16}" for sh in setup_headers)
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for f in self.features:
            row_items = []
            for s in self.setups:
                entry = self.get_entry(f.id, s.id)
                if entry and entry.accessible:
                    status = "ACCESSIBLE"
                elif entry and entry.holder_collision:
                    status = "COLLISION (H)"
                elif entry and entry.fixture_collision:
                    status = "COLLISION (F)"
                elif entry and not entry.tool_access:
                    status = "NO TOOL"
                elif entry and not entry.optical_access:
                    status = "INACCESSIBLE"
                else:
                    status = "INACCESSIBLE"
                row_items.append(f"{status:<16}")
            lines.append(f"{f.id:<24} | " + " | ".join(row_items))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_features": len(self.features),
            "num_setups": len(self.setups),
            "coverage_ratio": round(self.get_coverage_ratio(), 4),
            "uncovered_feature_ids": [f.id for f in self.uncovered_features()],
            "entries": [e.to_dict() for e in self.entries.values()],
            "summary_table": self.summary_table(),
        }


class AccessibilityAnalyzer:
    """Evaluates Feature × Setup × Orientation accessibility rigorously across geometry, tools, and fixtures."""

    def __init__(
        self,
        machine: Optional[MachineConfig] = None,
        tools: Optional[list[Tool]] = None,
        vertical_tol_deg: float = 15.0,
    ):
        self.machine = machine
        self.tools = tools or []
        self.vertical_tol_deg = float(vertical_tol_deg)
        self.cos_v = float(np.cos(np.radians(vertical_tol_deg)))

    def analyze(
        self,
        features: list[MachiningFeature],
        setups: list[Setup],
        shape: Optional[Shape] = None,
        mesh: Optional[TriangleMesh] = None,
    ) -> AccessibilityMatrix:
        """Compute the full Feature × Setup × Orientation accessibility matrix."""
        matrix = AccessibilityMatrix(features=features, setups=setups)

        # Build canonical model mesh if not provided
        canonical_mesh = mesh
        if canonical_mesh is None and shape is not None:
            faces = shape.faces() if hasattr(shape, "faces") and callable(shape.faces) else getattr(shape, "_faces", [])
            canonical_mesh = TriangleMesh.from_faces(faces)

        for feat in features:
            for s in setups:
                entry = self.analyze_feature_in_setup(
                    feature=feat,
                    setup=s,
                    shape=shape,
                    canonical_mesh=canonical_mesh,
                )
                matrix.entries[(feat.id, s.id)] = entry

        return matrix

    def analyze_feature_in_setup(
        self,
        feature: MachiningFeature,
        setup: Setup,
        shape: Optional[Shape] = None,
        canonical_mesh: Optional[TriangleMesh] = None,
    ) -> FeatureSetupAccessibilityEntry:
        """Evaluate accessibility of a single feature in a specific setup."""
        rejection_reasons: list[str] = []
        details: dict[str, Any] = {}

        # 1. Approach direction & optical access
        # Tool axis in setup coordinates is canonically +Z [0, 0, 1]
        # In canonical CAD space, the tool vector pointing toward the part is setup.canonical_orientation
        u_tool_approach = setup.canonical_orientation
        norm_u = float(np.linalg.norm(u_tool_approach))
        if norm_u > 1e-6:
            u_tool_approach /= norm_u
        else:
            u_tool_approach = np.array([0.0, 0.0, 1.0])

        # Feature approach direction
        feat_dir = feature.axis if feature.axis is not None else (
            feature.orientation if feature.orientation is not None else getattr(feature, "machining_direction", None)
        )
        if feat_dir is None:
            feat_dir = np.array([0.0, 0.0, 1.0])

        feat_dir = np.asarray(feat_dir, dtype=float)
        norm_f = float(np.linalg.norm(feat_dir))
        if norm_f > 1e-6:
            feat_dir /= norm_f
        else:
            feat_dir = np.array([0.0, 0.0, 1.0])

        # Alignment dot product
        dot_align = float(np.dot(u_tool_approach, feat_dir))
        details["dot_alignment"] = round(dot_align, 4)

        # Axial cylindrical features (holes, bores) require near-perfect coaxial alignment for standard 3-axis
        axial_types = {
            FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
            FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
            FeatureType.COUNTERBORE, FeatureType.COUNTERSINK, FeatureType.TAPPED_HOLE,
            FeatureType.THREADED_HOLE,
        }

        optical_access = True
        machine_req = MachineAxisRequirement.THREE_AXIS_VERTICAL

        if feature.type in axial_types:
            if dot_align < 0.90:
                optical_access = False
                rejection_reasons.append(
                    f"Feature axis {feat_dir.tolist()} does not align with setup approach vector {u_tool_approach.tolist()} (dot={dot_align:.3f} < 0.90)"
                )
                machine_req = MachineAxisRequirement.FIVE_AXIS_INDEXED
        else:
            # Planar / pocket / contour / chamfer / fillet / step features: tool approach must align with surface normal / approach
            if feature.type in (
                FeatureType.FACING_REGION, FeatureType.POCKET, FeatureType.OPEN_POCKET,
                FeatureType.SLOT, FeatureType.CHAMFER, FeatureType.FILLET, FeatureType.STEP,
            ):
                if dot_align < 0.20:
                    optical_access = False
                    rejection_reasons.append(
                        f"Machining direction {feat_dir.tolist()} misaligned with setup approach {u_tool_approach.tolist()} (dot={dot_align:.3f})"
                    )
                    machine_req = MachineAxisRequirement.FOUR_AXIS_ROTARY if abs(dot_align) < 0.2 else MachineAxisRequirement.FIVE_AXIS_INDEXED


        # Ray casting occlusion check in setup space
        if canonical_mesh is not None and len(canonical_mesh.triangles) > 0 and optical_access:
            # Transform feature test point into setup space
            feat_center = feature.center if feature.center is not None else np.array([
                0.5 * (feature.bounds_min[0] + feature.bounds_max[0]),
                0.5 * (feature.bounds_min[1] + feature.bounds_max[1]),
                feature.top_z,
            ])
            # Ray cast upward from feature center in canonical CAD space along tool retract direction
            ray_dir = u_tool_approach  # direction out of part toward spindle
            hit = canonical_mesh.ray_first_hit(feat_center + ray_dir * 1e-2, ray_dir, max_dist=1e4)
            if hit is not None and hit > 0.05:
                optical_access = False
                rejection_reasons.append(f"Feature occluded by part geometry along setup approach vector (hit distance {hit:.2f} mm)")

        # 2. Tool Access & Reach
        feature_depth = float(feature.depth if feature.depth is not None and feature.depth > 0 else (
            feature.top_z - feature.floor_z if (feature.top_z is not None and feature.floor_z is not None) else 0.0
        ))
        reach_required = feature_depth

        tool_access = True
        selected_tool: Optional[Tool] = None
        max_tool_reach = 0.0

        if self.tools:
            # Evaluate tool feasibility using GeometryDrivenToolSelector
            purpose_str = "drilling" if feature.type in axial_types else (
                "facing" if feature.type == FeatureType.FACING_REGION else "roughing"
            )
            selector = GeometryDrivenToolSelector(
                machine=self.machine or get_default_machine(),
                material=getattr(self.machine, "material", None) or get_material("aluminum_6061"),
                setup=setup,
            )
            res = selector.select_optimal_tool(feature, purpose_str, self.tools)
            if res.success and res.selected_tool is not None:
                selected_tool = res.selected_tool
                max_tool_reach = float(selected_tool.flute_length or selected_tool.overall_length or 50.0)
            else:
                tool_access = False
                if res.candidates_evaluated:
                    reasons = [r for c in res.candidates_evaluated for r in c.rejection_reasons]
                    rejection_reasons.extend(reasons[:2])
                else:
                    rejection_reasons.append(f"No compatible tool found for feature '{feature.id}'")

        # 3. Tool Holder Collision Check
        holder_collision = False
        if selected_tool and selected_tool.holder is not None and feature.is_concave:
            holder_dia = float(getattr(selected_tool.holder, "diameter", 40.0))
            flute_len = float(selected_tool.flute_length or 0.0)
            if flute_len < reach_required - 1e-4:
                opening = float(feature.diameter if feature.diameter is not None else (
                    min(feature.xy_extent) if len(feature.xy_extent) and min(feature.xy_extent) > 0 else 0.0
                ))
                if opening > 1e-4 and holder_dia > opening:
                    holder_collision = True
                    rejection_reasons.append(
                        f"Holder (Ø{holder_dia:.1f} mm) collides with cavity opening (Ø{opening:.1f} mm) at depth {reach_required:.1f} mm"
                    )

        # 4. Fixture Collision Check
        fixture_collision = False
        fixtures = getattr(setup, "fixtures", [])
        if fixtures and optical_access:
            # Transform feature bounding box into setup space
            feat_in_setup = setup.transform_feature(feature)
            bmin_s = feat_in_setup.bounds_min
            bmax_s = feat_in_setup.bounds_max

            for fix in fixtures:
                fix_min = getattr(fix, "bounds_min", None)
                fix_max = getattr(fix, "bounds_max", None)
                if fix_min is not None and fix_max is not None:
                    # Check 3D AABB overlap with fixture
                    overlap_x = (bmin_s[0] <= fix_max[0] and bmax_s[0] >= fix_min[0])
                    overlap_y = (bmin_s[1] <= fix_max[1] and bmax_s[1] >= fix_min[1])
                    overlap_z = (bmin_s[2] <= fix_max[2] and bmax_s[2] >= fix_min[2])
                    if overlap_x and overlap_y and overlap_z:
                        fixture_collision = True
                        rejection_reasons.append(
                            f"Toolpath volume overlaps clamping fixture '{getattr(fix, 'name', getattr(fix, 'id', 'fixture'))}'"
                        )
                        break

        # Clearance score derivation
        clearance_score = 100.0
        if not optical_access:
            clearance_score -= 60.0
        if not tool_access:
            clearance_score -= 30.0
        if holder_collision:
            clearance_score -= 40.0
        if fixture_collision:
            clearance_score -= 50.0
        clearance_score = max(0.0, clearance_score)

        is_accessible = bool(optical_access and tool_access and not holder_collision and not fixture_collision)

        return FeatureSetupAccessibilityEntry(
            feature_id=feature.id,
            setup_id=setup.id,
            work_offset=setup.WCS,
            orientation=u_tool_approach,
            accessible=is_accessible,
            optical_access=optical_access,
            tool_access=tool_access,
            holder_collision=holder_collision,
            fixture_collision=fixture_collision,
            machine_axis_requirement=machine_req,
            rejection_reasons=rejection_reasons,
            selected_tool_id=selected_tool.id if selected_tool else None,
            reach_required_mm=reach_required,
            reach_available_mm=max_tool_reach,
            clearance_score=clearance_score,
            details=details,
        )


class SetupPlanner:
    """Plans the minimal, optimal sequence of setups derived directly from the Accessibility Matrix."""

    def __init__(self, analyzer: Optional[AccessibilityAnalyzer] = None):
        self.analyzer = analyzer or AccessibilityAnalyzer()

    def plan_setups_from_matrix(
        self,
        features: list[MachiningFeature],
        candidate_setups: list[Setup],
        matrix: Optional[AccessibilityMatrix] = None,
        shape: Optional[Shape] = None,
        mesh: Optional[TriangleMesh] = None,
    ) -> tuple[list[Setup], dict[str, list[MachiningFeature]]]:
        """Compute the optimal subset of setups and partition features to their primary setup.
        
        Returns:
            (selected_setups, setup_to_features_map)
        """
        if matrix is None:
            matrix = self.analyzer.analyze(features, candidate_setups, shape=shape, mesh=mesh)

        remaining_features = set(f.id for f in features)
        selected_setups: list[Setup] = []
        setup_features_map: dict[str, list[MachiningFeature]] = {}
        feat_by_id = {f.id: f for f in features}

        # Greedy Set Cover to select minimal setups covering all accessible features
        while remaining_features:
            best_setup = None
            best_cover: set[str] = set()

            for s in candidate_setups:
                if s in selected_setups:
                    continue
                accessible_here = set(
                    fid for fid in remaining_features
                    if matrix.is_accessible(fid, s.id)
                )
                if len(accessible_here) > len(best_cover):
                    best_cover = accessible_here
                    best_setup = s

            if not best_setup or not best_cover:
                # No remaining features can be covered by candidate setups
                break

            selected_setups.append(best_setup)
            assigned_feats = [feat_by_id[fid] for fid in best_cover]
            setup_features_map[best_setup.id] = assigned_feats
            remaining_features -= best_cover

        # Assign any remaining setups if none were selected (e.g. all inaccessible)
        if not selected_setups and candidate_setups:
            selected_setups = [candidate_setups[0]]
            setup_features_map[candidate_setups[0].id] = []

        return selected_setups, setup_features_map
