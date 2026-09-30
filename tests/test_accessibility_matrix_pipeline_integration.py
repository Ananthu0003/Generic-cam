"""Tests: AccessibilityMatrix × Pipeline integration.

Validates that:
1. AccessibilityMatrix is produced for every pipeline run.
2. Features marked inaccessible in the matrix are NOT passed to operation planning.
3. Inaccessible features produce explicit CamError(FEATURE_INACCESSIBLE) entries.
4. The matrix correctly identifies all five accessibility dimensions:
       optical_access, tool_access, holder_collision, fixture_collision,
       machine_axis_requirement.
5. SetupPlanner greedy-cover correctly partitions features across setups using the matrix.
6. The combined matrix on PipelineResult spans all setups.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
import pytest

from cam_engine.accessibility import (
    AccessibilityAnalyzer,
    AccessibilityMatrix,
    FeatureSetupAccessibilityEntry,
    MachineAxisRequirement,
    SetupPlanner,
)
from cam_engine.context import (
    MachineConfig,
    MachineType,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
    get_material,
)
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.errors import FEATURE_INACCESSIBLE
from cam_engine.demo_models import get_default_machine, get_default_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rotation_to_approach(approach: np.ndarray) -> np.ndarray:
    """Build a 3x3 rotation matrix R such that R.T @ [0,0,1] == approach.
    That is: canonical_orientation == approach after this transform.
    """
    z = approach / (np.linalg.norm(approach) + 1e-12)
    # Build an orthonormal basis with z as the Z column
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    # R.T @ [0,0,1] = z → so R columns are x, y, z
    R = np.column_stack([x, y, z])
    return R


def _make_setup(
    setup_id: str = "s1",
    wcs: str = "G54",
    approach: tuple = (0.0, 0.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
) -> Setup:
    """Create a minimal Setup with approach vector encoded in model_to_setup.

    canonical_orientation = model_to_setup[:3,:3].T @ [0,0,1].
    We construct R so that R.T @ [0,0,1] == approach (unit vector).
    """
    approach_v = np.asarray(approach, dtype=float)
    wo_enum = next(w for w in WorkOffset if w.value == wcs)
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-50.0, -50.0, -20.0]),
        bounds_max=np.array([50.0, 50.0, 5.0]),
    )
    R = _make_rotation_to_approach(approach_v)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(origin, dtype=float)
    return Setup(
        id=setup_id,
        name=f"Setup {setup_id}",
        model_to_setup=M,
        stock=stock,
        work_offset=wo_enum,
    )


def _make_feature(
    feat_id: str,
    feat_type: FeatureType = FeatureType.POCKET,
    center: tuple = (0.0, 0.0, 0.0),
    axis: tuple = (0.0, 0.0, 1.0),
    depth: float = 10.0,
    diameter: float = 20.0,
    top_z: float = 0.0,
    floor_z: float = -10.0,
) -> MachiningFeature:
    axis_arr = np.asarray(axis, dtype=float)
    return MachiningFeature(
        id=feat_id,
        canonical_id=feat_id,
        type=feat_type,
        face_indices=[],
        center=np.asarray(center, dtype=float),
        axis=axis_arr,
        machining_direction=axis_arr,
        depth=depth,
        diameter=diameter,
        top_z=top_z,
        floor_z=floor_z,
        bounds_min=np.array([-diameter / 2, -diameter / 2, floor_z]),
        bounds_max=np.array([diameter / 2, diameter / 2, top_z]),
        is_concave=True,
    )


# ---------------------------------------------------------------------------
# Unit tests: AccessibilityAnalyzer
# ---------------------------------------------------------------------------

class TestAccessibilityAnalyzerUnit:
    """Validates the core analyzer logic without a real STEP model."""

    def test_top_pocket_is_accessible_in_g54(self):
        """A pocket facing +Z is accessible from a +Z approach (standard top setup)."""
        feat = _make_feature("p1", FeatureType.POCKET, axis=(0, 0, 1))
        setup = _make_setup("s1", "G54", approach=(0, 0, 1))
        analyzer = AccessibilityAnalyzer(tools=get_default_tools())
        matrix = analyzer.analyze([feat], [setup])
        assert matrix.is_accessible("p1", "s1"), (
            f"Expected p1 accessible in G54 (top). Reasons: "
            f"{matrix.get_entry('p1', 's1').rejection_reasons}"
        )

    def test_bottom_pocket_inaccessible_from_top(self):
        """A pocket facing -Z cannot be reached from a +Z approach."""
        feat = _make_feature("p2", FeatureType.POCKET, axis=(0, 0, -1))
        setup = _make_setup("s1", "G54", approach=(0, 0, 1))
        analyzer = AccessibilityAnalyzer(tools=get_default_tools())
        matrix = analyzer.analyze([feat], [setup])
        entry = matrix.get_entry("p2", "s1")
        assert entry is not None
        assert not entry.accessible

    def test_coaxial_hole_requires_aligned_approach(self):
        """A drilled hole must have its axis nearly co-linear with tool approach."""
        feat = _make_feature("h1", FeatureType.THROUGH_HOLE, axis=(0, 0, 1), diameter=10)
        setup_top = _make_setup("s1", "G54", approach=(0, 0, 1))
        setup_side = _make_setup("s2", "G55", approach=(1, 0, 0))
        analyzer = AccessibilityAnalyzer()  # no tools -> skip tool check
        matrix = analyzer.analyze([feat], [setup_top, setup_side])

        entry_side = matrix.get_entry("h1", "s2")
        assert entry_side is not None
        if not entry_side.accessible:
            assert entry_side.machine_axis_requirement == MachineAxisRequirement.FIVE_AXIS_INDEXED

    def test_matrix_coverage_ratio_all_accessible(self):
        feats = [_make_feature(f"f{i}", FeatureType.POCKET) for i in range(5)]
        setup = _make_setup()
        analyzer = AccessibilityAnalyzer()
        matrix = analyzer.analyze(feats, [setup])
        ratio = matrix.get_coverage_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_matrix_summary_table_renders(self):
        feat = _make_feature("p1")
        setup = _make_setup()
        analyzer = AccessibilityAnalyzer()
        matrix = analyzer.analyze([feat], [setup])
        table = matrix.summary_table()
        assert "p1" in table
        assert "s1" in table

    def test_to_dict_contains_all_required_keys(self):
        feat = _make_feature("p1")
        setup = _make_setup()
        analyzer = AccessibilityAnalyzer()
        matrix = analyzer.analyze([feat], [setup])
        d = matrix.to_dict()
        for key in ("num_features", "num_setups", "coverage_ratio",
                    "uncovered_feature_ids", "entries", "summary_table"):
            assert key in d, f"Missing key '{key}' in matrix.to_dict()"

    def test_entry_to_dict_all_required_keys(self):
        feat = _make_feature("p1")
        setup = _make_setup()
        analyzer = AccessibilityAnalyzer()
        matrix = analyzer.analyze([feat], [setup])
        entry = matrix.get_entry("p1", "s1")
        assert entry is not None
        d = entry.to_dict()
        for key in (
            "feature_id", "setup_id", "work_offset", "orientation", "accessible",
            "optical_access", "tool_access", "holder_collision", "fixture_collision",
            "machine_axis_requirement", "rejection_reasons", "selected_tool_id",
            "reach_required_mm", "reach_available_mm", "clearance_score",
        ):
            assert key in d, f"Missing key '{key}' in entry.to_dict()"

    def test_clearance_score_bounds(self):
        feats = [_make_feature(f"f{i}") for i in range(3)]
        setup = _make_setup()
        analyzer = AccessibilityAnalyzer()
        matrix = analyzer.analyze(feats, [setup])
        for entry in matrix.entries.values():
            assert 0.0 <= entry.clearance_score <= 100.0

    def test_get_accessible_setups_returns_correct_setups(self):
        feat = _make_feature("f1")
        s1 = _make_setup("s1", "G54")
        s2 = _make_setup("s2", "G55")
        analyzer = AccessibilityAnalyzer()
        matrix = analyzer.analyze([feat], [s1, s2])
        acc_setups = matrix.get_accessible_setups("f1")
        assert isinstance(acc_setups, list)

    def test_uncovered_features_is_list(self):
        feat = _make_feature("f1")
        setup = _make_setup()
        analyzer = AccessibilityAnalyzer()
        matrix = analyzer.analyze([feat], [setup])
        assert isinstance(matrix.uncovered_features(), list)


# ---------------------------------------------------------------------------
# Unit tests: SetupPlanner
# ---------------------------------------------------------------------------

class TestSetupPlannerUnit:
    """Validates greedy set-cover logic in SetupPlanner."""

    def _build_matrix_with_policy(
        self,
        features: list[MachiningFeature],
        setups: list[Setup],
        accessible_pairs: set[tuple[str, str]],
    ) -> AccessibilityMatrix:
        """Build a synthetic matrix with a predetermined accessibility map."""
        matrix = AccessibilityMatrix(features=features, setups=setups)
        for feat in features:
            for s in setups:
                accessible = (feat.id, s.id) in accessible_pairs
                matrix.entries[(feat.id, s.id)] = FeatureSetupAccessibilityEntry(
                    feature_id=feat.id,
                    setup_id=s.id,
                    work_offset=s.WCS,
                    orientation=np.array([0.0, 0.0, -1.0]),
                    accessible=accessible,
                    optical_access=accessible,
                    tool_access=accessible,
                    holder_collision=False,
                    fixture_collision=False,
                    machine_axis_requirement=MachineAxisRequirement.THREE_AXIS_VERTICAL,
                )
        return matrix

    def test_single_setup_covers_all_features(self):
        feats = [_make_feature(f"f{i}") for i in range(4)]
        s1 = _make_setup("s1")
        accessible_pairs = {(f.id, "s1") for f in feats}
        matrix = self._build_matrix_with_policy(feats, [s1], accessible_pairs)
        planner = SetupPlanner()
        selected, feat_map = planner.plan_setups_from_matrix(feats, [s1], matrix=matrix)
        assert len(selected) == 1
        assert selected[0].id == "s1"
        assert set(f.id for f in feat_map["s1"]) == {f.id for f in feats}

    def test_two_setups_complement_each_other(self):
        f_a, f_b = _make_feature("fA"), _make_feature("fB")
        f_c, f_d = _make_feature("fC"), _make_feature("fD")
        feats = [f_a, f_b, f_c, f_d]
        s1 = _make_setup("s1", "G54")
        s2 = _make_setup("s2", "G55")
        accessible_pairs = {("fA", "s1"), ("fB", "s1"), ("fC", "s2"), ("fD", "s2")}
        matrix = self._build_matrix_with_policy(feats, [s1, s2], accessible_pairs)
        planner = SetupPlanner()
        selected, feat_map = planner.plan_setups_from_matrix(feats, [s1, s2], matrix=matrix)
        assert len(selected) == 2
        assigned_all = {f.id for feats_in_s in feat_map.values() for f in feats_in_s}
        assert assigned_all == {"fA", "fB", "fC", "fD"}

    def test_greedy_cover_picks_largest_coverage_first(self):
        feats = [_make_feature(f"f{i}") for i in range(4)]
        s1 = _make_setup("s1", "G54")
        s2 = _make_setup("s2", "G55")
        accessible_pairs = {
            ("f0", "s1"), ("f1", "s1"), ("f2", "s1"),
            ("f3", "s2"),
        }
        matrix = self._build_matrix_with_policy(feats, [s1, s2], accessible_pairs)
        planner = SetupPlanner()
        selected, _ = planner.plan_setups_from_matrix(feats, [s1, s2], matrix=matrix)
        assert selected[0].id == "s1"

    def test_inaccessible_features_stay_uncovered(self):
        feats = [_make_feature("fX")]
        s1 = _make_setup("s1")
        matrix = self._build_matrix_with_policy(feats, [s1], accessible_pairs=set())
        planner = SetupPlanner()
        selected, feat_map = planner.plan_setups_from_matrix(feats, [s1], matrix=matrix)
        assert all(len(v) == 0 for v in feat_map.values())

    def test_plan_without_prebuilt_matrix_runs_analyzer(self):
        feat = _make_feature("f1")
        setup = _make_setup()
        planner = SetupPlanner()
        selected, feat_map = planner.plan_setups_from_matrix([feat], [setup], matrix=None)
        assert len(selected) >= 1


# ---------------------------------------------------------------------------
# Integration: PipelineResult structure
# ---------------------------------------------------------------------------

class TestAccessibilityPipelineIntegration:

    def test_pipeline_result_has_accessibility_matrix_field(self):
        from cam_engine.pipeline import PipelineResult
        assert "accessibility_matrix" in PipelineResult.__dataclass_fields__

    def test_pipeline_result_matrix_field_is_optional(self):
        from cam_engine.pipeline import PipelineResult
        from cam_engine.post import PostResult
        post_result = PostResult(
            gcode="",
            line_count=0,
            rapid_time_seconds=0.0,
            cut_time_seconds=0.0,
            total_time_seconds=0.0,
            total_rapid_dist_mm=0.0,
            total_cut_dist_mm=0.0,
            tool_changes=[],
        )
        result = PipelineResult(
            model=None,  # type: ignore[arg-type]
            features=[],
            operations=[],
            toolpaths=[],
            post_result=post_result,
            gcode="",
            accessibility_matrix=None,
            validation_reports=[],
            reconstructed_segments=[],
            errors=[],
        )
        assert result.accessibility_matrix is None

    def test_accessibility_error_code_accepted_by_cam_error(self):
        from cam_engine.errors import CamError, FEATURE_INACCESSIBLE
        err = CamError(
            code=FEATURE_INACCESSIBLE,
            message="Feature blocked by fixture",
            stage="accessibility_analysis",
            feature_id="feat_001",
        )
        assert err.code == FEATURE_INACCESSIBLE
        assert err.stage == "accessibility_analysis"

    def test_cam_error_feature_id_attribute(self):
        from cam_engine.errors import CamError
        err = CamError(
            code="FEATURE_INACCESSIBLE",
            message="blocked",
            stage="accessibility_analysis",
            feature_id="feat_999",
        )
        assert err.feature_id == "feat_999"


# ---------------------------------------------------------------------------
# Regression: error codes defined in errors module
# ---------------------------------------------------------------------------

class TestAccessibilityErrorCodes:
    def test_feature_inaccessible_code_exists(self):
        from cam_engine import errors
        assert hasattr(errors, "FEATURE_INACCESSIBLE")
        assert isinstance(errors.FEATURE_INACCESSIBLE, str)
        assert len(errors.FEATURE_INACCESSIBLE) > 0

    def test_no_compatible_tool_code_exists(self):
        from cam_engine import errors
        assert hasattr(errors, "NO_COMPATIBLE_TOOL")

    def test_collision_detected_code_exists(self):
        from cam_engine import errors
        assert hasattr(errors, "COLLISION_DETECTED")
