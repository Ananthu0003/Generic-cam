"""Tests for generic minimum-wall and web validation across features and tool geometry.

Verifies:
1. Automatic detection of thin remaining webs (e.g. 0.20 mm) between opposing/collinear blind holes.
2. Emits structured THIN_WALL_WARNING detailing nominal web, tool tip lengths, and tolerances.
3. If shoulder depth reference would violate the thin web, automatically clamps to tip depth
   and emits CRITICAL_WEB_THICKNESS.
4. Validates bottom stock web and adjacent XY wall thickness.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    Material, WorkOffset, DEFAULT_MATERIALS
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose


def test_opposing_blind_holes_thin_web_validation():
    """Test that a 0.20mm web between two opposing blind holes triggers thin web validation and warnings."""
    tool_drill = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=0.58,
        flute_length=10.0, overall_length=35.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-2.0, -2.0, 0.0]), bounds_max=np.array([2.0, 2.0, 3.22]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["brass_360"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_drill], material=mat)

    # Hole 1 from top: Floor Z = 1.71 mm (depth 1.51 mm)
    hole_top = MachiningFeature(
        id="hole_top", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.71]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.51, top_z=3.22, floor_z=1.71, diameter=0.58, is_concave=True,
    )

    # Hole 2 from bottom: Floor Z = 1.51 mm (depth 1.51 mm)
    # Distance between floors = 1.71 - 1.51 = 0.20 mm
    hole_bottom = MachiningFeature(
        id="hole_bottom", type=FeatureType.BLIND_HOLE, face_indices=[1],
        bounds_min=np.array([-0.29, -0.29, 0.0]), bounds_max=np.array([0.29, 0.29, 1.51]),
        depth=1.51, top_z=0.0, floor_z=1.51, diameter=0.58, is_concave=True,
    )

    planner = OperationPlanner(ctx, setup, [hole_top, hole_bottom])
    plan = planner.plan()

    # Must emit a THIN_WALL_WARNING for the 0.20 mm web
    thin_warnings = [w for w in planner.warnings if w.get("code") == "THIN_WALL_WARNING"]
    assert len(thin_warnings) > 0

    web_warn = [w for w in thin_warnings if "hole_top" in str(w) and "hole_bottom" in str(w)][0]
    assert "0.20" in web_warn["message"] or web_warn["details"]["nominal_web_mm"] == pytest.approx(0.20, abs=0.01)


def test_shoulder_depth_clamped_on_critical_web():
    """Test that attempting shoulder depth on a 0.20mm web is automatically overridden to tip depth."""
    tool_drill = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=0.58,
        flute_length=10.0, overall_length=35.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-2.0, -2.0, 0.0]), bounds_max=np.array([2.0, 2.0, 3.22]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["brass_360"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_drill], material=mat)

    # Explicitly requesting shoulder depth on a hole with 0.20mm floor web
    hole_top = MachiningFeature(
        id="hole_top", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.71]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.51, top_z=3.22, floor_z=1.71, diameter=0.58, is_concave=True,
        notes={"depth_reference": "shoulder"}
    )
    hole_bottom = MachiningFeature(
        id="hole_bottom", type=FeatureType.BLIND_HOLE, face_indices=[1],
        bounds_min=np.array([-0.29, -0.29, 0.0]), bounds_max=np.array([0.29, 0.29, 1.51]),
        depth=1.51, top_z=0.0, floor_z=1.51, diameter=0.58, is_concave=True,
    )

    planner = OperationPlanner(ctx, setup, [hole_top, hole_bottom])
    plan = planner.plan()

    # The planner must catch the critical web and override depth_reference to 'tip'
    drill_op = [op for op in plan if op.feature.id == "hole_top"][0]
    assert drill_op.notes["depth_reference"] == "tip"

    crit_warnings = [w for w in planner.warnings if w.get("code") == "CRITICAL_WEB_THICKNESS"]
    assert len(crit_warnings) > 0
    assert any("hole_top" in w["feature_id"] for w in crit_warnings)


def test_adjacent_holes_thin_wall_validation():
    """Test that two adjacent holes with a 0.30mm wall between them trigger thin wall validation."""
    tool_drill = Tool(
        id="T01", tool_number=1, type=ToolType.DRILL, diameter=2.0,
        flute_length=15.0, overall_length=45.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-10.0, -10.0, 0.0]), bounds_max=np.array([10.0, 10.0, 20.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_drill], material=mat)

    # Hole A at X=0, Hole B at X=2.3 (radius=1.0 each -> wall = 2.3 - 2.0 = 0.30 mm)
    hole_a = MachiningFeature(
        id="hole_a", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.0, -1.0, 5.0]), bounds_max=np.array([1.0, 1.0, 20.0]),
        depth=15.0, top_z=20.0, floor_z=5.0, diameter=2.0, is_concave=True,
    )
    hole_b = MachiningFeature(
        id="hole_b", type=FeatureType.BLIND_HOLE, face_indices=[1],
        bounds_min=np.array([1.3, -1.0, 5.0]), bounds_max=np.array([3.3, 1.0, 20.0]),
        depth=15.0, top_z=20.0, floor_z=5.0, diameter=2.0, is_concave=True,
    )

    planner = OperationPlanner(ctx, setup, [hole_a, hole_b])
    plan = planner.plan()

    thin_warnings = [w for w in planner.warnings if w.get("code") == "THIN_WALL_WARNING"]
    assert len(thin_warnings) > 0
    wall_warn = [w for w in thin_warnings if "hole_a" in str(w) and "hole_b" in str(w)][0]
    assert wall_warn["details"]["nominal_wall_mm"] == pytest.approx(0.30, abs=0.01)
