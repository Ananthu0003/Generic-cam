"""Tests for drill tool-tip geometry, depth reference (shoulder vs tip), and breakthrough calculation.

Verifies:
1. The CAM engine explicitly calculates tool tip length h_tip = (D/2) / tan(point_angle/2).
2. For through holes, the drill tip travels past the hole floor (Z_floor - h_tip - breakthrough)
   to ensure full-diameter cylindrical breakthrough.
3. For blind holes with web constraints, depth_reference="tip" ensures the drill point does not
   breach the web.
4. For blind holes requiring full cylindrical depth (depth_reference="shoulder"), the tip travels
   to Z_floor - h_tip.
"""

import math
import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    Material, WorkOffset, DEFAULT_MATERIALS
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.geometry import TriangleMesh
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose
from cam_engine.post import get_post_processor
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext


def _make_strategy_context(ctx: PlanningContext, setup: Setup) -> StrategyContext:
    mesh = TriangleMesh(triangles=np.zeros((1, 3, 3)))
    return StrategyContext(
        context=ctx,
        setup=setup,
        mesh=mesh,
        stock_voxels=None,
        clearance_z=10.0,
        finish_allowance_wall=0.2,
        finish_allowance_floor=0.1
    )


def test_through_hole_breakthrough_accounts_for_drill_tip():
    """Test that through-hole drilling depth adds drill tip length and breakthrough allowance."""
    # Tool: Ø1.0mm, 118 deg point angle -> tip length = 0.5 / tan(59 deg) = 0.3004 mm
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.DRILL, diameter=1.0,
        flute_length=15.0, overall_length=45.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 5.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    # Through hole from Z=5.0 to Z=0.0
    thru_hole = MachiningFeature(
        id="thru_hole_1", type=FeatureType.THROUGH_HOLE, face_indices=[0],
        bounds_min=np.array([-0.5, -0.5, 0.0]), bounds_max=np.array([0.5, 0.5, 5.0]),
        depth=5.0, top_z=5.0, floor_z=0.0, diameter=1.0, is_concave=True,
    )

    planner = OperationPlanner(ctx, setup, [thru_hole])
    plan = planner.plan()
    assert len(plan) == 1
    op = plan[0]

    s_ctx = _make_strategy_context(ctx, setup)
    strat = StrategyEngine(s_ctx)
    tp = strat.generate(op)

    intent = tp.metadata["canned_cycle_intent"]
    expected_tip_len = (1.0 / 2.0) / math.tan(math.radians(118.0 / 2.0))
    assert abs(intent["tip_length"] - expected_tip_len) < 1e-4

    # Final depth must penetrate past 0.0 by tip_length + breakthrough (0.5mm)
    expected_final_z = 0.0 - expected_tip_len - 0.5
    assert abs(intent["final_depth"] - expected_final_z) < 1e-4


def test_blind_hole_with_thin_web_preserves_floor():
    """Test that blind-hole drilling respects the floor limit and does not punch through thin webs."""
    # Tool: Ø0.58mm drill, point angle 118 deg -> tip length ~ 0.174 mm
    tool = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=0.58,
        flute_length=10.0, overall_length=35.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-2.0, -2.0, 0.0]), bounds_max=np.array([2.0, 2.0, 3.22]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["brass_360"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    # Top blind hole: Z=3.22 to Z=1.71 (leaving 0.20mm web above bottom hole at Z=1.51)
    blind_hole = MachiningFeature(
        id="blind_hole_top", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.71]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.51, top_z=3.22, floor_z=1.71, diameter=0.58, is_concave=True,
        notes={"depth_reference": "tip", "web_thickness": 0.20}
    )

    planner = OperationPlanner(ctx, setup, [blind_hole])
    plan = planner.plan()
    assert len(plan) == 1
    op = plan[0]

    s_ctx = _make_strategy_context(ctx, setup)
    strat = StrategyEngine(s_ctx)
    tp = strat.generate(op)

    intent = tp.metadata["canned_cycle_intent"]
    # Programmed tip depth must stop at floor_z (1.71), not breaching the 0.20mm web
    assert intent["final_depth"] == pytest.approx(1.71)
    assert intent["final_depth"] > 1.51  # Never breaches into the opposite hole


def test_blind_hole_shoulder_depth_reference():
    """Test that when depth_reference='shoulder' is explicitly requested, tip extends by tip_length."""
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.DRILL, diameter=2.0,
        flute_length=20.0, overall_length=50.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-10.0, -10.0, 0.0]), bounds_max=np.array([10.0, 10.0, 20.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    blind_hole = MachiningFeature(
        id="blind_shoulder", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.0, -1.0, 5.0]), bounds_max=np.array([1.0, 1.0, 20.0]),
        depth=15.0, top_z=20.0, floor_z=5.0, diameter=2.0, is_concave=True,
        notes={"depth_reference": "shoulder"}
    )

    planner = OperationPlanner(ctx, setup, [blind_hole])
    plan = planner.plan()
    assert len(plan) == 1
    op = plan[0]

    s_ctx = _make_strategy_context(ctx, setup)
    strat = StrategyEngine(s_ctx)
    tp = strat.generate(op)

    intent = tp.metadata["canned_cycle_intent"]
    expected_tip_len = (2.0 / 2.0) / math.tan(math.radians(118.0 / 2.0))
    assert intent["final_depth"] == pytest.approx(5.0 - expected_tip_len)
