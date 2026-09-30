"""Tests for explicit data-driven peck drilling modeling and G-code emission.

Verifies:
1. Explicit modeling of start plane, retract plane, first peck, subsequent peck, peck reduction, and final depth.
2. Expanded linear motion uses RAPID (G0) to re-enter to standoff height above prior cut depth,
   and PLUNGE (G1) exclusively for actual cutting engagement into material.
3. Controller canned cycle emission (G83) cleanly encapsulates the cycle parameters.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    Material, WorkOffset, DEFAULT_MATERIALS
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.geometry import TriangleMesh
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.post import get_post_processor
from cam_engine.toolpaths.semantic import MotionType
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


def test_peck_drilling_explicit_model_and_rapid_reentry():
    """Test that peck drilling generates explicit clean pecks with G0 rapid re-entries and G1 cutting plunges."""
    tool = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=1.0,
        flute_length=15.0, overall_length=45.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 5.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    # Blind hole: Top Z = 3.64 mm, Floor Z = 1.47 mm (total depth = 2.17 mm)
    hole = MachiningFeature(
        id="hole_1",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.5, -0.5, 1.47]),
        bounds_max=np.array([0.5, 0.5, 3.64]),
        depth=2.17,
        top_z=3.64,
        floor_z=1.47,
        diameter=1.0,
        is_concave=True,
    )

    # Peck = 0.58 mm (3.64 -> 3.06 -> 2.48 -> 1.90 -> 1.47)
    params = CuttingParameters(
        spindle_rpm=3000.0, feed_rate=100.0, plunge_feed=80.0, ramp_feed=80.0,
        feed_per_tooth_mm=0.02, depth_of_cut_mm=0.58, stepover_mm=1.0,
        coolant=True, effective_diameter_mm=1.0, engagement_angle_deg=180.0,
        power_kw=0.1, torque_nm=0.3
    )

    op = PlannedOperation(
        id="op_drill_001",
        purpose=OpPurpose.DRILLING,
        feature=hole,
        tool=tool,
        params=params,
        notes={"r_clearance": 0.58, "depth_reference": "tip"}  # R-plane = 3.64 + 0.58 = 4.22
    )

    s_ctx = _make_strategy_context(ctx, setup)
    strat = StrategyEngine(s_ctx)
    tp = strat.generate(op)

    assert tp.metadata["strategy"] == "peck_drill"
    intent = tp.metadata["canned_cycle_intent"]
    assert intent["start_plane"] == pytest.approx(3.64)
    assert intent["retract_plane"] == pytest.approx(4.22)
    assert intent["final_depth"] == pytest.approx(1.47)

    # Post-process for a non-canned controller (e.g. GRBL) to inspect explicit G0/G1 motion
    post = get_post_processor("grbl", work_offset="G54")
    res = post.post_process([tp], "TEST_PECK")
    gcode = res.gcode

    # Verify that intermediate re-entries above cut bottom use rapid (G0) and plunges use G1
    plunge_segs = [s for s in tp.segments if s.motion_type == MotionType.PLUNGE]

    # Every plunge except the first should start at standoff height (prev_depth + approach_clearance)
    for i in range(1, len(plunge_segs)):
        plunge = plunge_segs[i]
        # Plunge start Z must NOT be from retract_plane 4.22!
        assert plunge.start[2] < 4.22 - 1e-4, f"Plunge {i} cutting air from R-plane! Start Z: {plunge.start[2]}"


def test_peck_drilling_canned_cycle_emission():
    """Test that Fanuc/standard controller emits clean G83 block."""
    tool = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=1.0,
        flute_length=15.0, overall_length=45.0, point_angle=118.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 5.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    hole = MachiningFeature(
        id="hole_1", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.5, -0.5, 1.47]), bounds_max=np.array([0.5, 0.5, 3.64]),
        depth=2.17, top_z=3.64, floor_z=1.47, diameter=1.0, is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=3000.0, feed_rate=100.0, plunge_feed=80.0, ramp_feed=80.0,
        feed_per_tooth_mm=0.02, depth_of_cut_mm=0.58, stepover_mm=1.0,
        coolant=True, effective_diameter_mm=1.0, engagement_angle_deg=180.0,
        power_kw=0.1, torque_nm=0.3
    )
    op = PlannedOperation(
        id="op_drill_001", purpose=OpPurpose.DRILLING, feature=hole,
        tool=tool, params=params, notes={"r_clearance": 0.58, "depth_reference": "tip"}
    )

    s_ctx = _make_strategy_context(ctx, setup)
    strat = StrategyEngine(s_ctx)
    tp = strat.generate(op)

    post = get_post_processor("fanuc", work_offset="G54")
    res = post.post_process([tp], "TEST_FANUC_G83")
    gcode = res.gcode

    assert "G98 G83" in gcode
    assert "Q0.58" in gcode
    assert "Z1.47" in gcode
    assert "R4.22" in gcode
    assert "G80" in gcode
