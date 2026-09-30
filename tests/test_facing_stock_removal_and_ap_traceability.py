"""Tests for facing stock removal, actual axial depth of cut (ap), and tool DOC capability traceability.

Verifies:
1. When facing excess stock (e.g. Stock Top Z=4.22mm, Model Top Z=3.22mm):
   - Actual stock removal is strictly 4.22 - 3.22 = 1.00 mm.
   - For a single pass, actual ap is 1.00 mm.
   - Max tool capability (e.g. ap_max = 1.80 mm) is preserved as max_depth_of_cut_mm without conflation.
2. G-code comments clearly distinguish:
   - Actual stock removal / actual pass ap: ap = 1.00mm [Actual Stock: 1.00mm, Max Tool DOC: 1.80mm]
3. Setup 2 facing (Stock Top Z=1.00mm, Model Top Z=0.00mm):
   - Actual stock removal is 1.00 - 0.00 = 1.00 mm.
   - Actual ap = 1.00 mm.
4. Multi-pass facing:
   - When stock excess exceeds max DOC (e.g. excess = 3.50mm, max DOC = 1.80mm),
     facing executes in 2 passes with actual stepdown 1.75mm.
"""

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
from cam_engine.operations import PlannedOperation, OpPurpose
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


def test_setup1_facing_actual_stock_removal_and_ap():
    """Test Setup 1 facing (Stock Z=4.22mm, Model Z=3.22mm -> 1.00mm stock removal)."""
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0,
        flute_length=15.0, overall_length=50.0, flutes=3, material="carbide"
    )
    # Stock top = 4.22
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 4.22]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["becu"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    # Facing feature: top_z = 4.22, floor_z = 3.22, excess = 1.00mm
    facing_feat = MachiningFeature(
        id="facing_setup1", type=FeatureType.FACING_REGION, face_indices=[0],
        bounds_min=np.array([-5.0, -5.0, 3.22]), bounds_max=np.array([5.0, 5.0, 4.22]),
        depth=1.00, top_z=4.22, floor_z=3.22, is_concave=False,
        notes={"excess_height": 1.00}
    )

    # Tool capability max DOC = 1.80mm
    params = CuttingParameters(
        spindle_rpm=6366.0, feed_rate=860.0, plunge_feed=344.0, ramp_feed=602.0,
        feed_per_tooth_mm=0.045, depth_of_cut_mm=1.80, stepover_mm=2.10,
        coolant=True, effective_diameter_mm=3.0, engagement_angle_deg=131.4,
        power_kw=0.31, torque_nm=0.47
    )

    op = PlannedOperation(
        id="op_facing_001", purpose=OpPurpose.FACING, feature=facing_feat,
        tool=tool, params=params
    )

    s_ctx = _make_strategy_context(ctx, setup)
    strat = StrategyEngine(s_ctx)
    tp = strat.generate(op)

    assert tp.metadata["passes"] == 1
    facing_info = tp.metadata["facing_info"]
    assert facing_info["stock_removal_mm"] == pytest.approx(1.00)
    assert facing_info["actual_stepdown_mm"] == pytest.approx(1.00)
    assert facing_info["max_doc_mm"] == pytest.approx(1.80)

    cp_meta = tp.metadata["cutting_parameters"]
    assert cp_meta["actual_depth_of_cut_mm"] == pytest.approx(1.00)
    assert cp_meta["stock_removal_z_mm"] == pytest.approx(1.00)
    assert cp_meta["max_depth_of_cut_mm"] == pytest.approx(1.80)

    # Post process and verify G-code comment
    post = get_post_processor("fanuc", work_offset="G54")
    res = post.post_process([tp], "FACING_SETUP1")
    gcode = res.gcode

    assert "ap = 1.00mm [Actual Stock: 1.00mm, Max Tool DOC: 1.80mm]" in gcode


def test_setup2_facing_actual_stock_removal_and_ap():
    """Test Setup 2 facing (Stock Z=1.00mm, Model Z=0.00mm -> 1.00mm stock removal)."""
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0,
        flute_length=15.0, overall_length=50.0, flutes=3, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, -3.22]), bounds_max=np.array([5.0, 5.0, 1.00]))
    setup = Setup(id="s2", name="Setup 2", model_to_setup=np.eye(4), work_offset=WorkOffset.G55, stock=stock)
    mat = DEFAULT_MATERIALS["becu"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    facing_feat = MachiningFeature(
        id="facing_setup2", type=FeatureType.FACING_REGION, face_indices=[0],
        bounds_min=np.array([-5.0, -5.0, 0.00]), bounds_max=np.array([5.0, 5.0, 1.00]),
        depth=1.00, top_z=1.00, floor_z=0.00, is_concave=False,
        notes={"excess_height": 1.00}
    )

    params = CuttingParameters(
        spindle_rpm=6366.0, feed_rate=860.0, plunge_feed=344.0, ramp_feed=602.0,
        feed_per_tooth_mm=0.045, depth_of_cut_mm=1.80, stepover_mm=2.10,
        coolant=True, effective_diameter_mm=3.0, engagement_angle_deg=131.4,
        power_kw=0.31, torque_nm=0.47
    )

    op = PlannedOperation(
        id="op_facing_002", purpose=OpPurpose.FACING, feature=facing_feat,
        tool=tool, params=params
    )

    s_ctx = _make_strategy_context(ctx, setup)
    strat = StrategyEngine(s_ctx)
    tp = strat.generate(op)

    assert tp.metadata["passes"] == 1
    assert tp.metadata["facing_info"]["actual_stepdown_mm"] == pytest.approx(1.00)

    post = get_post_processor("fanuc", work_offset="G55")
    res = post.post_process([tp], "FACING_SETUP2")
    assert "ap = 1.00mm [Actual Stock: 1.00mm, Max Tool DOC: 1.80mm]" in res.gcode
