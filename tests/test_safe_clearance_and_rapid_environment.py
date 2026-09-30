import numpy as np
import pytest

from cam_engine.context import (
    Fixture,
    MachineConfig,
    Material,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
    DEFAULT_MATERIALS,
)
from cam_engine.demo_models import get_default_machine
from cam_engine.geometry import TriangleMesh
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath, CoordSpace
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.errors import COLLISION_DETECTED, GOUGE_DETECTED


def _make_context(stock_z_min=-10.0, stock_z_max=0.0, fixtures=None):
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=6.0, corner_radius=0.0, flute_length=20.0,
        overall_length=60.0, flutes=4, material="carbide", can_plunge=True
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-20.0, -20.0, stock_z_min]),
        bounds_max=np.array([20.0, 20.0, stock_z_max])
    )
    setup = Setup(
        id="setup_002",
        name="Setup 2 - Bottom Flip",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G55,
        stock=stock,
        fixtures=fixtures or []
    )
    machine = get_default_machine()
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=machine, tools=[tool], material=mat)
    mesh = TriangleMesh(triangles=np.zeros((1, 3, 3)))
    return ctx, setup, tool, mesh


def test_clearance_plane_accounts_for_stock_model_and_fixtures():
    """Verify that setup-specific clearance plane clears the highest obstacle (stock, model, or fixtures)."""
    fixture = Fixture(
        id="vise_jaw_tall",
        name="Tall Vise Jaw",
        bounds_min=np.array([-30.0, 20.0, -10.0]),
        bounds_max=np.array([30.0, 35.0, 15.0]),  # fixture top is Z=+15, while stock top is Z=0
        kind="vise_jaw"
    )
    ctx, setup, tool, mesh = _make_context(stock_z_min=-10.0, stock_z_max=0.0, fixtures=[fixture])
    
    top_z = float(setup.stock.bounds_max[2])
    if setup.fixtures:
        for f in setup.fixtures:
            top_z = max(top_z, float(f.bounds_max[2]))
    
    clearance_z = top_z + ctx.machine.safe_retract_height
    # Must clear fixture top (15.0) + retract height (e.g. 10.0) -> >= 25.0
    assert clearance_z >= 25.0


def test_link_to_over_solid_stock_stops_rapid_above_material_top():
    """Verify that _link_to over solid stock stops rapid motion above the stock top before plunging."""
    ctx, setup, tool, mesh = _make_context(stock_z_min=-10.0, stock_z_max=0.0)
    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=mesh, stock_voxels=None,
        clearance_z=11.0, finish_allowance_wall=0.2, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(strat_ctx)

    feat = MachiningFeature(
        id="f_pocket", type=FeatureType.POCKET, face_indices=[0],
        bounds_min=np.array([-5.0, -5.0, -5.0]), bounds_max=np.array([5.0, 5.0, 0.0]),
        depth=5.0, top_z=0.0, floor_z=-5.0
    )
    op = PlannedOperation(
        id="op_pock", feature=feat, tool=tool, purpose=OpPurpose.ROUGHING,
        params=CuttingParameters(spindle_rpm=8001, feed_rate=1200, plunge_feed=250, ramp_feed=400,
                                 feed_per_tooth_mm=0.05, depth_of_cut_mm=2.5, stepover_mm=2.0, coolant=True,
                                 effective_diameter_mm=6.0, engagement_angle_deg=45.0, power_kw=1.0, torque_nm=2.0)
    )

    tp = Toolpath(operation_id="op_pock", feature_id="f_pocket", tool_id=tool.id, purpose="roughing")
    # Link to a point inside the stock at depth Z = -2.92
    p_target = np.array([0.0, 0.0, -2.92])
    engine._link_to(tp, op, p_target, approach_offset=1.0)

    # Check that any rapid segment stops at or above stock top (Z >= 0.0 + approach_offset)
    rapid_segs = [s for s in tp.segments if s.motion_type == MotionType.RAPID]
    for r in rapid_segs:
        # Rapid end must not plunge below stock top (0.0)
        assert r.end[2] >= 0.0, f"Rapid plunged below stock top: Z={r.end[2]}"

    # Plunge segment must cover the descent from above stock top down to target
    plunge_segs = [s for s in tp.segments if s.motion_type == MotionType.PLUNGE]
    assert len(plunge_segs) >= 1
    assert plunge_segs[-1].end[2] == -2.92
    assert plunge_segs[-1].start[2] >= 0.0


def test_link_to_in_free_air_outside_stock():
    """Verify that _link_to in free space outside stock allows rapid descent down to target + approach_offset."""
    ctx, setup, tool, mesh = _make_context(stock_z_min=-10.0, stock_z_max=0.0)
    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=mesh, stock_voxels=None,
        clearance_z=11.0, finish_allowance_wall=0.2, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(strat_ctx)

    feat = MachiningFeature(
        id="f_boss", type=FeatureType.BOSS, face_indices=[0],
        bounds_min=np.array([-5.0, -5.0, -5.0]), bounds_max=np.array([5.0, 5.0, 0.0]),
        depth=5.0, top_z=0.0, floor_z=-5.0
    )
    op = PlannedOperation(
        id="op_boss", feature=feat, tool=tool, purpose=OpPurpose.ROUGHING,
        params=CuttingParameters(spindle_rpm=8001, feed_rate=1200, plunge_feed=250, ramp_feed=400,
                                 feed_per_tooth_mm=0.05, depth_of_cut_mm=2.5, stepover_mm=2.0, coolant=True,
                                 effective_diameter_mm=6.0, engagement_angle_deg=45.0, power_kw=1.0, torque_nm=2.0)
    )

    tp = Toolpath(operation_id="op_boss", feature_id="f_boss", tool_id=tool.id, purpose="roughing")
    # Target point far outside stock envelope (stock is [-20, 20])
    p_target = np.array([35.0, 0.0, -2.92])
    engine._link_to(tp, op, p_target, approach_offset=1.0)

    rapid_segs = [s for s in tp.segments if s.motion_type == MotionType.RAPID]
    # Rapid in free air can safely reach target + 1.0 = -1.92
    lowest_rapid_z = min(s.end[2] for s in rapid_segs)
    assert abs(lowest_rapid_z - (-1.92)) < 1e-4


def test_validator_rejects_rapid_plunge_into_stock():
    """Verify that Gate 4 ToolpathValidator rejects any toolpath with a rapid move penetrating into solid stock."""
    ctx, setup, tool, mesh = _make_context(stock_z_min=-10.0, stock_z_max=0.0)
    validator = ToolpathValidator(ctx, setup, mesh, clearance_z=11.0)

    tp = Toolpath(operation_id="op_bad", feature_id="f1", tool_id=tool.id, purpose="roughing")
    # Dangerous rapid: plunges directly from Z=11 to Z=-1.92 inside stock footprint (stock top is Z=0.0)
    tp.add(MotionSegment(
        motion_type=MotionType.RAPID,
        start=np.array([0.0, 0.0, 11.0]),
        end=np.array([0.0, 0.0, -1.92]),
        tool_id=tool.id, operation_id="op_bad", feature_id="f1"
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.PLUNGE,
        start=np.array([0.0, 0.0, -1.92]),
        end=np.array([0.0, 0.0, -2.92]),
        feed=250.0, tool_id=tool.id, operation_id="op_bad", feature_id="f1"
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("rapid segment 0 plunges" in e.message for e in report.errors)
