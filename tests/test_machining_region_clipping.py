"""Tests for Machining Region Derivation & Stock-Bounded Toolpath Validation.

Verifies:
1. Toolpaths are strictly generated within the machining region (Stock + Model + Tool).
2. Gate 4 ToolpathValidator validates cuts against both Cylindrical and Box stock envelopes.
3. Air-cutting segments extending outside the stock envelope are rejected with INVALID_TOOLPATH errors.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    DEFAULT_MATERIALS, WorkOffset, MachineConfig, MachineType, MachineAxis, Units
)
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath, CoordSpace
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.geometry.mesh import TriangleMesh


def make_test_context(stock: Stock) -> StrategyContext:
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.machine = MachineConfig(
        id="test-vmc", name="Test VMC", machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 15000.0, 5000.0),
            "Y": MachineAxis("Y", -500.0, 500.0, 15000.0, 5000.0),
            "Z": MachineAxis("Z", -400.0, 50.0, 10000.0, 3000.0),
        },
        spindle_min_rpm=100.0, spindle_max_rpm=15000.0, max_spindle_power_kw=7.5,
        controller="fanuc", units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0, hsm_enabled=True,
    )
    ctx.material = DEFAULT_MATERIALS["aluminum_6061"]
    ctx.tools = [
        Tool(id="t_endmill_2mm", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=2.0,
             flute_length=15.0, overall_length=45.0, flutes=2),
    ]
    setup = Setup(id="setup_1", name="Top Setup", model_to_setup=np.eye(4),
                  work_offset=WorkOffset.G54, stock=stock)
    
    plan_ctx = PlanningContext(
        model=None, setups=[setup], machine=ctx.machine, tools=ctx.tools,
        material=ctx.material
    )
    return StrategyContext(
        context=plan_ctx, setup=setup, mesh=mesh,
        stock_voxels=None, clearance_z=20.0, finish_allowance_wall=0.2, finish_allowance_floor=0.1
    )


def test_roughing_path_passes_gate4_validation():
    """Verify generated roughing toolpath passes Gate 4 stock envelope validation."""
    stock = Stock.from_cylinder(center_xy=np.array([0.0, 0.0]), z_min=0.0, z_max=15.0, radius=2.5)
    strat_ctx = make_test_context(stock)
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]

    boss_feature = MachiningFeature(
        id="pin_090mm",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 0.0]),
        bounds_max=np.array([0.45, 0.45, 10.0]),
        depth=10.0,
        top_z=10.0,
        floor_z=0.0,
        diameter=0.90,
        is_concave=False,
        notes={"is_circular": True}
    )
    params = CuttingParameters(
        spindle_rpm=10000.0, feed_rate=1000.0, plunge_feed=300.0, ramp_feed=500.0,
        feed_per_tooth_mm=0.05, depth_of_cut_mm=2.0, stepover_mm=0.5, coolant=True,
        effective_diameter_mm=2.0, engagement_angle_deg=45.0, power_kw=1.0, torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_rough_pin",
        purpose=OpPurpose.ROUGHING,
        feature=boss_feature,
        tool=tool,
        params=params,
    )

    tp = engine.generate(op)
    validator = ToolpathValidator(strat_ctx.context, strat_ctx.setup, strat_ctx.mesh, strat_ctx.clearance_z)
    report = validator.validate(tp)

    assert report.passed is True
    assert len(report.errors) == 0


def test_validator_rejects_out_of_stock_cutting():
    """Verify validator flags any cutting move that significantly exceeds stock boundaries."""
    stock_box = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-10.0, -10.0, 0.0]),
        bounds_max=np.array([10.0, 10.0, 20.0]),
    )
    strat_ctx = make_test_context(stock_box)
    validator = ToolpathValidator(strat_ctx.context, strat_ctx.setup, strat_ctx.mesh, strat_ctx.clearance_z)

    # Toolpath with cut move way out at X=50.0mm (air-cutting error)
    tp = Toolpath(operation_id="op_bad", feature_id="f1", tool_id="t_endmill_2mm", purpose="roughing", space=CoordSpace.SETUP)
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([45.0, 0.0, 10.0]),
        end=np.array([55.0, 0.0, 10.0]),
        feed=1000.0,
        tool_id="t_endmill_2mm",
        operation_id="op_bad",
        feature_id="f1"
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("extends outside the machining stock region" in str(e.message) for e in report.errors)
