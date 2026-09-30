"""Unit tests for dynamic finishing depth strategy and level calculations."""

import numpy as np
import pytest

from cam_engine.context import (
    MachineAxis,
    MachineConfig,
    MachineType,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
    DEFAULT_MATERIALS,
)
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.geometry import TriangleMesh
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.units import Units


def _make_dummy_env(tool_dia=3.0):
    tool = Tool(
        id="T01",
        tool_number=1,
        diameter=tool_dia,
        type=ToolType.FLAT_ENDMILL,
        flute_length=15.0,
        overall_length=50.0,
        flutes=3,
        material="carbide",
        can_plunge=True,
        diameter_offset_d=1,
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-20.0, -20.0, -10.0]),
        bounds_max=np.array([20.0, 20.0, 5.0])
    )
    setup = Setup(
        id="setup_01",
        name="Setup 1",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock
    )
    machine = MachineConfig(
        id="M01",
        name="3-Axis Mill",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="haas",
        spindle_max_rpm=15000.0,
        spindle_min_rpm=100.0,
        max_spindle_power_kw=15.0,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 10000.0, 5000.0),
            "Y": MachineAxis("Y", -300.0, 300.0, 10000.0, 5000.0),
            "Z": MachineAxis("Z", -200.0, 300.0, 10000.0, 5000.0),
        },
        safe_retract_height=10.0,
    )
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup],
        machine=machine,
        tools=[tool],
        material=DEFAULT_MATERIALS["aluminum_6061"]
    )
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    strat_ctx = StrategyContext(
        context=ctx,
        setup=setup,
        mesh=mesh,
        stock_voxels=None,
        clearance_z=15.0,
        finish_allowance_wall=0.1,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    return engine, tool


def test_dynamic_depth_levels_setup1_and_setup2():
    """Verify dynamic depth level calculation for 3.22 mm part in Setup 1 and Setup 2."""
    engine, tool = _make_dummy_env()

    # Setup 1: top_z = 3.22, floor_z = 0.0, max_stepdown = 1.2
    levels_s1_fixed = engine._compute_finishing_depth_levels(
        top_z=3.22, floor_z=0.0, max_stepdown=1.2, remaining_allowance=0.0, mode="fixed"
    )
    assert len(levels_s1_fixed) == 3
    assert pytest.approx(levels_s1_fixed[0], rel=1e-4) == 2.02
    assert pytest.approx(levels_s1_fixed[1], rel=1e-4) == 0.82
    assert pytest.approx(levels_s1_fixed[2], rel=1e-4) == 0.00
    # Top-down order check
    assert levels_s1_fixed[0] > levels_s1_fixed[1] > levels_s1_fixed[2]

    # Setup 2: top_z = 0.0, floor_z = -3.22, max_stepdown = 1.2
    levels_s2_fixed = engine._compute_finishing_depth_levels(
        top_z=0.0, floor_z=-3.22, max_stepdown=1.2, remaining_allowance=0.0, mode="fixed"
    )
    assert len(levels_s2_fixed) == 3
    assert pytest.approx(levels_s2_fixed[0], rel=1e-4) == -1.20
    assert pytest.approx(levels_s2_fixed[1], rel=1e-4) == -2.40
    assert pytest.approx(levels_s2_fixed[2], rel=1e-4) == -3.22
    # Top-down order check
    assert levels_s2_fixed[0] > levels_s2_fixed[1] > levels_s2_fixed[2]


def test_dynamic_depth_levels_equalized():
    """Verify equalized depth levels distribute passes evenly."""
    engine, tool = _make_dummy_env()

    # Setup 1 with equalized passes
    levels_s1_eq = engine._compute_finishing_depth_levels(
        top_z=3.22, floor_z=0.0, max_stepdown=1.2, remaining_allowance=0.0, mode="equalized"
    )
    assert len(levels_s1_eq) == 3
    expected_step = 3.22 / 3.0
    assert pytest.approx(levels_s1_eq[0], rel=1e-4) == 3.22 - expected_step
    assert pytest.approx(levels_s1_eq[1], rel=1e-4) == 3.22 - 2 * expected_step
    assert pytest.approx(levels_s1_eq[2], rel=1e-4) == 0.00
    # Every step is equal and <= max_stepdown
    for i in range(len(levels_s1_eq) - 1):
        step_taken = levels_s1_eq[i] - levels_s1_eq[i + 1]
        assert pytest.approx(step_taken, rel=1e-4) == expected_step
        assert step_taken <= 1.2 + 1e-6


def test_dynamic_depth_levels_with_floor_allowance():
    """Verify remaining floor allowance stops the finishing/semi-finishing passes above nominal floor."""
    engine, tool = _make_dummy_env()

    # top_z = 3.22, floor_z = 0.0, allowance = 0.22, max_stepdown = 1.5
    # total machining depth = 3.0 mm -> 2 passes of 1.5 mm
    levels = engine._compute_finishing_depth_levels(
        top_z=3.22, floor_z=0.0, max_stepdown=1.5, remaining_allowance=0.22, mode="fixed"
    )
    assert len(levels) == 2
    assert pytest.approx(levels[0], rel=1e-4) == 1.72
    assert pytest.approx(levels[1], rel=1e-4) == 0.22


def test_dynamic_depth_levels_single_pass_and_boundary():
    """Verify single pass when total depth is smaller than max_stepdown or zero."""
    engine, tool = _make_dummy_env()

    # Total depth = 0.5 mm, max_stepdown = 1.0 mm -> 1 pass at floor
    levels_single = engine._compute_finishing_depth_levels(
        top_z=0.5, floor_z=0.0, max_stepdown=1.0, remaining_allowance=0.0
    )
    assert levels_single == [0.0]

    # Total depth = 0.0 mm (facing/shallow surface) -> 1 pass at floor
    levels_zero = engine._compute_finishing_depth_levels(
        top_z=0.0, floor_z=0.0, max_stepdown=1.0, remaining_allowance=0.0
    )
    assert levels_zero == [0.0]


def test_boss_circular_contour_generates_top_down_passes():
    """Verify that _boss_circular_contour generates cuts descending strictly top-down at dynamic Z levels."""
    engine, tool = _make_dummy_env(tool_dia=3.0)

    feat = MachiningFeature(
        id="f_boss",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 0.0]),
        bounds_max=np.array([0.45, 0.45, 3.22]),
        depth=3.22,
        top_z=3.22,
        floor_z=0.0,
        diameter=0.90,
        is_concave=False,
        notes={"is_circular": True},
    )
    params = CuttingParameters(
        spindle_rpm=3000.0,
        feed_rate=500.0,
        plunge_feed=100.0,
        ramp_feed=250.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=1.2,
        stepover_mm=1.0,
        coolant=True,
        effective_diameter_mm=3.0,
        engagement_angle_deg=90.0,
        power_kw=1.0,
        torque_nm=2.0,
    )
    op = PlannedOperation(
        id="op_finish_boss",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.FINISHING,
        params=params,
        notes={"purpose": "finishing", "depth_mode": "fixed"},
    )

    tp = engine._boss_circular_contour(op)
    assert tp.metadata["depth_levels"] == [2.02, 0.82, 0.0]

    # Collect all cutting arc Z coordinates
    cut_z_values = [
        seg.start[2] for seg in tp.segments
        if seg.motion_type in (MotionType.ARC_CCW, MotionType.ARC_CW, MotionType.CUT)
    ]
    # Check that cut Z values strictly descend
    unique_z = []
    for z in cut_z_values:
        if not unique_z or abs(z - unique_z[-1]) > 1e-4:
            unique_z.append(z)

    assert unique_z == [2.02, 0.82, 0.0]


def test_finish_wall_floor_generates_top_down_passes():
    """Verify that _finish_wall_floor executes contour passes from top down, never bottom up."""
    engine, tool = _make_dummy_env(tool_dia=3.0)

    feat = MachiningFeature(
        id="f_step",
        type=FeatureType.STEP,
        face_indices=[1],
        bounds_min=np.array([-5.0, -5.0, 0.0]),
        bounds_max=np.array([5.0, 5.0, 3.22]),
        depth=3.22,
        top_z=3.22,
        floor_z=0.0,
        notes={"is_open": True},
    )
    params = CuttingParameters(
        spindle_rpm=3000.0,
        feed_rate=500.0,
        plunge_feed=100.0,
        ramp_feed=250.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=1.2,
        stepover_mm=1.0,
        coolant=True,
        effective_diameter_mm=3.0,
        engagement_angle_deg=90.0,
        power_kw=1.0,
        torque_nm=2.0,
    )
    op = PlannedOperation(
        id="op_finish_wall",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.FINISHING,
        params=params,
        notes={"purpose": "finishing", "depth_mode": "fixed"},
    )

    tp = engine._finish_wall_floor(op)
    assert tp.metadata["depth_levels"] == [2.02, 0.82, 0.0]

    # Check that entry/cut segment Z levels appear in strictly descending order
    entry_z_values = [
        seg.start[2] for seg in tp.segments
        if seg.motion_type == MotionType.ENTRY
    ]
    assert len(entry_z_values) == 3
    assert entry_z_values[0] > entry_z_values[1] > entry_z_values[2]
    assert pytest.approx(entry_z_values[0], rel=1e-4) == 2.02
    assert pytest.approx(entry_z_values[1], rel=1e-4) == 0.82
    assert pytest.approx(entry_z_values[2], rel=1e-4) == 0.00
