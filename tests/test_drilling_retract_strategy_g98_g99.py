"""Unit tests for operation-driven G98 (Initial Plane) vs G99 (R-Plane) retract strategy."""

import numpy as np
import pytest

from cam_engine.context import (
    Fixture,
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
from cam_engine.machinability import MachinabilityAnalyzer, CuttingParameters
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.post import get_post_processor
from cam_engine.units import Units


def _make_env(fixtures=None):
    tool = Tool(
        id="T_DRILL",
        tool_number=1,
        diameter=3.0,
        type=ToolType.DRILL,
        point_angle=118.0,
        flute_length=20.0,
        overall_length=50.0,
        flutes=2,
        material="carbide",
        can_plunge=True,
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-10.0, -10.0, 0.0]),
        bounds_max=np.array([10.0, 10.0, 5.0]),
    )
    setup = Setup(
        id="setup_01",
        name="Setup 1",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
        fixtures=fixtures or [],
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
        work_offsets={"G54": np.zeros(3)},
        tool_change_position=np.array([0.0, 0.0, 100.0]),
    )
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup],
        machine=machine,
        tools=[tool],
        material=mat,
    )
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    strat_ctx = StrategyContext(
        context=ctx,
        setup=setup,
        mesh=mesh,
        stock_voxels=None,
        clearance_z=15.0,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    analyzer = MachinabilityAnalyzer(machine, mat)
    return engine, tool, setup, ctx, analyzer, mesh


def test_single_hole_defaults_to_g98_initial_plane():
    """Verify that a single isolated hole operation defaults to G98 (Initial Plane return)."""
    engine, tool, setup, ctx, analyzer, mesh = _make_env()
    feat = MachiningFeature(
        id="f_single_hole", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]), bounds_max=np.array([1.5, 1.5, 5.0]),
        depth=5.0, top_z=5.0, floor_z=0.0, diameter=3.0, is_concave=True,
    )
    params = analyzer.analyze(tool, feat, "drilling", setup)
    op = PlannedOperation(
        id="op_single_drill", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={"r_clearance": 1.0}  # R plane = 5.0 + 1.0 = 6.0mm
    )
    tp = engine._drilling(op)
    cycle = tp.metadata.get("canned_cycle_intent", {})

    assert cycle["retract_mode"] == "G98"
    assert "Single Hole" in cycle["retract_reason"]

    post = get_post_processor("haas", ctx.machine)
    post_res = post.post_process([tp])
    gcode = post_res.gcode

    assert "Drill Retract Strategy: Mode=G98" in gcode
    assert "Initial Plane" in gcode
    assert "G98 G83" in gcode


def test_multi_hole_pattern_selects_g99_r_plane():
    """Verify that a multi-hole pattern on a planar face selects G99 (R-plane return)."""
    engine, tool, setup, ctx, analyzer, mesh = _make_env()
    feat = MachiningFeature(
        id="f_pattern_hole", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]), bounds_max=np.array([1.5, 1.5, 5.0]),
        depth=5.0, top_z=5.0, floor_z=0.0, diameter=3.0, is_concave=True,
    )
    params = analyzer.analyze(tool, feat, "drilling", setup)
    op = PlannedOperation(
        id="op_pattern_drill", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={"is_multi_hole": True, "r_clearance": 1.0}
    )
    tp = engine._drilling(op)
    cycle = tp.metadata.get("canned_cycle_intent", {})

    assert cycle["retract_mode"] == "G99"
    assert "Planar Pattern" in cycle["retract_reason"]

    post = get_post_processor("haas", ctx.machine)
    post_res = post.post_process([tp])
    gcode = post_res.gcode

    assert "Drill Retract Strategy: Mode=G99" in gcode
    assert "R-Plane" in gcode
    assert "G99 G83" in gcode


def test_tall_fixture_forces_g98_even_on_pattern():
    """Verify that an intervening fixture exceeding R-plane forces G98 return."""
    # Fixture with top Z = 8.0 mm (higher than R-plane Z = 6.0 mm)
    clamp = Fixture(
        id="toe_clamp_1",
        name="Toe Clamp 1",
        kind="clamp",
        bounds_min=np.array([-20.0, -20.0, 0.0]),
        bounds_max=np.array([-12.0, -12.0, 8.0]),
    )
    engine, tool, setup, ctx, analyzer, mesh = _make_env(fixtures=[clamp])
    feat = MachiningFeature(
        id="f_pattern_hole", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]), bounds_max=np.array([1.5, 1.5, 5.0]),
        depth=5.0, top_z=5.0, floor_z=0.0, diameter=3.0, is_concave=True,
    )
    params = analyzer.analyze(tool, feat, "drilling", setup)
    op = PlannedOperation(
        id="op_pattern_clamp", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={"is_multi_hole": True, "r_clearance": 1.0}
    )
    tp = engine._drilling(op)
    cycle = tp.metadata.get("canned_cycle_intent", {})

    # Must automatically downgrade to G98 due to tall fixture
    assert cycle["retract_mode"] == "G98"
    assert "Fixture Clearance" in cycle["retract_reason"]

    post = get_post_processor("haas", ctx.machine)
    post_res = post.post_process([tp])
    assert "G98 G83" in post_res.gcode


def test_operation_explicit_retract_override():
    """Verify that explicit operation notes override (G98/G99) are honored."""
    engine, tool, setup, ctx, analyzer, mesh = _make_env()
    feat = MachiningFeature(
        id="f_hole", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]), bounds_max=np.array([1.5, 1.5, 5.0]),
        depth=5.0, top_z=5.0, floor_z=0.0, diameter=3.0, is_concave=True,
    )
    params = analyzer.analyze(tool, feat, "drilling", setup)

    # Force G99 on single hole
    op1 = PlannedOperation(
        id="op1", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={"retract_mode": "G99"}
    )
    tp1 = engine._drilling(op1)
    assert tp1.metadata["canned_cycle_intent"]["retract_mode"] == "G99"

    # Force G98 on multi-hole pattern
    op2 = PlannedOperation(
        id="op2", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={"is_multi_hole": True, "retract_mode": "G98"}
    )
    tp2 = engine._drilling(op2)
    assert tp2.metadata["canned_cycle_intent"]["retract_mode"] == "G98"


def test_validator_catches_unsafe_g99_fixture_collision():
    """Verify that ToolpathValidator flags an error if G99 is forced with a fixture collision."""
    clamp = Fixture(
        id="tall_clamp",
        name="Tall Clamp",
        kind="clamp",
        bounds_min=np.array([-5.0, -5.0, 0.0]),
        bounds_max=np.array([5.0, 5.0, 10.0]),  # 10mm high
    )
    engine, tool, setup, ctx, analyzer, mesh = _make_env(fixtures=[clamp])
    feat = MachiningFeature(
        id="f_hole", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]), bounds_max=np.array([1.5, 1.5, 5.0]),
        depth=5.0, top_z=5.0, floor_z=0.0, diameter=3.0, is_concave=True,
    )
    params = analyzer.analyze(tool, feat, "drilling", setup)
    # Dangerously force G99 with R=6.0mm when fixture is 10.0mm high
    op = PlannedOperation(
        id="op_unsafe_g99", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={"retract_mode": "G99", "r_clearance": 1.0}
    )
    tp = engine._drilling(op)
    
    validator = ToolpathValidator(ctx, setup, mesh=mesh, clearance_z=15.0)
    report = validator.validate(tp)

    assert not report.passed
    assert any("G99 R-plane return" in e.message and "collides with fixture" in e.message for e in report.errors)
