"""Unit tests for rule-based drilling peck depth (Q) strategy, material scaling, and validation."""

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
from cam_engine.operations import PlannedOperation, OpPurpose, OperationPlanner
from cam_engine.machinability import MachinabilityAnalyzer, CuttingParameters
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.post import get_post_processor
from cam_engine.units import Units


def _make_env(tool_dia=0.58, mat_id="aluminum_6061", point_angle=118.0, drill_type="twist", has_tsc=False, max_peck_ratio=None):
    tool = Tool(
        id=f"T_DRILL_{tool_dia}",
        tool_number=1,
        diameter=tool_dia,
        type=ToolType.DRILL,
        point_angle=point_angle,
        drill_type=drill_type,
        has_through_coolant=has_tsc,
        max_peck_ratio=max_peck_ratio,
        flute_length=10.0,
        overall_length=38.0,
        flutes=2,
        material="carbide",
        can_plunge=True,
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -5.0, 0.0]),
        bounds_max=np.array([5.0, 5.0, 4.0]),
    )
    setup = Setup(
        id="setup_01",
        name="Setup 1",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
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
    mat = DEFAULT_MATERIALS.get(mat_id, DEFAULT_MATERIALS["aluminum_6061"])
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
        clearance_z=14.0,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    analyzer = MachinabilityAnalyzer(machine, mat)
    return engine, tool, setup, ctx, analyzer, mesh


def test_micro_drilling_peck_limits_setup1_and_setup2():
    """Verify that micro-drills (D < 1.0mm) have safe pecks (Q <= 0.50*D) in Aluminum and BeCu."""
    # Setup 1: Ø0.58 mm drill, hole depth 1.75 mm (L/D = 3.02)
    engine1, tool1, setup1, ctx1, analyzer1, mesh1 = _make_env(tool_dia=0.58, mat_id="aluminum_6061")
    feat1 = MachiningFeature(
        id="f_hole_setup1", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75, top_z=3.22, floor_z=1.47, diameter=0.58, is_concave=True,
    )
    strat1 = analyzer1.calculate_drilling_peck_strategy(tool1, feat1)
    
    # Q must be <= 0.50 * D (0.29mm)
    assert strat1["first_peck"] <= 0.50 * 0.58
    assert strat1["peck_ratio"] <= 0.50
    assert strat1["first_peck"] > 0.05
    assert strat1["cycle_type"] == "G83"

    # In BeCu C17200 (tough, work-hardening): peck should be even more conservative
    _, tool1_becu, _, _, analyzer1_becu, _ = _make_env(tool_dia=0.58, mat_id="becu")
    strat1_becu = analyzer1_becu.calculate_drilling_peck_strategy(tool1_becu, feat1)
    assert strat1_becu["first_peck"] < strat1["first_peck"]
    assert strat1_becu["first_peck"] <= 0.35 * 0.58

    # Setup 2: Ø0.51 mm drill, hole depth 1.27 mm (L/D = 2.49)
    engine2, tool2, setup2, ctx2, analyzer2, mesh2 = _make_env(tool_dia=0.51, mat_id="aluminum_6061")
    feat2 = MachiningFeature(
        id="f_hole_setup2", type=FeatureType.BLIND_HOLE, face_indices=[1],
        bounds_min=np.array([-0.255, -0.255, 0.0]), bounds_max=np.array([0.255, 0.255, 1.27]),
        depth=1.27, top_z=0.0, floor_z=-1.27, diameter=0.51, is_concave=True,
    )
    strat2 = analyzer2.calculate_drilling_peck_strategy(tool2, feat2)
    assert strat2["first_peck"] <= 0.50 * 0.51
    assert strat2["peck_ratio"] <= 0.50


def test_material_specific_peck_scaling():
    """Verify that different material classes scale peck depth appropriately for identical Ø3.0mm drill."""
    feat = MachiningFeature(
        id="f_hole_3mm", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]), bounds_max=np.array([1.5, 1.5, 9.0]),
        depth=9.0, top_z=9.0, floor_z=0.0, diameter=3.0, is_concave=True,
    )

    # 1. Aluminum 6061 (soft non-ferrous)
    _, t_al, _, _, an_al, _ = _make_env(tool_dia=3.0, mat_id="aluminum_6061")
    peck_al = an_al.calculate_drilling_peck(t_al, feat)

    # 2. BeCu C17200 (work-hardening copper alloy)
    _, t_becu, _, _, an_becu, _ = _make_env(tool_dia=3.0, mat_id="becu")
    peck_becu = an_becu.calculate_drilling_peck(t_becu, feat)

    # 3. Stainless 304 (austenitic, stringy)
    _, t_ss, _, _, an_ss, _ = _make_env(tool_dia=3.0, mat_id="stainless_304")
    peck_ss = an_ss.calculate_drilling_peck(t_ss, feat)

    # 4. Inconel 718 (superalloy, extremely tough)
    _, t_inc, _, _, an_inc, _ = _make_env(tool_dia=3.0, mat_id="inconel_718")
    peck_inc = an_inc.calculate_drilling_peck(t_inc, feat)

    # Verify expected physical hierarchy:
    # Q(Aluminum) > Q(BeCu) > Q(Stainless 304) > Q(Inconel 718)
    assert peck_al > peck_becu
    assert peck_becu > peck_ss
    assert peck_ss > peck_inc


def test_aspect_ratio_and_progressive_peck_reduction():
    """Verify that deep holes (L/D > 3.0) decay subsequent pecks while shallow holes do not."""
    _, tool, _, _, analyzer, _ = _make_env(tool_dia=2.0, mat_id="aluminum_6061")

    # Shallow hole: depth = 2.0 mm (L/D = 1.0)
    feat_shallow = MachiningFeature(
        id="f_shallow", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.0, -1.0, 0.0]), bounds_max=np.array([1.0, 1.0, 2.0]),
        depth=2.0, top_z=2.0, floor_z=0.0, diameter=2.0, is_concave=True,
    )
    strat_shallow = analyzer.calculate_drilling_peck_strategy(tool, feat_shallow)
    assert strat_shallow["peck_reduction"] == 0.0
    assert strat_shallow["subsequent_peck"] == strat_shallow["first_peck"]

    # Deep hole: depth = 12.0 mm (L/D = 6.0)
    feat_deep = MachiningFeature(
        id="f_deep", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.0, -1.0, 0.0]), bounds_max=np.array([1.0, 1.0, 12.0]),
        depth=12.0, top_z=12.0, floor_z=0.0, diameter=2.0, is_concave=True,
    )
    strat_deep = analyzer.calculate_drilling_peck_strategy(tool, feat_deep)
    assert strat_deep["peck_reduction"] > 0.0
    assert strat_deep["subsequent_peck"] < strat_deep["first_peck"]
    assert strat_deep["min_peck"] <= strat_deep["subsequent_peck"]
    assert strat_deep["cycle_type"] == "G83"


def test_through_spindle_coolant_and_tool_manufacturer_limit():
    """Verify that TSC increases allowable peck and max_peck_ratio strictly caps peck."""
    # Standard drill vs TSC drill
    feat = MachiningFeature(
        id="f_hole", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]), bounds_max=np.array([1.5, 1.5, 15.0]),
        depth=15.0, top_z=15.0, floor_z=0.0, diameter=3.0, is_concave=True,
    )
    _, t_std, _, _, an, _ = _make_env(tool_dia=3.0, mat_id="steel_1018", has_tsc=False)
    _, t_tsc, _, _, _, _ = _make_env(tool_dia=3.0, mat_id="steel_1018", has_tsc=True)

    strat_std = an.calculate_drilling_peck_strategy(t_std, feat)
    strat_tsc = an.calculate_drilling_peck_strategy(t_tsc, feat)
    assert strat_tsc["first_peck"] > strat_std["first_peck"]

    # Custom manufacturer limit: e.g. max_peck_ratio = 0.40
    _, t_custom, _, _, _, _ = _make_env(tool_dia=3.0, mat_id="steel_1018", max_peck_ratio=0.40)
    strat_custom = an.calculate_drilling_peck_strategy(t_custom, feat)
    assert strat_custom["first_peck"] <= 0.40 * 3.0 + 1e-4


def test_validator_detects_excessive_micro_drill_peck():
    """Verify that ToolpathValidator flags dangerous pecks on micro-drills."""
    engine, tool, setup, ctx, _, mesh = _make_env(tool_dia=0.58, mat_id="aluminum_6061")
    feat = MachiningFeature(
        id="f_hole", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75, top_z=3.22, floor_z=1.47, diameter=0.58, is_concave=True,
    )
    # Intentionally create dangerous operation with Q = 0.90 mm (1.55 x D on Ø0.58 mm drill)
    params = CuttingParameters(
        spindle_rpm=4000.0, feed_rate=100.0, plunge_feed=80.0, ramp_feed=50.0,
        feed_per_tooth_mm=0.01, depth_of_cut_mm=0.90, stepover_mm=0.58, coolant=True,
        effective_diameter_mm=0.58, engagement_angle_deg=180.0, power_kw=0.2, torque_nm=0.5
    )
    op = PlannedOperation(
        id="op_drill_bad", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={"first_peck": 0.90, "cycle_type": "G83"}
    )
    tp = engine._drilling(op)
    
    validator = ToolpathValidator(ctx, setup, mesh=mesh, clearance_z=14.0)
    report = validator.validate(tp)
    assert any("excessive for micro-drill" in w.message for w in report.warnings)


def test_post_processor_drill_peck_audit_comment_emission():
    """Verify that generated G-code contains Drill Peck Strategy audit comment."""
    engine, tool, setup, ctx, analyzer, _ = _make_env(tool_dia=0.58, mat_id="becu")
    feat = MachiningFeature(
        id="f_hole_becu", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75, top_z=3.22, floor_z=1.47, diameter=0.58, is_concave=True,
    )
    params = analyzer.analyze(tool, feat, OpPurpose.DRILLING.value, setup)
    strat = analyzer.calculate_drilling_peck_strategy(tool, feat)
    
    op = PlannedOperation(
        id="op_drill_becu", feature=feat, tool=tool, purpose=OpPurpose.DRILLING,
        params=params, notes={
            "depth_reference": "tip", "remaining_web": 0.20,
            "first_peck": strat["first_peck"],
            "subsequent_peck": strat["subsequent_peck"],
            "peck_reduction": strat["peck_reduction"],
            "min_peck": strat["min_peck"],
            "cycle_type": strat["cycle_type"],
            "peck_ratio": strat["peck_ratio"],
            "ld_ratio": strat["ld_ratio"],
        }
    )
    tp = engine._drilling(op)
    post = get_post_processor("haas", ctx.machine)
    post_res = post.post_process([tp])
    gcode = post_res.gcode

    assert "Drill Peck Strategy: Q=" in gcode
    assert "Cycle=G83" in gcode
    assert "Mat=Beryllium Copper" in gcode
    assert f"Q{strat['first_peck']:.3f}" in gcode
