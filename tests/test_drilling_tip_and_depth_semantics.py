"""Unit tests for drill-tip geometry and depth semantics."""

import numpy as np
import pytest

from cam_engine.context import (
    DepthReference,
    DrillDepthType,
    DrillingDepthSemantics,
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
from cam_engine.machinability import CuttingParameters
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.post import get_post_processor
from cam_engine.units import Units


def _make_drill_env(tool_dia=0.58, point_angle=118.0):
    tool = Tool(
        id="T_DRILL",
        tool_number=1,
        diameter=tool_dia,
        type=ToolType.DRILL,
        point_angle=point_angle,
        flute_length=10.0,
        overall_length=38.0,
        flutes=2,
        material="carbide",
        can_plunge=True,
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -5.0, 0.0]),
        bounds_max=np.array([5.0, 5.0, 3.22]),
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
    )
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup],
        machine=machine,
        tools=[tool],
        material=DEFAULT_MATERIALS["aluminum_6061"],
    )
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    strat_ctx = StrategyContext(
        context=ctx,
        setup=setup,
        mesh=mesh,
        stock_voxels=None,
        clearance_z=14.22,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    return engine, tool, setup, ctx


def test_blind_hole_tip_depth_reference():
    """Verify that tip-depth referencing stops the drill point exactly at the feature floor."""
    engine, tool, setup, ctx = _make_drill_env(tool_dia=0.58, point_angle=118.0)

    # Top blind hole: Z from 3.22 to 1.47 (cylindrical depth = 1.75 mm)
    feat = MachiningFeature(
        id="f_hole_top",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        diameter=0.58,
        is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=4000.0,
        feed_rate=100.0,
        plunge_feed=80.0,
        ramp_feed=50.0,
        feed_per_tooth_mm=0.01,
        depth_of_cut_mm=0.58,
        stepover_mm=0.58,
        coolant=True,
        effective_diameter_mm=0.58,
        engagement_angle_deg=180.0,
        power_kw=0.2,
        torque_nm=0.5,
    )
    op = PlannedOperation(
        id="op_drill_top",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.DRILLING,
        params=params,
        notes={"depth_reference": "tip", "remaining_web": 0.20},
    )

    tp = engine._drilling(op)
    intent = tp.metadata["canned_cycle_intent"]

    # Tip length for Ø0.58 mm @ 118 deg is (0.29 / tan(59 deg)) ≈ 0.1742 mm
    expected_tip_len = 0.29 / np.tan(np.radians(59.0))
    assert pytest.approx(intent["tip_length"], rel=1e-3) == expected_tip_len

    # Tip depth reference means cycle final depth is exactly floor_z = 1.470 mm
    assert pytest.approx(intent["final_depth"], rel=1e-4) == 1.470
    assert intent["depth_reference"] == "tip"

    # Postprocessor produces auditable G-code comment
    post = get_post_processor("haas", ctx.machine)
    post_res = post.post_process([tp])
    assert "Drill Depth Semantics: Ref=Tip Depth" in post_res.gcode
    assert "Z1.47" in post_res.gcode


def test_blind_hole_shoulder_depth_reference():
    """Verify that shoulder-depth referencing extends drill tip to ensure full cylindrical depth."""
    engine, tool, setup, ctx = _make_drill_env(tool_dia=0.58, point_angle=118.0)

    feat = MachiningFeature(
        id="f_hole_top",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        diameter=0.58,
        is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=4000.0,
        feed_rate=100.0,
        plunge_feed=80.0,
        ramp_feed=50.0,
        feed_per_tooth_mm=0.01,
        depth_of_cut_mm=0.58,
        stepover_mm=0.58,
        coolant=True,
        effective_diameter_mm=0.58,
        engagement_angle_deg=180.0,
        power_kw=0.2,
        torque_nm=0.5,
    )
    op = PlannedOperation(
        id="op_drill_top",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.DRILLING,
        params=params,
        notes={"depth_reference": "shoulder"},
    )

    tp = engine._drilling(op)
    intent = tp.metadata["canned_cycle_intent"]

    expected_tip_len = 0.29 / np.tan(np.radians(59.0))
    expected_cycle_z = 1.47 - expected_tip_len
    assert pytest.approx(intent["final_depth"], rel=1e-4) == expected_cycle_z
    assert intent["depth_reference"] == "shoulder"

    post = get_post_processor("haas", ctx.machine)
    post_res = post.post_process([tp])
    assert "Drill Depth Semantics: Ref=Shoulder (Full Dia)" in post_res.gcode
    assert f"Z{expected_cycle_z:.3f}" in post_res.gcode


def test_through_hole_breakthrough_depth():
    """Verify through-hole depth calculation adds tip length and breakthrough allowance."""
    engine, tool, setup, ctx = _make_drill_env(tool_dia=3.0, point_angle=118.0)

    feat = MachiningFeature(
        id="f_thru_hole",
        type=FeatureType.THROUGH_HOLE,
        face_indices=[0],
        bounds_min=np.array([-1.5, -1.5, 0.0]),
        bounds_max=np.array([1.5, 1.5, 10.0]),
        depth=10.0,
        top_z=10.0,
        floor_z=0.0,
        diameter=3.0,
        is_concave=True,
        notes={"is_through": True},
    )
    params = CuttingParameters(
        spindle_rpm=3000.0,
        feed_rate=200.0,
        plunge_feed=150.0,
        ramp_feed=100.0,
        feed_per_tooth_mm=0.025,
        depth_of_cut_mm=1.5,
        stepover_mm=3.0,
        coolant=True,
        effective_diameter_mm=3.0,
        engagement_angle_deg=180.0,
        power_kw=1.0,
        torque_nm=2.0,
    )
    op = PlannedOperation(
        id="op_drill_thru",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.DRILLING,
        params=params,
        notes={"breakthrough": 0.5},
    )

    tp = engine._drilling(op)
    intent = tp.metadata["canned_cycle_intent"]

    # Tip length for Ø3.0 mm @ 118 deg is 1.5 / tan(59 deg) ≈ 0.9012 mm
    expected_tip_len = 1.5 / np.tan(np.radians(59.0))
    expected_endpoint = 0.0 - expected_tip_len - 0.5
    assert pytest.approx(intent["final_depth"], rel=1e-4) == expected_endpoint
    assert intent["breakthrough"] == 0.5


def test_opposing_holes_web_protection_enforces_tip_reference():
    """Verify that when two opposing holes leave a 0.20mm web, OperationPlanner enforces tip depth."""
    _, tool1, setup1, ctx = _make_drill_env(tool_dia=0.58, point_angle=118.0)
    tool2 = Tool(
        id="T_DRILL2",
        tool_number=2,
        diameter=0.51,
        type=ToolType.DRILL,
        point_angle=118.0,
        flute_length=10.0,
        overall_length=38.0,
        flutes=2,
        material="carbide",
        can_plunge=True,
    )
    ctx.tools.append(tool2)

    # Top blind hole: Z from 3.22 to 1.47
    feat1 = MachiningFeature(
        id="f_hole_top",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        diameter=0.58,
        is_concave=True,
    )
    # Bottom blind hole: Z from 0.0 to 1.27
    feat2 = MachiningFeature(
        id="f_hole_bot",
        type=FeatureType.BLIND_HOLE,
        face_indices=[1],
        bounds_min=np.array([-0.255, -0.255, 0.0]),
        bounds_max=np.array([0.255, 0.255, 1.27]),
        depth=1.27,
        top_z=0.0,
        floor_z=1.27,
        diameter=0.51,
        is_concave=True,
    )

    planner = OperationPlanner(ctx, setup1, [feat1, feat2])
    # Run thin wall / opposing hole validation
    op_top = PlannedOperation(
        id="op_drill_1",
        feature=feat1,
        tool=tool1,
        purpose=OpPurpose.DRILLING,
        params=CuttingParameters(
            spindle_rpm=4000.0, feed_rate=100.0, plunge_feed=80.0, ramp_feed=50.0,
            feed_per_tooth_mm=0.01, depth_of_cut_mm=0.58, stepover_mm=0.58, coolant=True,
            effective_diameter_mm=0.58, engagement_angle_deg=180.0, power_kw=0.2, torque_nm=0.5
        ),
        notes={"depth_reference": "shoulder"}  # initially requested shoulder
    )
    op_bot = PlannedOperation(
        id="op_drill_2",
        feature=feat2,
        tool=tool2,
        purpose=OpPurpose.DRILLING,
        params=CuttingParameters(
            spindle_rpm=4000.0, feed_rate=100.0, plunge_feed=80.0, ramp_feed=50.0,
            feed_per_tooth_mm=0.01, depth_of_cut_mm=0.51, stepover_mm=0.51, coolant=True,
            effective_diameter_mm=0.51, engagement_angle_deg=180.0, power_kw=0.2, torque_nm=0.5
        ),
        notes={"depth_reference": "shoulder"}
    )

    planner._validate_feature_webs_and_walls([op_top, op_bot])

    # Because tool 1 tip length (0.174mm) and tool 2 tip length (0.153mm) violate the 0.20mm web,
    # the planner must automatically downgrade depth_reference to 'tip' to protect the web!
    assert op_top.notes["depth_reference"] == "tip"
    assert op_bot.notes["depth_reference"] == "tip"
    assert any(w["code"] == "CRITICAL_WEB_THICKNESS" for w in planner.warnings)


def test_tool_tip_geometry_model_dynamic_calculations():
    """Verify tool tip length formulas for various point angles and tip geometries."""
    # 1. 118 deg standard drill Ø0.58mm
    t_118_58 = Tool(
        id="T01", tool_number=1, diameter=0.58, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=118.0, tip_geometry="conical"
    )
    # L = (0.29 / tan(59°)) ≈ 0.17425mm
    assert pytest.approx(t_118_58.tip_length, abs=1e-3) == 0.174

    # 2. 118 deg standard drill Ø0.51mm (Setup 2)
    t_118_51 = Tool(
        id="T02", tool_number=2, diameter=0.51, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=118.0, tip_geometry="conical"
    )
    # L = (0.255 / tan(59°)) ≈ 0.15322mm
    assert pytest.approx(t_118_51.tip_length, abs=1e-3) == 0.153

    # 3. 135 deg split-point drill Ø0.58mm
    t_135 = Tool(
        id="T03", tool_number=3, diameter=0.58, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=135.0, tip_geometry="split_point"
    )
    # L = (0.29 / tan(67.5°)) ≈ 0.12012mm
    assert pytest.approx(t_135.tip_length, abs=1e-3) == 0.120

    # 4. 140 deg high-performance carbide drill Ø0.58mm
    t_140 = Tool(
        id="T04", tool_number=4, diameter=0.58, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=140.0, tip_geometry="faceted"
    )
    # L = (0.29 / tan(70°)) ≈ 0.10555mm
    assert pytest.approx(t_140.tip_length, abs=1e-3) == 0.106

    # 5. 90 deg spot drill / countersink Ø0.58mm
    t_90 = Tool(
        id="T05", tool_number=5, diameter=0.58, flute_length=10.0, overall_length=38.0,
        type=ToolType.SPOT_DRILL, point_angle=90.0, tip_geometry="conical"
    )
    # L = (0.29 / tan(45°)) = 0.290mm
    assert pytest.approx(t_90.tip_length, abs=1e-3) == 0.290

    # 6. Flat bottom drill (180 deg)
    t_flat = Tool(
        id="T06", tool_number=6, diameter=0.58, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=180.0, tip_geometry="flat"
    )
    assert t_flat.tip_length == 0.0

    # 7. Ball endmill
    t_ball = Tool(
        id="T07", tool_number=7, diameter=3.0, flute_length=10.0, overall_length=38.0,
        type=ToolType.BALL_ENDMILL, tip_radius=1.5
    )
    assert pytest.approx(t_ball.tip_length, abs=1e-4) == 1.5


def test_drawing_cylindrical_depth_to_programmed_tip_depth_chain():
    """Verify the dynamic parameter derivation chain:
    Drawing required cylindrical depth -> tool-tip geometry -> required programmed tip depth -> G83 Z
    """
    tool_118 = Tool(
        id="T_118", tool_number=1, diameter=0.58, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=118.0, tip_geometry="conical"
    )
    start_z = 3.220
    drawing_cyl_depth = 1.750

    # 1. Tip depth reference: Z_cycle = start_z - drawing_cyl_depth = 1.470
    z_tip = tool_118.compute_programmed_depth(start_z, drawing_cyl_depth, depth_reference="tip")
    assert pytest.approx(z_tip, abs=1e-4) == 1.470

    # 2. Shoulder depth reference: Z_cycle = start_z - (drawing_cyl_depth + tip_length) = 1.470 - 0.174 = 1.296
    z_shoulder = tool_118.compute_programmed_depth(start_z, drawing_cyl_depth, depth_reference="shoulder")
    assert pytest.approx(z_shoulder, abs=1e-3) == 1.296

    # 3. Setup 2: start_z = 0.0, drawing_cyl_depth = 1.270
    tool_setup2 = Tool(
        id="T_51", tool_number=2, diameter=0.51, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=118.0, tip_geometry="conical"
    )
    z_setup2_tip = tool_setup2.compute_programmed_depth(0.0, 1.270, depth_reference="tip")
    assert pytest.approx(z_setup2_tip, abs=1e-4) == -1.270

    z_setup2_shoulder = tool_setup2.compute_programmed_depth(0.0, 1.270, depth_reference="shoulder")
    # 0.0 - (1.270 + 0.153) = -1.423
    assert pytest.approx(z_setup2_shoulder, abs=1e-3) == -1.423


def test_post_processor_drill_audit_comment_includes_point_angle_and_geometry():
    """Verify that generated G-code contains full tool geometry audit info."""
    engine, tool, setup, ctx = _make_drill_env(tool_dia=0.58, point_angle=118.0)
    feat = MachiningFeature(
        id="f_hole",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        diameter=0.58,
        is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=4000.0, feed_rate=100.0, plunge_feed=80.0, ramp_feed=50.0,
        feed_per_tooth_mm=0.01, depth_of_cut_mm=0.61, stepover_mm=0.58, coolant=True,
        effective_diameter_mm=0.58, engagement_angle_deg=180.0, power_kw=0.2, torque_nm=0.5
    )
    op = PlannedOperation(
        id="op_drill_audit",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.DRILLING,
        params=params,
        notes={"depth_reference": "tip", "remaining_web": 0.20, "r_clearance": 1.0},
    )
    tp = engine._drilling(op)
    post = get_post_processor("haas", ctx.machine)
    post_res = post.post_process([tp])

    gcode = post_res.gcode
    # Expect: (--- Drill Depth Semantics: Ref=Tip Depth, Cycle Z=1.470, Tip Len=0.174mm [Point Angle: 118.0deg, Geom: conical], Cyl Depth=1.576mm, Tip Travel=1.750mm, Remaining Web: 0.200mm ---)
    assert "Drill Depth Semantics: Ref=Tip Depth" in gcode
    assert "Point Angle: 118.0deg" in gcode
    assert "Geom: conical" in gcode
    assert "Tip Len=0.174mm" in gcode
    assert "Cyl Depth=1.576mm" in gcode
    assert "Tip Travel=1.750mm" in gcode
    assert "Remaining Web: 0.200mm" in gcode
    assert "G83" in gcode
    assert "Z1.47" in gcode
    assert "R4.22" in gcode


def test_explicit_cam_data_model_drill_depth_semantics():
    """Verify that the CAM data model explicitly carries:
    depthReference = TOP_FACE
    depthType = CYLINDRICAL_DEPTH / TIP_DEPTH
    targetDepth = drawing depth
    toolTipGeometry = conical / split_point / faceted / flat
    programmedTipDepth = calculated
    """
    engine, tool, setup, ctx = _make_drill_env(tool_dia=0.58, point_angle=118.0)

    # 1. Setup 1 scenario: top_z=3.22, drawing depth=1.75
    # CASE A: Cylindrical Depth Target
    sem_cyl = tool.compute_drill_depth_semantics(
        start_z=3.22,
        target_depth=1.75,
        depth_type=DrillDepthType.CYLINDRICAL_DEPTH,
        depth_reference=DepthReference.TOP_FACE,
    )
    assert sem_cyl.depth_reference == DepthReference.TOP_FACE
    assert sem_cyl.depth_type == DrillDepthType.CYLINDRICAL_DEPTH
    assert sem_cyl.target_depth == 1.75
    assert sem_cyl.tool_point_angle_deg == 118.0
    assert sem_cyl.tool_tip_geometry == "conical"
    assert pytest.approx(sem_cyl.tool_tip_length, abs=1e-3) == 0.174
    # Tip must travel past cylindrical floor: 3.22 - (1.75 + 0.174) = 1.296
    assert pytest.approx(sem_cyl.programmed_tip_depth, abs=1e-3) == 1.296
    assert sem_cyl.achieved_cylindrical_depth == 1.75

    # CASE B: Tip Depth Target (Drawing depth specifies apex)
    sem_tip = tool.compute_drill_depth_semantics(
        start_z=3.22,
        target_depth=1.75,
        depth_type=DrillDepthType.TIP_DEPTH,
        depth_reference=DepthReference.TOP_FACE,
        remaining_web=0.20,
    )
    assert sem_tip.depth_reference == DepthReference.TOP_FACE
    assert sem_tip.depth_type == DrillDepthType.TIP_DEPTH
    assert sem_tip.target_depth == 1.75
    assert pytest.approx(sem_tip.programmed_tip_depth, abs=1e-4) == 1.470
    assert pytest.approx(sem_tip.achieved_cylindrical_depth, abs=1e-3) == 1.576
    assert sem_tip.total_hole_depth == 1.75
    assert sem_tip.remaining_web_thickness == 0.20

    # 2. Setup 2 scenario: top_z=0.0, drawing depth=1.27, tool Ø0.51
    tool2 = Tool(
        id="T02", tool_number=2, diameter=0.51, flute_length=10.0, overall_length=38.0,
        type=ToolType.DRILL, point_angle=118.0, tip_geometry="conical"
    )
    sem2_tip = tool2.compute_drill_depth_semantics(
        start_z=0.0,
        target_depth=1.27,
        depth_type=DrillDepthType.TIP_DEPTH,
        depth_reference=DepthReference.TOP_FACE,
    )
    assert sem2_tip.depth_reference == DepthReference.TOP_FACE
    assert sem2_tip.depth_type == DrillDepthType.TIP_DEPTH
    assert sem2_tip.target_depth == 1.27
    assert pytest.approx(sem2_tip.programmed_tip_depth, abs=1e-4) == -1.270
    assert pytest.approx(sem2_tip.achieved_cylindrical_depth, abs=1e-3) == 1.117

    # 3. Verify toolpath generation propagates semantics into canned_cycle_intent
    feat = MachiningFeature(
        id="f_hole_top",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        diameter=0.58,
        is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=4000.0, feed_rate=100.0, plunge_feed=80.0, ramp_feed=50.0,
        feed_per_tooth_mm=0.01, depth_of_cut_mm=0.58, stepover_mm=0.58, coolant=True,
        effective_diameter_mm=0.58, engagement_angle_deg=180.0, power_kw=0.2, torque_nm=0.5
    )
    op = PlannedOperation(
        id="op_drill_semantic",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.DRILLING,
        params=params,
        depth_semantics=sem_tip,
        notes={"depth_reference": "tip", "remaining_web": 0.20},
    )
    tp = engine._drilling(op)
    assert tp.metadata["depth_semantics"]["depth_reference"] == "TOP_FACE"
    assert tp.metadata["depth_semantics"]["depth_type"] == "TIP_DEPTH"
    assert tp.metadata["depth_semantics"]["target_depth"] == 1.75
    assert pytest.approx(tp.metadata["canned_cycle_intent"]["programmed_tip_depth"], rel=1e-4) == 1.47


