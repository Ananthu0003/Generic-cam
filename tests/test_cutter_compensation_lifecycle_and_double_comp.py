import numpy as np
import pytest

from cam_engine.context import (
    MachineAxis,
    MachineConfig,
    MachineType,
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
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.geometry import TriangleMesh
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath, CoordSpace
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.post import get_post_processor
from cam_engine.errors import CamError, INVALID_TOOLPATH
from cam_engine.units import Units


def _make_env():
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, flute_length=15.0, overall_length=50.0,
        flutes=3, material="carbide", can_plunge=True
    )
    drill = Tool(
        id="T02", tool_number=2, type=ToolType.DRILL,
        diameter=3.0, flute_length=15.0, overall_length=50.0,
        flutes=2, material="hss", can_plunge=True
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-20.0, -20.0, -10.0]),
        bounds_max=np.array([20.0, 20.0, 5.0])
    )
    setup = Setup(
        id="setup_01", name="Setup 1",
        model_to_setup=np.eye(4), work_offset=WorkOffset.G54,
        stock=stock
    )
    machine = MachineConfig(
        id="M01", name="3-Axis Mill", machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="haas", spindle_max_rpm=15000.0, spindle_min_rpm=100.0,
        max_spindle_power_kw=15.0,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 10000.0, 5000.0),
            "Y": MachineAxis("Y", -300.0, 300.0, 10000.0, 5000.0),
            "Z": MachineAxis("Z", -200.0, 300.0, 10000.0, 5000.0),
        },
        tool_change_position=np.array([0.0, 0.0, 100.0]),
        safe_retract_height=10.0,
        work_offsets={"G54": np.zeros(3)},
    )
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup], machine=machine,
        tools=[tool, drill], material=DEFAULT_MATERIALS["aluminum_6061"]
    )
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    validator = ToolpathValidator(ctx, setup, mesh, clearance_z=15.0)
    return ctx, setup, tool, drill, validator


def test_tool_type_compatibility_rejects_drills():
    """Verify that cutter compensation on a drill or tap is rejected."""
    _, setup, _, drill, validator = _make_env()
    tp = Toolpath(
        operation_id="op_drill_invalid", feature_id="f1",
        tool_id=drill.id, purpose="finishing", space=CoordSpace.SETUP
    )
    # Linear lead-in engaging G41 with drill
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([10.0, 0.0, 0.0]), end=np.array([5.0, 0.0, 0.0]),
        feed=500.0, tool_id=drill.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 2}}
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([5.0, 0.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
        feed=500.0, tool_id=drill.id, operation_id=tp.operation_id
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.EXIT,
        start=np.array([0.0, 0.0, 0.0]), end=np.array([-5.0, 0.0, 0.0]),
        feed=500.0, tool_id=drill.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "off"}}
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("cannot use cutter radius compensation" in e.message for e in report.errors)


def test_double_compensation_detected():
    """Verify that specifying control compensation on a tool-center path is rejected as double compensation."""
    _, setup, tool, _, validator = _make_env()
    tp = Toolpath(
        operation_id="op_double_comp", feature_id="f1",
        tool_id=tool.id, purpose="finishing", space=CoordSpace.SETUP,
        metadata={"path_type": "tool_center", "compensation_mode": "control"}
    )
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([10.0, 0.0, 0.0]), end=np.array([5.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.EXIT,
        start=np.array([5.0, 0.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "off"}}
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("Double compensation detected" in e.message for e in report.errors)


def test_uncompensated_part_contour_detected():
    """Verify that a toolpath declared on part_contour with compensation_mode off is rejected."""
    _, setup, tool, _, validator = _make_env()
    tp = Toolpath(
        operation_id="op_uncomp_contour", feature_id="f1",
        tool_id=tool.id, purpose="finishing", space=CoordSpace.SETUP,
        metadata={"path_type": "part_contour", "compensation_mode": "off"}
    )
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([5.0, 0.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("Uncompensated part contour" in e.message for e in report.errors)


def test_cutter_comp_lead_in_must_be_linear():
    """Verify that activating G41 on an arc or rapid is rejected."""
    _, setup, tool, _, validator = _make_env()
    tp = Toolpath(
        operation_id="op_arc_lead", feature_id="f1",
        tool_id=tool.id, purpose="finishing", space=CoordSpace.SETUP
    )
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=np.array([10.0, 0.0, 0.0]), end=np.array([5.0, 5.0, 0.0]),
        arc_center=np.array([10.0, 5.0, 0.0]), arc_ccw=False,
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.EXIT,
        start=np.array([5.0, 5.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "off"}}
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("cannot be activated directly on circular arc/helix" in e.message for e in report.errors)


def test_rapid_forbidden_under_active_cutter_comp():
    """Verify that executing a rapid move while G41 is active is rejected."""
    _, setup, tool, _, validator = _make_env()
    tp = Toolpath(
        operation_id="op_rapid_in_comp", feature_id="f1",
        tool_id=tool.id, purpose="finishing", space=CoordSpace.SETUP
    )
    # 1. Valid linear lead-in
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([10.0, 0.0, 0.0]), end=np.array([5.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
    ))
    # 2. Illegal rapid across workpiece while comp is active
    tp.add(MotionSegment(
        motion_type=MotionType.RAPID,
        start=np.array([5.0, 0.0, 0.0]), end=np.array([0.0, 10.0, 0.0]),
        tool_id=tool.id, operation_id=tp.operation_id
    ))
    # 3. Exit
    tp.add(MotionSegment(
        motion_type=MotionType.EXIT,
        start=np.array([0.0, 10.0, 0.0]), end=np.array([-5.0, 10.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "off"}}
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("rapid motion is illegal while cutter compensation" in e.message for e in report.errors)


def test_uncancelled_cutter_comp_at_toolpath_end():
    """Verify that completing a toolpath with active G41 and no G40 is rejected."""
    _, setup, tool, _, validator = _make_env()
    tp = Toolpath(
        operation_id="op_uncancelled_comp", feature_id="f1",
        tool_id=tool.id, purpose="finishing", space=CoordSpace.SETUP
    )
    # 1. Linear lead-in
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([10.0, 0.0, 0.0]), end=np.array([5.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id,
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
    ))
    # 2. Cut
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([5.0, 0.0, 0.0]), end=np.array([0.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id=tp.operation_id
    ))
    # Missing exit G40!

    report = validator.validate(tp)
    assert report.passed is False
    assert any("completed with uncancelled cutter compensation" in e.message for e in report.errors)


def test_valid_cutter_comp_lifecycle_and_post_audit():
    """Verify that a valid cutter compensation lifecycle passes Gate 4 and generates traceable G-code."""
    ctx, setup, tool, _, validator = _make_env()

    feat = MachiningFeature(
        id="f_boss", type=FeatureType.BOSS,
        face_indices=[0], bounds_min=np.array([-5, -5, 0]),
        bounds_max=np.array([5, 5, 5]), depth=5.0,
        top_z=5.0, floor_z=0.0, accessibility=Accessibility.TOOL_AXIS_OK,
        diameter=10.0,
    )
    params = CuttingParameters(
        spindle_rpm=12000.0, feed_rate=1500.0, plunge_feed=300.0,
        ramp_feed=600.0, depth_of_cut_mm=5.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.0417, coolant=True,
        effective_diameter_mm=3.0, engagement_angle_deg=90.0,
        power_kw=0.5, torque_nm=0.4
    )
    op = PlannedOperation(
        id="op_boss_finish", feature=feat, tool=tool,
        purpose=OpPurpose.FINISHING, params=params,
        notes={"cutter_comp": True},
    )


    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=15.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    # 1. Gate 4 Validation
    report = validator.validate(tp)
    assert report.passed is True
    assert len(report.errors) == 0

    # 2. Metadata Check
    assert tp.metadata["path_type"] in ("center_path", "tool_center")
    assert tp.metadata["compensation_mode"] == "wear"

    assert tp.metadata["cutter_comp_info"]["d_register"] == 1
    assert tp.metadata["cutter_comp_info"]["tool_diameter"] == 3.0

    # 3. Post-processing Check
    post = get_post_processor("haas", work_offset="G54")
    res = post.post_process([tp], program_name="TEST_CRC")

    assert "G41 D1" in res.gcode
    assert "G40" in res.gcode
    assert "Cutter Comp: G41 D1" in res.gcode
    assert "Tool: T01 Ø3.00mm" in res.gcode
    assert "Mode: Wear (Tool Center Path" in res.gcode


def test_cutter_comp_comment_and_gcode_exact_sync():
    """Verify that the operation audit block comment and the actual NC code block (G41/G42)
    NEVER disagree on the D register under any renumbering or mapping policy."""
    ctx, setup, tool, drill, validator = _make_env()
    # Tool with standard convention (diameter_offset_d=None -> D matches active T)
    tool_3mm = Tool(
        id="T03", tool_number=3, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, flute_length=12.0, overall_length=50.0,
        flutes=3, material="carbide", can_plunge=True,
        length_offset_h=3, diameter_offset_d=None
    )

    feat = MachiningFeature(
        id="f_boss", type=FeatureType.BOSS,
        face_indices=[0], bounds_min=np.array([-5, -5, 0]),
        bounds_max=np.array([5, 5, 5]), depth=5.0,
        top_z=5.0, floor_z=0.0, accessibility=Accessibility.TOOL_AXIS_OK,
        diameter=10.0,
    )
    params = CuttingParameters(
        spindle_rpm=8001.0, feed_rate=1200.0, plunge_feed=300.0,
        ramp_feed=600.0, depth_of_cut_mm=5.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=3.0, engagement_angle_deg=90.0,
        power_kw=0.5, torque_nm=0.4
    )
    op = PlannedOperation(
        id="op_boss_finish", feature=feat, tool=tool_3mm,
        purpose=OpPurpose.FINISHING, params=params,
        notes={"cutter_comp": True},
    )

    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=15.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    # 1. Case A: Auto-renumbered to T01 -> both comment and code must say D1
    post_auto = get_post_processor("haas", renumber_tools=True)
    res_auto = post_auto.post_process([tp], program_name="TEST_SYNC_AUTO")
    assert "Cutter Comp: G41 D1" in res_auto.gcode
    assert "G41 D1" in res_auto.gcode
    assert "D3" not in res_auto.gcode

    # 2. Case B: Preserved tool numbers (renumber_tools=False) -> both comment and code must say D3
    post_explicit = get_post_processor("haas", renumber_tools=False)
    res_explicit = post_explicit.post_process([tp], program_name="TEST_SYNC_EXPLICIT")
    assert "Cutter Comp: G41 D3" in res_explicit.gcode
    assert "G41 D3" in res_explicit.gcode
    assert "D1" not in res_explicit.gcode

    # 3. Case C: Explicit d_offset_map -> both comment and code must say D33
    post_mapped = get_post_processor("fanuc", d_offset_map={"T03": 33})
    res_mapped = post_mapped.post_process([tp], program_name="TEST_SYNC_MAPPED")
    assert "Cutter Comp: G41 D33" in res_mapped.gcode
    assert "G41 D33" in res_mapped.gcode


def test_multi_setup_finishing_d_register_synchronization():
    """Verify that in multi-setup jobs (Setup 1 Top, Setup 2 Bottom), finishing operations
    in Setup 2 maintain complete synchronization between tool number, H offset, D offset,
    audit comments, and actual emitted G41 G-code blocks."""
    ctx, setup1, tool1, drill, validator = _make_env()

    # Tool 3: Ø3.0mm finish endmill
    tool3 = Tool(
        id="T03", tool_number=3, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, flute_length=15.0, overall_length=50.0,
        flutes=3, material="carbide", can_plunge=True,
        length_offset_h=3, diameter_offset_d=3
    )

    feat_s1 = MachiningFeature(
        id="f_s1_boss", type=FeatureType.BOSS,
        face_indices=[0], bounds_min=np.array([-5, -5, 0]),
        bounds_max=np.array([5, 5, 5]), depth=5.0,
        top_z=5.0, floor_z=0.0, accessibility=Accessibility.TOOL_AXIS_OK,
        diameter=10.0,
    )
    feat_s2 = MachiningFeature(
        id="f_s2_contour", type=FeatureType.CONTOUR,
        face_indices=[0], bounds_min=np.array([-10, -10, -5]),
        bounds_max=np.array([10, 10, 0]), depth=5.0,
        top_z=0.0, floor_z=-5.0, accessibility=Accessibility.TOOL_AXIS_OK,
    )

    params = CuttingParameters(
        spindle_rpm=8001.0, feed_rate=1200.0, plunge_feed=300.0,
        ramp_feed=600.0, depth_of_cut_mm=5.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=3.0, engagement_angle_deg=90.0,
        power_kw=0.5, torque_nm=0.4
    )

    op1 = PlannedOperation(id="op1_s1_finish", feature=feat_s1, tool=tool3, purpose=OpPurpose.FINISHING, params=params, notes={"cutter_comp": True})
    op2 = PlannedOperation(id="op2_s2_finish", feature=feat_s2, tool=tool3, purpose=OpPurpose.FINISHING, params=params, notes={"cutter_comp": True})


    # Setup 2 context
    setup2 = Setup(
        id="setup_002", name="Setup 2 - Bottom Flip",
        model_to_setup=np.eye(4), work_offset=WorkOffset.G55,
        stock=Stock(kind=StockKind.BOX, bounds_min=np.array([-10, -10, -6]), bounds_max=np.array([10, 10, 0]))
    )

    engine1 = StrategyEngine(StrategyContext(
        context=ctx, setup=setup1, mesh=None, stock_voxels=None,
        clearance_z=15.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    ))
    engine2 = StrategyEngine(StrategyContext(
        context=ctx, setup=setup2, mesh=None, stock_voxels=None,
        clearance_z=15.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    ))

    tp1 = engine1.generate(op1)
    tp1.metadata["setup_id"] = "setup_001"
    tp1.metadata["work_offset"] = "G54"

    tp2 = engine2.generate(op2)
    tp2.metadata["setup_id"] = "setup_002"
    tp2.metadata["work_offset"] = "G55"

    # 1. Combined post-processing across Setup 1 and Setup 2 (renumber_tools=False)
    post_unrenumbered = get_post_processor("haas", renumber_tools=False)
    res_comb = post_unrenumbered.post_process([tp1, tp2], program_name="MULTI_SETUP_CRC")

    # In Setup 1: Tool 3 -> G43 H3, Cutter Comp: G41 D3, G41 D3
    # In Setup 2: Tool 3 -> G43 H3, Cutter Comp: G41 D3, G41 D3
    assert "Cutter Comp: G41 D3" in res_comb.gcode
    assert "G41 D3" in res_comb.gcode
    assert "G41 D1" not in res_comb.gcode

    # 2. Per-Setup 2 isolated export with job tool mapping (T3 mapped)
    post_s2 = get_post_processor("haas", work_offset="G55", renumber_tools=True, tool_map={"T03": 3})
    res_s2 = post_s2.post_process([tp2], program_name="SETUP2_PROG", setup_id="setup_002")

    assert "Cutter Comp: G41 D3" in res_s2.gcode
    assert "G41 D3" in res_s2.gcode
    assert "G41 D1" not in res_s2.gcode
