"""Tests for Automatic Tool Numbering and Tool Dimensions across Post-Processors, Setup Sheets, and API."""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    DEFAULT_MATERIALS, WorkOffset, MachineConfig, MachineType, MachineAxis, Units,
)
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.toolpaths.semantic import MotionType, MotionSegment, Toolpath
from cam_engine.post import (
    get_post_processor,
    GrblPostProcessor,
    FanucPostProcessor,
    HaasPostProcessor,
    LinuxCncPostProcessor,
)
from cam_engine.setup_sheet import generate_setup_sheet, generate_master_routing_sheet


def make_test_toolpath(
    op_id: str,
    purpose: str,
    tool_id: str,
    feature_id: str = "f1",
    tool_info: dict = None,
) -> Toolpath:
    tp = Toolpath(operation_id=op_id, purpose=purpose, tool_id=tool_id, feature_id=feature_id)
    if tool_info:
        tp.metadata["tool_info"] = tool_info

    # Tool change segment
    tp.add(MotionSegment(
        motion_type=MotionType.TOOL_CHANGE,
        start=np.array([0.0, 0.0, 20.0]),
        end=np.array([0.0, 0.0, 20.0]),
        tool_id=tool_id,
        operation_id=op_id,
        feature_id=feature_id,
        metadata={"tool_info": tool_info} if tool_info else {},
    ))
    # Rapid move
    tp.add(MotionSegment(
        motion_type=MotionType.RAPID,
        start=np.array([0.0, 0.0, 20.0]),
        end=np.array([10.0, 10.0, 5.0]),
        tool_id=tool_id,
        operation_id=op_id,
        feature_id=feature_id,
    ))
    # Cut move
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([10.0, 10.0, 5.0]),
        end=np.array([20.0, 10.0, 5.0]),
        feed=500.0,
        spindle=8001.0,
        tool_id=tool_id,
        operation_id=op_id,
        feature_id=feature_id,
    ))
    return tp


def test_grbl_automatic_sequential_tool_numbering():
    # Sequence of operations using arbitrary tool IDs
    tp1 = make_test_toolpath("op_facing_001", "facing", "t_face_cutter_50")
    tp2 = make_test_toolpath("op_spot_001", "spot_drilling", "t_spot_drill_90deg")
    tp3 = make_test_toolpath("op_rough_001", "roughing", "t_endmill_dia04")
    tp4 = make_test_toolpath("op_facing_002", "facing", "t_face_cutter_50")  # reuse tool 1

    post = GrblPostProcessor(work_offset="G54", renumber_tools=True)
    res = post.post_process([tp1, tp2, tp3, tp4], program_name="TEST_AUTO_NUM")

    gcode = res.gcode
    # 1st tool -> T1
    assert "T1 M0 (Change to T01" in gcode
    # 2nd tool -> T2
    assert "T2 M0 (Change to T02" in gcode
    # 3rd tool -> T3
    assert "T3 M0 (Change to T03" in gcode
    # Reused 1st tool -> should NOT create a new tool number T4, but reference T01
    assert "T01" in gcode
    assert "T4" not in gcode


def test_fanuc_automatic_tool_numbering_and_offset_sync():
    tp1 = make_test_toolpath("op_rough", "roughing", "t_rough_carbide")
    tp2 = make_test_toolpath("op_drill", "drilling", "t_micro_drill_058")

    # Add cutter compensation segment to op 1
    tp1.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([20.0, 10.0, 5.0]),
        end=np.array([25.0, 10.0, 5.0]),
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 99}},
        tool_id="t_rough_carbide",
        operation_id="op_rough",
        feature_id="f1",
    ))

    post = FanucPostProcessor(work_offset="G54", renumber_tools=True)
    res = post.post_process([tp1, tp2], program_name="TEST_FANUC_AUTO")

    gcode = res.gcode
    # T1 tool change and H1 length offset
    assert "T1 M6 (Tool: T01" in gcode
    assert "G43 H1" in gcode
    # D register synced to sequential tool number (D1 instead of D99)
    assert "G41 D1" in gcode
    # T2 tool change and H2 length offset
    assert "T2 M6 (Tool: T02" in gcode
    assert "G43 H2" in gcode


def test_explicit_job_tool_map():
    # Multi-setup sharing unified tool map
    tp1 = make_test_toolpath("op1", "facing", "T03")
    tp2 = make_test_toolpath("op2", "drilling", "T20")

    custom_map = {"T03": 1, "T20": 2}
    post = get_post_processor("grbl", work_offset="G54", tool_map=custom_map)
    res = post.post_process([tp1, tp2], program_name="TEST_MAP")

    assert "T1 M0 (Change to T01" in res.gcode
    assert "T2 M0 (Change to T02" in res.gcode


def test_tool_dimensions_in_gcode_header_and_operations():
    tool1_info = {
        "id": "t_face_50",
        "type": "face_mill",
        "diameter": 50.0,
        "flute_length": 15.0,
        "flutes": 4,
    }
    tool2_info = {
        "id": "t_drill_058",
        "type": "drill",
        "diameter": 0.58,
        "flute_length": 5.0,
        "flutes": 2,
    }

    tp1 = make_test_toolpath("op_face", "facing", "t_face_50", "f1", tool1_info)
    tp2 = make_test_toolpath("op_drill", "drilling", "t_drill_058", "f2", tool2_info)

    # Test Fanuc post
    fanuc_post = FanucPostProcessor(work_offset="G54")
    res_fanuc = fanuc_post.post_process([tp1, tp2], program_name="FANUC_DIM_TEST")
    
    # Check header tool table
    assert "TOOLS USED IN THIS PROGRAM" in res_fanuc.gcode
    assert "T01: Face Mill | Ø50mm | FL: 15.0mm | 4F" in res_fanuc.gcode
    assert "T02: Drill | Ø0.58mm | FL: 5.0mm | 2F" in res_fanuc.gcode

    # Check tool change comments
    assert "T1 M6 (Tool: T01 - Face Mill | Ø50mm | FL: 15.0mm | 4F)" in res_fanuc.gcode
    assert "T2 M6 (Tool: T02 - Drill | Ø0.58mm | FL: 5.0mm | 2F)" in res_fanuc.gcode

    # Test GRBL post
    grbl_post = GrblPostProcessor(work_offset="G54")
    res_grbl = grbl_post.post_process([tp1, tp2], program_name="GRBL_DIM_TEST")

    assert "; TOOLS USED IN THIS PROGRAM:" in res_grbl.gcode
    assert "; T01: Face Mill | Ø50mm | FL: 15.0mm | 4F" in res_grbl.gcode
    assert "; T02: Drill | Ø0.58mm | FL: 5.0mm | 2F" in res_grbl.gcode
    assert "T1 M0 (Change to T01 - Face Mill | Ø50mm | FL: 15.0mm | 4F)" in res_grbl.gcode


def test_setup_sheet_automatic_tool_numbering():
    t_face = Tool(id="t_face_50", tool_number=15, type=ToolType.FACE_MILL, diameter=50.0, flute_length=15.0, overall_length=60.0)
    t_drill = Tool(id="t_drill_058", tool_number=33, type=ToolType.DRILL, diameter=0.58, flute_length=5.0, overall_length=35.0)

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([0.0, 0.0, 0.0]), bounds_max=np.array([10.0, 10.0, 5.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    
    machine = MachineConfig(
        id="m1", name="CNC Mill", machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={"X": MachineAxis("X", -500, 500, 10000, 3000), "Y": MachineAxis("Y", -500, 500, 10000, 3000), "Z": MachineAxis("Z", -300, 50, 8001, 2000)},
        spindle_min_rpm=100, spindle_max_rpm=15000, max_spindle_power_kw=5.0,
        controller="grbl", units=Units.MM, tool_change_position=np.array([0, 0, 50]),
        work_offsets={"G54": np.array([0, 0, 0])}, safe_retract_height=10.0,
    )
    
    ctx = PlanningContext(
        model=None, setups=[setup], machine=machine, tools=[t_face, t_drill],
        material=DEFAULT_MATERIALS["aluminum_6061"],
    )

    feat1 = MachiningFeature(id="f1", type=FeatureType.FACING_REGION, face_indices=[0], bounds_min=np.zeros(3), bounds_max=np.ones(3), depth=1.0, top_z=5.0, floor_z=4.0)
    feat2 = MachiningFeature(id="f2", type=FeatureType.HOLE, face_indices=[1], bounds_min=np.zeros(3), bounds_max=np.ones(3), depth=2.0, top_z=4.0, floor_z=2.0, diameter=0.58)

    params = CuttingParameters(spindle_rpm=8001, feed_rate=500, plunge_feed=200, ramp_feed=300, feed_per_tooth_mm=0.05, depth_of_cut_mm=1.0, stepover_mm=2.0, coolant=True, effective_diameter_mm=50.0, engagement_angle_deg=45.0, power_kw=1.0, torque_nm=2.0)

    op1 = PlannedOperation(id="op1", purpose=OpPurpose.FACING, feature=feat1, tool=t_face, params=params)
    op2 = PlannedOperation(id="op2", purpose=OpPurpose.DRILLING, feature=feat2, tool=t_drill, params=params)

    tp1 = make_test_toolpath("op1", "facing", "t_face_50", "f1")
    tp2 = make_test_toolpath("op2", "drilling", "t_drill_058", "f2")

    sheet = generate_setup_sheet("TEST_PROG", "PART_TEST", ctx, setup, [feat1, feat2], [op1, op2], [tp1, tp2])
    
    # Verify sequential numbers 1 and 2 in the setup sheet tool list
    assert sheet.tool_list[0]["tool_number"] == 1
    assert sheet.tool_list[1]["tool_number"] == 2
    # Verify operation tools are labeled T01 and T02
    assert sheet.operations[0]["tool"] == "T01"
    assert sheet.operations[1]["tool"] == "T02"
