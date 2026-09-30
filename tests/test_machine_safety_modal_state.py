"""Tests for Explicit Machine Safety State and Deterministic Modal Management.

Verifies:
1. Every operation begins with a deterministic safety / modal initialization block.
2. Plane (G17), Distance mode (G90), Feed mode (G94), Units (G21), Cutter comp cancel (G40),
   and Canned cycle cancel (G80) are deterministically managed.
3. Canned cycles (G81-G85) in prior operations do not persist into subsequent milling operations.
4. Cutter compensation (G41/G42) in prior operations is cancelled (G40) across operation boundaries.
5. Work Coordinate System (WCS, e.g. G54 -> G55) is explicitly asserted per operation.
6. Tool length compensation (G43 H...) and spindle state (S... M3) are deterministically established.
7. Controller-specific dialects (Fanuc, Haas, LinuxCNC, GRBL) format safety preambles correctly.
"""

import numpy as np
import pytest

from cam_engine.post import (
    get_post_processor, FanucPostProcessor, HaasPostProcessor,
    LinuxCncPostProcessor, GrblPostProcessor
)
from cam_engine.toolpaths.semantic import (
    MotionSegment, MotionType, Toolpath, CoordSpace
)


def _build_test_toolpath(
    op_id: str,
    tool_id: str,
    purpose: str = "roughing",
    work_offset: str = "G54",
    tool_num: int = 1,
    length_offset_h: int = 1,
    spindle_rpm: float = 8001.0,
    feed: float = 600.0,
    with_canned_cycle: dict | None = None,
    with_cutter_comp: dict | None = None,
) -> Toolpath:
    """Helper to create a well-formed test toolpath with metadata and motion segments."""
    tp = Toolpath(operation_id=op_id, feature_id=f"feat_{op_id}", tool_id=tool_id, purpose=purpose, space=CoordSpace.SETUP)
    tool_info = {
        "id": tool_id,
        "tool_number": tool_num,
        "type": "drill" if "drill" in purpose else "flat_endmill",
        "diameter": 6.0 if "drill" in purpose else 10.0,
        "flute_length": 25.0,
        "overall_length": 60.0,
        "flutes": 2 if "drill" in purpose else 4,
        "length_offset_h": length_offset_h,
        "diameter_offset_d": tool_num,
    }
    tp.metadata["tool_info"] = tool_info
    tp.metadata["work_offset"] = work_offset
    tp.metadata["cutting_parameters"] = {
        "material_name": "Aluminum 6061-T6",
        "spindle_rpm": spindle_rpm,
        "feed_rate": feed,
        "cutting_speed_m_min": 150.0,
        "feed_per_tooth_mm": 0.05,
    }

    # Tool change segment
    tp.add(MotionSegment(
        motion_type=MotionType.TOOL_CHANGE,
        start=np.array([0.0, 0.0, 25.0]),
        end=np.array([0.0, 0.0, 25.0]),
        tool_id=tool_id,
        operation_id=op_id,
        metadata={"tool_number": tool_num, "length_offset_h": length_offset_h, "tool_info": tool_info}
    ))

    # Rapid to clearance
    tp.add(MotionSegment(
        motion_type=MotionType.RAPID,
        start=np.array([0.0, 0.0, 25.0]),
        end=np.array([10.0, 10.0, 5.0]),
        tool_id=tool_id,
        operation_id=op_id,
    ))

    if with_canned_cycle:
        # Canned cycle plunge
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=np.array([10.0, 10.0, 2.0]),
            end=np.array([10.0, 10.0, -10.0]),
            feed=feed,
            spindle=spindle_rpm,
            tool_id=tool_id,
            operation_id=op_id,
            metadata={"canned_cycle": with_canned_cycle}
        ))
    elif with_cutter_comp:
        # Comp entry
        tp.add(MotionSegment(
            motion_type=MotionType.ENTRY,
            start=np.array([10.0, 10.0, 0.0]),
            end=np.array([20.0, 10.0, 0.0]),
            feed=feed,
            tool_id=tool_id,
            operation_id=op_id,
            metadata={"cutter_comp": with_cutter_comp}
        ))
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([20.0, 10.0, 0.0]),
            end=np.array([50.0, 10.0, 0.0]),
            feed=feed,
            tool_id=tool_id,
            operation_id=op_id,
        ))
    else:
        # Standard linear cut
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([10.0, 10.0, 0.0]),
            end=np.array([40.0, 10.0, 0.0]),
            feed=feed,
            tool_id=tool_id,
            operation_id=op_id,
        ))

    # Retract
    tp.add(MotionSegment(
        motion_type=MotionType.RETRACT,
        start=np.array([40.0, 10.0, 0.0]),
        end=np.array([40.0, 10.0, 25.0]),
        tool_id=tool_id,
        operation_id=op_id,
    ))

    return tp


def test_operation_boundary_modal_safety_block():
    """Verify that consecutive operations with or without tool change each emit deterministic safety blocks."""
    # Op 1: Facing with T01
    tp1 = _build_test_toolpath("op_face_001", "T01", purpose="facing", tool_num=1, length_offset_h=1, spindle_rpm=6000.0)
    # Op 2: Pocket Roughing with same T01
    tp2 = _build_test_toolpath("op_pocket_001", "T01", purpose="roughing", tool_num=1, length_offset_h=1, spindle_rpm=7500.0)

    post = FanucPostProcessor(work_offset="G54")
    res = post.post_process([tp1, tp2], program_name="SAFETY_TEST")
    gcode = res.gcode

    # Both operations must have their own deterministic safe block: G00 G90 G17 G21 G94 G40 G80 G54
    lines = gcode.splitlines()
    op1_idx = next(i for i, l in enumerate(lines) if "op_face_001" in l)
    op2_idx = next(i for i, l in enumerate(lines) if "op_pocket_001" in l)

    # Check safe block in op1
    op1_block = "\n".join(lines[op1_idx:op1_idx + 10])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in op1_block
    assert "G43 H1" in op1_block
    assert "S6000 M3" in op1_block

    # Check safe block in op2 (even though tool did not change!)
    op2_block = "\n".join(lines[op2_idx:op2_idx + 10])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in op2_block
    assert "S7500 M3" in op2_block


def test_canned_cycle_isolation_across_operations():
    """Verify that a canned drilling cycle (G83) in Op 1 is cancelled (G80) before Op 2 starts."""
    cycle = {"type": "G83", "x": 10.0, "y": 10.0, "z_r": 2.0, "z_depth": -10.0, "peck": 2.0, "feed": 300.0}
    tp_drill = _build_test_toolpath("op_drill_001", "T02", purpose="drilling", tool_num=2, length_offset_h=2, with_canned_cycle=cycle)
    tp_mill = _build_test_toolpath("op_contour_001", "T01", purpose="finishing", tool_num=1, length_offset_h=1)

    post = FanucPostProcessor(work_offset="G54")
    res = post.post_process([tp_drill, tp_mill], program_name="CYCLE_TEST")
    gcode = res.gcode

    lines = gcode.splitlines()
    drill_idx = next(i for i, l in enumerate(lines) if "op_drill_001" in l)
    mill_idx = next(i for i, l in enumerate(lines) if "op_contour_001" in l)

    # In drill section, G83 must be emitted
    drill_section = "\n".join(lines[drill_idx:mill_idx])
    assert "G83" in drill_section

    # Before or at mill op boundary, G80 cancellation must appear
    assert "G80" in drill_section
    mill_section = "\n".join(lines[mill_idx:mill_idx + 10])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in mill_section


def test_cutter_compensation_isolation_across_operations():
    """Verify that active cutter compensation (G41) is cancelled (G40) across operation boundaries."""
    comp_on = {"action": "on", "side": "left", "d_register": 1}
    tp_comp = _build_test_toolpath("op_finish_comp", "T01", purpose="finishing", tool_num=1, length_offset_h=1, with_cutter_comp=comp_on)
    tp_next = _build_test_toolpath("op_chamfer", "T03", purpose="chamfering", tool_num=3, length_offset_h=3)

    post = HaasPostProcessor(work_offset="G54")
    res = post.post_process([tp_comp, tp_next], program_name="COMP_TEST")
    gcode = res.gcode

    lines = gcode.splitlines()
    comp_idx = next(i for i, l in enumerate(lines) if "op_finish_comp" in l)
    next_idx = next(i for i, l in enumerate(lines) if "op_chamfer" in l)

    comp_section = "\n".join(lines[comp_idx:next_idx])
    assert "G41 D1" in comp_section
    # G40 must be emitted before tool change and in next op preamble
    assert "G40" in comp_section

    next_section = "\n".join(lines[next_idx:next_idx + 10])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in next_section
    assert "G43 H3" in next_section


def test_multi_wcs_modal_independence():
    """Verify that switching from Setup 1 (G54) to Setup 2 (G55) explicitly establishes WCS."""
    tp_setup1 = _build_test_toolpath("op_s1_rough", "T01", purpose="roughing", work_offset="G54", tool_num=1, length_offset_h=1)
    tp_setup2 = _build_test_toolpath("op_s2_rough", "T01", purpose="roughing", work_offset="G55", tool_num=1, length_offset_h=1)

    post = LinuxCncPostProcessor(work_offset="G54")
    res = post.post_process([tp_setup1, tp_setup2], program_name="MULTI_WCS_TEST")
    gcode = res.gcode

    lines = gcode.splitlines()
    s1_idx = next(i for i, l in enumerate(lines) if "op_s1_rough" in l)
    s2_idx = next(i for i, l in enumerate(lines) if "op_s2_rough" in l)

    s1_block = "\n".join(lines[s1_idx:s1_idx + 10])
    assert "G90 G94 G17 G21 G40 G49 G80 G54" in s1_block

    s2_block = "\n".join(lines[s2_idx:s2_idx + 10])
    assert "G90 G94 G17 G21 G40 G49 G80 G55" in s2_block


def test_all_controllers_safety_preambles():
    """Verify dialect-specific modal safety lines across Fanuc, Haas, LinuxCNC, and GRBL."""
    tp = _build_test_toolpath("op_001", "T01", work_offset="G54", tool_num=1, length_offset_h=1)

    # Fanuc
    fanuc_res = FanucPostProcessor(work_offset="G54").post_process([tp])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in fanuc_res.gcode

    # Haas
    haas_res = HaasPostProcessor(work_offset="G54").post_process([tp])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in haas_res.gcode

    # LinuxCNC
    linux_res = LinuxCncPostProcessor(work_offset="G54").post_process([tp])
    assert "G90 G94 G17 G21 G40 G49 G80 G54" in linux_res.gcode

    # GRBL
    grbl_res = GrblPostProcessor(work_offset="G54").post_process([tp])
    assert "G54 G90 G17 G21 G94 G40" in grbl_res.gcode
