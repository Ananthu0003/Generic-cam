"""Tests for Generic Tool-Length Compensation (G43 H...) Architecture.

Verifies:
1. Explicit length_offset_h on Tool model propagates to G43 H{h}.
2. Post-processor h_offset_map overrides default tool number assumption.
3. Strict explicit policy raises error when unmapped tools are encountered.
4. All controllers (Fanuc, Haas, LinuxCNC, GRBL) emit G43 H... on tool changes.
5. Cutter compensation D registers resolve from tool metadata or d_offset_map.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, ToolHolder, CamError
)
from cam_engine.post import (
    get_post_processor, FanucPostProcessor, HaasPostProcessor,
    LinuxCncPostProcessor, GrblPostProcessor
)
from cam_engine.toolpaths.semantic import (
    MotionSegment, MotionType, Toolpath, CoordSpace
)


def make_test_toolpath(
    op_id: str,
    purpose: str,
    tool_id: str,
    tool_num: int = 1,
    length_offset_h: int | None = None,
    diameter_offset_d: int | None = None,
    overall_length: float | None = 60.0,
) -> Toolpath:
    tp = Toolpath(operation_id=op_id, feature_id="f1", tool_id=tool_id, purpose=purpose, space=CoordSpace.SETUP)
    tool_info = {
        "id": tool_id,
        "tool_number": tool_num,
        "type": "flat_endmill",
        "diameter": 10.0,
        "flute_length": 25.0,
        "overall_length": overall_length,
        "flutes": 4,
        "length_offset_h": length_offset_h,
        "diameter_offset_d": diameter_offset_d,
    }
    tp.metadata["tool_info"] = tool_info

    # Add tool change segment
    tp.add(MotionSegment(
        motion_type=MotionType.TOOL_CHANGE,
        start=np.array([0.0, 0.0, 20.0]),
        end=np.array([0.0, 0.0, 20.0]),
        tool_id=tool_id,
        operation_id=op_id,
        feature_id="f1",
        metadata={
            "tool_number": tool_num,
            "length_offset_h": length_offset_h,
            "diameter_offset_d": diameter_offset_d,
            "overall_length": overall_length,
            "tool_info": tool_info,
        }
    ))
    # Rapid and cut segments
    tp.add(MotionSegment(
        motion_type=MotionType.RAPID,
        start=np.array([0.0, 0.0, 20.0]),
        end=np.array([10.0, 10.0, 5.0]),
        tool_id=tool_id,
        operation_id=op_id,
        feature_id="f1",
    ))
    # Entry with cutter comp
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([10.0, 10.0, 0.0]),
        end=np.array([20.0, 10.0, 0.0]),
        feed=1000.0,
        tool_id=tool_id,
        operation_id=op_id,
        feature_id="f1",
        metadata={"cutter_comp": {"side": "left", "action": "on", "d_register": diameter_offset_d or 1}}
    ))
    # Linear cut
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([20.0, 10.0, 0.0]),
        end=np.array([50.0, 10.0, 0.0]),
        feed=1000.0,
        tool_id=tool_id,
        operation_id=op_id,
        feature_id="f1",
    ))
    return tp


def test_tool_model_validation():
    """Verify Tool model validates length_offset_h and diameter_offset_d positive values."""
    # Valid tool
    t = Tool(
        id="t1", tool_number=4, type=ToolType.FLAT_ENDMILL,
        diameter=10.0, flute_length=20.0, overall_length=60.0,
        length_offset_h=14, diameter_offset_d=24
    )
    t.validate()

    # Invalid H offset <= 0
    t_bad_h = Tool(
        id="t2", tool_number=4, type=ToolType.FLAT_ENDMILL,
        diameter=10.0, flute_length=20.0, overall_length=60.0,
        length_offset_h=0
    )
    with pytest.raises(CamError) as exc:
        t_bad_h.validate()
    assert "length_offset_h must be a positive integer" in str(exc.value)

    # Invalid D offset <= 0
    t_bad_d = Tool(
        id="t3", tool_number=4, type=ToolType.FLAT_ENDMILL,
        diameter=10.0, flute_length=20.0, overall_length=60.0,
        diameter_offset_d=-1
    )
    with pytest.raises(CamError) as exc:
        t_bad_d.validate()
    assert "diameter_offset_d must be a positive integer" in str(exc.value)


def test_explicit_h_and_d_offsets_from_tool_model():
    """Verify explicit H offset (e.g. H14 for Tool T4) is used in G43 instead of assuming H4."""
    tp = make_test_toolpath("op_rough", "roughing", "T04", tool_num=4, length_offset_h=14, diameter_offset_d=24)
    
    post = FanucPostProcessor(work_offset="G54", renumber_tools=False)
    res = post.post_process([tp], program_name="FANUC_H_TEST")
    
    gcode = res.gcode
    # Tool change T4
    assert "T4 M6" in gcode
    # Must emit G43 H14 (NOT H4!)
    assert "G43 H14" in gcode
    # Must emit G41 D24 (NOT D4!)
    assert "G41 D24" in gcode


def test_post_processor_h_offset_map_override():
    """Verify post processor h_offset_map overrides default tool number matching."""
    tp1 = make_test_toolpath("op1", "facing", "t_face", tool_num=1)
    tp2 = make_test_toolpath("op2", "roughing", "t_endmill", tool_num=2)

    custom_h_map = {"t_face": 31, "t_endmill": 42}
    custom_d_map = {"t_face": 51, "t_endmill": 62}

    post = HaasPostProcessor(
        work_offset="G54",
        renumber_tools=True,
        h_offset_map=custom_h_map,
        d_offset_map=custom_d_map,
    )
    res = post.post_process([tp1, tp2], program_name="HAAS_H_MAP")
    gcode = res.gcode

    assert "T1 M6" in gcode
    assert "G43 H31" in gcode
    assert "G41 D51" in gcode

    assert "T2 M6" in gcode
    assert "G43 H42" in gcode
    assert "G41 D62" in gcode


def test_strict_explicit_policy_raises_on_unmapped_tool():
    """Verify strict_explicit policy raises ValueError when an unmapped tool is used."""
    tp = make_test_toolpath("op1", "facing", "t_unmapped", tool_num=1, length_offset_h=None)

    post = FanucPostProcessor(work_offset="G54", h_offset_policy="strict_explicit")
    with pytest.raises(ValueError) as exc:
        post.post_process([tp], program_name="FAIL_PROG")
    assert "has no explicit H length offset defined" in str(exc.value)


def test_all_controllers_emit_g43_h():
    """Verify Fanuc, Haas, LinuxCNC, and GRBL emit G43 H... on tool changes."""
    tp = make_test_toolpath("op1", "facing", "T01", tool_num=1, length_offset_h=10)

    for controller in ["fanuc", "haas", "linuxcnc", "grbl"]:
        post = get_post_processor(controller=controller, work_offset="G54")
        res = post.post_process([tp], program_name=f"{controller.upper()}_TEST")
        assert "G43 H10" in res.gcode, f"Controller {controller} missing G43 H10"


def test_multi_tool_gcode_header_dimensions_and_offsets():
    """Verify tool dimensions including overall length (TL) are reported in program header."""
    tp1 = make_test_toolpath("op1", "facing", "T01", tool_num=1, length_offset_h=1, overall_length=75.0)
    tp2 = make_test_toolpath("op2", "profiling", "T02", tool_num=2, length_offset_h=2, overall_length=90.0)

    post = FanucPostProcessor(work_offset="G54")
    res = post.post_process([tp1, tp2], program_name="TEST_MULTI")
    gcode = res.gcode

    assert "TL: 75.0mm" in gcode
    assert "TL: 90.0mm" in gcode
    assert "G43 H1" in gcode
    assert "G43 H2" in gcode
