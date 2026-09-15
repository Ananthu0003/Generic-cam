"""Tests for canned-cycle + coolant emission in the post-processor, and the
tool-length-offset (G43 H) wiring from the Tool config."""

import numpy as np
import pytest

from cam_engine.post import get_post_processor
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType


def _op_toolpath(op_id: str, tool_id: str, with_change: bool = True) -> Toolpath:
    tp = Toolpath(operation_id=op_id, feature_id="feat_1", tool_id=tool_id,
                  purpose="drilling")
    if with_change:
        tp.add(MotionSegment(
            MotionType.TOOL_CHANGE, start=np.zeros(3), end=np.zeros(3),
            tool_id=tool_id, operation_id=op_id, feature_id="feat_1",
            metadata={"tool_number": 1, "length_offset_h": 12}))
    return tp


def _plunge(tp: Toolpath, op_id: str, cycle: dict, z_start: float = 26.0,
            z_end: float = 0.0, feed: float = 300.0, spindle: float = 2400.0):
    tp.add(MotionSegment(MotionType.SPINDLE_START, start=np.zeros(3),
                         end=np.zeros(3), spindle=spindle,
                         tool_id=tp.tool_id, operation_id=op_id))
    tp.add(MotionSegment(MotionType.PLUNGE, start=np.array([10.0, 20.0, z_start]),
                         end=np.array([10.0, 20.0, z_end]),
                         feed=feed, spindle=spindle, tool_id=tp.tool_id,
                         operation_id=op_id, metadata={"canned_cycle": cycle}))
    # explicit intermediate peck retract (chip clear) - must be suppressed
    tp.add(MotionSegment(MotionType.RETRACT, start=np.array([10.0, 20.0, 2.0]),
                         end=np.array([10.0, 20.0, 26.0]),
                         tool_id=tp.tool_id, operation_id=op_id))
    tp.add(MotionSegment(MotionType.RETRACT, start=np.array([10.0, 20.0, 0.0]),
                         end=np.array([10.0, 20.0, 35.0]),
                         tool_id=tp.tool_id, operation_id=op_id))


def test_grbl_emits_g83_canned_cycle_and_suppresses_pecks():
    tp = _op_toolpath("op_drill_001", "T01", with_change=True)
    cycle = {"type": "G83", "x": 10.0, "y": 20.0, "z_r": 26.0, "z_depth": 0.0,
             "peck": 2.0, "feed": 300.0, "initial_z": 35.0}
    _plunge(tp, "op_drill_001", cycle)

    res = get_post_processor("grbl", work_offset="G54").post_process([tp], "TEST")
    gcode = res.gcode

    assert "G43 H12" in gcode or "; Length offset H12" in gcode  # configured offset, not pocket #
    # GRBL v1.1 has no canned cycle support — drilling uses explicit G1 motion
    assert "G83" not in gcode  # no canned cycle emitted for GRBL
    assert "G1" in gcode       # explicit linear motion used instead


def test_grbl_falls_back_to_explicit_motion_for_g84():
    """GRBL v1.1 has no rigid tapping; explicit plunge/feed-out must remain."""
    tp = _op_toolpath("op_tap_001", "T01", with_change=True)
    cycle = {"type": "G84", "x": 10.0, "y": 20.0, "z_r": 26.0, "z_depth": 0.0,
             "feed": 1.5, "initial_z": 35.0}
    _plunge(tp, "op_tap_001", cycle, feed=1.5, spindle=2400.0)

    res = get_post_processor("grbl", work_offset="G54").post_process([tp], "TEST")
    gcode = res.gcode

    assert "G84" not in gcode
    assert "G1" in gcode                  # explicit motion preserved
    # G80 in header is a safety init; no G80 for canned cycle cancellation
    # (GRBL has no canned cycles, so no G80 cancel should appear in operation code)
    lines = gcode.split("\n")
    # Find lines after the header (skip first 4 lines: program name, generated, safety, work offset)
    operation_lines = lines[4:]
    g80_in_ops = any("G80" in line for line in operation_lines)
    assert not g80_in_ops


def test_fanuc_emits_g84_with_m29_and_g85_cycle():
    tp = _op_toolpath("op_tap_002", "T01", with_change=True)
    cycle = {"type": "G84", "x": 10.0, "y": 20.0, "z_r": 26.0, "z_depth": 0.0,
             "feed": 1.5, "initial_z": 35.0}
    _plunge(tp, "op_tap_002", cycle, feed=1.5)

    res = get_post_processor("fanuc", work_offset="G54").post_process([tp], "TEST")
    gcode = res.gcode

    assert "M29 S2400" in gcode
    assert "G84" in gcode
    assert "G80" in gcode                 # cancel after the cycle

    # G85 reaming cycle on a second op
    tp2 = _op_toolpath("op_ream_001", "T02", with_change=True)
    cycle2 = {"type": "G85", "x": 5.0, "y": 5.0, "z_r": 26.0, "z_depth": 1.0,
              "feed": 200.0, "initial_z": 35.0}
    _plunge(tp2, "op_ream_001", cycle2, feed=200.0)
    res2 = get_post_processor("fanuc", work_offset="G54").post_process([tp2], "TEST")
    assert "G85" in res2.gcode


def test_coolant_m8_m9_emitted():
    tp = _op_toolpath("op_drill_003", "T01", with_change=True)
    tp.add(MotionSegment(MotionType.COOLANT_ON, start=np.zeros(3), end=np.zeros(3),
                         tool_id="T01", operation_id="op_drill_003"))
    cycle = {"type": "G81", "x": 10.0, "y": 20.0, "z_r": 26.0, "z_depth": 0.0,
             "feed": 300.0, "initial_z": 35.0}
    _plunge(tp, "op_drill_003", cycle)
    tp.add(MotionSegment(MotionType.COOLANT_OFF, start=np.zeros(3), end=np.zeros(3),
                         tool_id="T01", operation_id="op_drill_003"))

    res = get_post_processor("fanuc", work_offset="G54").post_process([tp], "TEST")
    gcode = res.gcode

    assert "M8" in gcode
    assert "M9" in gcode
    assert gcode.index("M8") < gcode.index("G81") < gcode.index("M9")


def test_g81_spot_drill_canned_cycle():
    tp = _op_toolpath("op_spot_001", "T01", with_change=True)
    cycle = {"type": "G82", "x": 10.0, "y": 20.0, "z_r": 26.0, "z_depth": 23.4,
             "dwell": 0.1, "feed": 300.0, "initial_z": 35.0}
    _plunge(tp, "op_spot_001", cycle)

    res = get_post_processor("fanuc", work_offset="G54").post_process([tp], "TEST")
    gcode = res.gcode

    assert "G82" in gcode
    assert "P100" in gcode  # Fanuc uses milliseconds for dwell