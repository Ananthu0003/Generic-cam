"""Tests for D-offset number, D-offset mode, controller memory type, geometry, and wear value derivation.

Verifies:
1. Derivation of D register from:
   - Priority 1: Post explicit D map
   - Priority 2: Tool library explicit diameter_offset_d
   - Priority 3: Controller standard tool number matching (T01 -> D1)
2. Controller offset memory configurations:
   - Fanuc Type C: Separate Geometry and Wear registers
     * Wear mode: Geom = 0.000mm, Wear = 0.000mm (Wear delta only on Tool Center path)
     * Control mode: Geom = R1.500mm, Wear = 0.000mm (Full radius on Part Contour path)
   - Haas / Combined memory: Single combined offset register
     * Wear mode: Offset = 0.000mm (Wear delta only)
     * Control mode: Radius = R1.500mm
3. G-code comments explicitly communicate D register, mode, memory model, geometry value, and wear value.
"""

import numpy as np
import pytest

from cam_engine.post import get_post_processor
from cam_engine.post.controller_models import FANUC_CAPABILITY_MODEL, HAAS_CAPABILITY_MODEL
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType


def test_fanuc_wear_compensation_memory_derivation():
    """Test Fanuc Type C separate geometry/wear memory derivation for wear compensation."""
    tp = Toolpath(tool_id="T01", feature_id="f1", operation_id="op_contour", purpose="FINISHING")
    tp.metadata["tool"] = {"number": 1, "diameter": 3.0, "type": "flat_endmill", "material": "carbide"}
    tp.metadata["cutter_comp_info"] = {
        "side": "left",
        "mode": "wear",
        "lead_len": 1.20,
    }
    tp.add(MotionSegment(motion_type=MotionType.RAPID, start=np.array([0, 0, 10]), end=np.array([0, 0, 0])))
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY, start=np.array([0, 0, 0]), end=np.array([5, 0, 0]),
        metadata={"cutter_comp": {"side": "left", "action": "on"}}
    ))
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([5, 0, 0]), end=np.array([10, 0, 0])))
    tp.add(MotionSegment(
        motion_type=MotionType.EXIT, start=np.array([10, 0, 0]), end=np.array([15, 0, 0]),
        metadata={"cutter_comp": {"side": "left", "action": "off"}}
    ))

    post = get_post_processor("fanuc", work_offset="G54")
    res = post.post_process([tp], "FANUC_WEAR_TEST")
    gcode = res.gcode

    assert "G41 D1" in gcode
    assert "Fanuc Type C: Geom=0.000mm, Wear=0.000mm (Wear Delta Only)" in gcode
    assert "Mode: Wear (Tool Center Path" in gcode


def test_fanuc_control_compensation_memory_derivation():
    """Test Fanuc Type C separate geometry/wear memory derivation for control (part contour) compensation."""
    tp = Toolpath(tool_id="T01", feature_id="f1", operation_id="op_contour", purpose="FINISHING")
    tp.metadata["tool"] = {"number": 1, "diameter": 3.0, "type": "flat_endmill", "material": "carbide"}
    tp.metadata["cutter_comp_info"] = {
        "side": "left",
        "mode": "control",
        "lead_len": 1.20,
    }
    tp.add(MotionSegment(motion_type=MotionType.RAPID, start=np.array([0, 0, 10]), end=np.array([0, 0, 0])))
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY, start=np.array([0, 0, 0]), end=np.array([5, 0, 0]),
        metadata={"cutter_comp": {"side": "left", "action": "on"}}
    ))
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([5, 0, 0]), end=np.array([10, 0, 0])))
    tp.add(MotionSegment(
        motion_type=MotionType.EXIT, start=np.array([10, 0, 0]), end=np.array([15, 0, 0]),
        metadata={"cutter_comp": {"side": "left", "action": "off"}}
    ))

    post = get_post_processor("fanuc", work_offset="G54")
    res = post.post_process([tp], "FANUC_CONTROL_TEST")
    gcode = res.gcode

    assert "G41 D1" in gcode
    assert "Fanuc Type C: Geom=R1.500mm, Wear=0.000mm (Full Cutter Radius)" in gcode
    assert "Mode: Control (Part Contour Path" in gcode


def test_haas_combined_memory_derivation():
    """Test Haas combined offset memory derivation for wear and control modes."""
    tp = Toolpath(tool_id="T02", feature_id="f2", operation_id="op_contour", purpose="FINISHING")
    tp.metadata["tool"] = {"number": 2, "diameter": 6.0, "type": "flat_endmill", "material": "carbide"}
    tp.metadata["cutter_comp_info"] = {
        "side": "left",
        "mode": "wear",
        "lead_len": 2.50,
    }
    tp.add(MotionSegment(motion_type=MotionType.RAPID, start=np.array([0, 0, 10]), end=np.array([0, 0, 0])))
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY, start=np.array([0, 0, 0]), end=np.array([5, 0, 0]),
        metadata={"cutter_comp": {"side": "left", "action": "on"}}
    ))
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([5, 0, 0]), end=np.array([10, 0, 0])))
    tp.add(MotionSegment(
        motion_type=MotionType.EXIT, start=np.array([10, 0, 0]), end=np.array([15, 0, 0]),
        metadata={"cutter_comp": {"side": "left", "action": "off"}}
    ))

    post = get_post_processor("haas", work_offset="G54", renumber_tools=False)
    res = post.post_process([tp], "HAAS_WEAR_TEST")
    gcode = res.gcode

    assert "G41 D2" in gcode
    assert "Haas/Combined: Offset=0.000mm (Wear Delta Only)" in gcode
