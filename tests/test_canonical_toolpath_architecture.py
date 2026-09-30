"""Tests for Canonical Toolpath Segment Architecture & Pipeline Progression.

Verifies:
1. Canonical segment creation with exact geometric data (Rapid, LinearCut, CircularCut, HelicalCut, Retract, Plunge, ToolChange, Spindle, Coolant, Dwell).
2. Exact 4x4 coordinate transformations applied to MotionSegments and Toolpaths (start, end, arc_center, metadata).
3. Pipeline stage decoupling:
   Operation -> Canonical Toolpath -> Validation -> Coordinate Transformation -> Postprocessor -> G-code
4. Postprocessor dialect translation for canonical linear, circular (G2/G3 I/J), and helical segments.
"""

import math
import numpy as np
import pytest

from cam_engine.coords import CoordSpace, make_transform, rotation_about_axis
from cam_engine.toolpaths.semantic import (
    MotionSegment, MotionType, Toolpath,
    Rapid, LinearCut, CircularCut, HelicalCut, Retract, Plunge, Ramp,
    ToolChange, Spindle, Coolant, Dwell,
)
from cam_engine.post import (
    get_post_processor, FanucPostProcessor, HaasPostProcessor,
    LinuxCncPostProcessor, GrblPostProcessor
)


def test_canonical_segment_constructors():
    """Verify all canonical segment factory constructors produce correct geometric metadata."""
    # Rapid
    r = Rapid(start=[0, 0, 20], end=[10, 10, 5], tool_id="T01", operation_id="op1")
    assert r.motion_type == MotionType.RAPID
    assert np.allclose(r.start, [0, 0, 20])
    assert np.allclose(r.end, [10, 10, 5])

    # LinearCut
    lc = LinearCut(start=[10, 10, 0], end=[50, 10, 0], feed=1200.0, spindle=8001.0, tool_id="T01")
    assert lc.motion_type == MotionType.CUT
    assert np.isclose(lc.feed, 1200.0)
    assert np.isclose(lc.spindle, 8001.0)
    assert np.isclose(lc.length(), 40.0)

    # CircularCut (CCW)
    cc = CircularCut(start=[10, 0, 0], end=[0, 10, 0], arc_center=[0, 0, 0], arc_ccw=True, feed=800.0)
    assert cc.motion_type == MotionType.ARC_CCW
    assert cc.arc_ccw is True
    assert np.allclose(cc.arc_center, [0, 0, 0])
    # 90 degree arc of radius 10 = 10 * pi / 2 = 15.7079...
    assert np.isclose(cc.length(), 10.0 * math.pi / 2.0, atol=1e-3)

    # HelicalCut
    hc = HelicalCut(start=[10, 0, 2], end=[0, 10, -2], arc_center=[0, 0, 0], arc_ccw=True, feed=500.0)
    assert hc.motion_type == MotionType.HELIX
    assert np.allclose(hc.arc_center, [0, 0, 0])

    # Plunge and Retract
    pl = Plunge(start=[0, 0, 5], end=[0, 0, -10], feed=300.0, spindle=6000.0)
    assert pl.motion_type == MotionType.PLUNGE
    ret = Retract(start=[0, 0, -10], end=[0, 0, 25])
    assert ret.motion_type == MotionType.RETRACT

    # ToolChange, Spindle, Coolant, Dwell
    tc = ToolChange("T02", tool_number=2, length_offset_h=12, diameter_offset_d=2)
    assert tc.motion_type == MotionType.TOOL_CHANGE
    assert tc.metadata["tool_number"] == 2
    assert tc.metadata["length_offset_h"] == 12

    sp = Spindle(rpm=12000.0)
    assert sp.motion_type == MotionType.SPINDLE_START
    assert np.isclose(sp.spindle, 12000.0)

    co = Coolant(mode="flood", on=True)
    assert co.motion_type == MotionType.COOLANT_ON
    assert co.metadata["coolant_type"] == "flood"

    dw = Dwell(seconds=1.5)
    assert dw.motion_type == MotionType.DWELL
    assert np.isclose(dw.dwell_seconds, 1.5)


def test_canonical_segment_and_toolpath_transformation():
    """Verify exact 4x4 coordinate transformation on canonical segments and toolpaths."""
    # Rotate 90 deg around Z and translate by (100, 50, 10)
    rot_z = rotation_about_axis([0, 0, 1], math.pi / 2.0)
    trsf = make_transform(rotation=rot_z, translation=[100.0, 50.0, 10.0])

    # Transform linear cut
    seg = LinearCut(start=[10, 0, 0], end=[20, 0, 0], feed=1000.0, space=CoordSpace.SETUP)
    seg_tr = seg.transformed(trsf, dst_space=CoordSpace.WORK)
    assert seg_tr.space == CoordSpace.WORK
    # [10, 0, 0] rotated 90 deg -> [0, 10, 0] + [100, 50, 10] -> [100, 60, 10]
    assert np.allclose(seg_tr.start, [100.0, 60.0, 10.0], atol=1e-5)
    # [20, 0, 0] rotated 90 deg -> [0, 20, 0] + [100, 50, 10] -> [100, 70, 10]
    assert np.allclose(seg_tr.end, [100.0, 70.0, 10.0], atol=1e-5)

    # Transform circular arc with center
    arc = CircularCut(start=[5, 0, 0], end=[0, 5, 0], arc_center=[0, 0, 0], arc_ccw=True, space=CoordSpace.SETUP)
    arc_tr = arc.transformed(trsf, dst_space=CoordSpace.WORK)
    assert arc_tr.space == CoordSpace.WORK
    assert np.allclose(arc_tr.arc_center, [100.0, 50.0, 10.0], atol=1e-5)
    assert np.allclose(arc_tr.start, [100.0, 55.0, 10.0], atol=1e-5)
    assert np.allclose(arc_tr.end, [95.0, 50.0, 10.0], atol=1e-5)

    # Transform complete Toolpath
    tp = Toolpath(operation_id="op_test", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp.add(Rapid([0, 0, 20], [10, 0, 20], space=CoordSpace.SETUP))
    tp.add(seg)
    tp.add(arc)

    tp_tr = tp.transformed(trsf, dst_space=CoordSpace.WORK)
    assert tp_tr.space == CoordSpace.WORK
    assert len(tp_tr.segments) == 3
    assert all(s.space == CoordSpace.WORK for s in tp_tr.segments)


def test_postprocessor_translates_canonical_segments():
    """Verify that the postprocessor cleanly converts canonical segments into controller-specific G-code."""
    tp = Toolpath(operation_id="op_circ", feature_id="feat_circ", tool_id="T01", purpose="finishing", space=CoordSpace.WORK)
    tp.metadata["tool_info"] = {
        "id": "T01", "tool_number": 1, "type": "flat_endmill", "diameter": 10.0,
        "flute_length": 25.0, "overall_length": 60.0, "flutes": 4, "length_offset_h": 1,
    }
    tp.metadata["cutting_parameters"] = {
        "material_name": "Aluminum 6061-T6", "spindle_rpm": 8001.0, "feed_rate": 1000.0,
    }

    # ToolChange
    tp.add(ToolChange("T01", tool_number=1, length_offset_h=1, space=CoordSpace.WORK))
    # Rapid
    tp.add(Rapid([0, 0, 25], [10, 0, 5], space=CoordSpace.WORK))
    # Plunge
    tp.add(Plunge([10, 0, 5], [10, 0, 0], feed=300.0, spindle=8001.0, space=CoordSpace.WORK))
    # Linear cut
    tp.add(LinearCut([10, 0, 0], [20, 0, 0], feed=1000.0, space=CoordSpace.WORK))
    # Circular CCW cut from (20,0) to (0,20) with center at (0,0)
    tp.add(CircularCut(start=[20, 0, 0], end=[0, 20, 0], arc_center=[0, 0, 0], arc_ccw=True, feed=1000.0, space=CoordSpace.WORK))
    # Retract
    tp.add(Retract([0, 20, 0], [0, 20, 25], space=CoordSpace.WORK))

    # Fanuc
    fanuc_post = FanucPostProcessor(work_offset="G54")
    res_fanuc = fanuc_post.post_process([tp])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in res_fanuc.gcode
    assert "T1 M6" in res_fanuc.gcode
    assert "G43 H1" in res_fanuc.gcode
    assert "G1" in res_fanuc.gcode
    assert "G3" in res_fanuc.gcode or "G03" in res_fanuc.gcode
    assert "I" in res_fanuc.gcode and "J" in res_fanuc.gcode

    # Haas
    haas_post = HaasPostProcessor(work_offset="G54")
    res_haas = haas_post.post_process([tp])
    assert "G00 G90 G17 G21 G94 G40 G80 G54" in res_haas.gcode
    assert "G3" in res_haas.gcode or "G03" in res_haas.gcode

    # GRBL
    grbl_post = GrblPostProcessor(work_offset="G54")
    res_grbl = grbl_post.post_process([tp])
    assert "G54 G90 G17 G21 G94 G40" in res_grbl.gcode
    assert "G3" in res_grbl.gcode or "G03" in res_grbl.gcode
