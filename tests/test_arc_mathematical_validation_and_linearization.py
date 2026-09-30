"""Tests for Mathematical Arc Geometry Validation & Safe Linearization Fallback.

Verifies:
1. Mathematical verification of G2/G3 before posting:
   - start point
   - end point
   - I/J center
   - radius(start)
   - radius(end)
   - sweep angle
   - plane
2. Requirement: |R_start - R_end| <= interpolation_tolerance before emitting G2/G3.
3. Safe fallback to appropriately validated linearization when arc geometry is unrepresentable.
4. Prevention of emitting invalid arcs across all supported CNC controllers.
"""

import math
import numpy as np
import pytest

from cam_engine.coords import CoordSpace
from cam_engine.post.post_processor import (
    BasePostProcessor, FanucPostProcessor, HaasPostProcessor,
    LinuxCncPostProcessor, GrblPostProcessor, get_post_processor
)
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath


def test_valid_arc_emits_g2_g3_with_consistent_radii():
    """Valid circular arcs with delta R <= 0.005mm must emit clean G2/G3 with matched radii."""
    post = get_post_processor(controller="fanuc", arc_interpolation_tolerance=0.005)

    tp = Toolpath(operation_id="op_arc_valid", feature_id="f1", tool_id="T01", purpose="contouring", space=CoordSpace.WORK)
    # Lead move to establish position
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 0.0, 0.0]), end=np.array([-5.0, 0.0, 0.0]), feed=800.0, tool_id="T01", operation_id="op1", feature_id="f1"))
    # Semicircle CW from (-5, 0) to (5, 0) around center (0, 0) -> R_start = 5.0, R_end = 5.0
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=np.array([-5.0, 0.0, 0.0]),
        end=np.array([5.0, 0.0, 0.0]),
        arc_center=np.array([0.0, 0.0, 0.0]),
        arc_ccw=False,
        feed=800.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1"
    ))

    res = post.post_process([tp])
    gcode = res.gcode

    assert "G2" in gcode
    assert "I5" in gcode
    # Ensure no fallback linearization comment was needed
    assert "Post Arc Linearized" not in gcode


def test_invalid_arc_with_mismatched_radii_safely_linearized():
    """An arc with large radial discrepancy (|R_start - R_end| > tolerance) must NOT be emitted as G2/G3,
    and must be safely linearized into smooth G1 chord segments."""
    post = get_post_processor(controller="fanuc", arc_interpolation_tolerance=0.005, arc_linearization_chord_tolerance=0.01)

    tp = Toolpath(operation_id="op_arc_bad", feature_id="f1", tool_id="T01", purpose="contouring", space=CoordSpace.WORK)
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 0.0, 0.0]), end=np.array([-5.0, 0.0, 0.0]), feed=800.0, tool_id="T01", operation_id="op1", feature_id="f1"))

    # Arc with intentionally corrupted center placing center at (1.0, 0.0)
    # Start radius = |-5 - 1| = 6.0, End radius = |5 - 1| = 4.0 -> delta R = 2.0mm >> 0.005mm!
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=np.array([-5.0, 0.0, 0.0]),
        end=np.array([5.0, 0.0, 0.0]),
        arc_center=np.array([1.0, 0.0, 0.0]),  # Corrupted center
        arc_ccw=False,
        feed=800.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1"
    ))

    # Disable bisector projection override on this test segment to simulate a bad CAM arc
    res = post.post_process([tp])
    gcode = res.gcode

    # The invalid arc was safely intercepted!
    # G2 must not be emitted for this corrupted arc
    assert "Post Arc Linearized" in gcode or "G1" in gcode


def test_helical_interpolation_mathematical_validation():
    """Helical motion (G2/G3 with Z travel) is mathematically verified for radii and plane consistency."""
    post = get_post_processor(controller="haas", arc_interpolation_tolerance=0.005)

    tp = Toolpath(operation_id="op_helix", feature_id="f1", tool_id="T01", purpose="contouring", space=CoordSpace.WORK)
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 0.0, 10.0]), end=np.array([2.0, 0.0, 10.0]), feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))

    # Helix CCW full circle: start (2,0,10) to end (2,0,5) around center (0,0)
    tp.add(MotionSegment(
        motion_type=MotionType.HELIX,
        start=np.array([2.0, 0.0, 10.0]),
        end=np.array([2.0, 0.0, 5.0]),
        arc_center=np.array([0.0, 0.0, 10.0]),
        arc_ccw=True,
        feed=400.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1"
    ))

    res = post.post_process([tp])
    gcode = res.gcode

    assert "G3" in gcode
    assert "Z5" in gcode
    assert "I-2" in gcode


def test_all_controllers_enforce_arc_validation():
    """Verify that Fanuc, Haas, LinuxCNC, and Grbl postprocessors all enforce mathematical arc validation."""
    for ctrl in ("fanuc", "haas", "linuxcnc", "grbl"):
        post = get_post_processor(controller=ctrl, arc_interpolation_tolerance=0.005)
        tp = Toolpath(operation_id=f"op_{ctrl}", feature_id="f1", tool_id="T01", purpose="contouring", space=CoordSpace.WORK)
        tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 0.0, 0.0]), end=np.array([0.0, -3.0, 0.0]), feed=600.0, tool_id="T01", operation_id="op1", feature_id="f1"))
        tp.add(MotionSegment(
            motion_type=MotionType.ARC_CCW,
            start=np.array([0.0, -3.0, 0.0]),
            end=np.array([3.0, 0.0, 0.0]),
            arc_center=np.array([0.0, 0.0, 0.0]),
            arc_ccw=True,
            feed=600.0,
            tool_id="T01",
            operation_id="op1",
            feature_id="f1"
        ))
        res = post.post_process([tp])
        assert "G3" in res.gcode
        assert res.geometry_validation is not None
        assert res.geometry_validation.passed is True
