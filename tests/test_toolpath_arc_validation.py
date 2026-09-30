import math
import numpy as np
import pytest

from cam_engine.context import MachineConfig, PlanningContext, Setup, Tool, ToolType, Stock, StockKind, Material, WorkOffset
from cam_engine.demo_models import get_default_machine
from cam_engine.geometry import TriangleMesh
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath, CoordSpace
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.errors import INVALID_TOOLPATH


def _make_validator():
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=6.0, corner_radius=0.0, flute_length=20.0,
        overall_length=60.0, flutes=4, material="carbide", can_plunge=True
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-50, -50, -10]), bounds_max=np.array([50, 50, 20]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = Material(id="al", name="Aluminum", v_carbide=300, v_hss=90, hardness_hb=95,
                   max_hm_ratio=1.0, max_ae_ratio=0.4, specific_cutting_force_kn_mm2=0.8, needs_coolant=True)
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)
    mesh = TriangleMesh(triangles=np.zeros((1, 3, 3)))
    return ToolpathValidator(ctx, setup, mesh, clearance_z=20.0), tool


def test_valid_arc_and_contour_passes():
    """Test that a perfectly formed circular contour with shared center and radius passes validation."""
    validator, tool = _make_validator()
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id=tool.id, purpose="finishing")

    c = np.array([0.0, 0.0, 5.0])
    r = 10.0
    # 4 quadrant arcs around (0, 0)
    quads = [
        (np.array([r, 0.0, 5.0]), np.array([0.0, r, 5.0])),
        (np.array([0.0, r, 5.0]), np.array([-r, 0.0, 5.0])),
        (np.array([-r, 0.0, 5.0]), np.array([0.0, -r, 5.0])),
        (np.array([0.0, -r, 5.0]), np.array([r, 0.0, 5.0])),
    ]
    for s, e in quads:
        tp.add(MotionSegment(
            motion_type=MotionType.ARC_CCW,
            start=s, end=e, arc_center=c, arc_ccw=True,
            feed=1000.0, tool_id=tool.id, operation_id="op1", feature_id="f1",
            metadata={"contour_id": "c1"}
        ))

    report = validator.validate(tp)
    assert report.passed is True
    assert len(report.errors) == 0


def test_arc_radius_inconsistency_fails():
    """Test that an arc segment with mismatched start and end radii is rejected."""
    validator, tool = _make_validator()
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id=tool.id, purpose="roughing")

    # Arc start radius = 10.0, end radius = 10.05 (delta = 0.05 > 0.005mm limit)
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=np.array([10.0, 0.0, 0.0]),
        end=np.array([0.0, 10.05, 0.0]),
        arc_center=np.array([0.0, 0.0, 0.0]),
        arc_ccw=False,
        feed=500.0,
        tool_id=tool.id, operation_id="op1", feature_id="f1"
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("radius inconsistent" in e.message for e in report.errors)


def test_arc_chord_exceeds_diameter_fails():
    """Test that an impossible arc with chord > 2*R is rejected."""
    validator, tool = _make_validator()
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id=tool.id, purpose="roughing")

    # Radius = 5.0 (diameter = 10.0), chord distance = 14.14 > 10.0
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CCW,
        start=np.array([0.0, 5.0, 0.0]),
        end=np.array([10.0, 5.0, 0.0]),
        arc_center=np.array([0.0, 0.0, 0.0]),
        arc_ccw=True,
        feed=500.0,
        tool_id=tool.id, operation_id="op1", feature_id="f1",
        metadata={"arc_radius": 4.0}
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("chord length" in e.message for e in report.errors)


def test_degenerate_zero_radius_arc_fails():
    """Test that zero-radius arc motion is rejected."""
    validator, tool = _make_validator()
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id=tool.id, purpose="roughing")

    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CCW,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([0.0, 0.0, 0.0]),
        arc_center=np.array([0.0, 0.0, 0.0]),
        arc_ccw=True,
        feed=500.0,
        tool_id=tool.id, operation_id="op1", feature_id="f1"
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("degenerate zero radius" in e.message for e in report.errors)


def test_c0_positional_discontinuity_fails():
    """Test that spatial jump between uninterrupted cutting moves without retract/link is flagged."""
    validator, tool = _make_validator()
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id=tool.id, purpose="roughing")

    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([5.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id="op1", feature_id="f1"
    ))
    # Discontinuous start at (5.5, 0.0) with gap 0.5mm
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([5.5, 0.0, 0.0]),
        end=np.array([10.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id="op1", feature_id="f1"
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("positional discontinuity" in e.message for e in report.errors)


def test_g1_tangency_violation_fails():
    """Test that blend transitions marked as tangent fail if tangent vectors misalign > 5 degrees."""
    validator, tool = _make_validator()
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id=tool.id, purpose="finishing")

    # Linear cut along +X direction
    tp.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([5.0, 0.0, 0.0]),
        feed=500.0, tool_id=tool.id, operation_id="op1", feature_id="f1",
        metadata={"tangent_blend": True}
    ))
    # Arc departing at 90 degrees (+Y direction at junction instead of +X)
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CCW,
        start=np.array([5.0, 0.0, 0.0]),
        end=np.array([0.0, 5.0, 0.0]),
        arc_center=np.array([0.0, 0.0, 0.0]),  # radius vector is (+5, 0), CCW tangent is (0, 1) (+Y), 90 deg off from (+1, 0)
        arc_ccw=True,
        feed=500.0, tool_id=tool.id, operation_id="op1", feature_id="f1",
        metadata={"tangent_blend": True}
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("tangency violation" in e.message for e in report.errors)


def test_consecutive_arc_contour_center_shift_fails():
    """Test that two consecutive arcs intended to form a single circular contour fail if their centers shift."""
    validator, tool = _make_validator()
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id=tool.id, purpose="finishing")

    # First arc centered near (-0.603, 0.118)
    c1 = np.array([-0.603, 0.118, 0.0])
    p0 = np.array([-10.0, 0.0, 0.0])
    p1 = np.array([-8.0, 5.0, 0.0])
    # Adjust p0 and p1 so r_start == r_end for c1
    r1 = float(np.linalg.norm(p0[:2] - c1[:2]))
    # Compute true point on circle c1 of radius r1
    a1 = 1.0
    p1 = np.array([c1[0] + r1 * math.cos(a1), c1[1] + r1 * math.sin(a1), 0.0])

    # Second arc starting at p1 but centered at (0.001, 0.001) with different radius
    c2 = np.array([0.001, 0.001, 0.0])
    r2 = float(np.linalg.norm(p1[:2] - c2[:2]))
    a2 = 2.0
    p2 = np.array([c2[0] + r2 * math.cos(a2), c2[1] + r2 * math.sin(a2), 0.0])

    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CCW,
        start=p0, end=p1, arc_center=c1, arc_ccw=True,
        feed=500.0, tool_id=tool.id, operation_id="op1", feature_id="f1"
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CCW,
        start=p1, end=p2, arc_center=c2, arc_ccw=True,
        feed=500.0, tool_id=tool.id, operation_id="op1", feature_id="f1"
    ))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("inconsistent centers" in e.message for e in report.errors)
