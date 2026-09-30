"""Tests for Post-Processing Reverse Validation Engine.

Verifies the complete reverse validation pipeline:
CAD -> Features -> Operations -> Canonical Toolpath -> Postprocessor -> G-code -> G-code parser -> Reconstructed Toolpath -> Compare (XYZ, Arcs, Feeds, Spindle, Tool, WCS, Comp, Boundaries, Retracts)
"""

import numpy as np
import pytest

from cam_engine.context import (
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
from cam_engine.errors import CamError, POST_REVERSE_VALIDATION_FAILED
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath, CoordSpace, ToolpathGeometryMode
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.post import get_post_processor, GCodeGeometryValidator, NCGeometryValidationReport
from cam_engine.units import Units


def _make_fixture():
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=6.0, flute_length=20.0, overall_length=50.0,
        flutes=3, material="carbide", can_plunge=True,
        length_offset_h=1, diameter_offset_d=1
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-30.0, -30.0, -15.0]),
        bounds_max=np.array([30.0, 30.0, 5.0])
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
        tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"]
    )
    return ctx, setup, tool


def test_reverse_validation_xyz_success_and_tampered_fail():
    """Verify XYZ trajectory comparison detects exact match vs coordinate corruption."""
    ctx, setup, tool = _make_fixture()
    tp = Toolpath(
        operation_id="op_contour",
        feature_id="f1",
        tool_id="T01",
        purpose="finishing",
        metadata={"tool_info": {"tool_number": 1}}
    )
    tp.add(MotionSegment(MotionType.RAPID, np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0])))
    tp.add(MotionSegment(MotionType.PLUNGE, np.array([10.0, 0.0, 10.0]), np.array([10.0, 0.0, 0.0]), feed=300.0))
    tp.add(MotionSegment(MotionType.CUT, np.array([10.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0]), feed=1000.0))
    tp.add(MotionSegment(MotionType.CUT, np.array([20.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0]), feed=1000.0))
    tp.add(MotionSegment(MotionType.RETRACT, np.array([30.0, 0.0, 0.0]), np.array([30.0, 0.0, 10.0]), feed=1500.0))

    valid_gcode = """
    G90 G54
    T1 M6
    S6000 M3
    G0 X10.0 Y0.0 Z10.0
    G1 Z0.0 F300.0
    G1 X20.0 Y0.0 F1000.0
    G1 X30.0 Y0.0
    G0 Z10.0
    M30
    """

    validator = GCodeGeometryValidator(tolerance_mm=0.05)
    report_valid = validator.validate_toolpaths_vs_gcode(valid_gcode, [tp], setup=setup)
    assert report_valid.passed is True
    assert report_valid.xyz_matches is True
    assert report_valid.max_deviation_mm <= 0.05

    # Tampered coordinate (X99.0 instead of X30.0)
    tampered_gcode = valid_gcode.replace("G1 X30.0", "G1 X99.0")
    report_invalid = validator.validate_toolpaths_vs_gcode(tampered_gcode, [tp], setup=setup)
    assert report_invalid.passed is False
    assert report_invalid.xyz_matches is False
    assert report_invalid.max_deviation_mm > 50.0


def test_reverse_validation_arc_center_and_direction():
    """Verify arc center (I, J) and CW/CCW direction validation."""
    tp = Toolpath(
        operation_id="op_arc",
        feature_id="f_circ",
        tool_id="T01",
        purpose="finishing",
        metadata={"tool_info": {"tool_number": 1}}
    )
    # 90-degree CCW arc from (10, 0, 0) to (0, 10, 0) centered at (0, 0, 0)
    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CCW,
        start=np.array([10.0, 0.0, 0.0]),
        end=np.array([0.0, 10.0, 0.0]),
        arc_center=np.array([0.0, 0.0, 0.0]),
        arc_ccw=True,
        feed=800.0,
    ))

    valid_arc_gcode = """
    G90 G54
    T1 M6
    G0 X10.0 Y0.0 Z0.0
    G3 X0.0 Y10.0 I-10.0 J0.0 F800.0
    M30
    """

    validator = GCodeGeometryValidator(tolerance_mm=0.05)
    report = validator.validate_toolpaths_vs_gcode(valid_arc_gcode, [tp])
    assert report.passed is True
    assert report.arc_matches is True
    assert report.max_arc_center_deviation_mm <= 0.05

    # Inverted direction (G2 instead of G3)
    wrong_dir_gcode = valid_arc_gcode.replace("G3", "G2")
    report_wrong_dir = validator.validate_toolpaths_vs_gcode(wrong_dir_gcode, [tp])
    assert report_wrong_dir.passed is False
    assert report_wrong_dir.arc_matches is False
    assert any("Arc direction mismatch" in err for err in report_wrong_dir.errors)

    # Corrupted arc center (I-50.0 instead of I-10.0)
    corrupted_center_gcode = valid_arc_gcode.replace("I-10.0", "I-50.0")
    report_bad_center = validator.validate_toolpaths_vs_gcode(corrupted_center_gcode, [tp])
    assert report_bad_center.passed is False
    assert report_bad_center.arc_matches is False
    assert any("Arc center deviation" in err for err in report_bad_center.errors)


def test_reverse_validation_tool_and_wcs_mismatch():
    """Verify tool number and work coordinate system (WCS) mismatches are flagged."""
    ctx, setup, tool = _make_fixture()
    tp = Toolpath(
        operation_id="op_drill",
        feature_id="f_hole",
        tool_id="T01",
        purpose="drilling",
        metadata={"tool_info": {"tool_number": 1}}
    )
    tp.add(MotionSegment(MotionType.CUT, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -10.0]), feed=200.0))

    # G-code calling T2 and G55 instead of T1 and G54
    mismatched_gcode = """
    G90 G55
    T2 M6
    G0 X0.0 Y0.0 Z5.0
    G1 Z-10.0 F200.0
    M30
    """

    validator = GCodeGeometryValidator()
    report = validator.validate_toolpaths_vs_gcode(mismatched_gcode, [tp], setup=setup)
    assert report.passed is False
    assert report.tool_matches is False
    assert report.wcs_matches is False
    assert any("requires T1, but G-code active tools are [2]" in err for err in report.errors)
    assert any("setup requires G54, but G-code uses ['G55']" in err for err in report.errors)


def test_reverse_validation_strict_mode_raises_cam_error():
    """Verify strict mode raises CamError with POST_REVERSE_VALIDATION_FAILED code."""
    tp = Toolpath(
        operation_id="op_test",
        feature_id="f1",
        tool_id="T01",
        purpose="finishing",
        metadata={"cutter_comp_info": {"d_register": 1, "mode": "wear", "side": "left"}}
    )
    tp.add(MotionSegment(MotionType.CUT, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])))

    # G-code with missing cutter comp
    bad_gcode = """
    G90 G54
    G1 X10.0 Y0.0 Z0.0 F500.0
    M30
    """

    validator = GCodeGeometryValidator()
    with pytest.raises(CamError) as exc_info:
        validator.validate_toolpaths_vs_gcode(bad_gcode, [tp], strict=True)

    assert exc_info.value.code == POST_REVERSE_VALIDATION_FAILED
    assert "Post-processing reverse validation failed" in exc_info.value.message
