"""Tests for NC-vs-Toolpath Geometric Fidelity and Cutter Compensation Verification.

Verifies that:
1. Generated G-code is parsed back and reconstructed into tool-center motion segments.
2. The reconstructed NC motion path geometrically matches the planned semantic toolpaths.
3. Any D register mismatch (e.g. D1 vs D3), missing G41/G40, or spatial coordinate divergence
   is automatically detected before export.
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
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.geometry import TriangleMesh
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath, CoordSpace
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.post import get_post_processor, GCodeGeometryValidator, NCGeometryValidationReport
from cam_engine.units import Units


def _make_fixture():
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, flute_length=15.0, overall_length=50.0,
        flutes=3, material="carbide", can_plunge=True,
        length_offset_h=1, diameter_offset_d=1
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
        tools=[tool], material=DEFAULT_MATERIALS["becu"]
    )
    return ctx, setup, tool


def test_post_result_includes_geometry_validation_report():
    """Verify that post-processing automatically attaches a clean NC geometry validation report."""
    ctx, setup, tool = _make_fixture()
    feat = MachiningFeature(
        id="f_boss", type=FeatureType.BOSS,
        face_indices=[0], bounds_min=np.array([-5, -5, 0]),
        bounds_max=np.array([5, 5, 5]), depth=5.0,
        top_z=5.0, floor_z=0.0, accessibility=Accessibility.TOOL_AXIS_OK,
        diameter=10.0,
    )
    params = CuttingParameters(
        spindle_rpm=6366.0, feed_rate=860.0, plunge_feed=344.0,
        ramp_feed=602.0, depth_of_cut_mm=5.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.045, coolant=True,
        effective_diameter_mm=3.0, engagement_angle_deg=90.0,
        power_kw=0.31, torque_nm=0.47
    )
    op = PlannedOperation(
        id="op_finish", feature=feat, tool=tool,
        purpose=OpPurpose.FINISHING, params=params,
        notes={"cutter_comp": True},
    )

    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=15.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    post = get_post_processor("haas", work_offset="G54")
    res = post.post_process([tp], program_name="TEST_GEO_VALIDATION")

    assert res.geometry_validation is not None
    assert res.geometry_validation.passed is True
    assert res.geometry_validation.max_deviation_mm <= 0.05
    assert res.geometry_validation.cutter_comp_matches is True
    assert 1 in res.geometry_validation.active_d_registers
    assert len(res.geometry_validation.errors) == 0


def test_validator_detects_tampered_coordinates_or_deviation():
    """Verify that GCodeGeometryValidator detects spatial coordinate divergence."""
    ctx, setup, tool = _make_fixture()
    tp = Toolpath(
        operation_id="op_contour",
        feature_id="f1",
        tool_id="T01",
        purpose="finishing",
        metadata={"cutter_comp_info": {"d_register": 1, "mode": "wear", "side": "left"}}
    )
    # Planned points at X10, X20, X30
    tp.add(MotionSegment(MotionType.RAPID, np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0])))
    tp.add(MotionSegment(MotionType.ENTRY, np.array([10.0, 0.0, 10.0]), np.array([10.0, 0.0, 0.0])))
    tp.add(MotionSegment(MotionType.CUT, np.array([10.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0])))
    tp.add(MotionSegment(MotionType.CUT, np.array([20.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0])))

    # G-code with tampered / deviating coordinate (e.g. X99.0 instead of X30.0)
    tampered_gcode = """
    G90 G54
    G0 X10.0 Y0.0 Z10.0
    G41 D1
    G1 Z0.0 F500.0
    G1 X20.0 Y0.0
    G1 X99.0 Y0.0
    G40
    M30
    """

    validator = GCodeGeometryValidator(tolerance_mm=0.05)
    report = validator.validate_toolpaths_vs_gcode(tampered_gcode, [tp])

    assert report.passed is False
    assert report.max_deviation_mm > 50.0
    assert any("exceeds tolerance" in err for err in report.errors)


def test_validator_detects_missing_cutter_compensation_in_gcode():
    """Verify that GCodeGeometryValidator detects when a toolpath planned with G41/G42 has no compensation emitted."""
    tp = Toolpath(
        operation_id="op_contour",
        feature_id="f1",
        tool_id="T01",
        purpose="finishing",
        metadata={"cutter_comp_info": {"d_register": 1, "mode": "wear", "side": "left"}}
    )
    tp.add(MotionSegment(MotionType.CUT, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])))

    # G-code lacking G41/G42
    uncomp_gcode = """
    G90 G54
    G1 X10.0 Y0.0 Z0.0 F500.0
    M30
    """

    validator = GCodeGeometryValidator()
    report = validator.validate_toolpaths_vs_gcode(uncomp_gcode, [tp])

    assert report.passed is False
    assert report.cutter_comp_matches is False
    assert any("emitted no G41/G42 commands" in err for err in report.errors)


def test_validator_detects_mismatched_d_register_in_gcode():
    """Verify that GCodeGeometryValidator catches D register divergence (e.g. planned D1, but G-code activated D3)."""
    tp = Toolpath(
        operation_id="op_contour",
        feature_id="f1",
        tool_id="T01",
        purpose="finishing",
        metadata={"cutter_comp_info": {"d_register": 1, "mode": "wear", "side": "left"}}
    )
    tp.add(MotionSegment(MotionType.ENTRY, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])))
    tp.add(MotionSegment(MotionType.CUT, np.array([10.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0])))
    tp.add(MotionSegment(MotionType.EXIT, np.array([20.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0])))

    # G-code activating D3 instead of planned D1
    mismatched_gcode = """
    G90 G54
    G0 X0.0 Y0.0 Z5.0
    G41 D3
    G1 X10.0 Y0.0 Z0.0 F500.0
    G1 X20.0 Y0.0
    G40
    G1 X30.0 Y0.0
    M30
    """

    validator = GCodeGeometryValidator()
    report = validator.validate_toolpaths_vs_gcode(mismatched_gcode, [tp])

    assert report.passed is False
    assert report.cutter_comp_matches is False
    assert any("Cutter compensation register mismatch" in err for err in report.errors)
    assert any("expects D1" in err and "activated D register(s) [3]" in err for err in report.errors)

