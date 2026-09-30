"""Tests for Cutter Compensation Architecture and Independent Register Mapping.

Verifies:
1. Toolpath Representation:
   - CENTER_PATH: points represent tool centerline.
   - PART_BOUNDARY_PATH: points lie directly on nominal CAD part boundary.
2. Compensation Modes:
   - NONE: No G41/G42 emitted in G-code (CAM calculates tool center path; G40 only).
   - WEAR: Toolpath is CENTER_PATH; G41/G42 emitted for operator wear delta (CNC D = 0.0 + delta).
   - GEOMETRY: Toolpath is PART_BOUNDARY_PATH; G41/G42 emitted for full tool radius (CNC D = Tool Radius).
3. Independent Register Mapping:
   - Tool number T, Length offset number H, Diameter offset number D are independently mapped.
   - E.g. T2 with H12 and D22 emits T02, G43 H12, and G41 D22 (never assumes T=H=D).
4. Strict Finishing Policy:
   - Finishing operations with default/NONE compensation mode DO NOT emit G41/G42.
   - Finishing operations only emit G41/G42 when explicitly configured with WEAR or GEOMETRY mode.
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
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.post import get_post_processor
from cam_engine.toolpaths.semantic import (
    ToolpathRepresentation,
    CompensationMode,
    CutterCompensationPolicy,
    CompensationSide,
)
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.units import Units


def _make_context(tools=None):
    if tools is None:
        tools = [
            Tool(
                id="T02", tool_number=2, type=ToolType.FLAT_ENDMILL,
                diameter=6.0, flute_length=20.0, overall_length=60.0,
                flutes=3, material="carbide", can_plunge=True,
                length_offset_h=12, diameter_offset_d=22,
            )
        ]
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-30.0, -30.0, -15.0]),
        bounds_max=np.array([30.0, 30.0, 5.0]),
    )
    setup = Setup(
        id="setup_01", name="Setup 1",
        model_to_setup=np.eye(4), work_offset=WorkOffset.G54,
        stock=stock,
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
        tools=tools, material=DEFAULT_MATERIALS["aluminum_6061"],
    )
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=mesh, stock_voxels=None,
        clearance_z=15.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0,
    )
    return strat_ctx, tools[0]


def test_independent_t_h_d_register_mapping():
    """Verify tool registers T=2, H=12, D=22 are preserved independently and never assumed T=H=D."""
    strat_ctx, tool = _make_context()
    assert tool.tool_number == 2
    assert tool.length_offset_h == 12
    assert tool.diameter_offset_d == 22
    assert tool.effective_h == 12
    assert tool.effective_d == 22

    policy = CutterCompensationPolicy(
        representation=ToolpathRepresentation.CENTER_PATH,
        mode=CompensationMode.WEAR,
        side=CompensationSide.LEFT,
        tool_number=tool.tool_number,
        length_offset_h=tool.length_offset_h,
        diameter_offset_d=tool.diameter_offset_d,
    )

    assert policy.tool_number == 2
    assert policy.effective_h == 12
    assert policy.effective_d == 22
    assert policy.emits_g41_g42 is True


def test_no_g41_simply_because_operation_is_finishing():
    """Verify finishing operation with default (NONE) compensation mode NEVER emits G41/G42."""
    strat_ctx, tool = _make_context()
    feat = MachiningFeature(
        id="boss_f", type=FeatureType.BOSS,
        face_indices=[0], bounds_min=np.array([-10, -10, 0]),
        bounds_max=np.array([10, 10, 10]), depth=10.0,
        top_z=10.0, floor_z=0.0, diameter=20.0,
        accessibility=Accessibility.TOOL_AXIS_OK,
    )
    params = CuttingParameters(
        spindle_rpm=6000.0, feed_rate=800.0, plunge_feed=300.0,
        ramp_feed=500.0, depth_of_cut_mm=10.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=6.0, engagement_angle_deg=90.0,
        power_kw=1.0, torque_nm=0.8,
    )
    # Finishing operation with NO compensation requested
    op = PlannedOperation(
        id="op_finish_no_comp", feature=feat, tool=tool,
        purpose=OpPurpose.FINISHING, params=params,
        notes={}, # No cutter_comp flag or compensation_mode specified
    )

    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    # Must be on CENTER_PATH with compensation_mode = NONE
    assert tp.geometry_mode == ToolpathRepresentation.CENTER_PATH
    assert tp.metadata.get("compensation_mode") == "none"

    # Segments must NOT carry cutter compensation metadata
    comp_segs = [s for s in tp.segments if s.metadata and s.metadata.get("cutter_comp")]
    assert len(comp_segs) == 0

    # Emitted G-code must NOT contain G41 or G42
    post = get_post_processor("haas", renumber_tools=False)
    res = post.post_process([tp], program_name="TEST_NO_COMP")
    assert "G41" not in res.gcode
    assert "G42" not in res.gcode
    # Tool change must use H12 for T02
    assert "T02" in res.gcode or "T2" in res.gcode
    assert "H12" in res.gcode
    assert "H02" not in res.gcode and "H2" not in res.gcode


def test_wear_compensation_mode_with_independent_registers():
    """Verify WEAR compensation mode generates CENTER_PATH and emits G41 D22 with T02 and H12."""
    strat_ctx, tool = _make_context()
    feat = MachiningFeature(
        id="boss_f", type=FeatureType.BOSS,
        face_indices=[0], bounds_min=np.array([-10, -10, 0]),
        bounds_max=np.array([10, 10, 10]), depth=10.0,
        top_z=10.0, floor_z=0.0, diameter=20.0,
        accessibility=Accessibility.TOOL_AXIS_OK,
    )
    params = CuttingParameters(
        spindle_rpm=6000.0, feed_rate=800.0, plunge_feed=300.0,
        ramp_feed=500.0, depth_of_cut_mm=10.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=6.0, engagement_angle_deg=90.0,
        power_kw=1.0, torque_nm=0.8,
    )
    # Explicit WEAR compensation requested
    op = PlannedOperation(
        id="op_finish_wear", feature=feat, tool=tool,
        purpose=OpPurpose.FINISHING, params=params,
        notes={"compensation_mode": "wear"},
    )

    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    assert tp.geometry_mode == ToolpathRepresentation.CENTER_PATH
    assert tp.metadata["compensation_mode"] == "wear"
    assert tp.metadata["cutter_comp_info"]["d_register"] == 22
    assert tp.metadata["cutter_comp_info"]["diameter_offset_d"] == 22

    post = get_post_processor("haas", renumber_tools=False)
    res = post.post_process([tp], program_name="TEST_WEAR_COMP")

    # Verify independent T2, H12, D22
    assert "G43 H12" in res.gcode or "H12" in res.gcode
    assert "G41 D22" in res.gcode
    assert "G40" in res.gcode
    assert "D2 " not in res.gcode


def test_geometry_compensation_mode_part_boundary():
    """Verify GEOMETRY compensation mode sets PART_BOUNDARY_PATH and emits G41 D22."""
    strat_ctx, tool = _make_context()
    feat = MachiningFeature(
        id="boss_f", type=FeatureType.BOSS,
        face_indices=[0], bounds_min=np.array([-10, -10, 0]),
        bounds_max=np.array([10, 10, 10]), depth=10.0,
        top_z=10.0, floor_z=0.0, diameter=20.0,
        accessibility=Accessibility.TOOL_AXIS_OK,
    )
    params = CuttingParameters(
        spindle_rpm=6000.0, feed_rate=800.0, plunge_feed=300.0,
        ramp_feed=500.0, depth_of_cut_mm=10.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=6.0, engagement_angle_deg=90.0,
        power_kw=1.0, torque_nm=0.8,
    )
    # Explicit GEOMETRY compensation requested
    op = PlannedOperation(
        id="op_finish_geom", feature=feat, tool=tool,
        purpose=OpPurpose.FINISHING, params=params,
        notes={"compensation_mode": "geometry"},
    )

    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    # Representation must be PART_BOUNDARY_PATH
    assert tp.geometry_mode == ToolpathRepresentation.PART_BOUNDARY_PATH
    assert tp.metadata["compensation_mode"] == "geometry"
    assert tp.metadata["cutter_comp_info"]["d_register"] == 22

    post = get_post_processor("fanuc", renumber_tools=False)
    res = post.post_process([tp], program_name="TEST_GEOM_COMP")
    assert "G43 H12" in res.gcode or "H12" in res.gcode
    assert "G41 D22" in res.gcode
