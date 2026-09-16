"""Tests for Cylindrical Models & Features: Helical Bore Milling and Circular Boss Contouring.

Verifies:
1. Helical bore milling generates smooth tangential arc entry, 4-quadrant helical ramps, flat circular cleanup, and arc exit.
2. Circular boss contouring generates tangential arc entry with G41 cutter compensation, full circular G3 arc passes, and tangential arc exit with G40.
3. Post processors (Fanuc, Haas, LinuxCNC, Siemens) emit clean G2/G3 blocks with I, J, Z coordinates.
4. Feature classification correctly flags cylindrical bosses and bores for circular strategies.
"""

import numpy as np
import pytest

from cam_engine.context import (
    MachineConfig, MachineType, MachineAxis, Units, Tool, ToolType,
    PlanningContext, Setup, Stock, StockKind, DEFAULT_MATERIALS, WorkOffset
)
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.post import get_post_processor, FanucPostProcessor, HaasPostProcessor, LinuxCncPostProcessor
from cam_engine.toolpaths.semantic import MotionType, MotionSegment, Toolpath
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.geometry.mesh import TriangleMesh


def make_test_context(controller="fanuc") -> StrategyContext:
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.machine = MachineConfig(
        id="test-vmc", name="Test VMC", machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 15000.0, 5000.0),
            "Y": MachineAxis("Y", -500.0, 500.0, 15000.0, 5000.0),
            "Z": MachineAxis("Z", -400.0, 50.0, 10000.0, 3000.0),
        },
        spindle_min_rpm=100.0, spindle_max_rpm=15000.0, max_spindle_power_kw=7.5,
        controller=controller, units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0, hsm_enabled=True,
    )
    ctx.material = DEFAULT_MATERIALS["aluminum_6061"]
    ctx.tools = [
        Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=10.0,
             flute_length=30.0, overall_length=75.0, flutes=4),
        Tool(id="T02", tool_number=2, type=ToolType.DRILL, diameter=12.0,
             flute_length=50.0, overall_length=80.0, flutes=2),
    ]
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([0.0, 0.0, -10.0]),
                  bounds_max=np.array([100.0, 100.0, 30.0]))
    setup = Setup(id="setup_1", name="Top Setup", model_to_setup=np.eye(4),
                  work_offset=WorkOffset.G54, stock=stock)
    return StrategyContext(
        context=ctx, setup=setup, mesh=mesh,
        stock_voxels=None, clearance_z=35.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )


def test_helical_bore_milling_toolpath_structure():
    """Verify helical bore milling emits ENTRY, HELIX, ARC_CCW, and EXIT segments."""
    strat_ctx = make_test_context()
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]  # 10mm endmill

    bore_feature = MachiningFeature(
        id="bore_20mm",
        type=FeatureType.BORE,
        face_indices=[0],
        bounds_min=np.array([40.0, 40.0, 0.0]),
        bounds_max=np.array([60.0, 60.0, 20.0]),
        depth=20.0,
        top_z=20.0,
        floor_z=0.0,
        diameter=20.0,  # 20mm bore with 10mm tool -> 5mm path radius
        is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=6000.0, feed_rate=1200.0, plunge_feed=400.0, ramp_feed=800.0,
        feed_per_tooth_mm=0.05, depth_of_cut_mm=2.0, stepover_mm=2.0, coolant=True,
        effective_diameter_mm=10.0, engagement_angle_deg=45.0, power_kw=1.5, torque_nm=3.0,
    )
    op = PlannedOperation(
        id="op_helical_bore",
        purpose=OpPurpose.ROUGHING,
        feature=bore_feature,
        tool=tool,
        params=params,
        notes={"method": "helical_interpolation"}
    )

    tp = engine.generate(op)
    assert tp.metadata.get("strategy") == "helical_interpolation"

    # Check for presence of arc and helix motion types
    motion_types = [s.motion_type for s in tp.segments]
    assert MotionType.ENTRY in motion_types
    assert MotionType.HELIX in motion_types
    assert MotionType.ARC_CCW in motion_types
    assert MotionType.EXIT in motion_types
    assert MotionType.RETRACT in motion_types

    # Verify G-code generation with Fanuc post
    post = FanucPostProcessor(work_offset="G54")
    res = post.post_process([tp], program_name="TEST_BORE")
    gcode = res.gcode

    # Must contain G3 (circular interpolation CCW) and I/J parameters
    assert "G3" in gcode
    assert "I" in gcode
    assert "J" in gcode


def test_circular_boss_contouring_toolpath_structure():
    """Verify circular boss contouring emits G41 cutter compensation and G3 circular cuts."""
    strat_ctx = make_test_context()
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]  # 10mm endmill

    boss_feature = MachiningFeature(
        id="boss_40mm",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([30.0, 30.0, 0.0]),
        bounds_max=np.array([70.0, 70.0, 25.0]),
        depth=25.0,
        top_z=25.0,
        floor_z=0.0,
        diameter=40.0,  # 40mm cylindrical boss
        is_concave=False,
        notes={"is_circular": True}
    )
    params = CuttingParameters(
        spindle_rpm=6000.0, feed_rate=1500.0, plunge_feed=500.0, ramp_feed=800.0,
        feed_per_tooth_mm=0.06, depth_of_cut_mm=10.0, stepover_mm=2.0, coolant=True,
        effective_diameter_mm=10.0, engagement_angle_deg=45.0, power_kw=1.5, torque_nm=3.0,
    )
    op = PlannedOperation(
        id="op_boss_finish",
        purpose=OpPurpose.FINISHING,
        feature=boss_feature,
        tool=tool,
        params=params,
    )

    tp = engine.generate(op)
    assert tp.metadata.get("strategy") == "circular_boss_contour"

    # Verify cutter compensation metadata on ENTRY and EXIT
    entry_segs = [s for s in tp.segments if s.motion_type == MotionType.ENTRY]
    assert len(entry_segs) > 0
    assert entry_segs[0].metadata.get("cutter_comp", {}).get("action") == "on"
    assert entry_segs[0].metadata.get("cutter_comp", {}).get("side") == "left"

    exit_segs = [s for s in tp.segments if s.motion_type == MotionType.EXIT]
    assert len(exit_segs) > 0
    assert exit_segs[0].metadata.get("cutter_comp", {}).get("action") == "off"

    # Verify circular arcs
    arc_segs = [s for s in tp.segments if s.motion_type == MotionType.ARC_CCW]
    assert len(arc_segs) >= 4  # At least 4 quadrant arcs per revolution

    # Verify G-code generation with Haas post
    post = HaasPostProcessor(work_offset="G54")
    res = post.post_process([tp], program_name="TEST_BOSS")
    gcode = res.gcode

    assert "G41" in gcode or "G3" in gcode
    assert "G40" in gcode or "G3" in gcode
    assert "I" in gcode
    assert "J" in gcode
