"""Tests for Cutter Radius Compensation (G41/G42/G40) and CNC High-Speed Lookahead (G05.1/G187/G64).

Verifies:
1. Toolpaths emit linear lead-in/out annotated with cutter_comp metadata.
2. BasePostProcessor and subclasses emit G41 D{tool_num} and G40 properly.
3. Fanuc post emits G05.1 Q1 (lookahead on) in header and G05.1 Q0 in footer.
4. Haas post emits G187 P1 (roughing) and G187 P3 (finishing).
5. LinuxCNC post emits G64 path blending mode.
"""

import numpy as np
import pytest

from cam_engine.context import (MachineConfig, MachineType, MachineAxis, Units,
                                Tool, ToolType, PlanningContext, Setup, Stock,
                                StockKind, DEFAULT_MATERIALS, WorkOffset)

from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.post import get_post_processor, FanucPostProcessor, HaasPostProcessor, LinuxCncPostProcessor, GrblPostProcessor
from cam_engine.toolpaths.semantic import MotionType, MotionSegment, Toolpath
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext


def make_machine(controller="fanuc") -> MachineConfig:
    return MachineConfig(
        id="test-vmc",
        name="Test VMC",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 15000.0, 5000.0),
            "Y": MachineAxis("Y", -500.0, 500.0, 15000.0, 5000.0),
            "Z": MachineAxis("Z", -400.0, 50.0, 10000.0, 3000.0),
        },
        spindle_min_rpm=100.0,
        spindle_max_rpm=15000.0,
        max_spindle_power_kw=7.5,
        controller=controller,
        units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0,
        hsm_enabled=True,
    )


def make_context(controller="fanuc") -> StrategyContext:
    from cam_engine.geometry.mesh import TriangleMesh
    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.machine = make_machine(controller)
    ctx.material = DEFAULT_MATERIALS["aluminum_6061"]
    ctx.tools = [
        Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=10.0,
             flute_length=30.0, overall_length=75.0, flutes=4),
    ]
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([0.0, 0.0, 0.0]),
                  bounds_max=np.array([50.0, 50.0, 25.0]))
    setup = Setup(id="setup_1", name="Top", model_to_setup=np.eye(4),
                  work_offset=WorkOffset.G54, stock=stock)


    return StrategyContext(
        context=ctx,
        setup=setup,
        mesh=mesh,
        stock_voxels=object(),
        clearance_z=35.0,
        finish_allowance_wall=0.3,
        finish_allowance_floor=0.3,
    )


def test_finish_wall_emits_cutter_comp_segments():
    strat_ctx = make_context()
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]

    feature = MachiningFeature(
        id="pocket_1",
        type=FeatureType.POCKET,
        face_indices=[0],
        bounds_min=np.array([10.0, 10.0, 0.0]),
        bounds_max=np.array([40.0, 40.0, 20.0]),
        depth=20.0,
        top_z=20.0,
        floor_z=0.0,
        is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=1200.0, plunge_feed=400.0, ramp_feed=600.0,
        feed_per_tooth_mm=0.06, depth_of_cut_mm=10.0, stepover_mm=2.0, coolant=True,
        effective_diameter_mm=10.0, engagement_angle_deg=45.0, power_kw=1.5, torque_nm=3.0,
    )
    op = PlannedOperation(id="op_finish_1", purpose=OpPurpose.FINISHING, feature=feature, tool=tool, params=params)

    tp = engine.generate(op)
    assert tp.metadata.get("hsm_mode") == "finishing"

    comp_on_segs = [s for s in tp.segments if s.metadata.get("cutter_comp", {}).get("action") == "on"]
    comp_off_segs = [s for s in tp.segments if s.metadata.get("cutter_comp", {}).get("action") == "off"]

    assert len(comp_on_segs) >= 1
    assert len(comp_off_segs) >= 1
    assert comp_on_segs[0].metadata["cutter_comp"]["side"] == "left"
    assert comp_on_segs[0].metadata["cutter_comp"]["d_register"] == 1


def test_fanuc_post_cutter_comp_and_aicc():
    strat_ctx = make_context("fanuc")
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]

    feature = MachiningFeature(
        id="pocket_1", type=FeatureType.POCKET, face_indices=[0],
        bounds_min=np.array([10.0, 10.0, 0.0]), bounds_max=np.array([40.0, 40.0, 20.0]),
        depth=20.0, top_z=20.0, floor_z=0.0, is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=1200.0, plunge_feed=400.0, ramp_feed=600.0,
        feed_per_tooth_mm=0.06, depth_of_cut_mm=10.0, stepover_mm=2.0, coolant=True,
        effective_diameter_mm=10.0, engagement_angle_deg=45.0, power_kw=1.5, torque_nm=3.0,
    )
    op = PlannedOperation(id="op_finish_1", purpose=OpPurpose.FINISHING, feature=feature, tool=tool, params=params)
    tp = engine.generate(op)

    post = FanucPostProcessor(work_offset="G54")
    res = post.post_process([tp], program_name="TEST_FANUC")

    gcode = res.gcode
    # Verify AICC Lookahead ON / OFF
    assert "G05.1 Q1" in gcode
    assert "G05.1 Q0" in gcode
    # Verify G41 D1 and G40 cutter comp
    assert "G41 D1" in gcode
    assert "G40" in gcode


def test_haas_post_g187_smoothing_modes():
    strat_ctx = make_context("haas")
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]

    feature = MachiningFeature(
        id="pocket_1", type=FeatureType.POCKET, face_indices=[0],
        bounds_min=np.array([10.0, 10.0, 0.0]), bounds_max=np.array([40.0, 40.0, 20.0]),
        depth=20.0, top_z=20.0, floor_z=0.0, is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=1200.0, plunge_feed=400.0, ramp_feed=600.0,
        feed_per_tooth_mm=0.06, depth_of_cut_mm=10.0, stepover_mm=2.0, coolant=True,
        effective_diameter_mm=10.0, engagement_angle_deg=45.0, power_kw=1.5, torque_nm=3.0,
    )
    op_rough = PlannedOperation(id="op_rough_1", purpose=OpPurpose.ROUGHING, feature=feature, tool=tool, params=params)
    op_finish = PlannedOperation(id="op_finish_1", purpose=OpPurpose.FINISHING, feature=feature, tool=tool, params=params)

    tp_rough = engine.generate(op_rough)
    tp_finish = engine.generate(op_finish)

    post = HaasPostProcessor(work_offset="G54")
    res = post.post_process([tp_rough, tp_finish], program_name="TEST_HAAS")

    gcode = res.gcode
    # Verify G187 P1 (roughing) and G187 P3 (finishing)
    assert "G187 P1" in gcode
    assert "G187 P3" in gcode
    assert "G187" in gcode


def test_linuxcnc_post_g64_blending():
    strat_ctx = make_context("linuxcnc")
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]

    feature = MachiningFeature(
        id="pocket_1", type=FeatureType.POCKET, face_indices=[0],
        bounds_min=np.array([10.0, 10.0, 0.0]), bounds_max=np.array([40.0, 40.0, 20.0]),
        depth=20.0, top_z=20.0, floor_z=0.0, is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=1200.0, plunge_feed=400.0, ramp_feed=600.0,
        feed_per_tooth_mm=0.06, depth_of_cut_mm=10.0, stepover_mm=2.0, coolant=True,
        effective_diameter_mm=10.0, engagement_angle_deg=45.0, power_kw=1.5, torque_nm=3.0,
    )
    op = PlannedOperation(id="op_finish_1", purpose=OpPurpose.FINISHING, feature=feature, tool=tool, params=params)
    tp = engine.generate(op)

    post = LinuxCncPostProcessor(work_offset="G54")
    res = post.post_process([tp], program_name="TEST_LINUXCNC")

    assert "G64" in res.gcode
    assert "G41 D1" in res.gcode
