"""Tests for spot drilling, reaming, tapping, and chamfering strategies.

These purposes were planned by OperationPlanner but had no strategy in
StrategyEngine, causing UNSUPPORTED_FEATURE errors at toolpath generation.
"""

import numpy as np
import pytest

from cam_engine.context import (Material, MachineAxis, MachineConfig,
                               MachineType, Tool, ToolType, Units)
from cam_engine.demo_models import get_default_tools
from cam_engine.errors import CamError, UNSUPPORTED_FEATURE, INVALID_TOOLPATH
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import OpPurpose, PlannedOperation
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.semantic import MotionType


def make_machine() -> MachineConfig:
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
        controller="grbl",
        units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0,
    )


def make_tool(tool_id: str = "T05", tool_type: ToolType = ToolType.CHAMFER_MILL,
              diameter: float = 8.0) -> Tool:
    return Tool(
        id=tool_id,
        tool_number=5,
        type=tool_type,
        diameter=diameter,
        flute_length=25.0,
        overall_length=60.0,
        flutes=3,
        material="carbide",
        can_plunge=True,
    )


def make_feature(**overrides) -> MachiningFeature:
    defaults = dict(
        id="hole_1",
        type=FeatureType.HOLE,
        face_indices=[0],
        bounds_min=np.array([24.0, 24.0, 0.0]),
        bounds_max=np.array([36.0, 36.0, 25.0]),
        depth=25.0,
        top_z=25.0,
        floor_z=0.0,
        diameter=12.0,
        is_concave=True,
    )
    defaults.update(overrides)
    return MachiningFeature(**defaults)


def make_params() -> CuttingParameters:
    return CuttingParameters(
        spindle_rpm=4000.0,
        feed_rate=800.0,
        plunge_feed=300.0,
        ramp_feed=400.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=2.0,
        stepover_mm=1.0,
        coolant=True,
        effective_diameter_mm=8.0,
        engagement_angle_deg=45.0,
        power_kw=1.0,
        torque_nm=2.0,
    )


def make_engine(tmp_mesh=None) -> tuple[StrategyEngine, Tool]:
    from cam_engine.geometry.mesh import TriangleMesh

    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    stock = object()  # unused by these strategies

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.machine = make_machine()

    tool = make_tool()
    strat_ctx = StrategyContext(
        context=ctx,
        setup=None,          # unused by these strategies
        mesh=mesh,
        stock_voxels=stock,
        clearance_z=35.0,
        finish_allowance_wall=0.3,
        finish_allowance_floor=0.3,
    )
    return StrategyEngine(strat_ctx), tool


def make_op(engine_tool: Tool, purpose: OpPurpose, feature: MachiningFeature,
            notes: dict | None = None) -> PlannedOperation:
    return PlannedOperation(
        id=f"op_{purpose.value}",
        purpose=purpose,
        feature=feature,
        tool=engine_tool,
        params=make_params(),
        notes=notes or {},
    )


# ------------------------------------------------------------- dispatch ----

def test_all_four_purposes_have_strategies():
    engine, tool = make_engine()
    f = make_feature()
    for purpose in (OpPurpose.SPOT_DRILLING, OpPurpose.REAMING,
                    OpPurpose.TAPPING, OpPurpose.CHAMFERING):
        op = make_op(tool, purpose, f, notes={"spot_depth": 1.0,
                                              "tap_feed_mm_per_rev": 1.5,
                                              "angle_from_horizontal": 45.0})
        tp = engine.generate(op)
        assert tp.purpose == purpose.value
        assert len(tp.segments) > 0


# -------------------------------------------------------- spot drilling ----

def test_spot_drilling_basic_motion():
    engine, tool = make_engine()
    f = make_feature()
    op = make_op(tool, OpPurpose.SPOT_DRILLING, f, notes={"spot_depth": 1.6})

    tp = engine.generate(op)

    plunges = [s for s in tp.segments if s.motion_type is MotionType.PLUNGE]
    dwells = [s for s in tp.segments if s.motion_type is MotionType.DWELL]
    retracts = [s for s in tp.segments if s.motion_type is MotionType.RETRACT]

    assert len(plunges) == 1
    assert len(dwells) == 1
    # final retract goes back to clearance
    assert retracts[-1].end[2] == pytest.approx(35.0)
    # plunge bottom = top - spot depth
    assert plunges[0].end[2] == pytest.approx(f.top_z - 1.6)
    assert tp.metadata["strategy"] == "spot_drill"
    assert tp.metadata["canned_cycle_intent"]["cycle"] == "G82"


def test_spot_drilling_missing_depth_raises():
    engine, tool = make_engine()
    op = make_op(tool, OpPurpose.SPOT_DRILLING, make_feature(), notes={})
    with pytest.raises(CamError) as exc:
        engine.generate(op)
    assert exc.value.code == INVALID_TOOLPATH
    assert "spot_depth" in exc.value.message


# ------------------------------------------------------------ reaming ----

def test_reaming_single_plunge_and_feed_out():
    engine, tool = make_engine()
    f = make_feature()
    op = make_op(tool, OpPurpose.REAMING, f)

    tp = engine.generate(op)

    plunges = [s for s in tp.segments if s.motion_type is MotionType.PLUNGE]
    # one feed down, one feed back out (as RETRACT with feed), then rapid retract
    assert len(plunges) == 1
    assert plunges[0].end[2] == pytest.approx(f.floor_z)

    feed_out = [s for s in tp.segments
                if s.motion_type is MotionType.RETRACT and s.feed is not None]
    assert len(feed_out) == 1
    assert feed_out[0].start[2] == pytest.approx(f.floor_z)
    assert tp.metadata["strategy"] == "reaming"
    assert tp.metadata["canned_cycle_intent"]["cycle"] == "G85"


# ------------------------------------------------------------ tapping ----

def test_tapping_uses_feed_per_rev():
    engine, tool = make_engine()
    op = make_op(engine_tool := tool, OpPurpose.TAPPING, make_feature(),
                 notes={"tap_feed_mm_per_rev": 1.5})
    tp = engine.generate(op)

    plunge = [s for s in tp.segments if s.motion_type is MotionType.PLUNGE][0]
    assert plunge.feed == pytest.approx(1.5)
    assert plunge.metadata.get("tap_feed_mm_per_rev") == 1.5
    assert tp.metadata["canned_cycle_intent"]["cycle"] == "G84"
    assert tp.metadata["tap_feed_mm_per_rev"] == 1.5


def test_tapping_missing_feed_raises():
    engine, tool = make_engine()
    op = make_op(tool, OpPurpose.TAPPING, make_feature(), notes={})
    with pytest.raises(CamError) as exc:
        engine.generate(op)
    assert exc.value.code == INVALID_TOOLPATH
    assert "tap_feed" in exc.value.message


# --------------------------------------------------------- chamfering ----

def test_chamfering_offsets_outside_boundary():
    engine, tool = make_engine()
    f = make_feature(
        id="chamfer_1",
        type=FeatureType.CHAMFER,
        bounds_min=np.array([10.0, 10.0, 24.0]),
        bounds_max=np.array([60.0, 50.0, 25.0]),
        depth=1.0,
        top_z=25.0,
        floor_z=24.0,
        is_concave=False,
    )
    op = make_op(tool, OpPurpose.CHAMFERING, f,
                 notes={"angle_from_horizontal": 45.0})

    tp = engine.generate(op)

    cuts = [s for s in tp.segments if s.motion_type is MotionType.CUT]
    assert len(cuts) >= 4  # closed contour

    # 45deg chamfer of depth 1.0 -> horizontal leg 1.0; tool radius 4.0
    # outermost X should be bmax_x + 1.0 + 4.0 = 65.0
    max_x = max(s.end[0] for s in cuts)
    assert max_x == pytest.approx(60.0 + 1.0 + 4.0)
    assert tp.metadata["strategy"] == "chamfer_contour"
    assert tp.metadata["chamfer_angle_deg"] == pytest.approx(45.0)


def test_chamfering_missing_angle_raises():
    engine, tool = make_engine()
    f = make_feature(id="chamfer_2", type=FeatureType.CHAMFER,
                     top_z=25.0, floor_z=24.0, depth=1.0)
    op = make_op(tool, OpPurpose.CHAMFERING, f, notes={})
    with pytest.raises(CamError) as exc:
        engine.generate(op)
    assert exc.value.code == INVALID_TOOLPATH


# ---------------------------------------------------- planner coverage ----

def test_planner_purposes_now_generate_toolpaths():
    """End-to-end: the purposes the planner emits must all be generatable."""
    engine, tool = make_engine()
    drill = make_tool("T03", ToolType.DRILL, diameter=12.0)
    chamfer = make_tool("T05", ToolType.CHAMFER_MILL, diameter=8.0)

    hole = make_feature()
    chamfer_feat = make_feature(
        id="chamfer_3", type=FeatureType.CHAMFER,
        bounds_min=np.array([10.0, 10.0, 24.0]),
        bounds_max=np.array([60.0, 50.0, 25.0]),
        depth=1.0, top_z=25.0, floor_z=24.0, is_concave=False,
    )

    cases = [
        (OpPurpose.SPOT_DRILLING, drill, {"spot_depth": 1.5}, hole),
        (OpPurpose.DRILLING, drill, {}, hole),
        (OpPurpose.REAMING, drill, {}, hole),
        (OpPurpose.TAPPING, drill, {"tap_feed_mm_per_rev": 1.5}, hole),
        (OpPurpose.CHAMFERING, chamfer, {"angle_from_horizontal": 45.0}, chamfer_feat),
    ]
    for purpose, t, notes, feat in cases:
        op = make_op(t, purpose, feat, notes=notes)
        tp = engine.generate(op)
        assert tp.purpose == purpose.value
