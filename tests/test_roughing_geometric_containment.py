"""Tests for Roughing Geometric Containment Validation.

Verifies:
1. Volumetric Boolean containment: Stock ∩ Tool Swept Volume (V_swept) ∩ Target Material.
2. Tool-center paths extending beyond finished geometry (e.g. Ø0.90 mm boss with Ø3 mm cutter
   having tool-center coordinates at X=-2.25, -3.85, -5.45) are validated as valid and safe.
3. Rejection of paths extending outside the stock volume.
4. Rejection of pure air-cutting passes with 0% stock intersection.
5. Gouge protection against finished CAD geometry and finish allowance violations.
6. Fixture collision avoidance during roughing.
7. Uncut material web detection when stepover exceeds cutter diameter.
8. Topological loop continuity, monotonic stepdown, and self-intersection prevention.
"""

import math
import numpy as np
import pytest

from cam_engine.context import (
    PlanningContext, Setup, Stock, StockKind, Tool, ToolType,
    DEFAULT_MATERIALS, WorkOffset, CamError
)
from cam_engine.coords import CoordSpace
from cam_engine.errors import GOUGE_DETECTED, INVALID_TOOLPATH, COLLISION_DETECTED
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.geometry.voxel import VoxelStock
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.demo_models import get_default_machine


def test_boss_roughing_tool_center_coordinates_pass_validation():
    """Ø0.90 mm boss (R=0.45 mm) machined with Ø3.0 mm endmill in Ø12 mm stock bar.
    Tool center passes naturally lie at X = -2.25, -3.85, -5.45 etc.
    These must be validated as geometrically valid containment without false positives."""
    feat = MachiningFeature(
        id="feat_pin_090",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 0.0]),
        bounds_max=np.array([0.45, 0.45, 5.0]),
        depth=5.0,
        top_z=5.0,
        floor_z=0.0,
        diameter=0.90,
        radius=0.45,
        is_concave=False,
        notes={"is_circular": True},
    )

    # Bar stock: Ø12.0 mm cylinder (radius 6.0 mm)
    stock = Stock.from_cylinder(center_xy=np.zeros(2), z_min=0.0, z_max=5.0, radius=6.0)
    setup = Setup(
        id="s1",
        name="Setup 1",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
    )

    tool = Tool(
        id="T01",
        tool_number=1,
        type=ToolType.FLAT_ENDMILL,
        diameter=3.0,
        flute_length=15.0,
        overall_length=50.0,
        flutes=4,
    )

    params = CuttingParameters(
        spindle_rpm=10000.0,
        feed_rate=800.0,
        plunge_feed=200.0,
        ramp_feed=400.0,
        feed_per_tooth_mm=0.02,
        depth_of_cut_mm=2.5,
        stepover_mm=1.5,
        coolant=True,
        effective_diameter_mm=3.0,
        engagement_angle_deg=90.0,
        power_kw=0.5,
        torque_nm=0.5,
    )

    op = PlannedOperation(
        id="op_rough_boss",
        purpose=OpPurpose.ROUGHING,
        feature=feat,
        tool=tool,
        params=params,
    )

    strat_ctx = StrategyContext(
        context=PlanningContext(
            model=None,
            setups=[setup],
            machine=get_default_machine(),
            tools=[tool],
            material=DEFAULT_MATERIALS["aluminum_6061"],
        ),
        setup=setup,
        mesh=TriangleMesh(triangles=np.zeros((0, 3, 3))),
        stock_voxels=VoxelStock.from_bounds(stock.bounds_min, stock.bounds_max, resolution=0.5),
        clearance_z=10.0,
        finish_allowance_wall=0.1,
        finish_allowance_floor=0.0,
        ramp_angle_deg=15.0,
    )

    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    # Verify that toolpath contains tool-center coordinates extending beyond finished geometry
    cut_pts = [s.start for s in tp.segments if s.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW)]
    max_reach = max(abs(p[0]) for p in cut_pts)
    assert max_reach > 2.0, f"Expected tool-center passes beyond Ø0.90 mm, got {max_reach:.2f}"

    # Validate with ToolpathValidator
    validator = ToolpathValidator(
        strat_ctx.context, setup, strat_ctx.mesh, strat_ctx.clearance_z,
        finish_allowance_wall=0.1, finish_allowance_floor=0.0
    )
    report = validator.validate(tp)

    assert report.passed is True
    assert len(report.errors) == 0


def test_validator_rejects_roughing_outside_stock():
    """Roughing pass that extends beyond the stock envelope must be caught and rejected."""
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0, flute_length=15.0, overall_length=50.0, flutes=4)

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"])
    validator = ToolpathValidator(ctx, setup, TriangleMesh(triangles=np.zeros((0, 3, 3))), clearance_z=15.0)

    # Toolpath that cuts at X=20.0mm (far outside 5.0mm stock)
    tp = Toolpath(operation_id="op_rough_out_of_bounds", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([18.0, 0.0, 5.0]), end=np.array([22.0, 0.0, 5.0]), feed=500.0, tool_id="T01", operation_id="op_rough_out_of_bounds", feature_id="f1"))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("roughing swept volume exceeds stock boundary" in str(e.message) or "extends outside the machining stock region" in str(e.message) for e in report.errors)


def test_validator_rejects_zero_intersection_air_cut():
    """A roughing toolpath operating entirely in free air with 0% stock intersection is rejected."""
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0, flute_length=15.0, overall_length=50.0, flutes=4)

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"])
    validator = ToolpathValidator(ctx, setup, TriangleMesh(triangles=np.zeros((0, 3, 3))), clearance_z=15.0)

    # Toolpath entirely outside Z stock range
    tp = Toolpath(operation_id="op_rough_air", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 0.0, 14.0]), end=np.array([2.0, 0.0, 14.0]), feed=500.0, tool_id="T01", operation_id="op_rough_air", feature_id="f1"))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("0% intersection with stock" in str(e.message) or "exceeds stock boundary" in str(e.message) for e in report.errors)


def test_validator_detects_roughing_gouge_and_finish_allowance_violation():
    """Roughing toolpath that cuts into finished part mesh or finish floor allowance is detected."""
    # Model mesh with a flat floor at Z = 5.0
    triangles = np.array([
        [[-10.0, -10.0, 5.0], [10.0, -10.0, 5.0], [10.0, 10.0, 5.0]],
        [[-10.0, -10.0, 5.0], [10.0, 10.0, 5.0], [-10.0, 10.0, 5.0]],
    ])
    mesh = TriangleMesh(triangles=triangles)

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-10.0, -10.0, 0.0]), bounds_max=np.array([10.0, 10.0, 15.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0, flute_length=15.0, overall_length=50.0, flutes=4)

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"])

    # 1. Direct model gouge (cutting at Z = 4.0, which is 1.0mm below floor Z = 5.0)
    validator_gouge = ToolpathValidator(ctx, setup, mesh, clearance_z=20.0, finish_allowance_floor=0.2)
    tp_gouge = Toolpath(operation_id="op_rough_gouge", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp_gouge.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([-2.0, 0.0, 4.0]), end=np.array([2.0, 0.0, 4.0]), feed=500.0, tool_id="T01", operation_id="op_rough_gouge", feature_id="f1"))

    report = validator_gouge.validate(tp_gouge)
    assert report.passed is False
    assert any("gouges" in str(e.message).lower() for e in report.errors)

    # 2. Finish allowance violation (cutting at Z = 5.1 when finish floor allowance is 0.3mm -> requires Z >= 5.3)
    validator_allowance = ToolpathValidator(ctx, setup, mesh, clearance_z=20.0, finish_allowance_floor=0.3)
    tp_allowance = Toolpath(operation_id="op_rough_allowance", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp_allowance.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([-2.0, 0.0, 5.1]), end=np.array([2.0, 0.0, 5.1]), feed=500.0, tool_id="T01", operation_id="op_rough_allowance", feature_id="f1"))

    report2 = validator_allowance.validate(tp_allowance)
    assert report2.passed is False
    assert any("finish floor allowance" in str(e.message).lower() or "gouges" in str(e.message).lower() for e in report2.errors)


def test_validator_detects_excessive_stepover_leaving_uncut_material():
    """Concentric roughing passes where radial stepover exceeds cutter diameter leave uncut material webs."""
    stock = Stock.from_cylinder(center_xy=np.zeros(2), z_min=0.0, z_max=10.0, radius=10.0)
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0, flute_length=15.0, overall_length=50.0, flutes=4)

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"])
    validator = ToolpathValidator(ctx, setup, TriangleMesh(triangles=np.zeros((0, 3, 3))), clearance_z=20.0)

    # Two concentric rings with ring 1 at R=2.0 and ring 2 at R=6.0 (stepover = 4.0mm > cutter dia 3.0mm)
    tp = Toolpath(operation_id="op_rough_excess_stepover", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp.metadata["strategy"] = "boss_circular_roughing"

    # Ring 1: R=2.0
    tp.add(MotionSegment(motion_type=MotionType.ARC_CW, start=np.array([-2.0, 0.0, 5.0]), end=np.array([2.0, 0.0, 5.0]), arc_center=np.array([0.0, 0.0, 5.0]), arc_ccw=False, feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))
    tp.add(MotionSegment(motion_type=MotionType.ARC_CW, start=np.array([2.0, 0.0, 5.0]), end=np.array([-2.0, 0.0, 5.0]), arc_center=np.array([0.0, 0.0, 5.0]), arc_ccw=False, feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))

    # Link move
    tp.add(MotionSegment(motion_type=MotionType.RETRACT, start=np.array([-2.0, 0.0, 5.0]), end=np.array([-2.0, 0.0, 20.0]), tool_id="T01", operation_id="op1", feature_id="f1"))
    tp.add(MotionSegment(motion_type=MotionType.RAPID, start=np.array([-2.0, 0.0, 20.0]), end=np.array([-6.0, 0.0, 20.0]), tool_id="T01", operation_id="op1", feature_id="f1"))
    tp.add(MotionSegment(motion_type=MotionType.PLUNGE, start=np.array([-6.0, 0.0, 20.0]), end=np.array([-6.0, 0.0, 5.0]), feed=200.0, tool_id="T01", operation_id="op1", feature_id="f1"))

    # Ring 2: R=6.0
    tp.add(MotionSegment(motion_type=MotionType.ARC_CW, start=np.array([-6.0, 0.0, 5.0]), end=np.array([6.0, 0.0, 5.0]), arc_center=np.array([0.0, 0.0, 5.0]), arc_ccw=False, feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))
    tp.add(MotionSegment(motion_type=MotionType.ARC_CW, start=np.array([6.0, 0.0, 5.0]), end=np.array([-6.0, 0.0, 5.0]), arc_center=np.array([0.0, 0.0, 5.0]), arc_ccw=False, feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("stepover" in str(e.message).lower() and "uncut" in str(e.message).lower() for e in report.errors)


def test_validator_detects_self_intersecting_roughing_loop():
    """Roughing loops with figure-8 or self-intersecting paths must be rejected."""
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-10.0, -10.0, 0.0]), bounds_max=np.array([10.0, 10.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0, flute_length=15.0, overall_length=50.0, flutes=4)

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"])
    validator = ToolpathValidator(ctx, setup, TriangleMesh(triangles=np.zeros((0, 3, 3))), clearance_z=20.0)

    # Figure-8 self-intersecting loop: (0,0) -> (4,4) -> (4,0) -> (0,4) -> (0,0)
    tp = Toolpath(operation_id="op_rough_self_intersect", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 0.0, 5.0]), end=np.array([4.0, 4.0, 5.0]), feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([4.0, 4.0, 5.0]), end=np.array([4.0, 0.0, 5.0]), feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([4.0, 0.0, 5.0]), end=np.array([0.0, 4.0, 5.0]), feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 4.0, 5.0]), end=np.array([0.0, 0.0, 5.0]), feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("self-intersecting" in str(e.message).lower() for e in report.errors)


def test_validator_detects_fixture_collision_during_roughing():
    """Roughing pass intersecting a setup fixture / clamp must be flagged as collision."""
    class MockFixture:
        id = "clamp_01"
        bounds_min = np.array([2.0, 2.0, 0.0])
        bounds_max = np.array([6.0, 6.0, 10.0])

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-10.0, -10.0, 0.0]), bounds_max=np.array([10.0, 10.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock, fixtures=[MockFixture()])
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=3.0, flute_length=15.0, overall_length=50.0, flutes=4)

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"])
    validator = ToolpathValidator(ctx, setup, TriangleMesh(triangles=np.zeros((0, 3, 3))), clearance_z=20.0)

    # Toolpath cutting directly through fixture bounds at (4.0, 4.0, 5.0)
    tp = Toolpath(operation_id="op_rough_fixture_hit", feature_id="f1", tool_id="T01", purpose="roughing", space=CoordSpace.SETUP)
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 4.0, 5.0]), end=np.array([8.0, 4.0, 5.0]), feed=500.0, tool_id="T01", operation_id="op1", feature_id="f1"))

    report = validator.validate(tp)
    assert report.passed is False
    assert any("fixture" in str(e.message).lower() for e in report.errors)

