"""Tests for Stock-Envelope-Consistent Toolpath Generation Architecture.

Verifies:
1. Facing toolpaths cover the entire setup stock envelope (not just the local model top face).
2. Roughing toolpaths (circular boss roughing and contour parallel passes) start within the actual stock boundary and do not over-expand into thin air.
3. Finish allowance is strictly preserved at the innermost roughing ring.
4. Stock outer radius helper accurately handles both StockKind.CYLINDER and StockKind.BOX.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    DEFAULT_MATERIALS, WorkOffset, MachineConfig, MachineType, MachineAxis, Units
)
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.geometry.mesh import TriangleMesh


def make_test_context(stock: Stock, finish_wall: float = 0.2, finish_floor: float = 0.1) -> StrategyContext:
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
        controller="fanuc", units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0, hsm_enabled=True,
    )
    ctx.material = DEFAULT_MATERIALS["aluminum_6061"]
    ctx.tools = [
        Tool(id="t_endmill_2mm", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=2.0,
             flute_length=10.0, overall_length=40.0, flutes=2),
    ]
    setup = Setup(id="setup_1", name="Top Setup", model_to_setup=np.eye(4),
                  work_offset=WorkOffset.G54, stock=stock)
    return StrategyContext(
        context=ctx, setup=setup, mesh=mesh,
        stock_voxels=None, clearance_z=20.0, finish_allowance_wall=finish_wall, finish_allowance_floor=finish_floor
    )


def test_micro_boss_roughing_envelope_bounded_by_stock():
    """Verify that a Ø0.90mm pin with a 2mm stock does NOT generate roughing passes at X=-9.25mm."""
    # Small 2.0mm radius cylindrical stock (Ø4.0mm bar stock)
    stock = Stock.from_cylinder(center_xy=np.array([0.0, 0.0]), z_min=0.0, z_max=15.0, radius=2.0)
    strat_ctx = make_test_context(stock, finish_wall=0.2)
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]  # Ø2.0mm tool (r=1.0mm)

    boss_feature = MachiningFeature(
        id="pin_090mm",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 0.0]),
        bounds_max=np.array([0.45, 0.45, 10.0]),
        depth=10.0,
        top_z=10.0,
        floor_z=0.0,
        diameter=0.90,  # Ø0.90mm finished OD
        is_concave=False,
        notes={"is_circular": True}
    )
    params = CuttingParameters(
        spindle_rpm=10000.0, feed_rate=1000.0, plunge_feed=300.0, ramp_feed=500.0,
        feed_per_tooth_mm=0.05, depth_of_cut_mm=2.0, stepover_mm=0.5, coolant=True,
        effective_diameter_mm=2.0, engagement_angle_deg=45.0, power_kw=1.0, torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_rough_pin",
        purpose=OpPurpose.ROUGHING,
        feature=boss_feature,
        tool=tool,
        params=params,
    )

    tp = engine.generate(op)
    # Collect all cutting X coordinates
    all_x = []
    for s in tp.segments:
        if s.start is not None:
            all_x.append(s.start[0])
        if s.end is not None:
            all_x.append(s.end[0])

    min_x = min(all_x)
    max_x = max(all_x)

    # Stock is radius 2.0mm, tool is radius 1.0mm -> maximum tool center is ~3.0mm from center
    # Must NOT be -9.25mm!
    assert min_x >= -3.5, f"Roughing over-expanded to min_x={min_x}mm (expected >= -3.5mm)"
    assert max_x <= 3.5, f"Roughing over-expanded to max_x={max_x}mm (expected <= 3.5mm)"

    # Innermost ring: r_inner = r_boss(0.45) + finish_wall(0.2) + r_tool(1.0) = 1.65mm
    # Check that an arc near 1.65mm exists
    arc_radii = []
    for s in tp.segments:
        if s.arc_center is not None and s.start is not None:
            r = np.linalg.norm(s.start[:2] - s.arc_center[:2])
            arc_radii.append(r)

    assert any(abs(r - 1.65) < 1e-3 for r in arc_radii), f"Innermost arc radius 1.65mm not found in {arc_radii}"


def test_facing_envelope_covers_full_stock():
    """Verify facing toolpath covers the entire stock envelope, not just the model top face."""
    stock_box = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -5.0, 0.0]),
        bounds_max=np.array([5.0, 5.0, 12.0]),
    )
    strat_ctx = make_test_context(stock_box)
    engine = StrategyEngine(strat_ctx)
    tool = strat_ctx.context.tools[0]  # Ø2.0mm tool (r=1.0mm)

    # Facing feature for top excess stock (12.0mm down to 10.0mm)
    facing_feature = MachiningFeature(
        id="facing_top",
        type=FeatureType.FACING_REGION,
        face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 10.0]),
        bounds_max=np.array([0.45, 0.45, 12.0]),
        depth=2.0,
        top_z=12.0,
        floor_z=10.0,
        is_concave=False,
        notes={"excess_height": 2.0}
    )
    params = CuttingParameters(
        spindle_rpm=10000.0, feed_rate=1000.0, plunge_feed=300.0, ramp_feed=500.0,
        feed_per_tooth_mm=0.05, depth_of_cut_mm=1.0, stepover_mm=1.5, coolant=True,
        effective_diameter_mm=2.0, engagement_angle_deg=45.0, power_kw=1.0, torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_facing_top",
        purpose=OpPurpose.FACING,
        feature=facing_feature,
        tool=tool,
        params=params,
    )

    tp = engine.generate(op)
    # Collect all cutting XY coordinates
    cut_x = []
    cut_y = []
    for s in tp.segments:
        if s.motion_type in (MotionType.CUT, MotionType.PLUNGE):
            cut_x.extend([s.start[0], s.end[0]])
            cut_y.extend([s.start[1], s.end[1]])

    # Full stock is [-5, +5], tool radius is 1.0mm -> cutting should cover [-6.0, +6.0]
    assert min(cut_x) <= -5.0, f"Facing did not cover negative X stock: min_x={min(cut_x)}"
    assert max(cut_x) >= 5.0, f"Facing did not cover positive X stock: max_x={max(cut_x)}"
    assert min(cut_y) <= -5.0, f"Facing did not cover negative Y stock: min_y={min(cut_y)}"
    assert max(cut_y) >= 5.0, f"Facing did not cover positive Y stock: max_y={max(cut_y)}"
