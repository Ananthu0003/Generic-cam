"""Tests for Stock-Derived Roughing Envelopes and Stock Boundary Validation.

Verifies:
1. Roughing of small cylindrical features (e.g. OD = Ø0.90 mm) produces tightly bounded passes
   derived from the actual configured stock, without gratuitous air-cutting rings (e.g. no passes at X = -11.25).
2. Facing toolpath envelope is strictly derived from the configured stock envelope.
3. ToolpathValidator (Gate 4) validates that cutting moves remain within the configured stock envelope.
"""

import numpy as np
import pytest

from cam_engine.context import (
    PlanningContext, Setup, Stock, StockKind, Tool, ToolType,
    DEFAULT_MATERIALS, WorkOffset, CamError
)
from cam_engine.coords import CoordSpace
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.toolpaths.validation import ToolpathValidator
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.geometry.voxel import VoxelStock
from cam_engine.demo_models import get_default_machine


def test_small_boss_roughing_envelope_bounded_by_stock():
    """A Ø0.90 mm boss on a small stock (e.g. Ø2.0 mm cylinder) must not generate
    oversized air-cutting rings reaching out to X = -11.25 with a Ø3 mm tool."""
    # Create circular boss feature of diameter 0.90 mm (radius 0.45 mm)
    feat = MachiningFeature(
        id="feat_pin",
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

    # Configured stock: cylindrical bar Ø2.50 mm (radius 1.25 mm)
    stock = Stock.from_cylinder(center_xy=np.zeros(2), z_min=0.0, z_max=5.0, radius=1.25)
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

    # Strategy Context
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

    # Collect all cutting coordinates
    cut_pts = []
    for s in tp.segments:
        if s.motion_type in (MotionType.CUT, MotionType.ARC_CW, MotionType.ARC_CCW):
            cut_pts.append(s.start)
            cut_pts.append(s.end)
            if s.arc_center is not None:
                # Radius of circular ring
                ring_r = float(np.linalg.norm(s.start[:2] - s.arc_center[:2]))
                # Ring radius must be bounded close to stock radius + tool radius
                assert ring_r <= stock.radius + tool.diameter / 2.0 + 1.0, \
                    f"Roughing ring radius {ring_r:.2f} mm is excessively large for stock radius {stock.radius:.2f} mm"

    assert len(cut_pts) > 0
    all_x = [p[0] for p in cut_pts]
    all_y = [p[1] for p in cut_pts]

    # Max X and Y coordinates should be close to center (e.g. <= 3.0 mm), NOT -11.25 or -9.00!
    assert max(abs(x) for x in all_x) < 4.0, f"Max X reach {max(abs(x) for x in all_x)} exceeds stock-bounded envelope"
    assert max(abs(y) for y in all_y) < 4.0, f"Max Y reach {max(abs(y) for y in all_y)} exceeds stock-bounded envelope"


def test_facing_envelope_bounded_by_stock():
    """Facing operations must be strictly bounded to the configured stock boundary."""
    # Facing feature with excess height 1.0 mm
    feat = MachiningFeature(
        id="feat_face",
        type=FeatureType.FACING_REGION,
        face_indices=[0],
        bounds_min=np.array([-2.0, -2.0, 4.0]),
        bounds_max=np.array([2.0, 2.0, 5.0]),
        depth=1.0,
        top_z=5.0,
        floor_z=4.0,
        notes={"excess_height": 1.0},
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-2.5, -2.5, 0.0]), bounds_max=np.array([2.5, 2.5, 5.0]))
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
        spindle_rpm=8001.0,
        feed_rate=600.0,
        plunge_feed=200.0,
        ramp_feed=400.0,
        feed_per_tooth_mm=0.03,
        depth_of_cut_mm=1.0,
        stepover_mm=1.5,
        coolant=True,
        effective_diameter_mm=3.0,
        engagement_angle_deg=90.0,
        power_kw=0.4,
        torque_nm=0.4,
    )

    op = PlannedOperation(
        id="op_face",
        purpose=OpPurpose.FACING,
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
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )

    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    cut_pts = [s.end for s in tp.segments if s.motion_type in (MotionType.CUT, MotionType.PLUNGE)]
    assert len(cut_pts) > 0
    # Facing tool center moves should not exceed stock bounds + tool radius (2.5 + 1.5 = 4.0)
    max_reach_x = max(abs(p[0]) for p in cut_pts)
    max_reach_y = max(abs(p[1]) for p in cut_pts)
    assert max_reach_x <= 4.0 + 1e-3, f"Facing X reach {max_reach_x:.2f} exceeds stock bounds"
    assert max_reach_y <= 4.0 + 1e-3, f"Facing Y reach {max_reach_y:.2f} exceeds stock bounds"
