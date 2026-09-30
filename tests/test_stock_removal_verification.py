"""Tests for stock-removal verification and rest machining skipping.

Validates that:
1. After every operation, RemainingStock is maintained and compared with TargetCAD.
2. RemainingStock ∩ TargetRemovalVolume determines whether rest machining is actually required.
3. If no material remains: REST OPERATION → SKIP (no empty operation or tool change emitted).
4. If material remains: REST OPERATION → generates only remaining volume.
5. Setup 2 empty op_rest_001 is cleanly skipped without ghost tool changes or G-code blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from cam_engine.geometry.voxel import VoxelStock
from cam_engine.context import (
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
    get_material,
)
from cam_engine.demo_models import get_default_machine, get_default_tools
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.post import get_post_processor
from cam_engine.machinability import CuttingParameters
from cam_engine.geometry import TriangleMesh, Shape, Face, SurfaceKind


def _make_dummy_op(
    op_id: str,
    feature: MachiningFeature,
    tool: Tool,
    purpose: OpPurpose = OpPurpose.REST_MACHINING,
) -> PlannedOperation:
    params = CuttingParameters(
        spindle_rpm=5000.0,
        feed_rate=600.0,
        plunge_feed=200.0,
        ramp_feed=300.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=2.0,
        stepover_mm=tool.diameter * 0.5,
        coolant=True,
        effective_diameter_mm=tool.diameter,
        engagement_angle_deg=90.0,
        power_kw=1.0,
        torque_nm=1.0,
    )
    return PlannedOperation(
        id=op_id,
        purpose=purpose,
        feature=feature,
        tool=tool,
        params=params,
    )


def _make_dummy_context(setup: Setup) -> PlanningContext:
    class DummyModel:
        shape = None
    return PlanningContext(
        model=DummyModel(),
        setups=[setup],
        machine=get_default_machine(),
        tools=get_default_tools(),
        material=get_material("aluminum_6061"),
    )


def test_remaining_stock_intersection_and_removal_volume():
    """Verify RemainingStock ∩ TargetRemovalVolume calculations in VoxelStock."""
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([20.0, 20.0, 10.0])
    res = 1.0

    raw_stock = VoxelStock.from_bounds(bmin, bmax, resolution=res)
    assert raw_stock.remaining_volume() == 4000.0

    # Target CAD is a cube from [5,5,0] to [15,15,5] (10x10x5 = 500mm^3)
    target_cad = VoxelStock.from_bounds(bmin, bmax, resolution=res)
    target_cad.voxels[:] = False
    target_cad.voxels[5:15, 5:15, 0:5] = True

    # TargetRemovalVolume = raw_stock MINUS target_cad
    target_removal = raw_stock.compute_removal_volume(target_cad)
    assert target_removal.remaining_volume() == pytest.approx(4000.0 - 500.0)

    # Initially, RemainingStock is identical to raw_stock
    rem_stock = VoxelStock.from_bounds(bmin, bmax, resolution=res)
    assert rem_stock.has_remaining_removal(target_removal) is True
    assert rem_stock.remaining_removal_volume(target_removal) == pytest.approx(3500.0)

    # Machine away all removal volume
    rem_stock.voxels[~target_cad.voxels] = False
    assert rem_stock.remaining_removal_volume(target_removal) == 0.0
    assert rem_stock.has_remaining_removal(target_removal) is False


def test_rest_operation_skipped_when_no_material_remains():
    """When a feature has no remaining stock above target floor, REST OPERATION must be skipped."""
    bmin = np.array([0.0, 0.0, -10.0])
    bmax = np.array([50.0, 50.0, 0.0])
    stock = Stock(kind=StockKind.BOX, bounds_min=bmin, bounds_max=bmax)
    setup = Setup(
        id="s1",
        name="Setup 1",
        model_to_setup=np.eye(4),
        stock=stock,
        work_offset=WorkOffset.G54,
    )
    context = _make_dummy_context(setup)

    pocket_feat = MachiningFeature(
        id="pocket_01",
        canonical_id="pocket_01",
        type=FeatureType.POCKET,
        face_indices=[],
        bounds_min=np.array([10.0, 10.0, -5.0]),
        bounds_max=np.array([40.0, 40.0, 0.0]),
        depth=5.0,
        top_z=0.0,
        floor_z=-5.0,
    )

    rest_tool = Tool(
        id="T03",
        tool_number=3,
        type=ToolType.FLAT_ENDMILL,
        diameter=4.0,
        corner_radius=0.0,
        flute_length=15.0,
        overall_length=50.0,
        flutes=3,
    )

    # In-process stock where pocket has ALREADY been cut down to floor_z (-5.0)
    stock_voxels = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    # Clear voxels inside pocket down to floor_z
    p_min_idx = stock_voxels.world_to_index(pocket_feat.bounds_min)
    p_max_idx = stock_voxels.world_to_index(pocket_feat.bounds_max)
    stock_voxels.voxels[
        p_min_idx[0]:p_max_idx[0],
        p_min_idx[1]:p_max_idx[1],
        p_min_idx[2]:p_max_idx[2] + 1,
    ] = False

    strat_ctx = StrategyContext(
        context=context,
        setup=setup,
        mesh=None,
        stock_voxels=stock_voxels,
        clearance_z=10.0,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    rest_op = _make_dummy_op("op_rest_001", pocket_feat, rest_tool, OpPurpose.REST_MACHINING)

    tp = engine.generate(rest_op)
    assert tp.metadata.get("skipped") is not None
    assert tp.is_empty_or_skipped is True
    assert tp.has_cutting_motion is False
    assert len(tp.segments) == 0


def test_rest_operation_generates_when_material_remains():
    """When material remains above target floor, REST OPERATION generates cutting passes."""
    bmin = np.array([0.0, 0.0, -10.0])
    bmax = np.array([50.0, 50.0, 0.0])
    stock = Stock(kind=StockKind.BOX, bounds_min=bmin, bounds_max=bmax)
    setup = Setup(
        id="s1",
        name="Setup 1",
        model_to_setup=np.eye(4),
        stock=stock,
        work_offset=WorkOffset.G54,
    )
    context = _make_dummy_context(setup)

    pocket_feat = MachiningFeature(
        id="pocket_01",
        canonical_id="pocket_01",
        type=FeatureType.POCKET,
        face_indices=[],
        bounds_min=np.array([10.0, 10.0, -5.0]),
        bounds_max=np.array([40.0, 40.0, 0.0]),
        depth=5.0,
        top_z=0.0,
        floor_z=-5.0,
    )

    rest_tool = Tool(
        id="T03",
        tool_number=3,
        type=ToolType.FLAT_ENDMILL,
        diameter=4.0,
        corner_radius=0.0,
        flute_length=15.0,
        overall_length=50.0,
        flutes=3,
    )

    # In-process stock still contains raw material in the pocket (Z=0 down to -5)
    stock_voxels = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)

    strat_ctx = StrategyContext(
        context=context,
        setup=setup,
        mesh=None,
        stock_voxels=stock_voxels,
        clearance_z=10.0,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    rest_op = _make_dummy_op("op_rest_001", pocket_feat, rest_tool, OpPurpose.REST_MACHINING)

    tp = engine.generate(rest_op)
    assert tp.metadata.get("skipped") is None
    assert tp.is_empty_or_skipped is False
    assert tp.has_cutting_motion is True
    assert tp.cutting_length() > 0.0


def test_post_processor_skips_empty_or_skipped_toolpaths():
    """Post-processor should not emit tool changes or operations for skipped toolpaths."""
    tool1 = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=10.0, corner_radius=0.0, flute_length=30.0, overall_length=60.0, flutes=3)
    tool2 = Tool(id="T02", tool_number=2, type=ToolType.FLAT_ENDMILL, diameter=3.0, corner_radius=0.0, flute_length=10.0, overall_length=40.0, flutes=2)

    tp_active = Toolpath(
        operation_id="op_rough_001",
        feature_id="pocket_01",
        tool_id="T01",
        purpose="roughing",
        segments=[
            MotionSegment(
                motion_type=MotionType.CUT,
                start=np.array([10.0, 10.0, -2.0]),
                end=np.array([30.0, 10.0, -2.0]),
                feed=600.0,
            )
        ],
    )

    tp_skipped = Toolpath(
        operation_id="op_rest_001",
        feature_id="pocket_01",
        tool_id="T02",
        purpose="rest_machining",
        segments=[],
        metadata={"skipped": "no remaining stock in feature"},
    )

    post = get_post_processor(controller="fanuc", work_offset="G54")
    result = post.post_process([tp_active, tp_skipped], program_name="TEST_POST")
    gcode = result.gcode

    assert "op_rough_001" in gcode
    assert "T01" in gcode
    assert "op_rest_001" not in gcode
    assert "T02" not in gcode  # Ghost tool T02 is NOT called!


def test_multi_setup_stock_transfer_skips_empty_setup2_rest_operation():
    """Verify that when Setup 1 evacuates material, transferred stock in Setup 2
    causes Setup 2 rest machining to be skipped (fixing empty Setup 2 op_rest_001)."""
    bmin = np.array([0.0, 0.0, -10.0])
    bmax = np.array([50.0, 50.0, 0.0])
    stock = Stock(kind=StockKind.BOX, bounds_min=bmin, bounds_max=bmax)

    # Setup 1 (G54 top)
    s1 = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), stock=stock, work_offset=WorkOffset.G54)
    # Setup 2 (G55 bottom: flipped 180 around X)
    M2 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -10.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    s2 = Setup(id="s2", name="Setup 2", model_to_setup=M2, stock=stock, work_offset=WorkOffset.G55)

    context = _make_dummy_context(s2)

    # Setup 1 machined away the feature completely in Setup 1 space
    stock_s1 = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    # Clear all material in the feature region [10:40, 10:40, -5:0]
    stock_s1.voxels[10:40, 10:40, 5:10] = False

    # Transfer stock from Setup 1 to Setup 2
    T_1_to_2 = s2.model_to_setup @ np.linalg.inv(s1.model_to_setup)
    stock_s2 = stock_s1.transformed(T_1_to_2, bmin, bmax, resolution=1.0)

    pocket_s2 = MachiningFeature(
        id="pocket_s2",
        canonical_id="pocket_s2",
        type=FeatureType.POCKET,
        face_indices=[],
        bounds_min=np.array([10.0, 10.0, -5.0]),
        bounds_max=np.array([40.0, 40.0, 0.0]),
        depth=5.0,
        top_z=0.0,
        floor_z=-5.0,
    )

    rest_tool = Tool(
        id="T03",
        tool_number=3,
        type=ToolType.FLAT_ENDMILL,
        diameter=4.0,
        corner_radius=0.0,
        flute_length=15.0,
        overall_length=50.0,
        flutes=3,
    )

    strat_ctx = StrategyContext(
        context=context,
        setup=s2,
        mesh=None,
        stock_voxels=stock_s2,
        clearance_z=10.0,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    rest_op_s2 = _make_dummy_op("op_rest_s2_001", pocket_s2, rest_tool, OpPurpose.REST_MACHINING)

    tp_s2 = engine.generate(rest_op_s2)
    assert tp_s2.is_empty_or_skipped is True
    assert tp_s2.metadata.get("skipped") is not None

    post = get_post_processor(controller="fanuc", work_offset="G55")
    result = post.post_process([tp_s2], program_name="TEST_S2")
    assert "op_rest_s2_001" not in result.gcode
    assert "T03" not in result.gcode

