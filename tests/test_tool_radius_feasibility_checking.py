"""Tests for Automatic Tool-Radius & Envelope Feasibility Checking.

Verifies:
1. Minimum feature radius / hole diameter checking.
2. Minimum channel / slot width checking.
3. Minimum corner / fillet radius checking.
4. Tool reach / flute length checking.
5. Tool holder envelope collision checking.
6. Generic diagnostic formatting:
   - Tool ØX cannot machine required feature
   - Required minimum access radius = Y
   - Available tool = ØX
   - Recommended action = select smaller tool / different orientation
7. Strict policy: Invalid toolpaths are NEVER generated when feasibility fails.
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
    ToolHolder,
    ToolType,
    WorkOffset,
    DEFAULT_MATERIALS,
)
from cam_engine.errors import CamError, NO_COMPATIBLE_TOOL, TOOL_REACH_INSUFFICIENT, COLLISION_DETECTED
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.validation import PreToolpathGeometryValidator


def _make_env(tool: Tool) -> tuple[StrategyEngine, PreToolpathGeometryValidator]:
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-50.0, -50.0, -30.0]),
        bounds_max=np.array([50.0, 50.0, 10.0]),
    )
    setup = Setup(
        id="setup_01", name="Setup 1",
        model_to_setup=np.eye(4), work_offset=WorkOffset.G54,
        stock=stock,
    )
    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.material = DEFAULT_MATERIALS["aluminum_6061"]
    ctx.tools = [tool]

    mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=mesh, stock_voxels=None,
        clearance_z=20.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    validator = PreToolpathGeometryValidator(context=ctx, setup=setup)
    return engine, validator


def test_oversized_tool_in_hole():
    """Verify Tool Ø12.7 cannot machine hole Ø6.0 and returns explicit access radius."""
    tool_12_7 = Tool(
        id="T_12_7", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=12.7, flute_length=30.0, overall_length=75.0,
    )
    engine, validator = _make_env(tool_12_7)

    feat_hole = MachiningFeature(
        id="hole_dia6",
        type=FeatureType.BLIND_HOLE,
        face_indices=[1],
        bounds_min=np.array([-3.0, -3.0, -10.0]),
        bounds_max=np.array([3.0, 3.0, 0.0]),
        depth=10.0,
        top_z=0.0,
        floor_z=-10.0,
        diameter=6.0, # Required radius = 3.0 mm
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=500.0, plunge_feed=250.0,
        ramp_feed=350.0, depth_of_cut_mm=2.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=12.7, engagement_angle_deg=90.0,
        power_kw=1.0, torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_hole_12_7",
        purpose=OpPurpose.ROUGHING,
        feature=feat_hole,
        tool=tool_12_7,
        params=params,
    )

    # 1. Validator level check
    errors = validator.check_tool_feasibility(tool_12_7, feat_hole, op_purpose=op.purpose, op_id=op.id)
    assert len(errors) > 0
    assert errors[0].code == NO_COMPATIBLE_TOOL
    msg = errors[0].message
    assert "Tool Ø12.70 cannot machine required feature 'hole_dia6'" in msg
    assert "Required minimum access radius = 3.00 mm" in msg
    assert "Available tool = Ø12.70" in msg
    assert "Recommended action = select smaller tool (Ø <= 6.00 mm)" in msg

    # 2. StrategyEngine level refusal (DO NOT generate invalid toolpath)
    with pytest.raises(CamError) as exc_info:
        engine.generate(op)
    assert exc_info.value.code == NO_COMPATIBLE_TOOL
    assert "Required minimum access radius = 3.00 mm" in str(exc_info.value)


def test_oversized_tool_in_channel_slot():
    """Verify Tool Ø12.7 cannot machine narrow channel width 8.0 mm."""
    tool_12_7 = Tool(
        id="T_12_7", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=12.7, flute_length=30.0, overall_length=75.0,
    )
    engine, validator = _make_env(tool_12_7)

    feat_slot = MachiningFeature(
        id="slot_narrow",
        type=FeatureType.SLOT,
        face_indices=[2],
        bounds_min=np.array([-4.0, -25.0, -8.0]),
        bounds_max=np.array([4.0, 25.0, 0.0]), # width = 8.0 mm, length = 50.0 mm
        depth=8.0,
        top_z=0.0,
        floor_z=-8.0,
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=500.0, plunge_feed=250.0,
        ramp_feed=350.0, depth_of_cut_mm=2.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=12.7, engagement_angle_deg=90.0,
        power_kw=1.0, torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_slot_12_7",
        purpose=OpPurpose.ROUGHING,
        feature=feat_slot,
        tool=tool_12_7,
        params=params,
    )

    errors = validator.check_tool_feasibility(tool_12_7, feat_slot, op_purpose=op.purpose, op_id=op.id)
    assert len(errors) > 0
    assert errors[0].code == NO_COMPATIBLE_TOOL
    assert "channel width 8.00 mm" in errors[0].message
    assert "Required minimum access radius = 4.00 mm" in errors[0].message
    assert "Recommended action = select smaller tool (Ø <= 8.00 mm)" in errors[0].message

    with pytest.raises(CamError):
        engine.generate(op)


def test_oversized_tool_in_internal_corner_finishing():
    """Verify Tool Ø12.7 (R=6.35 mm) cannot finish internal corner radius R=2.0 mm."""
    tool_12_7 = Tool(
        id="T_12_7", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=12.7, flute_length=30.0, overall_length=75.0,
    )
    engine, validator = _make_env(tool_12_7)

    feat_pocket = MachiningFeature(
        id="pocket_corner",
        type=FeatureType.POCKET,
        face_indices=[3],
        bounds_min=np.array([-20.0, -20.0, -10.0]),
        bounds_max=np.array([20.0, 20.0, 0.0]),
        depth=10.0,
        top_z=0.0,
        floor_z=-10.0,
        fillet_radius=2.0, # Internal corner blend radius R=2.0 mm
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=500.0, plunge_feed=250.0,
        ramp_feed=350.0, depth_of_cut_mm=2.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=12.7, engagement_angle_deg=90.0,
        power_kw=1.0, torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_pocket_finish",
        purpose=OpPurpose.FINISHING, # Finishing requires corner access
        feature=feat_pocket,
        tool=tool_12_7,
        params=params,
    )

    errors = validator.check_tool_feasibility(tool_12_7, feat_pocket, op_purpose=op.purpose, op_id=op.id)
    assert len(errors) > 0
    assert errors[0].code == NO_COMPATIBLE_TOOL
    assert "corner radius 2.00 mm" in errors[0].message
    assert "Required minimum access radius = 2.00 mm" in errors[0].message
    assert "Recommended action = select smaller tool (Ø <= 4.00 mm)" in errors[0].message


def test_tool_holder_envelope_collision():
    """Verify Tool Holder Ø40 mm collides when reaching cavity Ø20 mm beyond flute length."""
    holder = ToolHolder(diameter=40.0, length=45.0)
    tool_short = Tool(
        id="T_SHORT", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=10.0, flute_length=15.0, overall_length=50.0, # Flute is only 15mm, cavity depth is 30mm
        holder=holder,
    )
    engine, validator = _make_env(tool_short)

    feat_deep_cavity = MachiningFeature(
        id="cavity_deep",
        type=FeatureType.POCKET,
        face_indices=[4],
        bounds_min=np.array([-10.0, -10.0, -30.0]),
        bounds_max=np.array([10.0, 10.0, 0.0]),
        depth=30.0, # Depth 30mm > Flute length 15mm
        top_z=0.0,
        floor_z=-30.0,
        diameter=20.0, # Opening is Ø20mm < Holder Ø40mm
    )
    params = CuttingParameters(
        spindle_rpm=5000.0, feed_rate=500.0, plunge_feed=250.0,
        ramp_feed=350.0, depth_of_cut_mm=2.0, stepover_mm=1.0,
        feed_per_tooth_mm=0.05, coolant=True,
        effective_diameter_mm=10.0, engagement_angle_deg=90.0,
        power_kw=1.0, torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_cavity_deep",
        purpose=OpPurpose.ROUGHING,
        feature=feat_deep_cavity,
        tool=tool_short,
        params=params,
    )

    errors = validator.check_tool_feasibility(tool_short, feat_deep_cavity, op_purpose=op.purpose, op_id=op.id)
    assert any(e.code == COLLISION_DETECTED for e in errors)
    coll_err = next(e for e in errors if e.code == COLLISION_DETECTED)
    assert "holder (Ø40.00 mm) cannot access required feature" in coll_err.message
    assert "Required minimum access radius = 10.00 mm (holder envelope collision)" in coll_err.message
    assert "Recommended action = increase tool stickout / select smaller holder" in coll_err.message
