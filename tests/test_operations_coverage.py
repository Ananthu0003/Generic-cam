"""Comprehensive tests ensuring 100% operation planning and toolpath generation coverage for all CAD features."""

import numpy as np
import pytest

from cam_engine.context import (
    MachineConfig, MachineAxis, MachineType, PlanningContext, Setup, Stock, StockKind, Tool, ToolType, WorkOffset, DEFAULT_MATERIALS
)
from cam_engine.features import FeatureType, MachiningFeature, Accessibility
from cam_engine.operations import OpPurpose, OperationPlanner, PlannedOperation
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine


def _create_mock_context() -> tuple[PlanningContext, Setup]:
    axes = {
        "X": MachineAxis(name="X", travel_min=-500, travel_max=500, rapid_rate=10000, max_feed=5000),
        "Y": MachineAxis(name="Y", travel_min=-500, travel_max=500, rapid_rate=10000, max_feed=5000),
        "Z": MachineAxis(name="Z", travel_min=-300, travel_max=100, rapid_rate=8001, max_feed=3000),
    }
    machine = MachineConfig(
        id="vmc_1",
        name="3-Axis VMC",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes=axes,
        spindle_min_rpm=100,
        spindle_max_rpm=12000,
        max_spindle_power_kw=15.0,
        safe_retract_height=10.0,
        tool_change_position=np.array([0.0, 0.0, 50.0]),
    )
    material = DEFAULT_MATERIALS["aluminum_6061"]
    tools = [
        Tool(id="T01_FACE", tool_number=1, type=ToolType.FACE_MILL, diameter=50.0, flute_length=15.0, overall_length=80.0, flutes=5),
        Tool(id="T02_ROUGH", tool_number=2, type=ToolType.FLAT_ENDMILL, diameter=12.0, flute_length=35.0, overall_length=75.0, flutes=4, can_plunge=True),
        Tool(id="T03_FINISH", tool_number=3, type=ToolType.FLAT_ENDMILL, diameter=6.0, flute_length=25.0, overall_length=60.0, flutes=4, can_plunge=True),
        Tool(id="T04_BALL", tool_number=4, type=ToolType.BALL_ENDMILL, diameter=6.0, flute_length=20.0, overall_length=60.0, tip_radius=3.0, flutes=2, can_plunge=True),
        Tool(id="T05_DRILL", tool_number=5, type=ToolType.DRILL, diameter=8.5, flute_length=60.0, overall_length=90.0, flutes=2),
        Tool(id="T06_SPOT", tool_number=6, type=ToolType.CHAMFER_MILL, diameter=10.0, flute_length=15.0, overall_length=50.0, flutes=2),
        Tool(id="T07_TAP", tool_number=7, type=ToolType.TAP, diameter=10.0, flute_length=25.0, overall_length=70.0, flutes=3),
        Tool(id="T08_CHAMFER", tool_number=8, type=ToolType.CHAMFER_MILL, diameter=12.0, flute_length=15.0, overall_length=50.0, flutes=2),
        Tool(id="T09_CSINK", tool_number=9, type=ToolType.COUNTERSINK_TOOL, diameter=16.0, flute_length=20.0, overall_length=60.0, flutes=3),
        Tool(id="T10_GROOVE", tool_number=10, type=ToolType.GROOVE_CUTTER, diameter=4.0, flute_length=5.0, overall_length=50.0, flutes=2),
        Tool(id="T11_THREAD", tool_number=11, type=ToolType.THREAD_MILL, diameter=8.0, flute_length=20.0, overall_length=60.0, flutes=3),
        Tool(id="T12_REAMER", tool_number=12, type=ToolType.REAMER, diameter=10.0, flute_length=30.0, overall_length=70.0, flutes=6),
    ]
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-60.0, -60.0, -45.0]),
        bounds_max=np.array([60.0, 60.0, 5.0]),
    )
    setup = Setup(
        id="setup_1",
        name="Setup 1",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
    )
    context = PlanningContext(model=None, machine=machine, material=material, tools=tools, setups=[setup])
    return context, setup


def _create_feature(ftype: FeatureType, fid: str = "feat_1", **kwargs) -> MachiningFeature:
    defaults = {
        "id": fid,
        "type": ftype,
        "face_indices": [0],
        "bounds_min": np.array([-20.0, -20.0, -20.0]),
        "bounds_max": np.array([20.0, 20.0, 0.0]),
        "depth": 20.0,
        "top_z": 0.0,
        "floor_z": -20.0,
        "is_concave": True,
        "accessibility": Accessibility.TOOL_AXIS_OK,
    }
    defaults.update(kwargs)
    return MachiningFeature(**defaults)


def test_all_features_plan_operations_and_generate_toolpaths():
    """Verify that every feature type plans valid operations and produces valid toolpath segments."""
    context, setup = _create_mock_context()

    test_features = [
        _create_feature(FeatureType.FACING_REGION, "f_facing", depth=2.0, top_z=2.0, floor_z=0.0, is_concave=False),
        _create_feature(FeatureType.HOLE, "f_hole", diameter=8.5),
        _create_feature(FeatureType.THROUGH_HOLE, "f_thru_hole", diameter=8.5),
        _create_feature(FeatureType.BLIND_HOLE, "f_blind_hole", diameter=8.5),
        _create_feature(FeatureType.BORE, "f_bore", diameter=25.0),
        _create_feature(FeatureType.THROUGH_BORE, "f_thru_bore", diameter=30.0),
        _create_feature(FeatureType.BLIND_BORE, "f_blind_bore", diameter=25.0),
        _create_feature(FeatureType.COUNTERBORE, "f_cbore", diameter=16.0, depth=5.0, top_z=0.0, floor_z=-5.0),
        _create_feature(FeatureType.COUNTERSINK, "f_csink", diameter=14.0, depth=3.0, top_z=0.0, floor_z=-3.0, notes={"semi_angle_deg": 45.0}),
        _create_feature(FeatureType.TAPPED_HOLE, "f_tap", diameter=10.0, notes={"tap_feed_mm_per_rev": 1.5}),
        _create_feature(FeatureType.THREADED_HOLE, "f_thread", diameter=10.0, notes={"pitch": 1.5}),
        _create_feature(FeatureType.POCKET, "f_pocket", bounds_min=np.array([-25.0, -15.0, -10.0]), bounds_max=np.array([25.0, 15.0, 0.0]), depth=10.0, floor_z=-10.0),
        _create_feature(FeatureType.OPEN_POCKET, "f_open_pocket", depth=10.0, floor_z=-10.0, notes={"is_open": True}),
        _create_feature(FeatureType.SLOT, "f_slot", bounds_min=np.array([-30.0, -4.0, -8.0]), bounds_max=np.array([30.0, 4.0, 0.0]), depth=8.0, floor_z=-8.0),
        _create_feature(FeatureType.OPEN_SLOT, "f_open_slot", bounds_min=np.array([-30.0, -4.0, -8.0]), bounds_max=np.array([30.0, 4.0, 0.0]), depth=8.0, floor_z=-8.0, notes={"is_open": True}),
        _create_feature(FeatureType.STEP, "f_step", depth=15.0, floor_z=-15.0, notes={"is_open": True}),
        _create_feature(FeatureType.SHOULDER, "f_shoulder", depth=15.0, floor_z=-15.0, notes={"is_open": True}),
        _create_feature(FeatureType.BOSS, "f_boss", diameter=20.0, depth=15.0, top_z=0.0, floor_z=-15.0, is_concave=False),
        _create_feature(FeatureType.PLANAR_WALL, "f_planar_wall", bounds_min=np.array([-30.0, -2.0, -15.0]), bounds_max=np.array([30.0, 2.0, 0.0]), depth=15.0, floor_z=-15.0),
        _create_feature(FeatureType.CONTOUR, "f_contour", bounds_min=np.array([-50.0, -50.0, -40.0]), bounds_max=np.array([50.0, 50.0, 0.0]), depth=40.0, floor_z=-40.0, is_concave=False),
        _create_feature(FeatureType.CHAMFER, "f_chamfer", depth=2.0, top_z=0.0, floor_z=-2.0, notes={"angle_from_horizontal": 45.0}),
        _create_feature(FeatureType.FILLET, "f_fillet", radius=3.0, depth=3.0, top_z=0.0, floor_z=-3.0),
        _create_feature(FeatureType.GROOVE, "f_groove", diameter=4.0, depth=3.0, top_z=0.0, floor_z=-3.0),
    ]

    planner = OperationPlanner(features=test_features, context=context, setup=setup)
    planned_ops = planner.plan()

    assert len(planned_ops) > 0, "Operation planner returned no operations!"

    # Verify every feature received at least one planned operation
    planned_feature_ids = {op.feature.id for op in planned_ops}
    for feat in test_features:
        assert feat.id in planned_feature_ids, f"Feature {feat.id} ({feat.type}) received no operations!"

    # Now verify StrategyEngine generates valid toolpaths for each planned operation
    strat_ctx = StrategyContext(
        context=context,
        setup=setup,
        mesh=None,
        stock_voxels=None,
        clearance_z=15.0,
        finish_allowance_wall=0.5,
        finish_allowance_floor=0.3,
    )
    engine = StrategyEngine(strat_ctx)

    for op in planned_ops:
        tp = engine.generate(op)
        assert tp is not None
        assert tp.operation_id == op.id
        if tp.is_empty_or_skipped:
            continue
        assert len(tp.segments) > 0, f"Toolpath for {op.id} ({op.purpose}) on feature {op.feature.id} has no motion segments!"
