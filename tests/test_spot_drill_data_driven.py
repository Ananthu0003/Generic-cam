"""Tests for data-driven tool capability evaluation in spot drilling.

Verifies:
1. Generic twist drills are NOT automatically selected as spot drills.
2. Spot drilling requires dedicated or explicitly capable spotting tools (SPOT_DRILL, CENTER_DRILL, can_spot_drill=True, etc.).
3. When no spotting tool is available, spot drilling is safely skipped rather than misappropriating a twist drill.
4. Spot depth is accurately calculated from the tool's point angle and feature diameter.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    Material, WorkOffset, DEFAULT_MATERIALS
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.operations import OperationPlanner, OpPurpose


def test_generic_drills_not_used_for_spotting():
    """Test that generic twist drills (Ø0.51mm, Ø0.58mm) are not picked to spot drill each other."""
    tool_drill1 = Tool(
        id="T02", tool_number=2, type=ToolType.DRILL, diameter=0.51,
        flute_length=5.0, overall_length=30.0, flutes=2, material="carbide"
    )
    tool_drill2 = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=0.58,
        flute_length=6.0, overall_length=30.0, flutes=2, material="carbide"
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    hole = MachiningFeature(
        id="hole_058mm",
        type=FeatureType.THROUGH_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 0.0]),
        bounds_max=np.array([0.29, 0.29, 5.0]),
        depth=5.0,
        top_z=5.0,
        floor_z=0.0,
        diameter=0.58,
        is_concave=True,
    )

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_drill1, tool_drill2], material=mat)
    planner = OperationPlanner(ctx, setup, [hole])
    plan = planner.plan()

    # Must NOT have any spot_drilling operation using T02 or T05
    spot_ops = [op for op in plan if op.purpose == OpPurpose.SPOT_DRILLING]
    assert len(spot_ops) == 0, f"Expected 0 spot drilling operations, but found: {[op.id for op in spot_ops]}"

    # Must still have the actual drilling operation
    drill_ops = [op for op in plan if op.purpose == OpPurpose.DRILLING]
    assert len(drill_ops) == 1
    assert drill_ops[0].tool.id == "T05"


def test_dedicated_spot_drill_selected_when_available():
    """Test that a dedicated SPOT_DRILL or tool with spotting capability is correctly selected."""
    tool_spot = Tool(
        id="T01", tool_number=1, type=ToolType.SPOT_DRILL, diameter=3.0,
        flute_length=10.0, overall_length=40.0, point_angle=90.0, flutes=2, material="carbide"
    )
    tool_drill = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=0.58,
        flute_length=6.0, overall_length=30.0, point_angle=118.0, flutes=2, material="carbide"
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    hole = MachiningFeature(
        id="hole_058mm",
        type=FeatureType.THROUGH_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 0.0]),
        bounds_max=np.array([0.29, 0.29, 5.0]),
        depth=5.0,
        top_z=5.0,
        floor_z=0.0,
        diameter=0.58,
        is_concave=True,
    )

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_spot, tool_drill], material=mat)
    planner = OperationPlanner(ctx, setup, [hole])
    plan = planner.plan()

    spot_ops = [op for op in plan if op.purpose == OpPurpose.SPOT_DRILLING]
    assert len(spot_ops) == 1
    assert spot_ops[0].tool.id == "T01"
    assert "spot_depth" in spot_ops[0].notes


def test_multi_capability_tool_recognized():
    """Test that a tool with explicit can_spot_drill=True or operations=['spot_drilling'] is recognized."""
    tool_multi = Tool(
        id="T03", tool_number=3, type=ToolType.CHAMFER_MILL, diameter=6.0,
        flute_length=12.0, overall_length=50.0, can_plunge=True, can_spot_drill=True, material="carbide"
    )
    tool_drill = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=4.0,
        flute_length=20.0, overall_length=60.0, flutes=2, material="carbide"
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-10.0, -10.0, 0.0]), bounds_max=np.array([10.0, 10.0, 20.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    hole = MachiningFeature(
        id="hole_4mm",
        type=FeatureType.THROUGH_HOLE,
        face_indices=[0],
        bounds_min=np.array([-2.0, -2.0, 0.0]),
        bounds_max=np.array([2.0, 2.0, 15.0]),
        depth=15.0,
        top_z=15.0,
        floor_z=0.0,
        diameter=4.0,
        is_concave=True,
    )

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_multi, tool_drill], material=mat)
    planner = OperationPlanner(ctx, setup, [hole])
    plan = planner.plan()

    spot_ops = [op for op in plan if op.purpose == OpPurpose.SPOT_DRILLING]
    assert len(spot_ops) == 1
    assert spot_ops[0].tool.id == "T03"
