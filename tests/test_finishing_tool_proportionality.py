"""Tests for tool selection proportionality and rejection of oversized tools.

Verifies:
1. Operation planner does NOT force oversized tools (e.g. Ø12.7mm tool on Ø0.90mm OD pin).
2. Reports 'No compatible finishing tool available' when library lacks compatible tools.
3. Evaluates minimum accessible feature radius, tool diameter, holder clearance, and stock dimensions.
4. Successfully plans operations when compatible tools are present.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, ToolHolder, PlanningContext, Setup, Stock, StockKind,
    Material, WorkOffset, DEFAULT_MATERIALS
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.operations import OperationPlanner, OpPurpose


def test_rejects_oversized_finishing_tool_for_small_pin():
    """Test that Ø12.7mm tool is rejected for finishing a Ø0.90mm pin, reporting missing tool."""
    # Library only contains large cutters
    tool_giant = Tool(
        id="T04", tool_number=4, type=ToolType.FLAT_ENDMILL, diameter=12.7,
        flute_length=30.0, overall_length=75.0, flutes=4, material="carbide"
    )
    tool_huge = Tool(
        id="T05", tool_number=5, type=ToolType.FLAT_ENDMILL, diameter=20.0,
        flute_length=40.0, overall_length=90.0, flutes=4, material="carbide"
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    # 0.90 mm diameter pin/boss
    feat_pin = MachiningFeature(
        id="pin_090mm",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 0.0]),
        bounds_max=np.array([0.45, 0.45, 5.0]),
        depth=5.0,
        top_z=5.0,
        floor_z=0.0,
        diameter=0.90,
        is_concave=False,
    )

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_giant, tool_huge], material=mat)
    planner = OperationPlanner(ctx, setup, [feat_pin])
    plan = planner.plan()

    # Must NOT force operation with Ø12.7mm or Ø20mm tool
    assert len(plan) == 0, f"Expected 0 planned operations for incompatible tools, got {len(plan)}"

    # Must report MISSING_TOOL warning with descriptive message
    finish_warnings = [w for w in planner.warnings if w.get("code") == "MISSING_TOOL"]
    assert len(finish_warnings) > 0
    assert any("No compatible finishing tool available" in w["message"] for w in finish_warnings)
    assert any("pin_090mm" in w["feature_id"] for w in finish_warnings)

    # Must suggest a properly sized micro tool
    sugg = finish_warnings[0].get("suggested_tool")
    assert sugg is not None
    assert sugg["diameter"] <= 3.0


def test_accepts_proportional_finishing_tool_when_available():
    """Test that OperationPlanner successfully plans finishing when a proportional tool exists."""
    tool_micro = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=1.5,
        flute_length=6.0, overall_length=40.0, flutes=2, material="carbide"
    )
    tool_giant = Tool(
        id="T04", tool_number=4, type=ToolType.FLAT_ENDMILL, diameter=12.7,
        flute_length=30.0, overall_length=75.0, flutes=4, material="carbide"
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 10.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    feat_pin = MachiningFeature(
        id="pin_090mm",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 0.0]),
        bounds_max=np.array([0.45, 0.45, 5.0]),
        depth=5.0,
        top_z=5.0,
        floor_z=0.0,
        diameter=0.90,
        is_concave=False,
    )

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_micro, tool_giant], material=mat)
    planner = OperationPlanner(ctx, setup, [feat_pin])
    plan = planner.plan()

    # Should plan using T01 (Ø1.5mm), NOT T04 (Ø12.7mm)
    assert len(plan) > 0
    for op in plan:
        assert op.tool.id == "T01"
        assert op.tool.diameter == 1.5


def test_rejects_holder_collision_in_deep_pocket():
    """Test that tool with colliding holder is rejected for internal feature."""
    # Tool with flute length 4mm, but feature is 10mm deep, and holder diameter is 25mm in 12mm pocket
    holder = ToolHolder(diameter=25.0, length=40.0)
    tool_bad_holder = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=6.0,
        flute_length=4.0, overall_length=40.0, flutes=3, material="carbide",
        holder=holder
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-15.0, -15.0, 0.0]), bounds_max=np.array([15.0, 15.0, 20.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    pocket = MachiningFeature(
        id="pocket_deep",
        type=FeatureType.POCKET,
        face_indices=[0],
        bounds_min=np.array([-6.0, -6.0, 0.0]),
        bounds_max=np.array([6.0, 6.0, 10.0]),
        depth=10.0,
        top_z=10.0,
        floor_z=0.0,
        diameter=12.0,
        is_concave=True,
    )

    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool_bad_holder], material=mat)
    planner = OperationPlanner(ctx, setup, [pocket])
    plan = planner.plan()

    # The tool holder collides, so the operation must NOT be planned with this tool
    assert len(plan) == 0
    assert any("MISSING_TOOL" in w.get("code", "") for w in planner.warnings)
