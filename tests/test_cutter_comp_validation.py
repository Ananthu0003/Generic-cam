import numpy as np
import pytest

from cam_engine.context import Tool, ToolType, PlanningContext, Setup, Stock, StockKind, Material, WorkOffset
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.geometry import TriangleMesh
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.toolpaths.validation import ToolpathValidator


def test_proportional_tool_selection_for_small_features():
    """Test that OperationPlanner picks a proportionally sized tool for a small pin/boss rather than giant tools."""
    tool_micro = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=1.0, flute_length=5.0, overall_length=40.0, flutes=2, material="carbide")
    tool_small = Tool(id="T02", tool_number=2, type=ToolType.FLAT_ENDMILL, diameter=2.0, flute_length=8.0, overall_length=45.0, flutes=3, material="carbide")
    tool_medium = Tool(id="T03", tool_number=3, type=ToolType.FLAT_ENDMILL, diameter=6.0, flute_length=20.0, overall_length=60.0, flutes=4, material="carbide")
    tool_giant = Tool(id="T04", tool_number=4, type=ToolType.FLAT_ENDMILL, diameter=12.7, flute_length=30.0, overall_length=75.0, flutes=4, material="carbide")

    tools = [tool_micro, tool_small, tool_medium, tool_giant]
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5, -5, 0]), bounds_max=np.array([5, 5, 10]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = Material(id="al", name="Aluminum", v_carbide=300, v_hss=90, hardness_hb=95, max_hm_ratio=1.0, max_ae_ratio=0.4, specific_cutting_force_kn_mm2=0.8, needs_coolant=True)
    
    # 0.9mm diameter pin/boss
    feat_pin = MachiningFeature(
        id="pin_09mm", type=FeatureType.BOSS, face_indices=[0],
        bounds_min=np.array([-0.45, -0.45, 0.0]), bounds_max=np.array([0.45, 0.45, 5.0]),
        depth=5.0, top_z=5.0, floor_z=0.0, diameter=0.9, is_concave=False
    )
    
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=tools, material=mat)
    planner = OperationPlanner(ctx, setup, [feat_pin])
    plan = planner.plan()
    
    for op in plan:
        # Tool selected should be proportional (Ø1.0 or Ø2.0 mm), NOT Ø12.7 mm
        assert op.tool.diameter <= 3.0, f"Selected tool diameter {op.tool.diameter}mm is too large for Ø0.9mm feature!"


def test_validator_flags_cutter_comp_on_circular_arc():
    """Test that ToolpathValidator rejects cutter compensation turning on directly on a circular arc block."""
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=6.0, flute_length=20.0, overall_length=60.0, flutes=4, material="carbide")
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-20, -20, 0]), bounds_max=np.array([20, 20, 20]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = Material(id="al", name="Aluminum", v_carbide=300, v_hss=90, hardness_hb=95, max_hm_ratio=1.0, max_ae_ratio=0.4, specific_cutting_force_kn_mm2=0.8, needs_coolant=True)
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)
    mesh = TriangleMesh(triangles=np.zeros((1, 3, 3)))
    validator = ToolpathValidator(ctx, setup, mesh, clearance_z=20.0)

    tp_bad = Toolpath(operation_id="op1", feature_id="f1", tool_id="T01", purpose="finishing")
    # Illegal: G41 on ARC_CW
    tp_bad.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=np.array([-10.0, 0.0, 5.0]),
        end=np.array([0.0, 10.0, 5.0]),
        arc_center=np.array([0.0, 0.0, 5.0]),
        feed=1000.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1",
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
    ))
    rep = validator.validate(tp_bad)
    assert any("cannot be activated directly on circular arc" in e.message for e in rep.errors)


def test_validator_accepts_valid_linear_lead_in_comp():
    """Test that ToolpathValidator accepts G41 engaged on linear lead-in followed by tangential arc."""
    tool = Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=6.0, flute_length=20.0, overall_length=60.0, flutes=4, material="carbide")
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-20, -20, 0]), bounds_max=np.array([20, 20, 20]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = Material(id="al", name="Aluminum", v_carbide=300, v_hss=90, hardness_hb=95, max_hm_ratio=1.0, max_ae_ratio=0.4, specific_cutting_force_kn_mm2=0.8, needs_coolant=True)
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)
    mesh = TriangleMesh(triangles=np.zeros((1, 3, 3)))
    validator = ToolpathValidator(ctx, setup, mesh, clearance_z=20.0)

    tp_ok = Toolpath(operation_id="op1", feature_id="f1", tool_id="T01", purpose="finishing")
    # Linear lead-in move (length = 4.0mm >= tool radius 3.0mm)
    tp_ok.add(MotionSegment(
        motion_type=MotionType.ENTRY,
        start=np.array([-14.0, 4.0, 5.0]),
        end=np.array([-10.0, 4.0, 5.0]),
        feed=1000.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1",
        metadata={"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
    ))
    # Tangential arc
    tp_ok.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=np.array([-10.0, 4.0, 5.0]),
        end=np.array([-6.0, 0.0, 5.0]),
        arc_center=np.array([-6.0, 4.0, 5.0]),
        feed=1000.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1"
    ))
    # Linear lead-out move cancelling G40
    tp_ok.add(MotionSegment(
        motion_type=MotionType.EXIT,
        start=np.array([-6.0, 0.0, 5.0]),
        end=np.array([-6.0, -4.0, 5.0]),
        feed=1000.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1",
        metadata={"cutter_comp": {"action": "off"}}
    ))
    rep = validator.validate(tp_ok)
    assert len(rep.errors) == 0
