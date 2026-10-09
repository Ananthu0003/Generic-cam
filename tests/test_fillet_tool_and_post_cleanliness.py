import pytest
import numpy as np
from cam_engine.context import (
    PlanningContext, Tool, ToolType, Setup, Stock, StockKind, WorkOffset, DEFAULT_MATERIALS
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.operations import OperationPlanner, OpPurpose
from cam_engine.post.canonical import ToolChange, Spindle, SafetyBlock, Rapid
from cam_engine.post.post_processor import FanucPostProcessor


def test_face_mill_never_selected_for_filleting():
    """Verify that a face mill with corner radius is never selected for filleting."""
    face_mill = Tool(
        id="T01",
        tool_number=1,
        type=ToolType.FACE_MILL,
        diameter=50.0,
        flute_length=10.0,
        overall_length=90.0,
        corner_radius=0.8,
    )
    flat_mill = Tool(
        id="T02",
        tool_number=2,
        type=ToolType.FLAT_ENDMILL,
        diameter=12.0,
        flute_length=30.0,
        overall_length=70.0,
        corner_radius=0.0,
    )
    fillet_feat = MachiningFeature(
        id="fillet_001",
        face_indices=[1],
        type=FeatureType.FILLET,
        fillet_radius=0.8,
        depth=2.0,
        top_z=0.0,
        floor_z=-2.0,
        surface_type="torus",
        bounds_min=np.array([-5.0, -5.0, -2.0]),
        bounds_max=np.array([5.0, 5.0, 0.0]),
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-10.0, -10.0, -5.0]), bounds_max=np.array([10.0, 10.0, 5.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[face_mill, flat_mill], material=mat)
    planner = OperationPlanner(ctx, setup, [fillet_feat])
    
    # _fillet_tool must return None since no ball or bullnose mill exists
    assert planner._fillet_tool(fillet_feat) is None


def test_tool_change_does_not_emit_stale_previous_spindle_rpm():
    """Verify that a tool change without explicit restart RPM does not output the previous tool's spindle speed."""
    post = FanucPostProcessor(work_offset="G54")
    cmd = ToolChange(
        tool_id="T02",
        tool_number=2,
        length_offset_h=2,
        description="Chamfer Mill",
        spindle_rpm=None,
    )
    lines = post._render_tool_change(cmd)
    
    # Must contain tool change and G43 length offset, but no spurious S... M3 line
    assert any("T2" in line for line in lines)
    assert any("G43 H2" in line for line in lines)
    assert not any("S" in line and "M3" in line for line in lines)


def test_chamfers_and_countersinks_deduplicated():
    """Verify that multiple concentric chamfer/countersink features at the same location are deduplicated."""
    c1 = MachiningFeature(
        id="csink_001",
        face_indices=[1],
        type=FeatureType.COUNTERSINK,
        diameter=3.0,
        depth=1.0,
        top_z=0.0,
        floor_z=-1.0,
        bounds_min=np.array([15.0, -1.0, -1.0]),
        bounds_max=np.array([17.0, 1.0, 0.0]),
    )
    c2 = MachiningFeature(
        id="csink_002",
        face_indices=[2],
        type=FeatureType.COUNTERSINK,
        diameter=3.0,
        depth=1.0,
        top_z=0.0,
        floor_z=-1.0,
        bounds_min=np.array([15.0, -1.0, -1.0]),
        bounds_max=np.array([17.0, 1.0, 0.0]),
    )
    csink_tool = Tool(
        id="T06",
        tool_number=6,
        type=ToolType.COUNTERSINK_TOOL,
        diameter=6.0,
        flute_length=10.0,
        overall_length=50.0,
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([10.0, -5.0, -5.0]), bounds_max=np.array([20.0, 5.0, 5.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[csink_tool], material=mat)
    planner = OperationPlanner(ctx, setup, [c1, c2])
    ops = planner._plan_countersinking()
    
    # 2 duplicate features at the same location should only produce 1 operation
    assert len(ops) == 1


def test_non_circular_feature_not_treated_as_circular_boss():
    """Verify that a non-circular feature (e.g. rectangular boss) is never treated as a circular boss."""
    from cam_engine.toolpaths.strategies import StrategyEngine
    f_rect = MachiningFeature(
        id="f_rect",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-18.735, -3.735, 0.0]),
        bounds_max=np.array([18.735, 3.735, 7.14]),
        diameter=2.1,  # corner arc diameter
        depth=7.14,
        top_z=7.14,
        floor_z=0.0,
    )
    assert StrategyEngine._is_feature_circular(f_rect) is False

    f_circ = MachiningFeature(
        id="f_circ",
        type=FeatureType.BOSS,
        face_indices=[0],
        bounds_min=np.array([-5.0, -5.0, 0.0]),
        bounds_max=np.array([5.0, 5.0, 10.0]),
        diameter=10.0,
        depth=10.0,
        top_z=10.0,
        floor_z=0.0,
    )
    assert StrategyEngine._is_feature_circular(f_circ) is True

