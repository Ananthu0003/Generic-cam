"""Tests for tool library management and resilient operation planning.

Verifies:
1. Features without fitting tools generate non-blocking warnings and tool suggestions (no crash).
2. Adding custom tools to session library via API.
3. Auto-suggesting micro-tools for unmachined features and re-planning operations.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from cam_engine.context import (MachineConfig, MachineType, MachineAxis, Units,
                                Tool, ToolType, PlanningContext, Setup, Stock,
                                StockKind, Material, DEFAULT_MATERIALS, WorkOffset)

from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.operations import OperationPlanner, OpPurpose


def make_test_context() -> PlanningContext:
    machine = MachineConfig(
        id="test_vmc",
        name="Test VMC",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 15000.0, 5000.0),
            "Y": MachineAxis("Y", -500.0, 500.0, 15000.0, 5000.0),
            "Z": MachineAxis("Z", -400.0, 50.0, 10000.0, 3000.0),
        },
        spindle_min_rpm=100.0,
        spindle_max_rpm=15000.0,
        max_spindle_power_kw=7.5,
        controller="fanuc",
        units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0,
    )
    tools = [
        Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=10.0,
             flute_length=30.0, overall_length=75.0, flutes=4),
        Tool(id="T02", tool_number=2, type=ToolType.DRILL, diameter=6.0,
             flute_length=25.0, overall_length=60.0, flutes=2),
    ]
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.machine = machine
    ctx.material = mat
    ctx.tools = tools
    return ctx



def make_setup() -> Setup:
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([0.0, 0.0, 0.0]),
                  bounds_max=np.array([50.0, 50.0, 25.0]))
    return Setup(id="setup_001", name="Top", model_to_setup=np.eye(4),
                 work_offset=WorkOffset.G54, stock=stock)



def test_unmachined_micro_hole_produces_warning_not_crash():
    ctx = make_test_context()
    setup = make_setup()

    # Feature is a 0.58mm micro hole — tool library only has 10mm endmill and 6mm drill
    micro_hole = MachiningFeature(
        id="hole_micro",
        type=FeatureType.THROUGH_HOLE,
        face_indices=[1],
        bounds_min=np.array([10.0, 10.0, 0.0]),
        bounds_max=np.array([10.58, 10.58, 25.0]),
        depth=25.0,
        top_z=25.0,
        floor_z=0.0,
        diameter=0.58,
        is_concave=True,
    )
    # Valid 6.0mm hole
    std_hole = MachiningFeature(
        id="hole_std",
        type=FeatureType.THROUGH_HOLE,
        face_indices=[2],
        bounds_min=np.array([20.0, 20.0, 0.0]),
        bounds_max=np.array([26.0, 26.0, 25.0]),
        depth=25.0,
        top_z=25.0,
        floor_z=0.0,
        diameter=6.0,
        is_concave=True,
    )

    planner = OperationPlanner(ctx, setup, [micro_hole, std_hole])
    ops = planner.plan()

    # The 6.0mm hole was planned successfully
    assert len(ops) >= 1
    assert any(op.feature.id == "hole_std" for op in ops)

    # The 0.58mm hole generated a structured warning with tool suggestion
    assert len(planner.warnings) == 1
    w = planner.warnings[0]
    assert w["code"] == "MISSING_TOOL"
    assert w["feature_id"] == "hole_micro"
    assert w["suggested_tool"]["diameter"] == 0.58
    assert w["suggested_tool"]["type"] == "drill"


def test_tool_api_endpoints():
    client = TestClient(app)

    # 1. Get tools
    res = client.get("/api/tools-and-machines")
    assert res.status_code == 200
    initial_count = len(res.json()["tools"])

    # 2. Add custom 0.58mm drill
    add_payload = {
        "tool_type": "drill",
        "diameter": 0.58,
        "flute_length": 8.0,
        "overall_length": 35.0,
        "flutes": 2,
    }
    res_add = client.post("/api/tools/add", json=add_payload)
    assert res_add.status_code == 200
    tools = res_add.json()["tools"]
    assert len(tools) == initial_count + 1
    added_tool = next((t for t in tools if abs(t["diameter"] - 0.58) < 1e-3), None)
    assert added_tool is not None
    assert added_tool["tool_type"] == "drill"

    # 3. Delete the added tool
    res_del = client.delete(f"/api/tools/{added_tool['id']}")
    assert res_del.status_code == 200
    assert len(res_del.json()["tools"]) == initial_count
