"""Unit tests for material-, tool-, and depth-aware drilling peck depth (Q) calculation."""

import numpy as np
import pytest

from cam_engine.context import (
    MachineAxis,
    MachineConfig,
    MachineType,
    Material,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
    DEFAULT_MATERIALS,
)
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.geometry import TriangleMesh
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import MachinabilityAnalyzer, CuttingParameters
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.post import get_post_processor
from cam_engine.units import Units


def _make_machine():
    return MachineConfig(
        id="M01",
        name="3-Axis Mill",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="haas",
        spindle_max_rpm=15000.0,
        spindle_min_rpm=100.0,
        max_spindle_power_kw=15.0,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 10000.0, 5000.0),
            "Y": MachineAxis("Y", -300.0, 300.0, 10000.0, 5000.0),
            "Z": MachineAxis("Z", -200.0, 300.0, 10000.0, 5000.0),
        },
        safe_retract_height=10.0,
    )


def test_peck_depth_material_awareness():
    """Verify that peck depth (Q) adapts based on workpiece material machinability and hardness."""
    machine = _make_machine()
    tool = Tool(
        id="T_DRILL_058",
        tool_number=1,
        diameter=0.58,
        type=ToolType.DRILL,
        point_angle=118.0,
        flute_length=10.0,
        overall_length=38.0,
        flutes=2,
        material="carbide",
    )
    feat = MachiningFeature(
        id="f_hole",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        diameter=0.58,
    )

    # 1. Aluminum 6061 (soft non-ferrous)
    analyzer_al = MachinabilityAnalyzer(machine, DEFAULT_MATERIALS["aluminum_6061"])
    q_al = analyzer_al.calculate_drilling_peck(tool, feat)

    # 2. Steel 1018 (low carbon steel)
    analyzer_steel1018 = MachinabilityAnalyzer(machine, DEFAULT_MATERIALS["steel_1018"])
    q_steel1018 = analyzer_steel1018.calculate_drilling_peck(tool, feat)

    # 3. Steel 4140 (pre-hard alloy steel)
    analyzer_steel4140 = MachinabilityAnalyzer(machine, DEFAULT_MATERIALS["steel_4140"])
    q_steel4140 = analyzer_steel4140.calculate_drilling_peck(tool, feat)

    # 4. Stainless 304 (work-hardening stainless)
    analyzer_ss = MachinabilityAnalyzer(machine, DEFAULT_MATERIALS["stainless_304"])
    q_ss = analyzer_ss.calculate_drilling_peck(tool, feat)

    # Check strictly descending peck depth as material difficulty increases:
    # Q_aluminum > Q_steel1018 > Q_steel4140 > Q_stainless
    assert q_al > q_steel1018 > q_steel4140 > q_ss
    assert q_al <= 0.50 * 0.58  # Micro-drill safety limit (Q <= 0.50*D)
    assert q_ss < q_al


def test_peck_depth_hole_aspect_ratio_awareness():
    """Verify that deep holes receive reduced peck depth compared to shallow holes."""
    machine = _make_machine()
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    analyzer = MachinabilityAnalyzer(machine, mat)

    tool = Tool(
        id="T_DRILL_10",
        tool_number=1,
        diameter=1.0,
        type=ToolType.DRILL,
        flute_length=15.0,
        overall_length=50.0,
        flutes=2,
        material="carbide",
    )

    # Shallow hole: depth = 1.5 mm (L/D = 1.5)
    feat_shallow = MachiningFeature(
        id="f_shallow",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.5, -0.5, 0.0]),
        bounds_max=np.array([0.5, 0.5, 1.5]),
        depth=1.5,
        top_z=1.5,
        floor_z=0.0,
        diameter=1.0,
    )
    # Deep hole: depth = 8.0 mm (L/D = 8.0)
    feat_deep = MachiningFeature(
        id="f_deep",
        type=FeatureType.BLIND_HOLE,
        face_indices=[1],
        bounds_min=np.array([-0.5, -0.5, 0.0]),
        bounds_max=np.array([0.5, 0.5, 8.0]),
        depth=8.0,
        top_z=8.0,
        floor_z=0.0,
        diameter=1.0,
    )

    q_shallow = analyzer.calculate_drilling_peck(tool, feat_shallow)
    q_deep = analyzer.calculate_drilling_peck(tool, feat_deep)

    assert q_shallow > q_deep
    # Deep hole should reduce peck to prevent chip jamming
    assert q_deep < q_shallow


def test_drilling_canned_cycle_uses_material_aware_peck():
    """Verify that G83 canned cycle in G-code uses the material-derived Q peck depth."""
    machine = _make_machine()
    mat_ss = DEFAULT_MATERIALS["stainless_304"]
    analyzer = MachinabilityAnalyzer(machine, mat_ss)

    tool = Tool(
        id="T_DRILL_058",
        tool_number=1,
        diameter=0.58,
        type=ToolType.DRILL,
        point_angle=118.0,
        flute_length=10.0,
        overall_length=38.0,
        flutes=2,
        material="carbide",
    )
    feat = MachiningFeature(
        id="f_hole",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        diameter=0.58,
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 3.22]))
    setup = Setup(id="S1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup],
        machine=machine,
        tools=[tool],
        material=mat_ss,
    )

    # Derive cutting parameters via analyzer
    params = analyzer.analyze(tool, feat, "drilling", setup)
    assert params.depth_of_cut_mm < 0.58  # In stainless 304, peck is significantly smaller than diameter

    op = PlannedOperation(
        id="op_drill_ss",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.DRILLING,
        params=params,
        notes={"depth_reference": "tip"},
    )

    strat_ctx = StrategyContext(
        context=ctx,
        setup=setup,
        mesh=TriangleMesh(triangles=np.zeros((0, 3, 3))),
        stock_voxels=None,
        clearance_z=14.22,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine._drilling(op)

    post = get_post_processor("haas", machine)
    post_res = post.post_process([tp])

    # Verify G83 contains the material-aware Q value
    expected_q_formatted = post.format_coord("Q", params.depth_of_cut_mm)
    assert expected_q_formatted in post_res.gcode
