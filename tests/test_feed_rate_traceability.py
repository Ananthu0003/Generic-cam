"""Tests for Feed Rate and Spindle Speed Traceability & Auditability.

Verifies that:
1. CuttingParameters and MachinabilityAnalyzer derive and retain canonical machinability data:
   - material (name, hardness)
   - tool (id, diameter, flutes, material)
   - cutting speed Vc (m/min)
   - spindle RPM (target RPM and machine limit clamping)
   - chip load / feed per tooth fz (mm/tooth)
   - feed rate F (target feed, machine axis limit clamping, plunge feed, ramp feed)
   - engagement (ap stepdown, ae stepover, power kW, torque Nm)
2. Toolpath metadata retains all cutting parameters from the planned operation.
3. PostProcessor formats clear, auditable G-code comment blocks communicating how S and F were derived.
"""

import math
import numpy as np
import pytest

from cam_engine.context import (
    ImportedModel,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
    DEFAULT_MATERIALS,
)
from cam_engine.coords import CoordSpace
from cam_engine.demo_models import (
    create_prismatic_bracket_shape,
    get_default_machine,
    get_default_tools,
)
from cam_engine.features import FeatureRecognizer, FeatureType, MachiningFeature
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.geometry.topology import Shape
from cam_engine.geometry.voxel import VoxelStock
from cam_engine.machinability import CuttingParameters, MachinabilityAnalyzer
from cam_engine.operations import OperationPlanner, OpPurpose, PlannedOperation
from cam_engine.pipeline import run_cam_pipeline
from cam_engine.post import get_post_processor
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine


def test_cutting_parameters_traceability_fields():
    """Verify that MachinabilityAnalyzer derives and records complete machinability traceability."""
    machine = get_default_machine()
    material = DEFAULT_MATERIALS["aluminum_6061"]
    analyzer = MachinabilityAnalyzer(machine, material)

    tool = Tool(
        id="T01",
        type=ToolType.FLAT_ENDMILL,
        diameter=2.0,
        flute_length=10.0,
        overall_length=40.0,
        flutes=2,
        material="carbide",
        tool_number=1,
    )

    feature = MachiningFeature(
        id="feat_001",
        type=FeatureType.POCKET,
        face_indices=[0],
        bounds_min=np.array([0.0, 0.0, 0.0]),
        bounds_max=np.array([10.0, 10.0, 5.0]),
        depth=5.0,
        top_z=5.0,
        floor_z=0.0,
    )

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, -2.0]), bounds_max=np.array([15.0, 15.0, 6.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)

    params = analyzer.analyze(tool, feature, OpPurpose.ROUGHING.value, setup)

    # Verify theoretical derivations
    assert params.material_name == material.name
    assert params.material_hardness_hb == 95.0
    assert params.tool_id == "T01"
    assert params.tool_diameter_mm == 2.0
    assert params.flute_count == 2
    assert params.tool_material == "carbide"

    # Vc for aluminum carbide target is 450 m/min
    assert params.target_cutting_speed_m_min == material.v_carbide
    theoretical_rpm = 1000.0 * material.v_carbide / (math.pi * 2.0)
    assert np.isclose(params.target_rpm, theoretical_rpm, rtol=1e-3)
    # Machine spindle max is 15000 RPM, so RPM should be clamped to 15000
    assert params.spindle_rpm == min(machine.spindle_max_rpm, theoretical_rpm)
    assert params.rpm_clamped == (theoretical_rpm > machine.spindle_max_rpm)
    expected_effective_vc = (math.pi * 2.0 * params.spindle_rpm) / 1000.0
    assert np.isclose(params.effective_cutting_speed_m_min, expected_effective_vc, rtol=1e-3)

    # Chip load fz
    assert params.feed_per_tooth_mm > 0.0
    theoretical_feed = params.spindle_rpm * tool.flutes * params.feed_per_tooth_mm
    assert np.isclose(params.target_feed_rate, theoretical_feed, rtol=1e-3)
    assert params.feed_rate <= min(ax.max_feed for ax in machine.axes.values())
    assert params.plunge_feed == params.feed_rate * 0.40
    assert params.ramp_feed == params.feed_rate * 0.70


def test_toolpath_metadata_retains_cutting_parameters():
    """Verify that StrategyEngine passes complete cutting parameter traceability into Toolpath metadata."""
    model = create_prismatic_bracket_shape()
    machine = get_default_machine()
    tools = get_default_tools()
    material = DEFAULT_MATERIALS["aluminum_6061"]

    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, -5.0]), bounds_max=np.array([105.0, 65.0, 30.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    ctx = PlanningContext(model=model, setups=[setup], machine=machine, tools=tools, material=material)

    rec = FeatureRecognizer(model.shape, stock_top_z=float(stock.bounds_max[2]))
    features = rec.recognize()
    planner = OperationPlanner(ctx, setup, features)
    ops = planner.plan()
    assert len(ops) > 0

    mesh = TriangleMesh.from_faces(model.shape.faces())
    stock_voxels = VoxelStock.from_bounds(setup.stock.bounds_min, setup.stock.bounds_max, resolution=2.0)
    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=mesh, stock_voxels=stock_voxels,
        clearance_z=float(stock.bounds_max[2]) + 10.0,
        finish_allowance_wall=0.3, finish_allowance_floor=0.3,
    )
    engine = StrategyEngine(strat_ctx)

    for op in ops:
        tp = engine.generate(op)
        cp_meta = tp.metadata.get("cutting_parameters")
        assert cp_meta is not None, f"Toolpath for {op.id} missing cutting_parameters metadata"
        assert cp_meta["spindle_rpm"] == op.params.spindle_rpm
        assert cp_meta["feed_rate"] == op.params.feed_rate
        assert cp_meta["feed_per_tooth_mm"] == op.params.feed_per_tooth_mm
        assert cp_meta["cutting_speed_m_min"] > 0.0
        assert cp_meta["material_name"] == material.name


def test_postprocessor_generates_auditable_gcode_headers():
    """Verify that PostProcessor formats clear, traceable comments in G-code explaining S and F derivations."""
    tool = Tool(
        id="T01",
        type=ToolType.FLAT_ENDMILL,
        diameter=2.0,
        flute_length=10.0,
        overall_length=40.0,
        flutes=2,
        material="carbide",
        tool_number=1,
    )
    tp = Toolpath(operation_id="op_rough_001", feature_id="feat_001", tool_id="T01", purpose="roughing")
    tp.metadata["tool_info"] = {
        "id": "T01", "tool_number": 1, "type": "flat_endmill", "diameter": 2.0,
        "flute_length": 10.0, "overall_length": 40.0, "flutes": 2, "material": "carbide",
    }
    tp.metadata["cutting_parameters"] = {
        "spindle_rpm": 15000.0,
        "feed_rate": 1896.7,
        "plunge_feed": 758.7,
        "ramp_feed": 1327.7,
        "feed_per_tooth_mm": 0.0632,
        "depth_of_cut_mm": 2.0,
        "stepover_mm": 1.0,
        "cutting_speed_m_min": 120.0,
        "target_rpm": 19098.6,
        "rpm_clamped": True,
        "target_feed_rate": 2414.0,
        "feed_clamped": False,
        "material_name": "aluminum_6061",
        "material_hardness_hb": 95.0,
        "tool_id": "T01",
        "tool_type": "flat_endmill",
        "tool_material": "carbide",
        "tool_diameter_mm": 2.0,
        "flute_count": 2,
        "machine_max_rpm": 15000.0,
        "machine_max_feed": 5000.0,
        "power_kw": 0.12,
        "torque_nm": 0.08,
    }

    # Add rapid and cut motions
    tp.add(MotionSegment(motion_type=MotionType.RAPID, start=np.array([0, 0, 10]), end=np.array([0, 0, 2])))
    tp.add(MotionSegment(motion_type=MotionType.CUT, start=np.array([0, 0, 0]), end=np.array([10, 0, 0]), feed=1896.7, spindle=15000.0))

    post = get_post_processor(controller="fanuc", work_offset="G54")
    res = post.post_process([tp], program_name="AUDIT_TEST")
    gcode = res.gcode

    # Verify auditable comments in G-code
    assert "Material: aluminum_6061" in gcode
    assert "Speed: Target Vc = 120.0 m/min (Calc: S19099 RPM) | Machine Limit: 15000 RPM -> Applied S15000 RPM (Effective Vc = 94.2 m/min)" in gcode
    assert "Feeds: fz = 0.0632 mm/tooth x 2F -> F1896.7 mm/min [Plunge: F758.7, Ramp: F1327.7]" in gcode
    assert "Engagement: ap = 2.00mm, ae = 1.00mm" in gcode
    assert "S15000 M3" in gcode
    assert "F1896.7" in gcode
