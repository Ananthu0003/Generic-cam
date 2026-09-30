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
    ToolType,
    WorkOffset,
    DEFAULT_MATERIALS,
    derive_setup_clearance_plane,
)
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.post import get_post_processor


def _make_test_machine(travel_max_z: float = 300.0, safe_retract: float = 10.0) -> MachineConfig:
    return MachineConfig(
        id="M01",
        name="Test 3-Axis Mill",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="fanuc",
        spindle_max_rpm=15000.0,
        spindle_min_rpm=100.0,
        max_spindle_power_kw=15.0,
        axes={
            "X": MachineAxis("X", -500.0, 500.0, 10000.0, 5000.0),
            "Y": MachineAxis("Y", -300.0, 300.0, 10000.0, 5000.0),
            "Z": MachineAxis("Z", -200.0, travel_max_z, 10000.0, 5000.0),
        },
        tool_change_position=np.array([0.0, 0.0, travel_max_z]),
        safe_retract_height=safe_retract,
        work_offsets={"G54": np.zeros(3), "G55": np.array([100.0, 0.0, 0.0])},
    )


def _make_cut_params(rpm: float = 8001.0, feed: float = 600.0, doc: float = 1.0, stepover: float = 2.1) -> CuttingParameters:
    return CuttingParameters(
        spindle_rpm=rpm,
        feed_rate=feed,
        plunge_feed=feed * 0.3,
        ramp_feed=feed * 0.5,
        depth_of_cut_mm=doc,
        stepover_mm=stepover,
        feed_per_tooth_mm=0.05,
        coolant=True,
        effective_diameter_mm=3.0,
        engagement_angle_deg=90.0,
        power_kw=0.5,
        torque_nm=0.3,
    )


def test_setup_level_plane_hierarchy_derivation():
    """Verify that Setup 1 and Setup 2 derive clearance, retract, and approach planes
    from setup-level stock and geometry definitions."""
    machine = _make_test_machine(safe_retract=10.0)
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, corner_radius=0.8, flute_length=15.0,
        overall_length=50.0, flutes=3, material="carbide"
    )

    # Setup 1: Top Milling (Stock top = 4.22)
    s1_stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -5.0, -1.0]),
        bounds_max=np.array([15.0, 15.0, 4.22])
    )
    s1 = Setup(
        id="setup_001",
        name="Setup 1 - Top Milling",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=s1_stock,
        approach_margin=1.0,
        retract_margin=2.0,
    )
    s1_cl = s1.compute_clearance(machine=machine, tools=[tool])

    assert pytest.approx(s1_cl.clearance_z, 1e-4) == 15.02  # 4.22 + 10.0 + 0.8
    assert pytest.approx(s1_cl.approach_z, 1e-4) == 5.22    # 4.22 + 1.0
    assert pytest.approx(s1_cl.retract_z, 1e-4) == 6.22     # 4.22 + 2.0

    # Setup 2: Bottom Flip (Stock top = 1.00)
    s2_stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -15.0, -4.22]),
        bounds_max=np.array([15.0, 5.0, 1.00])
    )
    s2 = Setup(
        id="setup_002",
        name="Setup 2 - Bottom Flip",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G55,
        stock=s2_stock,
        approach_margin=1.0,
        retract_margin=2.0,
    )
    s2_cl = s2.compute_clearance(machine=machine, tools=[tool])

    assert pytest.approx(s2_cl.clearance_z, 1e-4) == 11.80  # 1.00 + 10.0 + 0.8
    assert pytest.approx(s2_cl.approach_z, 1e-4) == 2.00    # 1.00 + 1.0
    assert pytest.approx(s2_cl.retract_z, 1e-4) == 3.00     # 1.00 + 2.0


def test_planned_operation_plane_accessors():
    """Verify that PlannedOperation exposes start_z, final_z, and approach_z."""
    feat = MachiningFeature(
        id="feat_face_01",
        type=FeatureType.FACING_REGION,
        face_indices=[0],
        bounds_min=np.array([0.0, 0.0, 3.22]),
        bounds_max=np.array([20.0, 20.0, 4.22]),
        accessibility=Accessibility.TOOL_AXIS_OK,
        top_z=4.22,
        floor_z=3.22,
        depth=1.00,
    )
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, corner_radius=0.0, flute_length=15.0,
        overall_length=50.0, flutes=3, material="carbide"
    )
    params = _make_cut_params(rpm=8001.0, feed=600.0, doc=1.00, stepover=2.10)
    op = PlannedOperation(
        id="op_facing_01",
        purpose=OpPurpose.FACING,
        feature=feat,
        tool=tool,
        params=params,
        notes={"excess_height": 1.00}
    )

    assert pytest.approx(op.start_z, 1e-4) == 4.22
    assert pytest.approx(op.final_z, 1e-4) == 3.22
    assert pytest.approx(op.get_approach_z(1.0), 1e-4) == 5.22
    assert pytest.approx(op.get_retract_z(2.0), 1e-4) == 6.22


def test_approach_height_generation_setup1_and_setup2():
    """Verify that Toolpath linking motions generate G0 approach heights at Z5.22 for Setup 1
    and Z2.00 for Setup 2 from setup-relative planes."""
    machine = _make_test_machine(safe_retract=10.0)
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, corner_radius=0.8, flute_length=15.0,
        overall_length=50.0, flutes=3, material="carbide"
    )
    params = _make_cut_params(rpm=8001.0, feed=600.0, doc=1.00, stepover=2.10)

    # --- Setup 1 ---
    s1_stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -5.0, -1.0]),
        bounds_max=np.array([15.0, 15.0, 4.22])
    )
    s1 = Setup(
        id="setup_001", name="Setup 1", model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54, stock=s1_stock, approach_margin=1.0
    )
    s1_cl = s1.compute_clearance(machine=machine, tools=[tool])
    f1 = MachiningFeature(
        id="feat_facing_s1", type=FeatureType.FACING_REGION,
        face_indices=[0], bounds_min=np.array([-5.0, -5.0, 3.22]), bounds_max=np.array([15.0, 15.0, 4.22]),
        accessibility=Accessibility.TOOL_AXIS_OK, top_z=4.22, floor_z=3.22,
        depth=1.00
    )
    op1 = PlannedOperation(
        id="op_facing_s1", purpose=OpPurpose.FACING,
        feature=f1, tool=tool, params=params, notes={"excess_height": 1.00}
    )

    from cam_engine.units import Units
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[s1], machine=machine, tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"]
    )
    strat_ctx1 = StrategyContext(
        context=ctx, setup=s1, mesh=None, stock_voxels=None,
        clearance_z=s1_cl.clearance_z, finish_allowance_wall=0.0,
        finish_allowance_floor=0.0, clearance_info=s1_cl, approach_margin=1.0
    )
    engine1 = StrategyEngine(strat_ctx1)
    tp1 = engine1.generate(op1)

    # Find the rapid descend and plunge segments
    rapid_descend_1 = [s for s in tp1.segments if s.motion_type == MotionType.RAPID and s.end[2] < s.start[2] - 1e-4]
    assert len(rapid_descend_1) >= 1
    assert pytest.approx(rapid_descend_1[0].start[2], 1e-4) == 15.02
    assert pytest.approx(rapid_descend_1[0].end[2], 1e-4) == 5.22

    plunge_1 = [s for s in tp1.segments if s.motion_type == MotionType.PLUNGE and abs(s.start[2] - 5.22) < 1e-4]
    assert len(plunge_1) >= 1
    assert pytest.approx(plunge_1[0].end[2], 1e-4) == 3.22

    post_fanuc = get_post_processor("fanuc", work_offset="G54")
    res1 = post_fanuc.post_process([tp1], program_name="SETUP1_APPROACH")
    assert "Z15.02" in res1.gcode
    assert "Z5.22" in res1.gcode
    assert "Z3.22" in res1.gcode

    # --- Setup 2 ---
    s2_stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -15.0, -4.22]),
        bounds_max=np.array([15.0, 5.0, 1.00])
    )
    s2 = Setup(
        id="setup_002", name="Setup 2", model_to_setup=np.eye(4),
        work_offset=WorkOffset.G55, stock=s2_stock, approach_margin=1.0
    )
    s2_cl = s2.compute_clearance(machine=machine, tools=[tool])
    f2 = MachiningFeature(
        id="feat_facing_s2", type=FeatureType.FACING_REGION,
        face_indices=[0], bounds_min=np.array([-5.0, -15.0, 0.0]), bounds_max=np.array([15.0, 5.0, 1.00]),
        accessibility=Accessibility.TOOL_AXIS_OK, top_z=1.00, floor_z=0.00,
        depth=1.00
    )
    op2 = PlannedOperation(
        id="op_facing_s2", purpose=OpPurpose.FACING,
        feature=f2, tool=tool, params=params, notes={"excess_height": 1.00}
    )

    strat_ctx2 = StrategyContext(
        context=ctx, setup=s2, mesh=None, stock_voxels=None,
        clearance_z=s2_cl.clearance_z, finish_allowance_wall=0.0,
        finish_allowance_floor=0.0, clearance_info=s2_cl, approach_margin=1.0
    )
    engine2 = StrategyEngine(strat_ctx2)
    tp2 = engine2.generate(op2)

    rapid_descend_2 = [s for s in tp2.segments if s.motion_type == MotionType.RAPID and s.end[2] < s.start[2] - 1e-4]
    assert len(rapid_descend_2) >= 1
    assert pytest.approx(rapid_descend_2[0].start[2], 1e-4) == 11.80
    assert pytest.approx(rapid_descend_2[0].end[2], 1e-4) == 2.00

    plunge_2 = [s for s in tp2.segments if s.motion_type == MotionType.PLUNGE and abs(s.start[2] - 2.00) < 1e-4]
    assert len(plunge_2) >= 1
    assert pytest.approx(plunge_2[0].end[2], 1e-4) == 0.00


def test_dynamic_approach_margin_reconfiguration():
    """Verify that changing setup approach margin dynamically updates the approach plane
    across operations without hardcoded numbers."""
    machine = _make_test_machine(safe_retract=10.0)
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, corner_radius=0.0, flute_length=15.0,
        overall_length=50.0, flutes=3, material="carbide"
    )
    params = _make_cut_params(rpm=8001.0, feed=600.0, doc=1.00, stepover=2.10)

    s1_stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -5.0, -1.0]),
        bounds_max=np.array([15.0, 15.0, 4.22])
    )
    # Set custom approach margin = 2.5 mm
    s1_custom = Setup(
        id="setup_custom", name="Setup Custom Margin", model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54, stock=s1_stock, approach_margin=2.5
    )
    s1_cl = s1_custom.compute_clearance(machine=machine, tools=[tool])

    assert pytest.approx(s1_cl.approach_z, 1e-4) == 6.72  # 4.22 + 2.5

    f1 = MachiningFeature(
        id="feat_facing", type=FeatureType.FACING_REGION,
        face_indices=[0], bounds_min=np.array([-5.0, -5.0, 3.22]), bounds_max=np.array([15.0, 15.0, 4.22]),
        accessibility=Accessibility.TOOL_AXIS_OK, top_z=4.22, floor_z=3.22,
        depth=1.00
    )
    op1 = PlannedOperation(
        id="op_facing", purpose=OpPurpose.FACING,
        feature=f1, tool=tool, params=params, notes={"excess_height": 1.00}
    )

    from cam_engine.units import Units
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[s1_custom], machine=machine, tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"]
    )
    strat_ctx = StrategyContext(
        context=ctx, setup=s1_custom, mesh=None, stock_voxels=None,
        clearance_z=s1_cl.clearance_z, finish_allowance_wall=0.0,
        finish_allowance_floor=0.0, clearance_info=s1_cl, approach_margin=2.5
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op1)

    rapid_descend = [s for s in tp.segments if s.motion_type == MotionType.RAPID and s.end[2] < s.start[2] - 1e-4]
    assert len(rapid_descend) >= 1
    # Rapid stops at dynamically derived Z6.72
    assert pytest.approx(rapid_descend[0].end[2], 1e-4) == 6.72
