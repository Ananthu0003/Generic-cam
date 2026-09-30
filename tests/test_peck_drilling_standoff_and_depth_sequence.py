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
)
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters, MachinabilityAnalyzer
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.post import get_post_processor


def _make_test_machine() -> MachineConfig:
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
            "Z": MachineAxis("Z", -200.0, 300.0, 10000.0, 5000.0),
        },
        tool_change_position=np.array([0.0, 0.0, 300.0]),
        safe_retract_height=10.0,
        work_offsets={"G54": np.zeros(3), "G55": np.array([100.0, 0.0, 0.0])},
    )


def test_setup1_drill_peck_depth_sequence_and_standoff():
    """Verify that Setup 1 (Ø0.58 drill) advances by 0.102mm uniform pecks
    and that the first cut does not plunge from R-plane through air."""
    tool = Tool(
        id="T02", tool_number=2, type=ToolType.DRILL, diameter=0.58,
        point_angle=118.0, flute_length=10.0, overall_length=40.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, -1.0]), bounds_max=np.array([15.0, 15.0, 4.22]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    machine = _make_test_machine()

    # Blind hole: Surface Z = 3.22 mm, Cycle tip depth Z = 1.470 mm (Tip travel = 1.750 mm)
    hole = MachiningFeature(
        id="hole_s1", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75, top_z=3.22, floor_z=1.47, diameter=0.58, is_concave=True,
    )

    analyzer = MachinabilityAnalyzer(machine, mat)
    strat = analyzer.calculate_drilling_peck_strategy(tool, hole)

    first_peck = 0.102
    params = CuttingParameters(
        spindle_rpm=15000.0, feed_rate=120.7, plunge_feed=120.7, ramp_feed=60.0,
        feed_per_tooth_mm=0.004, depth_of_cut_mm=first_peck, stepover_mm=0.58,
        coolant=True, effective_diameter_mm=0.58, engagement_angle_deg=180.0,
        power_kw=0.05, torque_nm=0.03
    )

    op = PlannedOperation(
        id="op_drill_s1", purpose=OpPurpose.DRILLING, feature=hole,
        tool=tool, params=params,
        notes={
            "r_clearance": 1.00,  # R-plane = 3.22 + 1.00 = 4.22
            "depth_reference": "tip",
            "first_peck": first_peck,
            "subsequent_peck": first_peck,
            "peck_reduction": 0.0,
        }
    )

    from cam_engine.units import Units
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup], machine=machine, tools=[tool], material=mat
    )
    s_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=15.02, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(s_ctx)
    tp = engine.generate(op)

    plunges = [s for s in tp.segments if s.motion_type == MotionType.PLUNGE]
    assert len(plunges) >= 15

    # Target depths must follow 3.22 - 0.102 = 3.118, 3.016, 2.914, 2.812...
    expected_depths = [round(3.22 - (i + 1) * 0.102, 3) for i in range(len(plunges))]
    # Clamp last depth to 1.470
    expected_depths = [max(1.470, d) for d in expected_depths]

    for i, plunge in enumerate(plunges):
        assert pytest.approx(plunge.end[2], 1e-3) == expected_depths[i]

    # Standoff / Approach check:
    # First plunge MUST NOT start at R-plane (4.22)! It must start at standoff above 3.22
    standoff = min(0.20, max(0.02, round(0.10 * 0.58, 3)))  # 0.058 mm
    assert pytest.approx(plunges[0].start[2], 1e-3) == round(3.22 + standoff, 3)
    assert plunges[0].start[2] < 4.22 - 0.5

    # Subsequent plunges must start at prev_depth + standoff
    for i in range(1, len(plunges)):
        prev_d = plunges[i - 1].end[2]
        assert pytest.approx(plunges[i].start[2], 1e-3) == round(prev_d + standoff, 3)


def test_setup2_drill_peck_depth_sequence_and_standoff():
    """Verify that Setup 2 (Ø0.51 drill) advances by 0.108mm uniform pecks."""
    tool = Tool(
        id="T03", tool_number=3, type=ToolType.DRILL, diameter=0.51,
        point_angle=118.0, flute_length=10.0, overall_length=40.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -15.0, -4.22]), bounds_max=np.array([15.0, 5.0, 1.00]))
    setup = Setup(id="s2", name="Setup 2", model_to_setup=np.eye(4), work_offset=WorkOffset.G55, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    machine = _make_test_machine()

    # Blind hole: Surface Z = 0.00 mm, Cycle tip depth Z = -1.270 mm
    hole = MachiningFeature(
        id="hole_s2", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.255, -0.255, -1.27]), bounds_max=np.array([0.255, 0.255, 0.00]),
        depth=1.27, top_z=0.00, floor_z=-1.27, diameter=0.51, is_concave=True,
    )

    first_peck = 0.108
    params = CuttingParameters(
        spindle_rpm=15000.0, feed_rate=120.0, plunge_feed=120.0, ramp_feed=60.0,
        feed_per_tooth_mm=0.004, depth_of_cut_mm=first_peck, stepover_mm=0.51,
        coolant=True, effective_diameter_mm=0.51, engagement_angle_deg=180.0,
        power_kw=0.05, torque_nm=0.03
    )

    op = PlannedOperation(
        id="op_drill_s2", purpose=OpPurpose.DRILLING, feature=hole,
        tool=tool, params=params,
        notes={
            "r_clearance": 1.00,
            "depth_reference": "tip",
            "first_peck": first_peck,
            "subsequent_peck": first_peck,
            "peck_reduction": 0.0,
        }
    )

    from cam_engine.units import Units
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup], machine=machine, tools=[tool], material=mat
    )
    s_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=11.80, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(s_ctx)
    tp = engine.generate(op)

    plunges = [s for s in tp.segments if s.motion_type == MotionType.PLUNGE]
    assert len(plunges) >= 12

    # Target depths must follow 0.00 - 0.108 = -0.108, -0.216, -0.324, -0.432...
    expected_depths = [round(0.00 - (i + 1) * 0.108, 3) for i in range(len(plunges))]
    expected_depths = [max(-1.270, d) for d in expected_depths]

    for i, plunge in enumerate(plunges):
        assert pytest.approx(plunge.end[2], 1e-3) == expected_depths[i]


def test_grbl_expanded_peck_motion_fidelity():
    """Verify that a non-canned controller (e.g. GRBL) emits clean rapid re-entries and feed plunges."""
    tool = Tool(
        id="T02", tool_number=2, type=ToolType.DRILL, diameter=0.58,
        point_angle=118.0, flute_length=10.0, overall_length=40.0, flutes=2, material="carbide"
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, -1.0]), bounds_max=np.array([15.0, 15.0, 4.22]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    machine = _make_test_machine()

    hole = MachiningFeature(
        id="hole_s1", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75, top_z=3.22, floor_z=1.47, diameter=0.58, is_concave=True,
    )
    params = CuttingParameters(
        spindle_rpm=15000.0, feed_rate=120.7, plunge_feed=120.7, ramp_feed=60.0,
        feed_per_tooth_mm=0.004, depth_of_cut_mm=0.102, stepover_mm=0.58,
        coolant=True, effective_diameter_mm=0.58, engagement_angle_deg=180.0,
        power_kw=0.05, torque_nm=0.03
    )
    op = PlannedOperation(
        id="op_drill_s1", purpose=OpPurpose.DRILLING, feature=hole,
        tool=tool, params=params,
        notes={"r_clearance": 1.00, "depth_reference": "tip", "first_peck": 0.102, "peck_reduction": 0.0}
    )

    from cam_engine.units import Units
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup], machine=machine, tools=[tool], material=mat
    )
    s_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=15.02, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(s_ctx)
    tp = engine.generate(op)

    post_grbl = get_post_processor("grbl", work_offset="G54")
    res = post_grbl.post_process([tp], program_name="GRBL_PECK_TEST")

    # In GRBL (expanded motion):
    # First cut should rapid to ~3.278 and feed to 3.118
    assert "Z3.118" in res.gcode
    assert "Z3.016" in res.gcode
    assert "Z2.914" in res.gcode
    assert "Z2.812" in res.gcode
