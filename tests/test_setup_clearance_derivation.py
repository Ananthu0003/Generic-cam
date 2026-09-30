import numpy as np
import pytest

from cam_engine.context import (
    Fixture,
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
    SetupClearanceInfo,
    derive_setup_clearance_plane,
)
from cam_engine.geometry import TriangleMesh
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.machinability import CuttingParameters
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.post import get_post_processor
from cam_engine.errors import CamError, MACHINE_LIMIT_EXCEEDED


def _make_dummy_mesh(z_min: float = 0.0, z_max: float = 3.22) -> TriangleMesh:
    """Create a minimal TriangleMesh with known Z bounds."""
    triangles = np.array([
        [[0.0, 0.0, z_min], [10.0, 0.0, z_min], [10.0, 10.0, z_max]],
        [[0.0, 0.0, z_min], [10.0, 10.0, z_max], [0.0, 10.0, z_max]],
    ])
    return TriangleMesh(triangles=triangles)


def _make_test_machine(travel_max_z: float = 300.0, safe_retract: float = 10.0) -> MachineConfig:
    return MachineConfig(
        id="M01",
        name="Test 3-Axis Mill",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="haas",
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


def test_setup1_vs_setup2_clearance_derivation():
    """Verify that Setup 1 (Z=14.22) and Setup 2 (Z=11.00) clearance planes
    are derived from their respective transformed stock & model geometry."""
    machine = _make_test_machine(safe_retract=10.0)
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, corner_radius=0.0, flute_length=15.0,
        overall_length=50.0, flutes=3, material="carbide", can_plunge=True
    )

    # Setup 1: Top Milling (Model 0..3.22, Stock top 4.22)
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
    )
    s1_mesh = _make_dummy_mesh(0.0, 3.22)
    s1_cl = s1.compute_clearance(machine=machine, tools=[tool], mesh=s1_mesh)

    assert pytest.approx(s1_cl.z_stock_max, 1e-4) == 4.22
    assert pytest.approx(s1_cl.z_model_max, 1e-4) == 3.22
    assert pytest.approx(s1_cl.safe_retract_margin, 1e-4) == 10.0
    assert pytest.approx(s1_cl.clearance_z, 1e-4) == 14.22
    assert "stock=4.22" in s1_cl.derivation_notes
    assert "model=3.22" in s1_cl.derivation_notes
    assert "retract=10.00" in s1_cl.derivation_notes

    # Setup 2: Bottom Flip (Model -3.22..0, Stock top 1.00)
    s2_rot = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    s2_stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-5.0, -15.0, -4.22]),
        bounds_max=np.array([15.0, 5.0, 1.00])
    )
    s2 = Setup(
        id="setup_002",
        name="Setup 2 - Bottom Flip",
        model_to_setup=s2_rot,
        work_offset=WorkOffset.G55,
        stock=s2_stock,
    )
    s2_mesh = _make_dummy_mesh(-3.22, 0.0)
    s2_cl = s2.compute_clearance(machine=machine, tools=[tool], mesh=s2_mesh)

    assert pytest.approx(s2_cl.z_stock_max, 1e-4) == 1.00
    assert pytest.approx(s2_cl.z_model_max, 1e-4) == 0.00
    assert pytest.approx(s2_cl.safe_retract_margin, 1e-4) == 10.0
    assert pytest.approx(s2_cl.clearance_z, 1e-4) == 11.00
    assert "stock=1.00" in s2_cl.derivation_notes
    assert "model=0.00" in s2_cl.derivation_notes
    assert "retract=10.00" in s2_cl.derivation_notes


def test_fixture_dominance_in_clearance_plane():
    """Verify that tall clamping fixtures elevate the setup clearance plane above fixture top."""
    machine = _make_test_machine(safe_retract=10.0)
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([0.0, 0.0, 0.0]),
        bounds_max=np.array([50.0, 50.0, 10.0])
    )
    # Vise jaw fixture extending up to Z=25.0
    tall_fixture = Fixture(
        id="vise_jaw_high",
        name="High Vise Jaw",
        bounds_min=np.array([-10.0, -10.0, 0.0]),
        bounds_max=np.array([60.0, 0.0, 25.0]),
        kind="vise_jaw"
    )
    setup = Setup(
        id="setup_fixture_test",
        name="Setup with Vise",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
        fixtures=[tall_fixture]
    )

    cl = setup.compute_clearance(machine=machine)
    assert pytest.approx(cl.z_stock_max, 1e-4) == 10.0
    assert pytest.approx(cl.z_fixture_max, 1e-4) == 25.0
    assert pytest.approx(cl.highest_obstacle_z, 1e-4) == 25.0
    assert pytest.approx(cl.clearance_z, 1e-4) == 35.0  # 25.0 + 10.0
    assert "fixtures=25.00" in cl.derivation_notes


def test_machine_z_travel_limit_detection():
    """Verify that clearance planes exceeding machine Z limits are flagged."""
    # Machine with restricted Z travel max = 20.0
    machine = _make_test_machine(travel_max_z=20.0, safe_retract=10.0)
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([0.0, 0.0, 0.0]),
        bounds_max=np.array([50.0, 50.0, 15.0])
    )
    setup = Setup(
        id="setup_limit_test",
        name="Setup Limit Test",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
    )

    # Clearance = 15.0 + 10.0 = 25.0 > machine travel max 20.0
    cl = setup.compute_clearance(machine=machine)
    assert pytest.approx(cl.clearance_z, 1e-4) == 25.0
    assert cl.machine_limit_ok is False
    assert pytest.approx(cl.machine_travel_max_z, 1e-4) == 20.0


def test_gcode_operation_audit_includes_clearance_derivation():
    """Verify that the post-processor emits clear mathematical explanation of clearance plane in G-code."""
    machine = _make_test_machine(safe_retract=10.0)
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=3.0, flute_length=15.0, overall_length=50.0,
        flutes=3, material="carbide", can_plunge=True
    )
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([0.0, 0.0, 0.0]),
        bounds_max=np.array([10.0, 10.0, 4.22])
    )
    setup = Setup(
        id="setup_audit",
        name="Setup Audit",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
    )
    cl = setup.compute_clearance(machine=machine, tools=[tool])

    feat = MachiningFeature(
        id="f1", type=FeatureType.FACING_REGION,
        face_indices=[0], bounds_min=np.array([0, 0, 3.22]),
        bounds_max=np.array([10, 10, 4.22]), depth=1.0,
        top_z=4.22, floor_z=3.22, accessibility=Accessibility.TOOL_AXIS_OK,
    )
    params = CuttingParameters(
        spindle_rpm=12000.0, feed_rate=1500.0, plunge_feed=300.0,
        ramp_feed=600.0, depth_of_cut_mm=1.0, stepover_mm=1.5,
        feed_per_tooth_mm=0.0417, coolant=True,
        effective_diameter_mm=3.0, engagement_angle_deg=90.0,
        power_kw=0.5, torque_nm=0.4
    )
    op = PlannedOperation(
        id="op_facing_01", feature=feat, tool=tool,
        purpose=OpPurpose.FACING, params=params
    )

    from cam_engine.units import Units
    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        setups=[setup], machine=machine,
        tools=[tool], material=DEFAULT_MATERIALS["aluminum_6061"]
    )
    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=cl.clearance_z, finish_allowance_wall=0.0,
        finish_allowance_floor=0.0, clearance_info=cl
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    assert pytest.approx(tp.metadata["clearance_z"], 1e-4) == 14.22
    assert "clearance_info" in tp.metadata

    post = get_post_processor("haas", work_offset="G54")
    res = post.post_process([tp], program_name="TEST_CLEARANCE")

    assert "Setup Clearance: Z14.22" in res.gcode
    assert "stock=4.22" in res.gcode
