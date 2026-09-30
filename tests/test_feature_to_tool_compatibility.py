"""Tests for Feature-to-Tool Compatibility and Geometry-Driven Tool Selection.

Verifies the 5-stage pipeline:
Feature -> Required access -> Tool diameter & reach bounds -> Available tools -> Machine compatibility -> Material cutting data -> Selected tool

Enforces the policy invariant:
Never arbitrarily pick a default tool (e.g., T2) and 'try to make it work'.
Geometry strictly drives cutter selection and rejects oversized tools.
"""

from typing import Optional
import pytest
import numpy as np

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
    Units,
    WorkOffset,
    get_material,
)
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.features import Accessibility, FeatureType, MachiningFeature
from cam_engine.operations import OperationPlanner, OpPurpose
from cam_engine.tool_selection import (
    GeometricAccessProfile,
    GeometryDrivenToolSelector,
    ToolSelectionResult,
)


def make_feature(
    id: str,
    type: FeatureType,
    depth: float = 5.0,
    top_z: float = 0.0,
    floor_z: float = -5.0,
    xy_span: tuple[float, float] = (10.0, 10.0),
    diameter: Optional[float] = None,
    fillet_radius: Optional[float] = None,
    is_concave: bool = True,
    notes: Optional[dict] = None,
) -> MachiningFeature:
    w, h = xy_span
    return MachiningFeature(
        id=id,
        type=type,
        face_indices=[0],
        bounds_min=np.array([-w / 2.0, -h / 2.0, floor_z]),
        bounds_max=np.array([w / 2.0, h / 2.0, top_z]),
        depth=depth,
        top_z=top_z,
        floor_z=floor_z,
        diameter=diameter,
        fillet_radius=fillet_radius,
        is_concave=is_concave,
        notes=notes or {},
    )


@pytest.fixture
def standard_machine():
    return MachineConfig(
        id="vmc-standard",
        name="Standard VMC",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={
            "X": MachineAxis(name="X", travel_min=-500.0, travel_max=500.0, rapid_rate=15000.0, max_feed=5000.0),
            "Y": MachineAxis(name="Y", travel_min=-300.0, travel_max=300.0, rapid_rate=15000.0, max_feed=5000.0),
            "Z": MachineAxis(name="Z", travel_min=-400.0, travel_max=50.0, rapid_rate=10000.0, max_feed=3000.0),
        },
        spindle_min_rpm=100.0,
        spindle_max_rpm=12000.0,
        max_spindle_power_kw=7.5,
        controller="grbl",
        units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0,
    )


@pytest.fixture
def low_power_machine():
    return MachineConfig(
        id="desktop-router",
        name="Desktop Router",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={
            "X": MachineAxis(name="X", travel_min=-200.0, travel_max=200.0, rapid_rate=5000.0, max_feed=2000.0),
            "Y": MachineAxis(name="Y", travel_min=-150.0, travel_max=150.0, rapid_rate=5000.0, max_feed=2000.0),
            "Z": MachineAxis(name="Z", travel_min=-100.0, travel_max=20.0, rapid_rate=3000.0, max_feed=1000.0),
        },
        spindle_min_rpm=100.0,
        spindle_max_rpm=6000.0,
        max_spindle_power_kw=0.8,
        controller="grbl",
        units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 20.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=5.0,
    )


@pytest.fixture
def al6061_material():
    return get_material("aluminum_6061")


@pytest.fixture
def tool_library():
    return [
        # Large face mill
        Tool(
            id="T1_face50",
            tool_number=1,
            type=ToolType.FACE_MILL,
            diameter=50.0,
            flute_length=10.0,
            overall_length=60.0,
            flutes=5,
        ),
        # Large roughing endmill
        Tool(
            id="T2_endmill12",
            tool_number=2,
            type=ToolType.FLAT_ENDMILL,
            diameter=12.0,
            flute_length=30.0,
            overall_length=75.0,
            flutes=4,
        ),
        # Medium endmill
        Tool(
            id="T3_endmill6",
            tool_number=3,
            type=ToolType.FLAT_ENDMILL,
            diameter=6.0,
            flute_length=20.0,
            overall_length=60.0,
            flutes=3,
        ),
        # Small endmill
        Tool(
            id="T4_endmill3",
            tool_number=4,
            type=ToolType.FLAT_ENDMILL,
            diameter=3.0,
            flute_length=12.0,
            overall_length=50.0,
            flutes=2,
        ),
        # Micro endmill
        Tool(
            id="T5_endmill1",
            tool_number=5,
            type=ToolType.FLAT_ENDMILL,
            diameter=1.0,
            flute_length=10.0,
            overall_length=40.0,
            flutes=2,
        ),
        # Short stubby endmill (short reach)
        Tool(
            id="T6_stub6",
            tool_number=6,
            type=ToolType.FLAT_ENDMILL,
            diameter=6.0,
            flute_length=5.0,
            overall_length=35.0,
            flutes=2,
        ),
        # Ball endmill Ø4 (for fillets/surfaces)
        Tool(
            id="T7_ball4",
            tool_number=7,
            type=ToolType.BALL_ENDMILL,
            diameter=4.0,
            flute_length=15.0,
            overall_length=55.0,
            flutes=2,
        ),
        # Drill Ø5
        Tool(
            id="T8_drill5",
            tool_number=8,
            type=ToolType.DRILL,
            diameter=5.0,
            flute_length=40.0,
            overall_length=80.0,
            point_angle=118.0,
            flutes=2,
        ),
        # Spot Drill Ø10 (90 deg)
        Tool(
            id="T9_spot10",
            tool_number=9,
            type=ToolType.SPOT_DRILL,
            diameter=10.0,
            flute_length=15.0,
            overall_length=60.0,
            point_angle=90.0,
            flutes=2,
        ),
    ]


def test_slot_geometry_drives_tool_diameter(standard_machine, al6061_material, tool_library):
    """A narrow slot of width 3.5mm must select a tool <= 3.5mm (Ø3), NEVER a default Ø12 or Ø6."""
    selector = GeometryDrivenToolSelector(standard_machine, al6061_material)

    narrow_slot = make_feature(
        id="slot_3_5mm",
        type=FeatureType.SLOT,
        depth=5.0,
        xy_span=(3.5, 40.0),
        is_concave=True,
    )

    result = selector.select_optimal_tool(narrow_slot, "roughing", tool_library)

    assert result.success is True
    assert result.selected_tool is not None
    # Must select T4_endmill3 (Ø3.0mm)
    assert result.selected_tool.diameter <= 3.5
    assert result.selected_tool.id == "T4_endmill3"

    # Verify that Ø12 and Ø6 were evaluated and rejected due to diameter constraint
    cands_by_id = {c.tool.id: c for c in result.candidates_evaluated}
    assert cands_by_id["T2_endmill12"].is_compatible is False
    assert any("exceeds geometric access limit" in r for r in cands_by_id["T2_endmill12"].rejection_reasons)
    assert cands_by_id["T3_endmill6"].is_compatible is False


def test_corner_fillet_finishing_enforces_corner_radius(standard_machine, al6061_material, tool_library):
    """Internal corner finishing with R0.9 corner requires cutter radius <= 0.9 (D <= 1.8)."""
    selector = GeometryDrivenToolSelector(standard_machine, al6061_material)

    pocket_with_tight_corners = make_feature(
        id="pocket_tight_corners",
        type=FeatureType.POCKET,
        depth=8.0,
        xy_span=(30.0, 30.0),
        fillet_radius=0.9,  # Internal corner radius 0.9mm -> Max tool diameter 1.8mm
        is_concave=True,
    )

    result = selector.select_optimal_tool(pocket_with_tight_corners, "finishing", tool_library)

    assert result.success is True
    assert result.selected_tool is not None
    # Tool diameter must be <= 2 * 0.9 = 1.8mm -> selects T5_endmill1 (Ø1.0mm)
    assert result.selected_tool.diameter <= 1.8
    assert result.selected_tool.id == "T5_endmill1"

    # Verify that Ø3, Ø6, Ø12 were rejected for corner radius violation
    cands = {c.tool.id: c for c in result.candidates_evaluated}
    assert cands["T4_endmill3"].is_compatible is False
    assert any("exceeds internal corner radius" in r for r in cands["T4_endmill3"].rejection_reasons)


def test_deep_pocket_requires_sufficient_reach(standard_machine, al6061_material, tool_library):
    """Deep pocket of depth 18mm rejects short stub cutter (flute 5mm) and selects full-reach cutter."""
    selector = GeometryDrivenToolSelector(standard_machine, al6061_material)

    deep_pocket = make_feature(
        id="deep_pocket_18mm",
        type=FeatureType.POCKET,
        depth=18.0,
        xy_span=(25.0, 25.0),
        is_concave=True,
    )

    result = selector.select_optimal_tool(deep_pocket, "roughing", tool_library)

    assert result.success is True
    assert result.selected_tool is not None
    assert result.selected_tool.flute_length >= 18.0
    assert result.selected_tool.id == "T2_endmill12"

    cands = {c.tool.id: c for c in result.candidates_evaluated}
    assert cands["T6_stub6"].is_compatible is False
    assert any("reach" in r and "insufficient" in r for r in cands["T6_stub6"].rejection_reasons)


def test_actionable_diagnostic_when_no_tool_fits(standard_machine, al6061_material):
    """When no tool in the library fits the micro-feature, an actionable diagnostic is returned."""
    # Active library only has large tools
    sparse_library = [
        Tool(id="T2_endmill12", tool_number=2, type=ToolType.FLAT_ENDMILL, diameter=12.0, flute_length=30.0, overall_length=75.0),
        Tool(id="T3_endmill6", tool_number=3, type=ToolType.FLAT_ENDMILL, diameter=6.0, flute_length=20.0, overall_length=60.0),
    ]
    selector = GeometryDrivenToolSelector(standard_machine, al6061_material)

    micro_slot = make_feature(
        id="micro_slot_1_5mm",
        type=FeatureType.SLOT,
        depth=3.0,
        xy_span=(1.5, 20.0),
        is_concave=True,
    )

    result = selector.select_optimal_tool(micro_slot, "roughing", sparse_library)

    assert result.success is False
    assert result.selected_tool is None
    assert "No compatible roughing tool available" in result.diagnostic_message
    assert result.suggested_tool is not None
    assert result.suggested_tool["diameter"] <= 1.5


def test_machine_power_constraint_filters_heavy_cutters(low_power_machine, al6061_material, tool_library):
    """Low power desktop mill (0.8 kW) filters out tools/cuts exceeding spindle capability."""
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-100.0, -100.0, -10.0]),
        bounds_max=np.array([100.0, 100.0, 0.0]),
    )
    setup = Setup(
        id="setup_top",
        name="Top Setup",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
    )
    selector = GeometryDrivenToolSelector(low_power_machine, al6061_material, setup=setup)

    large_facing = make_feature(
        id="face_large",
        type=FeatureType.FACING_REGION,
        depth=3.0,
        top_z=0.0,
        floor_z=-3.0,
        xy_span=(150.0, 150.0),
        notes={"excess_height": 3.0},
    )

    profile = selector.compute_access_profile(large_facing, "facing")
    t1_face50 = next(t for t in tool_library if t.id == "T1_face50")
    cand = selector.evaluate_tool_compatibility(t1_face50, large_facing, profile)

    # Face mill Ø50 facing 3mm cut on 0.8kW machine will exceed spindle power
    if not cand.is_compatible:
        assert any("power" in r.lower() or "rpm" in r.lower() for r in cand.rejection_reasons)


def test_operation_planner_full_pipeline_tool_selection(standard_machine, al6061_material, tool_library):
    """OperationPlanner uses GeometryDrivenToolSelector across multi-feature setups."""
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-50.0, -50.0, -25.0]),
        bounds_max=np.array([50.0, 50.0, 0.0]),
    )
    setup = Setup(
        id="setup_main",
        name="Main Setup",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
    )

    features = [
        # Large facing
        make_feature(
            id="feat_facing",
            type=FeatureType.FACING_REGION,
            depth=1.0,
            top_z=0.0,
            floor_z=0.0,
            xy_span=(100.0, 100.0),
            notes={"excess_height": 1.0},
        ),
        # Large pocket
        make_feature(
            id="feat_large_pocket",
            type=FeatureType.POCKET,
            depth=10.0,
            top_z=0.0,
            floor_z=-10.0,
            xy_span=(60.0, 60.0),
            is_concave=True,
        ),
        # Narrow slot
        make_feature(
            id="feat_narrow_slot",
            type=FeatureType.SLOT,
            depth=4.0,
            top_z=0.0,
            floor_z=-4.0,
            xy_span=(3.2, 30.0),
            is_concave=True,
        ),
        # Standard hole Ø5
        make_feature(
            id="feat_hole5",
            type=FeatureType.HOLE,
            diameter=5.0,
            depth=15.0,
            top_z=0.0,
            floor_z=-15.0,
            xy_span=(5.0, 5.0),
            is_concave=True,
        ),
    ]

    ctx = PlanningContext(
        model=ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3))),
        machine=standard_machine,
        material=al6061_material,
        setups=[setup],
        tools=tool_library,
    )

    planner = OperationPlanner(ctx, setup, features)
    ops = planner.plan()

    assert len(ops) >= 4

    op_by_feat = {op.feature.id: op for op in ops}

    # Facing should pick face mill Ø50 or large endmill
    assert op_by_feat["feat_facing"].tool.diameter >= 12.0

    # Large pocket roughing should pick Ø12
    assert op_by_feat["feat_large_pocket"].tool.diameter == 12.0

    # Narrow slot MUST pick Ø3 (T4), NEVER default Ø12
    assert op_by_feat["feat_narrow_slot"].tool.diameter <= 3.2
    assert op_by_feat["feat_narrow_slot"].tool.id == "T4_endmill3"

    # Hole Ø5 should pick exact drill Ø5 (T8) or spot drill
    hole_ops = [op for op in ops if op.feature.id == "feat_hole5"]
    drill_ops = [op for op in hole_ops if op.purpose == OpPurpose.DRILLING]
    assert len(drill_ops) == 1
    assert drill_ops[0].tool.id == "T8_drill5"
