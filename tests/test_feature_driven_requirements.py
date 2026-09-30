"""Feature-driven operation planning.

Every operation is derived from the decision chain

    detected feature -> required surface -> stock condition -> accessibility
                    -> tool availability -> operation type

so the planner must emit a stage only when that chain demands it: no standing
stock means no facing, no bulk material means no roughing, no residual corner
stock means no rest pass, stock that fits in one finishing pass means no semi
pass, and a bore is bored/reamed only when its surface carries a precision
requirement.
"""

import numpy as np
import pytest

from cam_engine.context import (
    DEFAULT_MATERIALS,
    FinishRequirement,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    ToleranceModel,
    Tool,
    ToolType,
    WorkOffset,
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import Accessibility, FeatureType, MachiningFeature
from cam_engine.operations import OpPurpose, OperationPlanner
from cam_engine.requirements import (
    StockCondition,
    evaluate_feature_requirement,
    measure_stock,
)
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine


# --------------------------------------------------------------------- setup --
def _setup(tools, stock) -> tuple[PlanningContext, Setup]:
    setup = Setup(
        id="s1", name="Setup 1", model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54, stock=stock,
    )
    ctx = PlanningContext(
        model=None, setups=[setup], machine=get_default_machine(),
        tools=tools, material=DEFAULT_MATERIALS["aluminum_6061"],
    )
    return ctx, setup


def _flat_tool(tid, diameter, flute_length=30.0, corner_radius=0.0):
    return Tool(
        id=tid, tool_number=int("".join(ch for ch in tid if ch.isdigit()) or 1),
        type=ToolType.FLAT_ENDMILL, diameter=diameter, flute_length=flute_length,
        overall_length=flute_length + 40.0, flutes=4, corner_radius=corner_radius,
        can_plunge=True,
    )


def _drill(tid, diameter, flute_length=60.0):
    return Tool(
        id=tid, tool_number=int("".join(ch for ch in tid if ch.isdigit()) or 1),
        type=ToolType.DRILL, diameter=diameter, flute_length=flute_length,
        overall_length=flute_length + 40.0, flutes=2,
    )


def _feature(ftype, fid="feat", **kwargs) -> MachiningFeature:
    defaults = dict(
        id=fid, type=ftype, face_indices=[0],
        bounds_min=np.array([-20.0, -20.0, -20.0]),
        bounds_max=np.array([20.0, 20.0, 0.0]),
        depth=20.0, top_z=0.0, floor_z=-20.0,
        is_concave=True, accessibility=Accessibility.TOOL_AXIS_OK,
    )
    defaults.update(kwargs)
    return MachiningFeature(**defaults)


# ------------------------------------------------------- stock condition gate --
def test_no_operations_when_stock_matches_the_part():
    """Stock flush with the part: nothing has to be cut."""
    tools = [_flat_tool("T1_rough", 12.0), _flat_tool("T2_finish", 6.0)]
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-50.0, -50.0, -20.0]),
                  bounds_max=np.array([50.0, 50.0, 0.0]))
    ctx, setup = _setup(tools, stock)
    contour = _feature(
        FeatureType.CONTOUR, "flush_contour", is_concave=False,
        bounds_min=np.array([-50.0, -50.0, -20.0]),
        bounds_max=np.array([50.0, 50.0, 0.0]),
    )

    planner = OperationPlanner(ctx, setup, [contour])
    plan = planner.plan()

    assert plan == []
    assert planner.warnings == []


def test_facing_demanded_only_when_stock_stands_above_the_face():
    tools = [Tool(id="T1_face", tool_number=1, type=ToolType.FACE_MILL,
                  diameter=50.0, flute_length=15.0, overall_length=80.0, flutes=5)]
    facing = _feature(
        FeatureType.FACING_REGION, "top_face", is_concave=False,
        bounds_min=np.array([-40.0, -40.0, -1.0]),
        bounds_max=np.array([40.0, 40.0, 0.0]),
        depth=1.0, top_z=0.0, floor_z=-1.0, notes={"excess_height": 1.0},
    )

    # 2 mm of billet above the model top face -> facing is required.
    with_stock = Stock(kind=StockKind.BOX,
                       bounds_min=np.array([-50.0, -50.0, -20.0]),
                       bounds_max=np.array([50.0, 50.0, 2.0]))
    ctx, setup = _setup(tools, with_stock)
    plan = OperationPlanner(ctx, setup, [facing]).plan()
    assert [op.purpose for op in plan] == [OpPurpose.FACING]
    assert plan[0].notes["excess_height"] == pytest.approx(2.0)
    assert plan[0].notes["requirement"]["required_surface"] == "planar_face"

    # Stock top exactly on the model top face -> no facing pass.
    flush_stock = Stock(kind=StockKind.BOX,
                        bounds_min=np.array([-50.0, -50.0, -20.0]),
                        bounds_max=np.array([50.0, 50.0, 0.0]))
    ctx, setup = _setup(tools, flush_stock)
    flush_face = _feature(
        FeatureType.FACING_REGION, "flush_face", is_concave=False,
        bounds_min=np.array([-40.0, -40.0, -1.0]),
        bounds_max=np.array([40.0, 40.0, 0.0]),
        depth=0.0, top_z=0.0, floor_z=0.0, notes={"excess_height": 0.0},
    )
    assert OperationPlanner(ctx, setup, [flush_face]).plan() == []


def test_roughing_skipped_when_the_cavity_is_already_open():
    """A feature the stock no longer fills has no bulk left to remove."""
    tools = [_flat_tool("T1_rough", 12.0), _flat_tool("T2_finish", 6.0)]
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-50.0, -50.0, -30.0]),
                  bounds_max=np.array([50.0, 50.0, 0.0]))
    ctx, setup = _setup(tools, stock)

    # The slot lies entirely outside the stock: no material to remove.
    outside = _feature(
        FeatureType.SLOT, "outside_slot",
        bounds_min=np.array([60.0, 60.0, -8.0]),
        bounds_max=np.array([80.0, 64.0, 0.0]),
        depth=8.0, top_z=0.0, floor_z=-8.0,
    )
    assert measure_stock(outside, setup).has_stock is False
    assert OperationPlanner(ctx, setup, [outside]).plan() == []


# ---------------------------------------------------------- corner stock gate --
def test_rest_machining_only_when_corner_stock_remains():
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-50.0, -50.0, -30.0]),
                  bounds_max=np.array([50.0, 50.0, 5.0]))
    pocket = _feature(
        FeatureType.POCKET, "pocket",
        bounds_min=np.array([-25.0, -15.0, -12.0]),
        bounds_max=np.array([25.0, 15.0, 0.0]),
        depth=12.0, floor_z=-12.0,
    )

    # Flat roughing cutter in a sharp corner -> full cutter radius survives.
    sharp_tools = [_flat_tool("T1_rough", 12.0), _flat_tool("T2_finish", 6.0)]
    ctx, setup = _setup(sharp_tools, stock)
    purposes = {op.purpose for op in OperationPlanner(ctx, setup, [pocket]).plan()}
    assert OpPurpose.REST_MACHINING in purposes

    # Bull-nose cutter whose corner radius already matches the corner -> nothing left.
    bull_tools = [
        _flat_tool("T1_rough", 12.0, corner_radius=6.0),
        _flat_tool("T2_finish", 6.0),
    ]
    ctx, setup = _setup(bull_tools, stock)
    bull_pocket = _feature(
        FeatureType.POCKET, "pocket_round",
        bounds_min=np.array([-25.0, -15.0, -12.0]),
        bounds_max=np.array([25.0, 15.0, 0.0]),
        depth=12.0, floor_z=-12.0, fillet_radius=6.0,
    )
    planner = OperationPlanner(ctx, setup, [bull_pocket])
    purposes = {op.purpose for op in planner.plan()}
    assert OpPurpose.REST_MACHINING not in purposes


def test_corner_stock_calculation_uses_rough_tool_and_corner_radius():
    stock = StockCondition(kind="box", axial_stock_mm=0.0, radial_stock_mm=0.0,
                           interior_stock_mm=10.0, finish_allowance_mm=0.3)
    tool = _flat_tool("T1_rough", 12.0)

    # Sharp corner: the whole cutter radius survives there.
    assert stock.corner_stock_mm(tool.diameter / 2.0, None) == pytest.approx(6.0)
    assert stock.demands_rest(tool, None)[0] is True

    # A known corner radius swallows the cutter radius one-to-one.
    assert stock.corner_stock_mm(tool.diameter / 2.0, 6.0) == pytest.approx(0.0)
    assert stock.demands_rest(tool, 6.0)[0] is False
    # ... and anything inside the finish allowance is left to finishing.
    assert stock.demands_rest(tool, 5.5)[0] is False


# ------------------------------------------------------------ remaining stock --
def test_no_semi_finishing_when_stock_fits_in_one_finishing_pass():
    """Flat roughing cutter leaves no cusp: the finish allowance alone is taken."""
    tools = [_flat_tool("T1_rough", 12.0), _flat_tool("T2_finish", 6.0)]
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-50.0, -50.0, -30.0]),
                  bounds_max=np.array([50.0, 50.0, 5.0]))
    ctx, setup = _setup(tools, stock)
    pocket = _feature(
        FeatureType.POCKET, "pocket",
        bounds_min=np.array([-25.0, -15.0, -12.0]),
        bounds_max=np.array([25.0, 15.0, 0.0]),
        depth=12.0, floor_z=-12.0,
    )

    planner = OperationPlanner(ctx, setup, [pocket])
    purposes = {op.purpose for op in planner.plan()}
    assert OpPurpose.ROUGHING in purposes
    assert OpPurpose.SEMI_FINISHING not in purposes


def test_semi_finishing_demanded_when_remaining_stock_exceeds_one_pass():
    stock = StockCondition(kind="box", axial_stock_mm=0.0, radial_stock_mm=0.0,
                           interior_stock_mm=10.0, finish_allowance_mm=0.3)
    rough = _flat_tool("T1_rough", 12.0, corner_radius=3.0)

    # 1.8 mm of cusps + 0.3 mm allowance does not fit into a 0.6 mm finish pass.
    assert stock.demands_semi(rough, 6.0, 2.0)[0] is True
    # A finer 4 mm stepover leaves cusps that a 6 mm finish pass can take.
    assert stock.demands_semi(rough, 4.0, 6.0)[0] is False


# ------------------------------------------------------- precision hole chain --
def _bore_feature(precision: bool) -> MachiningFeature:
    return _feature(
        FeatureType.BORE, "bore", diameter=10.0, depth=15.0, top_z=0.0,
        floor_z=-15.0, bounds_min=np.array([-5.0, -5.0, -15.0]),
        bounds_max=np.array([5.0, 5.0, 0.0]),
        tolerances=ToleranceModel(upper_tol_mm=0.01, lower_tol_mm=-0.01)
        if precision else None,
    )


def _hole_tools():
    return [
        _drill("T1_drill", 9.8),
        Tool(id="T2_bore", tool_number=2, type=ToolType.BORING_BAR,
             diameter=10.0, flute_length=30.0, overall_length=90.0),
        Tool(id="T3_ream", tool_number=3, type=ToolType.REAMER,
             diameter=10.0, flute_length=30.0, overall_length=80.0, flutes=6),
        _flat_tool("T4_mill", 6.0, flute_length=30.0),
    ]


def test_precision_bore_is_bored_and_reamed():
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-30.0, -30.0, -20.0]),
                  bounds_max=np.array([30.0, 30.0, 10.0]))
    ctx, setup = _setup(_hole_tools(), stock)
    planner = OperationPlanner(ctx, setup, [_bore_feature(precision=True)])
    plan = planner.plan()
    purposes = {op.purpose for op in plan}

    assert OpPurpose.DRILLING in purposes
    assert OpPurpose.BORING in purposes
    assert OpPurpose.REAMING in purposes
    for op in plan:
        assert op.notes["requirement"]["precision_required"] is True


def test_ordinary_bore_is_only_drilled():
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-30.0, -30.0, -20.0]),
                  bounds_max=np.array([30.0, 30.0, 10.0]))
    ctx, setup = _setup(_hole_tools(), stock)
    planner = OperationPlanner(ctx, setup, [_bore_feature(precision=False)])
    purposes = {op.purpose for op in planner.plan()}

    assert OpPurpose.DRILLING in purposes
    assert OpPurpose.BORING not in purposes
    assert OpPurpose.REAMING not in purposes


def test_finish_requirement_alone_declares_a_precision_surface():
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-30.0, -30.0, -20.0]),
                  bounds_max=np.array([30.0, 30.0, 10.0]))
    ctx, setup = _setup(_hole_tools(), stock)
    bore = _feature(
        FeatureType.BORE, "bore", diameter=10.0, depth=15.0, top_z=0.0,
        floor_z=-15.0, bounds_min=np.array([-5.0, -5.0, -15.0]),
        bounds_max=np.array([5.0, 5.0, 0.0]),
        finish=FinishRequirement(tolerance_mm=0.02, surface_finish_um=1.6),
    )
    req = evaluate_feature_requirement(bore, setup, ctx)
    assert req.precision_required is True
    assert req.demands_precision_hole_finish()[0] is True
    assert "tolerance" in req.evidence["precision"]


# --------------------------------------------------------------- deep pocket --
def _deep_pocket(depth: float, span: float) -> MachiningFeature:
    return _feature(
        FeatureType.POCKET, f"pocket_{depth}", diameter=span,
        bounds_min=np.array([-span / 2, -span / 2, -depth]),
        bounds_max=np.array([span / 2, span / 2, 0.0]),
        depth=depth, top_z=0.0, floor_z=-depth,
    )


def test_deep_pocket_plans_the_pocket_strategy():
    tools = [_flat_tool("T1_rough", 12.0, flute_length=45.0),
             _flat_tool("T2_finish", 6.0)]
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-40.0, -40.0, -50.0]),
                  bounds_max=np.array([40.0, 40.0, 5.0]))
    ctx, setup = _setup(tools, stock)

    planner = OperationPlanner(ctx, setup, [_deep_pocket(depth=40.0, span=20.0)])
    plan = planner.plan()
    roughs = [op for op in plan if op.purpose is OpPurpose.ROUGHING]
    assert len(roughs) == 1
    assert roughs[0].notes["strategy"] == "pocket"
    assert roughs[0].notes["method"] == "adaptive"

    # A shallow pocket keeps the plain cavity clearing strategy.
    shallow = OperationPlanner(ctx, setup, [_deep_pocket(depth=4.0, span=40.0)])
    shallow_roughs = [op for op in shallow.plan() if op.purpose is OpPurpose.ROUGHING]
    assert shallow_roughs[0].notes["strategy"] == "cavity"
    assert shallow_roughs[0].notes.get("method") != "adaptive"


def test_deep_pocket_strategy_generates_a_toolpath():
    tools = [_flat_tool("T1_rough", 12.0, flute_length=45.0),
             _flat_tool("T2_finish", 6.0)]
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-40.0, -40.0, -50.0]),
                  bounds_max=np.array([40.0, 40.0, 5.0]))
    ctx, setup = _setup(tools, stock)
    plan = OperationPlanner(ctx, setup, [_deep_pocket(depth=40.0, span=20.0)]).plan()
    rough = next(op for op in plan if op.purpose is OpPurpose.ROUGHING)

    engine = StrategyEngine(StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=10.0, finish_allowance_wall=0.5, finish_allowance_floor=0.3,
    ))
    tp = engine.generate(rough)
    assert tp is not None and tp.operation_id == rough.id
    assert not tp.is_empty_or_skipped
    assert len(tp.segments) > 0
    assert tp.metadata.get("strategy") == "adaptive_trochoidal"


# --------------------------------------------------------------------- trace --
def test_planned_operations_carry_their_requirement_trace():
    tools = [_flat_tool("T1_rough", 12.0), _flat_tool("T2_finish", 6.0),
             _drill("T3_drill", 5.0)]
    stock = Stock(kind=StockKind.BOX,
                  bounds_min=np.array([-50.0, -50.0, -30.0]),
                  bounds_max=np.array([50.0, 50.0, 5.0]))
    ctx, setup = _setup(tools, stock)
    features = [
        _feature(FeatureType.POCKET, "pocket",
                 bounds_min=np.array([-25.0, -15.0, -12.0]),
                 bounds_max=np.array([25.0, 15.0, 0.0]),
                 depth=12.0, floor_z=-12.0),
        _feature(FeatureType.HOLE, "hole", diameter=5.0, depth=15.0,
                 top_z=0.0, floor_z=-15.0,
                 bounds_min=np.array([-2.5, -2.5, -15.0]),
                 bounds_max=np.array([2.5, 2.5, 0.0])),
    ]

    plan = OperationPlanner(ctx, setup, features).plan()
    assert plan
    for op in plan:
        trace = op.notes["requirement"]
        assert trace["purpose"] == op.purpose.value
        assert trace["feature_type"] == op.feature.type.value
        assert trace["required_surface"]
        assert trace["decision"]
        assert "stock_condition" in trace
        assert trace["accessible"] is True
