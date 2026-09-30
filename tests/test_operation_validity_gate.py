"""Operation validity gate: questionable G-code is never emitted.

Before an operation is accepted the following must hold:

    geometry exists, target surface exists, tool compatible, tool access valid,
    depth valid, toolpath intersects stock, toolpath reaches target,
    compensation valid, feed valid, rpm valid, holder clearance valid,
    setup valid

A critical failure blocks G-code generation entirely.
"""

import numpy as np
import pytest

from cam_engine.context import (
    DEFAULT_MATERIALS,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
)
from cam_engine.demo_models import get_default_machine
from cam_engine.errors import CamError, GCODE_VALIDATION_FAILED
from cam_engine.features import Accessibility, FeatureType, MachiningFeature
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import OpPurpose, PlannedOperation
from cam_engine.post import get_post_processor
from cam_engine.toolpaths.semantic import (
    CoordSpace,
    MotionSegment,
    MotionType,
    Toolpath,
)
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from cam_engine.validity import (
    CHECK_ORDER,
    COMPENSATION_VALID,
    DEPTH_VALID,
    FEED_VALID,
    GEOMETRY_EXISTS,
    HOLDER_CLEARANCE_VALID,
    OperationValidityChecker,
    RPM_VALID,
    SETUP_VALID,
    TARGET_SURFACE_EXISTS,
    TOOL_ACCESS_VALID,
    TOOL_COMPATIBLE,
    TOOLPATH_INTERSECTS_STOCK,
    TOOLPATH_REACHES_TARGET,
    assert_valid_for_post,
)


# ------------------------------------------------------------------ fixtures --
def _feature(**kwargs) -> MachiningFeature:
    defaults = dict(
        id="pocket_1",
        type=FeatureType.POCKET,
        face_indices=[3],
        bounds_min=np.array([-20.0, -20.0, -10.0]),
        bounds_max=np.array([20.0, 20.0, 0.0]),
        depth=10.0,
        top_z=0.0,
        floor_z=-10.0,
        is_concave=True,
        accessibility=Accessibility.TOOL_AXIS_OK,
    )
    defaults.update(kwargs)
    return MachiningFeature(**defaults)


def _tool(diameter=6.0, flute_length=30.0, tool_type=ToolType.FLAT_ENDMILL):
    return Tool(id="T01", tool_number=1, type=tool_type, diameter=diameter,
                flute_length=flute_length, overall_length=flute_length + 40.0,
                flutes=4, can_plunge=True)


def _setup(bounds_min=(-30.0, -30.0, -25.0), bounds_max=(30.0, 30.0, 5.0),
           transform=None) -> Setup:
    return Setup(
        id="s1", name="Setup 1",
        model_to_setup=np.eye(4) if transform is None else transform,
        work_offset=WorkOffset.G54,
        stock=Stock(kind=StockKind.BOX,
                    bounds_min=np.array(bounds_min, dtype=float),
                    bounds_max=np.array(bounds_max, dtype=float)),
    )


def _context(setup, tools=None) -> PlanningContext:
    return PlanningContext(
        model=None, setups=[setup], machine=get_default_machine(),
        tools=tools if tools is not None else [_tool()],
        material=DEFAULT_MATERIALS["aluminum_6061"],
    )


def _params() -> CuttingParameters:
    return CuttingParameters(
        spindle_rpm=6000.0, feed_rate=1200.0, plunge_feed=300.0,
        ramp_feed=400.0, feed_per_tooth_mm=0.05, depth_of_cut_mm=2.0,
        stepover_mm=3.0, coolant=True, effective_diameter_mm=6.0,
        engagement_angle_deg=90.0, power_kw=1.0, torque_nm=1.6,
        machine_max_rpm=15000.0, machine_min_rpm=100.0,
        machine_max_feed=3000.0,
    )


def _op(feature=None, tool=None, purpose=OpPurpose.ROUGHING) -> PlannedOperation:
    return PlannedOperation(
        id="op_test", purpose=purpose,
        feature=feature if feature is not None else _feature(),
        tool=tool if tool is not None else _tool(),
        params=_params(),
    )


def _toolpath(feed=1200.0, spindle=6000.0, z_start=0.0, z_end=-10.0,
              xy=(0.0, 0.0)) -> Toolpath:
    tp = Toolpath(operation_id="op_test", feature_id="pocket_1", tool_id="T01",
                  purpose="roughing", space=CoordSpace.SETUP)
    tp.metadata["compensation_mode"] = "none"
    tp.metadata["toolpath_representation"] = "center_path"
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([xy[0], xy[1], z_start]),
        end=np.array([xy[0] + 5.0, xy[1], z_end]),
        space=CoordSpace.SETUP, feed=feed, spindle=spindle,
        tool_id="T01", operation_id="op_test", feature_id="pocket_1",
    ))
    return tp


def _generate(op) -> Toolpath:
    """Run a real strategy so the toolpath carries a full 12-check stamp."""
    setup = _setup()
    ctx = _context(setup)
    strat = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=15.0, finish_allowance_wall=0.3, finish_allowance_floor=0.3,
    )
    return StrategyEngine(strat).generate(op)


# ------------------------------------------------------------- the 12 checks --
def test_every_operation_carries_all_twelve_checks():
    tp = _generate(_op())
    stamp = tp.metadata["validity"]

    assert [c["name"] for c in stamp["checks"]] == list(CHECK_ORDER)
    assert len(CHECK_ORDER) == 12
    assert stamp["passed"] is True
    assert all(c["evaluated"] or c["passed"] for c in stamp["checks"])


@pytest.mark.parametrize("check_name", CHECK_ORDER)
def test_each_check_reports_when_it_fails(check_name):
    """Every check can say 'no' — nothing silently passes."""
    setup = _setup()
    ctx = _context(setup)
    checker = OperationValidityChecker(context=ctx, setup=setup)

    feature = _feature()
    tool = _tool()
    tp = _toolpath()

    if check_name == GEOMETRY_EXISTS:
        feature = _feature(bounds_min=np.array([np.nan, 0.0, -10.0]))
    elif check_name == TARGET_SURFACE_EXISTS:
        feature = _feature(face_indices=[], source_edges=[], surface_type=None)
    elif check_name == TOOL_COMPATIBLE:
        tool = _tool(diameter=90.0)          # far wider than the 40 mm pocket
    elif check_name == TOOL_ACCESS_VALID:
        feature = _feature(accessibility=Accessibility.INACCESSIBLE)
    elif check_name == DEPTH_VALID:
        feature = _feature(floor_z=2.0, top_z=0.0)   # floor above the top face
    elif check_name == TOOLPATH_INTERSECTS_STOCK:
        tp = _toolpath(xy=(400.0, 400.0))             # far outside the stock
    elif check_name == TOOLPATH_REACHES_TARGET:
        tp = _toolpath(z_start=-1.0, z_end=-2.0)      # stops 8 mm above the floor
    elif check_name == COMPENSATION_VALID:
        tp = _toolpath()
        tp.metadata["compensation_mode"] = "geometry"
        tp.metadata["toolpath_representation"] = "center_path"
    elif check_name == FEED_VALID:
        tp = _toolpath(feed=-100.0)
    elif check_name == RPM_VALID:
        tp = _toolpath(spindle=900_000.0)
    elif check_name == HOLDER_CLEARANCE_VALID:
        tool = _tool(flute_length=3.0)                # 3 mm flute, 10 mm deep
    elif check_name == SETUP_VALID:
        transform = np.eye(4)
        transform[0, 0] = 0.0                         # singular
        setup = _setup(transform=transform)
        ctx = _context(setup)
        checker = OperationValidityChecker(context=ctx, setup=setup)

    report = checker.check(_op(feature=feature, tool=tool), tp)
    failed = {c.name for c in report.critical_failures}
    assert check_name in failed, f"{check_name} did not fail: {report.summary()}"
    assert report.passed is False

    failing = next(c for c in report.critical_failures if c.name == check_name)
    assert failing.message
    assert failing.code
    assert {e.context["check"] for e in report.blocking_errors()} >= {check_name}


def test_healthy_operation_passes_every_check():
    setup = _setup()
    ctx = _context(setup)
    checker = OperationValidityChecker(context=ctx, setup=setup)
    report = checker.check(_op(), _toolpath())

    assert report.passed is True, report.summary()
    assert report.critical_failures == []
    assert [c.name for c in report.checks] == list(CHECK_ORDER)


def test_skipped_toolpath_is_not_an_operation():
    tp = _toolpath()
    tp.metadata["skipped"] = "no stock left"
    report = OperationValidityChecker().check(_op(), tp)
    assert report.skipped is True
    assert report.passed is True
    assert report.critical_failures == []


# ------------------------------------------------------------ blocking gate --
def test_stamped_failure_blocks_gcode_generation():
    tp = _toolpath()
    tp.metadata["validity"] = {
        "operation_id": "op_test", "tool_id": "T01", "purpose": "roughing",
        "skipped": False, "passed": False,
        "checks": [{"name": TOOLPATH_REACHES_TARGET, "passed": False,
                    "message": "stops 8 mm above the target floor",
                    "critical": True, "evaluated": True,
                    "code": "INVALID_TOOLPATH"}],
    }

    with pytest.raises(CamError) as exc:
        get_post_processor("grbl", work_offset="G54").post_process(
            [tp], program_name="SHOULD_NOT_EXIST")
    assert exc.value.code == GCODE_VALIDATION_FAILED
    failures = exc.value.context["blocking_failures"]
    assert [f["check"] for f in failures] == [TOOLPATH_REACHES_TARGET]
    assert "SHOULD_NOT_EXIST" not in exc.value.message


def test_invalid_feed_blocks_gcode_without_a_stamp():
    """Local checks work even on a toolpath that was never stamped."""
    tp = _toolpath(feed=0.0)
    with pytest.raises(CamError) as exc:
        get_post_processor("grbl", work_offset="G54").post_process([tp])
    assert exc.value.code == GCODE_VALIDATION_FAILED
    assert any(f["check"] == FEED_VALID for f in exc.value.context["blocking_failures"])


def test_rpm_outside_the_controller_range_blocks_gcode():
    tp = _toolpath(spindle=60_000.0)
    with pytest.raises(CamError) as exc:
        get_post_processor("grbl", work_offset="G54").post_process([tp])
    assert exc.value.code == GCODE_VALIDATION_FAILED
    assert any(f["check"] == RPM_VALID for f in exc.value.context["blocking_failures"])


def test_failed_operation_blocks_the_whole_program():
    """One bad operation stops the entire program, not just that toolpath."""
    good = _toolpath()
    good.operation_id = "op_good"
    bad = _toolpath(feed=-1.0)
    bad.operation_id = "op_bad"

    with pytest.raises(CamError) as exc:
        get_post_processor("grbl", work_offset="G54").post_process([good, bad])
    assert exc.value.code == GCODE_VALIDATION_FAILED
    assert any(f["operation_id"] == "op_bad"
               for f in exc.value.context["blocking_failures"])


def test_valid_operations_still_produce_gcode():
    tp = _toolpath()
    tp.operation_id = "op_ok"
    result = get_post_processor("grbl", work_offset="G54").post_process(
        [tp], program_name="VALID")
    assert "G54" in result.gcode
    assert result.line_count > 5


def test_skipped_toolpaths_never_block():
    tp = _toolpath(feed=-1.0)
    tp.metadata["skipped"] = "deliberately not machined"
    result = get_post_processor("grbl", work_offset="G54").post_process([tp])
    assert "G54" in result.gcode


def test_gate_can_be_bypassed_explicitly():
    tp = _toolpath(feed=0.0)
    result = get_post_processor("grbl", work_offset="G54").post_process(
        [tp], require_validity=False)
    assert result.line_count > 0


def test_empty_job_is_not_a_failure():
    assert_valid_for_post([])


def test_generated_toolpath_stamp_is_trusted_by_the_gate():
    """The stamp written by StrategyEngine is what the gate reads."""
    tp = _generate(_op())
    stamp = tp.metadata["validity"]
    stamp["checks"][0]["passed"] = False          # corrupt the geometry check
    stamp["checks"][0]["critical"] = True
    stamp["checks"][0]["message"] = "geometry missing"

    with pytest.raises(CamError) as exc:
        get_post_processor("grbl", work_offset="G54").post_process([tp])
    assert exc.value.code == GCODE_VALIDATION_FAILED
    assert any(f["check"] == GEOMETRY_EXISTS
               for f in exc.value.context["blocking_failures"])


# ------------------------------------------------------------- HTTP surface --
@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


def _api_session_through_toolpaths(client):
    stock = {"margin_x": 5.0, "margin_y": 5.0, "margin_z_top": 1.0, "margin_z_bottom": 3.0}
    client.get("/api/model/demo")
    assert client.post("/api/recognize-features").status_code == 200
    assert client.post("/api/plan-operations", json={"stock": stock}).status_code == 200
    res = client.post("/api/generate-toolpaths", json={"stock": stock})
    assert res.status_code == 200
    return res


def test_api_reports_all_twelve_validity_checks_per_toolpath(api_client):
    items = _api_session_through_toolpaths(api_client).json()["toolpaths"]
    assert items
    for item in items:
        stamp = item["validity"]
        assert stamp is not None, item["operation_id"]
        assert [c["name"] for c in stamp["checks"]] == list(CHECK_ORDER)
        assert stamp["passed"] is True, stamp
        assert stamp["operation_id"] == item["operation_id"]


def test_api_blocks_gcode_when_an_operation_check_fails(api_client):
    from cam_engine.api.routes import _default_session

    _api_session_through_toolpaths(api_client)
    assert _default_session.toolpaths

    saved = []
    try:
        for tp in _default_session.toolpaths:
            for seg in tp.segments:
                saved.append((seg, seg.feed))
                if seg.is_cutting:
                    seg.feed = -5.0

        blocked = api_client.post(
            "/api/generate-gcode", json={"controller": "grbl", "work_offset": "G54"})
        assert blocked.status_code == 400
        body = blocked.json()
        assert body["code"] == GCODE_VALIDATION_FAILED
        assert body["stage"] == "gcode_gate"
        assert "gcode" not in body
        checks = {f["check"] for f in body["context"]["blocking_failures"]}
        assert FEED_VALID in checks
        assert body["detail"]
    finally:
        for seg, feed in saved:
            seg.feed = feed

    healthy = api_client.post(
        "/api/generate-gcode", json={"controller": "grbl", "work_offset": "G54"})
    assert healthy.status_code == 200
    assert healthy.json()["line_count"] > 10


def test_api_download_endpoint_also_returns_400_not_500(api_client):
    from cam_engine.api.routes import _default_session

    _api_session_through_toolpaths(api_client)
    setup_id = _default_session.setups[0].id

    saved = []
    try:
        for tp in _default_session.toolpaths:
            for seg in tp.segments:
                saved.append((seg, seg.spindle))
                if seg.is_cutting:
                    seg.spindle = 999_999.0
        res = api_client.get(f"/api/gcode/download/{setup_id}", params={"controller": "grbl"})
        assert res.status_code == 400
        body = res.json()
        assert body["code"] == GCODE_VALIDATION_FAILED
        assert {f["check"] for f in body["context"]["blocking_failures"]} == {RPM_VALID}
    finally:
        for seg, spindle in saved:
            seg.spindle = spindle
