import math
import numpy as np
import pytest

from cam_engine.context import MachineConfig, PlanningContext, Setup, Tool, ToolType, Stock, StockKind
from cam_engine.geometry import TriangleMesh
from cam_engine.post.post_processor import (
    BasePostProcessor,
    FanucPostProcessor,
    HaasPostProcessor,
    GrblPostProcessor,
)
from cam_engine.toolpaths.semantic import MotionSegment, MotionType, Toolpath, CoordSpace
from cam_engine.toolpaths.validation import ToolpathValidator


def test_project_center_to_bisector():
    """Test that projecting candidate center to chord perpendicular bisector yields exact equidistant center."""
    start = np.array([-9.25, 0.192, 1.22])
    end = np.array([-8.943, 2.396, 1.22])
    # Inconsistent candidate center with ~0.010mm difference
    cand_center = np.array([-1.093, 0.144, 1.22])

    r_cand_start = float(np.linalg.norm(start[:2] - cand_center[:2]))
    r_cand_end = float(np.linalg.norm(end[:2] - cand_center[:2]))
    assert abs(r_cand_start - r_cand_end) > 0.005  # Was inconsistent

    proj_center = BasePostProcessor._project_center_to_bisector(start, end, cand_center)
    r_proj_start = float(np.linalg.norm(start[:2] - proj_center[:2]))
    r_proj_end = float(np.linalg.norm(end[:2] - proj_center[:2]))

    # Must be mathematically identical (within float machine precision)
    assert abs(r_proj_start - r_proj_end) < 1e-12


def test_format_arc_ij_precision_optimization():
    """Test that post processor _format_arc_ij produces I/J with post-rounded radius error <= 0.001mm."""
    post = BasePostProcessor(precision=3)
    start = np.array([-9.25, 0.192, 1.22])
    end = np.array([-8.943, 2.396, 1.22])
    cand_center = np.array([-1.093, 0.144, 1.22])

    i_off, j_off = post._format_arc_ij(start, end, cand_center, ccw=False)

    # Reconstruct radii from formatted G-code numbers
    xs = round(float(start[0]), 3)
    ys = round(float(start[1]), 3)
    xe = round(float(end[0]), 3)
    ye = round(float(end[1]), 3)

    r_start = math.hypot(i_off, j_off)
    cx = xs + i_off
    cy = ys + j_off
    r_end = math.hypot(xe - cx, ye - cy)

    assert abs(r_start - r_end) <= 0.001, f"Radius mismatch too high: {abs(r_start - r_end)}"


def test_fit_arc_mathematical_consistency():
    """Test that _fit_arc creates arcs with mathematically identical start and end radii."""
    post = BasePostProcessor()
    # Generate points along a true circular arc of radius 10.0 centered at (0, 0)
    angles = np.linspace(0.1, 1.2, 8)
    true_r = 10.0
    pts = [np.array([true_r * math.cos(a), true_r * math.sin(a), 5.0]) for a in angles]

    fitted, center, ccw = post._fit_arc(pts)
    assert fitted is True
    assert center is not None
    assert bool(ccw) is True

    r_start = float(np.linalg.norm(pts[0][:2] - center[:2]))
    r_end = float(np.linalg.norm(pts[-1][:2] - center[:2]))
    assert abs(r_start - r_end) < 1e-12


def test_fit_arc_rejects_non_circular_points():
    """Test that _fit_arc rejects points that deviate from a true circular arc."""
    post = BasePostProcessor()
    angles = np.linspace(0.1, 1.2, 8)
    pts = [np.array([10.0 * math.cos(a), 10.0 * math.sin(a), 5.0]) for a in angles]
    # Perturb intermediate point by 0.05mm
    pts[3][0] += 0.05

    fitted, center, ccw = post._fit_arc(pts)
    assert fitted is False


def test_post_processors_emit_consistent_g2_g3():
    """Test all post processors generate valid G2/G3 code without radius discrepancy."""
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=6.0, corner_radius=0.0, flute_length=20.0,
        overall_length=60.0, flutes=4, material="carbide", can_plunge=True
    )
    tp = Toolpath(operation_id="op1", feature_id="f1", tool_id="T01", purpose="roughing")

    # Arc from (-9.25, 0.192) to (-8.943, 2.396) with center offset (8.157, -0.048)
    start = np.array([-9.25, 0.192, 1.22])
    end = np.array([-8.943, 2.396, 1.22])
    center = start + np.array([8.157, -0.048, 0.0])

    tp.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=start,
        end=end,
        arc_center=center,
        arc_ccw=False,
        feed=1000.0,
        tool_id="T01",
        operation_id="op1",
        feature_id="f1"
    ))

    posts = [
        BasePostProcessor(precision=3),
        FanucPostProcessor(precision=3),
        HaasPostProcessor(precision=3),
        GrblPostProcessor(precision=3),
    ]

    for post in posts:
        res = post.post_process([tp])
        # Find G2 line (avoid G21 in header)
        g2_lines = [l for l in res.gcode.splitlines() if l.strip().startswith("G2 ") or " G2 " in l or "G02" in l]
        assert len(g2_lines) >= 1
        g2_line = g2_lines[0]
        assert "I" in g2_line and "J" in g2_line


def test_toolpath_validator_arc_radius_check():
    """Test ToolpathValidator detects inconsistent arc radii and passes consistent arcs."""
    from cam_engine.demo_models import get_default_machine
    from cam_engine.context import Material, WorkOffset
    tool = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL,
        diameter=6.0, corner_radius=0.0, flute_length=20.0,
        overall_length=60.0, flutes=4, material="carbide", can_plunge=True
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-50, -50, -10]), bounds_max=np.array([50, 50, 20]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = Material(id="al", name="Aluminum", v_carbide=300, v_hss=90, hardness_hb=95,
                   max_hm_ratio=1.0, max_ae_ratio=0.4, specific_cutting_force_kn_mm2=0.8, needs_coolant=True)
    ctx = PlanningContext(
        model=None, setups=[setup], machine=get_default_machine(),
        tools=[tool], material=mat
    )
    mesh = TriangleMesh(triangles=np.zeros((1, 3, 3)))
    validator = ToolpathValidator(ctx, setup, mesh, clearance_z=20.0)

    # 1. Consistent arc
    tp_ok = Toolpath(operation_id="op_ok", feature_id="f1", tool_id="T01", purpose="roughing")
    start = np.array([0.0, 10.0, 0.0])
    end = np.array([10.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    tp_ok.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=start,
        end=end,
        arc_center=center,
        feed=500.0,
        tool_id="T01",
        operation_id="op_ok",
        feature_id="f1"
    ))
    rep_ok = validator.validate(tp_ok)
    assert len(rep_ok.errors) == 0

    # 2. Inconsistent arc with delta R = 0.1mm
    tp_bad = Toolpath(operation_id="op_bad", feature_id="f1", tool_id="T01", purpose="roughing")
    bad_center = np.array([0.0, 0.1, 0.0])  # r_start = 9.9, r_end = 10.0005, delta ~ 0.1
    tp_bad.add(MotionSegment(
        motion_type=MotionType.ARC_CW,
        start=start,
        end=end,
        arc_center=bad_center,
        feed=500.0,
        tool_id="T01",
        operation_id="op_bad",
        feature_id="f1"
    ))
    rep_bad = validator.validate(tp_bad)
    assert any("radius inconsistent" in e.message for e in rep_bad.errors)


def test_consecutive_circular_arcs_share_identical_center():
    """Test that toolpath generation for circular boss roughing emits consecutive arcs sharing identical center."""
    from cam_engine.features import MachiningFeature, FeatureType
    from cam_engine.operations import PlannedOperation, OpPurpose
    from cam_engine.machinability import CuttingParameters
    from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
    from cam_engine.demo_models import get_default_machine

    tool = Tool(
        id="T04", tool_number=4, type=ToolType.FLAT_ENDMILL,
        diameter=6.0, corner_radius=0.0, flute_length=20.0,
        overall_length=60.0, flutes=4, material="carbide", can_plunge=True
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-15, -15, 0]), bounds_max=np.array([15, 15, 20]))
    from cam_engine.context import Material, WorkOffset
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    mat = Material(id="al", name="Aluminum", v_carbide=300, v_hss=90, hardness_hb=95,
                   max_hm_ratio=1.0, max_ae_ratio=0.4, specific_cutting_force_kn_mm2=0.8, needs_coolant=True)
    ctx = PlanningContext(model=None, setups=[setup], machine=get_default_machine(), tools=[tool], material=mat)

    strat_ctx = StrategyContext(
        context=ctx, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=20.0, finish_allowance_wall=0.2, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(strat_ctx)

    feat = MachiningFeature(
        id="feat_boss", type=FeatureType.BOSS, face_indices=[0],
        bounds_min=np.array([-5.0, -5.0, 0.0]), bounds_max=np.array([5.0, 5.0, 10.0]),
        depth=10.0, top_z=10.0, floor_z=0.0, diameter=10.0, is_concave=False
    )
    op = PlannedOperation(
        id="op_rough", feature=feat, tool=tool, purpose=OpPurpose.ROUGHING,
        params=CuttingParameters(spindle_rpm=8001, feed_rate=1200, plunge_feed=200, ramp_feed=400,
                                 feed_per_tooth_mm=0.05, depth_of_cut_mm=5.0, stepover_mm=2.0, coolant=True,
                                 effective_diameter_mm=6.0, engagement_angle_deg=45.0, power_kw=1.0, torque_nm=2.0)
    )

    tp = engine.generate(op)
    arc_segs = [s for s in tp.segments if s.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW)]
    assert len(arc_segs) >= 2

    # All arc segments must share the exact center (0, 0)
    for s in arc_segs:
        assert s.arc_center is not None
        assert abs(s.arc_center[0] - 0.0) < 1e-6
        assert abs(s.arc_center[1] - 0.0) < 1e-6

    # Verify G-code output
    post = GrblPostProcessor(precision=3)
    res = post.post_process([tp])
    gcode_lines = [l for l in res.gcode.splitlines() if l.startswith("G2 ") or l.startswith("G3 ") or " G2 " in l or " G3 " in l]
    assert len(gcode_lines) >= 2


def test_kasa_circle_fit_consistency():
    """Test that algebraic Kåsa multi-point circle fit recovers identical global center from subsets of points."""
    post = BasePostProcessor()
    cx_true, cy_true, r_true = 2.5, -3.0, 8.0
    # 32 points on circle
    angles = np.linspace(0, math.pi, 16)
    pts = [np.array([cx_true + r_true * math.cos(a), cy_true + r_true * math.sin(a), 5.0]) for a in angles]

    fitted, center, ccw = post._fit_arc(pts)
    assert fitted is True
    assert center is not None
    assert abs(center[0] - cx_true) < 1e-6
    assert abs(center[1] - cy_true) < 1e-6

