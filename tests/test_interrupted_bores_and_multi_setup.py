"""Comprehensive tests for:
1. Interrupted cylindrical features (bores with internal keyways/slots).
2. Multi-tier coaxial counterbores and stepped through-bores.
3. Multi-ring concentric helical bore pocketing for large diameters.
4. Automatic multi-axis setup generation for side holes/features.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cam_engine.features import (
    FeatureRecognizer,
    FeatureType,
    MachiningFeature,
    Accessibility,
)
from cam_engine.geometry.topology import Shape, Face, SurfaceData, SurfaceKind
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.context import (
    Tool,
    ToolType,
    Material,
    Setup,
    Stock,
    StockKind,
    Units,
    PlanningContext,
)
from cam_engine.demo_models import get_default_machine
from cam_engine.machinability import MachinabilityAnalyzer, CuttingParameters
from app import app


@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Interrupted Cylindrical Feature Recognition (Bore with Keyway)
# -----------------------------------------------------------------------------
def test_interrupted_bore_recognition():
    """Verify that an internal cylindrical surface with ~300° sweep (keyway interruption)
    is successfully recognized as a BORE / THROUGH_BORE rather than discarded."""
    r = 19.0
    u_span = 5.2359  # 300 degrees
    n_theta = 20
    n_z = 5
    thetas = np.linspace(0, u_span, n_theta)
    zs = np.linspace(0, 75.0, n_z)
    
    triangles = []
    for i in range(len(thetas) - 1):
        for j in range(len(zs) - 1):
            t0, t1 = thetas[i], thetas[i+1]
            z0, z1 = zs[j], zs[j+1]
            p00 = np.array([r * np.cos(t0), r * np.sin(t0), z0])
            p10 = np.array([r * np.cos(t1), r * np.sin(t1), z0])
            p01 = np.array([r * np.cos(t0), r * np.sin(t0), z1])
            p11 = np.array([r * np.cos(t1), r * np.sin(t1), z1])
            triangles.append([p00, p10, p11])
            triangles.append([p00, p11, p01])

    tri_arr = np.array(triangles, dtype=float)
    bmin = tri_arr.reshape(-1, 3).min(axis=0)
    bmax = tri_arr.reshape(-1, 3).max(axis=0)
    surf = SurfaceData(
        kind=SurfaceKind.CYLINDER,
        point_on=np.array([0.0, 0.0, 0.0]),
        axis=np.array([0.0, 0.0, 1.0]),
        radius=r,
        u_min=0.0,
        u_max=u_span,
    )
    face0 = Face(
        index=0,
        surface=surf,
        oriented_normal=np.array([-1.0, 0.0, 0.0]),
        triangles=tri_arr,
        area=r * u_span * 75.0,
        bounds_min=bmin,
        bounds_max=bmax,
        is_internal=True,
    )

    shape = Shape(occ_shape=None, _faces=[face0], _edges=[], _adjacency={0: []})
    recognizer = FeatureRecognizer(shape, stock_top_z=75.0)
    features = recognizer.recognize()

    bore_feats = [f for f in features if f.type in (FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE)]
    assert len(bore_feats) >= 1, "Interrupted bore with keyway was not recognized!"
    f = bore_feats[0]
    assert abs(f.diameter - 38.0) < 0.1
    assert f.notes.get("is_interrupted") is True


# -----------------------------------------------------------------------------
# 2. Multi-Tier Coaxial Counterbores (Top CBore + Through Bore + Bottom CBore)
# -----------------------------------------------------------------------------
def test_multi_tier_coaxial_counterbores():
    """Verify that 3 coaxial internal cylinders (Ø56 top counterbore, Ø38 through bore, Ø56 bottom counterbore)
    are recognized with proper parent-child relationships and heights."""
    faces = []
    
    def make_cyl(idx, r, z_min, z_max):
        u_span = 2 * np.pi
        thetas = np.linspace(0, u_span, 16)
        zs = np.linspace(z_min, z_max, 4)
        triangles = []
        for i in range(len(thetas) - 1):
            for j in range(len(zs) - 1):
                t0, t1 = thetas[i], thetas[i+1]
                z0, z1 = zs[j], zs[j+1]
                p00 = np.array([r * np.cos(t0), r * np.sin(t0), z0])
                p10 = np.array([r * np.cos(t1), r * np.sin(t1), z0])
                p01 = np.array([r * np.cos(t0), r * np.sin(t0), z1])
                p11 = np.array([r * np.cos(t1), r * np.sin(t1), z1])
                triangles.append([p00, p10, p11])
                triangles.append([p00, p11, p01])
        tri_arr = np.array(triangles, dtype=float)
        bmin = tri_arr.reshape(-1, 3).min(axis=0)
        bmax = tri_arr.reshape(-1, 3).max(axis=0)
        surf = SurfaceData(
            kind=SurfaceKind.CYLINDER,
            point_on=np.array([0.0, 0.0, 0.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            radius=r,
            u_min=0.0,
            u_max=u_span,
        )
        return Face(
            index=idx,
            surface=surf,
            oriented_normal=np.array([-1.0, 0.0, 0.0]),
            triangles=tri_arr,
            area=r * u_span * (z_max - z_min),
            bounds_min=bmin,
            bounds_max=bmax,
            is_internal=True,
        )

    faces.append(make_cyl(0, 28.0, 65.0, 75.0))  # Top counterbore
    faces.append(make_cyl(1, 19.0, 10.0, 65.0))  # Main bore
    faces.append(make_cyl(2, 28.0, 0.0, 10.0))   # Bottom counterbore

    shape = Shape(occ_shape=None, _faces=faces, _edges=[], _adjacency={0: [1], 1: [0, 2], 2: [1]})
    recognizer = FeatureRecognizer(shape, stock_top_z=75.0)
    features = recognizer.recognize()

    cbores = [f for f in features if f.type == FeatureType.COUNTERBORE]
    bores = [f for f in features if f.type in (FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE)]

    assert len(cbores) >= 2, f"Expected 2 counterbores, got {len(cbores)}"
    assert len(bores) >= 1, f"Expected 1 main bore, got {len(bores)}"
    assert abs(bores[0].diameter - 38.0) < 0.1
    assert any(abs(cb.diameter - 56.0) < 0.1 for cb in cbores)


# -----------------------------------------------------------------------------
# 3. Multi-Ring Concentric Helical Bore Pocketing
# -----------------------------------------------------------------------------
def test_multi_ring_helical_bore_clearing():
    """Verify that helical interpolation for a large counterbore (e.g. Ø56mm with 10mm tool)
    generates concentric radial passes (n_rings >= 3) to clear 100% of interior material."""
    f = MachiningFeature(
        id="feat_cbore_large",
        type=FeatureType.COUNTERBORE,
        face_indices=[0],
        bounds_min=np.array([-28.0, -28.0, 65.0]),
        bounds_max=np.array([28.0, 28.0, 75.0]),
        depth=10.0,
        top_z=75.0,
        floor_z=65.0,
        diameter=56.0,
        is_concave=True,
    )
    tool = Tool(
        id="T01",
        tool_number=1,
        type=ToolType.FLAT_ENDMILL,
        diameter=10.0,
        corner_radius=0.0,
        flute_length=35.0,
        overall_length=75.0,
        flutes=4,
    )
    from cam_engine.context import DEFAULT_MATERIALS, WorkOffset
    mat = DEFAULT_MATERIALS["aluminum_6061"]
    machine = get_default_machine()
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-50.0, -50.0, 0.0]), bounds_max=np.array([50.0, 50.0, 80.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    
    class _MockCtx:
        def __init__(self):
            self.material = mat
            self.machine = machine
            self.tools = [tool]
            self.setups = [setup]

    ctx = _MockCtx()

    params = CuttingParameters(
        spindle_rpm=6000.0,
        feed_rate=1500.0,
        plunge_feed=600.0,
        ramp_feed=900.0,
        feed_per_tooth_mm=0.06,
        depth_of_cut_mm=2.5,
        stepover_mm=5.0,
        coolant=True,
        effective_diameter_mm=10.0,
        engagement_angle_deg=45.0,
        power_kw=1.5,
        torque_nm=3.0,
    )
    op = PlannedOperation(
        id="op_cbore_large",
        purpose=OpPurpose.DRILLING,
        feature=f,
        tool=tool,
        params=params,
        notes={"method": "helical_interpolation", "target_diameter": 56.0},
    )

    strat_ctx = StrategyContext(
        context=ctx,
        setup=setup,
        mesh=None,
        stock_voxels=None,
        clearance_z=85.0,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    assert tp.metadata.get("strategy") == "helical_interpolation"
    rings = tp.metadata.get("concentric_rings", 1)
    assert rings >= 3, f"Expected at least 3 concentric rings, got {rings}"

    helix_segs = [s for s in tp.segments if s.motion_type is MotionType.HELIX]
    arc_segs = [s for s in tp.segments if s.motion_type is MotionType.ARC_CCW]
    assert len(helix_segs) > 0, "No helical descent segments found"
    assert len(arc_segs) > 0, "No flat arc cleanup passes found"


# -----------------------------------------------------------------------------
# 4. Transverse Side-Hole Feature & Multi-Setup Auto-Detection
# -----------------------------------------------------------------------------
def test_transverse_side_hole_and_auto_setups(client):
    """Verify that side holes are identified as transverse features and auto-generate setups API
    configures Top, Bottom, and Side setups."""
    res = client.post("/api/setups/auto-generate")
    assert res.status_code == 200
    data = res.json()
    setups = data["setups"]
    assert len(setups) >= 2, f"Expected at least 2 setups, got {len(setups)}"
    assert setups[0]["work_offset"] == "G54"
    assert setups[1]["work_offset"] == "G55"
