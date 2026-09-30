import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from app import app
from cam_engine.geometry.topology import Shape
from cam_engine.geometry.step_import import import_step
from cam_engine.features import FeatureRecognizer, FeatureType
from cam_engine.pipeline import run_cam_pipeline
from cam_engine.toolpaths.semantic import MotionType


def test_outer_contour_finishing_has_no_helical_spiral_ramp():
    """Verify that outer contour finishing on the demo model does not generate
    a spurious descending helical spiral at the corner."""
    client = TestClient(app)
    client.get("/api/model/demo")
    client.post("/api/recognize-features")
    client.post("/api/plan-operations", json={
        "stock": {"mode": "bounding_box", "margin_x": 5.0, "margin_y": 5.0, "margin_z_top": 1.0, "margin_z_bottom": 5.0}
    })
    res = client.post("/api/generate-toolpaths", json={
        "stock": {"mode": "bounding_box", "margin_x": 5.0, "margin_y": 5.0, "margin_z_top": 1.0, "margin_z_bottom": 5.0}
    })
    assert res.status_code == 200
    data = res.json()
    
    finish_ops = [tp for tp in data["toolpaths"] if tp.get("purpose") == "finishing"]
    assert len(finish_ops) > 0

    outer_finish = next((tp for tp in finish_ops if "contour_outer" in tp.get("feature_id", "") or tp.get("operation_id") == "op_finish_001"), None)
    assert outer_finish is not None

    helix_segments = [s for s in outer_finish["segments"] if s["motion_type"] == "helix"]
    # There should be 0 helical ramp plunges on the outer contour
    assert len(helix_segments) == 0, f"Expected 0 helix segments in outer contour finish, found {len(helix_segments)}"


def test_corner_fillet_is_not_classified_as_hole():
    """Verify that a part with a corner fillet does NOT classify the fillet as a hole/bore
    and does not generate any drilling or boring operations for it."""
    from OCP.gp import gp_Pnt
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    box = BRepPrimAPI_MakeBox(100.0, 60.0, 25.0).Shape()
    pocket = BRepPrimAPI_MakeBox(gp_Pnt(20.0, 20.0, 10.0), gp_Pnt(80.0, 40.0, 30.0)).Shape()
    cut = BRepAlgoAPI_Cut(box, pocket).Shape()

    # Fillet the inner vertical edge at (20, 20) with radius 5.0mm
    fillet_builder = BRepFilletAPI_MakeFillet(cut)
    exp = TopExp_Explorer(cut, TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        c = BRepAdaptor_Curve(e)
        p0 = c.Value(c.FirstParameter())
        p1 = c.Value(c.LastParameter())
        if abs(p0.X() - 20) < 1 and abs(p0.Y() - 20) < 1 and abs(p0.X() - p1.X()) < 1e-3 and abs(p0.Y() - p1.Y()) < 1e-3:
            fillet_builder.Add(5.0, e)
        exp.Next()

    fillet_builder.Build()
    shape_with_fillet = Shape(fillet_builder.Shape())
    rec = FeatureRecognizer(shape_with_fillet, stock_top_z=25.0)
    feats = rec.recognize()

    # Should have NO holes/bores (since there are no drilled holes in this geometry)
    hole_feats = [f for f in feats if f.type in (
        FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
        FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE
    )]
    assert len(hole_feats) == 0, f"Expected 0 hole features on filleted pocket, got: {[f.id for f in hole_feats]}"

    # The fillet should be recognized as a FILLET or part of pocket/wall
    fillet_feats = [f for f in feats if f.type == FeatureType.FILLET]
    assert len(fillet_feats) >= 1


def test_full_hole_is_still_correctly_classified_and_drilled():
    """Verify that full 360-degree through holes and blind holes continue to be recognized
    as holes and drilled properly."""
    bracket_path = Path("sample_bracket.step")
    if bracket_path.exists():
        res = run_cam_pipeline(bracket_path, controller="grbl", work_offset="G54")
        hole_ops = [op for op in res.operations if op.purpose.value in ("drilling", "boring")]
        assert len(hole_ops) >= 1


def test_fillet_toolpath_generation_without_major_radius_error():
    """Verify that fillet features generated from B-Rep or prismatic geometry
    can have toolpaths generated cleanly without raising major radius unavailable error."""
    from cam_engine.features import MachiningFeature
    from cam_engine.operations import PlannedOperation, OpPurpose
    from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
    from cam_engine.context import Tool, ToolType, Setup
    from cam_engine.machinability import CuttingParameters

    feat = MachiningFeature(
        id="setup_001_fillet_51",
        type=FeatureType.FILLET,
        face_indices=[51],
        depth=5.0,
        top_z=10.0,
        floor_z=5.0,
        radius=3.0,
        fillet_radius=3.0,
        surface_type="cylinder",
        bounds_min=np.array([10.0, 10.0, 5.0]),
        bounds_max=np.array([20.0, 20.0, 10.0]),
    )
    ball_tool = Tool(
        id="T01",
        tool_number=1,
        type=ToolType.BALL_ENDMILL,
        diameter=6.0,
        flute_length=20.0,
        overall_length=50.0,
        corner_radius=3.0,
        tip_radius=3.0,
    )
    params = CuttingParameters(
        spindle_rpm=8000.0,
        feed_rate=1200.0,
        plunge_feed=400.0,
        ramp_feed=600.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=1.0,
        stepover_mm=1.5,
        coolant=True,
        effective_diameter_mm=6.0,
        engagement_angle_deg=90.0,
        power_kw=1.0,
        torque_nm=1.0,
    )
    op = PlannedOperation(
        id="op_fillet_001",
        purpose=OpPurpose.FILLETING,
        feature=feat,
        tool=ball_tool,
        params=params,
        notes={"fillet_radius": 3.0, "surface_type": "cylinder"}
    )
    setup = Setup(id="setup_001", name="Top Setup")
    ctx = StrategyContext(
        context=None, setup=setup, mesh=None, stock_voxels=None,
        clearance_z=25.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0
    )
    engine = StrategyEngine(ctx)
    tp = engine.generate(op)
    assert tp is not None
    assert len(tp.segments) > 0


