import pytest
import numpy as np
import math

from cam_engine.geometry.topology import Shape
from cam_engine.features import FeatureRecognizer, FeatureType
from cam_engine.post.post_processor import BasePostProcessor, FanucPostProcessor
from cam_engine.toolpaths.semantic import MotionType, MotionSegment

def test_circular_face_wire_discretization():
    """Verify that a circular face wire is sampled with high resolution (>= 32 points)
    and not collapsed into a 3-point triangle."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    cyl = BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape()
    shape = Shape(cyl)
    faces = shape.faces()
    
    # Top and bottom faces are planar circles
    planar_faces = [f for f in faces if len(f.wires) > 0]
    for pf in planar_faces:
        for w in pf.wires:
            # Wire should have >= 32 points, not 4 (triangle)
            assert len(w.points) >= 32, f"Wire should have at least 32 points, got {len(w.points)}"
            # Verify points form a circle of radius 5.0
            radii = np.linalg.norm(w.points[:, :2], axis=1)
            assert np.allclose(radii, 5.0, atol=1e-3), "Wire points must lie on radius 5.0"


def test_blind_hole_bottom_not_classified_as_step_or_pocket():
    """Verify that a blind hole in a cylinder is not misclassified as a STEP or POCKET."""
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

    stock_cyl = BRepPrimAPI_MakeCylinder(10.0, 20.0).Shape()
    hole_ax = gp_Ax2(gp_Pnt(0, 0, 10.0), gp_Dir(0, 0, 1))
    hole_cyl = BRepPrimAPI_MakeCylinder(hole_ax, 3.0, 15.0).Shape()
    part = BRepAlgoAPI_Cut(stock_cyl, hole_cyl).Shape()

    shape = Shape(part)
    recognizer = FeatureRecognizer(shape, stock_top_z=20.0)
    features = recognizer.recognize()

    feature_types = [f.type for f in features]
    assert FeatureType.STEP not in feature_types, "Blind hole bottom must not be classified as STEP"
    assert FeatureType.POCKET not in feature_types, "Blind hole bottom must not be classified as POCKET"
    assert any(f.type in (FeatureType.BLIND_HOLE, FeatureType.BLIND_BORE, FeatureType.HOLE) for f in features)


def test_post_processor_arc_fitting_exact_circumcenter():
    """Verify that _fit_arc computes the true circumcenter in 2D without skew."""
    post = FanucPostProcessor()
    
    # Create points on a circle centered at (2.0, 3.0) with radius 4.0
    cx, cy, r = 2.0, 3.0, 4.0
    angles = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    pts = [np.array([cx + r * math.cos(a), cy + r * math.sin(a), 5.0]) for a in angles]

    fitted, center, ccw = post._fit_arc(pts)
    assert fitted is True
    assert center is not None
    assert np.allclose(center[:2], [cx, cy], atol=1e-4)
    assert bool(ccw) is True
