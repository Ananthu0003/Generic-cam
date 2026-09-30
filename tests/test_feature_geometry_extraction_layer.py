"""Tests for Real Feature-Geometry Extraction Layer.

Verifies that for every machining-relevant B-Rep feature, the extraction layer
extracts and retains:
- type
- face_ids
- edge_ids
- surface_type
- axis
- center
- diameter / radius
- axial_start
- axial_end
- depth
- fillet_radius
- orientation
- tolerances
- source_geometry_reference

Ensures cylindrical faces carry actual CAD diameter, axis, and axial start/end
boundaries directly from CAD B-Rep topology without reconstruction from bounding boxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cam_engine.context import ToleranceModel, FinishRequirement
from cam_engine.coords import CoordSpace
from cam_engine.features import FeatureRecognizer, FeatureType, MachiningFeature
from cam_engine.geometry import Face, Shape, SurfaceData, SurfaceKind


def _create_mock_cylinder_face(
    index: int,
    radius: float,
    top_z: float,
    bottom_z: float,
    center_xy: tuple[float, float] = (0.0, 0.0),
    is_internal: bool = True,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    edge_indices: list[int] | None = None,
) -> Face:
    cx, cy = center_xy
    ax = np.array(axis, dtype=float)
    ax /= np.linalg.norm(ax)
    n_pts = 16
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    triangles = []
    for i in range(n_pts):
        a1 = angles[i]
        a2 = angles[(i + 1) % n_pts]
        p1 = np.array([cx + radius * np.cos(a1), cy + radius * np.sin(a1), top_z])
        p2 = np.array([cx + radius * np.cos(a2), cy + radius * np.sin(a2), top_z])
        p3 = np.array([cx + radius * np.cos(a1), cy + radius * np.sin(a1), bottom_z])
        p4 = np.array([cx + radius * np.cos(a2), cy + radius * np.sin(a2), bottom_z])
        # Two triangles per sector
        triangles.append([p1, p3, p2])
        triangles.append([p2, p3, p4])
    tri_arr = np.array(triangles, dtype=float)

    sd = SurfaceData(
        kind=SurfaceKind.CYLINDER,
        point_on=np.array([cx, cy, 0.0]),
        axis=ax,
        radius=radius,
        u_min=0.0,
        u_max=2 * np.pi,
        u_span=2 * np.pi,
    )

    oriented_norm = np.array([-np.cos(angles[0]), -np.sin(angles[0]), 0.0]) if is_internal else np.array([np.cos(angles[0]), np.sin(angles[0]), 0.0])

    return Face(
        index=index,
        surface=sd,
        oriented_normal=oriented_norm,
        triangles=tri_arr,
        area=float(2 * np.pi * radius * (top_z - bottom_z)),
        bounds_min=np.array([cx - radius, cy - radius, bottom_z]),
        bounds_max=np.array([cx + radius, cy + radius, top_z]),
        edge_indices=edge_indices or [index * 2, index * 2 + 1],
        is_internal=is_internal,
    )


class MockShape(Shape):
    def __init__(self, faces: list[Face], adjacency: dict[int, list[int]] | None = None):
        super().__init__(occ_shape=None, _faces=faces, _edges=None, _adjacency=adjacency or {})


def test_cylindrical_feature_extraction_layer_exact_cad_properties():
    """Verify cylindrical feature extraction extracts exact diameter, axis, and axial start/end."""
    dia = 6.000
    radius = dia / 2.0
    top_z = 25.0
    floor_z = 10.0
    depth = top_z - floor_z
    center_xy = (15.0, 20.0)

    face = _create_mock_cylinder_face(
        index=0,
        radius=radius,
        top_z=top_z,
        bottom_z=floor_z,
        center_xy=center_xy,
        is_internal=True,
        axis=(0.0, 0.0, 1.0),
        edge_indices=[101, 102],
    )

    shape = MockShape(faces=[face])
    reqs = {0: FinishRequirement(tolerance_mm=0.02)}

    recognizer = FeatureRecognizer(shape=shape, stock_top_z=25.0, requirements=reqs)
    features = recognizer.recognize()

    hole_feats = [f for f in features if f.type != FeatureType.CONTOUR]
    assert len(hole_feats) == 1
    feat = hole_feats[0]

    # Verify complete hierarchy and retained fields
    assert feat.type in (FeatureType.HOLE, FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE)
    assert feat.face_ids == [0]
    assert feat.edge_ids == [101, 102]
    assert feat.surface_type == "cylinder"
    np.testing.assert_allclose(feat.axis, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(feat.center, [15.0, 20.0, 17.5], atol=1e-6)
    assert feat.diameter == pytest.approx(6.000, abs=1e-4)
    assert feat.radius == pytest.approx(3.000, abs=1e-4)
    assert feat.diameter_or_radius == pytest.approx(6.000, abs=1e-4)
    assert feat.axial_start == pytest.approx(25.0, abs=1e-4)
    assert feat.axial_end == pytest.approx(10.0, abs=1e-4)
    assert feat.depth == pytest.approx(15.0, abs=1e-4)
    assert feat.tolerances is not None
    assert feat.tolerances.upper_tol_mm == pytest.approx(0.02)
    assert "Face#0:CYLINDER:Ø6.000" in feat.source_geometry_reference


def test_boss_feature_extraction_layer_exact_cad_properties():
    """Verify external boss feature extraction retains CAD diameter, axis, and start/end boundaries."""
    dia = 12.500
    radius = dia / 2.0
    top_z = 30.0
    floor_z = 15.0
    center_xy = (-5.0, 12.0)

    face = _create_mock_cylinder_face(
        index=0,
        radius=radius,
        top_z=top_z,
        bottom_z=floor_z,
        center_xy=center_xy,
        is_internal=False,
        axis=(0.0, 0.0, 1.0),
        edge_indices=[201, 202],
    )

    shape = MockShape(faces=[face])
    recognizer = FeatureRecognizer(shape=shape, stock_top_z=30.0)
    features = recognizer.recognize()

    boss_feats = [f for f in features if f.type == FeatureType.BOSS]
    assert len(boss_feats) == 1
    feat = boss_feats[0]

    assert feat.type == FeatureType.BOSS
    assert feat.face_ids == [0]
    assert feat.edge_ids == [201, 202]
    assert feat.surface_type == "cylinder"
    np.testing.assert_allclose(feat.axis, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(feat.center, [-5.0, 12.0, 22.5], atol=1e-6)
    assert feat.diameter == pytest.approx(12.500, abs=1e-4)
    assert feat.axial_start == pytest.approx(30.0, abs=1e-4)
    assert feat.axial_end == pytest.approx(15.0, abs=1e-4)
    assert feat.depth == pytest.approx(15.0, abs=1e-4)
    assert feat.source_geometry_reference == "Face#0:CYLINDER:Ø12.500"


def test_feature_transformation_preserves_cad_invariants():
    """Verify rigid transformation does not alter diameter, radius, depth, or geometry reference."""
    feat = MachiningFeature(
        id="hole_1",
        type=FeatureType.HOLE,
        face_indices=[1],
        bounds_min=np.array([10.0, 10.0, 0.0]),
        bounds_max=np.array([20.0, 20.0, 15.0]),
        depth=15.0,
        top_z=15.0,
        floor_z=0.0,
        axis=np.array([0.0, 0.0, 1.0]),
        diameter=10.0,
        radius=5.0,
        surface_type="cylinder",
        center=np.array([15.0, 15.0, 7.5]),
        axial_start=15.0,
        axial_end=0.0,
        source_edges=[10, 11],
        source_geometry_reference="Face#1:CYLINDER:Ø10.000",
        tolerances=ToleranceModel(upper_tol_mm=0.015, lower_tol_mm=-0.015),
    )

    # 90 degree rotation around X and translation of [50, 100, 200]
    t = np.array([
        [1.0,  0.0,  0.0,  50.0],
        [0.0,  0.0, -1.0, 100.0],
        [0.0,  1.0,  0.0, 200.0],
        [0.0,  0.0,  0.0,   1.0],
    ])

    xf_feat = feat.transformed(t, dst_space=CoordSpace.SETUP)

    # Invariants MUST remain identical
    assert xf_feat.diameter == 10.0
    assert xf_feat.radius == 5.0
    assert xf_feat.depth == 15.0
    assert xf_feat.surface_type == "cylinder"
    assert xf_feat.source_geometry_reference == "Face#1:CYLINDER:Ø10.000"
    assert xf_feat.face_ids == [1]
    assert xf_feat.edge_ids == [10, 11]
    assert xf_feat.tolerances.upper_tol_mm == 0.015

    # Transformed vectors & points
    np.testing.assert_allclose(xf_feat.axis, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(xf_feat.center, [65.0, 92.5, 215.0], atol=1e-6)
