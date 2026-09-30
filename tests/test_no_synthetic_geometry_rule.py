"""Tests for the architectural 'No Synthetic Geometry' Rule.

The engine must never create final geometry from:
- bounding box
- silhouette
- tool diameter
- DOC / AE
- number of passes
- default radius / default depth
- hardcoded coordinate
- previous operation depth
- stock dimensions

unless that value is explicitly being used as stock/roughing geometry.

If CAD topology extraction fails, the pipeline fails with FEATURE_EXTRACTION_FAILED / SYNTHETIC_GEOMETRY_FORBIDDEN.
"""

import numpy as np
import pytest

from cam_engine.context import (
    PlanningContext, Setup, Stock, StockKind, Tool, ToolType, WorkOffset, ToolHolder, get_material
)
from cam_engine.errors import (
    CamError, SYNTHETIC_GEOMETRY_FORBIDDEN, INVALID_GEOMETRY
)
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.geometry.topology import Shape, Face, SurfaceData, SurfaceKind
from cam_engine.toolpaths.validation import PreToolpathGeometryValidator


def create_mock_shape_with_faces(faces: list[Face]) -> Shape:
    shape = Shape(None)
    shape._faces = faces
    return shape


def test_synthetic_cylinder_from_planar_box_rejected():
    """A feature claiming to be a cylindrical hole Ø10.0mm on a planar face (inferred from bounding box) must be rejected."""
    # CAD Face is planar, NOT a cylinder
    planar_face = Face(
        index=0,
        surface=SurfaceData(kind=SurfaceKind.PLANE, normal=np.array([0, 0, 1]), point_on=np.array([0, 0, 10])),
        oriented_normal=np.array([0, 0, 1]),
        triangles=np.zeros((2, 3, 3)),
        area=100.0,
        bounds_min=np.array([-5.0, -5.0, 10.0]),
        bounds_max=np.array([5.0, 5.0, 10.0]),
    )
    shape = create_mock_shape_with_faces([planar_face])

    # Synthetic feature guessing hole Ø10.0mm from bounding box span (-5 to +5)
    feat_synthetic = MachiningFeature(
        id="feat_synthetic_hole",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],  # points to planar face
        bounds_min=np.array([-5.0, -5.0, 0.0]),
        bounds_max=np.array([5.0, 5.0, 10.0]),
        depth=10.0,
        top_z=10.0,
        floor_z=0.0,
        axis=np.array([0.0, 0.0, 1.0]),
        diameter=10.0,  # guessed from box width
    )

    validator = PreToolpathGeometryValidator(shape=shape)
    report = validator.validate_feature(feat_synthetic)
    assert not report.passed
    assert any(e.code == SYNTHETIC_GEOMETRY_FORBIDDEN and "synthetic" in e.message.lower() for e in report.errors)


def test_synthetic_divergent_diameter_rejected():
    """A feature whose diameter diverges from actual CAD cylinder surface radius is rejected."""
    # Real CAD cylinder has R = 3.00 mm (Ø6.00 mm)
    cyl_face = Face(
        index=0,
        surface=SurfaceData(kind=SurfaceKind.CYLINDER, radius=3.00, axis=np.array([0, 0, 1]), point_on=np.array([0, 0, 0])),
        oriented_normal=np.array([-1, 0, 0]),
        triangles=np.zeros((4, 3, 3)),
        area=50.0,
        bounds_min=np.array([-3.0, -3.0, 0.0]),
        bounds_max=np.array([3.0, 3.0, 10.0]),
    )
    shape = create_mock_shape_with_faces([cyl_face])

    # Feature incorrectly synthesized/overridden to Ø8.00 mm (e.g. from tool diameter)
    feat_divergent = MachiningFeature(
        id="feat_divergent_dia",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        bounds_min=np.array([-3.0, -3.0, 0.0]),
        bounds_max=np.array([3.0, 3.0, 10.0]),
        depth=10.0,
        top_z=10.0,
        floor_z=0.0,
        axis=np.array([0.0, 0.0, 1.0]),
        diameter=8.0,  # Mismatch: 8.0 != 2 * 3.0
    )

    validator = PreToolpathGeometryValidator(shape=shape)
    report = validator.validate_feature(feat_divergent)
    assert not report.passed
    assert any(e.code == SYNTHETIC_GEOMETRY_FORBIDDEN for e in report.errors)


def test_true_cad_cylinder_passes():
    """A feature directly matching CAD cylinder surface radius passes cleanly."""
    cyl_face = Face(
        index=0,
        surface=SurfaceData(kind=SurfaceKind.CYLINDER, radius=4.00, axis=np.array([0, 0, 1]), point_on=np.array([0, 0, 0])),
        oriented_normal=np.array([-1, 0, 0]),
        triangles=np.zeros((4, 3, 3)),
        area=60.0,
        bounds_min=np.array([-4.0, -4.0, 0.0]),
        bounds_max=np.array([4.0, 4.0, 15.0]),
    )
    shape = create_mock_shape_with_faces([cyl_face])

    feat_exact = MachiningFeature(
        id="feat_exact_cad_hole",
        type=FeatureType.THROUGH_HOLE,
        face_indices=[0],
        bounds_min=np.array([-4.0, -4.0, 0.0]),
        bounds_max=np.array([4.0, 4.0, 15.0]),
        depth=15.0,
        top_z=15.0,
        floor_z=0.0,
        axis=np.array([0.0, 0.0, 1.0]),
        diameter=8.0,  # Matches 2 * 4.00
    )

    validator = PreToolpathGeometryValidator(shape=shape)
    report = validator.validate_feature(feat_exact)
    assert report.passed
    assert len(report.errors) == 0


def test_empty_provenance_rejected():
    """A feature with empty face indices (purely synthetic) is rejected."""
    feat_no_faces = MachiningFeature(
        id="feat_ghost",
        type=FeatureType.POCKET,
        face_indices=[],  # No CAD face
        bounds_min=np.array([0, 0, 0]),
        bounds_max=np.array([10, 10, 5]),
        depth=5.0,
        top_z=5.0,
        floor_z=0.0,
    )

    validator = PreToolpathGeometryValidator()
    report = validator.validate_feature(feat_no_faces)
    assert not report.passed
    assert any(e.code == INVALID_GEOMETRY and "empty" in e.message.lower() for e in report.errors)
