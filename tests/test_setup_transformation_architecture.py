"""Tests for the complete Setup Transformation Architecture.

Verifies the foundational CAM architecture:
Canonical CAD coordinates
          ↓
   Setup transform (model_to_setup)
          ↓
Machine/setup coordinates (G54 / G55 / etc.)

Validates that every setup has:
- setup_id
- WCS
- origin
- rotation
- translation
- orientation
- fixture transform
- model_to_setup_transform

Guarantees that physical surfaces retain their canonical identity and tolerances
across multiple setups (G54, G55, etc.) without independently regenerating geometry.
"""

import numpy as np
import pytest

from cam_engine.coords import (
    CoordSpace,
    make_transform,
    rotation_about_axis,
    validate_rigid,
    transform_points,
    transform_direction,
)
from cam_engine.context import (
    Setup,
    Stock,
    StockKind,
    WorkOffset,
    Fixture,
)
from cam_engine.features import (
    MachiningFeature,
    FeatureType,
    Accessibility,
)
from cam_engine.geometry import (
    CadSurfaceKind,
    NominalCadSurface,
    DesignGeometry,
)


def test_setup_attributes_and_transformation_pipeline():
    """Verify setup_id, WCS, origin, rotation, translation, orientation,
    fixture_transform, model_to_setup_transform, and coordinate transformation pipeline."""
    # Setup 1 (G54): Translated +100X, +50Y, +25Z
    rot1 = np.eye(3)
    trans1 = np.array([100.0, 50.0, 25.0])
    M1 = make_transform(rotation=rot1, translation=trans1)

    stock1 = Stock(kind=StockKind.BOX, bounds_min=np.array([90.0, 40.0, 0.0]), bounds_max=np.array([210.0, 160.0, 30.0]))
    setup1 = Setup(
        setup_id="setup_op10",
        name="OP10 - Top Side",
        WCS="G54",
        model_to_setup_transform=M1,
        stock=stock1,
    )

    # 1. Attribute access
    assert setup1.setup_id == "setup_op10"
    assert setup1.id == "setup_op10"
    assert setup1.WCS == "G54"
    assert setup1.wcs == "G54"
    assert setup1.work_offset == WorkOffset.G54
    np.testing.assert_allclose(setup1.origin, trans1)
    np.testing.assert_allclose(setup1.translation, trans1)
    np.testing.assert_allclose(setup1.rotation, rot1)
    np.testing.assert_allclose(setup1.model_to_setup_transform, M1)
    np.testing.assert_allclose(setup1.setup_transform, M1)
    np.testing.assert_allclose(setup1.orientation, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(setup1.fixture_transform, np.eye(4))

    # 2. Setup 2 (G55): Flipped 180° around X axis (bottom machining)
    rot_flip_x = rotation_about_axis(np.array([1.0, 0.0, 0.0]), np.pi)
    trans2 = np.array([0.0, 200.0, 50.0])
    M2 = make_transform(rotation=rot_flip_x, translation=trans2)

    fix_mat = make_transform(translation=np.array([0.0, 0.0, -10.0]))
    setup2 = Setup(
        setup_id="setup_op20",
        name="OP20 - Bottom Side",
        WCS="G55",
        model_to_setup=M2,
        fixture_transform=fix_mat,
        stock=stock1,
    )

    assert setup2.setup_id == "setup_op20"
    assert setup2.WCS == "G55"
    np.testing.assert_allclose(setup2.rotation, rot_flip_x, atol=1e-6)
    np.testing.assert_allclose(setup2.translation, trans2, atol=1e-6)
    np.testing.assert_allclose(setup2.fixture_transform, fix_mat, atol=1e-6)
    np.testing.assert_allclose(setup2.canonical_orientation, np.array([0.0, 0.0, -1.0]), atol=1e-6)


def test_single_canonical_feature_transformed_across_setups():
    """Verify that the SAME canonical feature is transformed to G54 and G55 setups
    without regenerating geometry or losing physical surface identity."""
    # Canonical CAD feature: Blind Bore Ø20.0mm, Depth 15.0mm, Face index 7
    canonical_feat = MachiningFeature(
        id="bore_canonical_007",
        canonical_id="bore_canonical_007",
        type=FeatureType.BLIND_BORE,
        face_indices=[7],
        bounds_min=np.array([10.0, 15.0, 5.0]),
        bounds_max=np.array([30.0, 35.0, 20.0]),
        depth=15.0,
        top_z=20.0,
        floor_z=5.0,
        axis=np.array([0.0, 0.0, 1.0]),
        diameter=20.0,
        radius=10.0,
        is_concave=True,
        accessibility=Accessibility.TOOL_AXIS_OK,
        center=np.array([20.0, 25.0, 12.5]),
        surface_type="cylinder",
        source_geometry_reference="Face#7:CYLINDER:Ø20.000",
        space=CoordSpace.MODEL,
    )

    # Setup 1 (G54, +50 X, +100 Y, +10 Z)
    M1 = make_transform(translation=np.array([50.0, 100.0, 10.0]))
    setup1 = Setup(setup_id="op10_top", WCS="G54", model_to_setup=M1)

    s1_feat = setup1.transform_feature(canonical_feat, new_id="s1_bore_007")

    # Verify nominal CAD invariants in Setup 1
    assert s1_feat.canonical_id == "bore_canonical_007"
    assert s1_feat.face_indices == [7]
    assert s1_feat.diameter == 20.0
    assert s1_feat.radius == 10.0
    assert s1_feat.depth == 15.0
    assert s1_feat.source_geometry_reference == "Face#7:CYLINDER:Ø20.000"
    assert s1_feat.space == CoordSpace.SETUP
    # Transformed spatial bounds
    np.testing.assert_allclose(s1_feat.bounds_min, np.array([60.0, 115.0, 15.0]))
    np.testing.assert_allclose(s1_feat.bounds_max, np.array([80.0, 135.0, 30.0]))
    np.testing.assert_allclose(s1_feat.center, np.array([70.0, 125.0, 22.5]))
    assert s1_feat.top_z == 30.0
    assert s1_feat.floor_z == 15.0

    # Setup 2 (G55, Flipped 180° around X axis + translation)
    rot_flip = rotation_about_axis(np.array([1.0, 0.0, 0.0]), np.pi)
    M2 = make_transform(rotation=rot_flip, translation=np.array([0.0, 100.0, 50.0]))
    setup2 = Setup(setup_id="op20_bottom", WCS="G55", model_to_setup=M2)

    s2_feat = setup2.transform_feature(canonical_feat, new_id="s2_bore_007")

    # Verify nominal CAD invariants in Setup 2
    assert s2_feat.canonical_id == "bore_canonical_007"
    assert s2_feat.face_indices == [7]
    assert s2_feat.diameter == 20.0
    assert s2_feat.radius == 10.0
    assert s2_feat.depth == 15.0
    assert s2_feat.source_geometry_reference == "Face#7:CYLINDER:Ø20.000"
    assert s2_feat.space == CoordSpace.SETUP

    # Inverse mapping from Setup 2 restores exact canonical point
    restored_center = setup2.inverse_transform_point(s2_feat.center)
    np.testing.assert_allclose(restored_center, canonical_feat.center, atol=1e-6)

    # Design geometry surface identity is preserved
    nom_surface = s1_feat.design_geometry.primary_surface
    assert nom_surface.face_ids == [7]
    assert nom_surface.diameter == 20.0
    assert nom_surface.radius == 10.0
