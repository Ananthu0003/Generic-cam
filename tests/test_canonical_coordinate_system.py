"""Tests for Canonical Part Coordinate System and Setup Transformations.

Verifies:
1. Features originate in Canonical CAD/B-Rep coordinates.
2. Setup 1 and Setup 2 rigidly transform the SAME canonical feature without regenerating or modifying nominal geometry.
3. Invariant nominal geometry (diameter, depth, corner radius) is perfectly preserved.
4. Boundaries, axes, and heights are transformed via the exact rigid 4x4 matrix.
"""

import numpy as np
import pytest

from cam_engine.coords import (
    CoordSpace,
    make_transform,
    rotation_about_axis,
    transform_points,
    transform_direction,
)
from cam_engine.features import MachiningFeature, FeatureType, Accessibility


def test_canonical_feature_rigid_transform_invariance():
    """Verify transforming a canonical feature preserves nominal geometry while updating spatial vectors."""
    canon_feat = MachiningFeature(
        id="hole_42",
        canonical_id="hole_42",
        space=CoordSpace.MODEL,
        type=FeatureType.BLIND_HOLE,
        face_indices=[42, 43],
        bounds_min=np.array([10.0, 20.0, -15.0]),
        bounds_max=np.array([16.0, 26.0, 0.0]),
        depth=15.0,
        top_z=0.0,
        floor_z=-15.0,
        axis=np.array([0.0, 0.0, 1.0]),
        diameter=6.0,
        radius=3.0,
        is_concave=True,
        accessibility=Accessibility.TOOL_AXIS_OK,
        machining_direction=np.array([0.0, 0.0, -1.0]),
        boundary_loop=[np.array([[10.0, 20.0, 0.0], [16.0, 20.0, 0.0], [16.0, 26.0, 0.0], [10.0, 26.0, 0.0]])],
    )

    # 1. Setup 1: Translation only (+50 X, +100 Y, +20 Z)
    t_setup1 = make_transform(translation=np.array([50.0, 100.0, 20.0]))
    s1_feat = canon_feat.transformed(t_setup1, dst_space=CoordSpace.SETUP, new_id="s1_hole_42")

    # Invariants
    assert s1_feat.canonical_id == "hole_42"
    assert s1_feat.diameter == 6.0
    assert s1_feat.depth == 15.0
    assert s1_feat.radius == 3.0
    assert s1_feat.face_indices == [42, 43]

    # Spatial transformation
    np.testing.assert_allclose(s1_feat.bounds_min, np.array([60.0, 120.0, 5.0]))
    np.testing.assert_allclose(s1_feat.bounds_max, np.array([66.0, 126.0, 20.0]))
    np.testing.assert_allclose(s1_feat.axis, np.array([0.0, 0.0, 1.0]))
    assert s1_feat.top_z == 20.0
    assert s1_feat.floor_z == 5.0

    # 2. Setup 2: 180 degree flip about X axis (Z inverted, Y inverted)
    rot_180_x = rotation_about_axis(np.array([1.0, 0.0, 0.0]), np.pi)
    t_setup2 = make_transform(rotation=rot_180_x, translation=np.array([0.0, 50.0, 100.0]))
    s2_feat = canon_feat.transformed(t_setup2, dst_space=CoordSpace.SETUP, new_id="s2_hole_42")

    # Invariants must remain exact
    assert s2_feat.canonical_id == "hole_42"
    assert s2_feat.diameter == 6.0
    assert s2_feat.depth == 15.0
    assert s2_feat.radius == 3.0

    # Flipped axis and direction
    np.testing.assert_allclose(s2_feat.axis, np.array([0.0, 0.0, -1.0]), atol=1e-6)
    np.testing.assert_allclose(s2_feat.machining_direction, np.array([0.0, 0.0, 1.0]), atol=1e-6)
