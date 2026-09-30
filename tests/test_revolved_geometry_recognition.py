"""Tests for rotational/revolved geometry recognition.

Validates that:
1. RevolvedGeometryRecognizer discovers coaxial rotational axes from B-Rep faces.
2. build_profile() constructs accurate radial/axial profiles:
   - Cylindrical steps (CYLINDRICAL_STEP)  -> correct diameter and axial span
   - Conical faces (CONICAL_TRANSITION)     -> correct semi-angle and endpoint radii
   - Toroidal fillets (TOROIDAL_FILLET)     -> correct minor curvature radius
   - Planar shoulders (PLANAR_SHOULDER)     -> correct inner/outer diameter and z-plane
3. Profile-level invariants (min/max diameter, axial extent, face coverage).
4. FeatureRecognizer._recognize_revolved_profile() emits the correct feature types
   (CONTOUR, BOSS, BORE, SHOULDER, CHAMFER, FILLET) with geometry sourced entirely
   from analytical B-Rep data -- NO synthetic coordinates.
5. Pure-revolution parts suppress the generic outer-contour fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from cam_engine.geometry.revolved import (
    RevolvedGeometryRecognizer,
    RevolvedProfileSegment,
    RevolvedProfileSegmentKind,
    TransitionKind,
)
from cam_engine.geometry.topology import SurfaceKind


# ---------------------------------------------------------------------------
# Minimal stub classes replicating Face / SurfaceData contract
# ---------------------------------------------------------------------------

@dataclass
class _SurfaceData:
    kind: SurfaceKind
    axis: Optional[np.ndarray] = None
    point_on: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    radius: Optional[float] = None
    ref_radius: Optional[float] = None
    semi_angle: Optional[float] = None
    u_span: Optional[float] = None
    u_min: Optional[float] = None
    u_max: Optional[float] = None


@dataclass
class _Face:
    index: int
    surface: _SurfaceData
    triangles: np.ndarray
    is_internal: Optional[bool] = None
    area: float = 1.0
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_max: np.ndarray = field(default_factory=lambda: np.ones(3))
    oriented_normal: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    edge_indices: list = field(default_factory=list)
    center: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    wires: list = field(default_factory=list)


def _cylinder_face(index, radius, z_min, z_max,
                   axis=None, origin=None, is_internal=True, n=12):
    axis = axis if axis is not None else np.array([0.0, 0.0, 1.0])
    origin = origin if origin is not None else np.zeros(3)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    tris = []
    for ang in angles:
        x = origin[0] + radius * math.cos(ang)
        y = origin[1] + radius * math.sin(ang)
        x2 = origin[0] + radius * math.cos(ang + 2 * math.pi / n)
        y2 = origin[1] + radius * math.sin(ang + 2 * math.pi / n)
        tris.append([[x, y, origin[2] + z_min],
                     [x, y, origin[2] + z_max],
                     [x2, y2, origin[2] + z_max]])
    surface = _SurfaceData(kind=SurfaceKind.CYLINDER, axis=axis.copy(),
                           point_on=origin.copy(), radius=radius,
                           u_span=2 * math.pi)
    area = 2 * math.pi * radius * abs(z_max - z_min)
    return _Face(index=index, surface=surface, triangles=np.array(tris),
                 is_internal=is_internal, area=area,
                 bounds_min=origin + np.array([-radius, -radius, z_min]),
                 bounds_max=origin + np.array([radius, radius, z_max]))


def _cone_face(index, r_min, r_max, z_min, z_max,
               axis=None, origin=None, n=12):
    axis = axis if axis is not None else np.array([0.0, 0.0, 1.0])
    origin = origin if origin is not None else np.zeros(3)
    semi_angle = math.atan2(abs(r_max - r_min), abs(z_max - z_min))
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    tris = []
    for ang in angles:
        tris.append([
            [origin[0] + r_min * math.cos(ang), origin[1] + r_min * math.sin(ang), origin[2] + z_min],
            [origin[0] + r_max * math.cos(ang), origin[1] + r_max * math.sin(ang), origin[2] + z_max],
            [origin[0] + r_max * math.cos(ang + 0.1), origin[1] + r_max * math.sin(ang + 0.1), origin[2] + z_max],
        ])
    axial_len = abs(z_max - z_min)
    area = math.pi * (r_min + r_max) * math.sqrt((r_max - r_min) ** 2 + axial_len ** 2)
    surface = _SurfaceData(kind=SurfaceKind.CONE, axis=axis.copy(),
                           point_on=origin.copy(), semi_angle=semi_angle)
    return _Face(index=index, surface=surface, triangles=np.array(tris),
                 is_internal=False, area=area,
                 bounds_min=origin + np.array([-r_max, -r_max, z_min]),
                 bounds_max=origin + np.array([r_max, r_max, z_max]))


def _torus_face(index, r_major, r_minor, z_min, z_max,
                axis=None, origin=None, n=12):
    axis = axis if axis is not None else np.array([0.0, 0.0, 1.0])
    origin = origin if origin is not None else np.zeros(3)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    tris = []
    for ang in angles:
        x = origin[0] + r_major * math.cos(ang)
        y = origin[1] + r_major * math.sin(ang)
        x2 = origin[0] + (r_major + r_minor) * math.cos(ang + 0.1)
        y2 = origin[1] + (r_major + r_minor) * math.sin(ang + 0.1)
        tris.append([[x, y, origin[2] + z_min],
                     [x, y, origin[2] + z_max],
                     [x2, y2, origin[2] + 0.5 * (z_min + z_max)]])
    area = 4 * math.pi ** 2 * r_major * r_minor
    surface = _SurfaceData(kind=SurfaceKind.TORUS, axis=axis.copy(),
                           point_on=origin.copy(), radius=r_major, ref_radius=r_minor)
    return _Face(index=index, surface=surface, triangles=np.array(tris),
                 is_internal=False, area=area,
                 bounds_min=origin + np.array([-(r_major + r_minor)] * 2 + [z_min]),
                 bounds_max=origin + np.array([(r_major + r_minor)] * 2 + [z_max]))


def _plane_face(index, z_level, r_inner, r_outer,
                axis=None, origin=None, n=12):
    axis = axis if axis is not None else np.array([0.0, 0.0, 1.0])
    origin = origin if origin is not None else np.zeros(3)
    z = origin[2] + z_level
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    tris = []
    for ang in angles:
        tris.append([
            [origin[0] + r_inner * math.cos(ang), origin[1] + r_inner * math.sin(ang), z],
            [origin[0] + r_outer * math.cos(ang), origin[1] + r_outer * math.sin(ang), z],
            [origin[0] + r_outer * math.cos(ang + 0.2), origin[1] + r_outer * math.sin(ang + 0.2), z],
        ])
    normal = axis / np.linalg.norm(axis)
    area = math.pi * (r_outer ** 2 - r_inner ** 2)
    surface = _SurfaceData(kind=SurfaceKind.PLANE, normal=normal, point_on=origin.copy())
    return _Face(index=index, surface=surface, triangles=np.array(tris),
                 is_internal=False, area=area, oriented_normal=normal,
                 bounds_min=origin + np.array([-r_outer, -r_outer, z_level]),
                 bounds_max=origin + np.array([r_outer, r_outer, z_level]))


# ===========================================================================
# 1. Axis discovery
# ===========================================================================

class TestAxisDiscovery:
    def test_single_cylinder_one_axis(self):
        f = _cylinder_face(0, radius=10.0, z_min=0.0, z_max=30.0)
        axes = RevolvedGeometryRecognizer([f]).discover_rotational_axes()
        assert len(axes) == 1
        assert 0 in axes[0][2]

    def test_two_coaxial_cylinders_merged(self):
        f0 = _cylinder_face(0, radius=10.0, z_min=0.0,  z_max=20.0)
        f1 = _cylinder_face(1, radius=15.0, z_min=20.0, z_max=40.0)
        axes = RevolvedGeometryRecognizer([f0, f1]).discover_rotational_axes()
        assert len(axes) == 1
        face_ids = axes[0][2]
        assert 0 in face_ids and 1 in face_ids

    def test_offset_cylinders_form_two_clusters(self):
        f0 = _cylinder_face(0, radius=5.0, z_min=0.0, z_max=20.0, origin=np.zeros(3))
        f1 = _cylinder_face(1, radius=5.0, z_min=0.0, z_max=20.0,
                            origin=np.array([50.0, 0.0, 0.0]))
        axes = RevolvedGeometryRecognizer([f0, f1]).discover_rotational_axes()
        assert len(axes) == 2

    def test_no_revolved_faces_returns_empty(self):
        f = _plane_face(0, z_level=0.0, r_inner=0.0, r_outer=20.0)
        axes = RevolvedGeometryRecognizer([f]).discover_rotational_axes()
        assert axes == []

    def test_cone_grouped_with_coaxial_cylinder(self):
        fc = _cylinder_face(0, radius=10.0, z_min=0.0, z_max=20.0)
        fk = _cone_face(1, r_min=10.0, r_max=15.0, z_min=20.0, z_max=25.0)
        axes = RevolvedGeometryRecognizer([fc, fk]).discover_rotational_axes()
        assert len(axes) == 1
        assert 0 in axes[0][2] and 1 in axes[0][2]


# ===========================================================================
# 2. Cylindrical profile
# ===========================================================================

class TestCylindricalProfile:
    def _profile(self, faces):
        rec = RevolvedGeometryRecognizer(faces)
        orig, ax, _ = rec.discover_rotational_axes()[0]
        return rec.build_profile(orig, ax)

    def test_single_external_cylinder(self):
        f = _cylinder_face(0, radius=15.0, z_min=0.0, z_max=50.0, is_internal=False)
        prof = self._profile([f])
        assert prof is not None
        assert len(prof.external_segments) == 1
        seg = prof.external_segments[0]
        assert seg.kind == RevolvedProfileSegmentKind.CYLINDRICAL_STEP
        assert abs(seg.r_start - 15.0) < 0.5
        assert seg.d_start == pytest.approx(2 * seg.r_start, abs=0.01)
        assert seg.axial_length == pytest.approx(50.0, abs=0.5)

    def test_single_internal_bore(self):
        f = _cylinder_face(0, radius=8.0, z_min=0.0, z_max=25.0, is_internal=True)
        prof = self._profile([f])
        assert prof is not None
        assert len(prof.internal_segments) >= 1
        seg = prof.internal_segments[0]
        assert seg.is_internal is True
        assert abs(seg.r_start - 8.0) < 0.5

    def test_stepped_profile_two_cylinders(self):
        f0 = _cylinder_face(0, radius=15.0, z_min=0.0,  z_max=30.0, is_internal=False)
        f1 = _cylinder_face(1, radius=10.0, z_min=30.0, z_max=60.0, is_internal=False)
        prof = self._profile([f0, f1])
        assert len(prof.external_segments) == 2
        dias = sorted(s.d_start for s in prof.external_segments)
        assert dias[0] == pytest.approx(20.0, abs=1.0)
        assert dias[1] == pytest.approx(30.0, abs=1.0)

    def test_profile_min_max_diameter(self):
        f0 = _cylinder_face(0, radius=5.0,  z_min=0.0,  z_max=10.0, is_internal=False)
        f1 = _cylinder_face(1, radius=20.0, z_min=10.0, z_max=30.0, is_internal=False)
        prof = self._profile([f0, f1])
        assert prof.min_diameter == pytest.approx(10.0, abs=1.0)
        assert prof.max_diameter == pytest.approx(40.0, abs=1.0)

    def test_profile_axial_extent(self):
        f0 = _cylinder_face(0, radius=12.0, z_min=5.0,  z_max=25.0, is_internal=False)
        f1 = _cylinder_face(1, radius=12.0, z_min=25.0, z_max=55.0, is_internal=False)
        prof = self._profile([f0, f1])
        assert prof.axial_min == pytest.approx(5.0, abs=0.5)
        assert prof.axial_max == pytest.approx(55.0, abs=0.5)
        assert prof.total_axial_length == pytest.approx(50.0, abs=1.0)

    def test_radius_at_query(self):
        f = _cylinder_face(0, radius=10.0, z_min=0.0, z_max=30.0, is_internal=False)
        prof = self._profile([f])
        r = prof.radius_at(15.0, internal=False)
        assert r is not None
        assert r == pytest.approx(10.0, abs=0.5)

    def test_diameter_at_query(self):
        f = _cylinder_face(0, radius=7.5, z_min=0.0, z_max=20.0, is_internal=False)
        prof = self._profile([f])
        d = prof.diameter_at(10.0, internal=False)
        assert d is not None
        assert d == pytest.approx(15.0, abs=1.0)


# ===========================================================================
# 3. Conical profile
# ===========================================================================

class TestConicalProfile:
    def _profile(self, faces):
        rec = RevolvedGeometryRecognizer(faces)
        orig, ax, _ = rec.discover_rotational_axes()[0]
        return rec.build_profile(orig, ax)

    def test_cone_creates_conical_segment(self):
        f = _cone_face(0, r_min=5.0, r_max=15.0, z_min=0.0, z_max=10.0)
        prof = self._profile([f])
        assert prof is not None
        cone_segs = [s for s in prof.segments
                     if s.kind == RevolvedProfileSegmentKind.CONICAL_TRANSITION]
        assert len(cone_segs) >= 1

    def test_cone_semi_angle_approximately_correct(self):
        # atan(10/10) = 45 degrees
        f = _cone_face(0, r_min=5.0, r_max=15.0, z_min=0.0, z_max=10.0)
        prof = self._profile([f])
        seg = next(s for s in prof.segments
                   if s.kind == RevolvedProfileSegmentKind.CONICAL_TRANSITION)
        assert seg.semi_angle_deg is not None
        assert 30.0 <= seg.semi_angle_deg <= 60.0

    def test_cone_radii_endpoints_ordered(self):
        f = _cone_face(0, r_min=3.0, r_max=12.0, z_min=0.0, z_max=8.0)
        prof = self._profile([f])
        seg = next(s for s in prof.segments
                   if s.kind == RevolvedProfileSegmentKind.CONICAL_TRANSITION)
        assert min(seg.r_start, seg.r_end) == pytest.approx(3.0, abs=1.5)
        assert max(seg.r_start, seg.r_end) == pytest.approx(12.0, abs=1.5)

    def test_chamfer_transition_type_detected(self):
        # Cone at higher z (s1) followed by cylinder (s2) -- triggers CHAMFER transition
        fk = _cone_face(0, r_min=8.0, r_max=10.0, z_min=20.0, z_max=25.0)
        fc = _cylinder_face(1, radius=10.0, z_min=0.0, z_max=20.0, is_internal=False)
        prof = self._profile([fk, fc])
        assert any(t.transition_type == TransitionKind.CHAMFER for t in prof.transitions)


# ===========================================================================
# 4. Toroidal profile
# ===========================================================================

class TestToroidalProfile:
    def _profile(self, faces):
        rec = RevolvedGeometryRecognizer(faces)
        orig, ax, _ = rec.discover_rotational_axes()[0]
        return rec.build_profile(orig, ax)

    def test_torus_creates_toroidal_segment(self):
        f = _torus_face(0, r_major=15.0, r_minor=3.0, z_min=0.0, z_max=6.0)
        prof = self._profile([f])
        tor_segs = [s for s in prof.segments
                    if s.kind == RevolvedProfileSegmentKind.TOROIDAL_FILLET]
        assert len(tor_segs) >= 1

    def test_torus_curvature_radius_stored(self):
        r_minor = 2.5
        f = _torus_face(0, r_major=12.0, r_minor=r_minor, z_min=0.0, z_max=5.0)
        prof = self._profile([f])
        seg = next(s for s in prof.segments
                   if s.kind == RevolvedProfileSegmentKind.TOROIDAL_FILLET)
        assert seg.curvature_radius is not None
        assert seg.curvature_radius == pytest.approx(r_minor, abs=0.1)

    def test_fillet_transition_type_detected(self):
        # Torus at higher z (s1) followed by cylinder (s2) -- triggers FILLET transition
        ft = _torus_face(0, r_major=10.0, r_minor=2.0, z_min=20.0, z_max=24.0)
        fc = _cylinder_face(1, radius=10.0, z_min=0.0, z_max=20.0, is_internal=False)
        prof = self._profile([ft, fc])
        assert any(t.transition_type == TransitionKind.FILLET for t in prof.transitions)


# ===========================================================================
# 5. Planar shoulders
# ===========================================================================

class TestPlanarShoulderProfile:
    def _profile(self, faces):
        rec = RevolvedGeometryRecognizer(faces)
        orig, ax, _ = rec.discover_rotational_axes()[0]
        return rec.build_profile(orig, ax)

    def test_shoulder_creates_planar_segment(self):
        axis = np.array([0.0, 0.0, 1.0])
        fp = _plane_face(0, z_level=20.0, r_inner=5.0, r_outer=15.0, axis=axis)
        fc = _cylinder_face(1, radius=15.0, z_min=0.0, z_max=20.0, is_internal=False)
        prof = self._profile([fp, fc])
        shoulder_segs = [s for s in prof.segments
                         if s.kind == RevolvedProfileSegmentKind.PLANAR_SHOULDER]
        assert len(shoulder_segs) >= 1

    def test_shoulder_inner_outer_diameter(self):
        axis = np.array([0.0, 0.0, 1.0])
        fp = _plane_face(0, z_level=20.0, r_inner=5.0, r_outer=15.0, axis=axis)
        fc = _cylinder_face(1, radius=15.0, z_min=0.0, z_max=20.0, is_internal=False)
        prof = self._profile([fp, fc])
        shoulder = next(s for s in prof.segments
                        if s.kind == RevolvedProfileSegmentKind.PLANAR_SHOULDER)
        assert shoulder.d_min == pytest.approx(10.0, abs=1.5)
        assert shoulder.d_max == pytest.approx(30.0, abs=1.5)

    def test_shoulder_width_stored(self):
        axis = np.array([0.0, 0.0, 1.0])
        fp = _plane_face(0, z_level=20.0, r_inner=5.0, r_outer=15.0, axis=axis)
        fc = _cylinder_face(1, radius=15.0, z_min=0.0, z_max=20.0, is_internal=False)
        prof = self._profile([fp, fc])
        shoulder = next(s for s in prof.segments
                        if s.kind == RevolvedProfileSegmentKind.PLANAR_SHOULDER)
        assert shoulder.shoulder_width is not None
        assert shoulder.shoulder_width == pytest.approx(10.0, abs=1.5)

    def test_square_step_transition_detected(self):
        axis = np.array([0.0, 0.0, 1.0])
        fp = _plane_face(0, z_level=20.0, r_inner=5.0, r_outer=15.0, axis=axis)
        f0 = _cylinder_face(1, radius=15.0, z_min=0.0,  z_max=20.0, is_internal=False)
        f1 = _cylinder_face(2, radius=5.0,  z_min=20.0, z_max=40.0, is_internal=False)
        prof = self._profile([fp, f0, f1])
        assert any(t.transition_type == TransitionKind.SQUARE_STEP for t in prof.transitions)


# ===========================================================================
# 6. Coverage and pure-revolution flag
# ===========================================================================

class TestCoverageAndPureRevolution:
    def test_pure_revolution_when_all_revolved(self):
        faces = [
            _cylinder_face(0, radius=10, z_min=0,  z_max=20, is_internal=False),
            _cylinder_face(1, radius=15, z_min=20, z_max=40, is_internal=False),
            _cylinder_face(2, radius=10, z_min=40, z_max=60, is_internal=False),
        ]
        rec = RevolvedGeometryRecognizer(faces)
        orig, ax, _ = rec.discover_rotational_axes()[0]
        prof = rec.build_profile(orig, ax)
        assert prof.face_coverage_ratio > 0.8
        assert prof.is_pure_revolution is True

    def test_partial_revolution_when_dominated_by_planar(self):
        # One small cylinder + a very large non-perpendicular plane face
        # (oblique normal => NOT counted as a perpendicular shoulder).
        fc = _cylinder_face(0, radius=10.0, z_min=0.0, z_max=20.0, is_internal=False)
        # Slanted planar face with large area -- oblique normal, not coaxial/perpendicular
        oblique_normal = np.array([1.0, 0.0, 0.1])
        oblique_normal /= np.linalg.norm(oblique_normal)
        oblique_surface = _SurfaceData(kind=SurfaceKind.PLANE, normal=oblique_normal)
        big_plane = _Face(index=1, surface=oblique_surface,
                          triangles=np.zeros((1, 3, 3)),
                          is_internal=False, area=500_000.0,
                          oriented_normal=oblique_normal)
        rec = RevolvedGeometryRecognizer([fc, big_plane])
        orig, ax, _ = rec.discover_rotational_axes()[0]
        prof = rec.build_profile(orig, ax)
        assert prof.face_coverage_ratio < 0.85

    def test_extract_primary_profile(self):
        f = _cylinder_face(0, radius=8.0, z_min=0.0, z_max=40.0, is_internal=False)
        prof = RevolvedGeometryRecognizer([f]).extract_primary_profile()
        assert prof is not None
        assert len(prof.segments) >= 1


# ===========================================================================
# 7. Serialisation
# ===========================================================================

class TestProfileSerialisation:
    def _profile(self, faces):
        rec = RevolvedGeometryRecognizer(faces)
        orig, ax, _ = rec.discover_rotational_axes()[0]
        return rec.build_profile(orig, ax)

    def test_segment_to_dict_keys(self):
        f = _cylinder_face(0, radius=10.0, z_min=0.0, z_max=20.0, is_internal=False)
        seg_dict = self._profile([f]).segments[0].to_dict()
        for key in ("index", "kind", "z_start", "z_end", "r_start", "r_end",
                    "d_start", "d_end", "is_internal"):
            assert key in seg_dict

    def test_profile_to_dict_keys(self):
        f = _cylinder_face(0, radius=10.0, z_min=0.0, z_max=20.0, is_internal=False)
        d = self._profile([f]).to_dict()
        for key in ("axis_origin", "axis_direction", "min_diameter", "max_diameter",
                    "axial_min", "axial_max", "segments", "transitions",
                    "is_pure_revolution", "face_coverage_ratio"):
            assert key in d

    def test_transition_to_dict_keys(self):
        axis = np.array([0.0, 0.0, 1.0])
        prof = self._profile([
            _plane_face(0, z_level=20.0, r_inner=5.0, r_outer=15.0, axis=axis),
            _cylinder_face(1, radius=15.0, z_min=0.0, z_max=20.0, is_internal=False),
        ])
        if prof.transitions:
            d = prof.transitions[0].to_dict()
            for key in ("z", "r_inner", "r_outer", "transition_type"):
                assert key in d

    def test_get_steps_returns_list(self):
        prof = self._profile([
            _plane_face(0, z_level=15.0, r_inner=5.0, r_outer=10.0),
            _cylinder_face(1, radius=10.0, z_min=0.0, z_max=15.0, is_internal=False),
        ])
        assert isinstance(prof.get_steps(), list)


# ===========================================================================
# 8. Segment analytical helpers
# ===========================================================================

class TestSegmentHelpers:
    def _cyl_seg(self, r=10.0, z0=0.0, z1=50.0):
        return RevolvedProfileSegment(
            index=0, kind=RevolvedProfileSegmentKind.CYLINDRICAL_STEP,
            face_indices=[0], surface_kind=SurfaceKind.CYLINDER,
            z_start=z1, z_end=z0, r_start=r, r_end=r, is_internal=False)

    def test_d_start_is_twice_r_start(self):
        assert self._cyl_seg(r=12.5).d_start == pytest.approx(25.0)

    def test_axial_length(self):
        assert self._cyl_seg(z0=5.0, z1=35.0).axial_length == pytest.approx(30.0)

    def test_z_min_z_max(self):
        seg = self._cyl_seg(z0=3.0, z1=27.0)
        assert seg.z_min == pytest.approx(3.0)
        assert seg.z_max == pytest.approx(27.0)

    def test_radius_at_constant_for_cylinder(self):
        seg = self._cyl_seg(r=8.0, z0=0.0, z1=20.0)
        assert seg.radius_at(10.0) == pytest.approx(8.0)
        assert seg.radius_at(0.0) == pytest.approx(8.0)
        assert seg.radius_at(20.0) == pytest.approx(8.0)

    def test_radius_at_none_outside_range(self):
        seg = self._cyl_seg(z0=0.0, z1=20.0)
        assert seg.radius_at(-1.0) is None
        assert seg.radius_at(21.0) is None

    def test_conical_radius_at_linear_interpolation(self):
        seg = RevolvedProfileSegment(
            index=0, kind=RevolvedProfileSegmentKind.CONICAL_TRANSITION,
            face_indices=[0], surface_kind=SurfaceKind.CONE,
            z_start=10.0, z_end=0.0, r_start=10.0, r_end=5.0,
            is_internal=False, semi_angle_deg=26.57)
        r_mid = seg.radius_at(5.0)
        assert r_mid is not None
        assert r_mid == pytest.approx(7.5, abs=0.1)


# ===========================================================================
# 9. FeatureRecognizer integration
# ===========================================================================

class _MockShape:
    def __init__(self, faces):
        self._faces = faces

    def faces(self):
        return self._faces

    def edges(self):
        return []

    def face_adjacency(self):
        return {}


class TestFeatureRecognizerIntegration:
    def _fr(self, faces):
        from cam_engine.features import FeatureRecognizer
        return FeatureRecognizer(shape=_MockShape(faces), stock_top_z=100.0)

    def test_pure_revolution_flag(self):
        faces = [
            _cylinder_face(0, radius=15.0, z_min=0.0,  z_max=50.0, is_internal=False),
            _cylinder_face(1, radius=10.0, z_min=50.0, z_max=80.0, is_internal=False),
        ]
        _, is_pure = self._fr(faces)._recognize_revolved_profile()
        assert is_pure is True

    def test_emits_contour_for_primary_axis(self):
        from cam_engine.features import FeatureType
        feats, _ = self._fr([_cylinder_face(0, radius=15.0, z_min=0.0, z_max=50.0,
                                             is_internal=False)])._recognize_revolved_profile()
        assert any(f.type == FeatureType.CONTOUR for f in feats)

    def test_contour_notes_have_profile_and_diameters(self):
        from cam_engine.features import FeatureType
        feats, _ = self._fr([_cylinder_face(0, radius=12.0, z_min=0.0, z_max=40.0,
                                             is_internal=False)])._recognize_revolved_profile()
        contour = next(f for f in feats if f.type == FeatureType.CONTOUR)
        assert "revolved_profile" in contour.notes
        assert "min_diameter" in contour.notes
        assert "max_diameter" in contour.notes

    def test_external_cylinder_emits_boss(self):
        from cam_engine.features import FeatureType
        feats, _ = self._fr([_cylinder_face(0, radius=15.0, z_min=0.0, z_max=50.0,
                                             is_internal=False)])._recognize_revolved_profile()
        assert any(f.type == FeatureType.BOSS for f in feats)

    def test_internal_cylinder_emits_bore(self):
        from cam_engine.features import FeatureType
        faces = [
            _cylinder_face(0, radius=15.0, z_min=0.0, z_max=40.0, is_internal=False),
            _cylinder_face(1, radius=5.0,  z_min=0.0, z_max=40.0, is_internal=True),
        ]
        feats, _ = self._fr(faces)._recognize_revolved_profile()
        assert any(f.type == FeatureType.BORE for f in feats)

    def test_planar_shoulder_emits_shoulder(self):
        from cam_engine.features import FeatureType
        axis = np.array([0.0, 0.0, 1.0])
        faces = [
            _cylinder_face(0, radius=15.0, z_min=0.0,  z_max=20.0, is_internal=False),
            _cylinder_face(1, radius=8.0,  z_min=20.0, z_max=40.0, is_internal=False),
            _plane_face(2, z_level=20.0, r_inner=8.0, r_outer=15.0, axis=axis),
        ]
        feats, _ = self._fr(faces)._recognize_revolved_profile()
        assert any(f.type == FeatureType.SHOULDER for f in feats)

    def test_conical_face_emits_chamfer(self):
        from cam_engine.features import FeatureType
        faces = [
            _cylinder_face(0, radius=10.0, z_min=8.0, z_max=30.0, is_internal=False),
            _cone_face(1, r_min=8.0, r_max=10.0, z_min=0.0, z_max=8.0),
        ]
        feats, _ = self._fr(faces)._recognize_revolved_profile()
        assert any(f.type == FeatureType.CHAMFER for f in feats)

    def test_toroidal_face_emits_fillet(self):
        from cam_engine.features import FeatureType
        faces = [
            _cylinder_face(0, radius=10.0, z_min=4.0, z_max=30.0, is_internal=False),
            _torus_face(1, r_major=10.0, r_minor=2.0, z_min=0.0, z_max=4.0),
        ]
        feats, _ = self._fr(faces)._recognize_revolved_profile()
        assert any(f.type == FeatureType.FILLET for f in feats)

    def test_boss_bore_features_have_positive_diameter(self):
        from cam_engine.features import FeatureType
        feats, _ = self._fr([_cylinder_face(0, radius=20.0, z_min=0.0, z_max=60.0,
                                             is_internal=False)])._recognize_revolved_profile()
        for feat in feats:
            if feat.type in (FeatureType.BOSS, FeatureType.BORE):
                assert feat.diameter is not None and feat.diameter > 0.0

    def test_all_features_have_nonzero_axis(self):
        feats, _ = self._fr([_cylinder_face(0, radius=20.0, z_min=0.0, z_max=60.0,
                                             is_internal=False)])._recognize_revolved_profile()
        for feat in feats:
            if feat.axis is not None:
                assert np.linalg.norm(feat.axis) > 0.5

    def test_empty_shape_no_features(self):
        feats, is_pure = self._fr([])._recognize_revolved_profile()
        assert feats == [] and is_pure is False

    def test_non_revolved_shape_no_features(self):
        f = _plane_face(0, z_level=0.0, r_inner=0.0, r_outer=50.0)
        feats, is_pure = self._fr([f])._recognize_revolved_profile()
        assert feats == [] and is_pure is False

    def test_source_geometry_reference_says_revolved(self):
        feats, _ = self._fr([_cylinder_face(0, radius=12.0, z_min=0.0, z_max=40.0,
                                             is_internal=False)])._recognize_revolved_profile()
        for feat in feats:
            if feat.source_geometry_reference is not None:
                assert "REVOLVED" in feat.source_geometry_reference

    def test_stepped_shaft_notes_reflect_brep_diameters(self):
        from cam_engine.features import FeatureType
        faces = [
            _cylinder_face(0, radius=20.0, z_min=0.0,  z_max=30.0, is_internal=False),
            _cylinder_face(1, radius=12.0, z_min=30.0, z_max=60.0, is_internal=False),
        ]
        feats, _ = self._fr(faces)._recognize_revolved_profile()
        contour = next(f for f in feats if f.type == FeatureType.CONTOUR)
        assert contour.notes["min_diameter"] == pytest.approx(24.0, abs=2.0)
        assert contour.notes["max_diameter"] == pytest.approx(40.0, abs=2.0)
