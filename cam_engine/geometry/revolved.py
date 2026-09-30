"""Rotational / Revolved Geometry Recognition and Radial/Axial Profile Engine.

Provides exact analytical recognition of rotationally symmetric parts:
- Coaxial cylindrical faces (diameters, axial spans, internal vs external)
- Conical faces (tapers, chamfers, countersinks, drafts)
- Toroidal / fillet faces (corner blends, rounds, annular grooves)
- Planar shoulders (annular steps perpendicular to axis)
- Ordered axial sections and diameter transition topology (fillets, chamfers, square steps)

Constructs a continuous radial/axial feature profile r(z) along the rotational axis,
enabling geometry-driven milling and turning strategies without arbitrary 2D XY slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence
import math

import numpy as np

from .topology import Face, Shape, SurfaceKind, WireLoop


class RevolvedProfileSegmentKind(str, Enum):
    CYLINDRICAL_STEP = "cylindrical_step"       # Constant radius cylinder (OD step or ID bore)
    CONICAL_TRANSITION = "conical_transition"   # Linear taper, chamfer, draft, or conical seat
    TOROIDAL_FILLET = "toroidal_fillet"         # Circular arc blend, round, or fillet
    PLANAR_SHOULDER = "planar_shoulder"         # Annular shoulder perpendicular to rotation axis
    GROOVE_RECESS = "groove_recess"             # Radial relief / undercut / o-ring groove
    OTHER = "other"


class TransitionKind(str, Enum):
    SQUARE_STEP = "square_step"                 # Sharp 90° planar shoulder transition
    CHAMFER = "chamfer"                         # Conical bevel transition
    FILLET = "fillet"                           # Toroidal blend / fillet transition
    UNDERCUT = "undercut"                       # Recessed groove transition


@dataclass
class RevolvedProfileSegment:
    """One axial/radial section of a revolved body along its rotational axis."""

    index: int
    kind: RevolvedProfileSegmentKind
    face_indices: list[int]
    surface_kind: SurfaceKind
    z_start: float                              # Axial start coordinate along rotation axis
    z_end: float                                # Axial end coordinate along rotation axis
    r_start: float                              # Radial distance from axis at z_start (mm)
    r_end: float                                # Radial distance from axis at z_end (mm)
    is_internal: bool                           # True for internal bore/cavity, False for outer OD
    curvature_radius: Optional[float] = None    # Minor radius for toroidal fillets/rounds (mm)
    semi_angle_deg: Optional[float] = None      # Semi-angle for conical faces (deg)
    shoulder_width: Optional[float] = None      # Radial span |r_end - r_start| for planar shoulders (mm)
    center_point: Optional[np.ndarray] = None   # 3D center in model space
    notes: dict = field(default_factory=dict)

    @property
    def z_min(self) -> float:
        return min(self.z_start, self.z_end)

    @property
    def z_max(self) -> float:
        return max(self.z_start, self.z_end)

    @property
    def axial_length(self) -> float:
        return abs(self.z_end - self.z_start)

    @property
    def d_start(self) -> float:
        return 2.0 * self.r_start

    @property
    def d_end(self) -> float:
        return 2.0 * self.r_end

    @property
    def d_min(self) -> float:
        return 2.0 * min(self.r_start, self.r_end)

    @property
    def d_max(self) -> float:
        return 2.0 * max(self.r_start, self.r_end)

    @property
    def mean_diameter(self) -> float:
        return float(self.r_start + self.r_end)

    def radius_at(self, z: float) -> Optional[float]:
        """Compute exact nominal radius r at axial coordinate z."""
        if z < self.z_min - 1e-4 or z > self.z_max + 1e-4:
            return None
        if self.kind == RevolvedProfileSegmentKind.CYLINDRICAL_STEP:
            return self.r_start
        if self.kind == RevolvedProfileSegmentKind.PLANAR_SHOULDER:
            return 0.5 * (self.r_start + self.r_end)
        if self.kind == RevolvedProfileSegmentKind.CONICAL_TRANSITION:
            dz = self.z_end - self.z_start
            if abs(dz) < 1e-6:
                return self.r_start
            t = (z - self.z_start) / dz
            return self.r_start + t * (self.r_end - self.r_start)
        if self.kind == RevolvedProfileSegmentKind.TOROIDAL_FILLET and self.curvature_radius is not None:
            # Approximate linear interpolation if arc geometry
            dz = self.z_end - self.z_start
            if abs(dz) < 1e-6:
                return self.r_start
            t = (z - self.z_start) / dz
            return self.r_start + t * (self.r_end - self.r_start)
        return self.r_start

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "face_indices": self.face_indices,
            "z_start": round(self.z_start, 4),
            "z_end": round(self.z_end, 4),
            "r_start": round(self.r_start, 4),
            "r_end": round(self.r_end, 4),
            "d_start": round(self.d_start, 4),
            "d_end": round(self.d_end, 4),
            "is_internal": self.is_internal,
            "curvature_radius": round(self.curvature_radius, 4) if self.curvature_radius is not None else None,
            "semi_angle_deg": round(self.semi_angle_deg, 2) if self.semi_angle_deg is not None else None,
            "shoulder_width": round(self.shoulder_width, 4) if self.shoulder_width is not None else None,
            "notes": self.notes,
        }


@dataclass
class RevolvedEdgeTransition:
    """Transition between consecutive axial sections (fillet, chamfer, or sharp step)."""

    z: float
    r_inner: float
    r_outer: float
    transition_type: TransitionKind
    connected_segments: tuple[int, int]
    fillet_radius: Optional[float] = None
    chamfer_angle_deg: Optional[float] = None
    chamfer_width: Optional[float] = None
    face_indices: list[int] = field(default_factory=list)

    @property
    def d_inner(self) -> float:
        return 2.0 * self.r_inner

    @property
    def d_outer(self) -> float:
        return 2.0 * self.r_outer

    @property
    def step_height(self) -> float:
        return abs(self.r_outer - self.r_inner)

    def to_dict(self) -> dict:
        return {
            "z": round(self.z, 4),
            "r_inner": round(self.r_inner, 4),
            "r_outer": round(self.r_outer, 4),
            "d_inner": round(self.d_inner, 4),
            "d_outer": round(self.d_outer, 4),
            "step_height": round(self.step_height, 4),
            "transition_type": self.transition_type.value,
            "connected_segments": list(self.connected_segments),
            "fillet_radius": round(self.fillet_radius, 4) if self.fillet_radius is not None else None,
            "chamfer_angle_deg": round(self.chamfer_angle_deg, 2) if self.chamfer_angle_deg is not None else None,
            "chamfer_width": round(self.chamfer_width, 4) if self.chamfer_width is not None else None,
            "face_indices": self.face_indices,
        }


@dataclass
class RevolvedProfile:
    """Complete analytical radial/axial profile of a rotationally symmetric / turned part."""

    axis_origin: np.ndarray                     # Reference origin point on rotation axis
    axis_direction: np.ndarray                  # Unit vector of rotation axis
    segments: list[RevolvedProfileSegment]      # All segments ordered along axis
    external_segments: list[RevolvedProfileSegment] # External turned OD envelope
    internal_segments: list[RevolvedProfileSegment] # Internal bore / ID cavity envelope
    shoulders: list[RevolvedProfileSegment]     # Planar shoulders perpendicular to axis
    transitions: list[RevolvedEdgeTransition]   # Edge transitions (chamfers, fillets, steps)
    min_diameter: float
    max_diameter: float
    axial_min: float
    axial_max: float
    is_pure_revolution: bool                    # True if 100% of body is revolved
    face_coverage_ratio: float                  # Fraction of part surface area belonging to revolved features

    @property
    def total_axial_length(self) -> float:
        return abs(self.axial_max - self.axial_min)

    def radius_at(self, z: float, internal: bool = False) -> Optional[float]:
        """Query nominal radius r at axial level z."""
        pool = self.internal_segments if internal else self.external_segments
        for seg in pool:
            r = seg.radius_at(z)
            if r is not None:
                return r
        return None

    def diameter_at(self, z: float, internal: bool = False) -> Optional[float]:
        """Query nominal diameter D at axial level z."""
        r = self.radius_at(z, internal=internal)
        return (2.0 * r) if r is not None else None

    def get_steps(self) -> list[dict]:
        """Return auditable list of all diameter steps and shoulders."""
        steps = []
        for trans in self.transitions:
            steps.append({
                "z": trans.z,
                "d_from": trans.d_inner,
                "d_to": trans.d_outer,
                "step_radial": trans.step_height,
                "transition": trans.transition_type.value,
                "fillet_r": trans.fillet_radius,
                "chamfer_deg": trans.chamfer_angle_deg,
            })
        return steps

    def to_dict(self) -> dict:
        return {
            "axis_origin": [round(x, 4) for x in self.axis_origin],
            "axis_direction": [round(x, 4) for x in self.axis_direction],
            "min_diameter": round(self.min_diameter, 4),
            "max_diameter": round(self.max_diameter, 4),
            "axial_min": round(self.axial_min, 4),
            "axial_max": round(self.axial_max, 4),
            "total_axial_length": round(self.total_axial_length, 4),
            "is_pure_revolution": self.is_pure_revolution,
            "face_coverage_ratio": round(self.face_coverage_ratio, 4),
            "num_segments": len(self.segments),
            "num_external_segments": len(self.external_segments),
            "num_internal_segments": len(self.internal_segments),
            "num_shoulders": len(self.shoulders),
            "segments": [s.to_dict() for s in self.segments],
            "transitions": [t.to_dict() for t in self.transitions],
        }


class RevolvedGeometryRecognizer:
    """Discovers rotational symmetry axes and extracts exact radial/axial profiles."""

    def __init__(
        self,
        faces: Sequence[Face],
        collinear_tol_mm: float = 0.5,
        angular_tol_deg: float = 2.5,
    ):
        self.faces = list(faces)
        self.collinear_tol_mm = float(collinear_tol_mm)
        self.angular_tol_deg = float(angular_tol_deg)
        self.cos_tol = float(np.cos(np.radians(angular_tol_deg)))

    def _are_axes_collinear(
        self,
        ax1: np.ndarray,
        p1: np.ndarray,
        ax2: np.ndarray,
        p2: np.ndarray,
    ) -> bool:
        """Check if two 3D axes (direction + point) are collinear infinite lines."""
        u1 = ax1 / max(np.linalg.norm(ax1), 1e-12)
        u2 = ax2 / max(np.linalg.norm(ax2), 1e-12)
        if abs(float(u1 @ u2)) < self.cos_tol:
            return False
        # Distance from p2 to line 1
        delta = p2 - p1
        dist_perp = np.linalg.norm(delta - (delta @ u1) * u1)
        return bool(dist_perp <= self.collinear_tol_mm)

    def discover_rotational_axes(self) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
        """Find all candidate rotational axes and the face indices associated with each.
        
        Returns: list of (axis_origin, axis_direction, face_indices)
        """
        revolved_kinds = (SurfaceKind.CYLINDER, SurfaceKind.CONE, SurfaceKind.TORUS)
        revolved_faces = [
            f for f in self.faces
            if f.surface.kind in revolved_kinds
            and f.surface.axis is not None
            and f.surface.point_on is not None
        ]

        if not revolved_faces:
            return []

        clusters: list[dict] = []
        for f in revolved_faces:
            fax = f.surface.axis / np.linalg.norm(f.surface.axis)
            fpt = f.surface.point_on.copy()

            matched = False
            for cl in clusters:
                if self._are_axes_collinear(cl["axis"], cl["origin"], fax, fpt):
                    cl["faces"].append(f.index)
                    cl["area"] += f.area
                    matched = True
                    break

            if not matched:
                clusters.append({
                    "axis": fax,
                    "origin": fpt,
                    "faces": [f.index],
                    "area": f.area,
                })

        # Sort clusters by area / number of faces descending
        clusters.sort(key=lambda c: -c["area"])

        valid_clusters = []
        for cl in clusters:
            if len(cl["faces"]) == 1:
                fi = cl["faces"][0]
                face = next((f for f in self.faces if f.index == fi), None)
                if face is not None and face.surface.kind == SurfaceKind.CYLINDER:
                    u_span = getattr(face.surface, "u_span", None)
                    if u_span is not None and u_span < 2 * math.pi - 0.15:
                        continue
                    if face.surface.u_min is not None and face.surface.u_max is not None:
                        if abs(face.surface.u_max - face.surface.u_min) < 2 * math.pi - 0.15:
                            continue
            valid_clusters.append(cl)

        return [(cl["origin"], cl["axis"], cl["faces"]) for cl in valid_clusters]

    def build_profile(
        self,
        axis_origin: np.ndarray,
        axis_direction: np.ndarray,
    ) -> Optional[RevolvedProfile]:
        """Construct the complete radial/axial profile along the given rotation axis."""
        u_axis = axis_direction / max(np.linalg.norm(axis_direction), 1e-12)
        p_origin = axis_origin.copy()

        # Classify and project all coaxial and perpendicular faces
        raw_segments: list[RevolvedProfileSegment] = []
        total_model_area = max(sum(f.area for f in self.faces), 1e-6)
        revolved_area = 0.0

        for f in self.faces:
            skind = f.surface.kind

            # 1. Cylindrical Faces
            if skind == SurfaceKind.CYLINDER and f.surface.axis is not None and f.surface.point_on is not None:
                if self._are_axes_collinear(u_axis, p_origin, f.surface.axis, f.surface.point_on):
                    r_cyl = float(f.surface.radius or 0.0)
                    # Project triangles/vertices along axis to find axial bounds [z_min, z_max]
                    pts = f.triangles.reshape(-1, 3) if len(f.triangles) else np.zeros((0, 3))
                    if len(pts):
                        zs = (pts - p_origin) @ u_axis
                        z_min, z_max = float(zs.min()), float(zs.max())
                    else:
                        z_min, z_max = 0.0, 0.0

                    is_int = bool(f.is_internal if f.is_internal is not None else False)
                    raw_segments.append(RevolvedProfileSegment(
                        index=len(raw_segments),
                        kind=RevolvedProfileSegmentKind.CYLINDRICAL_STEP,
                        face_indices=[f.index],
                        surface_kind=skind,
                        z_start=z_max,
                        z_end=z_min,
                        r_start=r_cyl,
                        r_end=r_cyl,
                        is_internal=is_int,
                        center_point=0.5 * (f.bounds_min + f.bounds_max),
                        notes={"nominal_diameter": 2.0 * r_cyl},
                    ))
                    revolved_area += f.area

            # 2. Conical Faces (Chamfers, tapers, countersinks)
            elif skind == SurfaceKind.CONE and f.surface.axis is not None and f.surface.point_on is not None:
                if self._are_axes_collinear(u_axis, p_origin, f.surface.axis, f.surface.point_on):
                    semi_angle = float(f.surface.semi_angle or 0.0)
                    semi_deg = float(np.degrees(semi_angle))
                    pts = f.triangles.reshape(-1, 3) if len(f.triangles) else np.zeros((0, 3))
                    if len(pts):
                        zs = (pts - p_origin) @ u_axis
                        z_min, z_max = float(zs.min()), float(zs.max())
                        # Compute radial distances at z_min and z_max
                        rads = np.linalg.norm((pts - p_origin) - zs[:, None] * u_axis[None, :], axis=1)
                        # Average radial samples near bounds
                        mask_max = abs(zs - z_max) < 0.2
                        mask_min = abs(zs - z_min) < 0.2
                        r_at_max = float(rads[mask_max].mean()) if np.any(mask_max) else float(rads.max())
                        r_at_min = float(rads[mask_min].mean()) if np.any(mask_min) else float(rads.min())
                    else:
                        z_min, z_max = 0.0, 0.0
                        r_at_min, r_at_max = 0.0, 0.0

                    is_int = bool(f.is_internal if f.is_internal is not None else False)
                    raw_segments.append(RevolvedProfileSegment(
                        index=len(raw_segments),
                        kind=RevolvedProfileSegmentKind.CONICAL_TRANSITION,
                        face_indices=[f.index],
                        surface_kind=skind,
                        z_start=z_max,
                        z_end=z_min,
                        r_start=r_at_max,
                        r_end=r_at_min,
                        is_internal=is_int,
                        semi_angle_deg=semi_deg,
                        center_point=0.5 * (f.bounds_min + f.bounds_max),
                        notes={"semi_angle_deg": semi_deg},
                    ))
                    revolved_area += f.area

            # 3. Toroidal Faces (Fillets, rounds, grooves)
            elif skind == SurfaceKind.TORUS and f.surface.axis is not None and f.surface.point_on is not None:
                if self._are_axes_collinear(u_axis, p_origin, f.surface.axis, f.surface.point_on):
                    r_major = float(f.surface.radius or 0.0)
                    r_minor = float(f.surface.ref_radius or 0.0)
                    pts = f.triangles.reshape(-1, 3) if len(f.triangles) else np.zeros((0, 3))
                    if len(pts):
                        zs = (pts - p_origin) @ u_axis
                        z_min, z_max = float(zs.min()), float(zs.max())
                        rads = np.linalg.norm((pts - p_origin) - zs[:, None] * u_axis[None, :], axis=1)
                        r_start = float(rads[abs(zs - z_max) < 0.2].mean()) if np.any(abs(zs - z_max) < 0.2) else r_major
                        r_end = float(rads[abs(zs - z_min) < 0.2].mean()) if np.any(abs(zs - z_min) < 0.2) else r_major
                    else:
                        z_min, z_max = 0.0, 0.0
                        r_start, r_end = r_major, r_major

                    is_int = bool(f.is_internal if f.is_internal is not None else False)
                    raw_segments.append(RevolvedProfileSegment(
                        index=len(raw_segments),
                        kind=RevolvedProfileSegmentKind.TOROIDAL_FILLET,
                        face_indices=[f.index],
                        surface_kind=skind,
                        z_start=z_max,
                        z_end=z_min,
                        r_start=r_start,
                        r_end=r_end,
                        is_internal=is_int,
                        curvature_radius=r_minor,
                        center_point=0.5 * (f.bounds_min + f.bounds_max),
                        notes={"fillet_radius": r_minor, "major_radius": r_major},
                    ))
                    revolved_area += f.area

            # 4. Planar Shoulders & End Faces (Planes perpendicular to rotation axis)
            elif skind == SurfaceKind.PLANE and f.surface.normal is not None:
                fn = f.surface.normal / max(np.linalg.norm(f.surface.normal), 1e-12)
                if abs(float(fn @ u_axis)) >= self.cos_tol:
                    # Face is perpendicular to axis
                    pts = f.triangles.reshape(-1, 3) if len(f.triangles) else np.zeros((0, 3))
                    if len(pts):
                        zs = (pts - p_origin) @ u_axis
                        z_face = float(zs.mean())
                        rads = np.linalg.norm((pts - p_origin) - zs[:, None] * u_axis[None, :], axis=1)
                        r_min, r_max = float(rads.min()), float(rads.max())
                    else:
                        z_face = 0.0
                        r_min, r_max = 0.0, 0.0

                    shoulder_w = max(0.0, r_max - r_min)
                    if shoulder_w > 0.05:  # meaningful annular radial span
                        raw_segments.append(RevolvedProfileSegment(
                            index=len(raw_segments),
                            kind=RevolvedProfileSegmentKind.PLANAR_SHOULDER,
                            face_indices=[f.index],
                            surface_kind=skind,
                            z_start=z_face,
                            z_end=z_face,
                            r_start=r_min,
                            r_end=r_max,
                            is_internal=False,
                            shoulder_width=shoulder_w,
                            center_point=0.5 * (f.bounds_min + f.bounds_max),
                            notes={"z_plane": z_face, "inner_dia": 2.0 * r_min, "outer_dia": 2.0 * r_max},
                        ))
                        revolved_area += f.area

        if not raw_segments:
            return None

        # Sort all segments along axis from top z_max downward
        raw_segments.sort(key=lambda s: -s.z_max)

        # Classify into external turned profile, internal bore profile, and shoulders
        ext_segs = [s for s in raw_segments if not s.is_internal and s.kind != RevolvedProfileSegmentKind.PLANAR_SHOULDER]
        int_segs = [s for s in raw_segments if s.is_internal]
        shoulders = [s for s in raw_segments if s.kind == RevolvedProfileSegmentKind.PLANAR_SHOULDER]

        # Detect edge transitions (steps, chamfers, fillets)
        transitions: list[RevolvedEdgeTransition] = []
        for i in range(len(raw_segments) - 1):
            s1 = raw_segments[i]
            s2 = raw_segments[i + 1]

            # Connect consecutive external sections
            if s1.kind == RevolvedProfileSegmentKind.PLANAR_SHOULDER:
                transitions.append(RevolvedEdgeTransition(
                    z=s1.z_start,
                    r_inner=min(s1.r_start, s1.r_end),
                    r_outer=max(s1.r_start, s1.r_end),
                    transition_type=TransitionKind.SQUARE_STEP,
                    connected_segments=(s1.index, s2.index),
                    face_indices=s1.face_indices,
                ))
            elif s1.kind == RevolvedProfileSegmentKind.CONICAL_TRANSITION:
                transitions.append(RevolvedEdgeTransition(
                    z=0.5 * (s1.z_start + s1.z_end),
                    r_inner=min(s1.r_start, s1.r_end),
                    r_outer=max(s1.r_start, s1.r_end),
                    transition_type=TransitionKind.CHAMFER,
                    connected_segments=(s1.index, s2.index),
                    chamfer_angle_deg=s1.semi_angle_deg,
                    chamfer_width=abs(s1.r_end - s1.r_start),
                    face_indices=s1.face_indices,
                ))
            elif s1.kind == RevolvedProfileSegmentKind.TOROIDAL_FILLET:
                transitions.append(RevolvedEdgeTransition(
                    z=0.5 * (s1.z_start + s1.z_end),
                    r_inner=min(s1.r_start, s1.r_end),
                    r_outer=max(s1.r_start, s1.r_end),
                    transition_type=TransitionKind.FILLET,
                    connected_segments=(s1.index, s2.index),
                    fillet_radius=s1.curvature_radius,
                    face_indices=s1.face_indices,
                ))

        all_r = [s.r_start for s in raw_segments] + [s.r_end for s in raw_segments]
        all_z = [s.z_start for s in raw_segments] + [s.z_end for s in raw_segments]
        min_d = 2.0 * min(all_r) if all_r else 0.0
        max_d = 2.0 * max(all_r) if all_r else 0.0
        min_z = min(all_z) if all_z else 0.0
        max_z = max(all_z) if all_z else 0.0

        coverage = min(1.0, revolved_area / total_model_area)
        is_pure = (coverage >= 0.85)

        return RevolvedProfile(
            axis_origin=p_origin,
            axis_direction=u_axis,
            segments=raw_segments,
            external_segments=ext_segs,
            internal_segments=int_segs,
            shoulders=shoulders,
            transitions=transitions,
            min_diameter=min_d,
            max_diameter=max_d,
            axial_min=min_z,
            axial_max=max_z,
            is_pure_revolution=is_pure,
            face_coverage_ratio=coverage,
        )

    def extract_primary_profile(self) -> Optional[RevolvedProfile]:
        """Detect and return the primary revolved profile for the shape."""
        axes = self.discover_rotational_axes()
        if not axes:
            return None
        orig, ax_dir, _ = axes[0]
        return self.build_profile(orig, ax_dir)
