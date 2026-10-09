"""Geometry-driven machining feature recognition.

Features are derived from actual B-Rep topology: surface kinds, oriented normals,
adjacency, boundary loops, coaxiality, concavity, and exact CAD dimensions.
Nothing is keyed on synthetic coordinates, names, or feature counts.
Every recognized feature carries semantic data, topological evidence, and provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .context import FinishRequirement, ToleranceModel
from .coords import CoordSpace, transform_points, transform_direction, transform_bounding_box
from .errors import CamError
from .geometry import Face, Shape, SurfaceKind, TriangleMesh
from .geometry.revolved import (
    RevolvedGeometryRecognizer,
    RevolvedProfile,
    RevolvedProfileSegment,
    RevolvedProfileSegmentKind,
    TransitionKind,
)

DEFAULT_TOOL_AXIS = np.array([0.0, 0.0, 1.0])


class FeatureType(str, Enum):
    HOLE = "hole"
    THROUGH_HOLE = "through_hole"
    BLIND_HOLE = "blind_hole"
    BORE = "bore"
    THROUGH_BORE = "through_bore"
    BLIND_BORE = "blind_bore"
    COUNTERBORE = "counterbore"
    COUNTERSINK = "countersink"
    TAPPED_HOLE = "tapped_hole"
    POCKET = "pocket"
    OPEN_POCKET = "open_pocket"
    SLOT = "slot"
    OPEN_SLOT = "open_slot"
    STEP = "step"
    SHOULDER = "shoulder"
    BOSS = "boss"
    FACING_REGION = "facing_region"
    PLANAR_WALL = "planar_wall"
    CONTOUR = "contour"
    FREEFORM_SURFACE = "freeform_surface"
    CHAMFER = "chamfer"
    FILLET = "fillet"
    THREADED_HOLE = "threaded_hole"
    GROOVE = "groove"


class Accessibility(str, Enum):
    TOOL_AXIS_OK = "tool_axis_ok"
    NEEDS_TILTED_TOOL = "needs_tilted_tool"
    INACCESSIBLE = "inaccessible"


@dataclass
class MachiningFeature:
    """Semantic machining feature extracted from real B-Rep geometry in canonical or setup space."""

    id: str
    type: FeatureType
    face_indices: list[int]
    bounds_min: np.ndarray            # coordinate space bounds
    bounds_max: np.ndarray
    depth: float                      # machining depth below local top surface (invariant)
    top_z: float                      # Z of material top over the feature
    floor_z: float                    # Z of feature floor (for pockets/holes/steps)
    axis: Optional[np.ndarray] = None # feature axis (holes/bores/bosses)
    diameter: Optional[float] = None  # holes/bores/bosses (invariant)
    radius: Optional[float] = None    # fillets/corners (invariant)
    is_concave: bool = True           # pockets/holes concave; bosses convex
    accessibility: Accessibility = Accessibility.TOOL_AXIS_OK
    machining_direction: np.ndarray = field(default_factory=lambda: DEFAULT_TOOL_AXIS.copy())
    wall_faces: list[int] = field(default_factory=list)
    floor_faces: list[int] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    parent_feature_id: Optional[str] = None
    child_feature_ids: list[str] = field(default_factory=list)
    recognition_evidence: list[str] = field(default_factory=list)
    source_edges: list[int] = field(default_factory=list)
    finish: Optional[FinishRequirement] = None
    notes: dict = field(default_factory=dict)
    boundary_loop: Optional[list[np.ndarray]] = None
    island_loops: list[list[np.ndarray]] = field(default_factory=list)
    canonical_id: Optional[str] = None
    space: CoordSpace = CoordSpace.MODEL

    # Detailed CAD Feature-Geometry Extraction Layer
    surface_type: Optional[str] = None          # plane | cylinder | cone | torus | bspline | freeform
    center: Optional[np.ndarray] = None         # exact CAD 3D center / origin point
    axial_start: Optional[float] = None         # exact CAD axial start coordinate along axis / Z
    axial_end: Optional[float] = None           # exact CAD axial end coordinate along axis / Z
    fillet_radius: Optional[float] = None       # exact CAD blend / fillet radius
    orientation: Optional[np.ndarray] = None    # surface normal or tool axis orientation vector
    tolerances: Optional[ToleranceModel] = None # CAD tolerance model
    source_geometry_reference: Optional[str] = None # provenance reference (e.g. Face#42:CYLINDER:Ø6.000)

    @property
    def face_ids(self) -> list[int]:
        """Alias for face_indices reflecting CAD B-Rep face IDs."""
        return self.face_indices

    @property
    def edge_ids(self) -> list[int]:
        """Alias for source_edges reflecting CAD B-Rep edge IDs."""
        return self.source_edges

    @property
    def diameter_or_radius(self) -> Optional[float]:
        """Nominal CAD diameter or radius."""
        return self.diameter if self.diameter is not None else self.radius

    @property
    def center_xy(self) -> np.ndarray:
        if self.center is not None:
            return self.center[:2]
        return 0.5 * (self.bounds_min[:2] + self.bounds_max[:2])

    @property
    def xy_extent(self) -> np.ndarray:
        return self.bounds_max[:2] - self.bounds_min[:2]

    @property
    def design_geometry(self):
        """Nominal CAD Design Geometry (immutable CAD representation)."""
        from .geometry import (
            DesignGeometry,
            NominalCadSurface,
            NominalCadBoundary,
            CadSurfaceKind,
        )
        kind_str = (self.surface_type or "other").lower()
        try:
            kind_enum = CadSurfaceKind(kind_str)
        except ValueError:
            kind_enum = CadSurfaceKind.OTHER

        surf = NominalCadSurface(
            kind=kind_enum,
            face_ids=list(self.face_indices or []),
            diameter=self.diameter,
            radius=self.radius or (self.diameter / 2.0 if self.diameter is not None else None),
            axis=self.axis.copy() if self.axis is not None else None,
            center=self.center.copy() if self.center is not None else None,
            normal=self.orientation.copy() if self.orientation is not None else None,
            axial_start=self.axial_start if self.axial_start is not None else self.top_z,
            axial_end=self.axial_end if self.axial_end is not None else self.floor_z,
            depth=float(self.depth) if self.depth is not None else 0.0,
            fillet_radius=self.fillet_radius,
            is_internal=not (self.type == FeatureType.BOSS or not self.is_concave),
            source_reference=self.source_geometry_reference or f"Feature:{self.id}",
            tolerances=self.tolerances,
        )

        bound = None
        if self.boundary_loop:
            bound_arr = np.asarray(self.boundary_loop, dtype=float)
            islands = [np.asarray(isl, dtype=float) for isl in (self.island_loops or [])]
            bound = NominalCadBoundary(
                outer_loop=bound_arr,
                island_loops=islands,
                edge_ids=list(self.source_edges or []),
                z_level=self.floor_z,
            )


        dims = {"depth": float(self.depth)}
        if self.diameter is not None:
            dims["diameter"] = float(self.diameter)
        if self.radius is not None:
            dims["radius"] = float(self.radius)
        if self.fillet_radius is not None:
            dims["fillet_radius"] = float(self.fillet_radius)

        return DesignGeometry(
            surfaces=[surf],
            boundaries=[bound] if bound else [],
            primary_surface=surf,
            primary_boundary=bound,
            nominal_dimensions=dims,
            space=self.space,
        )

    def compute_manufacturing_geometry(
        self,
        tool_radius: float,
        radial_allowance: float = 0.0,
        axial_allowance: float = 0.0,
        operation_id: Optional[str] = None,
        tool_id: Optional[str] = None,
        tool_length: float = 50.0,
    ):
        """Compute the corresponding ManufacturingGeometry via explicit offset transformation."""
        from .geometry import ManufacturingOffsetCalculator
        return ManufacturingOffsetCalculator.build_manufacturing_geometry(
            design_geometry=self.design_geometry,
            tool_radius=tool_radius,
            radial_allowance=radial_allowance,
            axial_allowance=axial_allowance,
            feature_id=self.id,
            operation_id=operation_id,
            tool_id=tool_id,
            tool_length=tool_length,
            space=self.space,
        )

    def transformed(
        self,
        transform_matrix: np.ndarray,
        dst_space: CoordSpace = CoordSpace.SETUP,
        new_id: Optional[str] = None,
    ) -> MachiningFeature:
        """Rigidly map a canonical feature into setup space without altering its nominal geometry."""
        t = np.asarray(transform_matrix, dtype=float)
        new_bmin, new_bmax = transform_bounding_box(self.bounds_min, self.bounds_max, t)
        new_axis = transform_direction(self.axis, t) if self.axis is not None else None
        new_mach_dir = transform_direction(self.machining_direction, t)
        new_center = transform_points(self.center.reshape(1, 3), t)[0] if self.center is not None else None
        new_orientation = transform_direction(self.orientation, t) if self.orientation is not None else None

        new_boundary_loop = None
        if self.boundary_loop:
            new_boundary_loop = [transform_points(np.asarray(p, dtype=float), t) for p in self.boundary_loop]

        new_island_loops = []
        if self.island_loops:
            new_island_loops = [
                [transform_points(np.asarray(p, dtype=float), t) for p in loop]
                for loop in self.island_loops
            ]

        # Project reference center point to obtain transformed top and floor Z levels
        mid_xy = self.center_xy
        top_pt = transform_points(np.array([[mid_xy[0], mid_xy[1], self.top_z]]), t)[0]
        floor_pt = transform_points(np.array([[mid_xy[0], mid_xy[1], self.floor_z]]), t)[0]

        # In setup space, tool approaches along +Z towards -Z.
        # Thus the entry/material top is the higher Z, and the floor is the lower Z.
        z1 = float(top_pt[2])
        z2 = float(floor_pt[2])
        setup_top_z = max(z1, z2)
        setup_floor_z = min(z1, z2)

        feat_id = new_id or self.id
        canon_id = self.canonical_id or self.id

        return MachiningFeature(
            id=feat_id,
            canonical_id=canon_id,
            space=dst_space,
            type=self.type,
            face_indices=list(self.face_indices),
            bounds_min=new_bmin,
            bounds_max=new_bmax,
            depth=float(self.depth),
            top_z=setup_top_z,
            floor_z=setup_floor_z,
            axis=new_axis,
            diameter=self.diameter,
            radius=self.radius,
            is_concave=self.is_concave,
            accessibility=self.accessibility,
            machining_direction=new_mach_dir,
            wall_faces=list(self.wall_faces),
            floor_faces=list(self.floor_faces),
            dependencies=list(self.dependencies),
            parent_feature_id=self.parent_feature_id,
            child_feature_ids=list(self.child_feature_ids),
            recognition_evidence=list(self.recognition_evidence),
            source_edges=list(self.source_edges),
            finish=self.finish,
            notes=dict(self.notes),
            boundary_loop=new_boundary_loop,
            island_loops=new_island_loops,
            surface_type=self.surface_type,
            center=new_center,
            axial_start=setup_top_z if self.axial_start is not None else None,
            axial_end=setup_floor_z if self.axial_end is not None else None,
            fillet_radius=self.fillet_radius,
            orientation=new_orientation,
            tolerances=self.tolerances,
            source_geometry_reference=self.source_geometry_reference,
        )



class FeatureRecognizer:
    """Extracts relational machining features from a transformed B-Rep shape in setup space."""

    def __init__(
        self,
        shape: Shape,
        stock_top_z: float,
        tool_axis: Optional[np.ndarray] = None,
        requirements: Optional[dict[int, FinishRequirement]] = None,
        vertical_tol_deg: float = 12.0,
        horizontal_tol_deg: float = 12.0,
        drill_diameters: Optional[list[float]] = None,
    ):
        self.shape = shape
        self.faces = shape.faces()
        self.edges = shape.edges()
        self.adjacency = shape.face_adjacency()
        self.stock_top_z = float(stock_top_z)
        self.tool_axis = (
            tool_axis / np.linalg.norm(tool_axis)
            if tool_axis is not None
            else DEFAULT_TOOL_AXIS.copy()
        )
        self.requirements = requirements or {}
        self._vertical_tol_deg = vertical_tol_deg
        self._horizontal_tol_deg = horizontal_tol_deg
        self._drill_diameters: Optional[list[float]] = sorted(drill_diameters) if drill_diameters else None
        self._model_top_z = self._compute_model_top_z()
        self._tags = self._tag_faces()

    def _edges_for_faces(self, face_ids: list[int]) -> list[int]:
        """Extract sorted list of unique B-Rep edge indices belonging to the given faces."""
        edges: list[int] = []
        for fid in face_ids:
            if 0 <= fid < len(self.faces):
                for eid in self.faces[fid].edge_indices:
                    if eid not in edges:
                        edges.append(eid)
        return sorted(edges)

    def _tolerance_for_faces(self, face_ids: list[int]) -> Optional[ToleranceModel]:
        """Extract CAD tolerance model for the given faces if present in requirements."""
        for fid in face_ids:
            req = self.requirements.get(fid)
            if req is not None:
                if getattr(req, "tolerance", None) is not None:
                    return req.tolerance
                if getattr(req, "tolerance_mm", None) is not None:
                    return ToleranceModel(upper_tol_mm=float(req.tolerance_mm), lower_tol_mm=-float(req.tolerance_mm))
        return None

    def _compute_model_top_z(self) -> float:
        valid_faces = [f for f in self.faces if len(f.triangles)]
        if not valid_faces:
            return 0.0
        return max(float(f.triangles[:, :, 2].max()) for f in valid_faces)

    def _face_u_span(self, f: Face) -> float:
        """Return angular span of face in radians (exact from CAD or estimated from mesh)."""
        if f.surface.u_span is not None:
            return float(f.surface.u_span)
        if f.surface.u_min is not None and f.surface.u_max is not None:
            return float(abs(f.surface.u_max - f.surface.u_min))
        if len(f.triangles) and f.surface.point_on is not None:
            pts = f.triangles.reshape(-1, 3)
            c = f.surface.point_on[:2]
            angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            sorted_a = np.sort(np.unique(np.round(angles, 3)))
            if len(sorted_a) >= 2:
                gaps = np.diff(sorted_a)
                wrap_gap = (2 * np.pi + sorted_a[0]) - sorted_a[-1]
                max_gap = max(float(gaps.max()), float(wrap_gap))
                expected_facet_gap = 2 * np.pi / max(len(sorted_a), 1)
                if max_gap > max(expected_facet_gap * 1.6, 0.25):
                    return float(max(0.1, 2 * np.pi - max_gap))
                else:
                    return float(2 * np.pi)
        return float(2 * np.pi)

    def _tag_faces(self) -> dict[int, str]:
        """Classify each face by orientation relative to setup tool axis."""
        tags: dict[int, str] = {}
        cos_v = np.cos(np.radians(self._vertical_tol_deg))
        cos_h = np.cos(np.radians(90.0 - self._horizontal_tol_deg))

        for f in self.faces:
            if f.surface.kind == SurfaceKind.CYLINDER:
                u_span = self._face_u_span(f)
                if u_span >= 2 * np.pi - 0.15:
                    tags[f.index] = "cylinder"
                    continue
                # If partial cylinder (corner fillet / round), classify by axis orientation
                if f.surface.axis is not None:
                    axis = f.surface.axis / max(np.linalg.norm(f.surface.axis), 1e-12)
                    dot_axis = abs(float(axis @ self.tool_axis))
                    if dot_axis >= cos_v:
                        tags[f.index] = "wall"
                    else:
                        tags[f.index] = "slanted"
                else:
                    tags[f.index] = "wall"
                continue
            if f.surface.kind == SurfaceKind.CONE:
                tags[f.index] = "cone"
                continue

            n = f.oriented_normal / max(np.linalg.norm(f.oriented_normal), 1e-12)
            dot_up = float(n @ self.tool_axis)

            if dot_up >= cos_v:
                # Planar face pointing along tool axis (upward in setup)
                # Check if it's the top face or an internal floor face below top
                face_max_z = float(f.triangles[:, :, 2].max()) if len(f.triangles) else 0.0
                if abs(face_max_z - self._model_top_z) < 0.5:
                    tags[f.index] = "top"
                else:
                    tags[f.index] = "floor"
            elif dot_up <= -cos_v:
                tags[f.index] = "bottom"
            elif abs(dot_up) <= cos_h:
                tags[f.index] = "wall"
            else:
                tags[f.index] = "slanted"
        return tags

    def recognize(self) -> list[MachiningFeature]:
        """Execute full feature recognition pipeline and build relational hierarchy."""
        features: list[MachiningFeature] = []
        features += self._recognize_cylindrical_features()
        features += self._recognize_countersinks(features)  # must follow cylindrical
        features += self._recognize_pockets_and_slots()
        features += self._recognize_facing_regions()
        features += self._recognize_chamfers()
        features += self._recognize_fillets()
        features += self._recognize_freeform()
        features += self._recognize_threaded_holes(features)
        features += self._recognize_grooves(features)
        # Revolved/turned geometry recognition — for pure-revolution parts, the analytical
        # profile provides the exact geometry hierarchy and suppresses generic outer contour.
        # For mixed/prismatic parts, it provides supplementary revolved features without duplicating claimed faces.
        revolved_features, is_pure_revolution = self._recognize_revolved_profile()
        existing_faces = {fi for feat in features for fi in feat.face_indices}
        for rf in revolved_features:
            if rf.type == FeatureType.CONTOUR:
                if is_pure_revolution:
                    features.append(rf)
                continue
            if not any(fi in existing_faces for fi in rf.face_indices):
                features.append(rf)
                existing_faces.update(rf.face_indices)
        if not is_pure_revolution:
            features += self._recognize_outer_contours()
        return self._dedupe_and_sort(features)

    def _classify_hole_or_bore(self, diameter: float, is_through: bool) -> FeatureType:
        """Classify an internal cylinder as a hole (drillable) or bore (requires endmill /
        boring bar). When a drill_diameters list is provided, match against it;
        otherwise fall back to a 12mm diameter threshold (traditional shop practice)."""
        if self._drill_diameters is not None:
            can_drill = any(abs(d - diameter) < 0.15 for d in self._drill_diameters)
            if can_drill:
                return FeatureType.THROUGH_HOLE if is_through else FeatureType.BLIND_HOLE
            else:
                return FeatureType.THROUGH_BORE if is_through else FeatureType.BLIND_BORE
        # Fallback: diameter threshold (12mm is the traditional max twist-drill size in
        # most general-purpose shops; can be overridden by passing drill_diameters)
        bore_threshold = 12.0
        if diameter > bore_threshold:
            return FeatureType.THROUGH_BORE if is_through else FeatureType.BLIND_BORE
        return FeatureType.THROUGH_HOLE if is_through else FeatureType.BLIND_HOLE

    # -------------------------------------------------------------------------
    def _recognize_cylindrical_features(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        cyl_groups = self._group_cylinder_faces()

        # Step 1: Process each individual cylinder group aligned with tool axis
        cyl_info: list[dict] = []
        transverse_cyl_groups: list[list[int]] = []

        for g in cyl_groups:
            faces = [self.faces[i] for i in g]
            f0 = faces[0]
            sd = f0.surface
            axis = sd.axis / np.linalg.norm(sd.axis)
            axis_alignment = abs(float(axis @ self.tool_axis))

            if axis_alignment < 0.7:
                # Transverse/horizontal feature (needs tilted tool / side setup)
                transverse_cyl_groups.append(g)
                continue

            # Ensure axis points in setup tool axis direction
            if axis @ self.tool_axis < 0:
                axis = -axis

            radius = float(sd.radius)
            zs = np.concatenate([f.triangles[:, :, 2].ravel() for f in faces if len(f.triangles)])
            if not len(zs):
                continue
            top_z = float(zs.max())
            bottom_z = float(zs.min())
            depth = top_z - bottom_z
            if depth <= 1e-4:
                continue

            center_xy = np.array([sd.point_on[0], sd.point_on[1]])

            # Check internal void vs external boss
            is_internal = f0.is_internal
            if is_internal is None:
                # Evaluate via radial vector
                fc = f0.center[:2] - sd.point_on[:2]
                is_internal = bool(float(f0.oriented_normal[:2] @ fc) < -1e-4)

            total_u_span = sum(self._face_u_span(f) for f in faces)
            is_full_circle = total_u_span >= 2 * np.pi - 0.15
            is_interrupted_bore = is_internal and (total_u_span >= np.pi - 0.25)

            # Partial internal cylinders (corner fillets, open arcs < 155 deg) are not holes/bores
            if is_internal and not (is_full_circle or is_interrupted_bore):
                continue

            cyl_info.append({
                "group": g,
                "faces": faces,
                "axis": axis,
                "radius": radius,
                "diameter": 2.0 * radius,
                "top_z": top_z,
                "bottom_z": bottom_z,
                "depth": depth,
                "center_xy": center_xy,
                "is_internal": is_internal,
                "is_interrupted": is_interrupted_bore and not is_full_circle,
                "total_u_span": total_u_span,
            })

        # Step 2: Classify bosses directly
        for info in cyl_info:
            if not info["is_internal"]:
                # Partial external rounds / corner fillets (< 165 deg) are corner rounds, not standalone bosses
                if info["total_u_span"] < np.pi - 0.25:
                    continue
                bmin = np.array([info["center_xy"][0] - info["radius"], info["center_xy"][1] - info["radius"], info["bottom_z"]])

                bmax = np.array([info["center_xy"][0] + info["radius"], info["center_xy"][1] + info["radius"], info["top_z"]])
                cyl_center = np.array([info["center_xy"][0], info["center_xy"][1], 0.5 * (info["top_z"] + info["bottom_z"])])
                out.append(MachiningFeature(
                    id=f"boss_{info['group'][0]}",
                    type=FeatureType.BOSS,
                    face_indices=info["group"],
                    bounds_min=bmin,
                    bounds_max=bmax,
                    depth=info["depth"],
                    top_z=info["top_z"],
                    floor_z=info["bottom_z"],
                    axis=info["axis"].copy(),
                    diameter=info["diameter"],
                    radius=info["radius"],
                    is_concave=False,
                    accessibility=self._accessibility_of(info["group"]),
                    wall_faces=info["group"],
                    source_edges=self._edges_for_faces(info["group"]),
                    finish=self.requirements.get(info["group"][0]),
                    tolerances=self._tolerance_for_faces(info["group"]),
                    recognition_evidence=["external_cylindrical_surface", f"radius_{info['radius']:.2f}"],
                    surface_type="cylinder",
                    center=cyl_center,
                    axial_start=info["top_z"],
                    axial_end=info["bottom_z"],
                    orientation=info["axis"].copy(),
                    source_geometry_reference=f"Face#{info['group'][0]}:CYLINDER:Ø{info['diameter']:.3f}",
                ))

        # Step 3: Group coaxial internal cylinders to recognize counterbores / stepped bores
        internal_cyls = [c for c in cyl_info if c["is_internal"]]
        coaxial_clusters: list[list[dict]] = []
        used_internal: set[int] = set()

        for i, c0 in enumerate(internal_cyls):
            if i in used_internal:
                continue
            cluster = [c0]
            used_internal.add(i)
            # Expand cluster transitively for any adjacent connected cylinder
            added = True
            while added:
                added = False
                for j, c2 in enumerate(internal_cyls):
                    if j in used_internal:
                        continue
                    same_axis = (np.linalg.norm(c0["center_xy"] - c2["center_xy"]) < 0.5 and
                                 abs(abs(c0["axis"] @ c2["axis"]) - 1.0) < 1e-3)
                    if same_axis:
                        # Check if c2 is connected to any cylinder already in cluster (no solid web between them)
                        is_connected = False
                        for member in cluster:
                            upper = member if member["top_z"] >= c2["top_z"] else c2
                            lower = c2 if member["top_z"] >= c2["top_z"] else member
                            z_gap = upper["bottom_z"] - lower["top_z"]
                            if abs(z_gap) < 0.1 or z_gap < 0:
                                is_connected = True
                                break
                        if is_connected:
                            cluster.append(c2)
                            used_internal.add(j)
                            added = True
            # Sort cluster from top Z downward
            cluster.sort(key=lambda x: -x["top_z"])
            coaxial_clusters.append(cluster)

        for cluster in coaxial_clusters:
            if len(cluster) == 1:
                # Single internal cylinder: hole or bore
                c = cluster[0]
                bmin = np.array([c["center_xy"][0] - c["radius"], c["center_xy"][1] - c["radius"], c["bottom_z"]])
                bmax = np.array([c["center_xy"][0] + c["radius"], c["center_xy"][1] + c["radius"], c["top_z"]])
                cyl_center = np.array([c["center_xy"][0], c["center_xy"][1], 0.5 * (c["top_z"] + c["bottom_z"])])

                # Check through vs blind: does it reach model bottom?
                model_bottom_z = min(float(f.bounds_min[2]) for f in self.faces if len(f.triangles))
                is_through = c["bottom_z"] <= model_bottom_z + 0.5
                ftype = self._classify_hole_or_bore(c["diameter"], is_through)

                top_pts = [np.array([c["center_xy"][0], c["center_xy"][1], c["top_z"]])]
                for angle in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
                    top_pts.append(np.array([
                        c["center_xy"][0] + c["radius"] * 0.5 * np.cos(angle),
                        c["center_xy"][1] + c["radius"] * 0.5 * np.sin(angle),
                        c["top_z"]
                    ]))
                acc = self._accessibility_of(c["group"], top_sample_pts=top_pts)

                out.append(MachiningFeature(
                    id=f"hole_{c['group'][0]}",
                    type=ftype,
                    face_indices=c["group"],
                    bounds_min=bmin,
                    bounds_max=bmax,
                    depth=c["depth"],
                    top_z=c["top_z"],
                    floor_z=c["bottom_z"],
                    axis=c["axis"].copy(),
                    diameter=c["diameter"],
                    radius=c["radius"],
                    is_concave=True,
                    accessibility=acc,
                    wall_faces=c["group"],
                    source_edges=self._edges_for_faces(c["group"]),
                    finish=self.requirements.get(c["group"][0]),
                    tolerances=self._tolerance_for_faces(c["group"]),
                    notes={
                        "is_interrupted": c.get("is_interrupted", False),
                        "u_span_deg": float(np.degrees(c.get("total_u_span", 2 * np.pi))),
                    },
                    recognition_evidence=[
                        "internal_cylinder",
                        f"diameter_{c['diameter']:.2f}",
                        "interrupted_bore" if c.get("is_interrupted") else "full_cylinder",
                        "through_opening" if is_through else "blind_bottom",
                    ],
                    surface_type="cylinder",
                    center=cyl_center,
                    axial_start=c["top_z"],
                    axial_end=c["bottom_z"],
                    orientation=c["axis"].copy(),
                    source_geometry_reference=f"Face#{c['group'][0]}:CYLINDER:Ø{c['diameter']:.3f}",
                ))
            else:
                # Multi-tier coaxial cylinders (e.g. Counterbore -> Through Bore -> Bottom Counterbore)
                cluster.sort(key=lambda x: -x["top_z"])
                min_dia = min(c["diameter"] for c in cluster)
                model_bottom_z = min(float(f.bounds_min[2]) for f in self.faces if len(f.triangles))

                cluster_feats: list[MachiningFeature] = []
                for k, c in enumerate(cluster):
                    bmin_k = np.array([c["center_xy"][0] - c["radius"], c["center_xy"][1] - c["radius"], c["bottom_z"]])
                    bmax_k = np.array([c["center_xy"][0] + c["radius"], c["center_xy"][1] + c["radius"], c["top_z"]])
                    cyl_center = np.array([c["center_xy"][0], c["center_xy"][1], 0.5 * (c["top_z"] + c["bottom_z"])])
                    is_main_bore = abs(c["diameter"] - min_dia) < 1e-3
                    is_through = c["bottom_z"] <= model_bottom_z + 0.5

                    if is_main_bore:
                        ftype = self._classify_hole_or_bore(c["diameter"], is_through)
                    else:
                        ftype = FeatureType.COUNTERBORE

                    top_pts = [np.array([c["center_xy"][0], c["center_xy"][1], c["top_z"]])]
                    for angle in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
                        top_pts.append(np.array([
                            c["center_xy"][0] + c["radius"] * 0.5 * np.cos(angle),
                            c["center_xy"][1] + c["radius"] * 0.5 * np.sin(angle),
                            c["top_z"]
                        ]))
                    acc = self._accessibility_of(c["group"], top_sample_pts=top_pts)

                    feat = MachiningFeature(
                        id=f"{'cbore' if ftype == FeatureType.COUNTERBORE else 'hole'}_{c['group'][0]}",
                        type=ftype,
                        face_indices=c["group"],
                        bounds_min=bmin_k,
                        bounds_max=bmax_k,
                        depth=c["depth"],
                        top_z=c["top_z"],
                        floor_z=c["bottom_z"],
                        axis=c["axis"].copy(),
                        diameter=c["diameter"],
                        radius=c["radius"],
                        is_concave=True,
                        accessibility=acc,
                        wall_faces=c["group"],
                        source_edges=self._edges_for_faces(c["group"]),
                        finish=self.requirements.get(c["group"][0]),
                        tolerances=self._tolerance_for_faces(c["group"]),
                        notes={
                            "is_interrupted": c.get("is_interrupted", False),
                            "u_span_deg": float(np.degrees(c.get("total_u_span", 2 * np.pi))),
                            "tier_index": k,
                            "tier_count": len(cluster),
                        },
                        recognition_evidence=[
                            "coaxial_compound_feature",
                            f"diameter_{c['diameter']:.2f}",
                            f"tier_{k+1}_of_{len(cluster)}",
                        ],
                        surface_type="cylinder",
                        center=cyl_center,
                        axial_start=c["top_z"],
                        axial_end=c["bottom_z"],
                        orientation=c["axis"].copy(),
                        source_geometry_reference=f"Face#{c['group'][0]}:CYLINDER:Ø{c['diameter']:.3f}",
                    )
                    cluster_feats.append(feat)

                # Link parent/child relationships
                for k, feat in enumerate(cluster_feats):
                    if k == 0:
                        feat.child_feature_ids = [cf.id for cf in cluster_feats[1:]]
                    else:
                        feat.parent_feature_id = cluster_feats[0].id
                out.extend(cluster_feats)

        # Step 4: Classify transverse / horizontal cross-holes (axis perpendicular to setup tool axis)
        for g in transverse_cyl_groups:
            faces = [self.faces[i] for i in g]
            f0 = faces[0]
            sd = f0.surface
            axis = sd.axis / np.linalg.norm(sd.axis)

            is_internal = f0.is_internal
            if is_internal is None:
                fc = f0.center - sd.point_on
                is_internal = bool(float(f0.oriented_normal @ fc) < -1e-4)

            total_u_span = sum(self._face_u_span(f) for f in faces)
            if is_internal and total_u_span < np.pi - 0.25:
                continue

            all_pts = np.concatenate([f.triangles.reshape(-1, 3) for f in faces if len(f.triangles)])
            if not len(all_pts):
                continue
            bmin = all_pts.min(axis=0)
            bmax = all_pts.max(axis=0)
            dia = 2.0 * float(sd.radius)
            length = float(np.linalg.norm(bmax - bmin))

            ftype = FeatureType.THROUGH_HOLE if is_internal else FeatureType.BOSS
            sugg_rot = [0.0, 90.0, 0.0] if abs(axis[0]) > abs(axis[1]) else [90.0, 0.0, 0.0]
            out.append(MachiningFeature(
                id=f"side_hole_{g[0]}",
                type=ftype,
                face_indices=g,
                bounds_min=bmin,
                bounds_max=bmax,
                depth=length,
                top_z=float(bmax[2]),
                floor_z=float(bmin[2]),
                axis=axis.copy(),
                diameter=dia,
                radius=float(sd.radius),
                is_concave=is_internal,
                accessibility=Accessibility.NEEDS_TILTED_TOOL,
                wall_faces=g,
                source_edges=self._edges_for_faces(g),
                tolerances=self._tolerance_for_faces(g),
                notes={
                    "is_side_feature": True,
                    "hole_axis": axis.tolist(),
                    "suggested_setup_rotation": sugg_rot,
                },
                recognition_evidence=[
                    "transverse_cylindrical_feature",
                    f"axis_{[round(float(x), 2) for x in axis]}",
                    f"diameter_{dia:.2f}",
                ],
                surface_type="cylinder",
                center=0.5 * (bmin + bmax),
                axial_start=float(bmax[2]),
                axial_end=float(bmin[2]),
                orientation=axis.copy(),
                source_geometry_reference=f"Face#{g[0]}:TRANSVERSE_CYLINDER:Ø{dia:.3f}",
            ))

        return out

    # -------------------------------------------------------------------------
    # Countersink recognition (CONE faces coaxial with holes)
    # -------------------------------------------------------------------------
    def _recognize_countersinks(self, existing: list[MachiningFeature]) -> list[MachiningFeature]:
        """Find CONE faces that form countersinks above existing hole features.
        A countersink is a conical chamfer at a hole entry:
        - SurfaceKind.CONE with semi_angle in [10°, 70°] (covers 82°, 90°, 118°, 120° tools)
        - Cone axis parallel to the tool axis (within 15°)
        - Cone centre XY coincides with an existing hole feature (within 1.5 mm)"""
        out: list[MachiningFeature] = []
        for cf in self.faces:
            if cf.surface.kind is not SurfaceKind.CONE:
                continue
            sd = cf.surface
            if sd.axis is None or sd.semi_angle is None or sd.point_on is None:
                continue
            cone_axis = sd.axis / max(float(np.linalg.norm(sd.axis)), 1e-12)
            # Cone axis must be coaxial with tool axis (up to 15° deviation)
            if abs(abs(float(cone_axis @ self.tool_axis)) - 1.0) > 0.26:  # cos(15°)≈0.97 tolerance
                continue
            semi_deg = float(np.degrees(abs(sd.semi_angle)))
            if not (10.0 < semi_deg < 70.0):
                continue
            if not len(cf.triangles):
                continue
            zs = cf.triangles[:, :, 2].ravel()
            top_z = float(zs.max())
            bottom_z = float(zs.min())
            depth = top_z - bottom_z
            if depth < 0.05:
                continue
            ref_r = float(abs(sd.ref_radius)) if sd.ref_radius is not None else 0.0
            major_r = ref_r + depth * float(np.tan(float(abs(sd.semi_angle))))
            major_dia = 2.0 * major_r
            if major_dia < 0.5:
                continue
            centre_xy = sd.point_on[:2].copy()
            # Find parent hole (nearest coaxial internal cylindrical feature)
            parent_id: Optional[str] = None
            hole_types = (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
                          FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
                          FeatureType.COUNTERBORE)
            for ef in existing:
                if ef.type not in hole_types:
                    continue
                if float(np.linalg.norm(ef.center_xy - centre_xy)) < 1.5:
                    parent_id = ef.id
                    break
            if parent_id is None:
                continue
            csink_id = f"csink_{cf.index}"
            csink_center = np.array([centre_xy[0], centre_xy[1], 0.5 * (top_z + bottom_z)])
            out.append(MachiningFeature(
                id=csink_id,
                type=FeatureType.COUNTERSINK,
                face_indices=[cf.index],
                bounds_min=np.array([centre_xy[0] - major_r, centre_xy[1] - major_r, bottom_z]),
                bounds_max=np.array([centre_xy[0] + major_r, centre_xy[1] + major_r, top_z]),
                depth=depth,
                top_z=top_z,
                floor_z=bottom_z,
                axis=cone_axis.copy(),
                diameter=major_dia,
                is_concave=True,
                accessibility=Accessibility.TOOL_AXIS_OK,
                parent_feature_id=parent_id,
                source_edges=self._edges_for_faces([cf.index]),
                finish=self.requirements.get(cf.index),
                tolerances=self._tolerance_for_faces([cf.index]),
                recognition_evidence=[
                    f"cone_semi_angle_{semi_deg:.1f}deg",
                    f"included_angle_{2 * semi_deg:.1f}deg",
                    f"major_diameter_{major_dia:.2f}mm",
                ],
                notes={
                    "semi_angle_deg": semi_deg,
                    "included_angle_deg": 2.0 * semi_deg,
                    "countersink_angle_deg": 90.0 - semi_deg,
                },
                surface_type="cone",
                center=csink_center,
                axial_start=top_z,
                axial_end=bottom_z,
                orientation=cf.oriented_normal.copy() if cf.oriented_normal is not None else cone_axis.copy(),
                source_geometry_reference=f"Face#{cf.index}:CONE:ANG{semi_deg:.1f}deg:Ø{major_dia:.3f}",
            ))
        return out

    def _group_cylinder_faces(self) -> list[list[int]]:
        """Group adjacent cylinder faces sharing an axis line."""
        cyl = [f for f in self.faces if f.surface.kind is SurfaceKind.CYLINDER]
        groups: list[list[int]] = []
        used: set[int] = set()

        for f in cyl:
            if f.index in used:
                continue
            group = [f.index]
            used.add(f.index)
            axis = f.surface.axis / np.linalg.norm(f.surface.axis)
            center = f.surface.point_on
            radius = f.surface.radius

            stack = [f.index]
            while stack:
                cur = stack.pop()
                for nbr in self.adjacency.get(cur, []):
                    if nbr in used or nbr < 0 or nbr >= len(self.faces):
                        continue
                    nf = self.faces[nbr]
                    if nf.surface.kind is not SurfaceKind.CYLINDER:
                        continue
                    naxis = nf.surface.axis / np.linalg.norm(nf.surface.axis)
                    if abs(abs(naxis @ axis) - 1.0) > 1e-4:
                        continue
                    if abs(nf.surface.radius - radius) > 1e-4:
                        continue
                    d = nf.surface.point_on - center
                    perp = d - (d @ axis) * axis
                    if np.linalg.norm(perp) > 1e-4:
                        continue
                    used.add(nbr)
                    group.append(nbr)
                    stack.append(nbr)
            groups.append(sorted(group))
        return groups

    # -------------------------------------------------------------------------
    # Pockets, Slots, Steps, Shoulders
    # -------------------------------------------------------------------------
    def _recognize_pockets_and_slots(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        floor_ids = [f.index for f in self.faces if self._tags.get(f.index) == "floor"]

        used_floors: set[int] = set()
        for fid in floor_ids:
            if fid in used_floors:
                continue
            floor_face = self.faces[fid]
            floor_z = float(floor_face.triangles[:, :, 2].mean()) if len(floor_face.triangles) else 0.0

            # Find connected floor faces at same Z
            floor_group = [fid]
            used_floors.add(fid)
            for other_id in floor_ids:
                if other_id in used_floors:
                    continue
                other_face = self.faces[other_id]
                other_z = float(other_face.triangles[:, :, 2].mean())
                if abs(other_z - floor_z) < 0.2:
                    # check adjacency
                    if any(other_id in self.adjacency.get(fg, []) for fg in floor_group):
                        floor_group.append(other_id)
                        used_floors.add(other_id)

            # Find surrounding wall faces that rise above the floor
            wall_faces: set[int] = set()
            for fg in floor_group:
                for nbr in self.adjacency.get(fg, []):
                    if nbr >= 0 and nbr < len(self.faces) and self._tags.get(nbr) in ("wall", "cylinder"):
                        nbr_face = self.faces[nbr]
                        if len(nbr_face.triangles):
                            nbr_zmax = float(nbr_face.triangles[:, :, 2].max())
                            if nbr_zmax > floor_z + 0.5:
                                wall_faces.add(nbr)

            if not wall_faces:
                continue

            wall_faces_list = sorted(wall_faces)
            # If all surrounding walls are internal cylinder faces forming a hole/bore,
            # this floor is the bottom of a hole/bore, not a prismatic step/pocket.
            if all(self.faces[w].surface.kind is SurfaceKind.CYLINDER and self.faces[w].is_internal for w in wall_faces_list):
                continue

            zs_wall = np.concatenate([self.faces[w].triangles[:, :, 2].ravel() for w in wall_faces_list if len(self.faces[w].triangles)])
            top_z = min(self.stock_top_z, float(zs_wall.max())) if len(zs_wall) else self.stock_top_z
            depth = top_z - floor_z
            if depth <= 1e-4:
                continue

            all_feature_faces = floor_group + wall_faces_list
            floor_pts = np.concatenate([self.faces[i].triangles[:, :, :2].reshape(-1, 2) for i in floor_group if len(self.faces[i].triangles)])
            bmin2 = floor_pts.min(axis=0)
            bmax2 = floor_pts.max(axis=0)
            extent = bmax2 - bmin2

            # Extract 2D boundary loop from floor wire loops if available
            boundary_loop: Optional[list[np.ndarray]] = None
            island_loops: list[list[np.ndarray]] = []
            for fg in floor_group:
                for w in self.faces[fg].wires:
                    if len(w.points) < 3:
                        continue
                    loop_2d = [p[:2].copy() for p in w.points]
                    if w.is_outer:
                        if boundary_loop is None:
                            boundary_loop = loop_2d
                    else:
                        # Inner wire = island (boss/obstacle inside the pocket)
                        island_loops.append(loop_2d)

            # Check open vs closed boundary
            # If wall faces don't fully close around the floor, or floor touches stock outer bounds -> STEP / OPEN_SLOT
            is_open = False
            for w in wall_faces_list:
                nbr_walls = sum(1 for n in self.adjacency.get(w, []) if n in wall_faces_list)
                if nbr_walls < 2:
                    is_open = True
                    break

            aspect_ratio = max(extent) / max(min(extent), 1e-6)

            if aspect_ratio >= 2.5:
                ftype = FeatureType.OPEN_SLOT if is_open else FeatureType.SLOT
            elif is_open:
                ftype = FeatureType.STEP
            else:
                ftype = FeatureType.POCKET

            pocket_center = np.array([0.5 * (bmin2[0] + bmax2[0]), 0.5 * (bmin2[1] + bmax2[1]), floor_z])
            floor_norm = self.faces[floor_group[0]].oriented_normal if self.faces[floor_group[0]].oriented_normal is not None else self.tool_axis.copy()

            out.append(MachiningFeature(
                id=f"{ftype.value}_{floor_group[0]}",
                type=ftype,
                face_indices=all_feature_faces,
                bounds_min=np.array([bmin2[0], bmin2[1], floor_z]),
                bounds_max=np.array([bmax2[0], bmax2[1], top_z]),
                depth=depth,
                top_z=top_z,
                floor_z=floor_z,
                axis=self.tool_axis.copy(),
                is_concave=True,
                accessibility=self._accessibility_of(floor_group),
                wall_faces=wall_faces_list,
                floor_faces=floor_group,
                boundary_loop=boundary_loop,
                island_loops=island_loops if island_loops else None,
                source_edges=self._edges_for_faces(all_feature_faces),
                finish=self.requirements.get(floor_group[0]),
                tolerances=self._tolerance_for_faces(all_feature_faces),
                recognition_evidence=[
                    f"type_{ftype.value}",
                    f"depth_{depth:.2f}",
                    f"floor_area_{sum(self.faces[i].area for i in floor_group):.1f}",
                    "open_boundary" if is_open else "closed_cavity",
                    f"islands_{len(island_loops)}" if island_loops else "no_islands",
                ],
                surface_type="plane",
                center=pocket_center,
                axial_start=top_z,
                axial_end=floor_z,
                orientation=floor_norm.copy(),
                source_geometry_reference=f"Face#{floor_group[0]}:{ftype.value.upper()}:Z{floor_z:.3f}",
            ))
        return out

    # -------------------------------------------------------------------------
    # Facing Regions: Unified material removal region on setup top
    # -------------------------------------------------------------------------
    def _recognize_facing_regions(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        top_faces = [f for f in self.faces if self._tags.get(f.index) == "top" and f.surface.kind is SurfaceKind.PLANE]
        if not top_faces:
            return out

        excess_height = self.stock_top_z - self._model_top_z
        if excess_height <= 1e-4:
            return out  # No excess stock above model top

        # Group all top planar faces
        face_indices = [f.index for f in top_faces]
        all_pts = np.concatenate([f.triangles[:, :, :2].reshape(-1, 2) for f in top_faces if len(f.triangles)])
        bmin2 = all_pts.min(axis=0)
        bmax2 = all_pts.max(axis=0)
        facing_center = np.array([0.5 * (bmin2[0] + bmax2[0]), 0.5 * (bmin2[1] + bmax2[1]), self._model_top_z])

        out.append(MachiningFeature(
            id=f"facing_{top_faces[0].index}",
            type=FeatureType.FACING_REGION,
            face_indices=face_indices,
            bounds_min=np.array([bmin2[0], bmin2[1], self._model_top_z]),
            bounds_max=np.array([bmax2[0], bmax2[1], self.stock_top_z]),
            depth=excess_height,
            top_z=self.stock_top_z,
            floor_z=self._model_top_z,
            axis=self.tool_axis.copy(),
            is_concave=False,
            accessibility=Accessibility.TOOL_AXIS_OK,
            floor_faces=face_indices,
            source_edges=self._edges_for_faces(face_indices),
            finish=self.requirements.get(top_faces[0].index),
            tolerances=self._tolerance_for_faces(face_indices),
            notes={"excess_height": excess_height},
            recognition_evidence=[
                "unified_stock_top_material_removal",
                f"excess_height_{excess_height:.2f}",
            ],
            surface_type="plane",
            center=facing_center,
            axial_start=self.stock_top_z,
            axial_end=self._model_top_z,
            orientation=self.tool_axis.copy(),
            source_geometry_reference=f"Face#{top_faces[0].index}:FACING:Z{self._model_top_z:.3f}",
        ))
        return out

    # -------------------------------------------------------------------------
    # Chamfers, Fillets, Freeform
    # -------------------------------------------------------------------------
    def _recognize_chamfers(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        for f in self.faces:
            if self._tags.get(f.index) != "slanted" or f.surface.kind is not SurfaceKind.PLANE:
                continue
            n = f.oriented_normal
            if float(n @ self.tool_axis) <= 0.05:
                continue
            angle_from_horiz = abs(float(np.degrees(np.arcsin(np.clip(n[2], -1, 1)))))
            if 15.0 < angle_from_horiz < 75.0:
                zs = f.triangles[:, :, 2]
                top_z, bottom_z = float(zs.max()), float(zs.min())
                depth = top_z - bottom_z
                if 0.05 < depth < 10.0:
                    out.append(MachiningFeature(
                        id=f"chamfer_{f.index}",
                        type=FeatureType.CHAMFER,
                        face_indices=[f.index],
                        bounds_min=f.bounds_min,
                        bounds_max=f.bounds_max,
                        depth=depth,
                        top_z=top_z,
                        floor_z=bottom_z,
                        axis=f.surface.axis.copy() if f.surface.axis is not None else None,
                        is_concave=False,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        source_edges=self._edges_for_faces([f.index]),
                        finish=self.requirements.get(f.index),
                        tolerances=self._tolerance_for_faces([f.index]),
                        notes={"angle_from_horizontal": angle_from_horiz},
                        recognition_evidence=[f"chamfer_angle_{angle_from_horiz:.1f}deg"],
                        surface_type="plane",
                        center=f.center.copy() if f.center is not None else 0.5 * (f.bounds_min + f.bounds_max),
                        axial_start=top_z,
                        axial_end=bottom_z,
                        orientation=f.oriented_normal.copy() if f.oriented_normal is not None else None,
                        source_geometry_reference=f"Face#{f.index}:CHAMFER:ANG{angle_from_horiz:.1f}deg",
                    ))
        return out

    def _recognize_fillets(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        for f in self.faces:
            if f.surface.kind in (SurfaceKind.CYLINDER, SurfaceKind.TORUS) and self._tags.get(f.index) in ("slanted", "wall"):
                u_span = self._face_u_span(f)
                if f.surface.kind == SurfaceKind.CYLINDER and u_span >= 2 * np.pi - 0.15:
                    continue
                if f.surface.kind == SurfaceKind.TORUS:
                    fillet_r = f.surface.ref_radius if f.surface.ref_radius is not None and f.surface.ref_radius > 0 else f.surface.radius
                    major_r = f.surface.radius
                else:
                    fillet_r = f.surface.radius
                    major_r = None

                if fillet_r and fillet_r < 15.0 and f.area < 500.0:
                    zs = f.triangles[:, :, 2] if len(f.triangles) else np.array([0.0])
                    top_z, bottom_z = float(zs.max()), float(zs.min())
                    stype = "cylinder" if f.surface.kind == SurfaceKind.CYLINDER else "torus"
                    is_internal = f.is_internal
                    if is_internal is None:
                        # For cylindrical/torus fillet, evaluate oriented normal against radius center
                        if f.surface.point_on is not None:
                            radial = f.center[:2] - f.surface.point_on[:2]
                            is_internal = bool(float(f.oriented_normal[:2] @ radial) < -1e-4)
                        else:
                            is_internal = False

                    xy_ext = float(np.linalg.norm(f.bounds_max[:2] - f.bounds_min[:2])) if (f.bounds_max is not None and f.bounds_min is not None) else 0.0
                    diameter = (2.0 * major_r) if major_r is not None else (xy_ext if xy_ext > 0 else (2.0 * fillet_r if fillet_r else None))
                    notes_dict = {}
                    if major_r is not None:
                        notes_dict["r_start"] = major_r
                        notes_dict["r_end"] = major_r
                        notes_dict["curvature_radius"] = fillet_r
                    notes_dict["z_start"] = top_z
                    notes_dict["z_end"] = bottom_z

                    out.append(MachiningFeature(
                        id=f"fillet_{f.index}",
                        type=FeatureType.FILLET,
                        face_indices=[f.index],
                        bounds_min=f.bounds_min,
                        bounds_max=f.bounds_max,
                        depth=top_z - bottom_z,
                        top_z=top_z,
                        floor_z=bottom_z,
                        axis=f.surface.axis.copy() if f.surface.axis is not None else None,
                        diameter=diameter,
                        radius=fillet_r,
                        fillet_radius=fillet_r,
                        is_concave=bool(is_internal),
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        wall_faces=[f.index],
                        source_edges=self._edges_for_faces([f.index]),
                        finish=self.requirements.get(f.index),
                        tolerances=self._tolerance_for_faces([f.index]),
                        notes=notes_dict,
                        recognition_evidence=[f"fillet_radius_{fillet_r:.2f}"],
                        surface_type=stype,
                        center=f.center.copy() if f.center is not None else 0.5 * (f.bounds_min + f.bounds_max),
                        axial_start=top_z,
                        axial_end=bottom_z,
                        orientation=f.oriented_normal.copy() if f.oriented_normal is not None else None,
                        source_geometry_reference=f"Face#{f.index}:FILLET:R{fillet_r:.3f}",
                    ))
        return out

    def _recognize_freeform(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        for f in self.faces:
            if f.surface.kind in (SurfaceKind.BSPLINE, SurfaceKind.SPHERE, SurfaceKind.TORUS) and self._tags.get(f.index) not in ("top", "floor"):
                zs = f.triangles[:, :, 2]
                top_z, bottom_z = float(zs.max()), float(zs.min())
                out.append(MachiningFeature(
                    id=f"freeform_{f.index}",
                    type=FeatureType.FREEFORM_SURFACE,
                    face_indices=[f.index],
                    bounds_min=f.bounds_min,
                    bounds_max=f.bounds_max,
                    depth=top_z - bottom_z,
                    top_z=top_z,
                    floor_z=bottom_z,
                    axis=f.surface.axis.copy() if f.surface.axis is not None else None,
                    is_concave=True,
                    accessibility=self._accessibility_of([f.index]),
                    source_edges=self._edges_for_faces([f.index]),
                    finish=self.requirements.get(f.index),
                    tolerances=self._tolerance_for_faces([f.index]),
                    recognition_evidence=[f"surface_{f.surface.kind.value}"],
                    surface_type=f.surface.kind.value,
                    center=f.center.copy() if f.center is not None else 0.5 * (f.bounds_min + f.bounds_max),
                    axial_start=top_z,
                    axial_end=bottom_z,
                    orientation=f.oriented_normal.copy() if f.oriented_normal is not None else None,
                    source_geometry_reference=f"Face#{f.index}:FREEFORM:{f.surface.kind.value.upper()}",
                ))
        return out

    def _recognize_threaded_holes(self, existing: list[MachiningFeature]) -> list[MachiningFeature]:
        """Upgrade existing HOLE/THROUGH_HOLE features to THREADED_HOLE if they have
        thread annotations in notes. Thread detection from pure geometry is unreliable,
        so this relies on user-provided annotations or notes from upstream planners."""
        out: list[MachiningFeature] = []
        for f in existing:
            if f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                if f.notes.get("tapped") or f.notes.get("threaded"):
                    out.append(MachiningFeature(
                        id=f.id,
                        type=FeatureType.THREADED_HOLE,
                        face_indices=f.face_indices,
                        bounds_min=f.bounds_min.copy(),
                        bounds_max=f.bounds_max.copy(),
                        depth=f.depth,
                        top_z=f.top_z,
                        floor_z=f.floor_z,
                        axis=f.axis.copy() if f.axis is not None else None,
                        diameter=f.diameter,
                        radius=f.radius,
                        is_concave=True,
                        accessibility=f.accessibility,
                        wall_faces=f.wall_faces,
                        floor_faces=f.floor_faces,
                        parent_feature_id=f.parent_feature_id,
                        child_feature_ids=f.child_feature_ids,
                        source_edges=list(f.source_edges),
                        finish=f.finish,
                        tolerances=f.tolerances,
                        recognition_evidence=f.recognition_evidence + ["threaded_hole_upgrade"],
                        notes=f.notes,
                        boundary_loop=f.boundary_loop,
                        island_loops=f.island_loops,
                        surface_type=f.surface_type or "cylinder",
                        center=f.center.copy() if f.center is not None else None,
                        axial_start=f.axial_start,
                        axial_end=f.axial_end,
                        orientation=f.orientation.copy() if f.orientation is not None else None,
                        source_geometry_reference=f.source_geometry_reference or f"Face#{f.face_indices[0]}:THREADED_HOLE",
                    ))
        return out

    def _recognize_grooves(self, existing: list[MachiningFeature]) -> list[MachiningFeature]:
        """Recognize groove features: narrow slots or annular channels that require
        a groove cutter. Detects features with groove annotations or narrow slot geometry."""
        out: list[MachiningFeature] = []
        for f in existing:
            if f.type == FeatureType.SLOT and f.diameter is not None:
                # Narrow slot that might be a groove
                if f.notes.get("groove") or (f.depth > 0 and f.xy_extent.min() < 5.0):
                    out.append(MachiningFeature(
                        id=f.id,
                        type=FeatureType.GROOVE,
                        face_indices=f.face_indices,
                        bounds_min=f.bounds_min.copy(),
                        bounds_max=f.bounds_max.copy(),
                        depth=f.depth,
                        top_z=f.top_z,
                        floor_z=f.floor_z,
                        axis=f.axis.copy() if f.axis is not None else None,
                        diameter=f.diameter,
                        radius=f.radius,
                        is_concave=True,
                        accessibility=f.accessibility,
                        wall_faces=f.wall_faces,
                        floor_faces=f.floor_faces,
                        source_edges=list(f.source_edges),
                        finish=f.finish,
                        tolerances=f.tolerances,
                        recognition_evidence=f.recognition_evidence + ["groove_from_slot"],
                        notes=f.notes,
                        boundary_loop=f.boundary_loop,
                        island_loops=f.island_loops,
                        surface_type=f.surface_type or "plane",
                        center=f.center.copy() if f.center is not None else None,
                        axial_start=f.axial_start,
                        axial_end=f.axial_end,
                        orientation=f.orientation.copy() if f.orientation is not None else None,
                        source_geometry_reference=f.source_geometry_reference or f"Face#{f.face_indices[0]}:GROOVE",
                    ))
        return out

    def _recognize_outer_contours(self) -> list[MachiningFeature]:
        """Recognize the outer perimeter / external boundary contour of the part.
        This feature drives outer profile clearing (stock removal) and perimeter finishing."""
        out: list[MachiningFeature] = []
        if not self.faces:
            return out

        model_bottom_z = min(float(f.bounds_min[2]) for f in self.faces if len(f.triangles))
        depth = self._model_top_z - model_bottom_z
        if depth <= 1e-4:
            return out

        all_pts_list = [f.triangles.reshape(-1, 3) for f in self.faces if len(f.triangles)]
        if not all_pts_list:
            return out
        all_pts = np.concatenate(all_pts_list)
        bmin = all_pts.min(axis=0)
        bmax = all_pts.max(axis=0)
        extent = bmax[:2] - bmin[:2]
        if extent[0] <= 1e-4 or extent[1] <= 1e-4:
            return out

        wall_faces = [f.index for f in self.faces if self._tags.get(f.index) in ("wall", "slanted", "cylinder")]

        def _compute_hull(pts: np.ndarray) -> list[np.ndarray]:
            pts_2d = sorted({(round(float(p[0]), 3), round(float(p[1]), 3)) for p in pts})
            if len(pts_2d) >= 3:
                lower: list[tuple[float, float]] = []
                for p in pts_2d:
                    while len(lower) >= 2 and (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) - (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0]) <= 1e-7:
                        lower.pop()
                    lower.append(p)
                upper: list[tuple[float, float]] = []
                for p in reversed(pts_2d):
                    while len(upper) >= 2 and (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) - (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0]) <= 1e-7:
                        upper.pop()
                    upper.append(p)
                hull = lower[:-1] + upper[:-1]
                return [np.array([p[0], p[1]], dtype=float) for p in hull]
            pts_min = pts.min(axis=0)
            pts_max = pts.max(axis=0)
            return [
                np.array([pts_min[0], pts_min[1]]),
                np.array([pts_max[0], pts_min[1]]),
                np.array([pts_max[0], pts_max[1]]),
                np.array([pts_min[0], pts_max[1]]),
            ]

        # Check if the part has continuous full-height outer wall faces spanning from bottom to top.
        # If full-height outer walls exist (e.g. prismatic brackets with steps/pockets), the outer contour
        # is the full-height perimeter of the part, NOT a stepped boss.
        has_full_height_outer_walls = any(
            self.faces[i].bounds_min[2] <= model_bottom_z + 0.5
            and self.faces[i].bounds_max[2] >= self._model_top_z - 0.5
            for i in wall_faces
        )

        stepped_features_created = False
        if not has_full_height_outer_walls:
            step_floors = []
            for f in self.faces:
                if self._tags.get(f.index) == "floor" and f.area > 2.0:
                    zs = f.triangles[:, :, 2]
                    z_fl = float(zs.mean())
                    if model_bottom_z + 0.5 < z_fl < self._model_top_z - 0.5:
                        f_ext = f.bounds_max[:2] - f.bounds_min[:2]
                        # Flange floor must extend to the outer extent of the base
                        if abs(f_ext[0] - extent[0]) < 1.0 or abs(f_ext[1] - extent[1]) < 1.0:
                            step_floors.append((z_fl, f))

            if step_floors:
                step_floors.sort(key=lambda x: x[0])
                for z_fl, fl_face in step_floors:
                    u_wall_faces = [
                        f.index for f in self.faces
                        if self._tags.get(f.index) in ("wall", "slanted", "cylinder")
                        and f.bounds_min[2] >= z_fl - 0.15
                        and f.bounds_max[2] > z_fl + 0.3
                    ]
                    if not u_wall_faces:
                        continue
                    u_pts_list = [self.faces[i].triangles.reshape(-1, 3) for i in u_wall_faces if len(self.faces[i].triangles)]
                    if not u_pts_list:
                        continue
                    u_pts = np.concatenate(u_pts_list)
                    u_bmin = u_pts.min(axis=0)
                    u_bmax = u_pts.max(axis=0)
                    u_extent = u_bmax[:2] - u_bmin[:2]

                    if (u_extent[0] < extent[0] - 0.5 or u_extent[1] < extent[1] - 0.5) and (u_extent[0] > 0.5 and u_extent[1] > 0.5):
                        u_loop = _compute_hull(u_pts)
                        u_depth = self._model_top_z - z_fl
                        u_center = np.array([0.5 * (u_bmin[0] + u_bmax[0]), 0.5 * (u_bmin[1] + u_bmax[1]), 0.5 * (self._model_top_z + z_fl)])

                        out.append(MachiningFeature(
                            id=f"contour_boss_{fl_face.index}",
                            type=FeatureType.CONTOUR,
                            face_indices=u_wall_faces,
                            bounds_min=np.array([u_bmin[0], u_bmin[1], z_fl]),
                            bounds_max=np.array([u_bmax[0], u_bmax[1], self._model_top_z]),
                            depth=u_depth,
                            top_z=self._model_top_z,
                            floor_z=z_fl,
                            axis=self.tool_axis.copy(),
                            is_concave=False,
                            accessibility=Accessibility.TOOL_AXIS_OK,
                            wall_faces=u_wall_faces,
                            boundary_loop=u_loop,
                            source_edges=self._edges_for_faces(u_wall_faces),
                            tolerances=self._tolerance_for_faces(u_wall_faces),
                            notes={"perimeter": True, "outer_contour": True, "step_boss": True, "floor_face": fl_face.index},
                            recognition_evidence=[
                                "stepped_boss_outer_contour",
                                f"depth_{u_depth:.2f}",
                                f"floor_z_{z_fl:.2f}",
                            ],
                            surface_type="contour",
                            center=u_center,
                            axial_start=self._model_top_z,
                            axial_end=z_fl,
                            orientation=self.tool_axis.copy(),
                            source_geometry_reference=f"Face#{fl_face.index}:STEP_BOSS_CONTOUR",
                        ))

                        l_wall_faces = [
                            f.index for f in self.faces
                            if self._tags.get(f.index) in ("wall", "slanted", "cylinder")
                            and f.bounds_max[2] <= z_fl + 0.15
                            and f.bounds_min[2] < z_fl - 0.2
                        ]
                        if l_wall_faces:
                            l_pts_list = [self.faces[i].triangles.reshape(-1, 3) for i in l_wall_faces if len(self.faces[i].triangles)]
                            if l_pts_list:
                                l_pts = np.concatenate(l_pts_list)
                                l_bmin = l_pts.min(axis=0)
                                l_bmax = l_pts.max(axis=0)
                                l_loop = _compute_hull(l_pts)
                                l_depth = z_fl - model_bottom_z
                                l_center = np.array([0.5 * (l_bmin[0] + l_bmax[0]), 0.5 * (l_bmin[1] + l_bmax[1]), 0.5 * (z_fl + model_bottom_z)])

                                out.append(MachiningFeature(
                                    id="contour_outer",
                                    type=FeatureType.CONTOUR,
                                    face_indices=l_wall_faces,
                                    bounds_min=np.array([l_bmin[0], l_bmin[1], model_bottom_z]),
                                    bounds_max=np.array([l_bmax[0], l_bmax[1], z_fl]),
                                    depth=l_depth,
                                    top_z=z_fl,
                                    floor_z=model_bottom_z,
                                    axis=self.tool_axis.copy(),
                                    is_concave=False,
                                    accessibility=Accessibility.TOOL_AXIS_OK,
                                    wall_faces=l_wall_faces,
                                    boundary_loop=l_loop,
                                    source_edges=self._edges_for_faces(l_wall_faces),
                                    tolerances=self._tolerance_for_faces(l_wall_faces),
                                    notes={"perimeter": True, "outer_contour": True, "base_contour": True},
                                    recognition_evidence=[
                                        "base_outer_contour",
                                        f"depth_{l_depth:.2f}",
                                    ],
                                    surface_type="contour",
                                    center=l_center,
                                    axial_start=z_fl,
                                    axial_end=model_bottom_z,
                                    orientation=self.tool_axis.copy(),
                                    source_geometry_reference="MODEL:BASE_PERIMETER_CONTOUR",
                                ))
                                stepped_features_created = True
                                break

        if stepped_features_created:
            return out

        boundary_loop: Optional[list[np.ndarray]] = None
        for f in self.faces:
            if self._tags.get(f.index) in ("top", "bottom", "step_floor"):
                f_extent = f.bounds_max[:2] - f.bounds_min[:2]
                if abs(f_extent[0] - extent[0]) < 0.5 and abs(f_extent[1] - extent[1]) < 0.5:
                    for w in f.wires:
                        if w.is_outer and len(w.points) >= 3:
                            boundary_loop = [p[:2].copy() for p in w.points]
                            break
            if boundary_loop is not None:
                break

        if boundary_loop is None:
            boundary_loop = _compute_hull(all_pts)


        contour_center = np.array([0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1]), 0.5 * (self._model_top_z + model_bottom_z)])
        target_faces = wall_faces if wall_faces else [f.index for f in self.faces]

        # Check if outer boundary is circular/cylindrical
        is_circular = False
        dia: Optional[float] = None
        rad: Optional[float] = None
        if abs(extent[0] - extent[1]) < 0.2 and extent[0] > 0.1:
            mean_r = 0.5 * (extent[0] + extent[1]) / 2.0
            if boundary_loop and len(boundary_loop) >= 6:
                dists = [float(np.linalg.norm(p[:2] - contour_center[:2])) for p in boundary_loop]
                if float(np.std(dists)) < 0.08 * mean_r:
                    is_circular = True
            elif wall_faces and all(self._tags.get(idx) in ("cylinder", "revolved") or getattr(self.faces[idx], "surface_type", "") == "cylinder" for idx in wall_faces):
                is_circular = True
            elif getattr(self, "_is_revolved_model", False):
                is_circular = True

            if is_circular:
                dia = float((extent[0] + extent[1]) / 2.0)
                rad = float(dia / 2.0)
                n_pts = 64
                boundary_loop = [
                    np.array([contour_center[0] + rad * math.cos(a), contour_center[1] + rad * math.sin(a)])
                    for a in np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
                ]

        notes = {"perimeter": True, "outer_contour": True}
        if is_circular:
            notes["is_circular"] = True
            notes["center_xy"] = [float(contour_center[0]), float(contour_center[1])]

        out.append(MachiningFeature(
            id="contour_outer",
            type=FeatureType.CONTOUR,
            face_indices=target_faces,
            bounds_min=np.array([bmin[0], bmin[1], model_bottom_z]),
            bounds_max=np.array([bmax[0], bmax[1], self._model_top_z]),
            depth=depth,
            top_z=self._model_top_z,
            floor_z=model_bottom_z,
            axis=self.tool_axis.copy(),
            diameter=dia,
            radius=rad,
            is_concave=False,
            accessibility=Accessibility.TOOL_AXIS_OK,
            wall_faces=wall_faces,
            boundary_loop=boundary_loop,
            source_edges=self._edges_for_faces(target_faces),
            tolerances=self._tolerance_for_faces(target_faces),
            notes=notes,
            recognition_evidence=[
                "outer_part_boundary_profile",
                f"depth_{depth:.2f}",
                f"bounds_min_{bmin[0]:.1f}_{bmin[1]:.1f}",
                f"bounds_max_{bmax[0]:.1f}_{bmax[1]:.1f}",
            ],
            surface_type="cylinder" if is_circular else "contour",
            center=contour_center,
            axial_start=self._model_top_z,
            axial_end=model_bottom_z,
            orientation=self.tool_axis.copy(),
            source_geometry_reference="MODEL:OUTER_PERIMETER_CONTOUR",
        ))
        return out

    # -------------------------------------------------------------------------
    # Revolved / Turned Geometry Recognition
    # -------------------------------------------------------------------------
    def _recognize_revolved_profile(self) -> tuple[list[MachiningFeature], bool]:
        """Detect rotational symmetry axes and build exact radial/axial profiles.

        Returns (features, is_pure_revolution).
        ``is_pure_revolution`` is True when the dominant rotational axis covers
        ≥ 85 % of the part surface area — used by `recognize()` to suppress the
        generic outer-contour fallback for turned parts.

        Recognition flow
        ----------------
        Axis discovery
         └─ For each coaxial axis cluster:
             ├─ One top-level CONTOUR feature embedding the full RevolvedProfile
             │  (all segments, transitions, min/max diameters, axial extent)
             ├─ Per CYLINDRICAL_STEP segment  → BOSS (external) or BORE (internal)
             ├─ Per PLANAR_SHOULDER segment   → SHOULDER feature
             ├─ Per GROOVE_RECESS segment     → GROOVE feature
             └─ Per CONICAL_TRANSITION        → CHAMFER / STEP feature

        No synthetic bounding-box coordinates are used. All geometry originates
        from analytical B-Rep surface data (axis, radius, semi_angle, point_on).
        """
        out: list[MachiningFeature] = []
        is_pure_revolution = False

        try:
            recognizer = RevolvedGeometryRecognizer(self.faces)
            axes = recognizer.discover_rotational_axes()
        except Exception:
            return out, False

        if not axes:
            return out, False

        for axis_idx, (ax_origin, ax_dir, axis_face_ids) in enumerate(axes):
            try:
                profile = recognizer.build_profile(ax_origin, ax_dir)
            except Exception:
                continue
            if profile is None or not profile.segments:
                continue

            # Determine pure-revolution flag from the primary (largest area) axis
            if axis_idx == 0 and profile.is_pure_revolution:
                is_pure_revolution = True

            # Only generate turned revolved profiles for true rotationally-symmetric / turned parts
            if not profile.is_pure_revolution:
                continue

            axis_tag = f"rev{axis_idx}"
            all_face_ids = sorted(set(
                fi for seg in profile.segments for fi in seg.face_indices
            ))

            # ------------------------------------------------------------------
            # 1. Top-level CONTOUR anchoring the complete revolved profile
            # ------------------------------------------------------------------
            profile_dict = profile.to_dict()
            model_bottom_z = profile.axial_min
            model_top_z = profile.axial_max
            depth = profile.total_axial_length

            # Bounding box in model space from profile bounds
            r_max = profile.max_diameter / 2.0
            bmin_profile = np.array([-r_max, -r_max, model_bottom_z]) + ax_origin
            bmax_profile = np.array([ r_max,  r_max, model_top_z])   + ax_origin

            profile_center = ax_origin + ax_dir * (
                0.5 * (profile.axial_min + profile.axial_max)
            )

            out.append(MachiningFeature(
                id=f"revolved_profile_{axis_tag}",
                type=FeatureType.CONTOUR,
                face_indices=all_face_ids,
                bounds_min=bmin_profile,
                bounds_max=bmax_profile,
                depth=depth,
                top_z=float(model_top_z),
                floor_z=float(model_bottom_z),
                axis=ax_dir.copy(),
                is_concave=False,
                accessibility=Accessibility.TOOL_AXIS_OK,
                wall_faces=all_face_ids,
                source_edges=self._edges_for_faces(all_face_ids),
                tolerances=self._tolerance_for_faces(all_face_ids),
                notes={
                    "revolved_profile": profile_dict,
                    "is_pure_revolution": profile.is_pure_revolution,
                    "face_coverage_ratio": profile.face_coverage_ratio,
                    "num_external_segments": len(profile.external_segments),
                    "num_internal_segments": len(profile.internal_segments),
                    "num_shoulders": len(profile.shoulders),
                    "min_diameter": profile.min_diameter,
                    "max_diameter": profile.max_diameter,
                    "axial_length": profile.total_axial_length,
                },
                recognition_evidence=[
                    "revolved_rotational_axis",
                    f"coverage_{profile.face_coverage_ratio:.2f}",
                    f"segments_{len(profile.segments)}",
                    f"min_dia_{profile.min_diameter:.2f}",
                    f"max_dia_{profile.max_diameter:.2f}",
                ],
                surface_type="revolved",
                center=profile_center,
                axial_start=float(model_top_z),
                axial_end=float(model_bottom_z),
                orientation=ax_dir.copy(),
                source_geometry_reference=(
                    f"REVOLVED_AXIS:{axis_idx}"
                    f":Ø{profile.min_diameter:.3f}-{profile.max_diameter:.3f}"
                    f":L{profile.total_axial_length:.3f}"
                ),
            ))

            # ------------------------------------------------------------------
            # 2. Per-segment features from the analytical profile
            # ------------------------------------------------------------------
            for seg in profile.segments:
                seg_bmin = np.array([
                    ax_origin[0] - seg.d_max / 2.0,
                    ax_origin[1] - seg.d_max / 2.0,
                    seg.z_min,
                ])
                seg_bmax = np.array([
                    ax_origin[0] + seg.d_max / 2.0,
                    ax_origin[1] + seg.d_max / 2.0,
                    seg.z_max,
                ])
                seg_center = ax_origin + ax_dir * (0.5 * (seg.z_start + seg.z_end))
                seg_id = f"{axis_tag}_seg{seg.index}"

                if seg.kind == RevolvedProfileSegmentKind.CYLINDRICAL_STEP:
                    ftype = FeatureType.BORE if seg.is_internal else FeatureType.BOSS
                    out.append(MachiningFeature(
                        id=seg_id,
                        type=ftype,
                        face_indices=seg.face_indices,
                        bounds_min=seg_bmin,
                        bounds_max=seg_bmax,
                        depth=seg.axial_length,
                        top_z=float(seg.z_max),
                        floor_z=float(seg.z_min),
                        axis=ax_dir.copy(),
                        diameter=seg.d_start,
                        radius=seg.r_start,
                        is_concave=seg.is_internal,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        wall_faces=seg.face_indices,
                        source_edges=self._edges_for_faces(seg.face_indices),
                        tolerances=self._tolerance_for_faces(seg.face_indices),
                        notes=seg.to_dict(),
                        recognition_evidence=[
                            "revolved_cylindrical_step",
                            f"diameter_{seg.d_start:.3f}",
                            f"axial_length_{seg.axial_length:.3f}",
                            "internal" if seg.is_internal else "external",
                        ],
                        surface_type="cylinder",
                        center=seg_center,
                        axial_start=float(seg.z_max),
                        axial_end=float(seg.z_min),
                        orientation=ax_dir.copy(),
                        source_geometry_reference=(
                            f"REVOLVED_SEG:{seg.index}:CYLINDER"
                            f":Ø{seg.d_start:.3f}"
                            f":L{seg.axial_length:.3f}"
                        ),
                    ))

                elif seg.kind == RevolvedProfileSegmentKind.PLANAR_SHOULDER:
                    out.append(MachiningFeature(
                        id=seg_id,
                        type=FeatureType.SHOULDER,
                        face_indices=seg.face_indices,
                        bounds_min=seg_bmin,
                        bounds_max=seg_bmax,
                        depth=float(seg.shoulder_width or (seg.r_end - seg.r_start)),
                        top_z=float(seg.z_start),
                        floor_z=float(seg.z_end),
                        axis=ax_dir.copy(),
                        diameter=seg.d_max,
                        radius=seg.r_end,
                        is_concave=True,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        wall_faces=[],
                        floor_faces=seg.face_indices,
                        source_edges=self._edges_for_faces(seg.face_indices),
                        tolerances=self._tolerance_for_faces(seg.face_indices),
                        notes=seg.to_dict(),
                        recognition_evidence=[
                            "revolved_planar_shoulder",
                            f"inner_dia_{seg.d_min:.3f}",
                            f"outer_dia_{seg.d_max:.3f}",
                            f"shoulder_width_{seg.shoulder_width or 0.0:.3f}",
                        ],
                        surface_type="plane",
                        center=seg_center,
                        axial_start=float(seg.z_start),
                        axial_end=float(seg.z_end),
                        orientation=ax_dir.copy(),
                        source_geometry_reference=(
                            f"REVOLVED_SEG:{seg.index}:SHOULDER"
                            f":Z{seg.z_start:.3f}"
                        ),
                    ))

                elif seg.kind == RevolvedProfileSegmentKind.GROOVE_RECESS:
                    out.append(MachiningFeature(
                        id=seg_id,
                        type=FeatureType.GROOVE,
                        face_indices=seg.face_indices,
                        bounds_min=seg_bmin,
                        bounds_max=seg_bmax,
                        depth=seg.axial_length,
                        top_z=float(seg.z_max),
                        floor_z=float(seg.z_min),
                        axis=ax_dir.copy(),
                        diameter=seg.d_min,
                        radius=seg.r_start,
                        is_concave=True,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        wall_faces=seg.face_indices,
                        source_edges=self._edges_for_faces(seg.face_indices),
                        tolerances=self._tolerance_for_faces(seg.face_indices),
                        notes=seg.to_dict(),
                        recognition_evidence=[
                            "revolved_groove_recess",
                            f"groove_dia_{seg.d_min:.3f}",
                            f"axial_length_{seg.axial_length:.3f}",
                        ],
                        surface_type="cylinder",
                        center=seg_center,
                        axial_start=float(seg.z_max),
                        axial_end=float(seg.z_min),
                        orientation=ax_dir.copy(),
                        source_geometry_reference=(
                            f"REVOLVED_SEG:{seg.index}:GROOVE"
                            f":Ø{seg.d_min:.3f}"
                        ),
                    ))

                elif seg.kind == RevolvedProfileSegmentKind.CONICAL_TRANSITION:
                    out.append(MachiningFeature(
                        id=seg_id,
                        type=FeatureType.CHAMFER,
                        face_indices=seg.face_indices,
                        bounds_min=seg_bmin,
                        bounds_max=seg_bmax,
                        depth=seg.axial_length,
                        top_z=float(seg.z_max),
                        floor_z=float(seg.z_min),
                        axis=ax_dir.copy(),
                        diameter=seg.d_max,
                        radius=seg.r_start,
                        is_concave=False,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        wall_faces=seg.face_indices,
                        source_edges=self._edges_for_faces(seg.face_indices),
                        tolerances=self._tolerance_for_faces(seg.face_indices),
                        notes=seg.to_dict(),
                        recognition_evidence=[
                            "revolved_conical_transition",
                            f"semi_angle_{seg.semi_angle_deg or 0.0:.1f}deg",
                            f"d_start_{seg.d_start:.3f}",
                            f"d_end_{seg.d_end:.3f}",
                        ],
                        surface_type="cone",
                        center=seg_center,
                        axial_start=float(seg.z_max),
                        axial_end=float(seg.z_min),
                        orientation=ax_dir.copy(),
                        source_geometry_reference=(
                            f"REVOLVED_SEG:{seg.index}:CONE"
                            f":angle_{seg.semi_angle_deg or 0.0:.1f}deg"
                        ),
                    ))

                elif seg.kind == RevolvedProfileSegmentKind.TOROIDAL_FILLET:
                    out.append(MachiningFeature(
                        id=seg_id,
                        type=FeatureType.FILLET,
                        face_indices=seg.face_indices,
                        bounds_min=seg_bmin,
                        bounds_max=seg_bmax,
                        depth=seg.axial_length,
                        top_z=float(seg.z_max),
                        floor_z=float(seg.z_min),
                        axis=ax_dir.copy(),
                        diameter=seg.d_max,
                        radius=seg.curvature_radius,
                        fillet_radius=seg.curvature_radius,
                        is_concave=False,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        wall_faces=seg.face_indices,
                        source_edges=self._edges_for_faces(seg.face_indices),
                        tolerances=self._tolerance_for_faces(seg.face_indices),
                        notes=seg.to_dict(),
                        recognition_evidence=[
                            "revolved_toroidal_fillet",
                            f"fillet_r_{seg.curvature_radius or 0.0:.3f}",
                        ],
                        surface_type="torus",
                        center=seg_center,
                        axial_start=float(seg.z_max),
                        axial_end=float(seg.z_min),
                        orientation=ax_dir.copy(),
                        source_geometry_reference=(
                            f"REVOLVED_SEG:{seg.index}:TORUS"
                            f":r{seg.curvature_radius or 0.0:.3f}"
                        ),
                    ))

        return out, is_pure_revolution

    def _accessibility_of(self, face_ids: list[int], top_sample_pts: Optional[list[np.ndarray]] = None) -> Accessibility:
        mesh = TriangleMesh.from_faces(self.faces)
        if not len(mesh.triangles):
            return Accessibility.INACCESSIBLE

        # If explicit top opening sample points are provided (for holes/bores/pockets)
        if top_sample_pts:
            for p in top_sample_pts:
                hit = mesh.ray_first_hit(p + self.tool_axis * 1e-3, self.tool_axis, max_dist=1e4)
                if hit is not None and hit > 1e-4:
                    return Accessibility.INACCESSIBLE
            return Accessibility.TOOL_AXIS_OK

        for fi in face_ids:
            f = self.faces[fi]
            pts = f.triangles.reshape(-1, 3)
            step = max(1, len(pts) // 12)
            for p in pts[::step]:
                hit = mesh.ray_first_hit(p + self.tool_axis * 1e-3, self.tool_axis, max_dist=1e4)
                if hit is not None and hit > 1e-4:
                    return Accessibility.INACCESSIBLE

        return Accessibility.TOOL_AXIS_OK

    def _dedupe_and_sort(self, features: list[MachiningFeature]) -> list[MachiningFeature]:
        seen: set[str] = set()
        out: list[MachiningFeature] = []
        for f in features:
            if f.id in seen:
                continue
            seen.add(f.id)
            out.append(f)
        out.sort(key=lambda f: (-f.depth, -float((f.bounds_max[:2] - f.bounds_min[:2]).prod())))
        return out
