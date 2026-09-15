"""Geometry-driven machining feature recognition.

Features are derived from actual B-Rep topology: surface kinds, oriented normals,
adjacency, boundary loops, coaxiality, concavity, and exact CAD dimensions.
Nothing is keyed on synthetic coordinates, names, or feature counts.
Every recognized feature carries semantic data, topological evidence, and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .context import FinishRequirement
from .errors import CamError
from .geometry import Face, Shape, SurfaceKind, TriangleMesh

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
    """Semantic machining feature extracted from real B-Rep geometry."""

    id: str
    type: FeatureType
    face_indices: list[int]
    bounds_min: np.ndarray            # setup space
    bounds_max: np.ndarray
    depth: float                      # machining depth below local top surface
    top_z: float                      # Z of material top over the feature (setup space)
    floor_z: float                    # Z of feature floor (for pockets/holes/steps)
    axis: Optional[np.ndarray] = None # feature axis (holes/bores/bosses)
    diameter: Optional[float] = None  # holes/bores/bosses
    radius: Optional[float] = None    # fillets/corners
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

    @property
    def center_xy(self) -> np.ndarray:
        return 0.5 * (self.bounds_min[:2] + self.bounds_max[:2])

    @property
    def xy_extent(self) -> np.ndarray:
        return self.bounds_max[:2] - self.bounds_min[:2]


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

    def _compute_model_top_z(self) -> float:
        valid_faces = [f for f in self.faces if len(f.triangles)]
        if not valid_faces:
            return 0.0
        return max(float(f.triangles[:, :, 2].max()) for f in valid_faces)

    def _tag_faces(self) -> dict[int, str]:
        """Classify each face by orientation relative to setup tool axis."""
        tags: dict[int, str] = {}
        cos_v = np.cos(np.radians(self._vertical_tol_deg))
        cos_h = np.cos(np.radians(90.0 - self._horizontal_tol_deg))

        for f in self.faces:
            if f.surface.kind == SurfaceKind.CYLINDER:
                tags[f.index] = "cylinder"
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
    # Cylindrical features: Holes, Bores, Counterbores, Countersinks, Bosses
    # -------------------------------------------------------------------------
    def _recognize_cylindrical_features(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        cyl_groups = self._group_cylinder_faces()

        # Step 1: Process each individual cylinder group
        cyl_info: list[dict] = []
        for g in cyl_groups:
            faces = [self.faces[i] for i in g]
            f0 = faces[0]
            sd = f0.surface
            axis = sd.axis / np.linalg.norm(sd.axis)
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
            })

        # Step 2: Classify bosses directly
        for info in cyl_info:
            if not info["is_internal"]:
                bmin = np.array([info["center_xy"][0] - info["radius"], info["center_xy"][1] - info["radius"], info["bottom_z"]])
                bmax = np.array([info["center_xy"][0] + info["radius"], info["center_xy"][1] + info["radius"], info["top_z"]])
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
                    is_concave=False,
                    accessibility=self._accessibility_of(info["group"]),
                    wall_faces=info["group"],
                    finish=self.requirements.get(info["group"][0]),
                    recognition_evidence=["external_cylindrical_surface", f"radius_{info['radius']:.2f}"],
                ))

        # Step 3: Group coaxial internal cylinders to recognize counterbores / stepped bores
        internal_cyls = [c for c in cyl_info if c["is_internal"]]
        coaxial_clusters: list[list[dict]] = []
        used_internal: set[int] = set()

        for i, c1 in enumerate(internal_cyls):
            if i in used_internal:
                continue
            cluster = [c1]
            used_internal.add(i)
            for j, c2 in enumerate(internal_cyls):
                if j in used_internal:
                    continue
                # Same axis location and direction
                if np.linalg.norm(c1["center_xy"] - c2["center_xy"]) < 0.1 and abs(abs(c1["axis"] @ c2["axis"]) - 1.0) < 1e-4:
                    cluster.append(c2)
                    used_internal.add(j)
            # Sort cluster from top Z downward
            cluster.sort(key=lambda x: -x["top_z"])
            coaxial_clusters.append(cluster)

        for cluster in coaxial_clusters:
            if len(cluster) == 1:
                # Single internal cylinder: hole or bore
                c = cluster[0]
                bmin = np.array([c["center_xy"][0] - c["radius"], c["center_xy"][1] - c["radius"], c["bottom_z"]])
                bmax = np.array([c["center_xy"][0] + c["radius"], c["center_xy"][1] + c["radius"], c["top_z"]])

                # Check through vs blind: does it reach model bottom?
                model_bottom_z = min(float(f.bounds_min[2]) for f in self.faces if len(f.triangles))
                is_through = c["bottom_z"] <= model_bottom_z + 0.5
                ftype = self._classify_hole_or_bore(c["diameter"], is_through)

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
                    is_concave=True,
                    accessibility=self._accessibility_of(c["group"]),
                    wall_faces=c["group"],
                    finish=self.requirements.get(c["group"][0]),
                    recognition_evidence=[
                        "internal_cylinder",
                        f"diameter_{c['diameter']:.2f}",
                        "through_opening" if is_through else "blind_bottom",
                    ],
                ))
            else:
                # Multiple coaxial cylinders -> Relational Counterbore + Through Hole / Bore
                parent_top = cluster[0]
                child_bottom = cluster[1]

                cbore_id = f"cbore_{parent_top['group'][0]}"
                child_id = f"hole_{child_bottom['group'][0]}"

                bmin_cb = np.array([parent_top["center_xy"][0] - parent_top["radius"], parent_top["center_xy"][1] - parent_top["radius"], parent_top["bottom_z"]])
                bmax_cb = np.array([parent_top["center_xy"][0] + parent_top["radius"], parent_top["center_xy"][1] + parent_top["radius"], parent_top["top_z"]])

                cbore_feat = MachiningFeature(
                    id=cbore_id,
                    type=FeatureType.COUNTERBORE,
                    face_indices=parent_top["group"],
                    bounds_min=bmin_cb,
                    bounds_max=bmax_cb,
                    depth=parent_top["depth"],
                    top_z=parent_top["top_z"],
                    floor_z=parent_top["bottom_z"],
                    axis=parent_top["axis"].copy(),
                    diameter=parent_top["diameter"],
                    is_concave=True,
                    accessibility=self._accessibility_of(parent_top["group"]),
                    wall_faces=parent_top["group"],
                    child_feature_ids=[child_id],
                    recognition_evidence=[
                        "coaxial_compound_feature",
                        f"cbore_outer_dia_{parent_top['diameter']:.2f}",
                        f"inner_dia_{child_bottom['diameter']:.2f}",
                    ],
                )

                bmin_ch = np.array([child_bottom["center_xy"][0] - child_bottom["radius"], child_bottom["center_xy"][1] - child_bottom["radius"], child_bottom["bottom_z"]])
                bmax_ch = np.array([child_bottom["center_xy"][0] + child_bottom["radius"], child_bottom["center_xy"][1] + child_bottom["radius"], child_bottom["top_z"]])

                model_bottom_z = min(float(f.bounds_min[2]) for f in self.faces if len(f.triangles))
                is_child_through = child_bottom["bottom_z"] <= model_bottom_z + 0.5
                child_ftype = self._classify_hole_or_bore(child_bottom["diameter"], is_child_through)
                child_feat = MachiningFeature(
                    id=child_id,
                    type=child_ftype,
                    face_indices=child_bottom["group"],
                    bounds_min=bmin_ch,
                    bounds_max=bmax_ch,
                    depth=child_bottom["depth"],
                    top_z=child_bottom["top_z"],
                    floor_z=child_bottom["bottom_z"],
                    axis=child_bottom["axis"].copy(),
                    diameter=child_bottom["diameter"],
                    is_concave=True,
                    accessibility=self._accessibility_of(child_bottom["group"]),
                    wall_faces=child_bottom["group"],
                    parent_feature_id=cbore_id,
                    recognition_evidence=[
                        "coaxial_child_bore",
                        f"diameter_{child_bottom['diameter']:.2f}",
                        "through_opening" if is_child_through else "blind_bottom",
                    ],
                )

                out.extend([cbore_feat, child_feat])

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
            csink_id = f"csink_{cf.index}"
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

            out.append(MachiningFeature(
                id=f"{ftype.value}_{floor_group[0]}",
                type=ftype,
                face_indices=all_feature_faces,
                bounds_min=np.array([bmin2[0], bmin2[1], floor_z]),
                bounds_max=np.array([bmax2[0], bmax2[1], top_z]),
                depth=depth,
                top_z=top_z,
                floor_z=floor_z,
                is_concave=True,
                accessibility=self._accessibility_of(floor_group),
                wall_faces=wall_faces_list,
                floor_faces=floor_group,
                boundary_loop=boundary_loop,
                island_loops=island_loops if island_loops else None,
                finish=self.requirements.get(floor_group[0]),
                recognition_evidence=[
                    f"type_{ftype.value}",
                    f"depth_{depth:.2f}",
                    f"floor_area_{sum(self.faces[i].area for i in floor_group):.1f}",
                    "open_boundary" if is_open else "closed_cavity",
                    f"islands_{len(island_loops)}" if island_loops else "no_islands",
                ],
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

        out.append(MachiningFeature(
            id=f"facing_{top_faces[0].index}",
            type=FeatureType.FACING_REGION,
            face_indices=face_indices,
            bounds_min=np.array([bmin2[0], bmin2[1], self._model_top_z]),
            bounds_max=np.array([bmax2[0], bmax2[1], self.stock_top_z]),
            depth=excess_height,
            top_z=self.stock_top_z,
            floor_z=self._model_top_z,
            is_concave=False,
            accessibility=Accessibility.TOOL_AXIS_OK,
            floor_faces=face_indices,
            notes={"excess_height": excess_height},
            recognition_evidence=[
                "unified_stock_top_material_removal",
                f"excess_height_{excess_height:.2f}",
            ],
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
                        is_concave=False,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        notes={"angle_from_horizontal": angle_from_horiz},
                        recognition_evidence=[f"chamfer_angle_{angle_from_horiz:.1f}deg"],
                    ))
        return out

    def _recognize_fillets(self) -> list[MachiningFeature]:
        out: list[MachiningFeature] = []
        for f in self.faces:
            if f.surface.kind in (SurfaceKind.CYLINDER, SurfaceKind.TORUS) and self._tags.get(f.index) in ("slanted", "wall"):
                radius = f.surface.radius
                if radius and radius < 10.0 and f.area < 200.0:
                    zs = f.triangles[:, :, 2]
                    top_z, bottom_z = float(zs.max()), float(zs.min())
                    out.append(MachiningFeature(
                        id=f"fillet_{f.index}",
                        type=FeatureType.FILLET,
                        face_indices=[f.index],
                        bounds_min=f.bounds_min,
                        bounds_max=f.bounds_max,
                        depth=top_z - bottom_z,
                        top_z=top_z,
                        floor_z=bottom_z,
                        radius=radius,
                        is_concave=True,
                        accessibility=Accessibility.TOOL_AXIS_OK,
                        recognition_evidence=[f"fillet_radius_{radius:.2f}"],
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
                    is_concave=True,
                    accessibility=self._accessibility_of([f.index]),
                    recognition_evidence=[f"surface_{f.surface.kind.value}"],
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
                        is_concave=True,
                        accessibility=f.accessibility,
                        wall_faces=f.wall_faces,
                        floor_faces=f.floor_faces,
                        parent_feature_id=f.parent_feature_id,
                        child_feature_ids=f.child_feature_ids,
                        recognition_evidence=f.recognition_evidence + ["threaded_hole_upgrade"],
                        notes=f.notes,
                        boundary_loop=f.boundary_loop,
                        island_loops=f.island_loops,
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
                        diameter=f.diameter,
                        is_concave=True,
                        accessibility=f.accessibility,
                        wall_faces=f.wall_faces,
                        floor_faces=f.floor_faces,
                        recognition_evidence=f.recognition_evidence + ["groove_from_slot"],
                        notes=f.notes,
                        boundary_loop=f.boundary_loop,
                        island_loops=f.island_loops,
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

        boundary_loop: Optional[list[np.ndarray]] = None
        for f in self.faces:
            if self._tags.get(f.index) in ("top", "bottom"):
                for w in f.wires:
                    if w.is_outer and len(w.points) >= 3:
                        boundary_loop = [p[:2].copy() for p in w.points]
                        break
            if boundary_loop is not None:
                break

        if boundary_loop is None:
            boundary_loop = [
                np.array([bmin[0], bmin[1]]),
                np.array([bmax[0], bmin[1]]),
                np.array([bmax[0], bmax[1]]),
                np.array([bmin[0], bmax[1]]),
            ]

        out.append(MachiningFeature(
            id="contour_outer",
            type=FeatureType.CONTOUR,
            face_indices=wall_faces if wall_faces else [f.index for f in self.faces],
            bounds_min=np.array([bmin[0], bmin[1], model_bottom_z]),
            bounds_max=np.array([bmax[0], bmax[1], self._model_top_z]),
            depth=depth,
            top_z=self._model_top_z,
            floor_z=model_bottom_z,
            is_concave=False,
            accessibility=Accessibility.TOOL_AXIS_OK,
            wall_faces=wall_faces,
            boundary_loop=boundary_loop,
            notes={"perimeter": True, "outer_contour": True},
            recognition_evidence=[
                "outer_part_boundary_profile",
                f"depth_{depth:.2f}",
                f"bounds_min_{bmin[0]:.1f}_{bmin[1]:.1f}",
                f"bounds_max_{bmax[0]:.1f}_{bmax[1]:.1f}",
            ],
        ))
        return out

    def _accessibility_of(self, face_ids: list[int]) -> Accessibility:
        mesh = TriangleMesh.from_faces(self.faces)
        if not len(mesh.triangles):
            return Accessibility.INACCESSIBLE
        worst = Accessibility.TOOL_AXIS_OK
        for fi in face_ids:
            f = self.faces[fi]
            pts = f.triangles.reshape(-1, 3)
            step = max(1, len(pts) // 12)
            for p in pts[::step]:
                hit = mesh.ray_first_hit(p + self.tool_axis * 1e-3, self.tool_axis, max_dist=1e4)
                if hit is not None and hit < 1e-1:
                    return Accessibility.INACCESSIBLE
            # Check if face normal requires tilted tool axis
            face_normal = f.oriented_normal
            if face_normal is not None:
                cos_angle = abs(np.dot(face_normal, self.tool_axis))
                # If face normal is more than vertical_tol_deg from tool axis,
                # it needs a tilted tool (but is still accessible)
                if cos_angle < np.cos(np.radians(self._vertical_tol_deg)):
                    worst = Accessibility.NEEDS_TILTED_TOOL
        return worst

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
