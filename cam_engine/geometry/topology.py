"""B-Rep topology extraction on top of OpenCascade.

Everything the feature recognizer uses comes from here: real faces, real edges,
real surface types, real adjacency, exact wire loops, and geometric normals.
No synthesized geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SurfaceKind(str, Enum):
    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    BSPLINE = "bspline"
    OTHER = "other"


class CurveKind(str, Enum):
    LINE = "line"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    BSPLINE = "bspline"
    OTHER = "other"


@dataclass
class SurfaceData:
    kind: SurfaceKind
    normal: Optional[np.ndarray] = None      # plane normal (unoriented), plane only
    point_on: Optional[np.ndarray] = None    # plane origin / axis location
    axis: Optional[np.ndarray] = None        # cylinder/cone axis direction
    radius: Optional[float] = None           # cylinder radius
    semi_angle: Optional[float] = None       # cone semi-angle (rad)
    ref_radius: Optional[float] = None       # cone reference radius
    u_min: Optional[float] = None            # first U parameter (radians for cylinder/cone/torus)
    u_max: Optional[float] = None            # last U parameter
    u_span: Optional[float] = None           # angular / parametric span in U


@dataclass
class CurveData:
    kind: CurveKind
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # discretized
    circle_center: Optional[np.ndarray] = None
    circle_axis: Optional[np.ndarray] = None
    circle_radius: Optional[float] = None
    closed: bool = False


@dataclass
class Edge:
    index: int
    curve: CurveData
    faces: tuple[int, int] = (-1, -1)  # face indices sharing this edge (-1 = open boundary)


@dataclass
class WireLoop:
    is_outer: bool
    points: np.ndarray                   # (N, 3) curve points in 3D
    edges: list[int] = field(default_factory=list)


@dataclass
class Face:
    index: int
    surface: SurfaceData
    oriented_normal: np.ndarray          # outward normal at surface param center
    triangles: np.ndarray                # (N,3,3) float, model space, outward winding
    area: float                          # approximated from tessellation
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    edge_indices: list[int] = field(default_factory=list)
    wires: list[WireLoop] = field(default_factory=list)
    is_internal: Optional[bool] = None   # for cylinders: True if internal hole/bore, False if boss
    kind_tag: str = ""                   # recognizer tag: wall/floor/top/freeform/...
    occ_face: Optional[object] = None

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.bounds_min + self.bounds_max)


@dataclass
class Shape:
    """TopoDS_Shape wrapper with lazy extraction into plain numpy structures."""

    occ_shape: object
    _faces: Optional[list[Face]] = None
    _edges: Optional[list[Edge]] = None
    _adjacency: Optional[dict[int, list[int]]] = None

    # ---------- basic queries ----------
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        box = Bnd_Box()
        BRepBndLib.Add_s(self.occ_shape, box, True)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        return np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])

    def scaled(self, factor: float) -> "Shape":
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.gp import gp_Trsf, gp_Pnt
        trsf = gp_Trsf()
        trsf.SetScale(gp_Pnt(0, 0, 0), factor)
        xf = BRepBuilderAPI_Transform(self.occ_shape, trsf, True)
        return Shape(xf.Shape())

    def transformed(self, trsf_4x4: np.ndarray) -> "Shape":
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.gp import gp_Trsf
        r = np.asarray(trsf_4x4)[:3, :3]
        t = np.asarray(trsf_4x4)[:3, 3]
        gp_trsf = gp_Trsf()
        gp_trsf.SetValues(r[0, 0], r[0, 1], r[0, 2], t[0],
                          r[1, 0], r[1, 1], r[1, 2], t[1],
                          r[2, 0], r[2, 1], r[2, 2], t[2])
        xf = BRepBuilderAPI_Transform(self.occ_shape, gp_trsf, True)
        return Shape(xf.Shape())

    def rotate(self, axis_name: str, angle_deg: float) -> "Shape":
        """Rotate the shape around X, Y, or Z axis by angle_deg degrees centered at bounding box center."""
        bmin, bmax = self.bounding_box()
        center = 0.5 * (bmin + bmax)
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        if axis_name.upper() == "X":
            rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
        elif axis_name.upper() == "Y":
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
        else:  # Z
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

        t1 = np.eye(4)
        t1[:3, 3] = -center
        r4 = np.eye(4)
        r4[:3, :3] = rot
        t2 = np.eye(4)
        t2[:3, 3] = center
        full_xf = t2 @ r4 @ t1
        return self.transformed(full_xf)

    def align_face_to_axis(self, face_index: int, target_axis: Optional[np.ndarray] = None) -> "Shape":
        """Rotate shape so that the normal of the specified face aligns with target_axis (default [0,0,1])."""
        target = np.array([0.0, 0.0, 1.0]) if target_axis is None else np.asarray(target_axis, dtype=float)
        target = target / np.linalg.norm(target)
        faces = self.faces()
        if face_index < 0 or face_index >= len(faces):
            return self
        fn = faces[face_index].oriented_normal
        n_len = np.linalg.norm(fn)
        if n_len < 1e-6:
            return self
        src = fn / n_len
        dot = np.clip(float(src @ target), -1.0, 1.0)
        if np.isclose(dot, 1.0):
            return self
        if np.isclose(dot, -1.0):
            ortho = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(src, ortho)
            axis = axis / np.linalg.norm(axis)
            angle = np.pi
        else:
            axis = np.cross(src, target)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)

        from ..coords import rotation_about_axis
        rot3 = rotation_about_axis(axis, angle)
        bmin, bmax = self.bounding_box()
        center = 0.5 * (bmin + bmax)
        t1 = np.eye(4)
        t1[:3, 3] = -center
        r4 = np.eye(4)
        r4[:3, :3] = rot3
        t2 = np.eye(4)
        t2[:3, 3] = center
        return self.transformed(t2 @ r4 @ t1)

    # ---------- extraction ----------
    def faces(self, linear_deflection: float = 0.2, angular_deflection: float = 0.25) -> list[Face]:
        """Extract faces with tessellation, surface kinds, outward normals, area, and wire loops."""
        if self._faces is not None:
            return self._faces
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.TopoDS import TopoDS

        BRepMesh_IncrementalMesh(self.occ_shape, linear_deflection, False,
                                 angular_deflection, True)

        fmap = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(self.occ_shape, TopAbs_FACE, fmap)

        faces: list[Face] = []
        for idx in range(1, fmap.Extent() + 1):
            occ_face = TopoDS.Face_s(fmap.FindKey(idx))
            faces.append(self._extract_face(idx - 1, occ_face))
        self._faces = faces
        return faces

    def _extract_face(self, index: int, occ_face) -> Face:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepLProp import BRepLProp_SLProps
        from OCP.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                                 GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface)
        from OCP.TopAbs import TopAbs_REVERSED, TopAbs_WIRE, TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.BRepTools import BRepTools

        adaptor = BRepAdaptor_Surface(occ_face)
        st = adaptor.GetType()

        u_min = float(adaptor.FirstUParameter())
        u_max = float(adaptor.LastUParameter())
        u_span = float(u_max - u_min)

        sd = SurfaceData(kind=SurfaceKind.OTHER, u_min=u_min, u_max=u_max, u_span=u_span)
        if st == GeomAbs_Plane:
            pln = adaptor.Plane()
            loc, ax = pln.Location(), pln.Axis().Direction()
            sd = SurfaceData(kind=SurfaceKind.PLANE,
                             point_on=np.array([loc.X(), loc.Y(), loc.Z()]),
                             normal=np.array([ax.X(), ax.Y(), ax.Z()]),
                             u_min=u_min, u_max=u_max, u_span=u_span)
        elif st == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            loc, ax = cyl.Location(), cyl.Axis().Direction()
            sd = SurfaceData(kind=SurfaceKind.CYLINDER,
                             point_on=np.array([loc.X(), loc.Y(), loc.Z()]),
                             axis=np.array([ax.X(), ax.Y(), ax.Z()]),
                             radius=cyl.Radius(),
                             u_min=u_min, u_max=u_max, u_span=u_span)
        elif st == GeomAbs_Cone:
            cone = adaptor.Cone()
            loc, ax = cone.Location(), cone.Axis().Direction()
            sd = SurfaceData(kind=SurfaceKind.CONE,
                             point_on=np.array([loc.X(), loc.Y(), loc.Z()]),
                             axis=np.array([ax.X(), ax.Y(), ax.Z()]),
                             semi_angle=cone.SemiAngle(), ref_radius=cone.RefRadius(),
                             u_min=u_min, u_max=u_max, u_span=u_span)
        elif st == GeomAbs_Sphere:
            sph = adaptor.Sphere()
            loc = sph.Location()
            sd = SurfaceData(kind=SurfaceKind.SPHERE,
                             point_on=np.array([loc.X(), loc.Y(), loc.Z()]),
                             radius=sph.Radius(),
                             u_min=u_min, u_max=u_max, u_span=u_span)
        elif st == GeomAbs_Torus:
            tor = adaptor.Torus()
            loc, ax = tor.Location(), tor.Axis().Direction()
            sd = SurfaceData(kind=SurfaceKind.TORUS,
                             point_on=np.array([loc.X(), loc.Y(), loc.Z()]),
                             axis=np.array([ax.X(), ax.Y(), ax.Z()]),
                             radius=tor.MajorRadius(), ref_radius=tor.MinorRadius(),
                             u_min=u_min, u_max=u_max, u_span=u_span)
        elif st == GeomAbs_BSplineSurface:
            sd = SurfaceData(kind=SurfaceKind.BSPLINE,
                             u_min=u_min, u_max=u_max, u_span=u_span)

        # Exact oriented surface normal evaluation
        sign = -1.0 if occ_face.Orientation() == TopAbs_REVERSED else 1.0
        u_mid = 0.5 * (adaptor.FirstUParameter() + adaptor.LastUParameter())
        v_mid = 0.5 * (adaptor.FirstVParameter() + adaptor.LastVParameter())
        
        oriented_normal = np.array([0.0, 0.0, 1.0])
        is_internal_cyl: Optional[bool] = None

        if st == GeomAbs_Plane and sd.normal is not None:
            oriented_normal = sign * sd.normal.copy()
            norm_len = np.linalg.norm(oriented_normal)
            if norm_len > 1e-12:
                oriented_normal /= norm_len
        else:
            try:
                props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)
                if props.IsNormalDefined():
                    gp_n = props.Normal()
                    raw_n = np.array([gp_n.X(), gp_n.Y(), gp_n.Z()])
                    oriented_normal = sign * raw_n
                    norm_len = np.linalg.norm(oriented_normal)
                    if norm_len > 1e-12:
                        oriented_normal /= norm_len
                    
                    # For cylinder: check if internal void (hole/bore) or external boss
                    if st == GeomAbs_Cylinder and sd.point_on is not None and sd.axis is not None:
                        p_mid_gp = adaptor.Value(u_mid, v_mid)
                        p_mid = np.array([p_mid_gp.X(), p_mid_gp.Y(), p_mid_gp.Z()])
                        axis_dir = sd.axis / np.linalg.norm(sd.axis)
                        d = p_mid - sd.point_on
                        p_axis = sd.point_on + (d @ axis_dir) * axis_dir
                        radial_vec = p_mid - p_axis
                        # If oriented normal opposes radial outward vector, it points into void -> internal hole
                        dot_rad = float(oriented_normal @ radial_vec)
                        is_internal_cyl = (dot_rad < -1e-4)
            except Exception:
                if sd.normal is not None:
                    oriented_normal = sign * sd.normal.copy()

        # Tessellation
        from OCP.TopLoc import TopLoc_Location
        loc_topo = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(occ_face, loc_topo)
        tris = np.zeros((0, 3, 3))
        if tri is not None:
            n = tri.NbNodes()
            pts = np.empty((n, 3))
            trsf = loc_topo.Transformation() if not loc_topo.IsIdentity() else None
            for i in range(1, n + 1):
                p = tri.Node(i)
                if trsf is not None:
                    p = p.Transformed(trsf)
                pts[i - 1] = (p.X(), p.Y(), p.Z())
            m = tri.NbTriangles()
            tris = np.empty((m, 3, 3))
            for i in range(1, m + 1):
                a, b, c = tri.Triangle(i).Get()
                tris[i - 1] = (pts[a - 1], pts[b - 1], pts[c - 1])
            if occ_face.Orientation() == TopAbs_REVERSED:
                tris = tris[:, [0, 2, 1], :]

        area = 0.0
        if len(tris):
            cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            area = 0.5 * float(np.linalg.norm(cross, axis=1).sum())

        flat = tris.reshape(-1, 3) if len(tris) else np.zeros((0, 3))
        bmin = flat.min(axis=0) if len(flat) else np.zeros(3)
        bmax = flat.max(axis=0) if len(flat) else np.zeros(3)

        # Extract wire loops
        wires: list[WireLoop] = []
        outer_shape = None
        try:
            outer_w = BRepTools.OuterWire_s(occ_face)
            if outer_w is not None and hasattr(outer_w, "IsNull") and not outer_w.IsNull():
                outer_shape = outer_w.TShape()
        except Exception:
            outer_shape = None

        wexp = TopExp_Explorer(occ_face, TopAbs_WIRE)
        while wexp.More():
            w = TopoDS.Wire_s(wexp.Current())
            is_out = (outer_shape is not None and hasattr(w, "IsNull") and not w.IsNull() and w.TShape() == outer_shape)
            wire_pts: list[np.ndarray] = []
            wire_edges: list[int] = []
            eexp = TopExp_Explorer(w, TopAbs_EDGE)
            while eexp.More():
                e = TopoDS.Edge_s(eexp.Current())
                try:
                    from OCP.BRepAdaptor import BRepAdaptor_Curve
                    crv = BRepAdaptor_Curve(e)
                    u0, u1 = crv.FirstParameter(), crv.LastParameter()
                    n_samples = max(4, min(100, int(crv.Value(u0).Distance(crv.Value(u1)) / 0.5) + 1))
                    for u in np.linspace(u0, u1, n_samples):
                        p = crv.Value(float(u))
                        wire_pts.append(np.array([p.X(), p.Y(), p.Z()]))
                except Exception:
                    pass
                eexp.Next()
            if wire_pts:
                wires.append(WireLoop(is_outer=is_out, points=np.array(wire_pts), edges=wire_edges))
            wexp.Next()

        return Face(
            index=index,
            surface=sd,
            oriented_normal=oriented_normal,
            triangles=tris,
            area=area,
            bounds_min=bmin,
            bounds_max=bmax,
            wires=wires,
            is_internal=is_internal_cyl,
            occ_face=occ_face,
        )

    # ---------- edges ----------
    def edges(self, deflection: float = 0.1) -> list[Edge]:
        if self._edges is not None:
            return self._edges
        from OCP.TopExp import TopExp
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.TopoDS import TopoDS

        emap = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(self.occ_shape, TopAbs_EDGE, emap)

        edges: list[Edge] = []
        for idx in range(1, emap.Extent() + 1):
            occ_edge = TopoDS.Edge_s(emap.FindKey(idx))
            edges.append(self._extract_edge(idx - 1, occ_edge, deflection))
        self._edges = edges
        return edges

    def _extract_edge(self, index: int, occ_edge, deflection: float) -> Edge:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve

        curve = BRepAdaptor_Curve(occ_edge)
        ctype = curve.GetType()
        u0, u1 = curve.FirstParameter(), curve.LastParameter()

        cd = CurveData(kind=CurveKind.OTHER)
        try:
            p0 = curve.Value(u0)
            p1 = curve.Value(u1)
            approx_len = p0.Distance(p1)
        except Exception:
            approx_len = 1.0
        npts = max(8, min(400, int(approx_len / max(deflection, 1e-3)) + 1))
        us = np.linspace(u0, u1, npts)
        pts = np.empty((npts, 3))
        for i, u in enumerate(us):
            p = curve.Value(float(u))
            pts[i] = (p.X(), p.Y(), p.Z())

        closed = bool(np.allclose(pts[0], pts[-1], atol=1e-6))
        if ctype == GeomAbs_Line:
            cd.kind = CurveKind.LINE
        elif ctype == GeomAbs_Circle:
            circ = curve.Circle()
            ax, loc = circ.Axis().Direction(), circ.Location()
            cd = CurveData(kind=CurveKind.CIRCLE, points=pts, closed=closed,
                           circle_center=np.array([loc.X(), loc.Y(), loc.Z()]),
                           circle_axis=np.array([ax.X(), ax.Y(), ax.Z()]),
                           circle_radius=circ.Radius())
        elif ctype == GeomAbs_Ellipse:
            cd.kind = CurveKind.ELLIPSE
        elif ctype == GeomAbs_BSplineCurve:
            cd.kind = CurveKind.BSPLINE
        cd.points = pts
        cd.closed = closed
        return Edge(index=index, curve=cd)

    # ---------- adjacency ----------
    def face_adjacency(self) -> dict[int, list[int]]:
        """Face adjacency via shared edges. Uses real topology, not proximity."""
        if self._adjacency is not None:
            return self._adjacency
        from OCP.TopExp import TopExp
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

        fmap = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(self.occ_shape, TopAbs_FACE, fmap)

        amap = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(self.occ_shape, TopAbs_EDGE, TopAbs_FACE, amap)

        adjacency: dict[int, list[int]] = {i: [] for i in range(fmap.Extent())}
        for i in range(1, amap.Extent() + 1):
            flist = amap.FindFromIndex(i)
            fidxs = [fmap.FindIndex(f) - 1 for f in flist]
            if len(fidxs) == 2 and fidxs[0] >= 0 and fidxs[1] >= 0:
                if fidxs[1] not in adjacency[fidxs[0]]:
                    adjacency[fidxs[0]].append(fidxs[1])
                if fidxs[0] not in adjacency[fidxs[1]]:
                    adjacency[fidxs[1]].append(fidxs[0])
        self._adjacency = adjacency
        return adjacency
