"""STL file import for triangle mesh geometry.

Supports both ASCII and binary STL formats. Imported meshes are converted
to the same TriangleMesh format used throughout the pipeline.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

from .mesh import TriangleMesh


class STLImportError(Exception):
    pass


def import_stl(path: str | Path) -> TriangleMesh:
    """Import an STL file (ASCII or binary) and return a TriangleMesh."""
    path = Path(path)
    if not path.exists():
        raise STLImportError(f"STL file not found: {path}")

    with open(path, "rb") as f:
        header = f.read(80)

        if header[:5] == b'solid' and b'\n' in header:
            try:
                f.seek(0)
                return _import_ascii_stl(f)
            except Exception:
                pass

        f.seek(0)
        return _import_binary_stl(f)


def _import_binary_stl(f) -> TriangleMesh:
    """Import binary STL format."""
    header = f.read(80)
    num_triangles = struct.unpack("<I", f.read(4))[0]

    if num_triangles <= 0 or num_triangles > 10_000_000:
        raise STLImportError(f"Invalid triangle count: {num_triangles}")

    triangles = np.zeros((num_triangles, 3, 3), dtype=np.float64)

    for i in range(num_triangles):
        data = struct.unpack("<12fH", f.read(50))
        for v in range(3):
            triangles[i, v, 0] = data[3 + v * 3]
            triangles[i, v, 1] = data[4 + v * 3]
            triangles[i, v, 2] = data[5 + v * 3]

    return TriangleMesh(triangles=triangles)


def _import_ascii_stl(f) -> TriangleMesh:
    """Import ASCII STL format."""
    triangles = []
    vertices = []
    in_triangle = False

    for line in f:
        line = line.decode("ascii", errors="ignore").strip().lower()
        if line.startswith("facet normal"):
            in_triangle = True
            vertices = []
        elif line.startswith("vertex"):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("endfacet"):
            if in_triangle and len(vertices) == 3:
                triangles.append(vertices)
            in_triangle = False
            vertices = []

    if not triangles:
        raise STLImportError("No triangles found in ASCII STL file")

    tri_array = np.array(triangles, dtype=np.float64)
    return TriangleMesh(triangles=tri_array)


def import_iges(path: str | Path) -> TriangleMesh:
    """Import an IGES file via OpenCascade tessellation.
    Falls back to error if OCP is not available."""
    try:
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.IGESCAFControl import IGESCAFControl_Reader
        from OCP.TDocStd_Document import TDocStd_Document
        from OCP.XSControl_WorkSession import XSControl_WorkSession

        path = Path(path)
        if not path.exists():
            raise STLImportError(f"IGES file not found: {path}")

        reader = IGESCAFControl_Reader()
        status = reader.ReadFile(str(path))
        if status != 1:
            raise STLImportError(f"Failed to read IGES file: {path}")

        doc = TDocStd_Document("XmlOcaf")
        reader.Transfer(doc)

        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS_Shape
        from OCP.BRep import BRep_Tool
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE

        explorer = TopExp_Explorer(doc.Main(), TopAbs_FACE)
        all_tris = []
        while explorer.More():
            face = explorer.Current()
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, loc)
            if triangulation is not None:
                nb_nodes = triangulation.NbNodes()
                nb_triangles = triangulation.NbTriangles()
                if nb_triangles > 0:
                    nodes = []
                    for i in range(1, nb_nodes + 1):
                        node = triangulation.Node(i)
                        nodes.append([node.X(), node.Y(), node.Z()])
                    nodes = np.array(nodes)

                    tris = []
                    for i in range(1, nb_triangles + 1):
                        tri = triangulation.Triangle(i)
                        n1, n2, n3 = tri.Get()
                        tris.append([nodes[n1 - 1], nodes[n2 - 1], nodes[n3 - 1]])
                    all_tris.extend(tris)
            explorer.Next()

        if not all_tris:
            raise STLImportError("No triangulation found in IGES file")

        return TriangleMesh(triangles=np.array(all_tris, dtype=np.float64))

    except ImportError:
        raise STLImportError(
            "IGES import requires OpenCascade (OCP) Python bindings. "
            "Install with: pip install cadquery-ocp"
        )
