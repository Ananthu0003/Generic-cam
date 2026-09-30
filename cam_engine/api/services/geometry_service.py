"""Geometry helper services.

Pure functions — no FastAPI coupling.  Extracted from the old ``utils.py``.
"""

from __future__ import annotations

import numpy as np

from ...demo_models import DemoCADModel
from ...geometry.topology import Shape
from ...context import Tool
from ..schemas import MeshData, BoundingBox


# ---------------------------------------------------------------------------
# Rotation matrix helpers
# ---------------------------------------------------------------------------

def euler_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build a 4×4 homogeneous rotation matrix from ZYX Euler angles (degrees)."""
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)

    R = Rz @ Ry @ Rx
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


# ---------------------------------------------------------------------------
# Shape / mesh conversion helpers
# ---------------------------------------------------------------------------

def shape_to_demo_cad_model(
    name: str, shape: Shape, desc: str = "Imported Model"
) -> DemoCADModel:
    """Convert a B-Rep ``Shape`` into a flat ``DemoCADModel`` (vertex soup)."""
    vertices: list[float] = []
    normals_list: list[float] = []
    indices: list[int] = []
    faces_meta: list[dict] = []

    for f in shape.faces():
        if not len(f.triangles):
            continue
        base_idx = len(vertices) // 3
        tris = f.triangles
        n_tris = len(tris)
        norm = f.oriented_normal
        for t in tris:
            for p in t:
                vertices.extend([float(p[0]), float(p[1]), float(p[2])])
                normals_list.extend([float(norm[0]), float(norm[1]), float(norm[2])])
        for i in range(n_tris * 3):
            indices.append(base_idx + i)
        faces_meta.append(
            {"id": f.index, "kind": f.surface.kind.value, "area": float(f.area)}
        )

    bmin, bmax = shape.bounding_box()
    return DemoCADModel(
        name=name,
        description=desc,
        bounds_min=[float(x) for x in bmin],
        bounds_max=[float(x) for x in bmax],
        vertices=vertices,
        normals=normals_list,
        indices=indices,
        faces_metadata=faces_meta,
    )


def mesh_to_response(demo: DemoCADModel) -> MeshData:
    """Convert a ``DemoCADModel`` to the ``MeshData`` API response schema."""
    bmin = demo.bounds_min
    bmax = demo.bounds_max
    size = [bmax[i] - bmin[i] for i in range(3)]
    return MeshData(
        vertices=demo.vertices,
        normals=demo.normals,
        indices=demo.indices,
        triangle_count=len(demo.indices) // 3,
        face_count=len(demo.faces_metadata),
        bounding_box=BoundingBox(min=bmin, max=bmax, size=size),
    )


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

def tool_name(t: Tool) -> str:
    """Return a human-readable label for a tool (used in operation responses)."""
    return f"{t.id} ({t.type.value.replace('_', ' ').title()} Ø{t.diameter}mm)"
