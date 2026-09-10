"""Geometry layer: STEP/STL/IGES import, topology, mesh queries, voxel stock."""

from .step_import import ImportedModel, import_step
from .stl_import import import_stl, import_iges, STLImportError
from .topology import Shape, Face, Edge, SurfaceData, CurveData, SurfaceKind, CurveKind
from .mesh import TriangleMesh
from .voxel import VoxelStock

__all__ = [
    "ImportedModel", "import_step",
    "import_stl", "import_iges", "STLImportError",
    "Shape", "Face", "Edge", "SurfaceData", "CurveData", "SurfaceKind", "CurveKind",
    "TriangleMesh", "VoxelStock",
]
