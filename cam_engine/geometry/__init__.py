"""Geometry layer: STEP/STL/IGES import, topology, mesh queries, voxel stock."""

from .step_import import ImportedModel, import_step
from .stl_import import import_stl, import_iges, STLImportError
from .topology import Shape, Face, Edge, SurfaceData, CurveData, SurfaceKind, CurveKind
from .mesh import TriangleMesh
from .voxel import VoxelStock
from .representation import (
    CadSurfaceKind,
    NominalCadSurface,
    NominalCadBoundary,
    NominalCadCurve,
    DesignGeometry,
    ToolpathBoundary,
    ToolCenterPath,
    StockBoundary,
    CuttingEnvelope,
    ManufacturingGeometry,
    ManufacturingOffsetCalculator,
)

from .revolved import (
    RevolvedProfileSegmentKind,
    RevolvedProfileSegment,
    RevolvedEdgeTransition,
    TransitionKind,
    RevolvedProfile,
    RevolvedGeometryRecognizer,
)

__all__ = [
    "ImportedModel", "import_step",
    "import_stl", "import_iges", "STLImportError",
    "Shape", "Face", "Edge", "SurfaceData", "CurveData", "SurfaceKind", "CurveKind",
    "TriangleMesh", "VoxelStock",
    "CadSurfaceKind",
    "NominalCadSurface",
    "NominalCadBoundary",
    "NominalCadCurve",
    "DesignGeometry",
    "ToolpathBoundary",
    "ToolCenterPath",
    "StockBoundary",
    "CuttingEnvelope",
    "ManufacturingGeometry",
    "ManufacturingOffsetCalculator",
    "RevolvedProfileSegmentKind",
    "RevolvedProfileSegment",
    "RevolvedEdgeTransition",
    "TransitionKind",
    "RevolvedProfile",
    "RevolvedGeometryRecognizer",
]

