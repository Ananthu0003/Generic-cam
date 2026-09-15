"""STEP import through OpenCascade (OCP). Geometry is the only source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import CamError, INVALID_GEOMETRY
from ..units import Units, UnitConverter
from .topology import Shape


@dataclass
class ImportedModel:
    """An imported CAD model in internal units (mm), model coordinate space."""

    shape: Shape
    declared_units: Units
    original_bounds: tuple[np.ndarray, np.ndarray]  # min, max in model space (mm)


def import_step(path: str | Path, file_units: Optional[Units] = None) -> ImportedModel:
    """Import a STEP file.

    file_units: if the STEP file declares units, OpenCascade normally converts
    them to mm on import; when a user overrides file_units we apply the explicit
    conversion ourselves. No silent scaling ever happens.
    """
    path = Path(path)
    if not path.exists():
        raise CamError(INVALID_GEOMETRY, f"STEP file not found: {path}", stage="import")

    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHELL

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise CamError(
            INVALID_GEOMETRY,
            f"STEP read failed with status code {status}",
            stage="import",
        )
    reader.TransferRoots()
    if reader.NbShapes() < 1:
        raise CamError(INVALID_GEOMETRY, "STEP file contains no shapes", stage="import")
    
    # Use reader.Shape() or OneShape() safely
    occ_shape = None
    try:
        occ_shape = reader.Shape()
        if occ_shape is not None and hasattr(occ_shape, "IsNull") and occ_shape.IsNull():
            occ_shape = reader.OneShape()
    except Exception:
        try:
            occ_shape = reader.OneShape()
        except Exception:
            pass

    if occ_shape is None or (hasattr(occ_shape, "IsNull") and occ_shape.IsNull()):
        raise CamError(INVALID_GEOMETRY, "Failed to extract valid 3D shape from STEP file", stage="import")

    # Count faces/solids to ensure machinable geometry exists
    exp_faces = TopExp_Explorer(occ_shape, TopAbs_FACE)
    n_faces = 0
    while exp_faces.More():
        n_faces += 1
        exp_faces.Next()

    if n_faces == 0:
        raise CamError(INVALID_GEOMETRY, "STEP contains no faces or solid geometry", stage="import")

    units = file_units if file_units is not None else Units.MM
    converter = UnitConverter(internal=Units.MM)
    scale = converter.factor_to_internal(units)

    shape = Shape(occ_shape)
    if abs(scale - 1.0) > 1e-12:
        shape = shape.scaled(scale)

    bbox = shape.bounding_box()
    return ImportedModel(
        shape=shape,
        declared_units=units,
        original_bounds=(bbox[0], bbox[1]),
    )
