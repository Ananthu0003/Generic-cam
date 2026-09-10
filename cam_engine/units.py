"""Unit handling: the CAM domain works internally in millimeters (consistent SI-derived).

Conversions happen ONLY at explicit boundaries:
- import: user-declared file units -> mm
- export: mm -> controller-configured units (post processor)
Never a hidden scale factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import UNIT_MISMATCH, CamError


class Units(str, Enum):
    MM = "mm"
    INCH = "inch"


MM_PER_INCH = 25.4  # exact definition, not a model-specific constant


@dataclass(frozen=True)
class UnitConverter:
    """Converts between the internal domain unit (mm) and a target unit."""

    internal: Units = Units.MM

    def to_internal(self, value: float, value_units: Units) -> float:
        if value_units is self.internal:
            return float(value)
        if self.internal is Units.MM and value_units is Units.INCH:
            return float(value) * MM_PER_INCH
        if self.internal is Units.INCH and value_units is Units.MM:
            return float(value) / MM_PER_INCH
        raise CamError(
            code=UNIT_MISMATCH,
            message=f"Unsupported unit conversion {self.internal} -> {value_units}",
            stage="units",
        )

    def from_internal(self, value: float, target_units: Units) -> float:
        return UnitConverter(target_units).to_internal(value, self.internal)

    def factor_to_internal(self, value_units: Units) -> float:
        return self.to_internal(1.0, value_units)
