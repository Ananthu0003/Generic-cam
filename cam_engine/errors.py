"""Explicit error taxonomy. No stage silently recovers from missing data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorSeverity(str, Enum):
    ERROR = "error"        # blocks the pipeline
    WARNING = "warning"    # recorded, pipeline continues


@dataclass
class CamError(Exception):
    """A planning / generation failure with full provenance."""

    code: str
    message: str
    stage: str = ""
    feature_id: Optional[str] = None
    operation_id: Optional[str] = None
    tool_id: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.ERROR
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        loc = []
        for key, val in (
            ("stage", self.stage),
            ("feature", self.feature_id),
            ("operation", self.operation_id),
            ("tool", self.tool_id),
        ):
            if val:
                loc.append(f"{key}={val}")
        prefix = f"[{self.code}]"
        if loc:
            prefix += " (" + ", ".join(loc) + ")"
        return f"{prefix}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "feature_id": self.feature_id,
            "operation_id": self.operation_id,
            "tool_id": self.tool_id,
            "severity": self.severity.value,
            "context": self.context,
        }


# --- error codes (explicit, never silently recovered) ---
MISSING_STOCK_GEOMETRY = "MISSING_STOCK_GEOMETRY"
INVALID_SETUP_TRANSFORM = "INVALID_SETUP_TRANSFORM"
FEATURE_INACCESSIBLE = "FEATURE_INACCESSIBLE"
NO_COMPATIBLE_TOOL = "NO_COMPATIBLE_TOOL"
TOOL_REACH_INSUFFICIENT = "TOOL_REACH_INSUFFICIENT"
MACHINE_AXIS_LIMIT = "MACHINE_AXIS_LIMIT"
COLLISION_DETECTED = "COLLISION_DETECTED"
GOUGE_DETECTED = "GOUGE_DETECTED"
INVALID_TOOLPATH = "INVALID_TOOLPATH"
UNSUPPORTED_FEATURE = "UNSUPPORTED_FEATURE"
INVALID_CUTTING_PARAMETERS = "INVALID_CUTTING_PARAMETERS"
SIMULATION_MISMATCH = "SIMULATION_MISMATCH"
INVALID_GEOMETRY = "INVALID_GEOMETRY"
INVALID_SETUP = "INVALID_SETUP"
INVALID_PLANNING = "INVALID_PLANNING"
INVALID_CONTROLLER_PROFILE = "INVALID_CONTROLLER_PROFILE"
UNSUPPORTED_G_CODE = "UNSUPPORTED_G_CODE"
UNSUPPORTED_M_CODE = "UNSUPPORTED_M_CODE"
INVALID_WORK_OFFSET = "INVALID_WORK_OFFSET"
MISSING_TOOL_OFFSET = "MISSING_TOOL_OFFSET"
INVALID_TOOL_LENGTH_OFFSET = "INVALID_TOOL_LENGTH_OFFSET"
INVALID_COORDINATE_SYSTEM = "INVALID_COORDINATE_SYSTEM"
INVALID_MODAL_STATE = "INVALID_MODAL_STATE"
MACHINE_LIMIT_EXCEEDED = "MACHINE_LIMIT_EXCEEDED"
INVALID_ARC = "INVALID_ARC"
INVALID_TOOL_STATE = "INVALID_TOOL_STATE"
INVALID_CANNED_CYCLE = "INVALID_CANNED_CYCLE"
POST_PROCESSING_FAILED = "POST_PROCESSING_FAILED"
GCODE_VALIDATION_FAILED = "GCODE_VALIDATION_FAILED"
SIMULATION_FAILED = "SIMULATION_FAILED"
MISSING_MATERIAL_DATA = "MISSING_MATERIAL_DATA"
MISSING_MACHINE_DATA = "MISSING_MACHINE_DATA"
UNIT_MISMATCH = "UNIT_MISMATCH"


class PlanningBlocker(Exception):
    """Aggregated blockers for an operation: the pipeline must stop, not guess."""

    def __init__(self, errors: list[CamError]):
        self.errors = errors
        super().__init__("; ".join(str(e) for e in errors))
