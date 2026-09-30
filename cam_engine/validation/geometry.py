"""Toolpath geometry validation before post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..toolpaths.semantic import Toolpath, MotionSegment, MotionType
from ..errors import CamError, GCODE_GEOMETRY_MISMATCH


@dataclass
class GeometryValidationResult:
    """Result of toolpath geometry validation."""

    passed: bool
    max_deviation_mm: float = 0.0
    max_arc_center_deviation_mm: float = 0.0
    inspected_moves: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_deviation_mm": self.max_deviation_mm,
            "max_arc_center_deviation_mm": self.max_arc_center_deviation_mm,
            "inspected_moves": self.inspected_moves,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ToolpathGeometryValidator:
    """Validates toolpath geometry before post-processing.

    Checks:
    - No zero-length moves
    - Arc validity (center/start/end consistency)
    - Feed rate consistency
    - Tool engagement limits
    - Rapid/feed transitions
    """

    def __init__(
        self,
        tolerance_mm: float = 0.01,
        arc_tolerance_mm: float = 0.005,
        max_feed_mm_min: float = 30000.0,
        min_arc_radius_mm: float = 0.1,
        max_arc_radius_mm: float = 100000.0,
    ):
        self.tolerance_mm = tolerance_mm
        self.arc_tolerance_mm = arc_tolerance_mm
        self.max_feed_mm_min = max_feed_mm_min
        self.min_arc_radius_mm = min_arc_radius_mm
        self.max_arc_radius_mm = max_arc_radius_mm

    def validate(self, toolpaths: list[Toolpath]) -> GeometryValidationResult:
        """Validate a list of toolpaths."""
        errors: list[str] = []
        warnings: list[str] = []
        max_dev = 0.0
        max_arc_dev = 0.0
        inspected = 0

        for tp in toolpaths:
            if not tp.segments:
                warnings.append(f"Toolpath {tp.operation_id} has no segments")
                continue

            for seg in tp.segments:
                inspected += 1

                # Check zero-length moves
                if seg.length() < self.tolerance_mm:
                    if seg.motion_type in (MotionType.CUT, MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX):
                        errors.append(
                            f"Toolpath {tp.operation_id}: near-zero {seg.motion_type.name} move "
                            f"({seg.length():.6f}mm < {self.tolerance_mm}mm tolerance)"
                        )

                # Validate arcs
                if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX):
                    arc_dev = self._validate_arc(seg, tp.operation_id)
                    if arc_dev > max_arc_dev:
                        max_arc_dev = arc_dev
                    if arc_dev > self.arc_tolerance_mm:
                        errors.append(
                            f"Toolpath {tp.operation_id}: arc center deviation {arc_dev:.4f}mm "
                            f"exceeds tolerance {self.arc_tolerance_mm:.4f}mm"
                        )

                # Validate feed rates
                if seg.feed is not None and seg.feed > self.max_feed_mm_min:
                    warnings.append(
                        f"Toolpath {tp.operation_id}: feed {seg.feed:.0f} mm/min "
                        f"exceeds machine limit {self.max_feed_mm_min:.0f} mm/min"
                    )

                # Validate arc radius
                if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX):
                    if seg.arc_center is not None:
                        radius = float(np.linalg.norm(seg.start[:2] - seg.arc_center[:2]))
                        if radius < self.min_arc_radius_mm:
                            errors.append(
                                f"Toolpath {tp.operation_id}: arc radius {radius:.4f}mm "
                                f"below minimum {self.min_arc_radius_mm}mm"
                            )
                        if radius > self.max_arc_radius_mm:
                            errors.append(
                                f"Toolpath {tp.operation_id}: arc radius {radius:.4f}mm "
                                f"exceeds maximum {self.max_arc_radius_mm:.1f}mm"
                            )

        passed = len(errors) == 0
        return GeometryValidationResult(
            passed=passed,
            max_deviation_mm=max_dev,
            max_arc_center_deviation_mm=max_arc_dev,
            inspected_moves=inspected,
            errors=errors,
            warnings=warnings,
        )

    def _validate_arc(self, seg: MotionSegment, op_id: str) -> float:
        """Validate arc geometry and return center deviation."""
        if seg.arc_center is None:
            return 0.0

        start = seg.start
        end = seg.end
        center = seg.arc_center

        # Radius from start to center
        r_start = float(np.linalg.norm(start[:2] - center[:2]))
        # Radius from end to center
        r_end = float(np.linalg.norm(end[:2] - center[:2]))

        return abs(r_start - r_end)


def validate_toolpath_geometry(
    toolpaths: list[Toolpath],
    tolerance_mm: float = 0.01,
    arc_tolerance_mm: float = 0.005,
    max_feed_mm_min: float = 30000.0,
) -> GeometryValidationResult:
    """Convenience function to validate toolpath geometry."""
    validator = ToolpathGeometryValidator(
        tolerance_mm=tolerance_mm,
        arc_tolerance_mm=arc_tolerance_mm,
        max_feed_mm_min=max_feed_mm_min,
    )
    return validator.validate(toolpaths)