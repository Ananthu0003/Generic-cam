"""Machine and controller validation for CAM operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..toolpaths.semantic import Toolpath, MotionSegment, MotionType
from ..post.controller_models import MachineProfile, get_machine_profile


@dataclass
class MachineValidationResult:
    """Result of machine/controller validation."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    spindle_violations: int = 0
    feed_violations: int = 0
    travel_violations: int = 0
    tool_change_issues: int = 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "spindle_violations": self.spindle_violations,
            "feed_violations": self.feed_violations,
            "travel_violations": self.travel_violations,
            "tool_change_issues": self.tool_change_issues,
        }


class MachineControllerValidator:
    """Validates toolpaths against machine/controller capabilities."""

    def __init__(
        self,
        machine_profile: Optional[str] = None,
        profile: Optional[object] = None,
    ):
        if profile is not None:
            self.machine_profile = profile
        elif machine_profile is not None:
            self.machine_profile = get_machine_profile(machine_profile)
        else:
            raise ValueError("Either machine_profile or profile must be provided")

    def validate(self, toolpaths: list) -> MachineValidationResult:
        """Validate toolpaths against machine profile."""
        errors: list[str] = []
        warnings: list[str] = []
        spindle_violations = 0
        feed_violations = 0
        travel_violations = 0
        tool_change_issues = 0

        profile = self.machine_profile
        controller = profile.controller

        for tp in toolpaths:
            if not tp.segments:
                continue

            # Check spindle speeds
            for seg in tp.segments:
                if seg.spindle is not None:
                    if seg.spindle > profile.max_spindle_rpm:
                        spindle_violations += 1
                        errors.append(
                            f"Operation {tp.operation_id}: spindle {seg.spindle:.0f} RPM "
                            f"exceeds machine max {profile.max_spindle_rpm:.0f} RPM"
                        )
                    elif seg.spindle < profile.machine_profile.spindle_feed_limits.min_spindle_rpm:
                        warnings.append(
                            f"Operation {tp.operation_id}: spindle {seg.spindle:.0f} RPM "
                            f"below machine min {profile.machine_profile.spindle_feed_limits.min_spindle_rpm:.0f} RPM"
                        )

                # Check feed rates
                if seg.feed is not None:
                    if seg.feed > profile.max_feed_mm_min:
                        feed_violations += 1
                        errors.append(
                            f"Operation {tp.operation_id}: feed {seg.feed:.0f} mm/min "
                            f"exceeds machine max {profile.max_feed_mm_min:.0f} mm/min"
                        )
                    elif seg.feed < 1.0:
                        warnings.append(
                            f"Operation {tp.operation_id}: feed {seg.feed:.2f} mm/min "
                            f"very low, may cause dwell marks"
                        )

                # Check rapid moves
                if seg.motion_type in (0, 1):  # RAPID, LINK
                    # Simplified - would check against machine travel limits
                    pass

            # Check tool changes
            if tp.tool_id:
                # Verify tool exists in magazine (simplified)
                pass

        # Check total travel within machine envelope
        all_points = np.vstack([seg.start for tp in toolpaths for seg in tp.segments])
        if len(all_points) > 0:
            bbox_min = all_points.min(axis=0)
            bbox_max = all_points.max(axis=0)
            travel_x = bbox_max[0] - bbox_min[0]
            travel_y = bbox_max[1] - bbox_min[1]
            travel_z = bbox_max[2] - bbox_min[2]

            # These would come from machine profile in real implementation
            if travel_x > 2000 or travel_y > 1000 or travel_z > 800:
                travel_violations += 1
                errors.append(
                    f"Part envelope ({travel_x:.0f}x{travel_y:.0f}x{travel_z:.0f}mm) "
                    f"may exceed typical machine travel"
                )

        passed = len(errors) == 0
        return MachineValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            spindle_violations=spindle_violations,
            feed_violations=feed_violations,
            travel_violations=travel_violations,
            tool_change_issues=tool_change_issues,
        )


def validate_for_machine(
    toolpaths: list,
    machine_profile: str,
) -> MachineValidationResult:
    """Convenience function to validate toolpaths for a machine."""
    validator = MachineControllerValidator(machine_profile=machine_profile)
    return validator.validate(toolpaths)