"""Final validation gate comparing reconstructed G-code path with canonical toolpath."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from ..toolpaths.semantic import Toolpath, MotionSegment, MotionType
from ..post.gcode_parser import GCodeValidator, ParsedBlock
from ..post.controller_models import MachineProfile


class ValidationOutcome(Enum):
    """Final validation outcome."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class PathComparisonResult:
    """Result of comparing a planned path with reconstructed path."""

    matched: bool
    max_position_deviation_mm: float = 0.0
    max_arc_center_deviation_mm: float = 0.0
    max_feed_deviation_pct: float = 0.0
    max_spindle_deviation_rpm: float = 0.0
    total_planned_moves: int = 0
    total_gcode_moves: int = 0
    matched_moves: int = 0
    unmatched_planned_moves: int = 0
    unmatched_gcode_moves: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "max_position_deviation_mm": self.max_position_deviation_mm,
            "max_arc_center_deviation_mm": self.max_arc_center_deviation_mm,
            "max_feed_deviation_pct": self.max_feed_deviation_pct,
            "max_spindle_deviation_rpm": self.max_spindle_deviation_rpm,
            "total_planned_moves": self.total_planned_moves,
            "total_gcode_moves": self.total_gcode_moves,
            "matched_moves": self.matched_moves,
            "unmatched_planned_moves": self.unmatched_planned_moves,
            "unmatched_gcode_moves": self.unmatched_gcode_moves,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class FinalValidationResult:
    """Result of final validation gate."""

    outcome: ValidationOutcome
    passed: bool
    max_position_deviation_mm: float = 0.0
    max_arc_center_deviation_mm: float = 0.0
    max_feed_deviation_pct: float = 0.0
    max_spindle_deviation_rpm: float = 0.0
    total_planned_moves: int = 0
    total_gcode_moves: int = 0
    matched_moves: int = 0
    unmatched_planned_moves: int = 0
    unmatched_gcode_moves: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_operation: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome.value,
            "passed": self.passed,
            "max_position_deviation_mm": self.max_position_deviation_mm,
            "max_arc_center_deviation_mm": self.max_arc_center_deviation_mm,
            "max_feed_deviation_pct": self.max_feed_deviation_pct,
            "max_spindle_deviation_rpm": self.max_spindle_deviation_rpm,
            "total_planned_moves": self.total_planned_moves,
            "total_gcode_moves": self.total_gcode_moves,
            "matched_moves": self.matched_moves,
            "unmatched_planned_moves": self.unmatched_planned_moves,
            "unmatched_gcode_moves": self.unmatched_gcode_moves,
            "errors": self.errors,
            "warnings": self.warnings,
            "per_operation": self.per_operation,
        }


class FinalValidationGate:
    """Final validation gate comparing canonical toolpaths with G-code output.

    This is the critical gate that catches toolpath-vs-G-code synchronization issues
    by reconstructing the toolpath from G-code and comparing it with the canonical toolpath.
    """

    def __init__(
        self,
        position_tolerance_mm: float = 0.05,
        arc_center_tolerance_mm: float = 0.01,
        feed_tolerance_pct: float = 5.0,
        spindle_tolerance_rpm: float = 100.0,
        require_full_match: bool = True,
    ):
        self.position_tolerance_mm = position_tolerance_mm
        self.arc_center_tolerance_mm = arc_center_tolerance_mm
        self.feed_tolerance_pct = feed_tolerance_pct
        self.spindle_tolerance_rpm = spindle_tolerance_rpm
        self.require_full_match = require_full_match
        self.parser = GCodeValidator()

    def validate(
        self,
        canonical_toolpaths: list,
        gcode: str,
        machine_profile: Optional[object] = None,
    ) -> FinalValidationResult:
        """Run final validation comparing canonical toolpaths with G-code.

        Args:
            canonical_toolpaths: Original toolpaths from toolpath generation
            gcode: Generated G-code string from post-processor
            machine_profile: Machine profile for controller-specific parsing

        Returns:
            FinalValidationResult with detailed comparison metrics
        """
        # Parse G-code into blocks
        blocks = self.parser.parse(gcode)

        # Reconstruct toolpath from G-code
        reconstructed_segments = self._reconstruct_segments(blocks)

        # Compare each operation
        per_op_results = []
        all_errors: list[str] = []
        all_warnings: list[str] = []

        total_planned = 0
        total_gcode = len(reconstructed_segments)
        total_matched = 0
        total_unmatched_planned = 0
        total_unmatched_gcode = 0

        max_pos_dev = 0.0
        max_arc_dev = 0.0
        max_feed_dev = 0.0
        max_spindle_dev = 0.0

        for tp in canonical_toolpaths:
            op_result = self._compare_operation(tp, reconstructed_segments)
            per_op_results.append(op_result.to_dict())

            total_planned += op_result.total_planned_moves
            total_matched += op_result.matched_moves
            total_unmatched_planned += op_result.unmatched_planned_moves
            total_unmatched_gcode += op_result.unmatched_gcode_moves

            max_pos_dev = max(max_pos_dev, op_result.max_position_deviation_mm)
            max_arc_dev = max(max_arc_dev, op_result.max_arc_center_deviation_mm)
            max_feed_dev = max(max_feed_dev, op_result.max_feed_deviation_pct)
            max_spindle_dev = max(max_spindle_dev, op_result.max_spindle_deviation_rpm)

            all_errors.extend(op_result.errors)
            all_warnings.extend(op_result.warnings)

        # Check for unmatched G-code moves (sum of unmatched cutting moves across operations)
        total_unmatched_gcode = sum(op.get("unmatched_gcode_moves", 0) for op in per_op_results)

        if self.require_full_match and (total_unmatched_planned > 0 or total_unmatched_gcode > 0):
            all_errors.append(
                f"Move count mismatch: {total_unmatched_planned} planned unmatched, "
                f"{total_unmatched_gcode} G-code unmatched"
            )

        # Determine outcome
        has_errors = len(all_errors) > 0
        has_warnings = len(all_warnings) > 0

        if has_errors:
            outcome = ValidationOutcome.FAIL
        elif has_warnings:
            outcome = ValidationOutcome.WARNING
        else:
            outcome = ValidationOutcome.PASS

        return FinalValidationResult(
            outcome=outcome,
            passed=not has_errors,
            max_position_deviation_mm=max_pos_dev,
            max_arc_center_deviation_mm=max_arc_dev,
            max_feed_deviation_pct=max_feed_dev,
            max_spindle_deviation_rpm=max_spindle_dev,
            total_planned_moves=total_planned,
            total_gcode_moves=total_gcode,
            matched_moves=total_matched,
            unmatched_planned_moves=total_unmatched_planned,
            unmatched_gcode_moves=total_unmatched_gcode,
            errors=all_errors,
            warnings=all_warnings,
            per_operation=per_op_results,
        )

    def _reconstruct_segments(self, blocks: list[ParsedBlock]) -> list[dict]:
        """Reconstruct motion segments from parsed G-code blocks."""
        segments: list[dict] = []
        current_pos = np.array([0.0, 0.0, 0.0])
        current_feed = None
        current_spindle = None

        for block in blocks:
            if block.target_pos is not None:
                # Motion block
                arc_c = (current_pos + block.arc_center_ijk) if block.arc_center_ijk is not None else None
                seg = {
                    "start": current_pos.copy(),
                    "end": block.target_pos.copy(),
                    "command": block.command,
                    "feed": block.feed,
                    "spindle": block.spindle,
                    "d_register": block.d_register,
                    "arc_center": arc_c,
                    "line_number": block.line_number,
                    "raw_text": block.raw_text,
                }
                segments.append(seg)
                current_pos = block.target_pos.copy()

            if block.feed is not None:
                current_feed = block.feed
            if block.spindle is not None:
                current_spindle = block.spindle

        return segments

    def _compare_operation(
        self,
        canonical_tp,
        gcode_segments: list[dict],
    ) -> PathComparisonResult:
        """Compare a canonical operation with G-code segments."""
        # Filter G-code segments for this operation
        # In practice, would match by operation_id or sequence
        op_segments = gcode_segments  # Simplified

        # Extract planned segments from canonical toolpath
        planned_segments = []
        for seg in canonical_tp.segments:
            if seg.motion_type in (
                MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
                MotionType.ENTRY, MotionType.EXIT,
                MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX,
                MotionType.RAPID, MotionType.LINK, MotionType.RETRACT
            ):
                planned_segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "motion_type": seg.motion_type,
                    "feed": seg.feed,
                    "spindle": seg.spindle,
                    "arc_center": getattr(seg, "arc_center", None),
                    "arc_ccw": getattr(seg, "arc_ccw", False),
                })

        # Match planned to G-code segments
        matched = 0
        unmatched_planned = 0
        unmatched_gcode = len(gcode_segments)

        max_pos_dev = 0.0
        max_arc_dev = 0.0
        max_feed_dev = 0.0
        max_spindle_dev = 0.0
        errors: list[str] = []
        warnings: list[str] = []

        used_gcode = set()

        for i, planned in enumerate(planned_segments):
            best_match = None
            best_dist = float("inf")
            best_idx = -1

            for j, gcode in enumerate(gcode_segments):
                if j in used_gcode:
                    continue
                dist = self._segment_distance(planned, gcode)
                if dist < best_dist:
                    best_dist = dist
                    best_match = gcode
                    best_idx = j

            if best_match and best_dist <= self.position_tolerance_mm:
                matched += 1
                used_gcode.add(best_idx)

                # Compare details
                pos_dev = best_dist
                max_pos_dev = max(max_pos_dev, pos_dev)

                # Feed comparison
                if planned["feed"] is not None and best_match.get("feed") is not None:
                    p_feed = planned["feed"]
                    g_feed = best_match["feed"]
                    if p_feed > 0:
                        feed_dev_pct = abs(g_feed - p_feed) / p_feed * 100
                        max_feed_dev = max(max_feed_dev, feed_dev_pct)
                        if feed_dev_pct > self.feed_tolerance_pct:
                            warnings.append(
                                f"Move {i}: feed deviation {feed_dev_pct:.1f}% "
                                f"(planned {p_feed:.0f}, G-code {g_feed:.0f})"
                            )

                # Spindle comparison
                if planned["spindle"] is not None and best_match.get("spindle") is not None:
                    spindle_dev = abs(best_match["spindle"] - planned["spindle"])
                    max_spindle_dev = max(max_spindle_dev, spindle_dev)
                    if spindle_dev > 100:
                        warnings.append(
                            f"Move {i}: spindle deviation {spindle_dev:.0f} RPM"
                        )

                # Arc center comparison
                if planned.get("arc_center") is not None and best_match.get("arc_center") is not None:
                    arc_dev = float(np.linalg.norm(
                        planned["arc_center"][:2] - best_match["arc_center"][:2]
                    ))
                    max_arc_dev = max(max_arc_dev, arc_dev)
                    if arc_dev > self.arc_center_tolerance_mm:
                        errors.append(
                            f"Move {i}: arc center deviation {arc_dev:.4f}mm "
                            f"exceeds tolerance {self.arc_center_tolerance_mm:.4f}mm"
                        )

            else:
                unmatched_planned += 1
                if planned["motion_type"] in (MotionType.CUT, MotionType.ARC_CW, MotionType.ARC_CCW):
                    errors.append(
                        f"Planned {planned['motion_type'].name} move {i} not found in G-code"
                    )
                else:
                    warnings.append(
                        f"Planned {planned['motion_type'].name} move {i} not matched in G-code"
                    )

        # Count unmatched G-code (unmatched cutting moves vs non-cutting machine/rapid moves)
        unmatched_cutting_gcode = 0
        for idx in range(len(gcode_segments)):
            if idx not in used_gcode:
                seg = gcode_segments[idx]
                raw = (seg.get("raw_text") or "").upper()
                is_machine = any(k in raw for k in ("G53", "G43", "G44", "G49", "SUPA", "M91", "M92", "G28", "G30", "M30", "M2", "M6", "M3", "M5", "M8", "M9", "G80", "G40"))
                if seg.get("command") in ("G1", "G2", "G3") and not is_machine:
                    unmatched_cutting_gcode += 1
                    errors.append(
                        f"Unplanned G-code cutting move at line {seg.get('line_number', '?')}: {seg.get('raw_text', '')}"
                    )
                elif seg.get("command") in ("G0", "G1", "G2", "G3"):
                    warnings.append(
                        f"Unmatched G-code move at line {seg.get('line_number', '?')}: {seg.get('raw_text', '')}"
                    )
        unmatched_gcode = unmatched_cutting_gcode

        return PathComparisonResult(
            matched=unmatched_planned == 0 and unmatched_gcode == 0,
            max_position_deviation_mm=max_pos_dev,
            max_arc_center_deviation_mm=max_arc_dev,
            max_feed_deviation_pct=max_feed_dev,
            max_spindle_deviation_rpm=max_spindle_dev,
            unmatched_planned_moves=unmatched_planned,
            unmatched_gcode_moves=unmatched_gcode,
            errors=errors,
            warnings=warnings,
        )

    def _segment_distance(self, planned: dict, gcode: dict) -> float:
        """Compute distance between planned and G-code segment endpoints."""
        return float(np.linalg.norm(planned["end"] - gcode["end"]))


def run_final_validation(
    canonical_toolpaths: list,
    gcode: str,
    machine_profile: Optional[object] = None,
    position_tolerance_mm: float = 0.05,
    arc_center_tolerance_mm: float = 0.01,
    feed_tolerance_pct: float = 5.0,
    spindle_tolerance_rpm: float = 100.0,
) -> FinalValidationResult:
    """Convenience function to run final validation gate."""
    gate = FinalValidationGate(
        position_tolerance_mm=position_tolerance_mm,
        arc_center_tolerance_mm=arc_center_tolerance_mm,
        feed_tolerance_pct=feed_tolerance_pct,
        spindle_tolerance_rpm=spindle_tolerance_rpm,
    )
    return gate.validate(canonical_toolpaths, gcode, machine_profile)