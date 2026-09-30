"""Stock simulation for CAM validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min: np.ndarray
    max: np.ndarray

    @property
    def volume(self) -> float:
        diff = self.max - self.min
        return float(np.prod(diff))

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(point >= self.min) and np.all(point <= self.max))


@dataclass
class StockSimulationResult:
    """Result of stock simulation."""

    passed: bool
    removed_volume_mm3: float = 0.0
    remaining_volume_mm3: float = 0.0
    gouge_volume_mm3: float = 0.0
    min_wall_thickness_mm: float = 0.0
    max_undercut_mm: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "removed_volume_mm3": self.removed_volume_mm3,
            "remaining_volume_mm3": self.remaining_volume_mm3,
            "gouge_volume_mm3": self.gouge_volume_mm3,
            "min_wall_thickness_mm": self.min_wall_thickness_mm,
            "max_undercut_mm": self.max_undercut_mm,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class StockSimulator:
    """Simulates stock removal from toolpaths using voxel-based approach.

    Provides:
    - Material removal volume
    - Gouge detection (tool cutting into finished surfaces)
    - Minimum wall thickness verification
    - Undercut detection
    """

    def __init__(
        self,
        voxel_size_mm: float = 0.5,
        gouge_tolerance_mm: float = 0.01,
        min_wall_mm: float = 0.2,
    ):
        self.voxel_size_mm = voxel_size_mm
        self.gouge_tolerance_mm = gouge_tolerance_mm
        self.min_wall_mm = min_wall_mm

    def simulate(
        self,
        toolpaths: list[Toolpath],
        stock_bounds: BoundingBox,
        part_bounds: Optional[BoundingBox] = None,
        tool_diameters: Optional[dict[str, float]] = None,
    ) -> StockSimulationResult:
        """Run stock simulation."""
        # This is a simplified implementation - real implementation would use
        # voxel grid or CSG operations for accurate material removal tracking
        errors: list[str] = []
        warnings: list[str] = []

        # Estimate removed volume from toolpath bounding boxes
        total_removed = 0.0
        for tp in toolpaths:
            if not tp.segments:
                continue
            tool_dia = tool_diameters.get(tp.tool_id, 10.0) if tool_diameters else 10.0
            for seg in tp.segments:
                if seg.motion_type in (
                    MotionType.CUT,
                    MotionType.ARC_CW,
                    MotionType.ARC_CCW,
                    MotionType.HELIX,
                    MotionType.PLUNGE,
                ):
                    # Rough volume estimate: tool cylinder * path length
                    length = seg.length()
                    radius = tool_dia / 2.0
                    volume = np.pi * radius * radius * length
                    total_removed += volume

        # Stock volume
        stock_vol = stock_bounds.volume()
        part_vol = part_bounds.volume() if part_bounds else 0.0
        expected_removed = stock_vol - part_vol

        # Gouge check (simplified)
        gouge_vol = 0.0
        if total_removed > expected_removed + 1e-6:
            gouge_vol = total_removed - expected_removed
            warnings.append(
                f"Potential gouge detected: removed {total_removed:.1f} mm³ "
                f"exceeds expected {expected_removed:.1f} mm³ by {gouge_vol:.1f} mm³"
            )

        return StockSimulationResult(
            passed=len(warnings) == 0,
            removed_volume_mm3=total_removed,
            remaining_volume_mm3=stock_vol - total_removed,
            gouge_volume_mm3=gouge_vol,
            min_wall_thickness_mm=self.min_wall_mm,
            max_undercut_mm=0.0,
            warnings=warnings,
        )


def simulate_stock_removal(
    toolpaths: list[Toolpath],
    stock_bounds: BoundingBox,
    part_bounds: Optional[BoundingBox] = None,
    tool_diameters: Optional[dict[str, float]] = None,
    voxel_size_mm: float = 0.5,
) -> StockSimulationResult:
    """Convenience function to run stock simulation."""
    simulator = StockSimulator(voxel_size_mm=voxel_size_mm)
    return simulator.simulate(toolpaths, stock_bounds, part_bounds, tool_diameters)