"""CAD deviation analysis for CAM validation."""

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


@dataclass
class CadDeviationResult:
    """Result of CAD deviation analysis."""

    passed: bool
    max_deviation_mm: float = 0.0
    mean_deviation_mm: float = 0.0
    rms_deviation_mm: float = 0.0
    inspected_points: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_deviation_mm": self.max_deviation_mm,
            "mean_deviation_mm": self.mean_deviation_mm,
            "rms_deviation_mm": self.rms_deviation_mm,
            "inspected_points": self.inspected_points,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class CadDeviationAnalyzer:
    """Analyzes deviation between machined surface and CAD model.

    Uses point-cloud sampling from CAD faces and compares with
    simulated tool contact points from toolpaths.
    """

    def __init__(
        self,
        tolerance_mm: float = 0.05,
        sample_density_mm2: float = 1.0,
    ):
        self.tolerance_mm = tolerance_mm
        self.sample_density_mm2 = sample_density_mm2

    def analyze(
        self,
        cad_faces: list[MeshData],
        toolpaths: list,  # Toolpath objects
        tool_diameters: dict[str, float],
        sample_points: Optional[np.ndarray] = None,
    ) -> CadDeviationResult:
        """Analyze CAD deviation from toolpaths."""
        errors: list[str] = []
        warnings: list[str] = []

        # Generate sample points on CAD faces if not provided
        if sample_points is None:
            sample_points = self._sample_cad_faces(cad_faces)
        else:
            sample_points = sample_points

        if len(sample_points) == 0:
            return CadDeviationResult(
                passed=True,
                errors=["No CAD faces provided for deviation analysis"],
            )

        # For each sample point, find nearest toolpath point
        deviations = []
        for pt in sample_points:
            min_dist = self._distance_to_toolpaths(pt, toolpaths, tool_diameters)
            deviations.append(min_dist)

        deviations = np.array(deviations)
        max_dev = float(np.max(deviations))
        mean_dev = float(np.mean(deviations))
        rms_dev = float(np.sqrt(np.mean(deviations ** 2)))

        if max_dev > self.tolerance_mm:
            errors.append(
                f"Max CAD deviation {max_dev:.3f}mm exceeds tolerance {self.tolerance_mm:.3f}mm"
            )
        elif mean_dev > self.tolerance_mm * 0.5:
            warnings.append(
                f"Mean CAD deviation {mean_dev:.3f}mm approaching tolerance {self.tolerance_mm:.3f}mm"
            )

        return CadDeviationResult(
            passed=len(errors) == 0,
            max_deviation_mm=max_dev,
            mean_deviation_mm=mean_dev,
            rms_deviation_mm=rms_dev,
            inspected_points=len(sample_points),
            errors=errors,
            warnings=warnings,
        )

    def _sample_cad_faces(self, faces: list[MeshData]) -> np.ndarray:
        """Generate sample points on CAD faces."""
        # Simplified - real implementation would use face area and sample density
        points = []
        for face in faces:
            if face.vertices is not None and len(face.vertices) >= 3:
                # Sample barycentric coordinates
                n_samples = max(1, int(face.area / 1.0)) if face.area else 1
                for _ in range(n_samples):
                    u = np.random.random()
                    v = np.random.random() * (1 - u)
                    w = 1 - u - v
                    pt = u * face.vertices[0] + v * face.vertices[1] + w * face.vertices[2]
                    points.append(pt)
        return np.array(points) if points else np.empty((0, 3))

    def _distance_to_toolpaths(
        self,
        point: np.ndarray,
        toolpaths: list,
        tool_diameters: dict[str, float],
    ) -> float:
        """Find minimum distance from point to any toolpath surface."""
        # Simplified - real implementation would use tool geometry
        # and compute exact distance to swept tool volume
        min_dist = float("inf")
        for tp in toolpaths:
            tool_dia = tool_diameters.get(tp.tool_id, 10.0)
            for seg in tp.segments:
                if seg.motion_type in (
                    # Only cutting motions
                    2, 3, 4, 5, 6, 7, 8, 9  # CUT, PLUNGE, RAMP, ENTRY, EXIT, ARC_CW, ARC_CCW, HELIX
                ):
                    # Distance from point to line segment
                    dist = self._point_to_segment_dist(point, seg.start, seg.end)
                    # Account for tool radius
                    effective_dist = max(0.0, dist - tool_dia / 2.0)
                    min_dist = min(min_dist, effective_dist)
        return min_dist

    def _point_to_segment_dist(self, pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        """Distance from point to line segment in 3D."""
        line_vec = end - start
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-12:
            return float(np.linalg.norm(pt - start))
        t = max(0.0, min(1.0, np.dot(pt - start, line_vec) / line_len_sq))
        closest = start + t * line_vec
        return float(np.linalg.norm(pt - closest))


def analyze_cad_deviation(
    cad_faces: list,
    toolpaths: list,
    tool_diameters: dict[str, float],
    tolerance_mm: float = 0.05,
) -> CadDeviationResult:
    """Convenience function for CAD deviation analysis."""
    analyzer = CadDeviationAnalyzer(tolerance_mm=tolerance_mm)
    return analyzer.analyze(cad_faces, toolpaths, tool_diameters)