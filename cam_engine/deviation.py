"""
CAD-vs-Machined Deviation Verification and Deviation Mapping Engine.

Compares simulated machined geometry (tool swept volume / simulated stock)
against the target CAD model (B-Rep / features / mesh / target voxels).

Computes:
1. Max positive deviation (excess remaining material / unmachined stock)
2. Max negative deviation (gouge / over-cut below CAD surface)
3. Feature diameter error (CAD nominal diameter vs achieved machined diameter)
4. Feature depth error (CAD nominal depth/floor vs achieved machined depth)
5. Surface deviation (point-by-point / face-by-face deviation metrics)
6. Fillet & corner radius deviation (nominal CAD radius vs cutter corner/profile)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Any
import numpy as np

from .context import Tool, ToolType
from .features import MachiningFeature, FeatureType
from .geometry import TriangleMesh, VoxelStock
from .toolpaths.semantic import Toolpath, MotionSegment, MotionType, CoordSpace


@dataclass
class FeatureDeviation:
    """Deviation metrics for a single CAD feature vs simulated machined geometry."""

    feature_id: str
    feature_type: str

    # Diameter deviation (cylindrical holes, bores, bosses, circular pockets)
    cad_diameter: Optional[float] = None
    machined_diameter: Optional[float] = None
    diameter_deviation: Optional[float] = None  # machined - CAD

    # Depth / Floor deviation
    cad_depth: Optional[float] = None
    machined_depth: Optional[float] = None
    depth_deviation: Optional[float] = None  # machined - CAD

    # Surface deviation
    surface_deviation_max: Optional[float] = None
    surface_deviation_min: Optional[float] = None
    surface_deviation_mean: Optional[float] = None

    # Fillet / Corner radius deviation
    cad_fillet_radius: Optional[float] = None
    machined_fillet_radius: Optional[float] = None
    fillet_deviation: Optional[float] = None

    # Status & human-readable formatting
    within_tolerance: bool = True
    formatted_summary: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class DeviationPoint:
    """Sample point on the CAD model with its local normal and signed deviation."""

    x: float
    y: float
    z: float
    deviation_mm: float  # Signed deviation: + = excess material, - = gouge / over-cut
    normal_x: float = 0.0
    normal_y: float = 0.0
    normal_z: float = 1.0


@dataclass
class DeviationMap:
    """Full CAD-vs-Machined Deviation Map and Analysis Report."""

    max_positive_deviation: float = 0.0   # Maximum excess unmachined material (+mm)
    max_negative_deviation: float = 0.0   # Maximum gouge / overcut (-mm, <= 0.0)
    mean_deviation: float = 0.0           # Mean signed deviation across sampled surface points
    rms_deviation: float = 0.0            # Root Mean Square deviation
    tolerance_mm: float = 0.02            # Permissible tolerance band (default ±0.02mm)
    within_tolerance: bool = True         # True if all deviations are within [-tolerance, +tolerance]

    feature_deviations: list[FeatureDeviation] = field(default_factory=list)
    deviation_points: list[DeviationPoint] = field(default_factory=list)
    summary_text: str = ""

    def get_feature_deviation(self, feature_id: str) -> Optional[FeatureDeviation]:
        for fd in self.feature_deviations:
            if fd.feature_id == feature_id:
                return fd
        return None

    def to_dict(self) -> dict:
        return {
            "max_positive_deviation": round(self.max_positive_deviation, 4),
            "max_negative_deviation": round(self.max_negative_deviation, 4),
            "mean_deviation": round(self.mean_deviation, 4),
            "rms_deviation": round(self.rms_deviation, 4),
            "tolerance_mm": round(self.tolerance_mm, 4),
            "within_tolerance": self.within_tolerance,
            "feature_deviations": [
                {
                    "feature_id": fd.feature_id,
                    "feature_type": fd.feature_type,
                    "cad_diameter": round(fd.cad_diameter, 4) if fd.cad_diameter is not None else None,
                    "machined_diameter": round(fd.machined_diameter, 4) if fd.machined_diameter is not None else None,
                    "diameter_deviation": round(fd.diameter_deviation, 4) if fd.diameter_deviation is not None else None,
                    "cad_depth": round(fd.cad_depth, 4) if fd.cad_depth is not None else None,
                    "machined_depth": round(fd.machined_depth, 4) if fd.machined_depth is not None else None,
                    "depth_deviation": round(fd.depth_deviation, 4) if fd.depth_deviation is not None else None,
                    "surface_deviation_max": round(fd.surface_deviation_max, 4) if fd.surface_deviation_max is not None else None,
                    "surface_deviation_min": round(fd.surface_deviation_min, 4) if fd.surface_deviation_min is not None else None,
                    "surface_deviation_mean": round(fd.surface_deviation_mean, 4) if fd.surface_deviation_mean is not None else None,
                    "cad_fillet_radius": round(fd.cad_fillet_radius, 4) if fd.cad_fillet_radius is not None else None,
                    "machined_fillet_radius": round(fd.machined_fillet_radius, 4) if fd.machined_fillet_radius is not None else None,
                    "fillet_deviation": round(fd.fillet_deviation, 4) if fd.fillet_deviation is not None else None,
                    "within_tolerance": fd.within_tolerance,
                    "formatted_summary": fd.formatted_summary,
                }
                for fd in self.feature_deviations
            ],
            "summary_text": self.summary_text,
        }


class CADMachinedDeviationAnalyzer:
    """Computes high-fidelity geometric deviations between CAD target and simulated machined model."""

    def __init__(
        self,
        tolerance_mm: float = 0.02,
        grid_resolution_mm: float = 1.0,
    ):
        self.tolerance_mm = tolerance_mm
        self.grid_resolution_mm = grid_resolution_mm

    def analyze(
        self,
        features: list[MachiningFeature],
        toolpaths: list[Toolpath],
        tools: list[Tool],
        mesh: Optional[TriangleMesh] = None,
        stock_voxels: Optional[VoxelStock] = None,
        target_cad_voxels: Optional[VoxelStock] = None,
    ) -> DeviationMap:
        """Run comprehensive CAD vs simulated machined deviation analysis."""
        tool_map = {t.id: t for t in tools}
        feat_tps: dict[str, list[Toolpath]] = {}
        for tp in toolpaths:
            if not tp.is_empty_or_skipped:
                feat_tps.setdefault(tp.feature_id, []).append(tp)

        feature_deviations: list[FeatureDeviation] = []
        all_point_deviations: list[float] = []
        deviation_points: list[DeviationPoint] = []

        # 1. Per-Feature Deviation Analysis
        for feat in features:
            f_id = feat.id
            f_type_str = feat.type.value if hasattr(feat.type, "value") else str(feat.type)
            f_tps = feat_tps.get(f_id, [])

            f_dev = self._analyze_feature(
                feat=feat,
                tps=f_tps,
                tool_map=tool_map,
                mesh=mesh,
                stock_voxels=stock_voxels,
            )
            feature_deviations.append(f_dev)

            # Collect surface deviations if available
            if f_dev.surface_deviation_mean is not None:
                all_point_deviations.append(f_dev.surface_deviation_mean)
            if f_dev.diameter_deviation is not None:
                all_point_deviations.append(f_dev.diameter_deviation)
            if f_dev.depth_deviation is not None:
                all_point_deviations.append(f_dev.depth_deviation)
            if f_dev.fillet_deviation is not None:
                all_point_deviations.append(f_dev.fillet_deviation)

        # 2. Voxel-based / Mesh-based Global Surface Deviation Sampling
        if mesh is not None and stock_voxels is not None:
            sampled_devs, sample_pts = self._sample_mesh_stock_deviations(mesh, stock_voxels)
            all_point_deviations.extend(sampled_devs)
            deviation_points.extend(sample_pts)

        # 3. Aggregate Global Statistics
        if all_point_deviations:
            arr = np.array(all_point_deviations, dtype=float)
            pos_mask = arr > 0
            neg_mask = arr < 0
            max_pos = float(arr[pos_mask].max()) if np.any(pos_mask) else 0.0
            max_neg = float(arr[neg_mask].min()) if np.any(neg_mask) else 0.0
            mean_dev = float(arr.mean())
            rms_dev = float(np.sqrt(np.mean(arr ** 2)))
        else:
            max_pos = 0.0
            max_neg = 0.0
            mean_dev = 0.0
            rms_dev = 0.0

        within_tol = (
            max_pos <= self.tolerance_mm + 1e-4 and
            max_neg >= -self.tolerance_mm - 1e-4 and
            all(fd.within_tolerance for fd in feature_deviations)
        )

        summary_lines = [
            "================ CAD-vs-Machined Deviation Report ================",
            f"Global Tolerance: ±{self.tolerance_mm:.3f} mm | Status: {'PASSED (Within Tolerance)' if within_tol else 'FAILED (Out of Tolerance)'}",
            f"Max Positive Deviation (Excess Material): {max_pos:+.4f} mm",
            f"Max Negative Deviation (Over-Cut/Gouge):  {max_neg:+.4f} mm",
            f"Mean Surface Deviation:                   {mean_dev:+.4f} mm",
            f"RMS Surface Deviation:                    {rms_dev:.4f} mm",
            "------------------------------------------------------------------",
            "Feature Deviations:",
        ]
        for fd in feature_deviations:
            status_tag = "[PASS]" if fd.within_tolerance else "[FAIL]"
            summary_lines.append(f"  • {fd.feature_id} ({fd.feature_type}): {fd.formatted_summary} {status_tag}")
        summary_lines.append("==================================================================")

        summary_text = "\n".join(summary_lines)

        return DeviationMap(
            max_positive_deviation=max_pos,
            max_negative_deviation=max_neg,
            mean_deviation=mean_dev,
            rms_deviation=rms_dev,
            tolerance_mm=self.tolerance_mm,
            within_tolerance=within_tol,
            feature_deviations=feature_deviations,
            deviation_points=deviation_points,
            summary_text=summary_text,
        )

    def _analyze_feature(
        self,
        feat: MachiningFeature,
        tps: list[Toolpath],
        tool_map: dict[str, Tool],
        mesh: Optional[TriangleMesh],
        stock_voxels: Optional[VoxelStock],
    ) -> FeatureDeviation:
        """Analyze diameter, depth, surface, and fillet deviations for a specific feature."""
        f_type_str = feat.type.value if hasattr(feat.type, "value") else str(feat.type)
        f_dev = FeatureDeviation(feature_id=feat.id, feature_type=f_type_str)
        summary_parts: list[str] = []

        if not tps:
            f_dev.within_tolerance = False
            f_dev.formatted_summary = "Unmachined (No cutting toolpath assigned)"
            f_dev.notes.append("No active toolpath machined this feature")
            return f_dev

        # Collect cutting segments across all toolpaths for this feature
        all_cuts: list[MotionSegment] = []
        for tp in tps:
            for s in tp.segments:
                if s.motion_type in (
                    MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
                    MotionType.HELIX, MotionType.ARC_CW, MotionType.ARC_CCW,
                    MotionType.ENTRY, MotionType.EXIT,
                ):
                    all_cuts.append(s)

        # Primary tool used
        primary_tool = tool_map.get(tps[0].tool_id)

        # --- A. Diameter Deviation Check ---
        cad_dia = getattr(feat, "diameter", None)
        if cad_dia is not None and cad_dia > 0:
            f_dev.cad_diameter = float(cad_dia)
            machined_dia = self._compute_machined_diameter(feat, tps, tool_map, all_cuts)
            if machined_dia is not None:
                f_dev.machined_diameter = float(machined_dia)
                f_dev.diameter_deviation = float(machined_dia - cad_dia)
                summary_parts.append(
                    f"Ø{cad_dia:.3f} CAD | machined = Ø{machined_dia:.3f} | deviation = {f_dev.diameter_deviation:+.3f} mm"
                )
                if abs(f_dev.diameter_deviation) > self.tolerance_mm + 1e-4:
                    f_dev.within_tolerance = False

        # --- B. Depth Deviation Check ---
        if f_type_str not in ("fillet", "chamfer"):
            cad_depth = getattr(feat, "depth", None)
            if (cad_depth is None or cad_depth <= 0) and feat.top_z is not None and feat.floor_z is not None:
                cad_depth = float(feat.top_z - feat.floor_z)

            if cad_depth is not None and cad_depth > 0:
                f_dev.cad_depth = float(cad_depth)
                machined_depth = self._compute_machined_depth(feat, tps, tool_map, all_cuts)
                if machined_depth is not None:
                    f_dev.machined_depth = float(machined_depth)
                    f_dev.depth_deviation = float(machined_depth - cad_depth)
                    if not summary_parts:
                        summary_parts.append(
                            f"Depth {cad_depth:.3f}mm CAD | machined = {machined_depth:.3f}mm | deviation = {f_dev.depth_deviation:+.3f} mm"
                        )
                    else:
                        summary_parts.append(f"Depth dev = {f_dev.depth_deviation:+.3f} mm")
                    if abs(f_dev.depth_deviation) > self.tolerance_mm + 1e-4:
                        f_dev.within_tolerance = False

        # --- C. Fillet & Corner Radius Deviation ---
        cad_fillet = getattr(feat, "corner_radius", None) or feat.notes.get("fillet_radius")
        if cad_fillet is not None and float(cad_fillet) > 0:
            cad_fillet_f = float(cad_fillet)
            f_dev.cad_fillet_radius = cad_fillet_f
            machined_fillet = self._compute_machined_fillet_radius(feat, tps, tool_map)
            if machined_fillet is not None:
                f_dev.machined_fillet_radius = float(machined_fillet)
                f_dev.fillet_deviation = float(machined_fillet - cad_fillet_f)
                if not summary_parts:
                    summary_parts.append(
                        f"R{cad_fillet_f:.3f} fillet CAD | machined = R{machined_fillet:.3f} | deviation = {f_dev.fillet_deviation:+.3f} mm"
                    )
                else:
                    summary_parts.append(f"Fillet R dev = {f_dev.fillet_deviation:+.3f} mm")
                if abs(f_dev.fillet_deviation) > self.tolerance_mm + 1e-4:
                    f_dev.within_tolerance = False

        # --- D. Surface Deviation ---
        if feat.floor_z is not None and all_cuts:
            lowest_cut_z = min(min(s.start[2], s.end[2]) for s in all_cuts)
            surf_dev = float(lowest_cut_z - feat.floor_z)
            f_dev.surface_deviation_min = surf_dev
            f_dev.surface_deviation_max = surf_dev
            f_dev.surface_deviation_mean = surf_dev
            if not summary_parts:
                summary_parts.append(f"Floor Z dev = {surf_dev:+.3f} mm")

        f_dev.formatted_summary = " | ".join(summary_parts) if summary_parts else "Machined nominal"
        return f_dev

    def _compute_machined_diameter(
        self,
        feat: MachiningFeature,
        tps: list[Toolpath],
        tool_map: dict[str, Tool],
        cuts: list[MotionSegment],
    ) -> Optional[float]:
        """Compute the resulting machined diameter from tool geometry and cutting passes."""
        # 1. Direct Drilling / Reaming / Spotting
        is_drilling = any(
            tp.purpose in ("drilling", "boring", "spot_drilling", "reaming", "tapping") or
            getattr(feat.type, "name", str(feat.type)) in ("HOLE", "BORE", "THREAD", "SPOT")
            for tp in tps
        )
        if is_drilling:
            # Diameter is determined by drilling tool diameter
            drill_tools = [tool_map[tp.tool_id] for tp in tps if tp.tool_id in tool_map and tool_map[tp.tool_id].type in (ToolType.DRILL, ToolType.REAMER, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL, ToolType.FLAT_ENDMILL)]
            if drill_tools:
                return float(drill_tools[-1].diameter)

        # 2. Circular / Helical Contour Milling
        center = np.array(feat.center[:2]) if feat.center is not None else np.zeros(2)
        r_max_cut = 0.0
        r_tool = (tool_map[tps[0].tool_id].diameter / 2.0) if (tps and tps[0].tool_id in tool_map) else 0.0

        for s in cuts:
            if s.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX) and s.arc_center is not None:
                c = s.arc_center[:2]
                r_orbit = float(np.linalg.norm(s.start[:2] - c))
                r_max_cut = max(r_max_cut, r_orbit + r_tool)
            else:
                d_start = float(np.linalg.norm(s.start[:2] - center))
                d_end = float(np.linalg.norm(s.end[:2] - center))
                r_max_cut = max(r_max_cut, max(d_start, d_end) + r_tool)

        if r_max_cut > 1e-4:
            return float(2.0 * r_max_cut)

        return None

    def _compute_machined_depth(
        self,
        feat: MachiningFeature,
        tps: list[Toolpath],
        tool_map: dict[str, Tool],
        cuts: list[MotionSegment],
    ) -> Optional[float]:
        """Compute the achieved depth from tool tip motion and tool geometry."""
        if not cuts:
            return None
        min_z = min(min(s.start[2], s.end[2]) for s in cuts)
        ref_top = feat.top_z if feat.top_z is not None else 0.0
        return float(ref_top - min_z)

    def _compute_machined_fillet_radius(
        self,
        feat: MachiningFeature,
        tps: list[Toolpath],
        tool_map: dict[str, Tool],
    ) -> Optional[float]:
        """Compute resulting fillet radius from cutter profile."""
        for tp in tps:
            tool = tool_map.get(tp.tool_id)
            if tool is not None:
                if tool.type == ToolType.BALL_ENDMILL:
                    return float(tool.diameter / 2.0)
                elif tool.type == ToolType.BULLNOSE_ENDMILL and tool.corner_radius > 0:
                    return float(tool.corner_radius)
        return None

    def _sample_mesh_stock_deviations(
        self, mesh: TriangleMesh, stock_voxels: VoxelStock
    ) -> tuple[list[float], list[DeviationPoint]]:
        """Sample surface heights across the CAD mesh and compare with remaining stock voxels."""
        deviations: list[float] = []
        points: list[DeviationPoint] = []

        bmin, bmax = mesh.bounds()
        if np.all(bmax <= bmin):
            return deviations, points

        xs = np.arange(bmin[0] + self.grid_resolution_mm / 2.0, bmax[0], self.grid_resolution_mm)
        ys = np.arange(bmin[1] + self.grid_resolution_mm / 2.0, bmax[1], self.grid_resolution_mm)

        hm = stock_voxels.heightmap() if hasattr(stock_voxels, "heightmap") else None
        if hm is None:
            return deviations, points

        for x in xs:
            for y in ys:
                cad_z = mesh.top_height(x, y)
                if cad_z is None:
                    continue

                idx = stock_voxels.world_to_index(np.array([x, y, cad_z]))
                ix, iy = int(idx[0]), int(idx[1])
                if 0 <= ix < hm.shape[0] and 0 <= iy < hm.shape[1]:
                    stock_z = hm[ix, iy]
                    if not np.isnan(stock_z):
                        dev = float(stock_z - cad_z)
                        deviations.append(dev)
                        points.append(DeviationPoint(x=float(x), y=float(y), z=float(cad_z), deviation_mm=dev))

        return deviations, points
