"""Machining boundary extraction, polygon offset, and multi-island region logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, List

import numpy as np

from ...features import FeatureType, MachiningFeature


@dataclass
class MachiningRegion:
    outer: List[np.ndarray]
    holes: List[List[np.ndarray]] = field(default_factory=list)
    z_top: float = 0.0
    z_bottom: float = 0.0
    remaining_stock: Optional[Any] = None


def _has_self_intersection(polygon: list[np.ndarray]) -> bool:
    """Check if any non-adjacent edges of a 2D polygon intersect (ray-casting)."""
    n = len(polygon)
    if n < 4:
        return False
    for i in range(n):
        a0 = polygon[i]
        a1 = polygon[(i + 1) % n]
        for j in range(i + 2, n):
            if j == (i - 1) % n or (i == 0 and j == n - 1):
                continue
            b0 = polygon[j]
            b1 = polygon[(j + 1) % n]
            d1 = a1 - a0
            d2 = b1 - b0
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 1e-12:
                continue
            t = ((b0[0] - a0[0]) * d2[1] - (b0[1] - a0[1]) * d2[0]) / cross
            u = ((b0[0] - a0[0]) * d1[1] - (b0[1] - a0[1]) * d1[0]) / cross
            if 0.0 < t < 1.0 and 0.0 < u < 1.0:
                return True
    return False


class BoundariesMixin:
    """Polygon offsetting, feature boundary recognition, and 2D nesting routines."""

    def _offset_polygon(self, polygon: list[np.ndarray], offset: float) -> list[np.ndarray]:
        """Offset a 2D polygon inward (negative offset) or outward (positive).
        Uses shapely when available (production-quality, handles concave/complex polygons);
        falls back to edge-parallel intersection math for environments without shapely."""
        try:
            from shapely.geometry import Polygon as _ShapelyPoly
            if len(polygon) < 3:
                return polygon[:]
            pts = [(float(p[0]), float(p[1])) for p in polygon]
            poly = _ShapelyPoly(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)  # repair self-intersections
            result_geom = poly.buffer(offset, join_style=2, cap_style=3)
            if result_geom.is_empty:
                return []
            # MultiPolygon: take the largest piece
            if hasattr(result_geom, 'geoms'):
                result_geom = max(result_geom.geoms, key=lambda g: g.area)
            coords = list(result_geom.exterior.coords)[:-1]  # exclude closing duplicate
            result = [np.array([x, y], dtype=float) for x, y in coords]
        except ImportError:
            result = self._offset_polygon_fallback(polygon, offset)

        if self.s.debug_stage in ("geometry", "compensation", "boundaries"):
            self.s.diagnostics.setdefault("offset_polygons", []).append({
                "offset": offset,
                "original": [[float(p[0]), float(p[1])] for p in polygon],
                "compensated": [[float(p[0]), float(p[1])] for p in result]
            })

        return result

    @staticmethod
    def _offset_polygon_fallback(polygon: list[np.ndarray], offset: float) -> list[np.ndarray]:
        """Fallback edge-parallel polygon offset (used when shapely is unavailable).
        Handles convex and simple concave polygons via consecutive edge intersections."""
        n = len(polygon)
        if n < 3:
            return polygon[:]
        offset_edges = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            edge = p1 - p0
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            normal = np.array([-edge[1], edge[0]], dtype=float) / edge_len
            op0 = p0 + normal * offset
            op1 = p1 + normal * offset
            offset_edges.append((op0, op1))
        if len(offset_edges) < 3:
            return polygon[:]
        result = []
        m = len(offset_edges)
        for i in range(m):
            _, e0_end = offset_edges[i]
            e1_start, _ = offset_edges[(i + 1) % m]
            d0 = e0_end - offset_edges[i][0]
            d1 = offset_edges[(i + 1) % m][1] - e1_start
            cross = d0[0] * d1[1] - d0[1] * d1[0]
            if abs(cross) < 1e-12:
                result.append(e0_end.copy())
            else:
                w = offset_edges[i][0] - e1_start
                t = (d1[0] * w[1] - d1[1] * w[0]) / cross
                result.append(offset_edges[i][0] + t * d0)
        if len(result) < 3:
            return result
        cleaned = [result[0]]
        for pt in result[1:]:
            if np.linalg.norm(pt - cleaned[-1]) > 1e-9:
                cleaned.append(pt)
        if len(cleaned) > 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) < 1e-9:
            cleaned.pop()
        if len(cleaned) < 3:
            return result
        if _has_self_intersection(cleaned):
            return []
        return cleaned

    @staticmethod
    def _convex_hull_2d(points: list[np.ndarray]) -> list[np.ndarray]:
        """Compute 2D convex hull using Monotone Chain algorithm."""
        pts = sorted(set((float(round(p[0], 4)), float(round(p[1], 4))) for p in points))
        if len(pts) <= 2:
            return [np.array(p, dtype=float) for p in pts]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return [np.array(p, dtype=float) for p in hull]

    def _feature_boundary_polygon(self, f: MachiningFeature, tool_radius: float = 0.0) -> MachiningRegion:
        """Extract a 2D boundary polygon and islands from the feature.
        For open features (Step, Shoulder, Open Pocket/Slot), extends open boundaries into open air
        by (tool_radius + 1.0mm) to prevent standing corner slivers."""
        is_open = f.type in (FeatureType.STEP, FeatureType.OPEN_POCKET, FeatureType.OPEN_SLOT, FeatureType.SHOULDER) or f.notes.get("is_open", False)

        outer = None

        if is_open and tool_radius > 0:
            extension = tool_radius * 1.2 + 1.0
            bmin, bmax = f.bounds_min[:2].copy(), f.bounds_max[:2].copy()
            model_bmin = self.s.setup.stock.bounds_min[:2]
            model_bmax = self.s.setup.stock.bounds_max[:2]
            if hasattr(self.s, "mesh") and self.s.mesh is not None and hasattr(self.s.mesh, "bounds_min"):
                model_bmin = self.s.mesh.bounds_min[:2]
                model_bmax = self.s.mesh.bounds_max[:2]

            if abs(bmin[0] - model_bmin[0]) < 1.0: bmin[0] -= extension
            if abs(bmax[0] - model_bmax[0]) < 1.0: bmax[0] += extension
            if abs(bmin[1] - model_bmin[1]) < 1.0: bmin[1] -= extension
            if abs(bmax[1] - model_bmax[1]) < 1.0: bmax[1] += extension

            outer = [
                np.array([bmin[0], bmin[1]]),
                np.array([bmax[0], bmin[1]]),
                np.array([bmax[0], bmax[1]]),
                np.array([bmin[0], bmax[1]]),
            ]

        # Use actual boundary geometry extracted by the recognizer
        if outer is None and f.boundary_loop and len(f.boundary_loop) >= 3:
            outer = [np.array([p[0], p[1]], dtype=float) for p in f.boundary_loop]

        # Fallback: exact bounding-box rectangle (do NOT use convex hull)
        if outer is None:
            bmin, bmax = f.bounds_min[:2], f.bounds_max[:2]
            outer = [
                np.array([bmin[0], bmin[1]]),
                np.array([bmax[0], bmin[1]]),
                np.array([bmax[0], bmax[1]]),
                np.array([bmin[0], bmax[1]]),
            ]

        region = MachiningRegion(
            outer=outer,
            holes=[[np.array([p[0], p[1]], dtype=float) for p in loop] for loop in (f.island_loops or [])],
            z_top=f.top_z,
            z_bottom=f.floor_z
        )

        if self.s.debug_stage in ("geometry", "boundary", "features"):
            self.s.diagnostics.setdefault("feature_boundaries", {})[f.id] = {
                "outer": [[float(p[0]), float(p[1])] for p in region.outer],
                "holes": [[[float(p[0]), float(p[1])] for p in hole] for hole in region.holes]
            }

        return region

    def _offset_machining_region(self, region: MachiningRegion, offset: float) -> list[list[np.ndarray]]:
        try:
            from shapely.geometry import Polygon as _ShapelyPoly
            pts = [(float(p[0]), float(p[1])) for p in region.outer]
            holes = [[(float(p[0]), float(p[1])) for p in h] for h in region.holes]
            poly = _ShapelyPoly(pts, holes=holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            result_geom = poly.buffer(offset, join_style=2, cap_style=3)

            loops = []
            if result_geom.is_empty:
                return loops

            geoms = result_geom.geoms if hasattr(result_geom, 'geoms') else [result_geom]
            for g in geoms:
                if g.is_empty: continue
                coords = list(g.exterior.coords)[:-1]
                loops.append([np.array([x, y], dtype=float) for x, y in coords])
                for interior in g.interiors:
                    coords = list(interior.coords)[:-1]
                    loops.append([np.array([x, y], dtype=float) for x, y in coords])
            return loops
        except ImportError:
            # Fallback to single outer loop if shapely isn't installed
            return [self._offset_polygon_fallback(region.outer, offset)]

    def _contour_parallel_passes(self, f: MachiningFeature, tool_radius: float,
                                 finish_wall: float, stepover: float,
                                 z: float) -> list[list[np.ndarray]]:
        """Generate offset polygon contour passes at a given Z level using MachiningRegion."""
        region = self._feature_boundary_polygon(f, tool_radius)
        inset0 = tool_radius + finish_wall
        bmin = np.min(region.outer, axis=0)
        bmax = np.max(region.outer, axis=0)
        extents = bmax - bmin
        min_extent = float(np.min(extents))
        n_offsets = max(1, int(np.ceil((min_extent / 2.0 - inset0) / stepover)) + 1)
        passes: list[list[np.ndarray]] = []

        # For outer contours / convex bosses: offset outward from part boundary into stock margin
        if not f.is_concave or f.type in (FeatureType.CONTOUR, FeatureType.BOSS):
            stock_bmin = self.s.setup.stock.bounds_min[:2]
            stock_bmax = self.s.setup.stock.bounds_max[:2]
            margin = max(float(np.max(stock_bmax - bmax)), float(np.max(bmin - stock_bmin)), 0.0)
            n_offsets = max(1, int(np.ceil((margin + inset0) / stepover)) + 1)
            for oi in range(n_offsets):
                outset = inset0 + (n_offsets - 1 - oi) * stepover
                loops = self._offset_machining_region(region, outset)
                for loop in loops:
                    passes.append([self._v3(p[0], p[1], z) for p in loop])
            return passes

        # Generate offset contours (inward for pockets)
        for oi in range(n_offsets):
            inset = inset0 + oi * stepover
            loops = self._offset_machining_region(region, -inset)
            for loop in loops:
                area = self._polygon_area(loop)
                if area > 1e-6:
                    passes.append([self._v3(p[0], p[1], z) for p in loop])

        return passes

    @staticmethod
    def _polygon_area(poly: list[np.ndarray]) -> float:
        """Shoelace formula for signed area."""
        n = len(poly)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i][0] * poly[j][1]
            area -= poly[j][0] * poly[i][1]
        return abs(area) * 0.5

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: list[np.ndarray]) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        x, y = point[0], point[1]
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside
