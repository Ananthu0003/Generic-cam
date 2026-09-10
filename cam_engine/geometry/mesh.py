"""Triangle-mesh queries: ray casting, point classification, height fields.

Used by collision checks, drop-cutter finishing, and entry/exit validation.
Operates on real tessellated B-Rep geometry only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriangleMesh:
    """Flat triangle soup with an acceleration grid over triangle AABBs."""

    triangles: np.ndarray  # (N, 3, 3)

    @classmethod
    def from_faces(cls, faces) -> "TriangleMesh":
        tris = [f.triangles for f in faces if len(f.triangles)]
        if not tris:
            return cls(triangles=np.zeros((0, 3, 3)))
        return cls(triangles=np.concatenate(tris, axis=0))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.triangles):
            return np.zeros(3), np.zeros(3)
        flat = self.triangles.reshape(-1, 3)
        return flat.min(axis=0), flat.max(axis=0)

    def _grid(self, cell: float = 5.0):
        if not len(self.triangles):
            return None
        bmin, bmax = self.bounds()
        dims = np.maximum(np.ceil((bmax - bmin) / cell).astype(int) + 1, 1)
        grid: dict[tuple[int, int, int], list[int]] = {}
        centers = self.triangles.mean(axis=1)
        cells = np.clip(((centers - bmin) / cell).astype(int), 0, dims - 1)
        for i, c in enumerate(cells):
            grid.setdefault((int(c[0]), int(c[1]), int(c[2])), []).append(i)
        return bmin, bmax, cell, grid, dims

    def ray_first_hit(self, origin: np.ndarray, direction: np.ndarray,
                      max_dist: float = 1e6) -> float | None:
        """Distance to the first triangle hit along a ray (Moller-Trumbore)."""
        if not len(self.triangles):
            return None
        g = self._grid()
        if g is None:
            return None
        bmin, bmax, cell, grid, dims = g
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return None
        d = d / n

        t_hit = max_dist
        step = max(1e-9, cell)
        # march along the ray through the grid
        t = 0.0
        while t <= min(max_dist, t_hit):
            p = o + d * t
            c = np.clip(((p - bmin) / cell).astype(int), 0, dims - 1)
            key = (int(c[0]), int(c[1]), int(c[2]))
            tris = grid.get(key, [])
            for ti in tris:
                tri = self.triangles[ti]
                e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
                h = np.cross(d, e2)
                a = e1 @ h
                if abs(a) < 1e-12:
                    continue
                f = 1.0 / a
                s = o - tri[0]
                u = f * (s @ h)
                if u < -1e-9 or u > 1.0 + 1e-9:
                    continue
                q = np.cross(s, e1)
                v = f * (d @ q)
                if v < -1e-9 or u + v > 1.0 + 1e-9:
                    continue
                dist = f * (e2 @ q)
                if 1e-9 < dist < t_hit:
                    t_hit = dist
            # advance to next cell boundary along the ray
            t_next = max_dist
            for ax in range(3):
                if abs(d[ax]) < 1e-12:
                    continue
                if d[ax] > 0:
                    boundary = bmin[ax] + (c[ax] + 1) * cell
                    t_next = min(t_next, (boundary - p[ax]) / d[ax])
                else:
                    boundary = bmin[ax] + c[ax] * cell
                    t_next = min(t_next, (boundary - p[ax]) / d[ax])
            if t_next <= t + 1e-12:
                t += step
            else:
                t = t_next + 1e-9
        return t_hit if t_hit < max_dist else None

    def is_point_inside(self, point: np.ndarray) -> bool:
        """Parity test via +Z ray casting; only meaningful for closed meshes."""
        if not len(self.triangles):
            return False
        p = np.asarray(point, dtype=float)
        bmin, bmax = self.bounds()
        if np.any(p < bmin) or np.any(p > bmax):
            return False
        # count crossings directly (simple, correct enough for closed solids)
        tri = self.triangles
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        d = np.array([0.0, 0.0, 1.0])
        e1, e2 = v1 - v0, v2 - v0
        h = np.cross(d, e2)
        a = (e1 * h).sum(axis=1)
        ok = np.abs(a) > 1e-12
        if not ok.any():
            return False
        f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        s = p - v0
        u = f * (s * h).sum(axis=1)
        q = np.cross(s, e1)
        v = f * (d * q).sum(axis=1)
        t = f * (e2 * q).sum(axis=1)
        hit = ok & (u >= 0) & (u + v <= 1) & (t > 1e-9)
        return bool(np.count_nonzero(hit) % 2 == 1)

    def top_height(self, x: float, y: float) -> float | None:
        """Highest surface Z at (x,y) within a small radius; None if open sky."""
        hits = self.heights_above(x, y)
        return max(hits) if hits else None

    def heights_above(self, x: float, y: float) -> list[float]:
        """All triangle surface Z values at (x,y) (intersection of vertical line)."""
        out: list[float] = []
        if not len(self.triangles):
            return out
        tri = self.triangles
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        # 2D point-in-triangle in XY, then interpolate Z
        px, py = x, y
        d0 = (v1[:, 0] - v0[:, 0]) * (py - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (px - v0[:, 0])
        d1 = (v2[:, 0] - v1[:, 0]) * (py - v1[:, 1]) - (v2[:, 1] - v1[:, 1]) * (px - v1[:, 0])
        d2 = (v0[:, 0] - v2[:, 0]) * (py - v2[:, 1]) - (v0[:, 1] - v2[:, 1]) * (px - v2[:, 0])
        neg = (d0 < 0) | (d1 < 0) | (d2 < 0)
        pos = (d0 > 0) | (d1 > 0) | (d2 > 0)
        inside = ~(neg & pos)
        for i in np.nonzero(inside)[0]:
            a, b, c = v0[i], v1[i], v2[i]
            area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(area) < 1e-12:
                continue
            w1 = ((px - a[0]) * (c[1] - a[1]) - (py - a[1]) * (c[0] - a[0])) / area
            w2 = ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) / area
            w0 = 1.0 - w1 - w2
            if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                continue
            out.append(w0 * a[2] + w1 * b[2] + w2 * c[2])
        return out
