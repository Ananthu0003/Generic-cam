"""Voxel stock representation for material-removal simulation.

The stock is real setup-defined geometry voxelized at a resolution derived from
the stock size and a configurable resolution setting. Voxel state is updated by
actual toolpath motions (tool swept volumes), never by assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelStock:
    """Solid voxel grid: True = material present. Origin at grid corner (0,0,0)."""

    voxels: np.ndarray            # bool[dx, dy, dz]
    voxel_size: float             # mm per voxel edge
    origin: np.ndarray            # world (setup) coords of voxel [0,0,0] center

    # ---------- construction ----------
    @classmethod
    def from_bounds(cls, bmin: np.ndarray, bmax: np.ndarray,
                    resolution: float) -> "VoxelStock":
        """Voxelize an axis-aligned box stock (setup space, Z up)."""
        bmin = np.asarray(bmin, dtype=float)
        bmax = np.asarray(bmax, dtype=float)
        if np.any(bmax <= bmin):
            raise ValueError("degenerate stock bounds")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        dims = np.maximum(np.ceil((bmax - bmin) / resolution).astype(int), 1)
        # origin = center of voxel [0,0,0]
        origin = bmin + resolution / 2.0
        voxels = np.ones(tuple(dims), dtype=bool)
        return cls(voxels=voxels, voxel_size=float(resolution), origin=origin)

    @classmethod
    def from_mesh(cls, mesh, bmin: np.ndarray, bmax: np.ndarray,
                  resolution: float) -> "VoxelStock":
        """Voxelize arbitrary stock geometry by point sampling each voxel center."""
        stock = cls.from_bounds(bmin, bmax, resolution)
        dims = stock.voxels.shape
        idx = np.indices(dims).reshape(3, -1).T
        centers = stock.origin + idx * stock.voxel_size
        inside = mesh.is_point_inside_batch(centers) if hasattr(mesh, "is_point_inside_batch") \
            else np.array([mesh.is_point_inside(c) for c in centers], dtype=bool)
        stock.voxels = inside.reshape(dims)
        return stock

    # ---------- queries ----------
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        d = np.array(self.voxels.shape) * self.voxel_size
        return self.origin - self.voxel_size / 2.0, self.origin - self.voxel_size / 2.0 + d

    def world_to_index(self, p: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(p, dtype=float) - self.origin) / self.voxel_size + 0.5).astype(int)

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(idx, dtype=float) * self.voxel_size

    def is_solid_at(self, p: np.ndarray) -> bool:
        i = self.world_to_index(p)
        d = np.array(self.voxels.shape)
        if np.any(i < 0) or np.any(i >= d):
            return False
        return bool(self.voxels[i[0], i[1], i[2]])

    def remaining_volume(self) -> float:
        return float(self.voxels.sum()) * self.voxel_size ** 3

    def solid_fraction(self) -> float:
        return float(self.voxels.mean())

    # ---------- removal ----------
    def remove_sphere(self, center: np.ndarray, radius: float) -> int:
        """Remove material inside a sphere; returns number of voxels removed."""
        c = np.asarray(center, dtype=float)
        d = np.array(self.voxels.shape)
        r_idx = int(np.ceil(radius / self.voxel_size))
        lo = np.maximum(self.world_to_index(c) - r_idx, 0)
        hi = np.minimum(self.world_to_index(c) + r_idx + 1, d)
        if np.any(lo >= hi):
            return 0
        grids = np.indices(hi - lo).reshape(3, -1).T + lo
        centers = self.index_to_world(grids)
        dist2 = ((centers - c) ** 2).sum(axis=1)
        mask = dist2 <= radius * radius
        cells = grids[mask]
        removed = 0
        for cell in cells:
            i, j, k = int(cell[0]), int(cell[1]), int(cell[2])
            if self.voxels[i, j, k]:
                self.voxels[i, j, k] = False
                removed += 1
        return removed

    def remove_capsule(self, p0: np.ndarray, p1: np.ndarray, radius: float) -> int:
        """Remove the swept volume of a sphere moving from p0 to p1 (tool motion)."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            return self.remove_sphere(p0, radius)
        n_steps = max(1, int(np.ceil(length / (self.voxel_size * 0.5))))
        removed = 0
        for i in range(n_steps + 1):
            t = i / n_steps
            removed += self.remove_sphere(p0 + seg * t, radius)
        return removed

    def diff(self, other: "VoxelStock") -> np.ndarray:
        """Boolean difference mask (True where material was removed)."""
        if other.voxels.shape != self.voxels.shape:
            raise ValueError("shape mismatch in diff")
        return self.voxels & ~other.voxels

    def removed_volume_between(self, before: "VoxelStock") -> float:
        return float((before.voxels & ~self.voxels).sum()) * self.voxel_size ** 3

    def heightmap(self) -> np.ndarray:
        """Top-most solid voxel Z per (x,y) column; NaN where no material."""
        dx, dy, dz = self.voxels.shape
        out = np.full((dx, dy), np.nan)
        # iterate from top down; first solid voxel per column wins
        for k in range(dz - 1, -1, -1):
            layer = self.voxels[:, :, k]
            update = np.isnan(out) & layer
            out[update] = self.origin[2] + k * self.voxel_size
        return out
