"""Voxel stock representation for material-removal simulation.

The stock is real setup-defined geometry voxelized at a resolution derived from
the stock size and a configurable resolution setting. Voxel state is updated by
actual toolpath motions (tool swept volumes), never by assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        """Voxelize arbitrary stock geometry into solid voxels."""
        stock = cls.from_bounds(bmin, bmax, resolution)
        if mesh is None or not len(getattr(mesh, "triangles", [])):
            stock.voxels[:] = False
            return stock

        dims = stock.voxels.shape
        voxels = np.zeros(dims, dtype=bool)
        
        x_coords = stock.origin[0] + np.arange(dims[0]) * stock.voxel_size
        y_coords = stock.origin[1] + np.arange(dims[1]) * stock.voxel_size
        z_coords = stock.origin[2] + np.arange(dims[2]) * stock.voxel_size

        for ix, x in enumerate(x_coords):
            for iy, y in enumerate(y_coords):
                hits = mesh.heights_above(x, y)
                if not hits:
                    continue
                hits.sort()
                # Deduplicate very close hits (e.g. hitting shared edge or vertex)
                unique_hits = []
                for h in hits:
                    if not unique_hits or abs(h - unique_hits[-1]) > 1e-4:
                        unique_hits.append(h)
                
                # Pairwise inside intervals
                for k in range(0, len(unique_hits) - 1, 2):
                    z_lo = unique_hits[k]
                    z_hi = unique_hits[k + 1]
                    mask = (z_coords >= z_lo) & (z_coords <= z_hi)
                    voxels[ix, iy, mask] = True

        stock.voxels = voxels
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
        lo_i, lo_j, lo_k = lo
        hi_i, hi_j, hi_k = hi
        # Coordinate grid for bounding box
        i_idx = np.arange(lo_i, hi_i)
        j_idx = np.arange(lo_j, hi_j)
        k_idx = np.arange(lo_k, hi_k)
        
        # Center in world coordinates
        ci = self.origin[0] + i_idx * self.voxel_size - c[0]
        cj = self.origin[1] + j_idx * self.voxel_size - c[1]
        ck = self.origin[2] + k_idx * self.voxel_size - c[2]
        
        # 3D broadcasted squared distance
        dist2_3d = ci[:, None, None] ** 2 + cj[None, :, None] ** 2 + ck[None, None, :] ** 2
        sphere_mask = dist2_3d <= (radius * radius)
        
        sub_grid = self.voxels[lo_i:hi_i, lo_j:hi_j, lo_k:hi_k]
        to_remove = sub_grid & sphere_mask
        removed = int(np.count_nonzero(to_remove))
        sub_grid[sphere_mask] = False
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

    def excess_material_volume(self, target_mesh) -> float:
        """Compare remaining stock against target model mesh.
        Returns volume of material that should have been removed but wasn't
        (excess stock above the finished model surface)."""
        if not hasattr(target_mesh, 'triangles') or not len(target_mesh.triangles):
            return 0.0
        # Get all occupied voxel centers as a batch
        occupied = np.argwhere(self.voxels)
        if len(occupied) == 0:
            return 0.0
        # Convert indices to world coordinates
        centers = self.origin + occupied * self.voxel_size
        excess_count = 0
        for c in centers:
            hs = target_mesh.heights_above(c[0], c[1])
            if hs:
                model_top = max(hs)
                if c[2] > model_top + 1e-6:
                    excess_count += 1
        return excess_count * self.voxel_size ** 3

    def transformed(self, matrix_4x4: np.ndarray, new_bmin: np.ndarray,
                    new_bmax: np.ndarray, resolution: float) -> "VoxelStock":
        """Transform this in-process voxel stock into a new setup's coordinate frame."""
        new_stock = VoxelStock.from_bounds(new_bmin, new_bmax, resolution)
        new_dims = new_stock.voxels.shape
        idx = np.indices(new_dims).reshape(3, -1).T
        centers_2 = new_stock.origin + idx * new_stock.voxel_size

        try:
            T_inv = np.linalg.inv(matrix_4x4)
        except np.linalg.LinAlgError:
            T_inv = np.eye(4)

        ones = np.ones((len(centers_2), 1))
        c2_hom = np.hstack([centers_2, ones])
        c1_hom = (T_inv @ c2_hom.T).T
        denom = c1_hom[:, 3:4]
        denom[np.abs(denom) < 1e-12] = 1.0
        c1 = c1_hom[:, :3] / denom

        idx_1 = np.floor((c1 - self.origin) / self.voxel_size + 0.5).astype(int)
        d = np.array(self.voxels.shape)
        valid = (idx_1[:, 0] >= 0) & (idx_1[:, 0] < d[0]) & \
                (idx_1[:, 1] >= 0) & (idx_1[:, 1] < d[1]) & \
                (idx_1[:, 2] >= 0) & (idx_1[:, 2] < d[2])

        new_voxels_flat = np.zeros(len(centers_2), dtype=bool)
        valid_idx = idx_1[valid]
        new_voxels_flat[valid] = self.voxels[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        new_stock.voxels = new_voxels_flat.reshape(new_dims)
        return new_stock

    def compute_removal_volume(self, protected_target: "VoxelStock") -> "VoxelStock":
        """Compute the active material removal volume: CurrentStock MINUS ProtectedTargetVolume.
        
        Toolpaths operate strictly within this volume to evacuate raw material without
        cutting air or violating protected islands, bosses, fixtures, or CAD walls.
        """
        if self.voxels.shape != protected_target.voxels.shape:
            raise ValueError("Voxel grid dimensions must match to compute removal volume")
        removal_mask = self.voxels & (~protected_target.voxels)
        return VoxelStock(
            voxels=removal_mask.copy(),
            voxel_size=self.voxel_size,
            origin=self.origin.copy(),
        )


    def evaluate_against_target(
        self,
        target_cad_voxels: "VoxelStock",
        tolerance_voxels: int = 1,
    ) -> dict[str, float | bool]:
        """Verify the simulated machined stock against the target CAD solid.
        
        - Overcut (Gouge): Material removed where CAD is solid (simulated is False, CAD is True).
        - Undercut (Excess): Material remaining outside allowable CAD tolerance (simulated is True, CAD is False).
        """
        if self.voxels.shape != target_cad_voxels.voxels.shape:
            raise ValueError("Voxel grid dimensions must match for target verification")

        
        # Gouge: CAD is solid (True) but simulated machined stock was removed (False)
        gouge_mask = target_cad_voxels.voxels & (~self.voxels)
        gouge_voxels = int(np.count_nonzero(gouge_mask))
        gouge_vol = float(gouge_voxels) * (self.voxel_size ** 3)
        
        # Excess: simulated stock is solid (True) but CAD is empty (False)
        excess_mask = self.voxels & (~target_cad_voxels.voxels)
        excess_voxels = int(np.count_nonzero(excess_mask))
        excess_vol = float(excess_voxels) * (self.voxel_size ** 3)
        
        return {
            "gouge_volume_mm3": round(gouge_vol, 6),
            "excess_volume_mm3": round(excess_vol, 6),
            "gouge_free": gouge_vol < 1e-9,
            "fully_cleared": excess_vol < 1e-9,
        }

    def is_empty(self) -> bool:
        """Return True if no solid voxels remain."""
        return bool(np.count_nonzero(self.voxels) == 0)

    def intersection(self, other: "VoxelStock") -> "VoxelStock":
        """Compute the boolean intersection of two voxel grids (RemainingStock ∩ TargetRemovalVolume)."""
        if self.voxels.shape != other.voxels.shape:
            raise ValueError("Voxel grid dimensions must match for intersection")
        return VoxelStock(
            voxels=(self.voxels & other.voxels).copy(),
            voxel_size=self.voxel_size,
            origin=self.origin.copy(),
        )

    def remaining_removal_volume(
        self,
        target_removal_volume: "VoxelStock",
        bounds_min: Optional[np.ndarray] = None,
        bounds_max: Optional[np.ndarray] = None,
    ) -> float:
        """Compute remaining material volume (mm³) to be removed: RemainingStock ∩ TargetRemovalVolume.
        
        Optionally restricted to a sub-region (e.g. a feature bounding box).
        """
        if self.voxels.shape != target_removal_volume.voxels.shape:
            raise ValueError("Voxel grid dimensions must match for remaining removal calculation")

        rem_mask = self.voxels & target_removal_volume.voxels
        if bounds_min is not None and bounds_max is not None:
            idx0 = self.world_to_index(bounds_min)
            idx1 = self.world_to_index(bounds_max)
            d = np.array(self.voxels.shape)
            i0, j0, k0 = np.clip(np.minimum(idx0, idx1), 0, d - 1)
            i1, j1, k1 = np.clip(np.maximum(idx0, idx1) + 1, 0, d)
            rem_mask = rem_mask[i0:i1, j0:j1, k0:k1]

        count = int(np.count_nonzero(rem_mask))
        return float(count) * (self.voxel_size ** 3)

    def has_remaining_removal(
        self,
        target_removal_volume: "VoxelStock",
        bounds_min: Optional[np.ndarray] = None,
        bounds_max: Optional[np.ndarray] = None,
        min_volume_mm3: float = 1e-4,
    ) -> bool:
        """Check if any material remains to be machined (RemainingStock ∩ TargetRemovalVolume > min_volume_mm3)."""
        return self.remaining_removal_volume(target_removal_volume, bounds_min, bounds_max) > min_volume_mm3


