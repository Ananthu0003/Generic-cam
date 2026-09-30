"""Tests for RemovalVolume = CurrentStock MINUS ProtectedTargetVolume and TargetCAD Verification.

Verifies:
1. Toolpath generation operates on RemovalVolume (CurrentStock MINUS ProtectedTargetVolume).
2. ProtectedTargetVolume protects CAD target solid, allowances, islands, bosses, and fixtures.
3. Final verification compares SimulatedMachinedStock versus TargetCAD (evaluating overcut/gouge vs undercut/excess).
"""

import numpy as np
import pytest

from cam_engine.geometry.voxel import VoxelStock


def test_removal_volume_subtraction():
    """Verify RemovalVolume isolates strictly material to be cut (excluding protected targets)."""
    # 10x10x10 grid with 1.0mm voxels
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([10.0, 10.0, 10.0])
    stock = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    assert stock.remaining_volume() == 1000.0

    # Protected target: A central boss/island (from [3,3,0] to [7,7,10]) = 4x4x10 = 160mm^3
    protected = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    # Initialize all False except protected zone
    protected.voxels[:] = False
    protected.voxels[3:7, 3:7, 0:10] = True
    assert protected.remaining_volume() == 160.0

    # Compute RemovalVolume = CurrentStock MINUS ProtectedTarget
    removal = stock.compute_removal_volume(protected)
    assert removal.remaining_volume() == pytest.approx(1000.0 - 160.0)

    # In the protected central boss zone, removal voxels MUST be False
    assert np.all(removal.voxels[3:7, 3:7, 0:10] == False)
    # Outside the protected zone, removal voxels MUST be True
    assert np.all(removal.voxels[0:3, 0:3, 0:10] == True)


def test_simulated_machined_stock_vs_target_cad_verification():
    """Verify evaluation of simulated machined stock vs target CAD (detecting gouges and excess)."""
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([10.0, 10.0, 10.0])

    # Target CAD solid: central cube [2:8, 2:8, 0:5] (6x6x5 = 180mm^3)
    target_cad = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    target_cad.voxels[:] = False
    target_cad.voxels[2:8, 2:8, 0:5] = True

    # Case 1: Perfect Machining (Simulated stock exactly matches Target CAD)
    sim_perfect = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    sim_perfect.voxels[:] = False
    sim_perfect.voxels[2:8, 2:8, 0:5] = True

    report_perfect = sim_perfect.evaluate_against_target(target_cad)
    assert report_perfect["gouge_free"] is True
    assert report_perfect["fully_cleared"] is True
    assert report_perfect["gouge_volume_mm3"] == 0.0
    assert report_perfect["excess_volume_mm3"] == 0.0

    # Case 2: Gouge (Cut too deep into CAD part)
    sim_gouge = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    sim_gouge.voxels[:] = False
    sim_gouge.voxels[2:8, 2:8, 0:5] = True
    # Accidentally cut a hole [4:6, 4:6, 2:5] inside the CAD body
    sim_gouge.voxels[4:6, 4:6, 2:5] = False

    report_gouge = sim_gouge.evaluate_against_target(target_cad)
    assert report_gouge["gouge_free"] is False
    assert report_gouge["gouge_volume_mm3"] > 0.0

    # Case 3: Excess (Left unmachined material outside CAD body)
    sim_excess = VoxelStock.from_bounds(bmin, bmax, resolution=1.0)
    sim_excess.voxels[:] = False
    sim_excess.voxels[2:8, 2:8, 0:5] = True
    # Left material at [0:2, 0:2, 0:5]
    sim_excess.voxels[0:2, 0:2, 0:5] = True

    report_excess = sim_excess.evaluate_against_target(target_cad)
    assert report_excess["fully_cleared"] is False
    assert report_excess["excess_volume_mm3"] > 0.0
