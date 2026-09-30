"""Tests for Feature-Boundary Depth Planning Layer.

Verifies:
1. CAD feature boundary -> Target Z/radial surface -> Roughing step-down planner
   -> Multiple intermediate passes -> Final pass strictly on target.
2. Given any arbitrary CAD target depth (e.g. 22.70mm) and stepdown (e.g. 5.08mm):
   The intermediate passes step down (D1, D2, D3, D4, ...) and the FINAL pass
   terminates EXACTLY on the CAD target geometry (target_depth = CAD_depth),
   never on an accumulated (previous_depth + DOC).
3. Both fixed stepdown and equalized stepdown modes land strictly on the exact
   CAD target depth.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext


def test_feature_boundary_depth_planner_exact_cad_termination():
    """Verify depth levels strictly terminate on CAD target depth, not previous_depth + DOC."""
    # Test case: Top Z = 22.70, Floor Z = 0.0, DOC = 5.08
    top_z = 22.70
    cad_floor_z = 0.00
    doc = 5.08

    engine = StrategyEngine(strat=None)  # _compute_depth_levels doesn't need context
    levels = engine._compute_depth_levels(
        top_z=top_z,
        target_z=cad_floor_z,
        max_stepdown=doc,
        remaining_allowance=0.0,
        mode="fixed",
    )

    # 22.70 / 5.08 = 4.468 -> 5 passes
    assert len(levels) == 5

    # Intermediate roughing passes
    assert levels[0] == pytest.approx(top_z - 1 * doc, abs=1e-5)  # 17.62
    assert levels[1] == pytest.approx(top_z - 2 * doc, abs=1e-5)  # 12.54
    assert levels[2] == pytest.approx(top_z - 3 * doc, abs=1e-5)  # 7.46
    assert levels[3] == pytest.approx(top_z - 4 * doc, abs=1e-5)  # 2.38

    # Final pass MUST be exactly CAD_depth (0.00), not previous_depth + DOC (which would be -2.70 or 20.32)
    assert levels[-1] == pytest.approx(cad_floor_z, abs=1e-6)
    assert levels[-1] == 0.00


def test_feature_boundary_depth_planner_arbitrary_fractional_target():
    """Verify arbitrary fractional CAD targets land on exact CAD floor."""
    top_z = 50.00
    cad_floor_z = 13.375  # arbitrary CAD boundary
    doc = 8.00

    engine = StrategyEngine(strat=None)
    levels = engine._compute_depth_levels(
        top_z=top_z,
        target_z=cad_floor_z,
        max_stepdown=doc,
        remaining_allowance=0.0,
        mode="fixed",
    )

    # Total depth = 36.625. 36.625 / 8 = 4.578 -> 5 passes
    assert len(levels) == 5

    # Check intermediate steps
    assert levels[0] == pytest.approx(50.0 - 8.0, abs=1e-5)   # 42.0
    assert levels[1] == pytest.approx(50.0 - 16.0, abs=1e-5)  # 34.0
    assert levels[2] == pytest.approx(50.0 - 24.0, abs=1e-5)  # 26.0
    assert levels[3] == pytest.approx(50.0 - 32.0, abs=1e-5)  # 18.0

    # Final pass is exact CAD target
    assert levels[-1] == pytest.approx(13.375, abs=1e-6)
    assert levels[-1] == 13.375


def test_feature_boundary_depth_planner_with_finish_allowance():
    """Verify roughing depth levels strictly land on CAD target + allowance."""
    top_z = 30.00
    cad_floor_z = 10.00
    allowance = 0.50
    doc = 6.00

    engine = StrategyEngine(strat=None)
    levels = engine._compute_depth_levels(
        top_z=top_z,
        target_z=cad_floor_z,
        max_stepdown=doc,
        remaining_allowance=allowance,
        mode="fixed",
    )

    # Target roughing floor = 10.50. Total depth = 19.50. 19.50 / 6 = 3.25 -> 4 passes
    assert len(levels) == 4
    assert levels[0] == pytest.approx(24.0, abs=1e-5)
    assert levels[1] == pytest.approx(18.0, abs=1e-5)
    assert levels[2] == pytest.approx(12.0, abs=1e-5)
    assert levels[-1] == pytest.approx(10.50, abs=1e-6)


def test_feature_boundary_depth_planner_equalized_mode():
    """Verify equalized mode distributes stepdown uniformly and lands exactly on CAD target."""
    top_z = 20.00
    cad_floor_z = 5.00
    doc = 6.00

    engine = StrategyEngine(strat=None)
    levels = engine._compute_depth_levels(
        top_z=top_z,
        target_z=cad_floor_z,
        max_stepdown=doc,
        remaining_allowance=0.0,
        mode="equalized",
    )

    # Total depth = 15.00. 15.00 / 6.00 = 2.5 -> 3 passes of 5.00mm each
    assert len(levels) == 3
    assert levels[0] == pytest.approx(15.00, abs=1e-5)
    assert levels[1] == pytest.approx(10.00, abs=1e-5)
    assert levels[-1] == pytest.approx(5.00, abs=1e-6)
    assert levels[-1] == 5.00


def test_feature_boundary_single_pass_when_depth_within_doc():
    """Verify when feature depth is within DOC, single pass lands on CAD target."""
    top_z = 10.00
    cad_floor_z = 7.50
    doc = 5.00

    engine = StrategyEngine(strat=None)
    levels = engine._compute_depth_levels(
        top_z=top_z,
        target_z=cad_floor_z,
        max_stepdown=doc,
        remaining_allowance=0.0,
    )

    assert len(levels) == 1
    assert levels[0] == pytest.approx(7.50, abs=1e-6)
    assert levels[0] == 7.50
