"""Tests for Finished Geometry Hierarchy: Nominal CAD vs Manufacturing Target vs Actual Machined Result.

Verifies:
1. Nominal CAD geometry is pure and unaltered by tolerances or cutting parameters.
2. ToleranceModel defines explicit upper/lower drawing limits and midpoints.
3. MachiningAllowance represents deliberate process offsets (e.g. wall/floor finishing stock, deflection bias).
4. FinalManufacturingTarget deterministically derives the CNC programmed target based on strategy.
5. SimulatedMachinedResult evaluates simulated/measured outcomes against both nominal and target limits.
"""

import pytest
from cam_engine.context import (
    NominalGeometry,
    ToleranceModel,
    MachiningAllowance,
    FinalManufacturingTarget,
    SimulatedMachinedResult,
    TargetStrategy,
)


def test_nominal_geometry_purity():
    """Verify nominal geometry stores unaltered CAD dimensions."""
    nom = NominalGeometry(
        dimension=6.000,
        dimension_type="hole_diameter",
        canonical_reference_id="face_42",
        description="CAD Nominal Ø6.000mm",
    )
    assert nom.dimension == 6.000
    assert nom.dimension_type == "hole_diameter"
    assert nom.canonical_reference_id == "face_42"


def test_tolerance_model_limits_and_midpoint():
    """Verify ToleranceModel calculations for bilateral and unilateral limits."""
    # Bilateral: 6.000 +/- 0.020
    tol = ToleranceModel(upper_tol_mm=0.020, lower_tol_mm=-0.020, tolerance_class="bilateral")
    assert tol.upper_limit(6.000) == 6.020
    assert tol.lower_limit(6.000) == 5.980
    assert tol.midpoint(6.000) == 6.000
    assert tol.tolerance_band_mm == pytest.approx(0.040)
    assert tol.is_in_tolerance(6.000, 6.003)
    assert tol.is_in_tolerance(6.000, 6.020)
    assert not tol.is_in_tolerance(6.000, 6.025)

    # Asymmetric: 6.000 +0.030 / -0.010 -> Midpoint is 6.010
    asym_tol = ToleranceModel(upper_tol_mm=0.030, lower_tol_mm=-0.010)
    assert asym_tol.midpoint(6.000) == pytest.approx(6.010)


def test_final_manufacturing_target_derivation_strategies():
    """Verify different manufacturing target strategies (nominal vs midpoint vs MMC vs allowance)."""
    nom = NominalGeometry(dimension=6.000, dimension_type="hole_diameter")
    tol = ToleranceModel(upper_tol_mm=0.030, lower_tol_mm=-0.010)

    # 1. Strategy: NOMINAL
    tgt_nom = FinalManufacturingTarget.derive(nom, tolerance=tol, strategy=TargetStrategy.NOMINAL)
    assert tgt_nom.target_dimension == 6.000

    # 2. Strategy: TOLERANCE_MIDPOINT (aiming for center of asymmetric tolerance band)
    tgt_mid = FinalManufacturingTarget.derive(nom, tolerance=tol, strategy=TargetStrategy.TOLERANCE_MIDPOINT)
    assert tgt_mid.target_dimension == 6.010

    # 3. Strategy: MMC for hole (smallest allowable hole = 5.990)
    tgt_mmc = FinalManufacturingTarget.derive(nom, tolerance=tol, strategy=TargetStrategy.MAXIMUM_MATERIAL_CONDITION)
    assert tgt_mmc.target_dimension == 5.990

    # 4. Strategy with Machining Allowance (e.g. leaving 0.200mm wall stock for reaming)
    allowance = MachiningAllowance(radial_stock_mm=0.200, purpose="roughing")
    tgt_allow = FinalManufacturingTarget.derive(nom, tolerance=tol, allowance=allowance, strategy=TargetStrategy.NOMINAL)
    assert tgt_allow.target_dimension == 6.200


def test_simulated_machined_result_evaluation():
    """Verify evaluation of actual/simulated machined result against target and tolerance."""
    nom = NominalGeometry(dimension=6.000, dimension_type="hole_diameter")
    tol = ToleranceModel(upper_tol_mm=0.020, lower_tol_mm=-0.020)
    target = FinalManufacturingTarget.derive(nom, tolerance=tol, strategy=TargetStrategy.NOMINAL)

    # Case 1: In tolerance (Ø6.003)
    res_pass = SimulatedMachinedResult.evaluate(target, measured_dimension=6.003)
    assert res_pass.in_tolerance is True
    assert res_pass.measured_dimension == 6.003
    assert res_pass.deviation_from_nominal_mm == pytest.approx(0.003)
    assert res_pass.deviation_from_target_mm == pytest.approx(0.003)
    assert 0.0 < res_pass.tolerance_consumed_pct < 50.0

    # Case 2: Out of tolerance (Ø6.025)
    res_fail = SimulatedMachinedResult.evaluate(target, measured_dimension=6.025)
    assert res_fail.in_tolerance is False
    assert res_fail.deviation_from_nominal_mm == pytest.approx(0.025)
    assert "OUT OF TOLERANCE" in res_fail.inspection_notes


def test_target_geometry_abstractions_surface_boundary_curve_depth():
    """Verify generalized final pass target geometry abstractions beyond simple floor_z:
    - TargetSurface (Plane, Cylinder, Cone, Torus, Blend/Fillet, Freeform)
    - TargetBoundary (Closed boundary loop)
    - TargetCurve (3D edge / blend guide)
    - TargetDepth (Axial termination)
    """
    import numpy as np
    from cam_engine.context import (
        TargetGeometryType,
        TargetSurface,
        TargetBoundary,
        TargetCurve,
        TargetDepth,
        DepthReference,
    )

    # 1. Planar target surface
    surf_plane = TargetSurface(
        surface_kind="plane",
        face_id=1,
        point_on=np.array([0.0, 0.0, 25.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        description="Top finish face",
    )
    assert surf_plane.distance_to_surface(np.array([10.0, 20.0, 25.1])) == pytest.approx(0.1)

    # 2. Cylindrical target surface (Ø8.00 mm step)
    surf_cyl = TargetSurface(
        surface_kind="cylinder",
        face_id=2,
        point_on=np.array([0.0, 0.0, 0.0]),
        axis=np.array([0.0, 0.0, 1.0]),
        radius=4.00,
        description="Punch Ø8.00mm cylindrical journal",
    )
    # Point at radius 4.05 -> distance 0.05mm
    assert surf_cyl.distance_to_surface(np.array([4.05, 0.0, 2.0])) == pytest.approx(0.05)

    # 3. Toroidal / Blend Fillet target surface (R = 1.00 mm)
    surf_fillet = TargetSurface(
        surface_kind="fillet",
        face_id=3,
        radius=1.00,
        description="Transition shoulder fillet R1.00mm",
    )
    assert surf_fillet.radius == 1.00
    assert surf_fillet.surface_kind == "fillet"

    # 4. TargetBoundary (Closed pocket loop)
    boundary = TargetBoundary(
        boundary_type="pocket_wall",
        points=[np.array([0, 0, 15]), np.array([40, 0, 15]), np.array([40, 30, 15]), np.array([0, 30, 15]), np.array([0, 0, 15])],
        is_closed=True,
        z_level=15.0,
    )
    assert boundary.is_closed is True
    assert len(boundary.points) == 5

    # 5. TargetCurve (Chamfer edge)
    curve = TargetCurve(
        curve_kind="circle",
        edge_id=10,
        radius=6.0,
        center=np.array([0.0, 0.0, 25.0]),
    )
    assert curve.radius == 6.0

    # 6. TargetDepth (Drill tip / cylindrical bottom depth)
    depth = TargetDepth(
        depth_reference=DepthReference.TOP_FACE,
        nominal_depth=12.50,
        reference_z=25.0,
        target_z=12.50,
        depth_type="cylindrical_depth",
    )
    assert depth.target_z == 12.50

    # 7. Link into FinalManufacturingTarget
    nom = NominalGeometry(dimension=8.000, dimension_type="cylinder_diameter")
    tgt = FinalManufacturingTarget(
        nominal=nom,
        target_surface=surf_cyl,
        target_dimension=8.000,
    )
    assert tgt.target_surface is not None
    assert tgt.target_surface.radius == 4.00
