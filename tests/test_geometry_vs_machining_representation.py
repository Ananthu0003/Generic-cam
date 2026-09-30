"""Tests for Separation of Design Geometry and Machining Representation.

Validates:
1. Design Geometry (Nominal CAD surface, boundary, curves) is immutable and holds exact CAD parameters.
2. Manufacturing Geometry (Toolpath boundary, Tool-center path, Stock boundary, Cutting envelope)
   is explicitly derived via ManufacturingOffsetCalculator.
3. Invariant: CAD Ø6 surface -> Manufacturing offset -> Tool-center path, NEVER mixed or randomized.
4. Cutting envelope swept volume and gouge validation against nominal CAD surfaces.
5. End-to-end integration across MachiningFeature, PlannedOperation, and Toolpath generation.
"""

import numpy as np
import pytest

from cam_engine.coords import CoordSpace
from cam_engine.features import FeatureType, MachiningFeature
from cam_engine.geometry.representation import (
    CadSurfaceKind,
    NominalCadSurface,
    NominalCadBoundary,
    NominalCadCurve,
    DesignGeometry,
    ToolpathBoundary,
    ToolCenterPath,
    StockBoundary,
    CuttingEnvelope,
    ManufacturingGeometry,
    ManufacturingOffsetCalculator,
)
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.toolpaths.semantic import ToolpathGeometryMode
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext


def test_design_geometry_cad_surface_integrity():
    """Verify nominal CAD surface holds pure CAD dimensions without tool offsets."""
    cad_surf = NominalCadSurface(
        kind=CadSurfaceKind.CYLINDER,
        face_ids=[101, 102],
        diameter=6.00,
        radius=3.00,
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.array([10.0, 20.0, 0.0]),
        axial_start=0.0,
        axial_end=-15.0,
        depth=15.0,
        is_internal=True,
        source_reference="Face#101:CYLINDER:Ø6.000",
    )

    design = DesignGeometry(
        surfaces=[cad_surf],
        primary_surface=cad_surf,
        nominal_dimensions={"diameter": 6.00, "depth": 15.0},
        space=CoordSpace.MODEL,
    )

    assert design.primary_surface.diameter == 6.00
    assert design.primary_surface.radius == 3.00
    assert design.primary_surface.depth == 15.0
    assert design.primary_surface.is_internal is True
    assert "diameter=6.000" in design.summary()


def test_manufacturing_offset_pipeline_hole_cad_o6():
    """Verify CAD Ø6 hole -> manufacturing offset calculation -> tool-center path."""
    cad_surf = NominalCadSurface(
        kind=CadSurfaceKind.CYLINDER,
        face_ids=[42],
        diameter=6.00,
        radius=3.00,
        center=np.array([0.0, 0.0, 0.0]),
        axial_start=0.0,
        axial_end=-10.0,
        depth=10.0,
        is_internal=True,
        source_reference="CAD_Ø6_SURFACE",
    )

    tool_radius = 1.50      # Ø3.00 mm endmill
    radial_allowance = 0.20 # 0.20 mm finish stock left on walls

    cad_r, bound_r, center_r = ManufacturingOffsetCalculator.calculate_cylindrical_offset(
        cad_surface=cad_surf,
        tool_radius=tool_radius,
        radial_allowance=radial_allowance,
        is_internal=True,
    )

    # 1. Nominal CAD radius: exactly 3.00 mm
    assert np.isclose(cad_r, 3.00)
    # 2. Toolpath boundary radius (with allowance): 3.00 - 0.20 = 2.80 mm
    assert np.isclose(bound_r, 2.80)
    # 3. Tool-center path radius: 2.80 - 1.50 = 1.30 mm
    assert np.isclose(center_r, 1.30)

    # Build the full circular tool center path
    tcp = ManufacturingOffsetCalculator.build_circular_tool_center_path(
        center_xy=np.array([0.0, 0.0]),
        nominal_surface=cad_surf,
        tool_radius=tool_radius,
        radial_allowance=radial_allowance,
        z_start=0.0,
        z_end=-10.0,
        is_internal=True,
    )

    assert np.isclose(tcp.center_radius, 1.30)
    assert np.isclose(tcp.source_toolpath_boundary.boundary_radius, 2.80)
    
    # Distance of all generated tool-center points from origin should be 1.30 mm
    radii = np.linalg.norm(tcp.points[:, :2], axis=1)
    assert np.allclose(radii, 1.30)


def test_manufacturing_offset_pipeline_boss_cad_o8():
    """Verify CAD Ø8 external boss -> manufacturing offset calculation -> tool-center path."""
    cad_surf = NominalCadSurface(
        kind=CadSurfaceKind.CYLINDER,
        face_ids=[88],
        diameter=8.00,
        radius=4.00,
        center=np.array([25.0, 30.0, 0.0]),
        axial_start=0.0,
        axial_end=-12.0,
        depth=12.0,
        is_internal=False, # External boss
        source_reference="CAD_Ø8_BOSS_SURFACE",
    )

    tool_radius = 2.00      # Ø4.00 mm endmill
    radial_allowance = 0.50 # 0.50 mm roughing allowance

    cad_r, bound_r, center_r = ManufacturingOffsetCalculator.calculate_cylindrical_offset(
        cad_surface=cad_surf,
        tool_radius=tool_radius,
        radial_allowance=radial_allowance,
        is_internal=False,
    )

    # 1. Nominal CAD radius: exactly 4.00 mm
    assert np.isclose(cad_r, 4.00)
    # 2. Toolpath boundary radius (outside boss): 4.00 + 0.50 = 4.50 mm
    assert np.isclose(bound_r, 4.50)
    # 3. Tool-center path radius (cutter center): 4.50 + 2.00 = 6.50 mm
    assert np.isclose(center_r, 6.50)


def test_swept_cutting_envelope_and_gouge_validation():
    """Verify cutting envelope swept volume and gouge validation against nominal CAD surface."""
    cad_surf = NominalCadSurface(
        kind=CadSurfaceKind.CYLINDER,
        diameter=6.00,
        radius=3.00,
        center=np.array([0.0, 0.0, 0.0]),
        axial_start=0.0,
        axial_end=-10.0,
        depth=10.0,
        is_internal=True,
    )

    design = DesignGeometry(
        surfaces=[cad_surf],
        primary_surface=cad_surf,
        nominal_dimensions={"diameter": 6.00},
        space=CoordSpace.SETUP,
    )

    # Valid machining geometry (cutter inside hole, allowance=0.0, tool_r=1.5 -> center_r=1.5)
    valid_mfg = ManufacturingOffsetCalculator.build_manufacturing_geometry(
        design_geometry=design,
        tool_radius=1.50,
        radial_allowance=0.0,
        feature_id="feat_hole_6",
        tool_length=30.0,
    )

    assert valid_mfg.validate_against_design() is True

    # Cutting envelope checks
    env = valid_mfg.cutting_envelope
    assert env is not None
    assert env.min_z == -10.0
    # Point at center (0, 0, -5) is inside cut envelope
    assert env.contains_point_xy_z(0.0, 0.0, -5.0) is True
    # Point at radius 2.9 mm is inside cut envelope (envelope outer radius = center_r 1.5 + tool_r 1.5 = 3.0)
    assert env.contains_point_xy_z(2.9, 0.0, -5.0) is True
    # Point at radius 3.5 mm is outside cut envelope
    assert env.contains_point_xy_z(3.5, 0.0, -5.0) is False


def test_feature_and_operation_representation_integration():
    """Verify MachiningFeature -> PlannedOperation -> Toolpath retains both Design & Manufacturing representations."""
    feat = MachiningFeature(
        id="hole_01",
        type=FeatureType.BLIND_HOLE,
        face_indices=[10, 11],
        bounds_min=np.array([-3.0, -3.0, -10.0]),
        bounds_max=np.array([3.0, 3.0, 0.0]),
        depth=10.0,
        top_z=0.0,
        floor_z=-10.0,
        axis=np.array([0.0, 0.0, 1.0]),
        diameter=6.00,
        surface_type="cylinder",
        center=np.array([0.0, 0.0, -5.0]),
        axial_start=0.0,
        axial_end=-10.0,
        source_geometry_reference="CAD:B-Rep:Face#10:CYLINDER:Ø6.000",
    )

    # Check Design Geometry extraction on feature
    design = feat.design_geometry
    assert design.primary_surface.diameter == 6.00
    assert design.primary_surface.radius == 3.00
    assert design.primary_surface.source_reference == "CAD:B-Rep:Face#10:CYLINDER:Ø6.000"

    class DummyTool:
        id = "T01"
        tool_number = 1
        type = "endmill"
        diameter = 4.00 # tool radius = 2.0 mm
        flute_length = 25.0
        overall_length = 50.0
        corner_radius = 0.0
        flutes = 3
        material = "carbide"
        length_offset_h = 1
        diameter_offset_d = 1

    params = CuttingParameters(
        spindle_rpm=5000,
        feed_rate=500.0,
        plunge_feed=250.0,
        ramp_feed=350.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=2.0,
        stepover_mm=1.5,
        coolant=True,
        effective_diameter_mm=4.0,
        engagement_angle_deg=90.0,
        power_kw=1.2,
        torque_nm=2.5,
    )


    op = PlannedOperation(
        id="op_helical_hole",
        purpose=OpPurpose.ROUGHING,
        feature=feat,
        tool=DummyTool(),
        params=params,
        notes={
            "method": "helical_interpolation",
            "finish_allowance_side": 0.2,
            "finish_allowance_floor": 0.1,
        },
    )


    # Derive manufacturing geometry
    mfg = op.get_manufacturing_geometry()
    assert mfg is not None
    assert mfg.feature_id == "hole_01"
    assert mfg.tool_center_path is not None
    # Nominal CAD radius = 3.0, allowance = 0.2 -> boundary = 2.8 -> tool r=2.0 -> center r = 0.8
    assert np.isclose(mfg.tool_center_path.center_radius, 0.80)

    from cam_engine.context import Setup, Stock, StockKind, WorkOffset, DEFAULT_MATERIALS
    from cam_engine.geometry.mesh import TriangleMesh

    class _Ctx:
        pass
    p_ctx = _Ctx()
    p_ctx.material = DEFAULT_MATERIALS.get("aluminum_6061")
    p_ctx.tools = [DummyTool()]
    
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-20.0, -20.0, -15.0]), bounds_max=np.array([20.0, 20.0, 5.0]))
    setup = Setup(id="setup_1", name="Top", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    dummy_mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))

    strat_ctx = StrategyContext(
        context=p_ctx,
        setup=setup,
        mesh=dummy_mesh,
        stock_voxels=None,
        clearance_z=10.0,
        finish_allowance_wall=0.2,
        finish_allowance_floor=0.1,
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    assert tp.design_geometry is not None
    assert tp.design_geometry.primary_surface.diameter == 6.00
    assert tp.manufacturing_geometry is not None
    assert np.isclose(tp.manufacturing_geometry.tool_center_path.center_radius, 0.80)


