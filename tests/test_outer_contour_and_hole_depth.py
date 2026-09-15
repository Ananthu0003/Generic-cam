"""Tests for outer contour stock removal / perimeter finishing and blind hole depth classification."""

import numpy as np
from pathlib import Path
import pytest

from cam_engine.context import (
    PlanningContext, Setup, Stock, StockKind, Material, DEFAULT_MATERIALS,
    Tool, ToolType, Units, WorkOffset, MachineConfig, MachineAxis, MachineType
)
from cam_engine.demo_models import get_default_tools, get_default_machine, create_prismatic_bracket_mesh
from cam_engine.features import FeatureRecognizer, FeatureType, MachiningFeature, Accessibility
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.toolpaths.semantic import MotionType
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.geometry.topology import Face, SurfaceData, SurfaceKind, Shape
from cam_engine.geometry.step_import import import_step


def test_blind_vs_through_hole_classification():
    """Test that blind holes are classified as BLIND_HOLE/BLIND_BORE with correct depth,
    and through holes are classified as THROUGH_HOLE/THROUGH_BORE."""
    # Part 100x60x30 mm, bottom at Z=0, top at Z=30
    # Hole 1: Blind hole from Z=30 down to Z=15 (depth 15mm, floor_z=15)
    # Hole 2: Through hole from Z=30 down to Z=0 (depth 30mm, floor_z=0)
    
    def make_cyl_face(idx, center_xy, r, z_top, z_bottom, is_internal=True):
        sd = SurfaceData(
            kind=SurfaceKind.CYLINDER,
            point_on=np.array([center_xy[0], center_xy[1], 0.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            radius=r
        )
        n_pts = 16
        triangles = []
        for i in range(n_pts):
            a0 = 2 * np.pi * i / n_pts
            a1 = 2 * np.pi * (i + 1) / n_pts
            p0 = [center_xy[0] + r * np.cos(a0), center_xy[1] + r * np.sin(a0), z_bottom]
            p1 = [center_xy[0] + r * np.cos(a1), center_xy[1] + r * np.sin(a1), z_bottom]
            p2 = [center_xy[0] + r * np.cos(a1), center_xy[1] + r * np.sin(a1), z_top]
            p3 = [center_xy[0] + r * np.cos(a0), center_xy[1] + r * np.sin(a0), z_top]
            triangles.append([p0, p1, p2])
            triangles.append([p0, p2, p3])
        tris = np.array(triangles, dtype=float)
        bmin = np.array([center_xy[0] - r, center_xy[1] - r, z_bottom])
        bmax = np.array([center_xy[0] + r, center_xy[1] + r, z_top])
        norm = np.array([-np.cos(0.0), -np.sin(0.0), 0.0])
        return Face(
            index=idx,
            surface=sd,
            oriented_normal=norm,
            triangles=tris,
            area=float(2 * np.pi * r * (z_top - z_bottom)),
            bounds_min=bmin,
            bounds_max=bmax,
            is_internal=is_internal,
        )

    # Base bottom plane at Z=0
    bot_sd = SurfaceData(kind=SurfaceKind.PLANE, normal=np.array([0.0, 0.0, -1.0]))
    bot_tris = np.array([
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 60.0, 0.0]],
        [[0.0, 0.0, 0.0], [100.0, 60.0, 0.0], [0.0, 60.0, 0.0]]
    ], dtype=float)
    bot_face = Face(
        index=0, surface=bot_sd, oriented_normal=np.array([0.0, 0.0, -1.0]),
        triangles=bot_tris, area=6000.0,
        bounds_min=np.array([0.0, 0.0, 0.0]), bounds_max=np.array([100.0, 60.0, 0.0])
    )

    # Top plane at Z=30
    top_sd = SurfaceData(kind=SurfaceKind.PLANE, normal=np.array([0.0, 0.0, 1.0]))
    top_tris = np.array([
        [[0.0, 0.0, 30.0], [100.0, 0.0, 30.0], [100.0, 60.0, 30.0]],
        [[0.0, 0.0, 30.0], [100.0, 60.0, 30.0], [0.0, 60.0, 30.0]]
    ], dtype=float)
    top_face = Face(
        index=1, surface=top_sd, oriented_normal=np.array([0.0, 0.0, 1.0]),
        triangles=top_tris, area=6000.0,
        bounds_min=np.array([0.0, 0.0, 30.0]), bounds_max=np.array([100.0, 60.0, 30.0])
    )

    # Face 2: Blind hole (D=8mm, Z: 12..30, depth 18mm)
    blind_face = make_cyl_face(2, [30.0, 30.0], 4.0, 30.0, 12.0)
    # Face 3: Through hole (D=8mm, Z: 0..30, depth 30mm)
    through_face = make_cyl_face(3, [70.0, 30.0], 4.0, 30.0, 0.0)

    shape = Shape(occ_shape=None, _faces=[bot_face, top_face, blind_face, through_face], _edges=[], _adjacency={})
    rec = FeatureRecognizer(shape, stock_top_z=31.0, drill_diameters=[8.0])
    features = rec.recognize()

    blind_feats = [f for f in features if f.id == "hole_2"]
    through_feats = [f for f in features if f.id == "hole_3"]

    assert len(blind_feats) == 1, "Blind hole should be recognized"
    assert blind_feats[0].type == FeatureType.BLIND_HOLE
    assert abs(blind_feats[0].depth - 18.0) < 1e-4
    assert abs(blind_feats[0].floor_z - 12.0) < 1e-4

    assert len(through_feats) == 1, "Through hole should be recognized"
    assert through_feats[0].type == FeatureType.THROUGH_HOLE
    assert abs(through_feats[0].depth - 30.0) < 1e-4
    assert abs(through_feats[0].floor_z - 0.0) < 1e-4


def test_compound_counterbore_blind_child_classification():
    """Verify that a compound counterbore with a blind child hole is correctly recognized as BLIND_HOLE,
    not forced to THROUGH_HOLE."""
    def make_cyl_face(idx, center_xy, r, z_top, z_bottom):
        sd = SurfaceData(
            kind=SurfaceKind.CYLINDER,
            point_on=np.array([center_xy[0], center_xy[1], 0.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            radius=r
        )
        n_pts = 16
        triangles = []
        for i in range(n_pts):
            a0 = 2 * np.pi * i / n_pts
            a1 = 2 * np.pi * (i + 1) / n_pts
            p0 = [center_xy[0] + r * np.cos(a0), center_xy[1] + r * np.sin(a0), z_bottom]
            p1 = [center_xy[0] + r * np.cos(a1), center_xy[1] + r * np.sin(a1), z_bottom]
            p2 = [center_xy[0] + r * np.cos(a1), center_xy[1] + r * np.sin(a1), z_top]
            p3 = [center_xy[0] + r * np.cos(a0), center_xy[1] + r * np.sin(a0), z_top]
            triangles.append([p0, p1, p2])
            triangles.append([p0, p2, p3])
        tris = np.array(triangles, dtype=float)
        bmin = np.array([center_xy[0] - r, center_xy[1] - r, z_bottom])
        bmax = np.array([center_xy[0] + r, center_xy[1] + r, z_top])
        return Face(
            index=idx, surface=sd, oriented_normal=np.array([-1.0, 0.0, 0.0]),
            triangles=tris, area=float(2 * np.pi * r * (z_top - z_bottom)),
            bounds_min=bmin, bounds_max=bmax, is_internal=True,
        )

    bot_face = Face(
        index=0, surface=SurfaceData(kind=SurfaceKind.PLANE, normal=np.array([0.0, 0.0, -1.0])),
        oriented_normal=np.array([0.0, 0.0, -1.0]),
        triangles=np.array([[[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 50.0, 0.0]]]),
        area=1250.0, bounds_min=np.array([0.0, 0.0, 0.0]), bounds_max=np.array([50.0, 50.0, 0.0])
    )
    top_face = Face(
        index=1, surface=SurfaceData(kind=SurfaceKind.PLANE, normal=np.array([0.0, 0.0, 1.0])),
        oriented_normal=np.array([0.0, 0.0, 1.0]),
        triangles=np.array([[[0.0, 0.0, 30.0], [50.0, 0.0, 30.0], [50.0, 50.0, 30.0]]]),
        area=1250.0, bounds_min=np.array([0.0, 0.0, 30.0]), bounds_max=np.array([50.0, 50.0, 30.0])
    )

    cbore_top = make_cyl_face(2, [25.0, 25.0], 7.0, 30.0, 20.0)
    cbore_child = make_cyl_face(3, [25.0, 25.0], 3.0, 20.0, 8.0)

    shape = Shape(occ_shape=None, _faces=[bot_face, top_face, cbore_top, cbore_child], _edges=[], _adjacency={})
    rec = FeatureRecognizer(shape, stock_top_z=30.0, drill_diameters=[6.0])
    features = rec.recognize()

    cbore_feat = next(f for f in features if f.type == FeatureType.COUNTERBORE)
    child_feat = next(f for f in features if f.id == cbore_feat.child_feature_ids[0])

    assert child_feat.type == FeatureType.BLIND_HOLE, f"Expected BLIND_HOLE but got {child_feat.type}"
    assert abs(child_feat.floor_z - 8.0) < 1e-4
    assert abs(child_feat.depth - 12.0) < 1e-4


def test_outer_contour_feature_and_toolpath():
    """Verify that outer contour feature is recognized and toolpath finishes the perimeter."""
    step_file = Path("sample_bracket.step")
    if not step_file.exists():
        pytest.skip("sample_bracket.step not found")

    imported = import_step(step_file)
    rec = FeatureRecognizer(imported.shape, stock_top_z=26.0)
    features = rec.recognize()

    contour_feats = [f for f in features if f.type == FeatureType.CONTOUR]
    assert len(contour_feats) >= 1, "Outer contour feature should be recognized"
    contour = contour_feats[0]
    assert contour.is_concave is False
    assert contour.depth > 0.0

    bmin, bmax = imported.shape.bounding_box()
    stock = Stock(kind=StockKind.BOX, bounds_min=bmin - np.array([5.0, 5.0, 3.0]),
                  bounds_max=bmax + np.array([5.0, 5.0, 1.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    tools = get_default_tools()
    machine = get_default_machine()
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    ctx = PlanningContext(model=imported, setups=[setup], machine=machine, tools=tools, material=mat)
    planner = OperationPlanner(ctx, setup, features)
    ops = planner.plan()

    contour_ops = [o for o in ops if o.feature.type == FeatureType.CONTOUR]
    assert len(contour_ops) >= 1, "At least one operation should be planned for the outer contour"

    mesh = TriangleMesh.from_faces(imported.shape.faces())
    strat_ctx = StrategyContext(context=ctx, setup=setup, mesh=mesh, stock_voxels=None,
                                clearance_z=float(stock.bounds_max[2]) + 10.0,
                                finish_allowance_wall=0.0, finish_allowance_floor=0.0,
                                ramp_angle_deg=15.0)
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(contour_ops[0])

    assert len(tp.segments) > 0
    assert tp.cutting_length() > 0.0
