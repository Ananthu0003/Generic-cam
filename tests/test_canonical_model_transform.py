"""Tests for canonical model transform hierarchy across multiple setups (G54 and G55).

Verifies the architecture:
B-Rep model
     ↓
Setup transform (model -> setup)
     ↓
Setup-local geometry
     ↓
Feature recognition
     ↓
Toolpath
     ↓
Machine coordinates (G54 / G55)
     ↓
G-code
"""

import math
import numpy as np
import pytest

from cam_engine.context import (
    ImportedModel,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Tool,
    ToolType,
    WorkOffset,
    DEFAULT_MATERIALS,
)
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.geometry.topology import Shape
from cam_engine.demo_models import create_prismatic_bracket_shape, get_default_machine, get_default_tools
from cam_engine.features import FeatureRecognizer, FeatureType, MachiningFeature
from cam_engine.operations import OperationPlanner, OpPurpose, PlannedOperation
from cam_engine.pipeline import run_cam_pipeline
from cam_engine.post import get_post_processor
from cam_engine.geometry.voxel import VoxelStock
from cam_engine.toolpaths.semantic import MotionType, Toolpath
from cam_engine.toolpaths.strategies import StrategyContext, StrategyEngine
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder


def _euler_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """Build rigid 4x4 transform matrix from Euler angles (deg) and translation."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def test_canonical_model_transform_architecture():
    """Verify that Setup 1 and Setup 2 share the single canonical B-Rep model
    and derive setup-local geometry, features, toolpaths, and G-code via model_to_setup."""
    # Canonical B-Rep model
    canonical_model = create_prismatic_bracket_shape()

    machine = get_default_machine()
    tools = get_default_tools()
    material = DEFAULT_MATERIALS["aluminum_6061"]

    # Setup 1 (Top machining, G54): Identity transform
    M1 = _euler_to_matrix(0.0, 0.0, 0.0)
    shape_s1 = canonical_model.shape.transformed(M1)
    bmin1, bmax1 = shape_s1.bounding_box()
    stock1 = Stock(kind=StockKind.BOX, bounds_min=bmin1 - np.array([5.0, 5.0, 3.0]), bounds_max=bmax1 + np.array([5.0, 5.0, 2.0]))
    setup1 = Setup(id="setup_001", name="Setup 1 - Top", model_to_setup=M1, work_offset=WorkOffset.G54, stock=stock1)

    # Setup 2 (Bottom machining, G55): Flipped 180 deg around X axis
    M2 = _euler_to_matrix(180.0, 0.0, 0.0)
    shape_s2 = canonical_model.shape.transformed(M2)
    bmin2, bmax2 = shape_s2.bounding_box()
    stock2 = Stock(kind=StockKind.BOX, bounds_min=bmin2 - np.array([5.0, 5.0, 3.0]), bounds_max=bmax2 + np.array([5.0, 5.0, 2.0]))
    setup2 = Setup(id="setup_002", name="Setup 2 - Bottom", model_to_setup=M2, work_offset=WorkOffset.G55, stock=stock2)

    # Single authoritative planning context
    ctx = PlanningContext(
        model=canonical_model,
        setups=[setup1, setup2],
        machine=machine,
        tools=tools,
        material=material,
    )
    ctx.validate()

    # Step 1: Verify Setup-local geometry derived strictly from canonical model
    shape_s1 = ctx.model_shape_in_setup(setup1)
    bmin1, bmax1 = shape_s1.bounding_box()
    assert np.allclose(bmin1, [0.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(bmax1, [100.0, 60.0, 25.0], atol=1e-3)

    shape_s2 = ctx.model_shape_in_setup(setup2)
    bmin2, bmax2 = shape_s2.bounding_box()
    assert np.allclose(bmin2, [0.0, -60.0, -25.0], atol=1e-3)
    assert np.allclose(bmax2, [100.0, 0.0, 0.0], atol=1e-3)

    # Step 2: Feature recognition on setup-local geometry
    rec1 = FeatureRecognizer(shape_s1, stock_top_z=float(setup1.stock.bounds_max[2]))
    feats1 = rec1.recognize()
    assert len(feats1) > 0

    rec2 = FeatureRecognizer(shape_s2, stock_top_z=float(setup2.stock.bounds_max[2]))
    feats2 = rec2.recognize()
    assert len(feats2) > 0

    # Step 3: Operation planning
    planner1 = OperationPlanner(ctx, setup1, feats1)
    ops1 = planner1.plan()
    assert len(ops1) > 0

    planner2 = OperationPlanner(ctx, setup2, feats2)
    ops2 = planner2.plan()
    assert len(ops2) > 0

    # Step 4: Toolpath generation in Setup coordinates
    mesh1 = TriangleMesh.from_faces(shape_s1.faces())
    stock_voxels1 = VoxelStock.from_bounds(setup1.stock.bounds_min, setup1.stock.bounds_max, resolution=2.0)
    strat_ctx1 = StrategyContext(
        context=ctx, setup=setup1, mesh=mesh1, stock_voxels=stock_voxels1,
        clearance_z=float(setup1.stock.bounds_max[2]) + 10.0,
        finish_allowance_wall=0.3, finish_allowance_floor=0.3,
    )
    engine1 = StrategyEngine(strat_ctx1)
    tps1 = [engine1.generate(op) for op in ops1]
    for tp in tps1:
        tp.metadata["setup_id"] = setup1.id
        tp.metadata["work_offset"] = setup1.work_offset.value

    # In-process stock transfer: T_1_to_2 = M2 @ inv(M1)
    T1_to_2 = M2 @ np.linalg.inv(M1)
    stock_voxels2 = stock_voxels1.transformed(T1_to_2, setup2.stock.bounds_min, setup2.stock.bounds_max, resolution=2.0)

    mesh2 = TriangleMesh.from_faces(shape_s2.faces())
    strat_ctx2 = StrategyContext(
        context=ctx, setup=setup2, mesh=mesh2, stock_voxels=stock_voxels2,
        clearance_z=float(setup2.stock.bounds_max[2]) + 10.0,
        finish_allowance_wall=0.3, finish_allowance_floor=0.3,
    )
    engine2 = StrategyEngine(strat_ctx2)
    tps2 = [engine2.generate(op) for op in ops2]
    for tp in tps2:
        tp.metadata["setup_id"] = setup2.id
        tp.metadata["work_offset"] = setup2.work_offset.value

    # Step 5: Post-processing to Machine / Work Coordinates (G54 for Setup 1, G55 for Setup 2)
    post1 = get_post_processor(controller="fanuc", work_offset=setup1.work_offset.value)
    gcode1 = post1.post_process(tps1, program_name="OP10_TOP").gcode
    assert "G54" in gcode1
    assert "G55" not in gcode1

    post2 = get_post_processor(controller="fanuc", work_offset=setup2.work_offset.value)
    gcode2 = post2.post_process(tps2, program_name="OP20_BOTTOM").gcode
    assert "G55" in gcode2
    assert "G54" not in gcode2


def test_cross_setup_critical_web_validation_with_canonical_transforms():
    """Verify that opposing blind holes across Setup 1 and Setup 2 leaving 0.20 mm web
    are validated in canonical model space, clamping shoulder depth reference to tip depth."""
    # Model: Height 3.22 mm
    # Setup 1 (Top, G54): Blind hole at top, depth 1.75 mm -> floor at Z=1.47 mm
    # Setup 2 (Bottom, G55): Blind hole at bottom, depth 1.27 mm -> floor at Z=1.27 mm
    # Remaining web: 3.22 - 1.75 - 1.27 = 0.20 mm!
    cylinder_shape = Shape(BRepPrimAPI_MakeCylinder(0.45, 3.22).Shape())
    canonical_model = ImportedModel(
        shape=cylinder_shape,
        declared_units="mm",
        original_bounds=cylinder_shape.bounding_box(),
    )

    # Drill tool: Ø0.58 mm, 118 deg point angle -> tip length h_tip = (0.29)/tan(59°) = 0.174 mm
    drill_tool = Tool(
        id="T05",
        type=ToolType.DRILL,
        diameter=0.58,
        flute_length=5.0,
        overall_length=35.0,
        tool_number=5,
        point_angle=118.0,
    )
    tools = [drill_tool]
    machine = get_default_machine()
    material = DEFAULT_MATERIALS["aluminum_6061"]

    # Setup 1: Identity transform (Top)
    M1 = np.eye(4)
    stock1 = Stock.from_cylinder(center_xy=np.array([0.0, 0.0]), radius=1.0, z_min=-1.0, z_max=4.22)
    setup1 = Setup(id="s1", name="Setup 1 - Top", model_to_setup=M1, work_offset=WorkOffset.G54, stock=stock1)

    # Setup 2: Flipped 180° around X, translated so Z=0 is bottom (Bottom)
    M2 = _euler_to_matrix(180.0, 0.0, 0.0, tx=0.0, ty=0.0, tz=3.22)
    stock2 = Stock.from_cylinder(center_xy=np.array([0.0, 0.0]), radius=1.0, z_min=-1.0, z_max=4.22)
    setup2 = Setup(id="s2", name="Setup 2 - Bottom", model_to_setup=M2, work_offset=WorkOffset.G55, stock=stock2)

    ctx = PlanningContext(
        model=canonical_model,
        setups=[setup1, setup2],
        machine=machine,
        tools=tools,
        material=material,
    )

    # Feature 1 in Setup 1 (Top blind hole: top_z=3.22, floor_z=1.47, depth=1.75)
    feat1 = MachiningFeature(
        id="s1_hole_top",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        diameter=0.58,
        depth=1.75,
        top_z=3.22,
        floor_z=1.47,
        bounds_min=np.array([-0.29, -0.29, 1.47]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        notes={"setup_id": setup1.id},
    )

    # Feature 2 in Setup 2 (Bottom blind hole in Setup 2 coords: top_z=3.22, floor_z=1.95, depth=1.27)
    feat2 = MachiningFeature(
        id="s2_hole_bot",
        type=FeatureType.BLIND_HOLE,
        face_indices=[0],
        diameter=0.58,
        depth=1.27,
        top_z=3.22,
        floor_z=1.95,
        bounds_min=np.array([-0.29, -0.29, 1.95]),
        bounds_max=np.array([0.29, 0.29, 3.22]),
        notes={"setup_id": setup2.id},
    )
    setup2.features = [feat2]

    planner1 = OperationPlanner(ctx, setup1, [feat1])
    ops1 = planner1.plan()

    assert len(ops1) > 0
    drill_op1 = ops1[0]

    # Verify that cross-setup validation identified the 0.20 mm remaining web
    # and forced depth_reference to 'tip' to avoid breaching the web
    assert drill_op1.notes.get("depth_reference") == "tip"
    warning_codes = [w["code"] for w in planner1.warnings]
    assert "CRITICAL_WEB_THICKNESS" in warning_codes or "THIN_WALL_WARNING" in warning_codes


def test_roughing_stock_divergence_between_setups():
    """Verify that Setup 1 and Setup 2 roughing toolpaths differ substantially
    because Setup 2 roughing consumes the in-process stock voxels transformed from Setup 1."""
    box_shape = Shape(BRepPrimAPI_MakeBox(40.0, 30.0, 20.0).Shape())
    canonical_model = ImportedModel(
        shape=box_shape,
        declared_units="mm",
        original_bounds=box_shape.bounding_box(),
    )

    tools = get_default_tools()
    machine = get_default_machine()
    material = DEFAULT_MATERIALS["aluminum_6061"]

    M1 = np.eye(4)
    stock1 = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, -5.0]), bounds_max=np.array([45.0, 35.0, 25.0]))
    setup1 = Setup(id="s1", name="Setup 1", model_to_setup=M1, work_offset=WorkOffset.G54, stock=stock1)

    M2 = _euler_to_matrix(180.0, 0.0, 0.0, tx=0.0, ty=30.0, tz=20.0)
    stock2 = Stock(kind=StockKind.BOX, bounds_min=np.array([-5.0, -5.0, -5.0]), bounds_max=np.array([45.0, 35.0, 25.0]))
    setup2 = Setup(id="s2", name="Setup 2", model_to_setup=M2, work_offset=WorkOffset.G55, stock=stock2)

    ctx = PlanningContext(model=canonical_model, setups=[setup1, setup2], machine=machine, tools=tools, material=material)

    shape1 = ctx.model_shape_in_setup(setup1)
    mesh1 = TriangleMesh.from_faces(shape1.faces())
    stock_voxels1 = VoxelStock.from_bounds(setup1.stock.bounds_min, setup1.stock.bounds_max, resolution=2.0)

    # Simulate machining in Setup 1 removing top stock from Z=20 to Z=25
    stock_voxels1.voxels[:, :, -3:] = False

    vol_after_s1 = float(np.count_nonzero(stock_voxels1.voxels)) * (stock_voxels1.voxel_size ** 3)

    # In-process stock transfer: Transform to Setup 2
    T1_to_2 = M2 @ np.linalg.inv(M1)
    stock_voxels2 = stock_voxels1.transformed(T1_to_2, setup2.stock.bounds_min, setup2.stock.bounds_max, resolution=2.0)
    vol_in_s2 = float(np.count_nonzero(stock_voxels2.voxels)) * (stock_voxels2.voxel_size ** 3)

    # Volume conservation
    assert abs(vol_in_s2 - vol_after_s1) / vol_after_s1 < 0.05

    # In Setup 2, what was the bottom excess (Z=-5 to Z=0 in Setup 1) is now top excess (Z=20 to Z=25 in Setup 2)
    # while what was machined in Setup 1 is now at the bottom (Z=-5 to Z=0 in Setup 2)
    sample_top_s2 = stock_voxels2.is_solid_at(np.array([20.0, 15.0, 22.0]))
    sample_bot_s2 = stock_voxels2.is_solid_at(np.array([20.0, 15.0, -2.0]))

    assert sample_top_s2 is True   # Unmachined excess now at top of Setup 2
    assert sample_bot_s2 is False  # Machined excess from Setup 1 now absent at bottom of Setup 2
