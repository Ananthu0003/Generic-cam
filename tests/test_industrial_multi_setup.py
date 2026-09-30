"""Tests for Industrial CAM Multi-Setup Functionality:
1. In-process stock transformation across setups (OP10 -> OP20).
2. Per-setup and master G-code post-processing with retracts and stops.
3. Master routing sheet and per-setup setup sheet generation.
4. ZIP CAM package export API endpoint.
"""

import io
import zipfile
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app import app
from cam_engine.geometry.voxel import VoxelStock
from cam_engine.context import (
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Material,
    DEFAULT_MATERIALS,
    Tool,
    ToolType,
    WorkOffset,
)
from cam_engine.features import FeatureRecognizer
from cam_engine.operations import OperationPlanner
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.demo_models import create_prismatic_bracket_shape, get_default_tools, get_default_machine
from cam_engine.setup_sheet import (
    generate_setup_sheet,
    format_setup_sheet,
    generate_master_routing_sheet,
    format_master_routing_sheet,
)
from cam_engine.post import get_post_processor


def _euler_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    R = Rz @ Ry @ Rx
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


@pytest.fixture
def client():
    return TestClient(app)


def test_voxel_stock_transformation_conservation():
    """Test that VoxelStock.transformed conserves volume and correctly transforms coordinates."""
    bmin1 = np.array([0.0, 0.0, 0.0])
    bmax1 = np.array([100.0, 60.0, 30.0])
    res = 2.0
    stock1 = VoxelStock.from_bounds(bmin1, bmax1, res)
    initial_vol = stock1.remaining_volume()
    assert initial_vol > 0.0

    # 180 flip around X axis
    M1 = np.eye(4)
    M2 = _euler_to_matrix(180.0, 0.0, 0.0)
    T1_to_2 = M2 @ np.linalg.inv(M1)

    # In setup 2, bounds are Y: -60 to 0, Z: -30 to 0
    bmin2 = np.array([0.0, -60.0, -30.0])
    bmax2 = np.array([100.0, 0.0, 0.0])

    stock2 = stock1.transformed(T1_to_2, bmin2, bmax2, res)
    transformed_vol = stock2.remaining_volume()

    # Volume should be conserved within numerical discretization tolerance (< 3%)
    rel_diff = abs(transformed_vol - initial_vol) / initial_vol
    assert rel_diff < 0.03, f"Volume not conserved: initial {initial_vol}, transformed {transformed_vol}"


def test_multi_setup_end_to_end_planning_and_inprocess_stock():
    """Test full multi-setup workflow where OP10 machines top and OP20 machines flipped bottom."""
    imported = create_prismatic_bracket_shape()
    tools = get_default_tools()
    machine = get_default_machine()
    mat = DEFAULT_MATERIALS["aluminum_6061"]

    # Setup 1 (Top, G54)
    M1 = _euler_to_matrix(0.0, 0.0, 0.0)
    shape_s1 = imported.shape.transformed(M1)
    bmin1, bmax1 = shape_s1.bounding_box()
    stock1 = Stock(kind=StockKind.BOX, bounds_min=bmin1 - np.array([2, 2, 2]), bounds_max=bmax1 + np.array([2, 2, 2]))
    setup1 = Setup(id="setup_001", name="Setup 1 - Top", model_to_setup=M1, work_offset=WorkOffset.G54, stock=stock1)

    # Setup 2 (Flipped 180 around X, G55)
    M2 = _euler_to_matrix(180.0, 0.0, 0.0)
    shape_s2 = imported.shape.transformed(M2)
    bmin2, bmax2 = shape_s2.bounding_box()
    stock2 = Stock(kind=StockKind.BOX, bounds_min=bmin2 - np.array([2, 2, 2]), bounds_max=bmax2 + np.array([2, 2, 2]))
    setup2 = Setup(id="setup_002", name="Setup 2 - Bottom", model_to_setup=M2, work_offset=WorkOffset.G55, stock=stock2)

    ctx = PlanningContext(model=imported, setups=[setup1, setup2], machine=machine, tools=tools, material=mat)

    # Recognize features in both setups
    rec1 = FeatureRecognizer(shape_s1, stock_top_z=float(bmax1[2] + 2.0))
    feats1 = rec1.recognize()
    for f in feats1:
        f.notes["setup_id"] = setup1.id

    rec2 = FeatureRecognizer(shape_s2, stock_top_z=float(bmax2[2] + 2.0))
    feats2 = rec2.recognize()
    for f in feats2:
        f.notes["setup_id"] = setup2.id

    assert len(feats1) > 0
    assert len(feats2) > 0

    # Plan ops
    planner1 = OperationPlanner(ctx, setup1, feats1)
    ops1 = planner1.plan()
    for op in ops1:
        op.notes["setup_id"] = setup1.id

    planner2 = OperationPlanner(ctx, setup2, feats2)
    ops2 = planner2.plan()
    for op in ops2:
        op.notes["setup_id"] = setup2.id

    assert len(ops1) > 0
    assert len(ops2) > 0

    # Generate toolpaths with in-process stock transformation
    mesh1 = TriangleMesh.from_faces(shape_s1.faces())
    stock_voxels1 = VoxelStock.from_bounds(setup1.stock.bounds_min, setup1.stock.bounds_max, resolution=2.0)
    strat_ctx1 = StrategyContext(
        context=ctx, setup=setup1, mesh=mesh1, stock_voxels=stock_voxels1,
        clearance_z=float(setup1.stock.bounds_max[2]) + 10.0,
        finish_allowance_wall=0.3,
        finish_allowance_floor=0.3,
    )
    engine1 = StrategyEngine(strat_ctx1)
    tps1 = [engine1.generate(op) for op in ops1]
    for tp in tps1:
        tp.metadata["setup_id"] = setup1.id
        tp.metadata["work_offset"] = setup1.work_offset.value

    # Transform remaining stock from OP10 to OP20
    T1_to_2 = M2 @ np.linalg.inv(M1)
    stock_voxels2 = stock_voxels1.transformed(T1_to_2, setup2.stock.bounds_min, setup2.stock.bounds_max, resolution=2.0)

    mesh2 = TriangleMesh.from_faces(shape_s2.faces())
    strat_ctx2 = StrategyContext(
        context=ctx, setup=setup2, mesh=mesh2, stock_voxels=stock_voxels2,
        clearance_z=float(setup2.stock.bounds_max[2]) + 10.0,
        finish_allowance_wall=0.3,
        finish_allowance_floor=0.3,
    )
    engine2 = StrategyEngine(strat_ctx2)
    tps2 = [engine2.generate(op) for op in ops2]
    for tp in tps2:
        tp.metadata["setup_id"] = setup2.id
        tp.metadata["work_offset"] = setup2.work_offset.value

    assert len(tps1) > 0
    assert len(tps2) > 0

    # Test master routing sheet generation
    all_ops = ops1 + ops2
    all_tps = tps1 + tps2
    all_feats = feats1 + feats2
    master_sheet = generate_master_routing_sheet(
        program_name="PART_TEST_MASTER",
        part_name="Test Bracket",
        planning_ctx=ctx,
        setups=[setup1, setup2],
        all_features=all_feats,
        all_operations=all_ops,
        all_toolpaths=all_tps,
    )
    formatted_master = format_master_routing_sheet(master_sheet)
    assert "MULTI-SETUP MASTER ROUTING SHEET" in formatted_master
    assert "OP10" in formatted_master
    assert "OP20" in formatted_master
    assert "G54" in formatted_master
    assert "G55" in formatted_master


def test_api_multi_setup_endpoints(client):
    """Test REST API routes for multi-setup operations, gcode download, and export package."""
    # 1. Load demo model
    r = client.get("/api/model/demo")
    assert r.status_code == 200

    # 2. Add OP20 setup
    r = client.post("/api/setups/add", json={
        "name": "Setup 2 - Bottom Flip",
        "work_offset": "G55",
        "preset": "flip_x_180",
    })
    assert r.status_code == 200
    setups_data = r.json()
    assert len(setups_data["setups"]) == 2
    setup2_id = setups_data["setups"][1]["id"]

    # 3. Recognize features
    r = client.post("/api/recognize-features")
    assert r.status_code == 200

    # 4. Plan operations
    r = client.post("/api/plan-operations", json={"stock": {}})
    assert r.status_code == 200

    # 5. Generate toolpaths
    r = client.post("/api/generate-toolpaths", json={"stock": {}})
    assert r.status_code == 200
    tp_data = r.json()
    assert len(tp_data["toolpaths"]) > 0

    # 6. Combined G-code with program stop and machine retract
    r = client.post("/api/generate-gcode", json={
        "controller": "haas",
        "program_name": "TEST_PART",
        "all_setups": True,
        "separate_files": False,
    })
    assert r.status_code == 200
    combined = r.json()
    assert "G54" in combined["gcode"]
    assert "G55" in combined["gcode"]
    assert "M00" in combined["gcode"]
    assert "G53 G00 Z0." in combined["gcode"]
    assert "M05" in combined["gcode"]

    # 7. Download individual setup NC
    r = client.get(f"/api/gcode/download/setup_001?controller=haas")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "G54" in r.text

    r = client.get(f"/api/gcode/download/{setup2_id}?controller=haas")
    assert r.status_code == 200
    assert "G55" in r.text

    # 8. Setup sheets
    r = client.get("/api/setup-sheet-master")
    assert r.status_code == 200
    sheet_data = r.json()
    assert "MASTER ROUTING SHEET" in sheet_data["text"]

    r = client.get("/api/setup-sheet/setup_001")
    assert r.status_code == 200
    assert "SETUP SHEET" in r.json()["text"]

    # 9. Export package ZIP
    r = client.get("/api/export-package?controller=haas")
    assert r.status_code == 200
    assert "application/zip" in r.headers["content-type"]

    zip_bytes = io.BytesIO(r.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        namelist = zf.namelist()
        # Verify contains NC_Programs and Setup_Sheets and MASTER_ROUTING_SHEET
        assert any("MASTER_ROUTING_SHEET.txt" in name for name in namelist)
        assert any("NC_Programs/" in name and "_OP10_G54.nc" in name for name in namelist)
        assert any("NC_Programs/" in name and "_OP20_G55.nc" in name for name in namelist)
        assert any("NC_Programs/" in name and "MASTER_ALL_SETUPS.nc" in name for name in namelist)
        assert any("Setup_Sheets/" in name and "SETUP_SHEET_" in name for name in namelist)
