"""Unit and integration tests for Real CAM Stock Engine (Fixed Billet & Cylindrical Stock)."""

import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from app import app
from cam_engine.context import Stock, StockKind, Setup, WorkOffset, DEFAULT_MATERIALS
from cam_engine.api.models import StockConfig, UpdateSetupRequest, PlanOpsRequest, GenerateToolpathsRequest
from cam_engine.api.routes import _create_stock_from_config
from cam_engine.pipeline import run_cam_pipeline
from cam_engine.setup_sheet import generate_setup_sheet, format_setup_sheet
from cam_engine.demo_models import get_default_tools, get_default_machine


@pytest.fixture
def client():
    return TestClient(app)


def test_stock_dataclass_box_and_cylinder():
    # Box stock
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([100.0, 50.0, 25.0])
    box_stock = Stock(kind=StockKind.BOX, bounds_min=bmin, bounds_max=bmax)
    assert box_stock.top_z() == 25.0
    assert "100.0 x 50.0 x 25.0 mm" in box_stock.dimension_description()
    assert "(Rectangular Block)" in box_stock.dimension_description()

    # Cylindrical stock
    cyl_stock = Stock.from_cylinder(
        center_xy=np.array([50.0, 25.0]),
        z_min=0.0,
        z_max=30.0,
        radius=40.0,
        axis=np.array([0.0, 0.0, 1.0]),
    )
    assert cyl_stock.kind == StockKind.CYLINDER
    assert cyl_stock.cylinder_radius == 40.0
    assert cyl_stock.top_z() == 30.0
    assert "Ø80.0 x 30.0 mm" in cyl_stock.dimension_description()
    assert "(Cylindrical Bar)" in cyl_stock.dimension_description()


def test_create_stock_from_config_relative():
    cfg = StockConfig(
        stock_mode="relative_box",
        margin_x=5.0,
        margin_y=5.0,
        margin_z_top=1.0,
        margin_z_bottom=3.0,
        offset_x=2.0,
    )
    bmin = np.array([10.0, 10.0, 0.0])
    bmax = np.array([90.0, 50.0, 20.0])

    stock = _create_stock_from_config(cfg, bmin, bmax)
    assert stock.kind == StockKind.BOX
    np.testing.assert_allclose(stock.bounds_min, np.array([7.0, 5.0, -3.0]))
    np.testing.assert_allclose(stock.bounds_max, np.array([97.0, 55.0, 21.0]))


def test_create_stock_from_config_fixed_box():
    cfg = StockConfig(
        stock_mode="fixed_box",
        fixed_size_x=120.0,
        fixed_size_y=80.0,
        fixed_size_z=30.0,
        margin_z_top=2.0,
    )
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([100.0, 60.0, 20.0])

    stock = _create_stock_from_config(cfg, bmin, bmax)
    assert stock.kind == StockKind.BOX
    dims = stock.bounds_max - stock.bounds_min
    np.testing.assert_allclose(dims, np.array([120.0, 80.0, 30.0]))
    assert stock.bounds_max[2] == 22.0  # bmax[2] + margin_z_top = 20 + 2
    assert stock.bounds_min[2] == -8.0  # 22 - 30


def test_create_stock_from_config_relative_cylinder():
    cfg = StockConfig(
        stock_mode="relative_cylinder",
        cylinder_margin_radial=3.0,
        cylinder_margin_axial_top=1.5,
        cylinder_margin_axial_bot=15.0,
        cylinder_axis="Z",
    )
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([40.0, 40.0, 20.0])

    stock = _create_stock_from_config(cfg, bmin, bmax)
    assert stock.kind == StockKind.CYLINDER
    # Part radius in XY: hypot(20, 20) = 28.284mm + 3.0mm margin = 31.284mm
    expected_r = np.hypot(20.0, 20.0) + 3.0
    assert abs(stock.cylinder_radius - expected_r) < 1e-4
    assert stock.top_z() == 21.5  # 20 + 1.5
    assert stock.bounds_min[2] == -15.0  # 0 - 15.0


def test_create_stock_from_config_fixed_cylinder():
    cfg = StockConfig(
        stock_mode="fixed_cylinder",
        cylinder_diameter=90.0,
        cylinder_length=50.0,
        cylinder_axis="Z",
        cylinder_margin_axial_top=2.0,
    )
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([60.0, 60.0, 25.0])

    stock = _create_stock_from_config(cfg, bmin, bmax)
    assert stock.kind == StockKind.CYLINDER
    assert stock.cylinder_radius == 45.0
    assert stock.top_z() == 27.0  # 25 + 2
    assert stock.bounds_min[2] == -23.0  # 27 - 50


def test_setup_sheet_with_cylindrical_stock():
    cyl_stock = Stock.from_cylinder(
        center_xy=np.array([0.0, 0.0]),
        z_min=-5.0,
        z_max=25.0,
        radius=35.0,
    )
    setup = Setup(
        id="setup_001",
        name="Setup 1 - Cylindrical Turning/Milling",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=cyl_stock,
    )
    from cam_engine.context import PlanningContext
    ctx = PlanningContext(
        model=None,
        setups=[setup],
        machine=get_default_machine(),
        tools=get_default_tools(),
        material=DEFAULT_MATERIALS["aluminum_6061"],
    )

    sheet = generate_setup_sheet(
        program_name="PROG_CYL",
        part_name="Round_Shaft",
        planning_ctx=ctx,
        setup=setup,
        features=[],
        operations=[],
        toolpaths=[],
    )

    formatted = format_setup_sheet(sheet)
    assert "Cylindrical Bar" in formatted or "Ø70.0" in formatted
    assert "Cylindrical stock" in formatted


def test_api_stock_mode_workflow(client):
    # 1. Load demo model
    res = client.get("/api/model/demo")
    assert res.status_code == 200

    # 2. Update setup to use cylindrical stock
    cyl_cfg = {
        "stock_mode": "cylinder",
        "cylinder_diameter": 95.0,
        "cylinder_length": 35.0,
        "cylinder_axis": "Z",
        "material": "aluminum_6061",
        "work_offset": "G54",
        "clamp_type": "vise_jaws",
    }
    update_res = client.post("/api/setups/update/setup_001", json={"stock": cyl_cfg})
    assert update_res.status_code == 200
    setups = update_res.json()["setups"]
    assert setups[0]["stock"]["stock_mode"] == "cylinder"
    assert setups[0]["stock"]["cylinder_diameter"] == 95.0

    # 3. Recognize features
    res_feat = client.post("/api/recognize-features")
    assert res_feat.status_code == 200

    # 4. Plan operations & generate toolpaths with cylindrical stock
    plan_res = client.post("/api/plan-operations", json={"stock": cyl_cfg})
    assert plan_res.status_code == 200

    tp_res = client.post("/api/generate-toolpaths", json={"stock": cyl_cfg})
    assert tp_res.status_code == 200
    data = tp_res.json()
    assert len(data["toolpaths"]) > 0
    assert data["total_cutting_length_mm"] > 0.0

    # 5. Generate post processing / G-code
    post_res = client.post("/api/generate-gcode", json={"controller": "grbl", "work_offset": "G54"})
    assert post_res.status_code == 200
    assert "G54" in post_res.json()["gcode"]


def test_pipeline_with_cylinder_and_fixed_stock():
    step_file = Path(__file__).parent.parent / "sample_bracket.step"
    if not step_file.exists():
        pytest.skip("sample_bracket.step not found")

    # Cylindrical pipeline run
    res_cyl = run_cam_pipeline(
        step_path=step_file,
        stock_mode="cylinder",
        cylinder_diameter=90.0,
        cylinder_length=40.0,
    )
    assert len(res_cyl.toolpaths) > 0
    assert res_cyl.post_result.total_time_seconds > 0.0

    # Fixed billet pipeline run
    res_fixed = run_cam_pipeline(
        step_path=step_file,
        stock_mode="fixed_box",
        fixed_size=(100.0, 70.0, 30.0),
    )
    assert len(res_fixed.toolpaths) > 0
    assert res_fixed.post_result.total_time_seconds > 0.0
