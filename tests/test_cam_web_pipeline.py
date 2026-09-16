"""End-to-end test suite for VexCAM post-processors, API endpoints, and web stack."""

import pytest
from fastapi.testclient import TestClient
from app import app
from cam_engine.post import get_post_processor
from cam_engine.post.gcode_parser import GCodeValidator
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType
import numpy as np


@pytest.fixture
def client():
    return TestClient(app)


def test_post_processor_grbl():
    tp = Toolpath(
        operation_id="op_test_01",
        feature_id="feat_01",
        tool_id="T01",
        purpose="roughing",
    )
    tp.add(MotionSegment(MotionType.RAPID, start=np.array([0.0, 0.0, 30.0]), end=np.array([10.0, 10.0, 30.0]), tool_id="T01"))
    tp.add(MotionSegment(MotionType.PLUNGE, start=np.array([10.0, 10.0, 30.0]), end=np.array([10.0, 10.0, 15.0]), feed=400.0, spindle=6000.0, tool_id="T01"))
    tp.add(MotionSegment(MotionType.CUT, start=np.array([10.0, 10.0, 15.0]), end=np.array([50.0, 10.0, 15.0]), feed=1800.0, spindle=6000.0, tool_id="T01"))

    post = get_post_processor("grbl", work_offset="G54")
    res = post.post_process([tp], program_name="TEST_PROG")

    assert res.line_count > 5
    assert "G54" in res.gcode
    assert "G0" in res.gcode or "G00" in res.gcode
    assert "G1" in res.gcode or "G01" in res.gcode
    assert "S6000" in res.gcode
    assert res.total_cut_dist_mm > 0.0

    # Parse-back verification
    validator = GCodeValidator()
    blocks = validator.parse(res.gcode)
    assert len(blocks) > 0


def test_post_processor_fanuc():
    tp = Toolpath(
        operation_id="op_test_02",
        feature_id="feat_02",
        tool_id="T02",
        purpose="finishing",
    )
    tp.add(MotionSegment(MotionType.CUT, start=np.array([0.0, 0.0, 0.0]), end=np.array([20.0, 0.0, 0.0]), feed=1200.0, spindle=8000.0, tool_id="T02"))

    post = get_post_processor("fanuc", work_offset="G54")
    res = post.post_process([tp], program_name="FANUC_TEST")

    assert "O1001" in res.gcode
    assert "M30" in res.gcode
    assert "%" in res.gcode


def test_api_demo_model(client):
    res = client.get("/api/model/demo")
    assert res.status_code == 200
    data = res.json()
    assert "mesh" in data
    assert len(data["mesh"]["vertices"]) > 0
    assert data["mesh"]["triangle_count"] > 0
    assert "bounding_box" in data["mesh"]


def test_api_tools_and_machines(client):
    res = client.get("/api/tools-and-machines")
    assert res.status_code == 200
    data = res.json()
    assert len(data["tools"]) >= 4
    assert data["machine"]["id"] == "vmc-3axis"


def test_api_feature_recognition_and_ops(client):
    # 1. Demo model
    client.get("/api/model/demo")

    # 2. Recognize features
    res_feat = client.post("/api/recognize-features")
    assert res_feat.status_code == 200
    feats = res_feat.json()["features"]
    assert len(feats) >= 4

    # 3. Plan operations
    res_ops = client.post("/api/plan-operations", json={"stock": {"margin_x": 5.0, "margin_y": 5.0, "margin_z_top": 1.0, "margin_z_bottom": 3.0}})
    assert res_ops.status_code == 200
    ops = res_ops.json()["operations"]
    assert len(ops) >= 4

    # 4. Generate toolpaths
    res_tp = client.post("/api/generate-toolpaths", json={"stock": {"margin_x": 5.0, "margin_y": 5.0, "margin_z_top": 1.0, "margin_z_bottom": 3.0}})
    assert res_tp.status_code == 200
    tp_data = res_tp.json()
    assert len(tp_data["toolpaths"]) >= 4
    assert tp_data["total_cutting_length_mm"] > 0

    # 5. Generate G-Code
    res_gc = client.post("/api/generate-gcode", json={"controller": "grbl", "work_offset": "G54"})
    assert res_gc.status_code == 200
    gc_data = res_gc.json()
    assert gc_data["line_count"] > 10
    assert "G54" in gc_data["gcode"]


def test_static_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "VexCAM Studio" in res.text


def test_api_upload_step(client):
    import tempfile
    from pathlib import Path
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

    box = BRepPrimAPI_MakeBox(45.0, 35.0, 15.0).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(box, STEPControl_AsIs)
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        tmp_path = f.name
    writer.Write(tmp_path)

    try:
        with open(tmp_path, "rb") as f:
            res = client.post("/api/model/upload", files={"file": ("custom_box.step", f, "application/octet-stream")})
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "custom_box.step"
        assert data["mesh"]["triangle_count"] > 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)
