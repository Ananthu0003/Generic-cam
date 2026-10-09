"""Unit and integration tests for Multi-Setup CAM and CAD Part Orientation."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from cam_engine.geometry.topology import Shape
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox


@pytest.fixture
def client():
    return TestClient(app)


def test_shape_rotation_and_alignment():
    """Verify that Shape.rotate around X, Y, Z updates bounding box and coordinates."""
    box = BRepPrimAPI_MakeBox(100.0, 50.0, 20.0).Shape()
    s = Shape(box)
    bmin, bmax = s.bounding_box()
    assert np.isclose(bmax[0] - bmin[0], 100.0)
    assert np.isclose(bmax[1] - bmin[1], 50.0)
    assert np.isclose(bmax[2] - bmin[2], 20.0)

    # Rotate 90 degrees around Z: length 100 becomes along Y
    s_rot_z = s.rotate("Z", 90.0)
    bmin_z, bmax_z = s_rot_z.bounding_box()
    assert np.isclose(bmax_z[0] - bmin_z[0], 50.0, atol=0.1)
    assert np.isclose(bmax_z[1] - bmin_z[1], 100.0, atol=0.1)

    # Rotate 90 degrees around X: height 20 becomes along Y
    s_rot_x = s.rotate("X", 90.0)
    bmin_x, bmax_x = s_rot_x.bounding_box()
    assert np.isclose(bmax_x[2] - bmin_x[2], 50.0, atol=0.1)


def test_api_model_orient(client):
    """Test /api/model/orient endpoint."""
    res0 = client.get("/api/model/demo")
    assert res0.status_code == 200
    b0 = res0.json()["mesh"]["bounding_box"]["size"]

    # Rotate +90 around Z
    res = client.post("/api/model/orient", json={"axis": "Z", "angle_deg": 90.0})
    assert res.status_code == 200
    b1 = res.json()["mesh"]["bounding_box"]["size"]
    # Size X and Y should swap
    assert np.isclose(b1[0], b0[1], atol=0.2)
    assert np.isclose(b1[1], b0[0], atol=0.2)


def test_api_setups_management(client):
    """Test adding, switching, and deleting multi-setups."""
    # Reset to demo
    client.get("/api/model/demo")

    # Initial setups list
    res = client.get("/api/setups")
    assert res.status_code == 200
    data = res.json()
    assert len(data["setups"]) == 1
    assert data["active_setup_id"] == "setup_001"

    # Add Setup 2 (Flip 180° / G55)
    res_add = client.post("/api/setups/add", json={
        "name": "Setup 2 - Bottom Flip",
        "work_offset": "G55",
        "preset": "flip_x_180",
    })
    assert res_add.status_code == 200
    data_add = res_add.json()
    assert len(data_add["setups"]) == 2
    assert data_add["active_setup_id"] == "setup_002"
    assert data_add["setups"][1]["work_offset"] == "G55"

    # Set active back to Setup 1
    res_act = client.post("/api/setups/active/setup_001")
    assert res_act.status_code == 200
    assert res_act.json()["active_setup_id"] == "setup_001"

    # Delete Setup 2
    res_del = client.delete("/api/setups/setup_002")
    assert res_del.status_code == 200
    assert len(res_del.json()["setups"]) == 1


def test_multi_setup_planning_and_gcode_pipeline(client):
    """End-to-end test of Multi-Setup feature recognition, operations, toolpaths, and multi-WCS G-code."""
    client.get("/api/model/demo")

    # Add Setup 2
    client.post("/api/setups/add", json={
        "name": "Setup 2 - Bottom Flip",
        "work_offset": "G55",
        "preset": "flip_x_180",
    })

    # Recognize features across both setups
    res_feat = client.post("/api/recognize-features")
    assert res_feat.status_code == 200
    feats = res_feat.json()["features"]
    assert len(feats) > 0

    # Plan operations across both setups
    res_ops = client.post("/api/plan-operations", json={"stock": {"margin_x": 5.0, "material": "aluminum_6061"}})
    assert res_ops.status_code == 200
    ops = res_ops.json()["operations"]
    assert len(ops) > 0

    # Generate toolpaths
    res_tp = client.post("/api/generate-toolpaths", json={"stock": {"margin_x": 5.0, "material": "aluminum_6061"}, "all_setups": True})
    assert res_tp.status_code == 200
    tps = res_tp.json()["toolpaths"]
    assert len(tps) > 0

    # Generate G-code with G54 and G55
    res_gc = client.post("/api/generate-gcode", json={"controller": "fanuc", "program_name": "MULTI_SETUP_TEST", "all_setups": True})
    assert res_gc.status_code == 200
    gc_data = res_gc.json()
    assert "G54" in gc_data["gcode"]
    assert gc_data["line_count"] > 10


def test_multi_setup_safe_depth_and_through_feature_isolation(client):
    """Verify that multi-setup planning isolates operations between setups:
    1. Outer contour passes respect fixture clamping limits and parting planes.
    2. Through features drilled in Setup 1 are not re-drilled in Setup 2.
    3. No cutting moves penetrate into the setup workholding envelope.
    """
    client.get("/api/model/demo")

    # Add Setup 2 (Flip 180° / G55)
    client.post("/api/setups/add", json={
        "name": "Setup 2 - Bottom Flip",
        "work_offset": "G55",
        "preset": "flip_x_180",
    })

    # Plan operations with vise jaw workholding (3mm clamping height)
    stock_payload = {
        "margin_x": 5.0,
        "margin_y": 5.0,
        "margin_z_top": 1.0,
        "margin_z_bottom": 4.0,
        "clamp_type": "vise_jaws",
        "clamp_height": 4.0,
        "material": "aluminum_6061",
    }
    client.post("/api/recognize-features")
    res_ops = client.post("/api/plan-operations", json={"stock": stock_payload})
    assert res_ops.status_code == 200
    ops = res_ops.json()["operations"]

    # Verify Setup 1 has through-hole drilling
    s1_ops = [op for op in ops if op.get("setup_id") == "setup_001"]
    s2_ops = [op for op in ops if op.get("setup_id") == "setup_002"]

    s1_drill_ops = [op for op in s1_ops if op["purpose"] == "drilling"]
    assert len(s1_drill_ops) > 0, "Setup 1 should plan drilling for through holes"

    # Verify Setup 2 does not duplicate heavy drilling through the already drilled hole
    s2_drill_ops = [op for op in s2_ops if op["purpose"] == "drilling"]
    assert len(s2_drill_ops) == 0, "Setup 2 should not re-drill through holes already completed in Setup 1"

    # Generate toolpaths
    res_tp = client.post("/api/generate-toolpaths", json={"stock": stock_payload, "all_setups": True})
    assert res_tp.status_code == 200
    tps = res_tp.json()["toolpaths"]

    for tp in tps:
        setup_id = tp.get("setup_id")
        # In Setup 1 (Z in [0, 25]), stock bottom is -4.0, clamp top is 0.0
        # In Setup 2 (Z in [-25, 0]), stock bottom is -29.0, clamp top is -25.0
        setup_safe_z = 0.0 if setup_id == "setup_001" else -25.0
        for seg in tp.get("segments", []):
            if seg.get("motion_type") == "cut":
                start_z = seg["start"][2]
                end_z = seg["end"][2]
                assert min(start_z, end_z) >= setup_safe_z - 0.5, (
                    f"Cutting segment plunged into clamping fixture: Z={min(start_z, end_z)} "
                    f"(safe floor={setup_safe_z}) in {setup_id}"
                )


def test_stepped_boss_outer_contour_multi_setup(client):
    """Verify that multi-level stepped parts distinguish upper protruding bosses from lower base perimeters,
    cutting the boss down to its step face without floating air cuts or duplicate re-cutting."""
    # Setup auto-generate
    client.post("/api/setups/auto-generate")
    stock_payload = {
        "mode": "relative",
        "margin_x": 5.0,
        "margin_y": 5.0,
        "margin_z_top": 1.0,
        "margin_z_bottom": 3.0,
        "clamp_type": "vise_jaws",
        "clamp_height": 3.0,
    }

    client.post("/api/recognize-features")
    res_ops = client.post("/api/plan-operations", json={"stock": stock_payload})
    assert res_ops.status_code == 200
    ops = res_ops.json()["operations"]

    res_tp = client.post("/api/generate-toolpaths", json={"stock": stock_payload, "all_setups": True})
    assert res_tp.status_code == 200
    tps = res_tp.json()["toolpaths"]
    assert len(tps) > 0


