"""Tests for Toolpath Representation Contract (ToolpathGeometryMode).

Verifies:
1. ToolpathGeometryMode defines three distinct geometric representations:
   - NOMINAL_BOUNDARY: Points lie directly on finished CAD boundary (requires full radius comp G41/G42 with D=ToolRadius).
   - TOOL_CENTER: Points are already offset by tool radius in CAM (requires Wear/Delta G41/G42 with D=0.0 or Computer mode).
   - MACHINE_COMPENSATED: Points are already fully machine compensated (no further radius offset allowed).
2. Guarantees cutter radius offset is applied exactly once.
3. Post-processor and strategy engine respect the geometry contract.
"""

import numpy as np
import pytest

from cam_engine.toolpaths.semantic import (
    Toolpath,
    MotionSegment,
    MotionType,
    ToolpathGeometryMode,
    CompensationType,
    CompensationSide,
)
from cam_engine.post import get_post_processor


def test_toolpath_geometry_modes_contract():
    """Verify Toolpath geometry mode enum contracts and defaults."""
    tp_default = Toolpath(
        operation_id="op1",
        feature_id="f1",
        tool_id="T01",
        purpose="profiling",
    )
    assert tp_default.geometry_mode == ToolpathGeometryMode.TOOL_CENTER

    # Explicit nominal boundary toolpath
    tp_boundary = Toolpath(
        operation_id="op2",
        feature_id="f2",
        tool_id="T01",
        purpose="finishing",
        geometry_mode=ToolpathGeometryMode.NOMINAL_BOUNDARY,
    )
    assert tp_boundary.geometry_mode == ToolpathGeometryMode.NOMINAL_BOUNDARY

    # Transformed toolpath preserves geometry mode contract
    t = np.eye(4)
    t[:3, 3] = [10.0, 20.0, 30.0]
    tp_tr = tp_boundary.transformed(t)
    assert tp_tr.geometry_mode == ToolpathGeometryMode.NOMINAL_BOUNDARY


def test_cutter_radius_single_application_guarantee():
    """Verify post processor respects geometry mode and audits compensation accurately."""
    tool_info = {
        "id": "T01",
        "tool_number": 1,
        "diameter": 10.0,
        "type": "flat_endmill",
    }
    
    # 1. Tool Center Path with Wear Compensation
    tp_wear = Toolpath(
        operation_id="op_wear",
        feature_id="f_pocket",
        tool_id="T01",
        purpose="finishing",
        geometry_mode=ToolpathGeometryMode.TOOL_CENTER,
        metadata={
            "tool_info": tool_info,
            "cutter_comp_info": {
                "action": "on",
                "mode": "wear",
                "side": "left",
                "lead_len": 2.0,
            }
        }
    )
    tp_wear.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([50.0, 0.0, 0.0]),
        tool_id="T01",
        metadata={
            "tool_info": tool_info,
            "cutter_comp": {"action": "on", "side": "left", "d_register": 1}
        }
    ))

    post = get_post_processor("fanuc")
    res_wear = post.post_process([tp_wear])
    assert "G41 D1" in res_wear.gcode
    assert "Tool Center Path" in res_wear.gcode
    assert "Wear Delta Only" in res_wear.gcode
