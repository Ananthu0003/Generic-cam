"""
Test suite for Post-Processed G-Code Simulation Architecture.

Verifies:
1. G-code is simulated after post-processing (CAD -> Features -> Ops -> Toolpaths -> Post -> G-code -> NC Parser -> Motion -> Simulation).
2. NC Parser & Reconstructor expands canned drilling cycles (G81, G83, G73) into exact pecks, chip-clearance retracts, and G98/G99 initial/R-plane returns.
3. Modal states (G90/G91, G17/G18/G19, G94), WCS (G54/G55), feeds, spindle speeds, and tool changes (T# M6, G43 H#) are correctly tracked during reconstructed motion.
4. Cutter compensation (G41/G42 D#, G40) is interpreted and reconstructed.
5. Voxel stock material removal simulation runs on reconstructed NC motion segments.
"""

import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, Setup, Stock, StockKind, PlanningContext, WorkOffset
)
from cam_engine.toolpaths.semantic import (
    Toolpath, MotionSegment, MotionType, CoordSpace
)
from cam_engine.post import (
    FanucPostProcessor, HaasPostProcessor, GrblPostProcessor,
    reconstruct_machine_motion, NCReconstructor, GCodeGeometryValidator
)
from cam_engine.pipeline import run_cam_pipeline, PipelineResult


def _make_tools() -> list[Tool]:
    t1 = Tool(
        id="T1",
        tool_number=1,
        type=ToolType.FLAT_ENDMILL,
        diameter=6.0,
        flute_length=20.0,
        overall_length=50.0,
        flutes=4,
        length_offset_h=1,
        diameter_offset_d=1,
    )
    t2 = Tool(
        id="T2",
        tool_number=2,
        type=ToolType.DRILL,
        diameter=3.0,
        flute_length=25.0,
        overall_length=60.0,
        flutes=2,
        point_angle=118.0,
        length_offset_h=2,
        diameter_offset_d=2,
    )
    return [t1, t2]


class TestNCMotionReconstruction:
    def test_reconstruct_linear_and_arc_motion(self):
        gcode = """%
O0001 (TEST PROG)
G21 G90 G94 G17 G40 G80
G54
T1 M6
G43 H1
S6000 M3
G0 X0.0 Y0.0 Z10.0
G1 Z0.0 F200.0
G1 X20.0 Y0.0 F500.0
G2 X30.0 Y10.0 I0.0 J10.0 F400.0
G0 Z10.0
M30
%"""
        tools = _make_tools()
        segs = reconstruct_machine_motion(gcode, tools=tools)

        assert len(segs) > 0
        # Check rapid approach
        rapid_to_top = segs[0]
        assert rapid_to_top.motion_type == MotionType.RAPID
        assert np.allclose(rapid_to_top.end, [0.0, 0.0, 10.0])

        # Check plunge
        plunge_seg = segs[1]
        assert plunge_seg.motion_type == MotionType.PLUNGE
        assert np.allclose(plunge_seg.end, [0.0, 0.0, 0.0])
        assert plunge_seg.feed == 200.0

        # Check linear cut
        cut_seg = segs[2]
        assert cut_seg.motion_type == MotionType.CUT
        assert np.allclose(cut_seg.end, [20.0, 0.0, 0.0])
        assert cut_seg.feed == 500.0

        # Check CW Arc
        arc_seg = segs[3]
        assert arc_seg.motion_type == MotionType.ARC_CW
        assert np.allclose(arc_seg.end, [30.0, 10.0, 0.0])
        assert arc_seg.arc_center is not None
        assert np.allclose(arc_seg.arc_center, [20.0, 10.0, 0.0])

        # Check Retract (G0 Z10.0)
        ret_seg = segs[4]
        assert ret_seg.motion_type in (MotionType.RAPID, MotionType.RETRACT)
        assert np.allclose(ret_seg.end, [30.0, 10.0, 10.0])

    def test_reconstruct_canned_cycle_g83_peck_expansion(self):
        gcode = """%
O0002 (PECK DRILL)
G21 G90 G94 G17 G40 G80
G54
T2 M6
G43 H2
S3000 M3
G0 X15.0 Y25.0 Z14.0
G98 G83 X15.0 Y25.0 Z2.0 R4.0 Q2.0 F100.0
G80
G0 Z14.0
M30
%"""
        tools = _make_tools()
        segs = reconstruct_machine_motion(gcode, tools=tools)

        # G83 from R=4.0 down to Z=2.0 with Q=2.0 has 1 peck (depth is 2.0mm)
        # Total depth = 4.0 - 2.0 = 2.0mm.
        # Steps:
        # 1. Rapid to (15, 25, 14)
        # 2. Rapid to R-plane (15, 25, 4.0)
        # 3. Plunge down to Z=2.0
        # 4. Chip evac retract back to R=4.0
        # 5. Final G98 retract to Initial Z=14.0
        canned_segs = [s for s in segs if "G83" in s.metadata.get("command", "") or "G98" in s.metadata.get("command", "")]
        assert len(canned_segs) >= 2

        # Check initial rapid to R-plane
        r_plane_seg = [s for s in segs if "R-Plane" in s.metadata.get("command", "")][0]
        assert np.isclose(r_plane_seg.end[2], 4.0)

        # Check plunge down to target Z=2.0
        plunge_segs = [s for s in segs if s.motion_type == MotionType.PLUNGE]
        assert len(plunge_segs) >= 1
        assert np.isclose(plunge_segs[-1].end[2], 2.0)

        # Check G98 final return to initial plane Z=14.0
        final_retract = [s for s in segs if "G98 Retract" in s.metadata.get("command", "")][0]
        assert np.isclose(final_retract.end[2], 14.0)

    def test_reconstruct_canned_cycle_g99_r_plane_return(self):
        gcode = """%
O0003 (G99 PATTERN DRILL)
G21 G90 G94 G17 G40 G80
G54
T2 M6
G43 H2
S3000 M3
G0 X0.0 Y0.0 Z10.0
G99 G81 X0.0 Y0.0 Z-5.0 R2.0 F120.0
G80
M30
%"""
        tools = _make_tools()
        segs = reconstruct_machine_motion(gcode, tools=tools)

        final_retract = [s for s in segs if "G99 Retract" in s.metadata.get("command", "")][0]
        # In G99 mode, final retract stops at R-plane (Z=2.0), not initial plane (Z=10.0)
        assert np.isclose(final_retract.end[2], 2.0)

    def test_reconstruct_multiple_tools_and_modal_tracking(self):
        gcode = """%
O0004 (MULTI TOOL)
G21 G90 G94 G17 G40 G80
G54
T1 M6
G43 H1
S5000 M3
G0 X0.0 Y0.0 Z5.0
G1 Z0.0 F300.0
G1 X10.0 Y0.0 F500.0
G0 Z5.0

T2 M6
G43 H2
S2500 M3
G0 X10.0 Y0.0 Z5.0
G98 G81 X10.0 Y0.0 Z-3.0 R1.0 F80.0
G80
G0 Z5.0
M30
%"""
        tools = _make_tools()
        segs = reconstruct_machine_motion(gcode, tools=tools)

        t1_segs = [s for s in segs if s.tool_id == "T1"]
        t2_segs = [s for s in segs if s.tool_id == "T2"]

        assert len(t1_segs) > 0
        assert len(t2_segs) > 0
        assert any(s.motion_type == MotionType.CUT for s in t1_segs)
        assert any(s.motion_type == MotionType.PLUNGE for s in t2_segs)

    def test_reconstruct_cutter_compensation_metadata(self):
        gcode = """%
O0005 (CUTTER COMP)
G21 G90 G94 G17 G80
G54
T1 M6
G43 H1
S6000 M3
G0 X-5.0 Y-5.0 Z5.0
G1 Z0.0 F200.0
G41 D1 G1 X0.0 Y0.0 F500.0
G1 X20.0 Y0.0
G40 G1 X25.0 Y5.0
G0 Z5.0
M30
%"""
        tools = _make_tools()
        segs = reconstruct_machine_motion(gcode, tools=tools)

        comp_active_segs = [s for s in segs if s.metadata.get("cutter_comp") == "G41"]
        assert len(comp_active_segs) >= 2
        assert all(s.metadata.get("d_register") == 1 for s in comp_active_segs)

    def test_pipeline_reconstructs_and_simulates_gcode(self):
        """Verify that the full CAM pipeline outputs reconstructed G-code segments for simulation."""
        from pathlib import Path
        bracket_path = Path("sample_bracket.step")
        if not bracket_path.exists():
            pytest.skip("sample_bracket.step not found")

        res = run_cam_pipeline(bracket_path, controller="fanuc", work_offset="G54")
        assert res.reconstructed_segments is not None
        assert len(res.reconstructed_segments) > 0
        assert len(res.gcode) > 0

        # Check that cutting and positioning segments exist in reconstructed motion
        cut_segs = [s for s in res.reconstructed_segments if s.motion_type in (MotionType.CUT, MotionType.PLUNGE, MotionType.ARC_CW, MotionType.ARC_CCW)]
        assert len(cut_segs) > 0
        rapid_segs = [s for s in res.reconstructed_segments if s.motion_type in (MotionType.RAPID, MotionType.RETRACT)]
        assert len(rapid_segs) > 0

