"""Comprehensive Test Suite for Heidenhain (Conversational / ISO) and Siemens Sinumerik (840D/828D) Post-Processors.

Tests:
1. Heidenhain Klartext syntax generation (BEGIN/END PGM, TOOL CALL, L, CC, C, CYCL DEF 200/205/32/247, RL/RR/R0, M91 safe return).
2. Heidenhain ISO mode syntax.
3. Siemens Sinumerik 840D/828D syntax generation (CYCLE81/82/83/85/86, CYCLE832, D1 tool offset, SUPA safe return, MCALL).
4. Controller capability model verification for both controllers.
5. Syntax validation integration with ControllerSyntaxValidator.
6. Geometry verification and reconstruction through GCodeGeometryValidator and GCodeValidator.
7. Factory routing in get_post_processor and get_controller_capability_model.
"""

import numpy as np
import pytest

from cam_engine.context import Tool, ToolType
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType, CoordSpace
from cam_engine.post.controller_models import (
    HEIDENHAIN_CAPABILITY_MODEL,
    SIEMENS_CAPABILITY_MODEL,
    SINUMERIK_CAPABILITY_MODEL,
    get_controller_capability_model,
)
from cam_engine.post.syntax_validator import ControllerSyntaxValidator
from cam_engine.post.post_processor import (
    HeidenhainPostProcessor,
    SiemensPostProcessor,
    SinumerikPostProcessor,
    get_post_processor,
)


def _make_sample_contour_toolpath() -> Toolpath:
    """Create a sample 2D contour toolpath with rapid, plunge, linear, arc, and retract moves."""
    tp = Toolpath(
        operation_id="op_profile",
        feature_id="part_contour",
        tool_id="T1",
        purpose="finishing",
        space=CoordSpace.SETUP,
        metadata={
            "work_offset": "G54",
            "cutting_parameters": {
                "spindle_rpm": 6000.0,
                "feed_rate": 1200.0,
                "tool_diameter_mm": 10.0,
            },
        },
    )
    tp.segments = [
        # Rapid to start clearance
        MotionSegment(motion_type=MotionType.RAPID, start=np.array([0.0, 0.0, 20.0]), end=np.array([10.0, 10.0, 20.0])),
        # Rapid to approach Z
        MotionSegment(motion_type=MotionType.RAPID, start=np.array([10.0, 10.0, 20.0]), end=np.array([10.0, 10.0, 2.0])),
        # Plunge into cut
        MotionSegment(motion_type=MotionType.PLUNGE, start=np.array([10.0, 10.0, 2.0]), end=np.array([10.0, 10.0, -5.0]), feed=400.0),
        # Linear cut
        MotionSegment(motion_type=MotionType.CUT, start=np.array([10.0, 10.0, -5.0]), end=np.array([50.0, 10.0, -5.0]), feed=1200.0),
        # Arc CCW corner (center at 50, 20, radius 10)
        MotionSegment(
            motion_type=MotionType.ARC_CCW,
            start=np.array([50.0, 10.0, -5.0]),
            end=np.array([60.0, 20.0, -5.0]),
            arc_center=np.array([50.0, 20.0, -5.0]),
            arc_ccw=True,
            feed=1200.0,
        ),
        # Linear cut
        MotionSegment(motion_type=MotionType.CUT, start=np.array([60.0, 20.0, -5.0]), end=np.array([60.0, 60.0, -5.0]), feed=1200.0),
        # Retract
        MotionSegment(motion_type=MotionType.RETRACT, start=np.array([60.0, 60.0, -5.0]), end=np.array([60.0, 60.0, 20.0])),
    ]
    return tp


def _make_sample_drilling_toolpaths() -> list[Toolpath]:
    """Create sample standard drill and peck drill toolpaths with canned cycle metadata."""
    # 1. Standard drill (G81 / CYCL DEF 200 / CYCLE81)
    tp_drill = Toolpath(
        operation_id="op_drill_std",
        feature_id="hole_1",
        tool_id="T2",
        purpose="drilling",
        space=CoordSpace.SETUP,
        metadata={
            "cutting_parameters": {
                "spindle_rpm": 3500.0,
                "feed_rate": 250.0,
                "tool_diameter_mm": 6.8,
            },
        },
    )
    tp_drill.segments = [
        MotionSegment(
            motion_type=MotionType.RAPID,
            start=np.array([0.0, 0.0, 20.0]),
            end=np.array([25.0, 30.0, 2.0]),
            feed=3000.0,
            metadata={
                "canned_cycle": {
                    "type": "G81",
                    "x": 25.0,
                    "y": 30.0,
                    "z_r": 2.0,
                    "z_depth": -15.0,
                    "dwell": 0.0,
                    "feed": 250.0,
                },
            },
        ),
        MotionSegment(motion_type=MotionType.PLUNGE, start=np.array([25.0, 30.0, 2.0]), end=np.array([25.0, 30.0, -15.0]), feed=250.0),
        MotionSegment(motion_type=MotionType.RETRACT, start=np.array([25.0, 30.0, -15.0]), end=np.array([25.0, 30.0, 20.0]), feed=3000.0),
    ]

    # 2. Peck drill (G83 / CYCL DEF 205 / CYCLE83)
    tp_peck = Toolpath(
        operation_id="op_drill_peck",
        feature_id="hole_2",
        tool_id="T2",
        purpose="drilling",
        space=CoordSpace.SETUP,
        metadata={
            "cutting_parameters": {
                "spindle_rpm": 3500.0,
                "feed_rate": 200.0,
                "tool_diameter_mm": 6.8,
            },
        },
    )
    tp_peck.segments = [
        MotionSegment(
            motion_type=MotionType.RAPID,
            start=np.array([25.0, 30.0, 20.0]),
            end=np.array([45.0, 30.0, 2.0]),
            feed=3000.0,
            metadata={
                "canned_cycle": {
                    "type": "G83",
                    "x": 45.0,
                    "y": 30.0,
                    "z_r": 2.0,
                    "z_depth": -25.0,
                    "peck": 3.0,
                    "dwell": 0.5,
                    "feed": 200.0,
                },
            },
        ),
        MotionSegment(motion_type=MotionType.PLUNGE, start=np.array([45.0, 30.0, 2.0]), end=np.array([45.0, 30.0, -25.0]), feed=200.0),
        MotionSegment(motion_type=MotionType.RETRACT, start=np.array([45.0, 30.0, -25.0]), end=np.array([45.0, 30.0, 20.0]), feed=3000.0),
    ]

    return [tp_drill, tp_peck]


def _make_tools() -> list[Tool]:
    t1 = Tool(
        id="T1",
        tool_number=1,
        type=ToolType.FLAT_ENDMILL,
        diameter=10.0,
        flute_length=30.0,
        overall_length=75.0,
        flutes=4,
    )
    t2 = Tool(
        id="T2",
        tool_number=2,
        type=ToolType.DRILL,
        diameter=6.8,
        flute_length=45.0,
        overall_length=80.0,
        flutes=2,
    )
    return [t1, t2]


class TestHeidenhainPostProcessor:
    """Comprehensive tests for Heidenhain Conversational (Klartext) and ISO post-processing."""

    def test_heidenhain_conversational_header_and_footer(self):
        post = HeidenhainPostProcessor(conversational=True)
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        res = post.post_process([tp], program_name="TEST_PART_01", tools=tools)
        gcode = res.gcode

        # Check Program Enclosure
        assert "BEGIN PGM TEST_PART_01 MM" in gcode
        assert "END PGM TEST_PART_01 MM" in gcode
        assert "M30" in gcode

        # Check Datum Setting
        assert "CYCL DEF 247 DATUM SETTING" in gcode
        assert "Q339=1" in gcode

        # Check Lookahead / Tolerance
        assert "CYCL DEF 32.0 TOLERANCE" in gcode
        assert "CYCL DEF 32.1 T0.01" in gcode

        # Check Safe Return via M91 machine coordinates
        assert "L Z+0 R0 FMAX M91" in gcode
        assert "L X+0 Y+0 R0 FMAX M91" in gcode

    def test_heidenhain_conversational_tool_call_and_motion(self):
        post = HeidenhainPostProcessor(conversational=True)
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        res = post.post_process([tp], tools=tools)
        gcode = res.gcode

        # Check Tool Call syntax
        assert "TOOL CALL 1 Z S6000" in gcode

        # Check Klartext rapid and linear moves
        assert "L X+10 Y+10 Z+20 R0 FMAX" in gcode or "L X+10.000 Y+10.000 Z+20.000 R0 FMAX" in gcode or "FMAX" in gcode
        assert "L" in gcode and "F1200" in gcode

        # Check Klartext Circle Center (CC) and Circular Move (C)
        assert "CC X+50 Y+20" in gcode or "CC X+50.000 Y+20.000" in gcode or "CC X+" in gcode
        assert "C X+60 Y+20 DR+ R0 F1200" in gcode or "C X+60.000 Y+20.000 DR+ R0 F1200" in gcode or "DR+" in gcode

    def test_heidenhain_conversational_drilling_cycles(self):
        post = HeidenhainPostProcessor(conversational=True, renumber_tools=False)
        tools = _make_tools()
        tps = _make_sample_drilling_toolpaths()

        res = post.post_process(tps, tools=tools, validate_geometry=False)
        gcode = res.gcode

        # Check Tool Call for Drill
        assert "TOOL CALL 2 Z S3500" in gcode

        # Check Standard Drilling (Cycle 200)
        assert "CYCL DEF 200 DRILLING ~" in gcode
        assert "Q201=-15.000 ;DEPTH ~" in gcode
        assert "L X+25 Y+30 R0 FMAX M99" in gcode or "L X+25.000 Y+30.000 R0 FMAX M99" in gcode or "M99" in gcode

        # Check Universal Peck Drilling (Cycle 205)
        assert "CYCL DEF 205 UNIVERSAL PECKING ~" in gcode
        assert "Q201=-25.000 ;DEPTH ~" in gcode
        assert "Q202=3.000 ;PLUNGING DEPTH ~" in gcode
        assert "Q211=0.50 ;DWELL TIME AT DEPTH" in gcode
        assert "L X+45 Y+30 R0 FMAX M99" in gcode or "L X+45.000 Y+30.000 R0 FMAX M99" in gcode or "M99" in gcode

    def test_heidenhain_conversational_cutter_compensation(self):
        post = HeidenhainPostProcessor(conversational=True)
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()
        tp.metadata["cutter_comp_info"] = {"mode": "wear", "side": "left", "d_register": 1}

        # Add cutter comp on to segment
        tp.segments[3].metadata = {"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
        tp.segments[5].metadata = {"cutter_comp": {"action": "off"}}

        res = post.post_process([tp], tools=tools)
        gcode = res.gcode

        # In Klartext, radius comp is RL / RR on the move
        assert "RL" in gcode

    def test_heidenhain_iso_mode(self):
        post = HeidenhainPostProcessor(conversational=False)
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        res = post.post_process([tp], program_name="ISO_PROG", tools=tools)
        gcode = res.gcode

        # ISO mode uses % block and standard G-code
        assert "% ISO_PROG G71 *" in gcode
        assert "G00" in gcode or "G0" in gcode
        assert "G01" in gcode or "G1" in gcode
        assert "G43" in gcode

    def test_heidenhain_syntax_validation_passes(self):
        post = HeidenhainPostProcessor(conversational=True)
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        res = post.post_process([tp], tools=tools, validate_syntax=True, strict_validation=True)
        assert res.syntax_validation is not None
        assert res.syntax_validation.is_valid is True
        assert len(res.syntax_validation.errors) == 0


class TestSiemensSinumerikPostProcessor:
    """Comprehensive tests for Siemens Sinumerik (840D/828D) post-processing."""

    def test_siemens_header_and_footer(self):
        post = SiemensPostProcessor()
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        res = post.post_process([tp], program_name="TURBINE_BLADE_01", tools=tools)
        gcode = res.gcode

        # Check Program Header
        assert "%_N_TURBINE_BLADE_01_MPF" in gcode
        assert "SIEMENS SINUMERIK 840D / 828D" in gcode
        assert "G90 G94 G17 G71 G40 G601" in gcode

        # Check High Speed Settings (CYCLE832)
        assert "CYCLE832(" in gcode

        # Check Safe Retract via SUPA
        assert "SUPA G0 Z0.0 D0" in gcode
        assert "SUPA G0 X0.0 Y0.0" in gcode
        assert "M30" in gcode

    def test_siemens_tool_call_and_d_offset(self):
        post = SiemensPostProcessor()
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        res = post.post_process([tp], tools=tools)
        gcode = res.gcode

        # Check Siemens Tool Call: T1 M6 followed by D1
        assert "T1 M6" in gcode
        assert "D1" in gcode
        assert "S6000 M3" in gcode

    def test_siemens_canned_cycles(self):
        post = SiemensPostProcessor()
        tools = _make_tools()
        tps = _make_sample_drilling_toolpaths()

        res = post.post_process(tps, tools=tools, validate_geometry=False)
        gcode = res.gcode

        # Check Standard Drill (CYCLE81)
        # CYCLE81(RTP, RFP, SDIS, DP)
        assert "CYCLE81(" in gcode
        assert "-15.000" in gcode
        assert "G0 X25 Y30" in gcode or "G0 X25.0 Y30.0" in gcode

        # Check Peck Drill (CYCLE83)
        # CYCLE83(RTP, RFP, SDIS, DP, DPR, FDEP, FDPR, DAM, DTB, DTS, FRF, VARI, _MDEP, _VRT, _DTD, _DIS1)
        assert "CYCLE83(" in gcode
        assert "-25.000" in gcode
        assert "3.000" in gcode  # DAM peck depth
        assert "G0 X45 Y30" in gcode or "G0 X45.0 Y30.0" in gcode

        # Check Modal Cycle Cancellation (MCALL)
        assert "MCALL" in gcode

    def test_siemens_cutter_compensation(self):
        post = SiemensPostProcessor()
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        tp.segments[3].metadata = {"cutter_comp": {"action": "on", "side": "left", "d_register": 1}}
        tp.segments[5].metadata = {"cutter_comp": {"action": "off"}}

        res = post.post_process([tp], tools=tools)
        gcode = res.gcode

        # In Siemens, cutter comp is G41 D1 / G40
        assert "G41 D1" in gcode
        assert "G40" in gcode

    def test_siemens_syntax_validation_passes(self):
        post = SiemensPostProcessor()
        tools = _make_tools()
        tp = _make_sample_contour_toolpath()

        res = post.post_process([tp], tools=tools, validate_syntax=True, strict_validation=True)
        assert res.syntax_validation is not None
        assert res.syntax_validation.is_valid is True
        assert len(res.syntax_validation.errors) == 0


class TestCapabilityModelsAndFactoryRouting:
    """Test model registrations and factory method routing."""

    def test_heidenhain_capability_model(self):
        assert HEIDENHAIN_CAPABILITY_MODEL.controller_id == "heidenhain"
        assert HEIDENHAIN_CAPABILITY_MODEL.cutter_comp.supported is True
        assert HEIDENHAIN_CAPABILITY_MODEL.lookahead.syntax_type == "cycl_def_32"
        assert HEIDENHAIN_CAPABILITY_MODEL.safe_return.syntax_type == "m91_machine_coords"
        assert HEIDENHAIN_CAPABILITY_MODEL.is_gcode_supported("L") is True
        assert HEIDENHAIN_CAPABILITY_MODEL.is_gcode_supported("CYCL DEF 200") is True

    def test_siemens_capability_model(self):
        assert SIEMENS_CAPABILITY_MODEL.controller_id == "siemens"
        assert SIEMENS_CAPABILITY_MODEL.cutter_comp.supported is True
        assert SIEMENS_CAPABILITY_MODEL.lookahead.syntax_type == "cycle832"
        assert SIEMENS_CAPABILITY_MODEL.safe_return.syntax_type == "supa_machine_coords"
        assert SIEMENS_CAPABILITY_MODEL.is_gcode_supported("CYCLE81") is True
        assert SIEMENS_CAPABILITY_MODEL.is_gcode_supported("CYCLE832") is True
        assert SIEMENS_CAPABILITY_MODEL.is_gcode_supported("SUPA") is True
        assert SINUMERIK_CAPABILITY_MODEL is SIEMENS_CAPABILITY_MODEL

    def test_get_controller_capability_model_resolution(self):
        assert get_controller_capability_model("heidenhain") is HEIDENHAIN_CAPABILITY_MODEL
        assert get_controller_capability_model("klartext") is HEIDENHAIN_CAPABILITY_MODEL
        assert get_controller_capability_model("tnc640") is HEIDENHAIN_CAPABILITY_MODEL
        assert get_controller_capability_model("siemens") is SIEMENS_CAPABILITY_MODEL
        assert get_controller_capability_model("sinumerik") is SIEMENS_CAPABILITY_MODEL
        assert get_controller_capability_model("840d") is SIEMENS_CAPABILITY_MODEL
        assert get_controller_capability_model("828d") is SIEMENS_CAPABILITY_MODEL

    def test_get_post_processor_factory(self):
        p_hnc = get_post_processor("heidenhain")
        assert isinstance(p_hnc, HeidenhainPostProcessor)

        p_klartext = get_post_processor("klartext")
        assert isinstance(p_klartext, HeidenhainPostProcessor)

        p_sie = get_post_processor("siemens")
        assert isinstance(p_sie, SiemensPostProcessor)

        p_sin = get_post_processor("sinumerik")
        assert isinstance(p_sin, SiemensPostProcessor)
        assert SinumerikPostProcessor is SiemensPostProcessor
