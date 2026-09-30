"""
Test suite for Controller Capability Model and Syntax Validation before export.
Validates:
1. Modal codes and whitelists per controller
2. Canned cycles validation (G81..G89, G73, G76, G98/G99)
3. Cutter compensation semantics (G41/G42/G40, D register limits)
4. Tool-length compensation syntax (G43/G49, H register limits)
5. Lookahead syntax (G05.1 Q1/Q0 Fanuc, G187 Haas, G64 LinuxCNC, none GRBL)
6. Safe return syntax (G91 G28 Fanuc, G53 Haas, etc.)
7. Feed rate (Fmax) and spindle speed (Smax) limits
8. Tape enclosure (%) and program numbering (O#### vs O#####)
"""

import numpy as np
import pytest
from cam_engine.context import Tool, ToolType
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType, CoordSpace
from cam_engine.post.controller_models import (
    FANUC_CAPABILITY_MODEL,
    HAAS_CAPABILITY_MODEL,
    GRBL_CAPABILITY_MODEL,
    LINUXCNC_CAPABILITY_MODEL,
    ControllerCapabilityModel,
)
from cam_engine.post.syntax_validator import (
    ControllerSyntaxValidator,
    ControllerSyntaxValidationReport,
)
from cam_engine.post.post_processor import (
    FanucPostProcessor,
    HaasPostProcessor,
    GrblPostProcessor,
    LinuxCncPostProcessor,
    get_post_processor,
)


def _make_dummy_toolpath() -> Toolpath:
    tp = Toolpath(operation_id="op_dummy", feature_id="f1", tool_id="T1", purpose="finishing", space=CoordSpace.SETUP)
    tp.segments = [
        MotionSegment(motion_type=MotionType.RAPID, start=np.array([0.0, 0.0, 5.0]), end=np.array([0.0, 0.0, 5.0]), feed=1500.0),
        MotionSegment(motion_type=MotionType.CUT, start=np.array([0.0, 0.0, 5.0]), end=np.array([10.0, 10.0, 0.0]), feed=500.0),
        MotionSegment(motion_type=MotionType.CUT, start=np.array([10.0, 10.0, 0.0]), end=np.array([20.0, 10.0, 0.0]), feed=500.0),
        MotionSegment(motion_type=MotionType.RAPID, start=np.array([20.0, 10.0, 0.0]), end=np.array([20.0, 10.0, 5.0]), feed=1500.0),
    ]
    return tp


class TestControllerCapabilityPresets:
    def test_fanuc_preset_defaults(self):
        assert "Fanuc" in FANUC_CAPABILITY_MODEL.name
        assert FANUC_CAPABILITY_MODEL.lookahead.supported is True
        assert FANUC_CAPABILITY_MODEL.lookahead.on_pattern == "G05.1 Q1"
        assert FANUC_CAPABILITY_MODEL.safe_return.z_retract_command == "G91 G28 Z0.0"
        assert FANUC_CAPABILITY_MODEL.block_format.requires_percent_enclosure is True
        assert FANUC_CAPABILITY_MODEL.program_numbering.digits == 4
        assert "G83" in FANUC_CAPABILITY_MODEL.supported_canned_cycles

    def test_haas_preset_defaults(self):
        assert "Haas" in HAAS_CAPABILITY_MODEL.name
        assert HAAS_CAPABILITY_MODEL.lookahead.on_pattern.startswith("G187")
        assert HAAS_CAPABILITY_MODEL.safe_return.z_retract_command.startswith("G53")
        assert HAAS_CAPABILITY_MODEL.program_numbering.digits == 5

    def test_grbl_preset_defaults(self):
        assert "GRBL" in GRBL_CAPABILITY_MODEL.name
        assert GRBL_CAPABILITY_MODEL.lookahead.supported is False
        assert len(GRBL_CAPABILITY_MODEL.supported_canned_cycles) == 0
        assert GRBL_CAPABILITY_MODEL.supports_g98_g99 is False
        assert GRBL_CAPABILITY_MODEL.block_format.requires_percent_enclosure is False
        assert GRBL_CAPABILITY_MODEL.program_numbering.supported is False


class TestPostProcessorSyntaxValidationIntegration:
    def test_fanuc_post_produces_valid_syntax(self):
        tool = Tool(
            id="T1",
            tool_number=1,
            type=ToolType.FLAT_ENDMILL,
            diameter=6.0,
            flute_length=20.0,
            overall_length=50.0,
            flutes=4,
            corner_radius=0.0,
        )
        post = FanucPostProcessor()
        tp = _make_dummy_toolpath()
        result = post.post_process([tp], [tool])

        assert result.syntax_validation is not None
        assert result.syntax_validation.is_valid is True, f"Errors: {result.syntax_validation.errors}"
        assert len(result.syntax_validation.errors) == 0

    def test_haas_post_produces_valid_syntax(self):
        tool = Tool(
            id="T2",
            tool_number=2,
            type=ToolType.FLAT_ENDMILL,
            diameter=10.0,
            flute_length=30.0,
            overall_length=60.0,
            flutes=4,
            corner_radius=0.0,
        )
        post = HaasPostProcessor()
        tp = _make_dummy_toolpath()
        result = post.post_process([tp], [tool])

        assert result.syntax_validation is not None
        assert result.syntax_validation.is_valid is True, f"Errors: {result.syntax_validation.errors}"
        assert len(result.syntax_validation.errors) == 0

    def test_grbl_post_produces_valid_syntax(self):
        tool = Tool(
            id="T1",
            tool_number=1,
            type=ToolType.FLAT_ENDMILL,
            diameter=3.175,
            flute_length=15.0,
            overall_length=40.0,
            flutes=2,
            corner_radius=0.0,
        )
        post = GrblPostProcessor()
        tp = _make_dummy_toolpath()
        result = post.post_process([tp], [tool])

        assert result.syntax_validation is not None
        assert result.syntax_validation.is_valid is True, f"Errors: {result.syntax_validation.errors}"
        assert len(result.syntax_validation.errors) == 0

    def test_linuxcnc_post_produces_valid_syntax(self):
        tool = Tool(
            id="T1",
            tool_number=1,
            type=ToolType.FLAT_ENDMILL,
            diameter=6.0,
            flute_length=20.0,
            overall_length=50.0,
            flutes=3,
            corner_radius=0.0,
        )
        post = LinuxCncPostProcessor()
        tp = _make_dummy_toolpath()
        result = post.post_process([tp], [tool])

        assert result.syntax_validation is not None
        assert result.syntax_validation.is_valid is True, f"Errors: {result.syntax_validation.errors}"
        assert len(result.syntax_validation.errors) == 0


class TestSyntaxValidatorErrorDetection:
    def test_grbl_rejects_canned_cycle(self):
        gcode = [
            "G21 G90",
            "G0 Z5.0",
            "G98 G83 X10.0 Y10.0 Z-5.0 R2.0 Q1.0 F100.0",
            "G80",
            "M30",
        ]
        validator = ControllerSyntaxValidator(GRBL_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "G83" in error_msgs or "canned cycle" in error_msgs.lower()

    def test_grbl_rejects_lookahead_code(self):
        gcode = [
            "G21 G90",
            "G05.1 Q1",
            "G0 X0 Y0",
            "G05.1 Q0",
            "M30",
        ]
        validator = ControllerSyntaxValidator(GRBL_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "lookahead" in error_msgs.lower() or "G05.1" in error_msgs

    def test_grbl_rejects_percent_delimiters(self):
        gcode = [
            "%",
            "G21 G90",
            "G0 X0 Y0",
            "M30",
            "%",
        ]
        validator = ControllerSyntaxValidator(GRBL_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "%" in error_msgs or "tape enclosure" in error_msgs.lower()

    def test_fanuc_rejects_missing_percent_enclosure(self):
        gcode = [
            "O0001 (PROGRAM)",
            "G21 G90 G40 G80",
            "G0 X0 Y0",
            "M30",
        ]
        validator = ControllerSyntaxValidator(FANUC_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "%" in error_msgs or "tape enclosure" in error_msgs.lower()

    def test_fanuc_rejects_haas_smoothing_g187(self):
        gcode = [
            "%",
            "O0001 (TEST)",
            "G187 P1 E0.01",
            "G0 X0 Y0",
            "M30",
            "%",
        ]
        validator = ControllerSyntaxValidator(FANUC_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "G187" in error_msgs or "unsupported" in error_msgs.lower()

    def test_fanuc_rejects_5digit_program_number_if_restricted(self):
        gcode = [
            "%",
            "O12345 (TEST)",
            "G0 X0 Y0",
            "M30",
            "%",
        ]
        validator = ControllerSyntaxValidator(FANUC_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "O12345" in error_msgs or "exceeds maximum digits" in error_msgs

    def test_rejects_feed_exceeding_fmax(self):
        gcode = [
            "%",
            "O0001 (TEST)",
            "G1 X10.0 Y10.0 F60000.0",
            "M30",
            "%",
        ]
        validator = ControllerSyntaxValidator(FANUC_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "F60000.0" in error_msgs or "exceeds maximum allowed feed" in error_msgs

    def test_rejects_spindle_exceeding_smax(self):
        gcode = [
            "%",
            "O0001 (TEST)",
            "S50000 M03",
            "M30",
            "%",
        ]
        validator = ControllerSyntaxValidator(FANUC_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "S50000" in error_msgs or "exceeds maximum allowed spindle speed" in error_msgs

    def test_rejects_d_register_out_of_bounds(self):
        gcode = [
            "%",
            "O0001 (TEST)",
            "G41 D999 X10.0 Y10.0",
            "M30",
            "%",
        ]
        validator = ControllerSyntaxValidator(FANUC_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "D999" in error_msgs or "register" in error_msgs

    def test_canned_cycle_missing_required_words(self):
        gcode = [
            "%",
            "O0001 (TEST)",
            "G83 X10.0 Y10.0",  # Missing Z, R, Q
            "G80",
            "M30",
            "%",
        ]
        validator = ControllerSyntaxValidator(FANUC_CAPABILITY_MODEL)
        report = validator.validate(gcode)

        assert report.is_valid is False
        error_msgs = " ".join(report.errors)
        assert "missing required" in error_msgs.lower()
