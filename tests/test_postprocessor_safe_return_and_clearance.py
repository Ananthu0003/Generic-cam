"""Tests for post-processor dynamic safe return, setup clearance, and machine homing derivation."""

import pytest
import numpy as np

from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType, CoordSpace
from cam_engine.post.post_processor import (
    get_post_processor,
    GrblPostProcessor,
    FanucPostProcessor,
    HaasPostProcessor,
    LinuxCncPostProcessor,
)
from cam_engine.context import MachineConfig, MachineAxis, MachineType


def create_sample_toolpath(setup_clearance: float = 15.02, initial_z: float = 15.02) -> Toolpath:
    """Helper to create a toolpath with specific setup clearance metadata."""
    tp = Toolpath(
        operation_id="op_drill",
        feature_id="hole_1",
        tool_id="T01",
        purpose="drilling",
        space=CoordSpace.SETUP,
    )
    tp.metadata = {
        "setup_id": "setup_001",
        "setup_clearance_z": setup_clearance,
        "work_offset": "G54",
    }
    tp.add(
        MotionSegment(
            motion_type=MotionType.RAPID,
            start=np.array([0.0, 0.0, 50.0]),
            end=np.array([10.0, 20.0, setup_clearance]),
            tool_id="T01",
            operation_id="op_drill",
            feature_id="hole_1",
        )
    )
    tp.add(
        MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([10.0, 20.0, setup_clearance]),
            end=np.array([10.0, 20.0, 0.0]),
            feed=200.0,
            tool_id="T01",
            operation_id="op_drill",
            feature_id="hole_1",
            metadata={
                "canned_cycle": {
                    "type": "G83",
                    "x": 10.0,
                    "y": 20.0,
                    "z_r": 5.0,
                    "z_depth": -10.0,
                    "peck": 2.0,
                    "initial_z": initial_z,
                }
            },
        )
    )
    return tp


def make_dummy_machine(controller: str = "grbl", **kwargs) -> MachineConfig:
    axes = {
        "X": MachineAxis("X", -500, 500, 10000, 5000),
        "Y": MachineAxis("Y", -300, 300, 10000, 5000),
        "Z": MachineAxis("Z", -200, 300, 10000, 5000),
    }
    return MachineConfig(
        id="M_TEST",
        name="Test Machine",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller=controller,
        spindle_min_rpm=100.0,
        spindle_max_rpm=15000.0,
        max_spindle_power_kw=15.0,
        axes=axes,
        tool_change_position=np.array([0.0, 0.0, 100.0]),
        safe_retract_height=10.0,
        work_offsets={"G54": np.zeros(3)},
        **kwargs,
    )


def test_grbl_dynamic_safe_retract_from_setup_clearance():
    """Verify GRBL derives final retract Z from setup clearance metadata."""
    tp = create_sample_toolpath(setup_clearance=35.5)
    post = get_post_processor("grbl")
    result = post.post_process([tp])
    lines = result.gcode.splitlines()

    # Should contain G0 Z35.5 instead of hardcoded Z20.0
    assert "G0 Z35.5" in lines
    assert "G0 X0 Y0" in lines
    assert "M2" in lines


def test_grbl_custom_machine_retract_and_homing():
    """Verify GRBL respects custom machine retract Z and home command overrides."""
    machine = make_dummy_machine(
        controller="grbl",
        machine_retract_z=45.0,
        home_command_z="G53 G0 Z0 (Safe Z)",
        home_command_xy="G53 G0 X0 Y0 (Park Table)",
    )
    post = get_post_processor("grbl", machine=machine)
    tp = create_sample_toolpath(setup_clearance=15.0)
    result = post.post_process([tp])
    lines = result.gcode.splitlines()

    assert "G53 G0 Z0 (Safe Z)" in lines
    assert "G53 G0 X0 Y0 (Park Table)" in lines
    assert "M2" in lines


def test_fanuc_safe_return_sequence():
    """Verify Fanuc post uses standard G91 G28 Z0.0 / G28 X0.0 Y0.0 / M30 / %."""
    tp = create_sample_toolpath()
    post = get_post_processor("fanuc")
    result = post.post_process([tp])
    lines = result.gcode.splitlines()

    assert "G49 G40" in lines
    assert "G91 G28 Z0.0" in lines
    assert "G28 X0.0 Y0.0" in lines
    assert "M30" in lines
    assert "%" in lines


def test_fanuc_custom_home_commands():
    """Verify Fanuc post respects machine profile custom homing commands."""
    machine = make_dummy_machine(
        controller="fanuc",
        home_command_z="G91 G30 Z0.0",
        home_command_xy="G30 X0.0 Y0.0",
    )
    post = get_post_processor("fanuc", machine=machine)
    tp = create_sample_toolpath()
    result = post.post_process([tp])
    lines = result.gcode.splitlines()

    assert "G91 G30 Z0.0" in lines
    assert "G30 X0.0 Y0.0" in lines
    assert "M30" in lines


def test_haas_safe_return_sequence():
    """Verify Haas post uses G53 G00 Z0.0 / G53 G00 X0.0 Y0.0 / M30."""
    tp = create_sample_toolpath()
    post = get_post_processor("haas")
    result = post.post_process([tp])
    lines = result.gcode.splitlines()

    assert "G49 G40" in lines
    assert "G53 G00 Z0.0" in lines
    assert "G53 G00 X0.0 Y0.0" in lines
    assert "M30" in lines


def test_linuxcnc_safe_return_from_fixture_and_setup():
    """Verify LinuxCNC uses highest safe clearance and M2 end command."""
    post = get_post_processor(
        "linuxcnc",
        fixture_clearance_z=55.0,
        setup_clearance_z=30.0,
    )
    tp = create_sample_toolpath(setup_clearance=30.0)
    result = post.post_process([tp])
    lines = result.gcode.splitlines()

    assert "G0 Z55" in lines or "G0 Z55.0" in lines or "G0 Z55.000" in lines
    assert "M2" in lines
