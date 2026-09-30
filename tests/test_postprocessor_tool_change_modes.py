"""Tests for Post-Processor-Aware Tool Change Architecture (ATC M6, manual M0, custom formatting, and next-tool staging).

Verifies:
1. Fanuc, Haas, and LinuxCNC default to ATC (Txx M6).
2. GRBL defaults to manual (Txx M0), but can be configured to ATC (Txx M6).
3. Custom tool change formatting templates (e.g. 'M6 T{tool_num}', 'T{tool_num_2d}{h_offset_2d}').
4. Next-tool staging / pre-calling (T{next_tool}) during multi-tool programs.
"""

import numpy as np
import pytest

from cam_engine.context import MachineConfig, MachineAxis, MachineType, Units
from cam_engine.post import (
    get_post_processor, FanucPostProcessor, HaasPostProcessor,
    LinuxCncPostProcessor, GrblPostProcessor, ToolChangeStrategy,
    ToolChangeCapability, ControllerCapabilityModel
)
from cam_engine.toolpaths.semantic import (
    MotionSegment, MotionType, Toolpath, CoordSpace
)


def make_test_toolpath(op_id: str, tool_id: str, tool_num: int = 1) -> Toolpath:
    tp = Toolpath(operation_id=op_id, feature_id="f1", tool_id=tool_id, purpose="facing", space=CoordSpace.SETUP)
    tool_info = {
        "id": tool_id,
        "tool_number": tool_num,
        "type": "flat_endmill",
        "diameter": 10.0,
        "flute_length": 25.0,
        "flutes": 4,
    }
    tp.metadata["tool_info"] = tool_info

    tp.add(MotionSegment(
        motion_type=MotionType.TOOL_CHANGE,
        start=np.array([0.0, 0.0, 20.0]),
        end=np.array([0.0, 0.0, 20.0]),
        tool_id=tool_id,
        operation_id=op_id,
        feature_id="f1",
        metadata={"tool_number": tool_num, "tool_info": tool_info}
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([20.0, 0.0, 0.0]),
        feed=1000.0,
        tool_id=tool_id,
        operation_id=op_id,
        feature_id="f1",
    ))
    return tp


def test_default_tool_change_modes_per_controller():
    """Verify default ATC (M6) on Fanuc/Haas/LinuxCNC and manual (M0) on GRBL."""
    tp1 = make_test_toolpath("op1", "T01", 1)
    tp2 = make_test_toolpath("op2", "T02", 2)

    # 1. Fanuc -> ATC M6
    fanuc_post = FanucPostProcessor(work_offset="G54")
    res_fanuc = fanuc_post.post_process([tp1, tp2], "FANUC_ATC")
    assert "T1 M6" in res_fanuc.gcode
    assert "T2 M6" in res_fanuc.gcode
    assert "M0" not in res_fanuc.gcode

    # 2. Haas -> ATC M6
    haas_post = HaasPostProcessor(work_offset="G54")
    res_haas = haas_post.post_process([tp1, tp2], "HAAS_ATC")
    assert "T1 M6" in res_haas.gcode
    assert "T2 M6" in res_haas.gcode

    # 3. LinuxCNC -> ATC M6
    linux_post = LinuxCncPostProcessor(work_offset="G54")
    res_linux = linux_post.post_process([tp1, tp2], "LINUX_ATC")
    assert "T1 M6" in res_linux.gcode
    assert "T2 M6" in res_linux.gcode

    # 4. GRBL -> Manual M0
    grbl_post = GrblPostProcessor(work_offset="G54")
    res_grbl = grbl_post.post_process([tp1, tp2], "GRBL_MANUAL")
    assert "T1 M0 (Change to T01" in res_grbl.gcode
    assert "T2 M0 (Change to T02" in res_grbl.gcode


def test_grbl_configured_for_atc():
    """Verify GRBL post configured with tool_change_mode='atc' emits M6."""
    tp = make_test_toolpath("op1", "T01", 1)
    post = GrblPostProcessor(work_offset="G54", tool_change_mode="atc")
    res = post.post_process([tp], "GRBL_ATC")
    assert "T1 M6" in res.gcode
    assert "M0" not in res.gcode


def test_fanuc_configured_for_manual():
    """Verify Fanuc post configured with tool_change_mode='manual' emits M0."""
    tp = make_test_toolpath("op1", "T01", 1)
    post = FanucPostProcessor(work_offset="G54", tool_change_mode="manual")
    res = post.post_process([tp], "FANUC_MANUAL")
    assert "T1 M0 (Change to T01" in res.gcode


def test_custom_tool_change_format_template():
    """Verify custom tool change formatting strings (e.g. lathe turrets or alternate M6 positions)."""
    tp = make_test_toolpath("op1", "T03", 3)

    # Prefix M6 format: 'M6 T{tool_num}'
    post_prefix = get_post_processor("fanuc", work_offset="G54", tool_change_format="M6 T{tool_num}")
    res_prefix = post_prefix.post_process([tp], "PREFIX_M6")
    assert "M6 T1" in res_prefix.gcode

    # Turret index 4-digit format: 'T{tool_num_2d}{h_offset_2d}'
    post_turret = get_post_processor("fanuc", work_offset="G54", tool_change_format="T{tool_num_2d}{h_offset_2d}")
    res_turret = post_turret.post_process([tp], "TURRET_INDEX")
    assert "T0101" in res_turret.gcode


def test_tool_staging_next_tool():
    """Verify stage_next_tool=True pre-calls the next distinct tool in carousel."""
    tp1 = make_test_toolpath("op1", "T01", 1)
    tp2 = make_test_toolpath("op2", "T02", 2)
    tp3 = make_test_toolpath("op3", "T03", 3)

    post = FanucPostProcessor(work_offset="G54", stage_next_tool=True)
    res = post.post_process([tp1, tp2, tp3], "STAGING_TEST")
    gcode = res.gcode

    # After T1, T2 should be staged
    assert "T1 M6" in gcode
    assert "T2 (Stage next tool in carousel)" in gcode

    # After T2, T3 should be staged
    assert "T2 M6" in gcode
    assert "T3 (Stage next tool in carousel)" in gcode

    # After T3 (last tool), no staging needed
    assert "T3 M6" in gcode


def test_machine_profile_derived_tool_change_strategy():
    """Verify toolChangeStrategy is derived from machine profile rather than hardcoded."""
    tp = make_test_toolpath("op1", "T01", 1)

    # 1. Machine with has_atc=True (default ATC M6)
    atc_machine = MachineConfig(
        id="M_ATC",
        name="VMC with ATC",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="haas",
        spindle_min_rpm=100.0,
        spindle_max_rpm=15000.0,
        max_spindle_power_kw=15.0,
        axes={"X": MachineAxis("X", -500, 500, 10000, 5000), "Y": MachineAxis("Y", -300, 300, 10000, 5000), "Z": MachineAxis("Z", -200, 300, 10000, 5000)},
        tool_change_position=np.array([0.0, 0.0, 100.0]),
        safe_retract_height=10.0,
        work_offsets={"G54": np.zeros(3)},
        has_atc=True,
        tool_change_strategy="atc",
        tool_change_command="M6",
    )
    post_atc = get_post_processor(machine=atc_machine)
    res_atc = post_atc.post_process([tp], "ATC_PROG")
    assert "T1 M6" in res_atc.gcode
    assert "M0" not in res_atc.gcode

    # 2. Machine with has_atc=False (Manual tool change with M0)
    manual_machine = MachineConfig(
        id="M_MANUAL",
        name="Manual Mill",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="fanuc",
        spindle_min_rpm=100.0,
        spindle_max_rpm=6000.0,
        max_spindle_power_kw=5.0,
        axes={"X": MachineAxis("X", -500, 500, 10000, 5000), "Y": MachineAxis("Y", -300, 300, 10000, 5000), "Z": MachineAxis("Z", -200, 300, 10000, 5000)},
        tool_change_position=np.array([0.0, 0.0, 100.0]),
        safe_retract_height=10.0,
        work_offsets={"G54": np.zeros(3)},
        has_atc=False,
        tool_change_strategy="manual",
        tool_change_command="M0",
    )
    post_manual = get_post_processor(machine=manual_machine)
    res_manual = post_manual.post_process([tp], "MANUAL_PROG")
    assert "T1 M0" in res_manual.gcode
    assert "M6" not in res_manual.gcode

    # 3. Machine with custom M00 stop or custom format template
    custom_machine = MachineConfig(
        id="M_CUSTOM",
        name="Custom Mill",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        controller="haas",
        spindle_min_rpm=100.0,
        spindle_max_rpm=12000.0,
        max_spindle_power_kw=10.0,
        axes={"X": MachineAxis("X", -500, 500, 10000, 5000), "Y": MachineAxis("Y", -300, 300, 10000, 5000), "Z": MachineAxis("Z", -200, 300, 10000, 5000)},
        tool_change_position=np.array([0.0, 0.0, 100.0]),
        safe_retract_height=10.0,
        work_offsets={"G54": np.zeros(3)},
        has_atc=True,
        tool_change_format="M06 T{tool_num} (LOAD TOOL {desc})",
    )
    post_custom = get_post_processor(machine=custom_machine)
    res_custom = post_custom.post_process([tp], "CUSTOM_PROG")
    assert "M06 T1 (LOAD TOOL" in res_custom.gcode

