"""Tests for unique CNC program number derivation, collision avoidance, and controller formatting.

Verifies:
- Setups on the same job get distinct program numbers (e.g., setup_001 vs setup_002).
- Different jobs/parts receive distinct program number blocks (not fixed 1001/1002 for every job).
- Explicit program number overrides are respected.
- Base program number configurations work seamlessly.
- Collision registry prevents collisions across batch exports.
- Controller-specific formatting adheres to Fanuc (4-digit O####) and Haas (5-digit O#####).
"""

import re
import numpy as np
import pytest

from cam_engine.post.post_processor import (
    resolve_program_number,
    get_post_processor,
    FanucPostProcessor,
    HaasPostProcessor,
    GrblPostProcessor,
    LinuxCncPostProcessor,
)
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType


def _make_dummy_toolpath(setup_id: str = "setup_001") -> Toolpath:
    tp = Toolpath(
        operation_id=f"op_{setup_id}",
        feature_id="f1",
        tool_id="T01",
        purpose="roughing",
        metadata={"setup_id": setup_id},
    )
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([10.0, 0.0, 0.0]),
        feed=500.0,
        spindle=6000.0,
        tool_id="T01",
    ))
    return tp


class TestProgramNumberUniqueness:
    """Test suite for unique CNC program number generation."""

    def test_setup_1_and_setup_2_get_distinct_numbers(self):
        """Setup 1 and Setup 2 of the same job must never collide on O1001."""
        post_s1 = get_post_processor("fanuc", work_offset="G54")
        post_s2 = get_post_processor("fanuc", work_offset="G55")

        tp1 = _make_dummy_toolpath("setup_001")
        tp2 = _make_dummy_toolpath("setup_002")

        res1 = post_s1.post_process([tp1], program_name="BRACKET_FRONT_setup_001")
        res2 = post_s2.post_process([tp2], program_name="BRACKET_FRONT_setup_002")

        assert res1.program_number != res2.program_number
        assert f"O{res1.program_number:04d}" in res1.gcode
        assert f"O{res2.program_number:04d}" in res2.gcode

        # Sequential setup numbers within same job
        assert res2.program_number == res1.program_number + 1

    def test_different_jobs_generate_distinct_number_blocks(self):
        """Two different jobs must not both map to fixed O1001/O1002."""
        tp1 = _make_dummy_toolpath("setup_001")

        post_job_a = get_post_processor("fanuc")
        post_job_b = get_post_processor("fanuc")

        res_a = post_job_a.post_process([tp1], program_name="FLANGE_HOUSING", job_id="job_flange_01")
        res_b = post_job_b.post_process([tp1], program_name="TURBINE_IMPELLER", job_id="job_turbine_02")

        assert res_a.program_number != res_b.program_number
        # Verify both are within standard Fanuc user program range (1000..8999)
        assert 1000 <= res_a.program_number < 9000
        assert 1000 <= res_b.program_number < 9000

    def test_explicit_program_number_override(self):
        """Explicitly passed program numbers take top precedence."""
        post = get_post_processor("fanuc")
        tp = _make_dummy_toolpath("setup_001")

        # Integer override
        res1 = post.post_process([tp], program_name="CUSTOM_PROG", program_number=4520)
        assert res1.program_number == 4520
        assert "O4520" in res1.gcode

        # String override with 'O' prefix
        res2 = post.post_process([tp], program_name="CUSTOM_PROG", program_number="O7890")
        assert res2.program_number == 7890
        assert "O7890" in res2.gcode

    def test_base_program_number_configuration(self):
        """Configured base_program_number increments cleanly per setup."""
        post1 = get_post_processor("haas", base_program_number=5000)
        post2 = get_post_processor("haas", base_program_number=5000)

        tp1 = _make_dummy_toolpath("setup_001")
        tp2 = _make_dummy_toolpath("setup_002")

        res1 = post1.post_process([tp1], setup_id="setup_001")
        res2 = post2.post_process([tp2], setup_id="setup_002")

        assert res1.program_number == 5001
        assert res2.program_number == 5002
        assert "O05001" in res1.gcode
        assert "O05002" in res2.gcode

    def test_batch_collision_registry(self):
        """Collision registry dynamically increments duplicate candidates."""
        registry: set[int] = set()

        num1 = resolve_program_number(program_name="PART_A", explicit_number=2000, used_numbers=registry)
        num2 = resolve_program_number(program_name="PART_B", explicit_number=2000, used_numbers=registry)
        num3 = resolve_program_number(program_name="PART_C", explicit_number=2000, used_numbers=registry)

        assert num1 == 2000
        assert num2 == 2001
        assert num3 == 2002
        assert len(registry) == 3

    def test_controller_formatting_fanuc_vs_haas(self):
        """Fanuc outputs 4-digit O#### and Haas outputs 5-digit O#####."""
        tp = _make_dummy_toolpath("setup_001")

        fanuc_post = FanucPostProcessor(program_number=123)
        haas_post = HaasPostProcessor(program_number=123)
        grbl_post = GrblPostProcessor(program_number=123)
        linux_post = LinuxCncPostProcessor(program_number=123)

        res_fanuc = fanuc_post.post_process([tp], program_name="TEST_FMT")
        res_haas = haas_post.post_process([tp], program_name="TEST_FMT")
        res_grbl = grbl_post.post_process([tp], program_name="TEST_FMT")
        res_linux = linux_post.post_process([tp], program_name="TEST_FMT")

        assert "O0123" in res_fanuc.gcode
        assert "O00123" in res_haas.gcode
        assert "(O0123)" in res_grbl.gcode
        assert "O0123" in res_linux.gcode

    def test_macro_range_safety(self):
        """Derived program numbers avoid reserved 9000-9999 macro program space."""
        for name in ["MACRO_TEST", "JOB_999", "COMPLEX_CAM_MODEL", "ZZZZ_PART"]:
            num = resolve_program_number(program_name=name)
            assert 1000 <= num < 9000
