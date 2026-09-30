"""Tests for physical spindle speed calculation, machine clamping, and effective cutting speed (Vc) traceability.

Verifies:
1. Standard spindle speed relationship: RPM_calc = 1000 * Vc_target / (pi * D)
2. Clamping to machine limits: RPM_applied = clamp(RPM_calc, RPM_min, RPM_max)
3. Effective cutting speed: Vc_effective = (pi * D * RPM_applied) / 1000
4. For micro-drills (e.g. Ø0.58mm and Ø0.51mm at Vc=120 m/min):
   - Ø0.58mm: RPM_calc ≈ 65,858 RPM -> Clamped to 15,000 RPM -> Vc_effective ≈ 27.3 m/min
   - Ø0.51mm: RPM_calc ≈ 74,898 RPM -> Clamped to 15,000 RPM -> Vc_effective ≈ 24.1 m/min
5. Unclamped cutters (e.g. Ø10.0mm at Vc=120 m/min):
   - Ø10.0mm: RPM_calc ≈ 3,820 RPM -> Applied 3,820 RPM -> Vc_effective = 120.0 m/min
6. G-code comments explicitly communicate Target Vc, Calculated RPM, Machine Limit, Applied RPM, and Effective Vc.
"""

import math
import numpy as np
import pytest

from cam_engine.context import (
    Tool, ToolType, PlanningContext, Setup, Stock, StockKind,
    Material, WorkOffset, DEFAULT_MATERIALS, MachineConfig
)
from cam_engine.demo_models import get_default_machine
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.geometry import TriangleMesh
from cam_engine.machinability import MachinabilityAnalyzer, CuttingParameters
from cam_engine.operations import OperationPlanner, PlannedOperation, OpPurpose
from cam_engine.post import get_post_processor
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext


def test_micro_drill_058_spindle_speed_and_effective_vc():
    """Test that Ø0.58mm drill calculates ~65,900 RPM and derives effective Vc ~27.3 m/min when clamped to 15,000 RPM."""
    machine = get_default_machine()  # max_rpm = 15000
    mat = DEFAULT_MATERIALS["becu"]  # v_carbide = 120.0 m/min
    analyzer = MachinabilityAnalyzer(machine, mat)

    tool_058 = Tool(
        id="T05", tool_number=5, type=ToolType.DRILL, diameter=0.58,
        flute_length=10.0, overall_length=35.0, point_angle=118.0, flutes=2, material="carbide"
    )
    feature = MachiningFeature(
        id="drill_058", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.29, -0.29, 1.47]), bounds_max=np.array([0.29, 0.29, 3.22]),
        depth=1.75, top_z=3.22, floor_z=1.47, diameter=0.58, is_concave=True
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-2.0, -2.0, 0.0]), bounds_max=np.array([2.0, 2.0, 4.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)

    cp = analyzer.analyze(tool_058, feature, OpPurpose.DRILLING.value, setup)

    # 1. Target Vc = 120.0 m/min
    assert cp.target_cutting_speed_m_min == pytest.approx(120.0)

    # 2. Calculated theoretical RPM = 1000 * 120 / (pi * 0.58) = 65,857.2 RPM
    expected_calc_rpm = 1000.0 * 120.0 / (math.pi * 0.58)
    assert cp.target_rpm == pytest.approx(expected_calc_rpm, rel=1e-3)
    assert cp.calculated_rpm == pytest.approx(expected_calc_rpm, rel=1e-3)
    assert round(cp.calculated_rpm) == 65857

    # 3. Machine max clamp
    assert cp.rpm_clamped is True
    assert cp.spindle_rpm == pytest.approx(15000.0)

    # 4. Effective cutting speed = pi * 0.58 * 15000 / 1000 = 27.33 m/min
    expected_effective_vc = (math.pi * 0.58 * 15000.0) / 1000.0
    assert cp.effective_cutting_speed_m_min == pytest.approx(expected_effective_vc, rel=1e-3)
    assert round(cp.effective_cutting_speed_m_min, 1) == 27.3
    assert cp.cutting_speed_m_min == pytest.approx(expected_effective_vc, rel=1e-3)


def test_micro_drill_051_spindle_speed_and_effective_vc():
    """Test that Ø0.51mm drill calculates ~74,900 RPM and derives effective Vc ~24.1 m/min when clamped to 15,000 RPM."""
    machine = get_default_machine()  # max_rpm = 15000
    mat = DEFAULT_MATERIALS["becu"]  # v_carbide = 120.0 m/min
    analyzer = MachinabilityAnalyzer(machine, mat)

    tool_051 = Tool(
        id="T06", tool_number=6, type=ToolType.DRILL, diameter=0.51,
        flute_length=8.0, overall_length=35.0, point_angle=118.0, flutes=2, material="carbide"
    )
    feature = MachiningFeature(
        id="drill_051", type=FeatureType.BLIND_HOLE, face_indices=[0],
        bounds_min=np.array([-0.255, -0.255, -1.27]), bounds_max=np.array([0.255, 0.255, 0.0]),
        depth=1.27, top_z=0.0, floor_z=-1.27, diameter=0.51, is_concave=True
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-2.0, -2.0, -2.0]), bounds_max=np.array([2.0, 2.0, 1.0]))
    setup = Setup(id="s2", name="Setup 2", model_to_setup=np.eye(4), work_offset=WorkOffset.G55, stock=stock)

    cp = analyzer.analyze(tool_051, feature, OpPurpose.DRILLING.value, setup)

    # 1. Target Vc = 120.0 m/min
    assert cp.target_cutting_speed_m_min == pytest.approx(120.0)

    # 2. Calculated theoretical RPM = 1000 * 120 / (pi * 0.51) = 74,896.4 RPM
    expected_calc_rpm = 1000.0 * 120.0 / (math.pi * 0.51)
    assert cp.target_rpm == pytest.approx(expected_calc_rpm, rel=1e-3)
    assert round(cp.calculated_rpm) == 74896

    # 3. Machine max clamp
    assert cp.rpm_clamped is True
    assert cp.spindle_rpm == pytest.approx(15000.0)

    # 4. Effective cutting speed = pi * 0.51 * 15000 / 1000 = 24.03 m/min
    expected_effective_vc = (math.pi * 0.51 * 15000.0) / 1000.0
    assert cp.effective_cutting_speed_m_min == pytest.approx(expected_effective_vc, rel=1e-3)
    assert round(cp.effective_cutting_speed_m_min, 1) == 24.0
    assert round(cp.effective_cutting_speed_m_min, 2) == 24.03


def test_unclamped_tool_speed_equivalence():
    """Test that larger cutter whose RPM is within machine limits maintains target Vc == effective Vc."""
    machine = get_default_machine()  # max_rpm = 15000
    mat = DEFAULT_MATERIALS["becu"]  # v_carbide = 120.0 m/min
    analyzer = MachinabilityAnalyzer(machine, mat)

    tool_10 = Tool(
        id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=10.0,
        flute_length=25.0, overall_length=75.0, flutes=3, material="carbide"
    )
    feature = MachiningFeature(
        id="pocket_1", type=FeatureType.OPEN_POCKET, face_indices=[0],
        bounds_min=np.array([-20.0, -20.0, 0.0]), bounds_max=np.array([20.0, 20.0, 10.0]),
        depth=10.0, top_z=10.0, floor_z=0.0, is_concave=True
    )
    stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-25.0, -25.0, 0.0]), bounds_max=np.array([25.0, 25.0, 15.0]))
    setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)

    cp = analyzer.analyze(tool_10, feature, OpPurpose.ROUGHING.value, setup)

    # RPM = 1000 * 120 / (pi * 10.0) = 3,819.7 RPM < 15,000 RPM
    assert cp.rpm_clamped is False
    assert cp.spindle_rpm == pytest.approx(3819.7, rel=1e-3)
    assert cp.target_cutting_speed_m_min == pytest.approx(120.0)
    assert cp.effective_cutting_speed_m_min == pytest.approx(120.0)


def test_post_processor_emits_transparent_speed_audit_comment():
    """Test that postprocessor emits complete and transparent speed comments in G-code."""
    tp = Toolpath(tool_id="T05", feature_id="drill_058", operation_id="op_drill_058", purpose="DRILLING")
    tp.metadata["tool"] = {"number": 5, "diameter": 0.58, "type": "drill", "material": "carbide"}
    tp.metadata["cutting_parameters"] = {
        "material_name": "brass_360",
        "material_hardness_hb": 110.0,
        "target_cutting_speed_m_min": 120.0,
        "effective_cutting_speed_m_min": 27.33,
        "calculated_rpm": 65858.4,
        "target_rpm": 65858.4,
        "spindle_rpm": 15000.0,
        "rpm_clamped": True,
        "machine_max_rpm": 15000.0,
        "feed_rate": 120.7,
        "feed_per_tooth_mm": 0.004,
        "flute_count": 2,
    }
    tp.add(MotionSegment(motion_type=MotionType.RAPID, start=np.array([0, 0, 10]), end=np.array([0, 0, 4.22])))
    tp.add(MotionSegment(motion_type=MotionType.PLUNGE, start=np.array([0, 0, 4.22]), end=np.array([0, 0, 1.47]), feed=120.7, spindle=15000.0))

    post = get_post_processor("fanuc", work_offset="G54")
    res = post.post_process([tp], "TEST_SPEED_COMMENT")
    gcode = res.gcode

    assert "Target Vc = 120.0 m/min" in gcode
    assert "Calc: S65858 RPM" in gcode
    assert "Machine Limit: 15000 RPM" in gcode
    assert "Applied S15000 RPM" in gcode
    assert "Effective Vc = 27.3 m/min" in gcode
