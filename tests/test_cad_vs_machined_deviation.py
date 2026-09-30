"""
Test suite for CAD-vs-Machined Deviation Verification & Deviation Mapping.

Verifies:
1. Feature diameter error (e.g. Ø6.000 CAD vs Ø6.004 machined -> deviation = +0.004 mm).
2. Feature depth error (nominal CAD depth vs achieved toolpath depth).
3. Max positive deviation (excess stock) and max negative deviation (gouge).
4. Surface deviation and fillet radius deviation.
5. Formatted summary strings matching CAD drawing inspection standards.
6. Pipeline integration returning DeviationMap in PipelineResult.
"""

from pathlib import Path
import numpy as np
import pytest

from cam_engine.context import Tool, ToolType, Setup, Stock, StockKind, PlanningContext, get_material
from cam_engine.demo_models import get_default_machine, get_default_tools
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType, CoordSpace
from cam_engine.geometry import TriangleMesh, VoxelStock
from cam_engine.deviation import (
    CADMachinedDeviationAnalyzer, DeviationMap, FeatureDeviation, DeviationPoint
)
from cam_engine.pipeline import run_cam_pipeline


def _make_sample_tools():
    t_drill = Tool(
        id="T01_DRILL_6",
        tool_number=1,
        type=ToolType.DRILL,
        diameter=6.004,  # Intentionally Ø6.004 mm to test +0.004mm diameter deviation
        flute_length=30.0,
        overall_length=70.0,
        point_angle=118.0,
    )
    t_endmill = Tool(
        id="T02_ENDMILL_10",
        tool_number=2,
        type=ToolType.FLAT_ENDMILL,
        diameter=10.0,
        flute_length=25.0,
        overall_length=65.0,
    )
    t_ball = Tool(
        id="T03_BALL_4",
        tool_number=3,
        type=ToolType.BALL_ENDMILL,
        diameter=4.0,  # Corner radius = 2.0mm
        flute_length=15.0,
        overall_length=50.0,
    )
    return [t_drill, t_endmill, t_ball]


class TestFeatureDiameterDeviation:
    """Test feature diameter error checking."""

    def test_drilled_hole_diameter_deviation_reporting(self):
        tools = _make_sample_tools()
        # CAD feature: Ø6.000 hole
        feat = MachiningFeature(
            id="f_hole_6mm",
            type=FeatureType.HOLE,
            face_indices=[1],
            bounds_min=np.array([7.0, 7.0, -15.0]),
            bounds_max=np.array([13.0, 13.0, 0.0]),
            depth=15.0,
            diameter=6.000,
            top_z=0.0,
            floor_z=-15.0,
            center=np.array([10.0, 10.0, 0.0]),
        )

        # Toolpath executed with Tool Ø6.004
        tp = Toolpath(operation_id="op_drill", feature_id="f_hole_6mm", tool_id="T01_DRILL_6", purpose="drilling")
        tp.add(MotionSegment(
            motion_type=MotionType.PLUNGE,
            start=np.array([10.0, 10.0, 2.0]),
            end=np.array([10.0, 10.0, -15.0]),
            tool_id="T01_DRILL_6",
        ))

        analyzer = CADMachinedDeviationAnalyzer(tolerance_mm=0.010)
        dev_map = analyzer.analyze(features=[feat], toolpaths=[tp], tools=tools)

        f_dev = dev_map.get_feature_deviation("f_hole_6mm")
        assert f_dev is not None
        assert f_dev.cad_diameter == 6.000
        assert np.isclose(f_dev.machined_diameter, 6.004, atol=1e-5)
        assert np.isclose(f_dev.diameter_deviation, 0.004, atol=1e-5)
        assert f_dev.within_tolerance is True

        # Check formatted string matches user requirement: Ø6.000 CAD | machined = Ø6.004 | deviation = +0.004 mm
        assert "Ø6.000 CAD" in f_dev.formatted_summary
        assert "machined = Ø6.004" in f_dev.formatted_summary
        assert "deviation = +0.004 mm" in f_dev.formatted_summary

    def test_circular_pocket_interpolated_diameter_deviation(self):
        tools = _make_sample_tools()
        # CAD feature: Circular pocket Ø30.000 mm
        feat = MachiningFeature(
            id="f_pocket_circ",
            type=FeatureType.POCKET,
            face_indices=[2],
            bounds_min=np.array([-15.0, -15.0, -5.0]),
            bounds_max=np.array([15.0, 15.0, 0.0]),
            depth=5.0,
            diameter=30.000,
            top_z=0.0,
            floor_z=-5.0,
            center=np.array([0.0, 0.0, 0.0]),
        )

        # Circular milling toolpath using 10mm endmill on orbit radius R = 10.010 mm
        # Machined diameter = 2 * (R_orbit + R_tool) = 2 * (10.010 + 5.000) = 30.020 mm
        tp = Toolpath(operation_id="op_circ", feature_id="f_pocket_circ", tool_id="T02_ENDMILL_10", purpose="finishing")
        tp.add(MotionSegment(
            motion_type=MotionType.ARC_CCW,
            start=np.array([10.010, 0.0, -5.0]),
            end=np.array([10.010, 0.0, -5.0]),
            arc_center=np.array([0.0, 0.0, -5.0]),
            arc_ccw=True,
            tool_id="T02_ENDMILL_10",
        ))

        analyzer = CADMachinedDeviationAnalyzer(tolerance_mm=0.015)
        dev_map = analyzer.analyze(features=[feat], toolpaths=[tp], tools=tools)

        f_dev = dev_map.get_feature_deviation("f_pocket_circ")
        assert f_dev is not None
        assert f_dev.cad_diameter == 30.000
        assert np.isclose(f_dev.machined_diameter, 30.020, atol=1e-4)
        assert np.isclose(f_dev.diameter_deviation, 0.020, atol=1e-4)
        # 0.020mm exceeds tolerance of 0.015mm
        assert f_dev.within_tolerance is False


class TestFeatureDepthAndSurfaceDeviation:
    """Test feature depth and floor surface error checking."""

    def test_pocket_depth_deviation(self):
        tools = _make_sample_tools()
        feat = MachiningFeature(
            id="f_rect_pocket",
            type=FeatureType.POCKET,
            face_indices=[3],
            bounds_min=np.array([-10.0, -10.0, -10.0]),
            bounds_max=np.array([10.0, 10.0, 0.0]),
            depth=10.000,
            top_z=0.0,
            floor_z=-10.000,
        )

        # Toolpath cutting at Z = -9.995 mm (0.005mm shallow / excess stock)
        tp = Toolpath(operation_id="op_pocket", feature_id="f_rect_pocket", tool_id="T02_ENDMILL_10", purpose="finishing")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([-5.0, 0.0, -9.995]),
            end=np.array([5.0, 0.0, -9.995]),
            tool_id="T02_ENDMILL_10",
        ))

        analyzer = CADMachinedDeviationAnalyzer(tolerance_mm=0.020)
        dev_map = analyzer.analyze(features=[feat], toolpaths=[tp], tools=tools)

        f_dev = dev_map.get_feature_deviation("f_rect_pocket")
        assert f_dev is not None
        assert f_dev.cad_depth == 10.000
        assert np.isclose(f_dev.machined_depth, 9.995, atol=1e-5)
        assert np.isclose(f_dev.depth_deviation, -0.005, atol=1e-5)
        assert f_dev.within_tolerance is True


class TestFilletAndCornerDeviation:
    """Test fillet and corner radius error checking."""

    def test_fillet_radius_deviation(self):
        tools = _make_sample_tools()
        # CAD fillet R = 2.000 mm
        feat = MachiningFeature(
            id="f_fillet_2mm",
            type=FeatureType.FILLET,
            face_indices=[4],
            bounds_min=np.array([0.0, 0.0, 0.0]),
            bounds_max=np.array([10.0, 2.0, 2.0]),
            depth=2.0,
            notes={"fillet_radius": 2.000},
            top_z=2.0,
            floor_z=0.0,
        )

        # Machined with T03_BALL_4 (ball nose Ø4.0 mm -> radius = 2.000 mm)
        tp = Toolpath(operation_id="op_fillet", feature_id="f_fillet_2mm", tool_id="T03_BALL_4", purpose="contour_fillet")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([0.0, 0.0, 2.0]),
            end=np.array([10.0, 0.0, 2.0]),
            tool_id="T03_BALL_4",
        ))

        analyzer = CADMachinedDeviationAnalyzer(tolerance_mm=0.010)
        dev_map = analyzer.analyze(features=[feat], toolpaths=[tp], tools=tools)

        f_dev = dev_map.get_feature_deviation("f_fillet_2mm")
        assert f_dev is not None
        assert f_dev.cad_fillet_radius == 2.000
        assert f_dev.machined_fillet_radius == 2.000
        assert np.isclose(f_dev.fillet_deviation, 0.0, atol=1e-5)
        assert f_dev.within_tolerance is True


class TestPipelineDeviationMapIntegration:
    """Validate that run_cam_pipeline outputs a complete deviation map."""

    def test_pipeline_computes_deviation_map(self):
        bracket_path = Path("sample_bracket.step")
        if not bracket_path.exists():
            pytest.skip("sample_bracket.step not found")

        res = run_cam_pipeline(bracket_path, controller="fanuc", work_offset="G54")
        assert res.deviation_map is not None
        assert isinstance(res.deviation_map, DeviationMap)

        dm = res.deviation_map
        assert len(dm.feature_deviations) > 0
        assert len(dm.summary_text) > 0
        assert "CAD-vs-Machined Deviation Report" in dm.summary_text
        assert "Max Positive Deviation" in dm.summary_text
        assert "Max Negative Deviation" in dm.summary_text

        # Verify to_dict serializability
        d = dm.to_dict()
        assert "max_positive_deviation" in d
        assert "max_negative_deviation" in d
        assert "feature_deviations" in d
        assert len(d["feature_deviations"]) > 0
