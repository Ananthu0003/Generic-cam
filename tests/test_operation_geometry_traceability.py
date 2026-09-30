"""Tests for Operation-to-Geometry Traceability across CAM pipeline.

Validates the full traceability chain:
CAD Face/Edge IDs -> Feature ID -> Operation ID -> Toolpath Segment ID -> Motion Segment
"""

import numpy as np
import pytest
from cam_engine.context import (
    Tool,
    ToolType,
    ToleranceModel,
    MachiningAllowance,
    FinalManufacturingTarget,
)
from cam_engine.coords import CoordSpace
from cam_engine.features import FeatureType, MachiningFeature, Accessibility
from cam_engine.operations import OpPurpose, PlannedOperation
from cam_engine.errors import CamError
from cam_engine.toolpaths.semantic import (
    MotionSegment,
    MotionType,
    Toolpath,
    ToolpathGeometryMode,
    Rapid,
    LinearCut,
    CircularCut,
)
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.api.models import MotionSegmentItem, ToolpathItem


def test_motion_segment_traceability_fields_defaults():
    """Verify MotionSegment holds all required traceability fields."""
    seg = MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([10.0, 0.0, 0.0]),
        segment_id="seg_001",
        operation_id="op_pocket_1",
        feature_id="feat_pocket_main",
        source_face_ids=[12, 14, 15],
        source_edge_ids=[101, 102],
        geometry_mode=ToolpathGeometryMode.TOOL_CENTER,
        target_surface="pocket_floor",
    )

    assert seg.segment_id == "seg_001"
    assert seg.operation_id == "op_pocket_1"
    assert seg.feature_id == "feat_pocket_main"
    assert seg.source_face_ids == [12, 14, 15]
    assert seg.source_edge_ids == [101, 102]
    assert seg.geometry_mode == ToolpathGeometryMode.TOOL_CENTER
    assert seg.target_surface == "pocket_floor"


def test_toolpath_auto_populates_segment_traceability():
    """Toolpath.add() should automatically inherit traceability from parent Toolpath if missing."""
    tp = Toolpath(
        operation_id="op_contour_1",
        feature_id="feat_profile_ext",
        tool_id="tool_em_6",
        purpose="contour",
        geometry_mode=ToolpathGeometryMode.NOMINAL_BOUNDARY,
        source_face_ids=[5, 6, 7],
        source_edge_ids=[20, 21, 22],
        target_surface="outer_profile",
    )

    # Add bare segment without explicit metadata
    seg1 = LinearCut(start=[0, 0, 0], end=[10, 0, 0])
    tp.add(seg1)

    assert seg1.segment_id == "op_contour_1_seg_0001"
    assert seg1.operation_id == "op_contour_1"
    assert seg1.feature_id == "feat_profile_ext"
    assert seg1.source_face_ids == [5, 6, 7]
    assert seg1.source_edge_ids == [20, 21, 22]
    assert seg1.geometry_mode == ToolpathGeometryMode.NOMINAL_BOUNDARY
    assert seg1.target_surface == "outer_profile"

    # Add second segment
    seg2 = LinearCut(start=[10, 0, 0], end=[10, 10, 0])
    tp.add(seg2)
    assert seg2.segment_id == "op_contour_1_seg_0002"


def test_coordinate_transformation_preserves_traceability():
    """Transforming a Toolpath across coordinate systems must preserve all traceability metadata."""
    tp = Toolpath(
        operation_id="op_drill_1",
        feature_id="feat_hole_8mm",
        tool_id="tool_drill_8",
        purpose="drilling",
        space=CoordSpace.SETUP,
        geometry_mode=ToolpathGeometryMode.TOOL_CENTER,
        source_face_ids=[42],
        source_edge_ids=[99],
        target_surface="cylinder_hole",
    )
    seg = LinearCut(start=[0, 0, 5], end=[0, 0, -15])
    tp.add(seg)

    # 90 deg rotation around Z
    matrix = np.array([
        [0, -1, 0, 10],
        [1,  0, 0, 20],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=float)

    tp_work = tp.transformed(matrix, dst_space=CoordSpace.WORK)

    assert tp_work.operation_id == "op_drill_1"
    assert tp_work.feature_id == "feat_hole_8mm"
    assert tp_work.source_face_ids == [42]
    assert tp_work.source_edge_ids == [99]
    assert tp_work.geometry_mode == ToolpathGeometryMode.TOOL_CENTER
    assert tp_work.target_surface == "cylinder_hole"
    assert len(tp_work.segments) == 1

    seg_work = tp_work.segments[0]
    assert seg_work.segment_id == "op_drill_1_seg_0001"
    assert seg_work.operation_id == "op_drill_1"
    assert seg_work.feature_id == "feat_hole_8mm"
    assert seg_work.source_face_ids == [42]
    assert seg_work.source_edge_ids == [99]
    assert seg_work.geometry_mode == ToolpathGeometryMode.TOOL_CENTER
    assert seg_work.target_surface == "cylinder_hole"


from cam_engine.machinability import CuttingParameters


def test_strategy_engine_populates_full_traceability_chain():
    """StrategyEngine must propagate CAD feature face/edge indices into generated toolpaths and segments."""
    feat = MachiningFeature(
        id="feat_hole_10",
        type=FeatureType.HOLE,
        face_indices=[101, 102],
        bounds_min=np.array([-5.0, -5.0, -20.0]),
        bounds_max=np.array([5.0, 5.0, 0.0]),
        depth=20.0,
        top_z=0.0,
        floor_z=-20.0,
        diameter=10.0,
        axis=np.array([0.0, 0.0, 1.0]),
        source_edges=[501, 502],
        notes={
            "target": FinalManufacturingTarget.derive(
                nominal=10.0,
                dimension_type="diameter",
            )
        }
    )

    tool = Tool(
        id="T01",
        tool_number=1,
        type=ToolType.DRILL,
        diameter=10.0,
        flute_length=30.0,
        overall_length=60.0,
    )

    params = CuttingParameters(
        spindle_rpm=2500.0,
        feed_rate=300.0,
        plunge_feed=150.0,
        ramp_feed=150.0,
        feed_per_tooth_mm=0.06,
        depth_of_cut_mm=20.0,
        stepover_mm=10.0,
        coolant=True,
        effective_diameter_mm=10.0,
        engagement_angle_deg=180.0,
        power_kw=1.0,
        torque_nm=3.8,
    )

    op = PlannedOperation(
        id="op_drill_hole_10",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.DRILLING,
        params=params,
    )

    strat_ctx = StrategyContext(
        context=None,
        setup=None,
        mesh=None,
        stock_voxels=None,
        clearance_z=20.0,
        finish_allowance_wall=0.0,
        finish_allowance_floor=0.0,
    )
    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    assert tp.operation_id == "op_drill_hole_10"
    assert tp.feature_id == "feat_hole_10"
    assert tp.source_face_ids == [101, 102]
    assert tp.source_edge_ids == [501, 502]
    assert len(tp.segments) > 0

    for seg in tp.segments:
        assert seg.segment_id is not None
        assert seg.segment_id.startswith("op_drill_hole_10_seg_")
        assert seg.operation_id == "op_drill_hole_10"
        assert seg.feature_id == "feat_hole_10"
        assert seg.source_face_ids == [101, 102]
        assert seg.source_edge_ids == [501, 502]


def test_api_models_traceability_serialization():
    """API Pydantic models must faithfully serialize and expose all traceability fields."""
    seg_item = MotionSegmentItem(
        segment_id="op_face_01_seg_0001",
        motion_type="cut",
        start=[0.0, 0.0, 0.0],
        end=[50.0, 0.0, 0.0],
        feed=1200.0,
        spindle=8001.0,
        tool_id="T02",
        operation_id="op_face_01",
        feature_id="feat_top_face",
        source_face_ids=[1, 2],
        source_edge_ids=[10, 11, 12, 13],
        geometry_mode="tool_center",
        target_surface="plane",
    )

    tp_item = ToolpathItem(
        operation_id="op_face_01",
        feature_id="feat_top_face",
        purpose="facing",
        tool_id="T02",
        segment_count=1,
        cutting_length_mm=50.0,
        rapid_length_mm=0.0,
        geometry_mode="tool_center",
        source_face_ids=[1, 2],
        source_edge_ids=[10, 11, 12, 13],
        target_surface="plane",
        segments=[seg_item],
    )

    d = tp_item.model_dump()
    assert d["feature_id"] == "feat_top_face"
    assert d["source_face_ids"] == [1, 2]
    assert d["source_edge_ids"] == [10, 11, 12, 13]
    assert d["geometry_mode"] == "tool_center"
    assert d["target_surface"] == "plane"
    assert d["segments"][0]["segment_id"] == "op_face_01_seg_0001"
    assert d["segments"][0]["source_face_ids"] == [1, 2]


def test_pocket_contour_traceability_chain():
    """Verify pocket/contour operation preserves CAD topology across every segment."""
    loop = [
        np.array([0.0, 0.0, 0.0]),
        np.array([40.0, 0.0, 0.0]),
        np.array([40.0, 30.0, 0.0]),
        np.array([0.0, 30.0, 0.0]),
    ]
    feat = MachiningFeature(
        id="feat_pocket_01",
        type=FeatureType.POCKET,
        face_indices=[21, 22, 23, 24, 25],
        bounds_min=np.array([0.0, 0.0, -10.0]),
        bounds_max=np.array([40.0, 30.0, 0.0]),
        depth=10.0,
        top_z=0.0,
        floor_z=-10.0,
        source_edges=[101, 102, 103, 104],
        boundary_loop=loop,
        wall_faces=[21, 22, 23, 24],
        floor_faces=[25],
    )

    tool = Tool(
        id="T02",
        tool_number=2,
        type=ToolType.FLAT_ENDMILL,
        diameter=6.0,
        flute_length=20.0,
        overall_length=50.0,
    )

    params = CuttingParameters(
        spindle_rpm=6000.0,
        feed_rate=1200.0,
        plunge_feed=300.0,
        ramp_feed=500.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=5.0,
        stepover_mm=3.0,
        coolant=True,
        effective_diameter_mm=6.0,
        engagement_angle_deg=90.0,
        power_kw=1.5,
        torque_nm=2.4,
    )

    op = PlannedOperation(
        id="op_pocket_rough",
        feature=feat,
        tool=tool,
        purpose=OpPurpose.ROUGHING,
        params=params,
    )

    strat_ctx = StrategyContext(
        context=None,
        setup=None,
        mesh=None,
        stock_voxels=None,
        clearance_z=15.0,
        finish_allowance_wall=0.2,
        finish_allowance_floor=0.1,
    )

    engine = StrategyEngine(strat_ctx)
    tp = engine.generate(op)

    assert tp.operation_id == "op_pocket_rough"
    assert tp.feature_id == "feat_pocket_01"
    assert tp.source_face_ids == [21, 22, 23, 24, 25]
    assert tp.source_edge_ids == [101, 102, 103, 104]
    assert len(tp.segments) > 0

    # Every single motion segment must be uniquely traceable back to the CAD feature
    for idx, seg in enumerate(tp.segments, start=1):
        assert seg.segment_id == f"op_pocket_rough_seg_{idx:04d}"
        assert seg.operation_id == "op_pocket_rough"
        assert seg.feature_id == "feat_pocket_01"
        assert seg.source_face_ids == [21, 22, 23, 24, 25]
        assert seg.source_edge_ids == [101, 102, 103, 104]

