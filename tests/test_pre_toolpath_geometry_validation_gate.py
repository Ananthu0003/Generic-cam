"""Tests for Gate 2: Mandatory Geometry Validation Gate BEFORE toolpath generation.

Validates:
1. Feature Geometry (face exists, edge exists, boundary closed, surface valid, axis valid, dimensions valid)
2. Manufacturing Geometry (target reachable, belongs to setup, tool access, diameter compatible, holder clearance)
3. Coordinate Transformation (CAD coords valid, setup transform valid, no unintended scale, no unintended shear/rotation)
4. Strict NO TOOLPATH guarantee on validation failure
"""

import math
import numpy as np
import pytest

from cam_engine.context import (
    PlanningContext, Setup, Stock, StockKind, Tool, ToolType, WorkOffset, ToolHolder
)
from cam_engine.coords import CoordSpace
from cam_engine.errors import (
    CamError, INVALID_GEOMETRY, FEATURE_INACCESSIBLE, NO_COMPATIBLE_TOOL,
    TOOL_REACH_INSUFFICIENT, INVALID_SETUP_TRANSFORM, COLLISION_DETECTED
)
from cam_engine.features import MachiningFeature, FeatureType, Accessibility
from cam_engine.machinability import CuttingParameters
from cam_engine.operations import PlannedOperation, OpPurpose
from cam_engine.toolpaths.validation import PreToolpathGeometryValidator, ValidationReport
from cam_engine.geometry.topology import Shape, Face, Edge, SurfaceData, SurfaceKind, CurveData, CurveKind


def create_dummy_feature(
    feat_id: str = "feat_001",
    feat_type: FeatureType = FeatureType.POCKET,
    depth: float = 5.0,
    top_z: float = 10.0,
    floor_z: float = 5.0,
    diameter: float = 12.0,
    axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    machining_dir: np.ndarray = np.array([0.0, 0.0, 1.0]),
    accessibility: Accessibility = Accessibility.TOOL_AXIS_OK,
    closed_loop: bool = True,
) -> MachiningFeature:
    bounds_min = np.array([-10.0, -10.0, floor_z])
    bounds_max = np.array([10.0, 10.0, top_z])
    
    if closed_loop:
        boundary_loop = [
            np.array([-5.0, -5.0, top_z]),
            np.array([5.0, -5.0, top_z]),
            np.array([5.0, 5.0, top_z]),
            np.array([-5.0, 5.0, top_z]),
            np.array([-5.0, -5.0, top_z]),
        ]
    else:
        boundary_loop = [
            np.array([-5.0, -5.0, top_z]),
            np.array([5.0, -5.0, top_z]),
            np.array([5.0, 5.0, top_z]),
            np.array([-5.0, 5.0, top_z]),  # not closed back to start
        ]

    return MachiningFeature(
        id=feat_id,
        type=feat_type,
        face_indices=[0, 1],
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        depth=depth,
        top_z=top_z,
        floor_z=floor_z,
        axis=axis.copy() if axis is not None else None,
        diameter=diameter,
        accessibility=accessibility,
        machining_direction=machining_dir.copy() if machining_dir is not None else None,
        boundary_loop=boundary_loop,
    )


def create_dummy_setup(transform: np.ndarray = np.eye(4)) -> Setup:
    stock = Stock(
        kind=StockKind.BOX,
        bounds_min=np.array([-20.0, -20.0, 0.0]),
        bounds_max=np.array([20.0, 20.0, 15.0]),
    )
    return Setup(
        id="setup_001",
        name="Setup 1",
        model_to_setup=transform,
        work_offset=WorkOffset.G54,
        stock=stock,
    )


def test_valid_setup_and_feature_pass():
    setup = create_dummy_setup()
    feat = create_dummy_feature()
    validator = PreToolpathGeometryValidator(setup=setup)

    setup_rep = validator.validate_setup_transform()
    assert setup_rep.passed
    assert len(setup_rep.errors) == 0

    feat_rep = validator.validate_feature(feat)
    assert feat_rep.passed
    assert len(feat_rep.errors) == 0


def test_setup_transform_unintended_scale_rejected():
    # Transform with 1.25x scaling factor
    T_scaled = np.eye(4)
    T_scaled[:3, :3] *= 1.25
    setup = create_dummy_setup(transform=T_scaled)
    validator = PreToolpathGeometryValidator(setup=setup)

    report = validator.validate_setup_transform()
    assert not report.passed
    assert any(e.code == INVALID_SETUP_TRANSFORM and "scale" in e.message.lower() for e in report.errors)


def test_setup_transform_unintended_shear_rejected():
    # Transform with shear distortion
    T_shear = np.eye(4)
    T_shear[0, 1] = 0.5
    setup = create_dummy_setup(transform=T_shear)
    validator = PreToolpathGeometryValidator(setup=setup)

    report = validator.validate_setup_transform()
    assert not report.passed
    assert any(e.code == INVALID_SETUP_TRANSFORM and "shear" in e.message.lower() for e in report.errors)


def test_setup_transform_nan_or_singular_rejected():
    T_nan = np.eye(4)
    T_nan[0, 0] = np.nan
    setup = create_dummy_setup(transform=T_nan)
    validator = PreToolpathGeometryValidator(setup=setup)

    report = validator.validate_setup_transform()
    assert not report.passed
    assert any(e.code == INVALID_SETUP_TRANSFORM for e in report.errors)

    T_singular = np.zeros((4, 4))
    T_singular[3, 3] = 1.0
    setup_sing = create_dummy_setup(transform=T_singular)
    validator_sing = PreToolpathGeometryValidator(setup=setup_sing)
    report_sing = validator_sing.validate_setup_transform()
    assert not report_sing.passed
    assert any(e.code == INVALID_SETUP_TRANSFORM and "singular" in e.message.lower() for e in report_sing.errors)


def test_feature_open_boundary_rejected():
    setup = create_dummy_setup()
    feat_open = create_dummy_feature(feat_type=FeatureType.POCKET, closed_loop=False)
    validator = PreToolpathGeometryValidator(setup=setup)

    report = validator.validate_feature(feat_open)
    assert not report.passed
    assert any(e.code == INVALID_GEOMETRY and "open" in e.message.lower() for e in report.errors)


def test_feature_invalid_axis_rejected():
    setup = create_dummy_setup()
    # Missing axis on axial hole
    feat_no_axis = create_dummy_feature(feat_type=FeatureType.THROUGH_HOLE, axis=None)
    validator = PreToolpathGeometryValidator(setup=setup)

    report = validator.validate_feature(feat_no_axis)
    assert not report.passed
    assert any(e.code == INVALID_GEOMETRY and "axis" in e.message.lower() for e in report.errors)

    # Degenerate zero axis
    feat_zero_axis = create_dummy_feature(feat_type=FeatureType.THROUGH_HOLE, axis=np.array([0.0, 0.0, 0.0]))
    report_zero = validator.validate_feature(feat_zero_axis)
    assert not report_zero.passed
    assert any(e.code == INVALID_GEOMETRY and "degenerate" in e.message.lower() for e in report_zero.errors)

    # Unnormalized axis
    feat_unnorm_axis = create_dummy_feature(feat_type=FeatureType.THROUGH_HOLE, axis=np.array([0.0, 0.0, 5.0]))
    report_unnorm = validator.validate_feature(feat_unnorm_axis)
    assert not report_unnorm.passed
    assert any(e.code == INVALID_GEOMETRY and "normalized" in e.message.lower() for e in report_unnorm.errors)


def test_feature_invalid_dimensions_and_inverted_bounds_rejected():
    setup = create_dummy_setup()
    # Negative depth
    feat_neg_depth = create_dummy_feature(depth=-2.0)
    validator = PreToolpathGeometryValidator(setup=setup)
    report = validator.validate_feature(feat_neg_depth)
    assert not report.passed
    assert any(e.code == INVALID_GEOMETRY and "depth" in e.message.lower() for e in report.errors)

    # Floor Z above Top Z
    feat_inv_z = create_dummy_feature(top_z=5.0, floor_z=10.0)
    report_inv = validator.validate_feature(feat_inv_z)
    assert not report_inv.passed
    assert any(e.code == INVALID_GEOMETRY and "floor z" in e.message.lower() for e in report_inv.errors)

    # Inverted bounding box
    feat_inv_box = create_dummy_feature()
    feat_inv_box.bounds_min = np.array([10.0, 10.0, 5.0])
    feat_inv_box.bounds_max = np.array([-10.0, -10.0, 10.0])
    report_box = validator.validate_feature(feat_inv_box)
    assert not report_box.passed
    assert any(e.code == INVALID_GEOMETRY and "inverted" in e.message.lower() for e in report_box.errors)


def test_feature_inaccessible_and_misaligned_direction_rejected():
    setup = create_dummy_setup()
    # Explicitly marked INACCESSIBLE
    feat_inacc = create_dummy_feature(accessibility=Accessibility.INACCESSIBLE)
    validator = PreToolpathGeometryValidator(setup=setup)
    report = validator.validate_feature(feat_inacc)
    assert not report.passed
    assert any(e.code == FEATURE_INACCESSIBLE for e in report.errors)

    # Machining direction opposite (+Z axis setup vs -Z feature approach)
    feat_wrong_dir = create_dummy_feature(machining_dir=np.array([0.0, 0.0, -1.0]))
    report_dir = validator.validate_feature(feat_wrong_dir)
    assert not report_dir.passed
    assert any(e.code == FEATURE_INACCESSIBLE and "machining direction" in e.message.lower() for e in report_dir.errors)


def create_dummy_params() -> CuttingParameters:
    return CuttingParameters(
        spindle_rpm=5000.0,
        feed_rate=500.0,
        plunge_feed=150.0,
        ramp_feed=250.0,
        feed_per_tooth_mm=0.05,
        depth_of_cut_mm=1.0,
        stepover_mm=2.0,
        coolant=True,
        effective_diameter_mm=6.0,
        engagement_angle_deg=90.0,
        power_kw=1.0,
        torque_nm=2.0,
    )


def test_operation_tool_reach_insufficient_rejected():
    setup = create_dummy_setup()
    feat = create_dummy_feature(depth=20.0, top_z=20.0, floor_z=0.0)
    tool_short = Tool(
        id="tool_short",
        tool_number=1,
        type=ToolType.FLAT_ENDMILL,
        diameter=6.0,
        flute_length=10.0,  # 10mm flute < 20mm depth
        overall_length=50.0,
    )
    params = create_dummy_params()
    op = PlannedOperation(
        id="op_pocket",
        purpose=OpPurpose.ROUGHING,
        feature=feat,
        tool=tool_short,
        params=params,
    )

    validator = PreToolpathGeometryValidator(setup=setup)
    report = validator.validate_operation(op)
    assert not report.passed
    assert any(e.code == TOOL_REACH_INSUFFICIENT for e in report.errors)


def test_operation_tool_diameter_incompatible_rejected():
    setup = create_dummy_setup()
    # Hole Ø5.0 mm but drill is Ø6.0 mm
    feat_hole = create_dummy_feature(
        feat_type=FeatureType.BLIND_HOLE,
        depth=10.0,
        diameter=5.0,
    )
    tool_large = Tool(
        id="drill_large",
        tool_number=2,
        type=ToolType.DRILL,
        diameter=6.0,
        flute_length=30.0,
        overall_length=60.0,
    )
    params = create_dummy_params()
    op = PlannedOperation(
        id="op_drill",
        purpose=OpPurpose.DRILLING,
        feature=feat_hole,
        tool=tool_large,
        params=params,
    )

    validator = PreToolpathGeometryValidator(setup=setup)
    report = validator.validate_operation(op)
    assert not report.passed
    assert any(e.code == NO_COMPATIBLE_TOOL for e in report.errors)


def test_operation_holder_clearance_collision_rejected():
    setup = create_dummy_setup()
    feat = create_dummy_feature(depth=25.0, top_z=25.0, floor_z=0.0)
    holder = ToolHolder(diameter=28.0, length=40.0)
    tool_with_holder = Tool(
        id="tool_holder_short",
        tool_number=3,
        type=ToolType.FLAT_ENDMILL,
        diameter=6.0,
        flute_length=15.0,  # 15mm < 25mm depth -> holder will collide
        overall_length=45.0,
        holder=holder,
    )
    params = create_dummy_params()
    op = PlannedOperation(
        id="op_deep_pocket",
        purpose=OpPurpose.ROUGHING,
        feature=feat,
        tool=tool_with_holder,
        params=params,
    )

    validator = PreToolpathGeometryValidator(setup=setup)
    report = validator.validate_operation(op)
    assert not report.passed
    assert any(e.code in (TOOL_REACH_INSUFFICIENT, COLLISION_DETECTED) for e in report.errors)
