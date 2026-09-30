"""
Test suite for Swept-Volume Verification Architecture.

Verifies:
1. Tool + Holder -> Swept Volume -> Stock -> CAD Target.
2. Gouge: machined_volume ∩ protected_CAD_volume flags GOUGE_DETECTED.
3. Remaining stock: stock_after_cut - CAD_target tracked with volume metrics.
4. Collision: tool/holder/fixture intersection flags COLLISION_DETECTED.
5. Boundary violation: toolpath outside permitted machining envelope flags boundary violation.
6. Execution before G-code generation across the full CAM pipeline.
"""

import numpy as np
import pytest

from cam_engine.context import (
    PlanningContext, Setup, Stock, StockKind, Tool, ToolType, ToolHolder, Fixture, MachineConfig, MachineAxis, get_material
)
from cam_engine.geometry import TriangleMesh, VoxelStock
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType, CoordSpace
from cam_engine.toolpaths.validation import (
    ToolpathValidator, SweptVolumeVerifier, SweptVolumeReport
)
from cam_engine.errors import GOUGE_DETECTED, COLLISION_DETECTED, INVALID_TOOLPATH, MACHINE_AXIS_LIMIT
from cam_engine.pipeline import run_cam_pipeline
from cam_engine.demo_models import get_default_machine


def _make_context_and_setup(stock_min=None, stock_max=None, fixtures=None):
    s_min = stock_min if stock_min is not None else np.array([-30.0, -30.0, -10.0])
    s_max = stock_max if stock_max is not None else np.array([30.0, 30.0, 10.0])
    stock = Stock(kind=StockKind.BOX, bounds_min=s_min, bounds_max=s_max)
    setup = Setup(
        id="setup_001",
        name="Setup 1",
        stock=stock,
        fixtures=fixtures or [],
    )

    t_flat = Tool(
        id="T01_ENDMILL",
        tool_number=1,
        type=ToolType.FLAT_ENDMILL,
        diameter=10.0,
        flute_length=20.0,
        overall_length=60.0,
        holder=ToolHolder(diameter=35.0, length=40.0),
    )
    t_ball = Tool(
        id="T02_BALL",
        tool_number=2,
        type=ToolType.BALL_ENDMILL,
        diameter=6.0,
        flute_length=15.0,
        overall_length=50.0,
        holder=ToolHolder(diameter=30.0, length=35.0),
    )

    machine = get_default_machine()

    ctx = PlanningContext(
        model=None,
        setups=[setup],
        machine=machine,
        tools=[t_flat, t_ball],
        material=get_material("aluminum_6061"),
    )
    return ctx, setup, [t_flat, t_ball]


def _make_flat_box_mesh(z_top=0.0):
    """Simple 40x40 box mesh with top face at z_top."""
    x0, x1 = -20.0, 20.0
    y0, y1 = -20.0, 20.0
    z0, z1 = -10.0, z_top
    triangles = [
        # Top face (2 triangles, normal +Z)
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1]],
        [[x0, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        # Bottom face
        [[x0, y0, z0], [x1, y1, z0], [x1, y0, z0]],
        [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
        # North (+Y)
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1]],
        [[x0, y1, z0], [x1, y1, z1], [x0, y1, z1]],
        # South (-Y)
        [[x0, y0, z0], [x1, y0, z1], [x1, y0, z0]],
        [[x0, y0, z0], [x0, y0, z1], [x1, y0, z1]],
        # East (+X)
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
        [[x1, y0, z0], [x1, y1, z1], [x1, y0, z1]],
        # West (-X)
        [[x0, y0, z0], [x0, y1, z1], [x0, y1, z0]],
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1]],
    ]
    return TriangleMesh(triangles=np.array(triangles, dtype=float))


class TestSweptVolumeGougeVerification:
    """Validate machined_volume ∩ protected_CAD_volume."""

    def test_gouge_detected_when_cutting_below_cad_surface(self):
        ctx, setup, tools = _make_context_and_setup()
        mesh = _make_flat_box_mesh(z_top=0.0)  # CAD part top at Z=0.0
        verifier = SweptVolumeVerifier(
            context=ctx, setup=setup, mesh=mesh, clearance_z=20.0
        )

        # Create toolpath that cuts at Z = -2.0 mm (penetrating 2mm into CAD part)
        tp = Toolpath(operation_id="op_gouge", feature_id="f1", tool_id="T01_ENDMILL", purpose="finishing")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([-10.0, 0.0, -2.0]),
            end=np.array([10.0, 0.0, -2.0]),
            tool_id="T01_ENDMILL",
        ))

        report = verifier.verify(tp, tools[0])
        assert not report.passed
        assert report.gouge_detected
        assert report.gouge_depth >= 1.9
        assert any(e.code == GOUGE_DETECTED for e in report.errors)

    def test_safe_finishing_toolpath_passes_gouge_check(self):
        ctx, setup, tools = _make_context_and_setup()
        mesh = _make_flat_box_mesh(z_top=0.0)  # CAD part top at Z=0.0
        verifier = SweptVolumeVerifier(
            context=ctx, setup=setup, mesh=mesh, clearance_z=20.0
        )

        # Create toolpath skimming exactly on the surface Z=0.0
        tp = Toolpath(operation_id="op_finish", feature_id="f1", tool_id="T01_ENDMILL", purpose="finishing")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([-15.0, 0.0, 0.0]),
            end=np.array([15.0, 0.0, 0.0]),
            tool_id="T01_ENDMILL",
        ))

        report = verifier.verify(tp, tools[0])
        assert report.passed
        assert not report.gouge_detected
        assert len(report.errors) == 0


class TestSweptVolumeRemainingStock:
    """Validate stock_after_cut - CAD_target and volume tracking."""

    def test_remaining_stock_volume_metrics_tracked(self):
        ctx, setup, tools = _make_context_and_setup()
        mesh = _make_flat_box_mesh(z_top=0.0)
        stock_voxels = VoxelStock.from_bounds(setup.stock.bounds_min, setup.stock.bounds_max, resolution=2.0)
        target_cad_voxels = VoxelStock.from_mesh(mesh, setup.stock.bounds_min, setup.stock.bounds_max, resolution=2.0)
        target_removal = stock_voxels.compute_removal_volume(target_cad_voxels)

        verifier = SweptVolumeVerifier(
            context=ctx,
            setup=setup,
            mesh=mesh,
            clearance_z=20.0,
            stock_voxels=stock_voxels,
            target_cad_voxels=target_cad_voxels,
            target_removal_volume=target_removal,
        )

        tp = Toolpath(operation_id="op_rough", feature_id="f1", tool_id="T01_ENDMILL", purpose="roughing")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([-15.0, 0.0, 5.0]),
            end=np.array([15.0, 0.0, 5.0]),
            tool_id="T01_ENDMILL",
        ))

        report = verifier.verify(tp, tools[0])
        assert report.passed
        assert report.initial_stock_volume > 0
        assert report.target_cad_volume > 0
        assert "swept_volume_metrics" in tp.metadata
        assert tp.metadata["swept_volume_metrics"]["initial_stock_volume"] > 0


class TestSweptVolumeCollisionVerification:
    """Validate tool/holder/fixture intersection."""

    def test_holder_collision_with_tall_wall_detected(self):
        ctx, setup, tools = _make_context_and_setup()
        # Mesh with a high feature / wall at Z = 25.0 mm
        # Tool flute length is 20mm, holder starts at Z = pt[2] + 20mm
        # If tool cuts at Z = 0.0, holder spans Z = 20.0 .. 60.0 mm
        # A CAD wall at Z = 25.0 mm within holder radius (17.5mm) will collide with holder
        tall_mesh = _make_flat_box_mesh(z_top=25.0)
        verifier = SweptVolumeVerifier(
            context=ctx, setup=setup, mesh=tall_mesh, clearance_z=40.0
        )

        tp = Toolpath(operation_id="op_holder_coll", feature_id="f1", tool_id="T01_ENDMILL", purpose="pocketing")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([0.0, 0.0, 0.0]),
            end=np.array([5.0, 0.0, 0.0]),
            tool_id="T01_ENDMILL",
        ))

        report = verifier.verify(tp, tools[0])
        assert not report.passed
        assert report.collision_detected
        assert any("Holder collision" in msg for msg in report.collision_details)
        assert any(e.code == COLLISION_DETECTED for e in report.errors)

    def test_fixture_collision_detected(self):
        # Workholding clamp fixture placed at X = 15..25, Y = -10..10, Z = 0..15
        clamp = Fixture(
            id="clamp_east",
            name="Toe Clamp",
            bounds_min=np.array([15.0, -10.0, 0.0]),
            bounds_max=np.array([25.0, 10.0, 15.0]),
        )
        ctx, setup, tools = _make_context_and_setup(fixtures=[clamp])
        mesh = _make_flat_box_mesh(z_top=0.0)
        verifier = SweptVolumeVerifier(
            context=ctx, setup=setup, mesh=mesh, clearance_z=25.0
        )

        # Toolpath runs directly through clamp at X = 20.0, Y = 0.0, Z = 5.0
        tp = Toolpath(operation_id="op_fixture_coll", feature_id="f1", tool_id="T01_ENDMILL", purpose="roughing")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([10.0, 0.0, 5.0]),
            end=np.array([25.0, 0.0, 5.0]),
            tool_id="T01_ENDMILL",
        ))

        report = verifier.verify(tp, tools[0])
        assert not report.passed
        assert report.collision_detected
        assert any("fixture" in msg for msg in report.collision_details)
        assert any(e.code == COLLISION_DETECTED for e in report.errors)


class TestSweptVolumeBoundaryViolations:
    """Validate toolpath outside permitted machining envelope."""

    def test_cutting_outside_stock_envelope_rejected(self):
        ctx, setup, tools = _make_context_and_setup(
            stock_min=np.array([-20.0, -20.0, -10.0]),
            stock_max=np.array([20.0, 20.0, 10.0]),
        )
        mesh = _make_flat_box_mesh(z_top=0.0)
        verifier = SweptVolumeVerifier(
            context=ctx, setup=setup, mesh=mesh, clearance_z=20.0
        )

        # Cutting at X = 100.0 (way outside stock bounds X=[-20, 20])
        tp = Toolpath(operation_id="op_oob", feature_id="f1", tool_id="T01_ENDMILL", purpose="roughing")
        tp.add(MotionSegment(
            motion_type=MotionType.CUT,
            start=np.array([100.0, 0.0, 5.0]),
            end=np.array([120.0, 0.0, 5.0]),
            tool_id="T01_ENDMILL",
        ))

        report = verifier.verify(tp, tools[0])
        assert not report.passed
        assert report.boundary_violation
        assert any(e.code == INVALID_TOOLPATH for e in report.errors)

    def test_motion_exceeding_machine_travel_rejected(self):
        ctx, setup, tools = _make_context_and_setup()
        mesh = _make_flat_box_mesh(z_top=0.0)
        verifier = SweptVolumeVerifier(
            context=ctx, setup=setup, mesh=mesh, clearance_z=20.0
        )

        # Move exceeding machine Y travel (-300 .. +300)
        tp = Toolpath(operation_id="op_travel_limit", feature_id="f1", tool_id="T01_ENDMILL", purpose="rapid")
        tp.add(MotionSegment(
            motion_type=MotionType.RAPID,
            start=np.array([0.0, 0.0, 20.0]),
            end=np.array([0.0, 600.0, 20.0]),
            tool_id="T01_ENDMILL",
        ))

        report = verifier.verify(tp, tools[0])
        assert not report.passed
        assert report.boundary_violation
        assert any(e.code == MACHINE_AXIS_LIMIT for e in report.errors)


class TestSweptVolumePipelineIntegration:
    """Validate full CAM pipeline integration of swept-volume verification."""

    def test_pipeline_validates_swept_volume_before_gcode(self):
        from pathlib import Path
        bracket_path = Path("sample_bracket.step")
        if not bracket_path.exists():
            pytest.skip("sample_bracket.step not found")

        res = run_cam_pipeline(bracket_path, controller="fanuc", work_offset="G54")
        assert res.toolpaths is not None
        assert len(res.toolpaths) > 0

        # Every toolpath generated by pipeline must be swept-volume verified
        for tp in res.toolpaths:
            assert tp.metadata.get("swept_volume_verified") is True
            assert "swept_volume_report" in tp.metadata
            assert isinstance(tp.metadata["swept_volume_report"], SweptVolumeReport)
            assert tp.metadata["swept_volume_report"].passed
