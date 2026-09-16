"""End-to-End CAM Pipeline: load STEP -> recognize -> plan -> toolpath -> post-process."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from .geometry.step_import import import_step, ImportedModel
from .geometry.mesh import TriangleMesh
from .geometry.voxel import VoxelStock
from .context import (
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Material,
    DEFAULT_MATERIALS,
    Units,
    WorkOffset,
)
from .errors import CamError, INVALID_TOOLPATH, INVALID_WORK_OFFSET, MISSING_MATERIAL_DATA
from .features import FeatureRecognizer, MachiningFeature
from .operations import OperationPlanner, PlannedOperation
from .toolpaths.strategies import StrategyEngine, StrategyContext
from .toolpaths.semantic import Toolpath, MotionType
from .toolpaths.validation import ToolpathValidator, ValidationReport
from .post import get_post_processor, PostResult
from .demo_models import get_default_tools, get_default_machine


@dataclass
class PipelineResult:
    model: ImportedModel
    features: list[MachiningFeature]
    operations: list[PlannedOperation]
    toolpaths: list[Toolpath]
    post_result: PostResult
    gcode: str
    validation_reports: list[ValidationReport] = field(default_factory=list)
    errors: list[CamError] = field(default_factory=list)


def run_cam_pipeline(
    step_path: str | Path,
    controller: str = "grbl",
    work_offset: str = "G54",
    margin_x: float = 5.0,
    margin_y: float = 5.0,
    margin_z_top: float = 1.0,
    margin_z_bottom: float = 3.0,
    material_name: str = "aluminum_6061",
    model_to_setup: Optional[np.ndarray] = None,
    output_gcode_path: Optional[str | Path] = None,
    stock_mode: str = "relative_box",
    fixed_size: Optional[tuple[float, float, float]] = None,
    cylinder_diameter: Optional[float] = None,
    cylinder_length: Optional[float] = None,
) -> PipelineResult:
    """Execute the full CAM pipeline on an input STEP file."""
    path = Path(step_path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found at: {path}")

    # 1. STEP Import & Geometry
    imported = import_step(path)
    setup_transform = model_to_setup if model_to_setup is not None else np.eye(4)

    # 2. Setup & Context (transformed into setup coordinates)
    shape_in_setup = imported.shape.transformed(setup_transform)
    bmin, bmax = shape_in_setup.bounding_box()

    if stock_mode == "cylinder":
        cx = 0.5 * (bmin[0] + bmax[0])
        cy = 0.5 * (bmin[1] + bmax[1])
        rx = 0.5 * (bmax[0] - bmin[0])
        ry = 0.5 * (bmax[1] - bmin[1])
        part_r = float(np.hypot(rx, ry))
        dia = float(cylinder_diameter if (cylinder_diameter and cylinder_diameter > 0) else (2.0 * (part_r + margin_x)))
        radius = dia / 2.0
        z_top = float(bmax[2] + margin_z_top)
        z_bot = z_top - float(cylinder_length) if (cylinder_length and cylinder_length > 0) else float(bmin[2] - margin_z_bottom)
        stock = Stock.from_cylinder(center_xy=np.array([cx, cy]), z_min=z_bot, z_max=z_top, radius=radius)
    elif stock_mode == "fixed_box" and fixed_size is not None:
        fx, fy, fz = fixed_size
        cx = 0.5 * (bmin[0] + bmax[0])
        cy = 0.5 * (bmin[1] + bmax[1])
        z_top = float(bmax[2] + margin_z_top)
        z_bot = z_top - float(fz)
        stock_min = np.array([cx - fx / 2.0, cy - fy / 2.0, z_bot])
        stock_max = np.array([cx + fx / 2.0, cy + fy / 2.0, z_top])
        stock = Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)
    else:
        stock_min = np.array([bmin[0] - margin_x, bmin[1] - margin_y, bmin[2] - margin_z_bottom])
        stock_max = np.array([bmax[0] + margin_x, bmax[1] + margin_y, bmax[2] + margin_z_top])
        stock = Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)

    wo_enum = None
    for w in WorkOffset:
        if w.value == work_offset:
            wo_enum = w
            break
    if wo_enum is None:
        valid = [w.value for w in WorkOffset]
        raise CamError(
            code=INVALID_WORK_OFFSET,
            message=f"unknown work offset '{work_offset}', valid: {valid}",
            stage="pipeline_setup",
        )

    setup = Setup(
        id="setup_001",
        name="Setup 1 - Top Milling",
        model_to_setup=setup_transform,
        work_offset=wo_enum,
        stock=stock,
    )

    machine = get_default_machine()
    tools = get_default_tools()
    material = DEFAULT_MATERIALS.get(material_name)
    if material is None:
        valid = list(DEFAULT_MATERIALS.keys())
        raise CamError(
            code=MISSING_MATERIAL_DATA,
            message=f"unknown material '{material_name}', valid: {valid}",
            stage="pipeline_setup",
        )

    ctx = PlanningContext(
        model=imported,
        setups=[setup],
        machine=machine,
        tools=tools,
        material=material,
    )
    ctx.validate()

    # 3–6: Process each setup
    all_features: list[MachiningFeature] = []
    all_operations: list[PlannedOperation] = []
    all_toolpaths: list[Toolpath] = []
    all_validation_reports: list[ValidationReport] = []
    all_errors: list[CamError] = []
    prev_remaining_stock = None  # for multi-setup stock transfer

    for s in ctx.setups:
        # 3. Feature Recognition
        setup_shape = ctx.model_shape_in_setup(s)
        stock_top = float(s.stock.bounds_max[2])
        # Pass available drill diameters so bore/hole classification reflects the
        # actual tool library instead of a fixed 12mm threshold
        from .context import ToolType as _ToolType
        drill_dias = sorted(set(
            t.diameter for t in ctx.tools if t.type is _ToolType.DRILL
        )) or None
        recognizer = FeatureRecognizer(
            setup_shape, stock_top_z=stock_top, drill_diameters=drill_dias
        )
        features = recognizer.recognize()
        all_features.extend(features)

        # 4. Operation Planning
        planner = OperationPlanner(ctx, s, features)
        operations = planner.plan()
        all_operations.extend(operations)

        # 5. Toolpath Generation
        mesh = TriangleMesh.from_faces(setup_shape.faces())

        dims = s.stock.bounds_max - s.stock.bounds_min
        resolution = max(0.5, min(dims) / 50.0)
        # Multi-setup stock transfer: use remaining stock from previous setup
        if prev_remaining_stock is not None:
            stock_voxels = prev_remaining_stock
        else:
            stock_voxels = VoxelStock.from_bounds(
                s.stock.bounds_min, s.stock.bounds_max, resolution,
            )

        clearance_z = float(s.stock.bounds_max[2]) + machine.safe_retract_height

        # Derive finish allowances from FinishRequirement entries (tolerance-driven).
        # Fallback to 0.3mm when no requirements are set.
        finish_allowance_wall, finish_allowance_floor = _derive_finish_allowances(
            ctx, material
        )
        # Ramp angle: conservative for hard materials, more aggressive for soft
        ramp_angle_deg = 10.0 if material.hardness_hb >= 200 else (
            15.0 if material.hardness_hb >= 100 else 20.0
        )

        strat_ctx = StrategyContext(
            context=ctx,
            setup=s,
            mesh=mesh,
            stock_voxels=stock_voxels,
            clearance_z=clearance_z,
            finish_allowance_wall=finish_allowance_wall,
            finish_allowance_floor=finish_allowance_floor,
            ramp_angle_deg=ramp_angle_deg,
        )
        engine = StrategyEngine(strat_ctx)

        setup_toolpaths: list[Toolpath] = []
        for op in operations:
            try:
                tp = engine.generate(op)
                setup_toolpaths.append(tp)
                # Update remaining stock after each operation
                tool_radius = op.tool.diameter / 2.0
                for seg in tp.segments:
                    if seg.motion_type in (
                        MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
                        MotionType.ARC_CW, MotionType.ARC_CCW,
                    ):
                        # For arc/helix motions, sample intermediate points
                        if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW):
                            import math
                            center = seg.arc_center if seg.arc_center is not None else None
                            if center is not None:
                                # Compute true arc path using center, start, end angles
                                r_start = seg.start[:2] - center[:2]
                                r_end = seg.end[:2] - center[:2]
                                angle_start = math.atan2(r_start[1], r_start[0])
                                angle_end = math.atan2(r_end[1], r_end[0])
                                if seg.motion_type == MotionType.ARC_CCW:
                                    if angle_end < angle_start:
                                        angle_end += 2 * math.pi
                                else:
                                    if angle_end > angle_start:
                                        angle_end -= 2 * math.pi
                                arc_radius = float(np.linalg.norm(r_start))
                                n_samples = max(4, int(math.ceil(arc_radius * abs(angle_end - angle_start) / 1.0)))
                                for k in range(n_samples):
                                    t0 = k / n_samples
                                    t1 = (k + 1) / n_samples
                                    a0 = angle_start + t0 * (angle_end - angle_start)
                                    a1 = angle_start + t1 * (angle_end - angle_start)
                                    z0 = seg.start[2] + t0 * (seg.end[2] - seg.start[2])
                                    z1 = seg.start[2] + t1 * (seg.end[2] - seg.start[2])
                                    p0 = np.array([center[0] + arc_radius * math.cos(a0),
                                                   center[1] + arc_radius * math.sin(a0), z0])
                                    p1 = np.array([center[0] + arc_radius * math.cos(a1),
                                                   center[1] + arc_radius * math.sin(a1), z1])
                                    stock_voxels.remove_capsule(p0, p1, tool_radius)
                            else:
                                # Fallback: linear interpolation if no arc center
                                n_samples = max(4, int(math.ceil(
                                    float(np.linalg.norm(seg.end - seg.start)) / 1.0)))
                                for k in range(n_samples):
                                    t0 = k / n_samples
                                    t1 = (k + 1) / n_samples
                                    p0 = seg.start + t0 * (seg.end - seg.start)
                                    p1 = seg.start + t1 * (seg.end - seg.start)
                                    stock_voxels.remove_capsule(p0, p1, tool_radius)
                        else:
                            stock_voxels.remove_capsule(seg.start, seg.end, tool_radius)
                    elif seg.motion_type == MotionType.HELIX:
                        stock_voxels.remove_capsule(seg.start, seg.end, tool_radius)
            except CamError as e:
                all_errors.append(e)
            except Exception as e:
                all_errors.append(CamError(
                    code=INVALID_TOOLPATH,
                    message=f"toolpath generation failed for {op.id}: {e}",
                    stage="toolpath_generation",
                    operation_id=op.id,
                ))

        # 6. Validation
        validator = ToolpathValidator(ctx, s, mesh, clearance_z)
        validated_toolpaths: list[Toolpath] = []
        for tp in setup_toolpaths:
            report = validator.validate(tp)
            all_validation_reports.append(report)
            if report.passed:
                validated_toolpaths.append(tp)
            else:
                for err in report.errors:
                    all_errors.append(err)
        all_toolpaths.extend(validated_toolpaths)
        # Store remaining stock for multi-setup transfer
        prev_remaining_stock = stock_voxels

    # 7. Post-Processing
    post = get_post_processor(controller=controller, work_offset=work_offset)
    program_name = path.stem.upper().replace(" ", "_")
    post_res = post.post_process(all_toolpaths, program_name=program_name)

    if output_gcode_path is not None:
        out_p = Path(output_gcode_path)
        out_p.write_text(post_res.gcode, encoding="utf-8")

    return PipelineResult(
        model=imported,
        features=all_features,
        operations=all_operations,
        toolpaths=all_toolpaths,
        post_result=post_res,
        gcode=post_res.gcode,
        validation_reports=all_validation_reports,
        errors=all_errors,
    )


def _derive_finish_allowances(
    ctx: PlanningContext,
    material: Material,
) -> tuple[float, float]:
    """Derive finish allowances from FinishRequirement entries in the planning context.
    When requirements carry explicit tolerances, the allowance is proportional to the
    tightest tolerance; otherwise a sensible default (0.3mm) is used.

    Returns:
        (finish_allowance_wall, finish_allowance_floor) in mm
    """
    default_wall = 0.3
    default_floor = 0.3

    # Gather all tolerances from any finish requirements in the context
    # (FinishRequirement objects may live on the context or on individual features)
    tols: list[float] = []
    if hasattr(ctx, "requirements") and ctx.requirements:
        for req in ctx.requirements.values():
            if hasattr(req, "tolerance_mm") and req.tolerance_mm and req.tolerance_mm > 0:
                tols.append(float(req.tolerance_mm))

    if not tols:
        return default_wall, default_floor

    min_tol = min(tols)
    # Wall allowance: 1/3 of tightest tolerance, clamped to [0.05, 0.5] mm
    wall = float(min(max(min_tol / 3.0, 0.05), 0.5))
    # Floor allowance: 1/4 of tightest tolerance, clamped to [0.05, 0.3] mm
    floor = float(min(max(min_tol / 4.0, 0.05), 0.3))
    return wall, floor
