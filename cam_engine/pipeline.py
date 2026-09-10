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
from .errors import CamError, INVALID_TOOLPATH
from .features import FeatureRecognizer, MachiningFeature
from .operations import OperationPlanner, PlannedOperation
from .toolpaths.strategies import StrategyEngine, StrategyContext
from .toolpaths.semantic import Toolpath
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

    stock_min = np.array([bmin[0] - margin_x, bmin[1] - margin_y, bmin[2] - margin_z_bottom])
    stock_max = np.array([bmax[0] + margin_x, bmax[1] + margin_y, bmax[2] + margin_z_top])
    stock = Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)

    wo_enum = WorkOffset.G54
    for w in WorkOffset:
        if w.value == work_offset:
            wo_enum = w
            break

    setup = Setup(
        id="setup_001",
        name="Setup 1 - Top Milling",
        model_to_setup=setup_transform,
        work_offset=wo_enum,
        stock=stock,
    )

    machine = get_default_machine()
    tools = get_default_tools()
    material = DEFAULT_MATERIALS.get(material_name, DEFAULT_MATERIALS["aluminum_6061"])

    ctx = PlanningContext(
        model=imported,
        setups=[setup],
        machine=machine,
        tools=tools,
        material=material,
    )

    # 3–6: Process each setup
    all_features: list[MachiningFeature] = []
    all_operations: list[PlannedOperation] = []
    all_toolpaths: list[Toolpath] = []
    all_validation_reports: list[ValidationReport] = []
    all_errors: list[CamError] = []

    for s in ctx.setups:
        # 3. Feature Recognition
        setup_shape = ctx.model_shape_in_setup(s)
        stock_top = float(s.stock.bounds_max[2])
        recognizer = FeatureRecognizer(setup_shape, stock_top_z=stock_top)
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
        stock_voxels = VoxelStock.from_bounds(
            s.stock.bounds_min, s.stock.bounds_max, resolution,
        )

        clearance_z = float(s.stock.bounds_max[2]) + machine.safe_retract_height

        strat_ctx = StrategyContext(
            context=ctx,
            setup=s,
            mesh=mesh,
            stock_voxels=stock_voxels,
            clearance_z=clearance_z,
            finish_allowance_wall=0.3,
            finish_allowance_floor=0.3,
        )
        engine = StrategyEngine(strat_ctx)

        setup_toolpaths: list[Toolpath] = []
        for op in operations:
            try:
                tp = engine.generate(op)
                all_toolpaths.append(tp)
                setup_toolpaths.append(tp)
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
        for tp in setup_toolpaths:
            report = validator.validate(tp)
            all_validation_reports.append(report)
            if not report.passed:
                for err in report.errors:
                    all_errors.append(err)

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
