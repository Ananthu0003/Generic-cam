"""FastAPI REST API routes for Generic CAM."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..context import (
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Material,
    DEFAULT_MATERIALS,
    Tool,
    ToolType,
    Units,
    WorkOffset,
)
from ..features import (
    FeatureRecognizer,
    FeatureType,
    Accessibility,
    MachiningFeature,
)
from ..operations import OperationPlanner, PlannedOperation, OpPurpose
from ..machinability import CuttingParameters
from ..toolpaths.strategies import StrategyEngine, StrategyContext
from ..toolpaths.semantic import Toolpath, MotionSegment, MotionType
from ..geometry.step_import import import_step, ImportedModel
from ..geometry.mesh import TriangleMesh
from ..geometry.voxel import VoxelStock
from ..post import get_post_processor
from ..errors import CamError
from ..demo_models import (
    create_prismatic_bracket_mesh,
    get_default_tools,
    get_default_machine,
    DemoCADModel,
)
from .models import (
    ModelInfoResponse,
    MeshData,
    BoundingBox,
    FeatureItem,
    RecognizeFeaturesResponse,
    ToolItem,
    MachineItem,
    StockConfig,
    PlanOpsRequest,
    PlannedOpItem,
    PlanOpsResponse,
    GenerateToolpathsRequest,
    GenerateToolpathsResponse,
    ToolpathItem,
    MotionSegmentItem,
    PostProcessRequest,
    PostProcessResponse,
)

router = APIRouter(prefix="/api", tags=["cam"])


class SessionState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.model_name: str = "Demo Prismatic Bracket"
        self.raw_mesh: Optional[DemoCADModel] = None
        self.imported_model: Optional[ImportedModel] = None
        self.features: list[MachiningFeature] = []
        self.tools: list[Tool] = get_default_tools()
        self.machine = get_default_machine()
        self.planned_ops: list[PlannedOperation] = []
        self.toolpaths: list[Toolpath] = []
        self.stock_config = StockConfig()
        self.load_demo()

    def load_demo(self):
        self.raw_mesh = create_prismatic_bracket_mesh()
        self.model_name = self.raw_mesh.name
        self.imported_model = None
        self.features = []
        self.planned_ops = []
        self.toolpaths = []


session = SessionState()


def _mesh_to_response(demo: DemoCADModel) -> MeshData:
    bmin = demo.bounds_min
    bmax = demo.bounds_max
    size = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]]
    return MeshData(
        vertices=demo.vertices,
        normals=demo.normals,
        indices=demo.indices,
        triangle_count=len(demo.indices) // 3,
        face_count=len(demo.faces_metadata),
        bounding_box=BoundingBox(min=bmin, max=bmax, size=size),
    )


@router.get("/model/demo", response_model=ModelInfoResponse)
def get_demo_model():
    """Loads and returns the demo CAD model mesh and bounding box."""
    session.load_demo()
    return ModelInfoResponse(
        name=session.model_name,
        units="mm",
        mesh=_mesh_to_response(session.raw_mesh),
        features_count=0,
    )


@router.post("/model/upload", response_model=ModelInfoResponse)
async def upload_model_file(file: UploadFile = File(...)):
    """Uploads a CAD file (.step, .stp, .stl), processes it, and caches the model."""
    filename = file.filename.lower()
    if not (filename.endswith(".step") or filename.endswith(".stp") or filename.endswith(".stl")):
        raise HTTPException(status_code=400, detail="Supported formats: .step, .stp, .stl")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if filename.endswith(".stl"):
            from ..geometry.stl_import import import_stl, STLImportError
            try:
                mesh = import_stl(tmp_path)
            except STLImportError as e:
                raise HTTPException(status_code=422, detail=f"Failed to process STL file: {e}")

            vertices = []
            normals = []
            indices = []
            faces_meta = []

            for i, tri in enumerate(mesh.triangles):
                base_idx = len(vertices) // 3
                v0, v1, v2 = tri[0], tri[1], tri[2]
                normal = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-12:
                    normal = normal / norm_len
                else:
                    normal = np.array([0.0, 0.0, 1.0])

                for p in (v0, v1, v2):
                    vertices.extend([float(p[0]), float(p[1]), float(p[2])])
                    normals.extend([float(normal[0]), float(normal[1]), float(normal[2])])

                indices.extend([base_idx, base_idx + 1, base_idx + 2])
                faces_meta.append({"id": i, "kind": "mesh_facet", "area": 0.5 * norm_len})

            bmin = [float(x) for x in mesh.bounds_min]
            bmax = [float(x) for x in mesh.bounds_max]

            session.raw_mesh = DemoCADModel(
                name=file.filename,
                description="Imported STL Mesh",
                bounds_min=bmin,
                bounds_max=bmax,
                vertices=vertices,
                normals=normals,
                indices=indices,
                faces_metadata=faces_meta,
            )
            session.imported_model = None
            session.model_name = file.filename
            session.features = []
            session.planned_ops = []
            session.toolpaths = []

            return ModelInfoResponse(
                name=session.model_name,
                units="mm",
                mesh=_mesh_to_response(session.raw_mesh),
                features_count=0,
            )
        else:
            imported = import_step(tmp_path)
            session.imported_model = imported
            session.model_name = file.filename
            session.features = []
            session.planned_ops = []
            session.toolpaths = []

            vertices = []
            normals_list = []
            indices = []
            faces_meta = []

            shape_faces = imported.shape.faces()
            for f in shape_faces:
                if not len(f.triangles):
                    continue
                base_idx = len(vertices) // 3
                tris = f.triangles
                n_tris = len(tris)
                for t in tris:
                    for p in t:
                        vertices.extend([float(p[0]), float(p[1]), float(p[2])])
                        norm = f.oriented_normal
                        normals_list.extend([float(norm[0]), float(norm[1]), float(norm[2])])
                for i in range(n_tris * 3):
                    indices.append(base_idx + i)

                faces_meta.append({
                    "id": f.index,
                    "kind": f.surface.kind.value,
                    "area": float(f.area),
                })

            bmin = [float(x) for x in imported.original_bounds[0]]
            bmax = [float(x) for x in imported.original_bounds[1]]

            session.raw_mesh = DemoCADModel(
                name=file.filename,
                description="Imported STEP Model",
                bounds_min=bmin,
                bounds_max=bmax,
                vertices=vertices,
                normals=normals_list,
                indices=indices,
                faces_metadata=faces_meta,
            )

            return ModelInfoResponse(
                name=session.model_name,
                units="mm",
                mesh=_mesh_to_response(session.raw_mesh),
                features_count=0,
            )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process file: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/tools-and-machines")
def get_tools_and_machines():
    """Returns available tools catalog and default machine configuration."""
    tool_items = [
        ToolItem(
            id=t.id,
            tool_number=t.tool_number,
            name=f"Tool {t.tool_number}: {t.type.value.replace('_', ' ').title()} (Ø{t.diameter}mm)",
            tool_type=t.type.value if hasattr(t.type, "value") else str(t.type),
            diameter=t.diameter,
            corner_radius=t.corner_radius,
            flute_length=t.flute_length,
            overall_length=t.overall_length,
            flutes=t.flutes,
            max_rpm=12000.0,
            max_feed=3000.0,
        )
        for t in session.tools
    ]

    machine = session.machine
    machine_item = MachineItem(
        id=machine.id,
        name=machine.name,
        machine_type=machine.machine_type.value,
        controller=machine.controller,
        spindle_max_rpm=machine.spindle_max_rpm,
        max_spindle_power_kw=machine.max_spindle_power_kw,
        axes={
            k: {
                "travel_min": v.travel_min,
                "travel_max": v.travel_max,
                "rapid_rate": v.rapid_rate,
                "max_feed": v.max_feed,
            }
            for k, v in machine.axes.items()
        },
    )

    return {"tools": tool_items, "machine": machine_item}


@router.post("/recognize-features", response_model=RecognizeFeaturesResponse)
def recognize_features():
    """Runs geometric feature recognition on the active part."""
    if session.imported_model is not None:
        bmin, bmax = session.imported_model.original_bounds
        stock_top = float(bmax[2]) + session.stock_config.margin_z_top
        rec = FeatureRecognizer(session.imported_model.shape, stock_top_z=stock_top)
        session.features = rec.recognize()
    else:
        # Fallback demo model features
        session.features = [
            MachiningFeature(
                id="feat_facing_001",
                type=FeatureType.FACING_REGION,
                face_indices=[5],
                bounds_min=np.array([0.0, 0.0, 25.0]),
                bounds_max=np.array([100.0, 60.0, 25.0]),
                depth=session.stock_config.margin_z_top,
                top_z=25.0 + session.stock_config.margin_z_top,
                floor_z=25.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=False,
                notes={"strategy": "zigzag", "stock_removal": f"{session.stock_config.margin_z_top}mm"},
            ),
            MachiningFeature(
                id="feat_pocket_001",
                type=FeatureType.POCKET,
                face_indices=[8, 9],
                bounds_min=np.array([15.0, 15.0, 15.0]),
                bounds_max=np.array([55.0, 45.0, 25.0]),
                depth=10.0,
                top_z=25.0,
                floor_z=15.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=True,
                notes={"width": 40.0, "length": 30.0, "corner_radius": 3.0},
            ),
            MachiningFeature(
                id="feat_step_001",
                type=FeatureType.STEP,
                face_indices=[6, 7],
                bounds_min=np.array([80.0, 0.0, 17.0]),
                bounds_max=np.array([100.0, 60.0, 25.0]),
                depth=8.0,
                top_z=25.0,
                floor_z=17.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=False,
                notes={"open_sides": ["+X", "-Y", "+Y"]},
            ),
            MachiningFeature(
                id="feat_hole_001",
                type=FeatureType.THROUGH_HOLE,
                face_indices=[10],
                bounds_min=np.array([64.0, 24.0, 0.0]),
                bounds_max=np.array([76.0, 36.0, 25.0]),
                depth=25.0,
                top_z=25.0,
                floor_z=0.0,
                diameter=12.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=True,
                notes={"center": [70.0, 30.0], "type": "through_bore"},
            ),
        ]

    resp_items = [
        FeatureItem(
            id=f.id,
            type=f.type.value if hasattr(f.type, "value") else str(f.type),
            bounds_min=[float(x) for x in f.bounds_min],
            bounds_max=[float(x) for x in f.bounds_max],
            depth=float(f.depth),
            top_z=float(f.top_z),
            floor_z=float(f.floor_z),
            diameter=float(f.diameter) if f.diameter is not None else None,
            radius=float(f.radius) if f.radius is not None else None,
            accessibility=f.accessibility.value if hasattr(f.accessibility, "value") else str(f.accessibility),
            is_concave=bool(f.is_concave),
            face_indices=f.face_indices,
            notes=f.notes,
        )
        for f in session.features
    ]

    return RecognizeFeaturesResponse(features=resp_items, count=len(resp_items))


def _tool_name(t: Tool) -> str:
    return f"{t.id} ({t.type.value.replace('_', ' ').title()} Ø{t.diameter}mm)"


def _build_context_and_setup(stock_cfg: StockConfig):
    bmin = session.raw_mesh.bounds_min if session.raw_mesh else [0, 0, 0]
    bmax = session.raw_mesh.bounds_max if session.raw_mesh else [100, 60, 25]
    stock_min = np.array([bmin[0] - stock_cfg.margin_x, bmin[1] - stock_cfg.margin_y, bmin[2] - stock_cfg.margin_z_bottom])
    stock_max = np.array([bmax[0] + stock_cfg.margin_x, bmax[1] + stock_cfg.margin_y, bmax[2] + stock_cfg.margin_z_top])
    stock = Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)

    setup = Setup(
        id="setup_001",
        name="Setup 1 - Top Milling",
        model_to_setup=np.eye(4),
        work_offset=WorkOffset.G54,
        stock=stock,
    )

    imported = session.imported_model or _make_dummy_imported()
    ctx = PlanningContext(
        model=imported,
        setups=[setup],
        machine=session.machine,
        tools=session.tools,
        material=DEFAULT_MATERIALS.get("aluminum_6061", DEFAULT_MATERIALS["aluminum_6061"]),
    )
    return ctx, setup


@router.post("/plan-operations", response_model=PlanOpsResponse)
def plan_operations(req: PlanOpsRequest):
    """Plans ordered machining operations based on recognized features and tooling."""
    session.stock_config = req.stock

    if not session.features:
        recognize_features()

    try:
        ctx, setup = _build_context_and_setup(req.stock)
        selected_features = [
            f for f in session.features
            if not req.selected_features or f.id in req.selected_features
        ]

        planner = OperationPlanner(ctx, setup, selected_features)
        ops = planner.plan()
        session.planned_ops = ops

        op_items: list[PlannedOpItem] = [
            PlannedOpItem(
                id=op.id,
                purpose=op.purpose.value if hasattr(op.purpose, "value") else str(op.purpose),
                feature_id=op.feature.id,
                tool_id=op.tool.id,
                tool_name=_tool_name(op.tool),
                feed_rate=op.params.feed_rate,
                spindle_rpm=op.params.spindle_rpm,
                stepover_mm=op.params.stepover_mm,
                stepdown_mm=op.params.depth_of_cut_mm,
                notes=op.notes,
            )
            for op in ops
        ]

        stock_min = [float(x) for x in setup.stock.bounds_min]
        stock_max = [float(x) for x in setup.stock.bounds_max]
        stock_size = [stock_max[i] - stock_min[i] for i in range(3)]

        return PlanOpsResponse(
            operations=op_items,
            stock_bounds=BoundingBox(min=stock_min, max=stock_max, size=stock_size),
        )
    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")


@router.post("/generate-toolpaths", response_model=GenerateToolpathsResponse)
def generate_toolpaths(req: GenerateToolpathsRequest):
    """Generates continuous semantic toolpaths with rapid, cut, plunge, and retract motion segments."""
    try:
        if not session.planned_ops:
            plan_operations(PlanOpsRequest(stock=req.stock))

        ctx, setup = _build_context_and_setup(req.stock)

        if session.imported_model:
            setup_shape = ctx.model_shape_in_setup(setup)
            mesh = TriangleMesh.from_faces(setup_shape.faces())
        else:
            demo = session.raw_mesh or create_prismatic_bracket_mesh()
            tri_pts = np.array(demo.vertices).reshape(-1, 3)
            indices = np.array(demo.indices).reshape(-1, 3)
            tris = tri_pts[indices]
            mesh = TriangleMesh(triangles=tris)

        dims = setup.stock.bounds_max - setup.stock.bounds_min
        resolution = max(0.5, min(dims) / 50.0)
        stock_voxels = VoxelStock.from_bounds(
            setup.stock.bounds_min, setup.stock.bounds_max, resolution,
        )
        clearance_z = float(setup.stock.bounds_max[2]) + session.machine.safe_retract_height

        strat_ctx = StrategyContext(
            context=ctx,
            setup=setup,
            mesh=mesh,
            stock_voxels=stock_voxels,
            clearance_z=clearance_z,
            finish_allowance_wall=0.3,
            finish_allowance_floor=0.3,
        )
        engine = StrategyEngine(strat_ctx)

        toolpaths: list[Toolpath] = []
        total_cut_len = 0.0
        total_rapid_len = 0.0

        for op in session.planned_ops:
            tp = engine.generate(op)
            toolpaths.append(tp)
            total_cut_len += tp.cutting_length()
            total_rapid_len += tp.rapid_length()

        session.toolpaths = toolpaths

        resp_tps: list[ToolpathItem] = []
        for tp in toolpaths:
            seg_items = [
                MotionSegmentItem(
                    motion_type=s.motion_type.value if hasattr(s.motion_type, "value") else str(s.motion_type),
                    start=[float(x) for x in s.start],
                    end=[float(x) for x in s.end],
                    feed=float(s.feed) if s.feed is not None else None,
                    spindle=float(s.spindle) if s.spindle is not None else None,
                    arc_center=[float(x) for x in s.arc_center] if s.arc_center is not None else None,
                    arc_ccw=bool(s.arc_ccw),
                    tool_id=s.tool_id,
                    operation_id=s.operation_id,
                )
                for s in tp.segments
            ]
            resp_tps.append(
                ToolpathItem(
                    operation_id=tp.operation_id,
                    purpose=tp.purpose,
                    tool_id=tp.tool_id,
                    segment_count=len(seg_items),
                    cutting_length_mm=tp.cutting_length(),
                    rapid_length_mm=tp.rapid_length(),
                    segments=seg_items,
                )
            )

        est_seconds = (total_cut_len / 1500.0) * 60.0 + (total_rapid_len / 5000.0) * 60.0

        return GenerateToolpathsResponse(
            toolpaths=resp_tps,
            total_cutting_length_mm=total_cut_len,
            total_rapid_length_mm=total_rapid_len,
            estimated_time_seconds=est_seconds,
        )
    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")


@router.post("/generate-gcode", response_model=PostProcessResponse)
def generate_gcode(req: PostProcessRequest):
    """Post-processes generated toolpaths into controller-specific CNC G-code."""
    try:
        if not session.toolpaths:
            generate_toolpaths(GenerateToolpathsRequest(stock=session.stock_config))

        post = get_post_processor(controller=req.controller, work_offset=req.work_offset)
        result = post.post_process(session.toolpaths, program_name=req.program_name)

        return PostProcessResponse(
            controller=req.controller,
            gcode=result.gcode,
            line_count=result.line_count,
            cycle_time_seconds=result.total_time_seconds,
            rapid_time_seconds=result.rapid_time_seconds,
            cut_time_seconds=result.cut_time_seconds,
            total_rapid_dist_mm=result.total_rapid_dist_mm,
            total_cut_dist_mm=result.total_cut_dist_mm,
            tool_changes=result.tool_changes,
            work_offset=result.work_offset,
        )
    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")


@router.get("/setup-sheet")
def get_setup_sheet():
    """Generates a setup sheet for shop floor documentation."""
    from ..setup_sheet import generate_setup_sheet, format_setup_sheet

    if not session.raw_mesh:
        raise HTTPException(status_code=400, detail="No model loaded")

    if not session.planned_ops:
        plan_operations(PlanOpsRequest(stock=session.stock_config))

    if not session.toolpaths:
        generate_toolpaths(GenerateToolpathsRequest(stock=session.stock_config))

    ctx, setup = _build_context_and_setup(session.stock_config)
    post = get_post_processor(controller="grbl", work_offset="G54")
    post_result = post.post_process(session.toolpaths, program_name="SETUP_SHEET")

    sheet = generate_setup_sheet(
        program_name=session.model_name.upper().replace(" ", "_"),
        part_name=session.model_name,
        planning_ctx=ctx,
        setup=setup,
        features=session.features,
        operations=session.planned_ops,
        toolpaths=session.toolpaths,
        post_result=post_result,
    )

    return {"text": format_setup_sheet(sheet), "sheet": {
        "program_name": sheet.program_name,
        "part_name": sheet.part_name,
        "material": sheet.material,
        "machine": sheet.machine,
        "stock_dimensions": sheet.stock_dimensions,
        "tool_count": len(sheet.tool_list),
        "operation_count": len(sheet.operations),
        "estimated_cycle_time": sheet.estimated_cycle_time,
        "notes": sheet.notes,
    }}


def _make_dummy_imported():
    from ..geometry.step_import import ImportedModel
    from ..geometry.topology import Shape
    return ImportedModel(
        shape=Shape.__new__(Shape),
        declared_units="mm",
        original_bounds=(np.zeros(3), np.ones(3)),
    )
