"""FastAPI REST API routes for VexCAM."""

from __future__ import annotations

import io
import zipfile
import tempfile
import uuid
import time
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Response


from ..context import (
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Fixture,
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
from ..geometry.topology import Shape, SurfaceKind
from ..geometry.mesh import TriangleMesh
from ..geometry.voxel import VoxelStock
from ..post import get_post_processor
from ..errors import CamError
from ..demo_models import (
    create_prismatic_bracket_shape,
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
    AddToolRequest,
    AutoSuggestToolRequest,
    MachineItem,
    StockConfig,
    OrientModelRequest,
    SetupItem,
    AddSetupRequest,
    UpdateSetupRequest,
    SetupsListResponse,
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


def _euler_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    R = Rz @ Ry @ Rx
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def _create_stock_from_config(cfg: StockConfig, bmin: list[float] | np.ndarray, bmax: list[float] | np.ndarray) -> Stock:
    bmin = np.asarray(bmin, dtype=float)
    bmax = np.asarray(bmax, dtype=float)
    mode = getattr(cfg, "stock_mode", "relative_box") or "relative_box"
    off_x = getattr(cfg, "offset_x", 0.0) or 0.0
    off_y = getattr(cfg, "offset_y", 0.0) or 0.0
    off_z = getattr(cfg, "offset_z", 0.0) or 0.0

    if mode in ("relative_cylinder", "cylinder", "fixed_cylinder"):
        cx = 0.5 * (bmin[0] + bmax[0]) + off_x
        cy = 0.5 * (bmin[1] + bmax[1]) + off_y
        rx = 0.5 * (bmax[0] - bmin[0])
        ry = 0.5 * (bmax[1] - bmin[1])
        part_r = float(np.hypot(rx, ry))

        top_margin = getattr(cfg, "cylinder_margin_axial_top", 1.0) if getattr(cfg, "cylinder_margin_axial_top", None) is not None else 1.0
        bot_margin = getattr(cfg, "cylinder_margin_axial_bot", 3.0) if getattr(cfg, "cylinder_margin_axial_bot", None) is not None else 3.0
        z_top = float(bmax[2] + top_margin + off_z)

        if mode == "fixed_cylinder":
            # Exact purchased bar diameter and length
            dia = float(cfg.cylinder_diameter if (cfg.cylinder_diameter and cfg.cylinder_diameter > 0) else (2.0 * (part_r + 2.5)))
            radius = dia / 2.0
            if cfg.cylinder_length and cfg.cylinder_length > 0:
                z_bot = z_top - float(cfg.cylinder_length)
            else:
                z_bot = float(bmin[2] - bot_margin + off_z)
        else:
            # relative_cylinder: auto-calculate diameter from part radius + radial margin
            radial_margin = getattr(cfg, "cylinder_margin_radial", 2.5) if getattr(cfg, "cylinder_margin_radial", None) is not None else 2.5
            dia = float(cfg.cylinder_diameter if (cfg.cylinder_diameter and cfg.cylinder_diameter > 0) else (2.0 * (part_r + radial_margin)))
            radius = dia / 2.0
            z_bot = float(bmin[2] - bot_margin + off_z)

        axis_str = str(getattr(cfg, "cylinder_axis", "Z") or "Z").upper()
        if axis_str == "X":
            axis_vec = np.array([1.0, 0.0, 0.0])
        elif axis_str == "Y":
            axis_vec = np.array([0.0, 1.0, 0.0])
        else:
            axis_vec = np.array([0.0, 0.0, 1.0])

        return Stock.from_cylinder(center_xy=np.array([cx, cy]), z_min=z_bot, z_max=z_top, radius=radius, axis=axis_vec)

    elif mode == "fixed_box":
        part_sx = float(bmax[0] - bmin[0])
        part_sy = float(bmax[1] - bmin[1])
        part_sz = float(bmax[2] - bmin[2])

        fx = float(cfg.fixed_size_x if (cfg.fixed_size_x and cfg.fixed_size_x > 0) else (part_sx + 2 * cfg.margin_x))
        fy = float(cfg.fixed_size_y if (cfg.fixed_size_y and cfg.fixed_size_y > 0) else (part_sy + 2 * cfg.margin_y))
        fz = float(cfg.fixed_size_z if (cfg.fixed_size_z and cfg.fixed_size_z > 0) else (part_sz + cfg.margin_z_top + cfg.margin_z_bottom))

        cx = 0.5 * (bmin[0] + bmax[0]) + off_x
        cy = 0.5 * (bmin[1] + bmax[1]) + off_y
        z_top = float(bmax[2] + cfg.margin_z_top + off_z)
        z_bot = z_top - fz

        stock_min = np.array([cx - fx / 2.0, cy - fy / 2.0, z_bot])
        stock_max = np.array([cx + fx / 2.0, cy + fy / 2.0, z_top])
        return Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)

    else:
        # relative_box
        stock_min = np.array([
            bmin[0] - cfg.margin_x + off_x,
            bmin[1] - cfg.margin_y + off_y,
            bmin[2] - cfg.margin_z_bottom + off_z,
        ])
        stock_max = np.array([
            bmax[0] + cfg.margin_x + off_x,
            bmax[1] + cfg.margin_y + off_y,
            bmax[2] + cfg.margin_z_top + off_z,
        ])
        return Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)


def _shape_to_demo_cad_model(name: str, shape: Shape, desc: str = "Imported Model") -> DemoCADModel:
    vertices = []
    normals_list = []
    indices = []
    faces_meta = []
    shape_faces = shape.faces()
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
    bmin, bmax = shape.bounding_box()
    return DemoCADModel(
        name=name,
        description=desc,
        bounds_min=[float(x) for x in bmin],
        bounds_max=[float(x) for x in bmax],
        vertices=vertices,
        normals=normals_list,
        indices=indices,
        faces_metadata=faces_meta,
    )


class SetupConfig:
    def __init__(self, id: str, name: str, work_offset: str = "G54",
                 rotation_deg: Optional[list[float]] = None,
                 stock_cfg: Optional[StockConfig] = None):
        self.id = id
        self.name = name
        self.work_offset = work_offset
        self.rotation_deg = rotation_deg or [0.0, 0.0, 0.0]
        self.stock_cfg = stock_cfg or StockConfig(work_offset=work_offset)
        self.feature_ids: list[str] = []
        self.operations_count: int = 0


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
        self.setups: list[SetupConfig] = [
            SetupConfig(id="setup_001", name="Setup 1 - Top Milling", work_offset="G54", rotation_deg=[0.0, 0.0, 0.0]),
        ]
        self.active_setup_id: str = "setup_001"
        self.load_demo()

    def load_demo(self):
        try:
            self.imported_model = create_prismatic_bracket_shape()
            self.model_name = "Prismatic CAM Test Bracket"
            self.raw_mesh = _shape_to_demo_cad_model(self.model_name, self.imported_model.shape, "Standard 100x60x25mm aerospace bracket with top facing, 40x30mm pocket, Ø12mm bore, and 8mm stepdown.")
        except Exception:
            self.raw_mesh = create_prismatic_bracket_mesh()
            self.model_name = self.raw_mesh.name
            self.imported_model = None
        self.features = []
        self.planned_ops = []
        self.toolpaths = []
        self.setups = [
            SetupConfig(id="setup_001", name="Setup 1 - Top Milling", work_offset="G54", rotation_deg=[0.0, 0.0, 0.0]),
        ]
        self.active_setup_id = "setup_001"


_sessions: dict[str, tuple[SessionState, float]] = {}
_default_session = SessionState()
_SESSION_TTL_SECONDS = 3600  # 1 hour


def _evict_expired_sessions():
    now = time.time()
    expired = [sid for sid, (_, ts) in _sessions.items() if now - ts > _SESSION_TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]


def _get_session(request: Request) -> SessionState:
    _evict_expired_sessions()
    sid = request.headers.get("X-Session-ID", "")
    if not sid:
        return _default_session
    if sid not in _sessions:
        _sessions[sid] = (SessionState(), time.time())
    state, _ = _sessions[sid]
    _sessions[sid] = (state, time.time())
    return state


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
def get_demo_model(request: Request):
    """Loads and returns the demo CAD model mesh and bounding box."""
    sess = _get_session(request)
    sess.load_demo()
    return ModelInfoResponse(
        name=sess.model_name,
        units="mm",
        mesh=_mesh_to_response(sess.raw_mesh),
        features_count=0,
    )


@router.post("/model/upload", response_model=ModelInfoResponse)
async def upload_model_file(request: Request, file: UploadFile = File(...)):
    """Uploads a CAD file (.step, .stp, .stl), processes it, and caches the model."""
    sess = _get_session(request)
    orig_filename = file.filename or "Uploaded Part"
    filename_lower = orig_filename.lower()
    if not (filename_lower.endswith(".step") or filename_lower.endswith(".stp") or filename_lower.endswith(".stl")):
        raise HTTPException(status_code=400, detail="Supported formats: .step, .stp, .stl")

    suffix = Path(orig_filename).suffix if Path(orig_filename).suffix else ".step"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if filename_lower.endswith(".stl"):
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

            sess.raw_mesh = DemoCADModel(
                name=orig_filename,
                description="Imported STL Mesh",
                bounds_min=bmin,
                bounds_max=bmax,
                vertices=vertices,
                normals=normals,
                indices=indices,
                faces_metadata=faces_meta,
            )
            sess.imported_model = None
            sess.model_name = orig_filename
            sess.features = []
            sess.planned_ops = []
            sess.toolpaths = []

            return ModelInfoResponse(
                name=sess.model_name,
                units="mm",
                mesh=_mesh_to_response(sess.raw_mesh),
                features_count=0,
            )
        else:
            imported = import_step(tmp_path)
            sess.imported_model = imported
            sess.model_name = orig_filename
            sess.features = []
            sess.planned_ops = []
            sess.toolpaths = []
            sess.raw_mesh = _shape_to_demo_cad_model(orig_filename, imported.shape, "Imported STEP Model")

            return ModelInfoResponse(
                name=sess.model_name,
                units="mm",
                mesh=_mesh_to_response(sess.raw_mesh),
                features_count=0,
            )
    except CamError as e:
        raise HTTPException(status_code=422, detail=f"{e.code}: {e.message}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process file: {str(e)}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@router.post("/model/orient", response_model=ModelInfoResponse)
def orient_model(request: Request, req: OrientModelRequest):
    """Interactively rotates the active CAD model around X, Y, or Z axis or aligns a selected face to +Z."""
    sess = _get_session(request)
    if sess.imported_model is not None:
        if req.align_face_id is not None:
            sess.imported_model.shape = sess.imported_model.shape.align_face_to_axis(req.align_face_id, np.array([0.0, 0.0, 1.0]))
        else:
            sess.imported_model.shape = sess.imported_model.shape.rotate(req.axis, req.angle_deg)
        bmin, bmax = sess.imported_model.shape.bounding_box()
        sess.imported_model.original_bounds = (bmin, bmax)
        sess.raw_mesh = _shape_to_demo_cad_model(sess.model_name, sess.imported_model.shape, "Oriented Model")
    elif sess.raw_mesh is not None:
        # Mesh rotation fallback
        pts = np.array(sess.raw_mesh.vertices).reshape(-1, 3)
        norms = np.array(sess.raw_mesh.normals).reshape(-1, 3)
        rad = np.radians(req.angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        if req.axis.upper() == "X":
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif req.axis.upper() == "Y":
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center = 0.5 * (np.array(sess.raw_mesh.bounds_min) + np.array(sess.raw_mesh.bounds_max))
        rotated_pts = (pts - center) @ R.T + center
        rotated_norms = norms @ R.T
        bmin = [float(x) for x in rotated_pts.min(axis=0)]
        bmax = [float(x) for x in rotated_pts.max(axis=0)]
        sess.raw_mesh.vertices = rotated_pts.flatten().tolist()
        sess.raw_mesh.normals = rotated_norms.flatten().tolist()
        sess.raw_mesh.bounds_min = bmin
        sess.raw_mesh.bounds_max = bmax

    # Invalidate previous cached features/ops/toolpaths
    sess.features = []
    sess.planned_ops = []
    sess.toolpaths = []

    return ModelInfoResponse(
        name=sess.model_name,
        units="mm",
        mesh=_mesh_to_response(sess.raw_mesh),
        features_count=0,
    )


# -----------------------------------------------------------------------------
# Multi-Setup Management Routes
# -----------------------------------------------------------------------------

@router.get("/setups", response_model=SetupsListResponse)
def get_setups(request: Request):
    """List all configured machining setups with calculated transformed bounds."""
    sess = _get_session(request)
    items = []
    for s in sess.setups:
        rot_mat = _euler_to_matrix(*s.rotation_deg)
        if sess.imported_model:
            shape_in_setup = sess.imported_model.shape.transformed(rot_mat)
            bmin, bmax = shape_in_setup.bounding_box()
        elif sess.raw_mesh and sess.raw_mesh.bounding_box:
            bmin = sess.raw_mesh.bounds_min
            bmax = sess.raw_mesh.bounds_max
        else:
            bmin = [0.0, 0.0, 0.0]
            bmax = [100.0, 60.0, 25.0]

        cfg = s.stock_cfg or StockConfig(work_offset=s.work_offset)
        stock = _create_stock_from_config(cfg, bmin, bmax)
        s_min = [float(x) for x in stock.bounds_min]
        s_max = [float(x) for x in stock.bounds_max]
        s_size = [float(s_max[i] - s_min[i]) for i in range(3)]
        stk_bounds = BoundingBox(min=s_min, max=s_max, size=s_size)

        items.append(SetupItem(
            id=s.id,
            name=s.name,
            work_offset=s.work_offset,
            rotation_deg=s.rotation_deg,
            stock=cfg,
            stock_bounds=stk_bounds,
            feature_ids=s.feature_ids,
            operations_count=s.operations_count,
            is_active=(s.id == sess.active_setup_id),
        ))
    return SetupsListResponse(setups=items, active_setup_id=sess.active_setup_id)


@router.post("/setups/add", response_model=SetupsListResponse)
def add_setup(request: Request, req: AddSetupRequest):
    """Add a new setup (e.g. OP20 Flip 180°, Side 90°, etc.)."""
    sess = _get_session(request)
    idx = len(sess.setups) + 1
    new_id = f"setup_{idx:03d}"

    rotation = req.rotation_deg or [0.0, 0.0, 0.0]
    if req.preset == "flip_x_180":
        rotation = [180.0, 0.0, 0.0]
    elif req.preset == "flip_y_180":
        rotation = [0.0, 180.0, 0.0]
    elif req.preset == "side_x_90":
        rotation = [90.0, 0.0, 0.0]
    elif req.preset == "side_y_90":
        rotation = [0.0, 90.0, 0.0]

    stk = req.stock or StockConfig(work_offset=req.work_offset)
    setup = SetupConfig(id=new_id, name=req.name, work_offset=req.work_offset, rotation_deg=rotation, stock_cfg=stk)
    sess.setups.append(setup)
    sess.active_setup_id = new_id

    return get_setups(request)


@router.post("/setups/update/{setup_id}", response_model=SetupsListResponse)
def update_setup(request: Request, setup_id: str, req: UpdateSetupRequest):
    """Update setup parameters such as name, work offset, or stock configuration."""
    sess = _get_session(request)
    target = None
    for s in sess.setups:
        if s.id == setup_id:
            target = s
            break
    if not target:
        raise HTTPException(status_code=404, detail=f"Setup {setup_id} not found")

    if req.name is not None:
        target.name = req.name
    if req.work_offset is not None:
        target.work_offset = req.work_offset
    if req.rotation_deg is not None:
        target.rotation_deg = req.rotation_deg
    if req.stock is not None:
        target.stock_cfg = req.stock

    return get_setups(request)


@router.post("/setups/active/{setup_id}", response_model=SetupsListResponse)
def set_active_setup(request: Request, setup_id: str):
    """Sets the active setup."""
    sess = _get_session(request)
    found = any(s.id == setup_id for s in sess.setups)
    if not found:
        raise HTTPException(status_code=404, detail=f"Setup {setup_id} not found")
    sess.active_setup_id = setup_id
    return get_setups(request)


@router.delete("/setups/{setup_id}", response_model=SetupsListResponse)
def delete_setup(request: Request, setup_id: str):
    """Deletes a setup if more than 1 exist."""
    sess = _get_session(request)
    if len(sess.setups) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the primary setup")
    sess.setups = [s for s in sess.setups if s.id != setup_id]
    if sess.active_setup_id == setup_id:
        sess.active_setup_id = sess.setups[0].id
    return get_setups(request)


@router.post("/setups/auto-generate", response_model=SetupsListResponse)
def auto_generate_setups(request: Request):
    """Automatically analyzes 3D model geometry to configure required multi-axis setups (Top, Bottom, Sides)."""
    sess = _get_session(request)
    setups: list[SetupConfig] = [
        SetupConfig(id="setup_001", name="Setup 1 - Top Milling", work_offset="G54", rotation_deg=[0.0, 0.0, 0.0])
    ]

    offsets = ["G55", "G56", "G57", "G58", "G59"]
    offset_idx = 0

    if sess.imported_model is not None:
        shape = sess.imported_model.shape
        faces = shape.faces()

        has_side_x = False
        has_side_y = False

        for f in faces:
            if f.surface.kind is SurfaceKind.CYLINDER and f.surface.axis is not None:
                axis = f.surface.axis / max(float(np.linalg.norm(f.surface.axis)), 1e-12)
                if abs(axis[2]) < 0.3:
                    if abs(axis[0]) >= abs(axis[1]):
                        has_side_x = True
                    else:
                        has_side_y = True
            elif f.surface.kind is SurfaceKind.PLANE and f.oriented_normal is not None:
                norm = f.oriented_normal
                if (abs(norm[0]) > 0.85 or abs(norm[1]) > 0.85) and f.area > 150.0:
                    # Significant side face
                    if abs(norm[0]) > abs(norm[1]):
                        has_side_x = True
                    else:
                        has_side_y = True

        # Standard 2-sided part: always include Bottom Flip (G55)
        if offset_idx < len(offsets):
            setups.append(SetupConfig(
                id=f"setup_{len(setups)+1:03d}",
                name="Setup 2 - Bottom Flip",
                work_offset=offsets[offset_idx],
                rotation_deg=[180.0, 0.0, 0.0]
            ))
            offset_idx += 1

        if has_side_x and offset_idx < len(offsets):
            setups.append(SetupConfig(
                id=f"setup_{len(setups)+1:03d}",
                name=f"Setup {len(setups)+1} - Side Milling (+X)",
                work_offset=offsets[offset_idx],
                rotation_deg=[0.0, 90.0, 0.0]
            ))
            offset_idx += 1

        if has_side_y and offset_idx < len(offsets):
            setups.append(SetupConfig(
                id=f"setup_{len(setups)+1:03d}",
                name=f"Setup {len(setups)+1} - End Face (+Y)",
                work_offset=offsets[offset_idx],
                rotation_deg=[90.0, 0.0, 0.0]
            ))
            offset_idx += 1
    else:
        # Fallback for demo model
        setups.append(SetupConfig(
            id="setup_002",
            name="Setup 2 - Bottom Flip",
            work_offset="G55",
            rotation_deg=[180.0, 0.0, 0.0]
        ))

    sess.setups = setups
    sess.active_setup_id = setups[0].id
    sess.features = []
    sess.planned_ops = []
    sess.toolpaths = []

    return get_setups(request)


@router.get("/tools-and-machines")
def get_tools_and_machines(request: Request):
    """Returns available tools catalog and default machine configuration."""
    sess = _get_session(request)
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
        for t in sess.tools
    ]

    machine = sess.machine
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


@router.post("/tools/add")
def add_custom_tool(request: Request, req: AddToolRequest):
    """Add a new custom tool to the active session tool library."""
    sess = _get_session(request)
    t_num = req.tool_number
    if t_num is None:
        used_numbers = [t.tool_number for t in sess.tools]
        t_num = max(used_numbers, default=0) + 1

    type_map = {
        "drill": ToolType.DRILL,
        "flat_endmill": ToolType.FLAT_ENDMILL,
        "ball_endmill": ToolType.BALL_ENDMILL,
        "bullnose_endmill": ToolType.BULLNOSE_ENDMILL,
        "face_mill": ToolType.FACE_MILL,
        "chamfer_mill": ToolType.CHAMFER_MILL,
        "countersink_tool": ToolType.COUNTERSINK_TOOL,
        "boring_bar": ToolType.BORING_BAR,
        "reamer": ToolType.REAMER,
        "tap": ToolType.TAP,
        "thread_mill": ToolType.THREAD_MILL,
        "groove_cutter": ToolType.GROOVE_CUTTER,
    }
    ttype = type_map.get(req.tool_type.lower(), ToolType.FLAT_ENDMILL)
    t_id = f"T{t_num:02d}"
    existing_ids = {t.id for t in sess.tools}
    if t_id in existing_ids:
        t_id = f"T{t_num:02d}_{len(sess.tools)+1}"

    new_tool = Tool(
        id=t_id,
        tool_number=t_num,
        type=ttype,
        diameter=float(req.diameter),
        corner_radius=float(req.corner_radius),
        flute_length=float(req.flute_length),
        overall_length=float(req.overall_length),
        flutes=int(req.flutes),
        material=req.material,
        can_plunge=True,
    )
    sess.tools.append(new_tool)
    return get_tools_and_machines(request)


@router.delete("/tools/{tool_id}")
def delete_custom_tool(request: Request, tool_id: str):
    """Remove a tool from the session tool library."""
    sess = _get_session(request)
    sess.tools = [t for t in sess.tools if t.id != tool_id]
    return get_tools_and_machines(request)


@router.post("/tools/auto-suggest")
def auto_suggest_tool(request: Request, req: AutoSuggestToolRequest):
    """Auto-generate a fitting tool for an unmachined feature and add it to library."""
    sess = _get_session(request)
    feature = next((f for f in sess.features if f.id == req.feature_id), None)
    
    if req.tool_type:
        ttype = req.tool_type
    elif feature and feature.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
        ttype = "drill"
    else:
        ttype = "flat_endmill"

    if req.diameter is not None and req.diameter > 0:
        dia = float(req.diameter)
    elif feature:
        if ttype == "drill":
            dia = float(feature.diameter or 1.0)
        else:
            dia = max(0.2, round((feature.diameter or 1.0) * 0.7, 2))
            if feature.diameter and dia >= feature.diameter:
                dia = round(feature.diameter * 0.5, 2)
    else:
        dia = 1.0
    
    depth = feature.depth if (feature and feature.depth) else 5.0
    add_req = AddToolRequest(
        tool_type=ttype,
        diameter=round(float(dia), 2),
        flute_length=round(max(depth * 1.3, 5.0), 1),
        overall_length=round(max(depth * 2.5, 35.0), 1),
        flutes=2 if ttype == "drill" else 3,
        material="carbide"
    )
    return add_custom_tool(request, add_req)



@router.post("/recognize-features", response_model=RecognizeFeaturesResponse)
def recognize_features(request: Request):
    """Runs geometric feature recognition on the active part across all setups."""
    sess = _get_session(request)
    all_features: list[MachiningFeature] = []

    if sess.imported_model is not None:
        for s in sess.setups:
            rot_mat = _euler_to_matrix(*s.rotation_deg)
            setup_shape = sess.imported_model.shape.transformed(rot_mat)
            bmin, bmax = setup_shape.bounding_box()
            stock_top = float(bmax[2]) + s.stock_cfg.margin_z_top
            rec = FeatureRecognizer(setup_shape, stock_top_z=stock_top)
            feats = rec.recognize()
            for f in feats:
                f.id = f"{s.id}_{f.id}"
                f.notes["setup_id"] = s.id
                f.notes["setup_name"] = s.name
            s.feature_ids = [f.id for f in feats]
            all_features.extend(feats)
        sess.features = all_features
    else:
        # Fallback demo model features
        s0 = sess.setups[0]
        margin_z = float(sess.stock_config.margin_z_top) if hasattr(sess.stock_config, 'margin_z_top') else 1.0
        sess.features = [
            MachiningFeature(
                id=f"{s0.id}_feat_facing_001",
                type=FeatureType.FACING_REGION,
                face_indices=[5],
                bounds_min=np.array([0.0, 0.0, 25.0]),
                bounds_max=np.array([100.0, 60.0, 25.0 + margin_z]),
                depth=margin_z,
                top_z=25.0 + margin_z,
                floor_z=25.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=False,
                notes={
                    "strategy": "zigzag",
                    "stock_removal": f"{margin_z:.1f}mm",
                    "excess_height": margin_z,
                    "setup_id": s0.id,
                    "setup_name": s0.name,
                },
            ),
            MachiningFeature(
                id=f"{s0.id}_feat_pocket_001",
                type=FeatureType.POCKET,
                face_indices=[8, 9],
                bounds_min=np.array([15.0, 15.0, 15.0]),
                bounds_max=np.array([55.0, 45.0, 25.0]),
                depth=10.0,
                top_z=25.0,
                floor_z=15.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=True,
                notes={"width": 40.0, "length": 30.0, "corner_radius": 3.0, "setup_id": s0.id, "setup_name": s0.name},
            ),
            MachiningFeature(
                id=f"{s0.id}_feat_step_001",
                type=FeatureType.STEP,
                face_indices=[6, 7],
                bounds_min=np.array([80.0, 0.0, 17.0]),
                bounds_max=np.array([100.0, 60.0, 25.0]),
                depth=8.0,
                top_z=25.0,
                floor_z=17.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=False,
                notes={"open_sides": ["+X", "-Y", "+Y"], "setup_id": s0.id, "setup_name": s0.name},
            ),
            MachiningFeature(
                id=f"{s0.id}_feat_hole_001",
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
                notes={"center": [70.0, 30.0], "type": "through_bore", "setup_id": s0.id, "setup_name": s0.name},
            ),
            MachiningFeature(
                id=f"{s0.id}_feat_contour_001",
                type=FeatureType.CONTOUR,
                face_indices=[1, 2, 3, 4],
                bounds_min=np.array([0.0, 0.0, 0.0]),
                bounds_max=np.array([100.0, 60.0, 25.0]),
                depth=25.0,
                top_z=25.0,
                floor_z=0.0,
                accessibility=Accessibility.TOOL_AXIS_OK,
                is_concave=False,
                notes={"closed": True, "type": "outer_profile", "setup_id": s0.id, "setup_name": s0.name},
            ),
        ]
        s0.feature_ids = [f.id for f in sess.features]

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
        for f in sess.features
    ]

    return RecognizeFeaturesResponse(features=resp_items, count=len(resp_items))


def _tool_name(t: Tool) -> str:
    return f"{t.id} ({t.type.value.replace('_', ' ').title()} Ø{t.diameter}mm)"


def _build_context_and_setups(sess: SessionState, stock_cfg: StockConfig):
    imported = sess.imported_model or _make_dummy_imported(sess)
    material = DEFAULT_MATERIALS.get(stock_cfg.material, DEFAULT_MATERIALS["aluminum_6061"])

    wo_map = {
        "G54": WorkOffset.G54, "G55": WorkOffset.G55, "G56": WorkOffset.G56,
        "G57": WorkOffset.G57, "G58": WorkOffset.G58, "G59": WorkOffset.G59,
    }

    setups_list: list[Setup] = []
    for sc in sess.setups:
        rot_mat = _euler_to_matrix(*sc.rotation_deg)
        if sess.imported_model:
            shape_in_setup = sess.imported_model.shape.transformed(rot_mat)
            bmin, bmax = shape_in_setup.bounding_box()
        else:
            bmin = sess.raw_mesh.bounds_min if sess.raw_mesh else [0, 0, 0]
            bmax = sess.raw_mesh.bounds_max if sess.raw_mesh else [100, 60, 25]

        cfg = sc.stock_cfg
        stock = _create_stock_from_config(cfg, bmin, bmax)
        stock_min = stock.bounds_min
        stock_max = stock.bounds_max

        # Clamping / Fixture setup modeling
        fixtures: list[Fixture] = []
        clamp_type = getattr(cfg, "clamp_type", "vise_jaws") or "vise_jaws"
        clamp_h = float(getattr(cfg, "clamp_height", cfg.margin_z_bottom) or cfg.margin_z_bottom or 3.0)
        clamp_w = float(getattr(cfg, "clamp_width", 12.0) or 12.0)

        if clamp_type == "vise_jaws" and clamp_h > 0:
            z_bot = float(stock_min[2])
            z_top = z_bot + clamp_h
            # Front Jaw (-Y)
            jaw_front = Fixture(
                id=f"{sc.id}_vise_jaw_front",
                name="Front Vise Jaw",
                bounds_min=np.array([stock_min[0] - 5.0, stock_min[1] - clamp_w, z_bot - 10.0]),
                bounds_max=np.array([stock_max[0] + 5.0, stock_min[1], z_top]),
                kind="vise_jaw",
            )
            # Rear Jaw (+Y)
            jaw_rear = Fixture(
                id=f"{sc.id}_vise_jaw_rear",
                name="Rear Vise Jaw",
                bounds_min=np.array([stock_min[0] - 5.0, stock_max[1], z_bot - 10.0]),
                bounds_max=np.array([stock_max[0] + 5.0, stock_max[1] + clamp_w, z_top]),
                kind="vise_jaw",
            )
            fixtures.extend([jaw_front, jaw_rear])
        elif clamp_type == "toe_clamps" and clamp_h > 0:
            z_bot = float(stock_min[2])
            z_top = z_bot + clamp_h
            fixtures.extend([
                Fixture(id=f"{sc.id}_toe_1", name="Toe Clamp 1",
                        bounds_min=np.array([stock_min[0] - 10, stock_min[1] - 10, z_bot]),
                        bounds_max=np.array([stock_min[0] + 5, stock_min[1] + 5, z_top]), kind="toe_clamp"),
                Fixture(id=f"{sc.id}_toe_2", name="Toe Clamp 2",
                        bounds_min=np.array([stock_max[0] - 5, stock_min[1] - 10, z_bot]),
                        bounds_max=np.array([stock_max[0] + 10, stock_min[1] + 5, z_top]), kind="toe_clamp"),
                Fixture(id=f"{sc.id}_toe_3", name="Toe Clamp 3",
                        bounds_min=np.array([stock_min[0] - 10, stock_max[1] - 5, z_bot]),
                        bounds_max=np.array([stock_min[0] + 5, stock_max[1] + 10, z_top]), kind="toe_clamp"),
                Fixture(id=f"{sc.id}_toe_4", name="Toe Clamp 4",
                        bounds_min=np.array([stock_max[0] - 5, stock_max[1] - 5, z_bot]),
                        bounds_max=np.array([stock_max[0] + 10, stock_max[1] + 10, z_top]), kind="toe_clamp"),
            ])

        wo = wo_map.get(sc.work_offset, WorkOffset.G54)
        setup = Setup(
            id=sc.id,
            name=sc.name,
            model_to_setup=rot_mat,
            work_offset=wo,
            stock=stock,
            fixtures=fixtures,
        )
        setups_list.append(setup)

    ctx = PlanningContext(
        model=imported,
        setups=setups_list,
        machine=sess.machine,
        tools=sess.tools,
        material=material,
    )
    return ctx, setups_list


@router.post("/plan-operations", response_model=PlanOpsResponse)
def plan_operations(request: Request, req: PlanOpsRequest):
    """Plans ordered machining operations based on recognized features and tooling across setups."""
    sess = _get_session(request)
    sess.stock_config = req.stock

    if not sess.features:
        recognize_features(request)

    try:
        ctx, setups = _build_context_and_setups(sess, req.stock)
        all_ops: list[PlannedOperation] = []

        all_warnings: list[dict] = []
        all_unmachined: list[dict] = []

        for s in setups:
            setup_features = [
                f for f in sess.features
                if f.notes.get("setup_id", sess.setups[0].id) == s.id
                and (not req.selected_features or f.id in req.selected_features)
            ]
            if setup_features:
                planner = OperationPlanner(ctx, s, setup_features)
                ops = planner.plan()
                all_warnings.extend(planner.warnings)
                all_unmachined.extend(planner.unmachined_features)
                for op in ops:
                    op.notes["setup_id"] = s.id
                    op.notes["setup_name"] = s.name
                    op.notes["work_offset"] = s.work_offset.value
                all_ops.extend(ops)
                for sc in sess.setups:
                    if sc.id == s.id:
                        sc.operations_count = len(ops)

        sess.planned_ops = all_ops

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
                setup_id=op.notes.get("setup_id"),
                notes=op.notes,
            )
            for op in all_ops
        ]

        s0 = setups[0]
        stock_min = [float(x) for x in s0.stock.bounds_min]
        stock_max = [float(x) for x in s0.stock.bounds_max]
        stock_size = [stock_max[i] - stock_min[i] for i in range(3)]

        return PlanOpsResponse(
            operations=op_items,
            stock_bounds=BoundingBox(min=stock_min, max=stock_max, size=stock_size),
            setup_id=sess.active_setup_id,
            warnings=all_warnings,
            unmachined_features=all_unmachined,
        )

    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")


@router.post("/generate-toolpaths", response_model=GenerateToolpathsResponse)
def generate_toolpaths(request: Request, req: GenerateToolpathsRequest):
    """Generates continuous semantic toolpaths with rapid, cut, plunge, and retract motion segments across setups."""
    sess = _get_session(request)
    try:
        if not sess.planned_ops:
            plan_operations(request, PlanOpsRequest(stock=req.stock))

        ctx, setups = _build_context_and_setups(sess, req.stock)
        all_toolpaths: list[Toolpath] = []
        total_cut_len = 0.0
        total_rapid_len = 0.0
        prev_remaining_stock = None
        prev_setup = None
        all_diagnostics = {}

        for s in setups:
            setup_ops = [op for op in sess.planned_ops if op.notes.get("setup_id") == s.id]
            if not setup_ops:
                continue

            if sess.imported_model:
                setup_shape = ctx.model_shape_in_setup(s)
                mesh = TriangleMesh.from_faces(setup_shape.faces())
            else:
                demo = sess.raw_mesh or create_prismatic_bracket_mesh()
                tri_pts = np.array(demo.vertices).reshape(-1, 3)
                indices = np.array(demo.indices).reshape(-1, 3)
                tris = tri_pts[indices]
                mesh = TriangleMesh(triangles=tris)

            dims = s.stock.bounds_max - s.stock.bounds_min
            resolution = max(0.5, min(dims) / 50.0)
            if prev_remaining_stock is not None and prev_setup is not None:
                M_curr = s.model_to_setup
                M_prev = prev_setup.model_to_setup
                try:
                    T_prev_to_curr = M_curr @ np.linalg.inv(M_prev)
                except np.linalg.LinAlgError:
                    T_prev_to_curr = np.eye(4)
                stock_voxels = prev_remaining_stock.transformed(
                    T_prev_to_curr, s.stock.bounds_min, s.stock.bounds_max, resolution
                )
            else:
                stock_voxels = VoxelStock.from_bounds(s.stock.bounds_min, s.stock.bounds_max, resolution)

            clearance_z = float(s.stock.bounds_max[2]) + sess.machine.safe_retract_height

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

            for op in setup_ops:
                tp = engine.generate(op)
                tp.metadata["setup_id"] = s.id
                tp.metadata["work_offset"] = s.work_offset.value
                all_toolpaths.append(tp)
                total_cut_len += tp.cutting_length()
                total_rapid_len += tp.rapid_length()

            prev_remaining_stock = stock_voxels
            prev_setup = s
            if strat_ctx.diagnostics:
                all_diagnostics.update(strat_ctx.diagnostics)

        sess.toolpaths = all_toolpaths

        resp_tps: list[ToolpathItem] = []
        for tp in all_toolpaths:
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
                    setup_id=tp.metadata.get("setup_id"),
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
            setup_id=sess.active_setup_id,
            diagnostics=all_diagnostics if all_diagnostics else None
        )
    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")


@router.post("/generate-gcode", response_model=PostProcessResponse)
def generate_gcode(request: Request, req: PostProcessRequest):
    """Post-processes generated toolpaths into controller-specific CNC G-code with multi-setup support."""
    sess = _get_session(request)
    try:
        if not sess.toolpaths:
            generate_toolpaths(request, GenerateToolpathsRequest(stock=sess.stock_config))

        unique_setup_ids = []
        for tp in sess.toolpaths:
            sid = tp.metadata.get("setup_id", "setup_001")
            if sid not in unique_setup_ids:
                unique_setup_ids.append(sid)

        # If separate_files requested, return per-setup data
        if req.separate_files:
            from .models import PostProcessPerSetupResponse
            setups_data = {}
            total_time = 0.0
            for i, sid in enumerate(unique_setup_ids):
                setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id", "setup_001") == sid]
                if not setup_tps:
                    continue
                matching_setup = next((s for s in sess.setups if s.id == sid), None)
                wo = matching_setup.work_offset if matching_setup else req.work_offset
                sname = matching_setup.name if matching_setup else f"Setup {i+1}"
                post = get_post_processor(controller=req.controller, work_offset=wo)
                result = post.post_process(setup_tps, program_name=f"{req.program_name}_{sid}")
                setups_data[sid] = {
                    "gcode": result.gcode,
                    "line_count": result.line_count,
                    "cycle_time_seconds": result.total_time_seconds,
                    "work_offset": wo,
                    "name": sname,
                    "tool_changes": result.tool_changes,
                    "rapid_dist_mm": result.total_rapid_dist_mm,
                    "cut_dist_mm": result.total_cut_dist_mm,
                }
                total_time += result.total_time_seconds
            return PostProcessPerSetupResponse(
                controller=req.controller,
                setups=setups_data,
                total_cycle_time_seconds=total_time,
            )

        # Original combined G-code logic
        gcode_blocks = []
        total_time = 0.0
        rapid_time = 0.0
        cut_time = 0.0
        total_rapid_dist = 0.0
        total_cut_dist = 0.0
        all_tools = []
        setups_processed = []

        per_setup_data = {}
        for i, sid in enumerate(unique_setup_ids):
            setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id", "setup_001") == sid]
            if not setup_tps:
                continue

            matching_setup = next((s for s in sess.setups if s.id == sid), None)
            wo = matching_setup.work_offset if matching_setup else req.work_offset
            sname = matching_setup.name if matching_setup else f"Setup {i+1}"
            setups_processed.append(f"{sname} ({wo})")

            post = get_post_processor(controller=req.controller, work_offset=wo)
            result = post.post_process(setup_tps, program_name=f"{req.program_name}_{sid}")

            per_setup_data[sid] = {
                "gcode": result.gcode,
                "line_count": result.line_count,
                "cycle_time_seconds": result.total_time_seconds,
                "work_offset": wo,
                "name": sname,
                "tool_changes": result.tool_changes,
                "rapid_dist_mm": result.total_rapid_dist_mm,
                "cut_dist_mm": result.total_cut_dist_mm,
            }

            if i > 0:
                retract_cmd = "G53 G00 Z0." if req.controller in ("fanuc", "haas") else "G28 G91 Z0.\nG90"
                msg_cmd = f"(MSG, FLIP PART - LOAD {sname.upper()} [{wo}])" if req.controller == "haas" else f"({sname.upper()} - FLIP / RE-FIXTURE PART)"
                gcode_blocks.append(
                    f"\n; ==========================================\n"
                    f"; (--- {sname.upper()} - FLIP / RE-FIXTURE PART ---)\n"
                    f"; ==========================================\n"
                    f"M05 (Spindle Stop)\n"
                    f"M09 (Coolant Off)\n"
                    f"{retract_cmd}\n"
                    f"{msg_cmd}\n"
                    f"M00 (Program Stop - Operator Flip/Reclamp)\n"
                    f"{wo} (Activate Setup Work Offset)\n"
                )

            gcode_blocks.append(result.gcode)
            total_time += result.total_time_seconds
            rapid_time += result.rapid_time_seconds
            cut_time += result.cut_time_seconds
            total_rapid_dist += result.total_rapid_dist_mm
            total_cut_dist += result.total_cut_dist_mm
            for t in result.tool_changes:
                if t not in all_tools:
                    all_tools.append(t)

        combined_gcode = "\n".join(gcode_blocks)
        line_count = len(combined_gcode.splitlines())

        return PostProcessResponse(
            controller=req.controller,
            gcode=combined_gcode,
            line_count=line_count,
            cycle_time_seconds=total_time,
            rapid_time_seconds=rapid_time,
            cut_time_seconds=cut_time,
            total_rapid_dist_mm=total_rapid_dist,
            total_cut_dist_mm=total_cut_dist,
            tool_changes=all_tools,
            work_offset=req.work_offset,
            setups=setups_processed,
            per_setup=per_setup_data,
        )
    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")


@router.get("/gcode/download/{setup_id}")
def download_setup_gcode(
    request: Request,
    setup_id: str,
    controller: str = "haas",
    program_name: Optional[str] = None,
):
    """Generates and downloads the G-code file for a specific setup."""
    sess = _get_session(request)
    if not sess.toolpaths:
        generate_toolpaths(request, GenerateToolpathsRequest(stock=sess.stock_config))

    target_setup = next((s for s in sess.setups if s.id == setup_id), None)
    if not target_setup:
        raise HTTPException(status_code=404, detail=f"Setup {setup_id} not found")

    setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id", "setup_001") == setup_id]

    prog_name = program_name or f"{sess.model_name.replace(' ', '_')}_{target_setup.name.replace(' ', '_')}"
    post = get_post_processor(controller=controller, work_offset=target_setup.work_offset)
    result = post.post_process(setup_tps, program_name=prog_name)

    safe_name = f"{sess.model_name.replace(' ', '_')}_{target_setup.name.replace(' ', '_')}_{target_setup.work_offset}.nc"
    return Response(
        content=result.gcode,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@router.get("/export-package")
def export_cam_package(
    request: Request,
    controller: str = "haas",
    program_name: Optional[str] = None,
):
    """Generates a complete multi-setup production package (ZIP) containing separate setup NC files,
    a master combined program, per-setup setup sheets, and the master routing sheet."""
    sess = _get_session(request)
    if not sess.toolpaths:
        generate_toolpaths(request, GenerateToolpathsRequest(stock=sess.stock_config))

    ctx, setups = _build_context_and_setups(sess, sess.stock_config)
    base_name = program_name or sess.model_name.replace(" ", "_")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        per_setup_results = {}
        for idx, s in enumerate(setups):
            setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id") == s.id]
            post = get_post_processor(controller=controller, work_offset=s.work_offset.value)
            s_prog_name = f"{base_name}_OP{(idx+1)*10}_{s.work_offset.value}"
            res = post.post_process(setup_tps, program_name=s_prog_name)
            per_setup_results[s.id] = {
                "cycle_time_seconds": res.total_time_seconds,
                "gcode": res.gcode,
            }

            nc_filename = f"{s_prog_name}.nc"
            zip_file.writestr(f"NC_Programs/{nc_filename}", res.gcode)

            # Per-setup sheet
            from ..setup_sheet import generate_setup_sheet, format_setup_sheet
            s_ops = [op for op in sess.planned_ops if op.notes.get("setup_id") == s.id]
            s_feats = [f for f in sess.features if f.notes.get("setup_id") == s.id]
            sheet = generate_setup_sheet(
                program_name=s_prog_name,
                part_name=sess.model_name,
                planning_ctx=ctx,
                setup=s,
                features=s_feats,
                operations=s_ops,
                toolpaths=setup_tps,
                post_result=res,
            )
            sheet_txt = format_setup_sheet(sheet)
            zip_file.writestr(f"Setup_Sheets/SETUP_SHEET_{s.id}_{s.work_offset.value}.txt", sheet_txt)

        # Master Combined NC program
        combined_req = PostProcessRequest(controller=controller, program_name=base_name, separate_files=False)
        combined_res = generate_gcode(request, combined_req)
        zip_file.writestr(f"NC_Programs/{base_name}_MASTER_ALL_SETUPS.nc", combined_res.gcode)

        # Master Routing Sheet
        from ..setup_sheet import generate_master_routing_sheet, format_master_routing_sheet
        master_sheet = generate_master_routing_sheet(
            program_name=f"{base_name}_MASTER",
            part_name=sess.model_name,
            planning_ctx=ctx,
            setups=setups,
            all_features=sess.features,
            all_operations=sess.planned_ops,
            all_toolpaths=sess.toolpaths,
            per_setup_results=per_setup_results,
        )
        master_txt = format_master_routing_sheet(master_sheet)
        zip_file.writestr("MASTER_ROUTING_SHEET.txt", master_txt)

    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()
    zip_name = f"{base_name}_CAM_Package.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@router.get("/setup-sheet-master")
def get_master_setup_sheet(request: Request):
    """Generates the consolidated master routing sheet across all setups."""
    from ..setup_sheet import generate_master_routing_sheet, format_master_routing_sheet
    sess = _get_session(request)

    if not sess.raw_mesh:
        raise HTTPException(status_code=400, detail="No model loaded")
    if not sess.planned_ops:
        plan_operations(request, PlanOpsRequest(stock=sess.stock_config))
    if not sess.toolpaths:
        generate_toolpaths(request, GenerateToolpathsRequest(stock=sess.stock_config))

    ctx, setups = _build_context_and_setups(sess, sess.stock_config)
    base_name = sess.model_name.replace(" ", "_")

    per_setup_results = {}
    for s in setups:
        setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id") == s.id]
        post = get_post_processor(controller=sess.machine.controller or "haas", work_offset=s.work_offset.value)
        res = post.post_process(setup_tps, program_name=f"{base_name}_{s.id}")
        per_setup_results[s.id] = {"cycle_time_seconds": res.total_time_seconds}

    master_sheet = generate_master_routing_sheet(
        program_name=f"{base_name}_MASTER",
        part_name=sess.model_name,
        planning_ctx=ctx,
        setups=setups,
        all_features=sess.features,
        all_operations=sess.planned_ops,
        all_toolpaths=sess.toolpaths,
        per_setup_results=per_setup_results,
    )

    return {
        "text": format_master_routing_sheet(master_sheet),
        "sheet": {
            "program_name": master_sheet.program_name,
            "part_name": master_sheet.part_name,
            "material": master_sheet.material,
            "machine": master_sheet.machine,
            "total_cycle_time": master_sheet.total_cycle_time,
            "setup_summaries": master_sheet.setup_summaries,
            "consolidated_tools": master_sheet.consolidated_tools,
            "notes": master_sheet.notes,
        },
    }


@router.get("/setup-sheet/{setup_id}")
def get_setup_sheet_for_setup(request: Request, setup_id: str):
    """Generates a setup sheet for a specific setup."""
    from ..setup_sheet import generate_setup_sheet, format_setup_sheet
    sess = _get_session(request)

    if not sess.raw_mesh:
        raise HTTPException(status_code=400, detail="No model loaded")
    if not sess.planned_ops:
        plan_operations(request, PlanOpsRequest(stock=sess.stock_config))
    if not sess.toolpaths:
        generate_toolpaths(request, GenerateToolpathsRequest(stock=sess.stock_config))

    ctx, setups = _build_context_and_setups(sess, sess.stock_config)
    setup = next((s for s in setups if s.id == setup_id), None)
    if not setup:
        raise HTTPException(status_code=404, detail=f"Setup {setup_id} not found")

    setup_tps = [tp for tp in sess.toolpaths if tp.metadata.get("setup_id") == setup_id]
    s_ops = [op for op in sess.planned_ops if op.notes.get("setup_id") == setup_id]
    s_feats = [f for f in sess.features if f.notes.get("setup_id") == setup_id]

    post = get_post_processor(controller=sess.machine.controller or "haas", work_offset=setup.work_offset.value)
    post_result = post.post_process(setup_tps, program_name=f"{sess.model_name.replace(' ', '_')}_{setup.id}")

    sheet = generate_setup_sheet(
        program_name=f"{sess.model_name.upper().replace(' ', '_')}_{setup.id}",
        part_name=sess.model_name,
        planning_ctx=ctx,
        setup=setup,
        features=s_feats,
        operations=s_ops,
        toolpaths=setup_tps,
        post_result=post_result,
    )

    return {"text": format_setup_sheet(sheet), "sheet": {
        "program_name": sheet.program_name,
        "part_name": sheet.part_name,
        "material": sheet.material,
        "machine": sheet.machine,
        "setup_name": sheet.setup_name,
        "work_offset": sheet.work_offset,
        "stock_dimensions": sheet.stock_dimensions,
        "tool_count": len(sheet.tool_list),
        "operation_count": len(sheet.operations),
        "estimated_cycle_time": sheet.estimated_cycle_time,
        "notes": sheet.notes,
    }}


@router.get("/setup-sheet")
def get_setup_sheet(request: Request):
    """Generates a setup sheet for shop floor documentation (primary setup)."""
    from ..setup_sheet import generate_setup_sheet, format_setup_sheet
    sess = _get_session(request)

    if not sess.raw_mesh:
        raise HTTPException(status_code=400, detail="No model loaded")

    if not sess.planned_ops:
        plan_operations(request, PlanOpsRequest(stock=sess.stock_config))

    if not sess.toolpaths:
        generate_toolpaths(request, GenerateToolpathsRequest(stock=sess.stock_config))

    ctx, setups = _build_context_and_setups(sess, sess.stock_config)
    setup = setups[0]
    post = get_post_processor(controller="grbl", work_offset="G54")
    post_result = post.post_process(sess.toolpaths, program_name="SETUP_SHEET")

    sheet = generate_setup_sheet(
        program_name=sess.model_name.upper().replace(" ", "_"),
        part_name=sess.model_name,
        planning_ctx=ctx,
        setup=setup,
        features=sess.features,
        operations=sess.planned_ops,
        toolpaths=sess.toolpaths,
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


def _make_dummy_imported(sess: SessionState):
    """Create a dummy ImportedModel for demo/STL models that lack B-Rep topology."""
    from ..geometry.step_import import ImportedModel
    from ..geometry.topology import Shape

    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    bmin = sess.raw_mesh.bounds_min if sess.raw_mesh else [0, 0, 0]
    bmax = sess.raw_mesh.bounds_max if sess.raw_mesh else [100, 60, 25]
    return ImportedModel(
        shape=Shape(compound),
        declared_units="mm",
        original_bounds=(np.array(bmin, dtype=float), np.array(bmax, dtype=float)),
    )
