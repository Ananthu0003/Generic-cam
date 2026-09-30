"""Model upload and orientation router."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ..dependencies import get_session
from ..exceptions import unprocessable
from ..schemas import ModelInfoResponse, OrientModelRequest
from ..services.geometry_service import shape_to_demo_cad_model, mesh_to_response
from ...demo_models import DemoCADModel
from ...geometry.step_import import import_step
from ...errors import CamError
from ..config import SUPPORTED_CAD_EXTENSIONS

router = APIRouter()


@router.get("/model/demo", response_model=ModelInfoResponse)
def get_demo_model(request: Request):
    """Load and return the demo CAD model mesh and bounding box."""
    sess = get_session(request)
    sess.load_demo()
    return ModelInfoResponse(
        name=sess.model_name,
        units="mm",
        mesh=mesh_to_response(sess.raw_mesh),
        features_count=0,
    )


@router.post("/model/upload", response_model=ModelInfoResponse)
async def upload_model_file(request: Request, file: UploadFile = File(...)):
    """Upload a CAD file (.step, .stp, .stl), process it, and cache the model."""
    sess = get_session(request)
    orig_filename = file.filename or "Uploaded Part"
    filename_lower = orig_filename.lower()

    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_CAD_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Supported formats: .step, .stp, .stl")

    suffix = Path(orig_filename).suffix or ".step"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if filename_lower.endswith(".stl"):
            return _process_stl(sess, orig_filename, tmp_path)
        else:
            return _process_step(sess, orig_filename, tmp_path)

    except CamError as e:
        raise unprocessable(f"{e.code}: {e.message}")
    except HTTPException:
        raise
    except Exception as e:
        raise unprocessable(f"Failed to process file: {e}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _process_stl(sess, orig_filename: str, tmp_path: Path) -> ModelInfoResponse:
    from ...geometry.stl_import import import_stl, STLImportError
    try:
        mesh = import_stl(tmp_path)
    except STLImportError as e:
        raise HTTPException(status_code=422, detail=f"Failed to process STL file: {e}")

    vertices: list[float] = []
    normals: list[float] = []
    indices: list[int] = []
    faces_meta: list[dict] = []

    for i, tri in enumerate(mesh.triangles):
        base_idx = len(vertices) // 3
        v0, v1, v2 = tri[0], tri[1], tri[2]
        normal = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normal)
        normal = normal / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0])

        for p in (v0, v1, v2):
            vertices.extend([float(p[0]), float(p[1]), float(p[2])])
            normals.extend([float(normal[0]), float(normal[1]), float(normal[2])])

        indices.extend([base_idx, base_idx + 1, base_idx + 2])
        faces_meta.append({"id": i, "kind": "mesh_facet", "area": 0.5 * norm_len})

    bmin = [float(x) for x in mesh.bounds_min]
    bmax = [float(x) for x in mesh.bounds_max]

    sess.raw_mesh = DemoCADModel(
        name=orig_filename, description="Imported STL Mesh",
        bounds_min=bmin, bounds_max=bmax,
        vertices=vertices, normals=normals, indices=indices,
        faces_metadata=faces_meta,
    )
    sess.imported_model = None
    sess.model_name = orig_filename
    sess.features = []
    sess.planned_ops = []
    sess.toolpaths = []

    return ModelInfoResponse(name=sess.model_name, units="mm", mesh=mesh_to_response(sess.raw_mesh), features_count=0)


def _process_step(sess, orig_filename: str, tmp_path: Path) -> ModelInfoResponse:
    imported = import_step(tmp_path)
    sess.imported_model = imported
    sess.model_name = orig_filename
    sess.features = []
    sess.planned_ops = []
    sess.toolpaths = []
    sess.raw_mesh = shape_to_demo_cad_model(orig_filename, imported.shape, "Imported STEP Model")

    return ModelInfoResponse(name=sess.model_name, units="mm", mesh=mesh_to_response(sess.raw_mesh), features_count=0)


@router.post("/model/orient", response_model=ModelInfoResponse)
def orient_model(request: Request, req: OrientModelRequest):
    """Rotate the active CAD model or align a selected face to +Z."""
    sess = get_session(request)

    if sess.imported_model is not None:
        if req.align_face_id is not None:
            sess.imported_model.shape = sess.imported_model.shape.align_face_to_axis(
                req.align_face_id, np.array([0.0, 0.0, 1.0])
            )
        else:
            sess.imported_model.shape = sess.imported_model.shape.rotate(req.axis, req.angle_deg)
        bmin, bmax = sess.imported_model.shape.bounding_box()
        sess.imported_model.original_bounds = (bmin, bmax)
        sess.raw_mesh = shape_to_demo_cad_model(sess.model_name, sess.imported_model.shape, "Oriented Model")

    elif sess.raw_mesh is not None:
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
        sess.raw_mesh.vertices = rotated_pts.flatten().tolist()
        sess.raw_mesh.normals = rotated_norms.flatten().tolist()
        sess.raw_mesh.bounds_min = [float(x) for x in rotated_pts.min(axis=0)]
        sess.raw_mesh.bounds_max = [float(x) for x in rotated_pts.max(axis=0)]

    sess.features = []
    sess.planned_ops = []
    sess.toolpaths = []

    return ModelInfoResponse(
        name=sess.model_name, units="mm",
        mesh=mesh_to_response(sess.raw_mesh), features_count=0,
    )
