"""Setups router — thin HTTP controller."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..dependencies import get_session
from ..exceptions import not_found, bad_request
from ..schemas import (
    AddSetupRequest,
    SetupsListResponse,
    StockConfig,
    UpdateSetupRequest,
)
from ..services.setup_service import build_setups_response, auto_generate_setups_for_session
from ..state import SetupConfig

router = APIRouter()

_SETUP_PRESETS: dict[str, list[float]] = {
    "flip_x_180": [180.0, 0.0, 0.0],
    "flip_y_180": [0.0, 180.0, 0.0],
    "side_x_90":  [90.0, 0.0, 0.0],
    "side_y_90":  [0.0, 90.0, 0.0],
}


@router.get("/setups", response_model=SetupsListResponse)
def get_setups(request: Request):
    """List all configured machining setups with calculated transformed bounds."""
    return build_setups_response(get_session(request))


@router.post("/setups/add", response_model=SetupsListResponse)
def add_setup(request: Request, req: AddSetupRequest):
    """Add a new setup."""
    sess = get_session(request)
    idx = len(sess.setups) + 1
    new_id = f"setup_{idx:03d}"
    rotation = _SETUP_PRESETS.get(req.preset or "", req.rotation_deg or [0.0, 0.0, 0.0])
    stk = req.stock or StockConfig(work_offset=req.work_offset)
    sess.setups.append(
        SetupConfig(id=new_id, name=req.name, work_offset=req.work_offset, rotation_deg=rotation, stock_cfg=stk)
    )
    sess.active_setup_id = new_id
    return build_setups_response(sess)


@router.post("/setups/update/{setup_id}", response_model=SetupsListResponse)
def update_setup(request: Request, setup_id: str, req: UpdateSetupRequest):
    """Update setup parameters."""
    sess = get_session(request)
    target = next((s for s in sess.setups if s.id == setup_id), None)
    if not target:
        raise not_found("Setup", setup_id)
    if req.name is not None:
        target.name = req.name
    if req.work_offset is not None:
        target.work_offset = req.work_offset
    if req.rotation_deg is not None:
        target.rotation_deg = req.rotation_deg
    if req.stock is not None:
        target.stock_cfg = req.stock
    return build_setups_response(sess)


@router.post("/setups/active/{setup_id}", response_model=SetupsListResponse)
def set_active_setup(request: Request, setup_id: str):
    """Set the active setup."""
    sess = get_session(request)
    if not any(s.id == setup_id for s in sess.setups):
        raise not_found("Setup", setup_id)
    sess.active_setup_id = setup_id
    return build_setups_response(sess)


@router.delete("/setups/{setup_id}", response_model=SetupsListResponse)
def delete_setup(request: Request, setup_id: str):
    """Delete a setup (at least one must remain)."""
    sess = get_session(request)
    if len(sess.setups) <= 1:
        raise bad_request("Cannot delete the primary setup")
    sess.setups = [s for s in sess.setups if s.id != setup_id]
    if sess.active_setup_id == setup_id:
        sess.active_setup_id = sess.setups[0].id
    return build_setups_response(sess)


@router.post("/setups/auto-generate", response_model=SetupsListResponse)
def auto_generate_setups(request: Request):
    """Multi-pass feature recognition with face-index deduplication to auto-create setups."""
    sess = get_session(request)
    auto_generate_setups_for_session(sess)
    return build_setups_response(sess)
