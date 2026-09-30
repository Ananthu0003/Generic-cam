"""Tools and machines router — thin HTTP controller."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...context import Tool, ToolType
from ..dependencies import get_session
from ..exceptions import not_found
from ..schemas import (
    AddToolRequest,
    AutoSuggestToolRequest,
    MachineItem,
    ToolItem,
)
from ..services.feature_service import recognize_features_for_session
from ...features import FeatureType

router = APIRouter()

_TOOL_TYPE_MAP: dict[str, ToolType] = {
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


def _tools_response(sess):
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


@router.get("/tools-and-machines")
def get_tools_and_machines(request: Request):
    """Return available tool catalog and default machine configuration."""
    return _tools_response(get_session(request))


@router.post("/tools/add")
def add_custom_tool(request: Request, req: AddToolRequest):
    """Add a new custom tool to the active session tool library."""
    sess = get_session(request)
    t_num = req.tool_number
    if t_num is None:
        t_num = max((t.tool_number for t in sess.tools), default=0) + 1

    ttype = _TOOL_TYPE_MAP.get(req.tool_type.lower(), ToolType.FLAT_ENDMILL)
    t_id = f"T{t_num:02d}"
    existing_ids = {t.id for t in sess.tools}
    if t_id in existing_ids:
        t_id = f"T{t_num:02d}_{len(sess.tools) + 1}"

    sess.tools.append(
        Tool(
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
    )
    return _tools_response(sess)


@router.delete("/tools/{tool_id}")
def delete_custom_tool(request: Request, tool_id: str):
    """Remove a tool from the session tool library."""
    sess = get_session(request)
    sess.tools = [t for t in sess.tools if t.id != tool_id]
    return _tools_response(sess)


@router.post("/tools/auto-suggest")
def auto_suggest_tool(request: Request, req: AutoSuggestToolRequest):
    """Auto-generate a fitting tool for an unmachined feature and add it to library."""
    sess = get_session(request)
    if not sess.features:
        recognize_features_for_session(sess)

    feature = next((f for f in sess.features if f.id == req.feature_id), None)

    ttype = req.tool_type or (
        "drill"
        if feature and feature.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
        else "flat_endmill"
    )

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
        material="carbide",
    )
    return add_custom_tool(request, add_req)
