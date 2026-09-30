"""Toolpaths router — thin HTTP controller."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..dependencies import get_session
from ..schemas import (
    GenerateToolpathsRequest,
    GenerateToolpathsResponse,
    MotionSegmentItem,
    ToolpathItem,
)
from ..services.toolpath_service import generate_toolpaths_for_session
from ...errors import CamError

router = APIRouter()


@router.post("/generate-toolpaths", response_model=GenerateToolpathsResponse)
def generate_toolpaths(request: Request, req: GenerateToolpathsRequest):
    """Generate continuous semantic toolpaths with rapid, cut, plunge, and retract segments across setups."""
    sess = get_session(request)

    try:
        diagnostics = generate_toolpaths_for_session(sess, req.stock)

        total_cut_len = sum(tp.cutting_length() for tp in sess.toolpaths)
        total_rapid_len = sum(tp.rapid_length() for tp in sess.toolpaths)

        resp_tps: list[ToolpathItem] = []
        for tp in sess.toolpaths:
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
            diagnostics=diagnostics if diagnostics else None,
        )

    except CamError as e:
        raise HTTPException(status_code=400, detail=f"[{e.code}] {e.message}")
