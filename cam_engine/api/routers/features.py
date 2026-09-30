"""Features router — thin HTTP controller."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..dependencies import get_session
from ..schemas import FeatureItem, RecognizeFeaturesResponse
from ..services.feature_service import recognize_features_for_session

router = APIRouter()


@router.post("/recognize-features", response_model=RecognizeFeaturesResponse)
def recognize_features(request: Request):
    """Return globally-recognised machining features for the active part.

    Features are recognised once in the model's native coordinate frame.
    Each feature is assigned to exactly one setup — the one whose tool-axis
    best covers it.
    """
    sess = get_session(request)
    recognize_features_for_session(sess)

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
            setup_id=f.setup_id,
            notes=f.notes,
        )
        for f in sess.features
    ]

    return RecognizeFeaturesResponse(features=resp_items, count=len(resp_items))
