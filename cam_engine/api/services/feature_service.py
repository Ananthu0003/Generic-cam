"""Feature recognition service.

Contains the domain logic for recognising machining features across all
session setups.  Pure Python — no FastAPI coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...features import FeatureRecognizer, FeatureType, MachiningFeature
from .geometry_service import euler_to_matrix
from .demo_service import populate_demo_features

if TYPE_CHECKING:
    from ..state import SessionState


def recognize_features_for_session(sess: "SessionState") -> None:
    """Populate ``sess.features`` by running feature recognition across all setups.

    If features were already computed this call is a no-op.
    Falls back to demo features when no STEP model has been imported.
    """
    if sess.features:
        return  # already computed

    if sess.imported_model is None:
        populate_demo_features(sess)
        return

    shape = sess.imported_model.shape
    all_feats: list[MachiningFeature] = []
    claimed_features: set[tuple] = set()
    claimed_hole_faces: set[int] = set()

    _HOLE_TYPES = frozenset([
        FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
        FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE, FeatureType.COUNTERBORE,
    ])

    for s in sess.setups:
        rot_mat = euler_to_matrix(*s.rotation_deg)
        setup_shape = shape.transformed(rot_mat)
        _, bmax_rot = setup_shape.bounding_box()
        margin_z_top = float(s.stock_cfg.margin_z_top) if hasattr(s.stock_cfg, "margin_z_top") else 1.0
        stock_top = float(bmax_rot[2]) + margin_z_top

        rec = FeatureRecognizer(setup_shape, stock_top_z=stock_top)
        raw_feats = rec.recognize()

        setup_feats: list[MachiningFeature] = []
        for f in raw_feats:
            # 1. Outer contour only on the primary (first) setup
            if f.type is FeatureType.CONTOUR and s.id != sess.setups[0].id:
                continue
            # 2. Facing regions: only keep if excess stock removal > 0
            if f.type is FeatureType.FACING_REGION:
                if f.notes.get("excess_height", 0.0) <= 1e-3:
                    continue
            # 3. Holes: skip if cylindrical face already claimed
            feat_faces = set(f.face_indices)
            if f.type in _HOLE_TYPES and feat_faces and (feat_faces & claimed_hole_faces):
                continue
            # 4. Exact duplicate
            key = (f.type, frozenset(f.face_indices))
            if key in claimed_features:
                continue

            f.id = f"{s.id}_{f.id}" if len(sess.setups) > 1 else f.id
            f.setup_id = s.id
            f.notes["setup_id"] = s.id
            f.notes["setup_name"] = s.name
            setup_feats.append(f)

        for f in setup_feats:
            claimed_features.add((f.type, frozenset(f.face_indices)))
            if f.type in _HOLE_TYPES:
                claimed_hole_faces.update(f.face_indices)

        s.feature_ids = [f.id for f in setup_feats]
        all_feats.extend(setup_feats)

    sess.features = all_feats
