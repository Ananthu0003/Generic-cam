"""Setup management service.

Contains domain logic for CRUD operations on machining setups and the
auto-generation algorithm.  Pure Python — no FastAPI coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...features import FeatureRecognizer, FeatureType, MachiningFeature
from ..schemas import BoundingBox, StockConfig, SetupsListResponse, SetupItem
from ..config import AUTO_SETUP_CANDIDATES
from .geometry_service import euler_to_matrix
from .stock_service import create_stock_from_config
from .demo_service import populate_demo_features

if TYPE_CHECKING:
    from ..state import SessionState, SetupConfig


def build_setups_response(sess: "SessionState") -> SetupsListResponse:
    """Build the ``SetupsListResponse`` from current session state."""
    items: list[SetupItem] = []

    for s in sess.setups:
        rot_mat = euler_to_matrix(*s.rotation_deg)
        if sess.imported_model:
            shape_in_setup = sess.imported_model.shape.transformed(rot_mat)
            bmin, bmax = shape_in_setup.bounding_box()
        elif sess.raw_mesh:
            bmin = sess.raw_mesh.bounds_min
            bmax = sess.raw_mesh.bounds_max
        else:
            bmin = [0.0, 0.0, 0.0]
            bmax = [100.0, 60.0, 25.0]

        cfg = s.stock_cfg or StockConfig(work_offset=s.work_offset)
        stock = create_stock_from_config(cfg, bmin, bmax)
        s_min = [float(x) for x in stock.bounds_min]
        s_max = [float(x) for x in stock.bounds_max]
        s_size = [float(s_max[i] - s_min[i]) for i in range(3)]

        items.append(SetupItem(
            id=s.id,
            name=s.name,
            work_offset=s.work_offset,
            rotation_deg=s.rotation_deg,
            stock=cfg,
            stock_bounds=BoundingBox(min=s_min, max=s_max, size=s_size),
            feature_ids=s.feature_ids,
            operations_count=s.operations_count,
            is_active=(s.id == sess.active_setup_id),
        ))

    return SetupsListResponse(setups=items, active_setup_id=sess.active_setup_id)


def auto_generate_setups_for_session(sess: "SessionState") -> None:
    """Run multi-pass feature recognition and generate setups automatically.

    Resets ``sess.setups``, ``sess.features``, ``sess.planned_ops``,
    and ``sess.toolpaths`` before the pass.
    """
    from ..state import SetupConfig

    sess.features = []
    sess.planned_ops = []
    sess.toolpaths = []

    # Demo / mesh fallback
    if sess.imported_model is None:
        sess.setups = [
            SetupConfig(id="setup_001", name="Setup 1 - Top Milling", work_offset="G54", rotation_deg=[0.0, 0.0, 0.0]),
            SetupConfig(id="setup_002", name="Setup 2 - Bottom Flip", work_offset="G55", rotation_deg=[180.0, 0.0, 0.0]),
        ]
        sess.active_setup_id = "setup_001"
        populate_demo_features(sess)
        return

    shape = sess.imported_model.shape
    STOCK_MARGIN = 1.0

    _HOLE_TYPES = frozenset([
        FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
        FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE, FeatureType.COUNTERBORE,
    ])

    claimed_features: set[tuple] = set()
    claimed_hole_faces: set[int] = set()
    direction_features: dict[str, list[MachiningFeature]] = {}

    for direction, rot_deg, label, wo in AUTO_SETUP_CANDIDATES:
        rot_mat = euler_to_matrix(*rot_deg)
        setup_shape = shape.transformed(rot_mat)
        _, bmax_rot = setup_shape.bounding_box()
        stock_top = float(bmax_rot[2]) + STOCK_MARGIN

        rec = FeatureRecognizer(setup_shape, stock_top_z=stock_top)
        raw_feats = rec.recognize()

        net_new: list[MachiningFeature] = []
        for feat in raw_feats:
            if feat.type is FeatureType.CONTOUR and direction != "top":
                continue
            if feat.type is FeatureType.FACING_REGION:
                if feat.notes.get("excess_height", 0.0) <= 1e-3:
                    continue
            feat_faces = set(feat.face_indices)
            if feat.type in _HOLE_TYPES and feat_faces and (feat_faces & claimed_hole_faces):
                continue
            key = (feat.type, frozenset(feat.face_indices))
            if key in claimed_features:
                continue
            net_new.append(feat)

        has_substantive = any(
            f.type not in (FeatureType.FACING_REGION, FeatureType.CONTOUR) for f in net_new
        )
        if direction != "top" and not has_substantive:
            continue

        for feat in net_new:
            claimed_features.add((feat.type, frozenset(feat.face_indices)))
            if feat.type in _HOLE_TYPES:
                claimed_hole_faces.update(feat.face_indices)

        if net_new:
            direction_features[direction] = net_new

    if "top" not in direction_features:
        direction_features["top"] = []

    WORK_OFFSETS = ["G54", "G55", "G56", "G57", "G58", "G59"]
    setups: list["SetupConfig"] = []
    setup_idx = 1
    direction_to_setup_id: dict[str, str] = {}

    for direction, rot_deg, label, _ in AUTO_SETUP_CANDIDATES:
        if direction not in direction_features:
            continue
        sid = f"setup_{setup_idx:03d}"
        name = f"Setup {setup_idx} - {label}"
        wo = WORK_OFFSETS[(setup_idx - 1) % len(WORK_OFFSETS)]
        setups.append(SetupConfig(id=sid, name=name, work_offset=wo, rotation_deg=rot_deg))
        direction_to_setup_id[direction] = sid
        setup_idx += 1

    all_features: list[MachiningFeature] = []
    for direction, feats in direction_features.items():
        sid = direction_to_setup_id[direction]
        sc_name = next((s.name for s in setups if s.id == sid), sid)
        for feat in feats:
            feat.setup_id = sid
            feat.notes["setup_id"] = sid
            feat.notes["setup_name"] = sc_name
        all_features.extend(feats)

    for sc in setups:
        sc.feature_ids = [f.id for f in all_features if f.setup_id == sc.id]

    sess.setups = setups
    sess.active_setup_id = setups[0].id
    sess.features = all_features
