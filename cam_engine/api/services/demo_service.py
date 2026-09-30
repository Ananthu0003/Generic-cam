"""Demo model helper service.

Pure functions — no FastAPI coupling.  Extracted from the old ``utils.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ...features import MachiningFeature, FeatureType, Accessibility

if TYPE_CHECKING:
    from ..state import SessionState


def make_dummy_imported(sess: "SessionState"):
    """Return a minimal ``ImportedModel`` for demo/STL sessions that lack B-Rep topology."""
    from ...geometry.step_import import ImportedModel
    from ...geometry.topology import Shape
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


def populate_demo_features(sess: "SessionState") -> None:
    """Populate ``sess.features`` with the hard-coded demo bracket features.

    All features are assigned to the primary setup (``setup_001``).
    Called when there is no imported STEP model and we need the demo feature set.
    """
    s0 = sess.setups[0]
    margin_z = float(sess.stock_config.margin_z_top) if hasattr(sess.stock_config, "margin_z_top") else 1.0

    demo_feats = [
        MachiningFeature(
            id="feat_facing_001",
            type=FeatureType.FACING_REGION,
            face_indices=[5],
            bounds_min=np.array([0.0, 0.0, 25.0]),
            bounds_max=np.array([100.0, 60.0, 25.0 + margin_z]),
            depth=margin_z,
            top_z=25.0 + margin_z,
            floor_z=25.0,
            accessibility=Accessibility.TOOL_AXIS_OK,
            is_concave=False,
            setup_id=s0.id,
            notes={
                "strategy": "zigzag",
                "stock_removal": f"{margin_z:.1f}mm",
                "excess_height": margin_z,
                "setup_id": s0.id,
                "setup_name": s0.name,
            },
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
            setup_id=s0.id,
            notes={"width": 40.0, "length": 30.0, "corner_radius": 3.0, "setup_id": s0.id, "setup_name": s0.name},
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
            setup_id=s0.id,
            notes={"open_sides": ["+X", "-Y", "+Y"], "setup_id": s0.id, "setup_name": s0.name},
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
            setup_id=s0.id,
            notes={"center": [70.0, 30.0], "type": "through_bore", "setup_id": s0.id, "setup_name": s0.name},
        ),
        MachiningFeature(
            id="feat_contour_001",
            type=FeatureType.CONTOUR,
            face_indices=[1, 2, 3, 4],
            bounds_min=np.array([0.0, 0.0, 0.0]),
            bounds_max=np.array([100.0, 60.0, 25.0]),
            depth=25.0,
            top_z=25.0,
            floor_z=0.0,
            accessibility=Accessibility.TOOL_AXIS_OK,
            is_concave=False,
            setup_id=s0.id,
            notes={"closed": True, "type": "outer_profile", "setup_id": s0.id, "setup_name": s0.name},
        ),
    ]

    sess.features = demo_feats
    s0.feature_ids = [f.id for f in demo_feats]
