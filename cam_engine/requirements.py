"""Feature-driven manufacturing requirement evaluation.

Operation planning is not a fixed sequence.  Every candidate operation is
derived from an explicit decision chain evaluated once per detected feature:

    Detected feature
            ↓
    Required surface        the surface that must exist after machining
            ↓
    Stock condition         the material that actually remains against it
            ↓
    Accessibility           can the tool axis reach the feature in this setup
            ↓
    Tool availability       does the library hold a cutter for that surface
            ↓
    Operation type          the operations that are actually required

Mapping implemented by this chain:

    planar face over standing stock            -> facing
    external profile inside bar/box stock      -> rough contour
    corner stock the rough tool cannot reach   -> rest machining
    stock thicker than one finish pass         -> semi finishing
    stock remaining on the required surface    -> finish contour
    toroidal blend                             -> 3D surface finishing
    deep enclosed cavity                       -> pocket strategy
    cylindrical hole                           -> drilling
    precision cylindrical hole                 -> boring / reaming

Nothing here is inferred from a stage that came before it: each test reads the
feature, the surface it demands, and the stock that is really there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .context import Setup, Tool
from .features import Accessibility, FeatureType, MachiningFeature

# ---------------------------------------------------------------- thresholds --
STOCK_EPS = 1e-4                 # mm: thinner stock is not worth an operation
REST_STOCK_THRESHOLD_MM = 0.2    # mm: corner stock above this needs a rest pass
SEMI_PASS_MIN_MM = 0.5           # mm: thinnest stock a finish pass may take
SEMI_PASS_RATIO = 0.30           # fraction of finish tool dia removable at once
DEFAULT_FINISH_ALLOWANCE_MM = 0.3
MIN_FINISH_ALLOWANCE_MM = 0.05
MAX_FINISH_ALLOWANCE_MM = 0.5
PRECISION_TOLERANCE_MM = 0.05    # tolerance at or below this demands precision work
PRECISION_RA_UM = 3.2            # Ra at or below this demands a precision surface

# ------------------------------------------------------------ feature groups --
HOLE_TYPES = frozenset({
    FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
    FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
    FeatureType.COUNTERBORE, FeatureType.TAPPED_HOLE, FeatureType.THREADED_HOLE,
})
BORE_TYPES = frozenset({
    FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
    FeatureType.COUNTERBORE,
})
REAMABLE_TYPES = frozenset({
    FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
    FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
})
CAVITY_TYPES = frozenset({
    FeatureType.POCKET, FeatureType.OPEN_POCKET,
    FeatureType.SLOT, FeatureType.OPEN_SLOT,
})
ROUGHABLE_TYPES = frozenset({
    FeatureType.POCKET, FeatureType.OPEN_POCKET,
    FeatureType.SLOT, FeatureType.OPEN_SLOT,
    FeatureType.STEP, FeatureType.SHOULDER,
    FeatureType.BOSS, FeatureType.PLANAR_WALL,
    FeatureType.CONTOUR,
})
FINISHABLE_TYPES = frozenset({
    FeatureType.POCKET, FeatureType.OPEN_POCKET,
    FeatureType.SLOT, FeatureType.OPEN_SLOT,
    FeatureType.STEP, FeatureType.SHOULDER,
    FeatureType.BOSS, FeatureType.PLANAR_WALL,
    FeatureType.CONTOUR, FeatureType.FREEFORM_SURFACE,
    FeatureType.FILLET, FeatureType.GROOVE,
})
PROFILE_TYPES = frozenset({FeatureType.STEP, FeatureType.SHOULDER, FeatureType.PLANAR_WALL})
EXTERNAL_TYPES = frozenset({FeatureType.BOSS, FeatureType.CONTOUR})


class RequiredSurface(str, Enum):
    """Surface the feature demands once machining is complete."""

    PLANAR_FACE = "planar_face"
    CYLINDRICAL_FACE = "cylindrical_face"
    CONICAL_FACE = "conical_face"
    TOROIDAL_BLEND = "toroidal_blend"
    FREEFORM_FACE = "freeform_face"
    CAVITY = "cavity"
    HOLE_BORE = "hole_bore"
    GROOVE_FACE = "groove_face"


def required_surface_of(feature: MachiningFeature) -> RequiredSurface:
    """Detected feature -> required surface."""
    ftype = feature.type
    st = (feature.surface_type or "").lower()

    if ftype is FeatureType.FACING_REGION:
        return RequiredSurface.PLANAR_FACE
    if ftype in HOLE_TYPES:
        return RequiredSurface.HOLE_BORE
    if ftype in (FeatureType.COUNTERSINK, FeatureType.CHAMFER) or st == "cone":
        return RequiredSurface.CONICAL_FACE
    if ftype is FeatureType.FILLET or st == "torus":
        return RequiredSurface.TOROIDAL_BLEND
    if ftype is FeatureType.FREEFORM_SURFACE or st in ("bspline", "freeform", "nurbs"):
        return RequiredSurface.FREEFORM_FACE
    if ftype is FeatureType.GROOVE:
        return RequiredSurface.GROOVE_FACE
    if ftype in CAVITY_TYPES:
        return RequiredSurface.CAVITY
    if st == "cylinder":
        return RequiredSurface.CYLINDRICAL_FACE
    if ftype in EXTERNAL_TYPES and (
        feature.diameter is not None or bool(feature.notes.get("is_circular"))
    ):
        return RequiredSurface.CYLINDRICAL_FACE
    return RequiredSurface.PLANAR_FACE


def internal_corner_radius(feature: MachiningFeature) -> Optional[float]:
    """Smallest internal corner radius the feature exposes, if it is known."""
    if feature.fillet_radius is not None and feature.fillet_radius > STOCK_EPS:
        return float(feature.fillet_radius)
    if feature.type in (FeatureType.FILLET, *CAVITY_TYPES) and feature.radius is not None \
            and feature.radius > STOCK_EPS:
        return float(feature.radius)
    note = feature.notes.get("corner_radius")
    if note is not None and float(note) > STOCK_EPS:
        return float(note)
    return None


def _is_deep_pocket(feature: MachiningFeature) -> bool:
    """Enclosed cavity whose depth rules out a plain profile-parallel clearing."""
    if feature.type not in (FeatureType.POCKET, FeatureType.OPEN_POCKET,
                            FeatureType.SLOT, FeatureType.OPEN_SLOT):
        return False
    if feature.notes.get("is_open"):
        return False
    extent = np.asarray(feature.xy_extent, dtype=float)
    if extent.size == 0 or float(extent.max()) <= STOCK_EPS:
        return False
    depth = float(feature.depth or 0.0)
    return depth >= 1.5 * float(extent.min()) or depth > float(extent.max())


def _finish_allowance_of(feature: MachiningFeature, context) -> float:
    """Stock deliberately left on the surface after roughing (tolerance driven)."""
    finish = feature.finish
    if finish is None:
        reqs = getattr(context, "requirements", None) or {}
        for fi in feature.face_indices:
            if fi in reqs:
                finish = reqs[fi]
                break
    tolerance = getattr(finish, "tolerance_mm", None) if finish is not None else None
    if tolerance and float(tolerance) > 0:
        return float(min(max(float(tolerance) / 3.0, MIN_FINISH_ALLOWANCE_MM),
                         MAX_FINISH_ALLOWANCE_MM))
    return DEFAULT_FINISH_ALLOWANCE_MM


def _precision_of(feature: MachiningFeature, context) -> tuple[bool, Optional[float], str]:
    """Required surface -> precision demand (drawing tolerance / fine finish)."""
    finish = feature.finish
    if finish is None:
        reqs = getattr(context, "requirements", None) or {}
        for fi in feature.face_indices:
            if fi in reqs:
                finish = reqs[fi]
                break
    if finish is not None:
        tol = float(getattr(finish, "tolerance_mm", 0.0) or 0.0)
        ra = float(getattr(finish, "surface_finish_um", 0.0) or 0.0)
        if 0 < tol <= PRECISION_TOLERANCE_MM:
            return True, tol, f"surface tolerance {tol:.3f} mm demands a precision pass"
        if 0 < ra <= PRECISION_RA_UM:
            return True, (tol or None), f"surface finish Ra {ra:.1f} um demands a precision pass"

    model = feature.tolerances
    if model is not None:
        band = float(model.tolerance_band_mm)
        ra = getattr(model, "surface_finish_ra_um", None)
        if band <= 2 * PRECISION_TOLERANCE_MM:
            return True, band, f"tolerance band {band:.3f} mm demands a precision pass"
        if ra is not None and 0 < float(ra) <= PRECISION_RA_UM:
            return True, band, f"surface finish Ra {float(ra):.1f} um demands a precision pass"
        if getattr(model, "tolerance_class", None):
            return True, band, f"tolerance class {model.tolerance_class} demands a precision pass"

    return False, None, "no precision requirement attached to the surface"


# ---------------------------------------------------------------- stock state --
@dataclass
class StockCondition:
    """Material that actually remains against one feature, measured on the stock."""

    kind: str
    axial_stock_mm: float        # stock standing above the feature's top face
    radial_stock_mm: float       # stock outside an external profile
    interior_stock_mm: float     # bulk material inside a concave feature
    finish_allowance_mm: float   # stock roughing deliberately leaves behind

    @property
    def has_stock(self) -> bool:
        return (self.axial_stock_mm > STOCK_EPS
                or self.radial_stock_mm > STOCK_EPS
                or self.interior_stock_mm > STOCK_EPS)

    # -- stock condition -> operation demand --------------------------------
    def demands_facing(self) -> tuple[bool, str]:
        if self.axial_stock_mm > STOCK_EPS:
            return True, f"{self.axial_stock_mm:.3f} mm of stock stands above the planar face"
        return False, "no stock remains above the planar face"

    def demands_roughing(self) -> tuple[bool, str]:
        if self.interior_stock_mm > STOCK_EPS:
            return True, f"{self.interior_stock_mm:.3f} mm of bulk material fills the feature"
        if self.radial_stock_mm > STOCK_EPS:
            return True, f"{self.radial_stock_mm:.3f} mm of stock surrounds the profile"
        return False, "stock already matches the required profile"

    def demands_finishing(self, rough_planned: bool) -> tuple[bool, str]:
        if rough_planned and self.finish_allowance_mm > STOCK_EPS:
            return True, (f"{self.finish_allowance_mm:.3f} mm allowance left by roughing "
                          "must be removed to reach the required surface")
        if not rough_planned and self.has_stock:
            return True, "raw stock still stands against the required surface"
        if rough_planned:
            return False, "roughing already cut to the required surface"
        return False, "no stock remains against the required surface"

    def corner_stock_mm(self, rough_tool_radius: float,
                        corner_radius: Optional[float]) -> float:
        """Radial stock a roughing cutter leaves in an internal corner.

        An unknown corner radius is treated as a sharp corner: the full cutter
        radius survives there until a smaller tool removes it."""
        if corner_radius is None:
            return max(0.0, float(rough_tool_radius))
        return max(0.0, float(rough_tool_radius) - float(corner_radius))

    def demands_rest(self, rough_tool: Tool,
                     corner_radius: Optional[float]) -> tuple[bool, str]:
        stock = self.corner_stock_mm(rough_tool.diameter / 2.0, corner_radius)
        residual = stock - self.finish_allowance_mm
        if residual <= REST_STOCK_THRESHOLD_MM:
            if stock <= STOCK_EPS:
                return False, "the roughing tool already reaches every internal corner"
            return False, (f"{stock:.3f} mm of corner stock is absorbed by the "
                           f"{self.finish_allowance_mm:.3f} mm finish allowance")
        return True, (f"{residual:.3f} mm of radial stock remains in internal corners "
                      f"after roughing with {rough_tool.id}")

    def demands_semi(self, rough_tool: Tool, rough_stepover_mm: float,
                     finish_tool_diameter: float) -> tuple[bool, str]:
        """Stock left by roughing exceeds what one finishing pass may remove."""
        corner_r = float(getattr(rough_tool, "corner_radius", 0.0) or 0.0)
        cusp = 0.0
        if corner_r > STOCK_EPS and rough_stepover_mm > STOCK_EPS:
            cusp = (rough_stepover_mm ** 2) / (8.0 * corner_r)
        remaining = self.finish_allowance_mm + cusp
        capacity = max(SEMI_PASS_MIN_MM, SEMI_PASS_RATIO * float(finish_tool_diameter))
        if remaining <= capacity:
            return False, (f"{remaining:.3f} mm of stock after roughing fits in one "
                           f"finishing pass (<= {capacity:.3f} mm)")
        return True, (f"{remaining:.3f} mm of stock after roughing exceeds one finishing "
                      f"pass (<= {capacity:.3f} mm)")

    def demands_hole_work(self) -> tuple[bool, str]:
        if self.interior_stock_mm > STOCK_EPS:
            return True, f"{self.interior_stock_mm:.3f} mm of material fills the bore"
        return False, "no material remains inside the bore"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "axial_mm": round(float(self.axial_stock_mm), 4),
            "radial_mm": round(float(self.radial_stock_mm), 4),
            "interior_mm": round(float(self.interior_stock_mm), 4),
            "finish_allowance_mm": round(float(self.finish_allowance_mm), 4),
        }


def measure_stock(feature: MachiningFeature, setup: Setup) -> StockCondition:
    """Required surface + configured stock -> measured stock condition."""
    stock = getattr(setup, "stock", None)
    if stock is None or stock.bounds_min is None or stock.bounds_max is None:
        return StockCondition(kind="none", axial_stock_mm=0.0, radial_stock_mm=0.0,
                              interior_stock_mm=0.0,
                              finish_allowance_mm=DEFAULT_FINISH_ALLOWANCE_MM)

    smin = np.asarray(stock.bounds_min, dtype=float)
    smax = np.asarray(stock.bounds_max, dtype=float)
    fmin = np.asarray(feature.bounds_min, dtype=float)
    fmax = np.asarray(feature.bounds_max, dtype=float)

    kind = getattr(getattr(stock, "kind", None), "value", str(getattr(stock, "kind", "")))

    # Footprint overlap between feature and stock (XY).
    overlap_x = min(float(fmax[0]), float(smax[0])) - max(float(fmin[0]), float(smin[0]))
    overlap_y = min(float(fmax[1]), float(smax[1])) - max(float(fmin[1]), float(smin[1]))
    footprint_overlap = overlap_x > STOCK_EPS and overlap_y > STOCK_EPS
    if not footprint_overlap:
        # Nothing of the feature sits inside the stock: there is no material.
        return StockCondition(
            kind=kind, axial_stock_mm=0.0, radial_stock_mm=0.0,
            interior_stock_mm=0.0,
            finish_allowance_mm=DEFAULT_FINISH_ALLOWANCE_MM,
        )

    # Axial: material standing above the feature's top face.  A facing region
    # reports the measured excess from recognition (stock top vs model top).
    if feature.type is FeatureType.FACING_REGION:
        axial = max(
            float(feature.notes.get("excess_height", 0.0) or 0.0),
            float(feature.depth or 0.0),
            float(smax[2] - fmax[2]),
        )
        axial = max(0.0, axial)
    else:
        axial = max(0.0, float(smax[2] - float(feature.top_z)))

    # Interior: bulk material inside a concave feature that the stock still fills.
    interior = 0.0
    if feature.is_concave and footprint_overlap:
        z_top = min(float(feature.top_z), float(smax[2]))
        z_bot = max(float(feature.floor_z), float(smin[2]))
        interior = max(0.0, z_top - z_bot)

    # Radial: stock standing outside an external profile.  The largest standoff
    # decides: material on any side still has to be cut away.
    radial = 0.0
    if not feature.is_concave and footprint_overlap:
        gaps = [
            float(fmin[0]) - float(smin[0]),
            float(smax[0]) - float(fmax[0]),
            float(fmin[1]) - float(smin[1]),
            float(smax[1]) - float(fmax[1]),
        ]
        radial = max(0.0, max(gaps))

    return StockCondition(
        kind=kind,
        axial_stock_mm=axial,
        radial_stock_mm=radial,
        interior_stock_mm=interior,
        finish_allowance_mm=0.0,   # filled by evaluate_feature_requirement
    )


# ---------------------------------------------------------------- requirement --
@dataclass
class FeatureRequirement:
    """Full decision chain state for one detected feature."""

    feature: MachiningFeature
    required_surface: RequiredSurface
    stock: StockCondition
    accessible: bool
    precision_required: bool
    tolerance_mm: Optional[float] = None
    strategy: str = "profile"
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def feature_id(self) -> str:
        return self.feature.id

    @property
    def accessibility_rejection(self) -> Optional[str]:
        if self.accessible:
            return None
        return (f"feature {self.feature.id} is not reachable along the tool axis "
                f"({self.feature.accessibility.value})")

    def demands_roughing(self) -> tuple[bool, str]:
        if not self.accessible:
            return False, self.accessibility_rejection or "feature is inaccessible"
        if self.feature.type not in ROUGHABLE_TYPES:
            return False, f"{self.feature.type.value} is not bulk-removed by contour roughing"
        return self.stock.demands_roughing()

    def demands_finishing(self, rough_planned: bool) -> tuple[bool, str]:
        if not self.accessible:
            return False, self.accessibility_rejection or "feature is inaccessible"
        if self.feature.type not in FINISHABLE_TYPES:
            return False, f"{self.feature.type.value} is finished by a dedicated strategy"
        return self.stock.demands_finishing(rough_planned)

    def demands_hole_work(self) -> tuple[bool, str]:
        if not self.accessible:
            return False, self.accessibility_rejection or "feature is inaccessible"
        return self.stock.demands_hole_work()

    def demands_precision_hole_finish(self) -> tuple[bool, str]:
        ok, reason = self.demands_hole_work()
        if not ok:
            return False, reason
        if not self.precision_required:
            return False, "the bore carries no precision requirement"
        return True, self.evidence.get("precision", "precision bore surface required")

    def demands_surface_pass(self, purpose: str) -> tuple[bool, str]:
        """Dedicated surface passes (chamfer, countersink, fillet, groove)."""
        if not self.accessible:
            return False, self.accessibility_rejection or "feature is inaccessible"
        if not self.stock.has_stock:
            return False, "no stock remains against the required surface"
        return True, (f"{self.stock.axial_stock_mm + self.stock.radial_stock_mm + self.stock.interior_stock_mm:.3f} mm "
                      f"of stock remains on the {self.required_surface.value}")

    def trace(self, purpose: str, reason: str) -> dict:
        """Auditable record of why this operation type was required."""
        return {
            "purpose": purpose,
            "feature_type": self.feature.type.value,
            "required_surface": self.required_surface.value,
            "strategy": self.strategy,
            "precision_required": self.precision_required,
            "tolerance_mm": self.tolerance_mm,
            "accessible": self.accessible,
            "stock_condition": self.stock.to_dict(),
            "decision": reason,
            "evidence": dict(self.evidence),
        }

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature.id,
            "required_surface": self.required_surface.value,
            "strategy": self.strategy,
            "precision_required": self.precision_required,
            "tolerance_mm": self.tolerance_mm,
            "accessible": self.accessible,
            "stock_condition": self.stock.to_dict(),
            "evidence": dict(self.evidence),
        }


def _strategy_for(feature: MachiningFeature, surface: RequiredSurface) -> str:
    ftype = feature.type
    if ftype is FeatureType.FACING_REGION:
        return "face"
    if ftype in HOLE_TYPES:
        return "hole"
    if ftype in (FeatureType.COUNTERSINK, FeatureType.CHAMFER):
        return "chamfer"
    if surface in (RequiredSurface.TOROIDAL_BLEND, RequiredSurface.FREEFORM_FACE):
        return "3d_surface"
    if ftype is FeatureType.GROOVE:
        return "groove"
    if ftype in CAVITY_TYPES:
        return "pocket" if _is_deep_pocket(feature) else "cavity"
    if ftype in EXTERNAL_TYPES:
        return "contour"
    if ftype in PROFILE_TYPES:
        return "profile"
    return "profile"


def evaluate_feature_requirement(feature: MachiningFeature, setup: Setup,
                                 context=None) -> FeatureRequirement:
    """Run the full feature -> surface -> stock -> accessibility chain."""
    surface = required_surface_of(feature)
    stock = measure_stock(feature, setup)
    stock.finish_allowance_mm = _finish_allowance_of(feature, context)
    precision, tolerance, precision_reason = _precision_of(feature, context)
    accessible = feature.accessibility is not Accessibility.INACCESSIBLE
    strategy = _strategy_for(feature, surface)

    evidence: dict[str, str] = {
        "detected_feature": f"{feature.type.value} ({feature.id})",
        "required_surface": surface.value,
        "precision": precision_reason,
        "accessibility": feature.accessibility.value,
        "stock_kind": stock.kind,
    }

    if feature.type is FeatureType.FACING_REGION:
        evidence["stock_condition"] = (
            f"{stock.axial_stock_mm:.3f} mm axial stock above the planar face")
    elif feature.type in CAVITY_TYPES:
        evidence["stock_condition"] = (
            f"{stock.interior_stock_mm:.3f} mm bulk material inside the cavity")
    elif feature.is_concave:
        evidence["stock_condition"] = (
            f"{stock.interior_stock_mm:.3f} mm material inside the feature")
    else:
        evidence["stock_condition"] = (
            f"{stock.radial_stock_mm:.3f} mm stock outside the profile")
    if strategy == "pocket":
        evidence["strategy"] = "deep cavity: pocket clearing strategy"
    elif stock.kind == "cylinder" and feature.type in EXTERNAL_TYPES:
        evidence["strategy"] = "cylindrical stock: rough contour strategy"

    return FeatureRequirement(
        feature=feature,
        required_surface=surface,
        stock=stock,
        accessible=accessible,
        precision_required=precision,
        tolerance_mm=tolerance,
        strategy=strategy,
        evidence=evidence,
    )
