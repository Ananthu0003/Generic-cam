"""Operation validity checks: the gate between planning and G-code.

Every operation must prove itself before it is accepted:

    geometry exists?            the feature has real, finite geometry
    target surface exists?      a CAD surface (or boundary) is actually targeted
    tool compatible?            the cutter fits the feature it must cut
    tool access valid?          the tool axis can reach the feature here
    depth valid?                the requested depth exists inside the machine
    toolpath intersects stock?  the generated motion actually touches material
    toolpath reaches target?    the motion gets down to the target surface
    compensation valid?         G41/G42/G40 is consistent and legal
    feed valid?                 every cutting move carries a legal feed
    rpm valid?                  every spindle command is inside the spindle range
    holder clearance valid?     flute/holder can reach without colliding
    setup valid?                the setup transform and stock are well formed

Checks are evaluated when a toolpath is generated and stamped onto
``toolpath.metadata["validity"]``.  ``assert_valid_for_post`` re-reads those
stamps (plus the checks that only need the toolpath itself) immediately before
G-code emission; a single critical failure raises and **no G-code is produced**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .context import PlanningContext, Setup, ToolType
from .errors import (
    CamError,
    COLLISION_DETECTED,
    ErrorSeverity,
    FEATURE_INACCESSIBLE,
    GCODE_VALIDATION_FAILED,
    INVALID_CUTTING_PARAMETERS,
    INVALID_GEOMETRY,
    INVALID_SETUP,
    INVALID_TOOLPATH,
    MACHINE_LIMIT_EXCEEDED,
    NO_COMPATIBLE_TOOL,
    TOOL_REACH_INSUFFICIENT,
)

# ------------------------------------------------------------ the 12 checks --
GEOMETRY_EXISTS = "geometry_exists"
TARGET_SURFACE_EXISTS = "target_surface_exists"
TOOL_COMPATIBLE = "tool_compatible"
TOOL_ACCESS_VALID = "tool_access_valid"
DEPTH_VALID = "depth_valid"
TOOLPATH_INTERSECTS_STOCK = "toolpath_intersects_stock"
TOOLPATH_REACHES_TARGET = "toolpath_reaches_target"
COMPENSATION_VALID = "compensation_valid"
FEED_VALID = "feed_valid"
RPM_VALID = "rpm_valid"
HOLDER_CLEARANCE_VALID = "holder_clearance_valid"
SETUP_VALID = "setup_valid"

CHECK_ORDER: tuple[str, ...] = (
    GEOMETRY_EXISTS,
    TARGET_SURFACE_EXISTS,
    TOOL_COMPATIBLE,
    TOOL_ACCESS_VALID,
    DEPTH_VALID,
    TOOLPATH_INTERSECTS_STOCK,
    TOOLPATH_REACHES_TARGET,
    COMPENSATION_VALID,
    FEED_VALID,
    RPM_VALID,
    HOLDER_CLEARANCE_VALID,
    SETUP_VALID,
)

#: purposes whose target is the top face of the feature, not its floor
_TOP_FACE_PURPOSES = frozenset({"spot_drilling", "facing"})

#: tools that must never carry cutter radius compensation
_NO_COMP_TOOL_TYPES = frozenset({
    ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL,
    ToolType.TAP, ToolType.REAMER, ToolType.COUNTERSINK_TOOL,
})

#: deepest cut may stop this far above the feature floor and still be "on target"
TARGET_REACH_TOLERANCE_MM = 0.5
STOCK_OVERLAP_EPS = 1e-6
LENGTH_EPS = 1e-6


# ---------------------------------------------------------------- reporting --
@dataclass
class ValidityCheck:
    """Outcome of one named check."""

    name: str
    passed: bool
    message: str = ""
    critical: bool = True
    evaluated: bool = True
    code: str = ""

    @property
    def blocking(self) -> bool:
        return self.evaluated and not self.passed and self.critical

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "critical": self.critical,
            "evaluated": self.evaluated,
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidityCheck":
        return cls(
            name=str(data.get("name", "")),
            passed=bool(data.get("passed", False)),
            message=str(data.get("message", "")),
            critical=bool(data.get("critical", True)),
            evaluated=bool(data.get("evaluated", True)),
            code=str(data.get("code", "")),
        )


@dataclass
class ValidityReport:
    """All checks evaluated for one operation/toolpath."""

    operation_id: str = ""
    tool_id: str = ""
    purpose: str = ""
    checks: list[ValidityCheck] = field(default_factory=list)
    skipped: bool = False

    @property
    def critical_failures(self) -> list[ValidityCheck]:
        return [c for c in self.checks if c.blocking]

    @property
    def failures(self) -> list[ValidityCheck]:
        return [c for c in self.checks if c.evaluated and not c.passed]

    @property
    def passed(self) -> bool:
        return self.skipped or not self.critical_failures

    @property
    def failed_names(self) -> list[str]:
        return [c.name for c in self.critical_failures]

    def summary(self) -> str:
        fails = "; ".join(
            f"{c.name}: {c.message or 'failed'}" for c in self.critical_failures
        )
        return fails or "all checks passed"

    def to_dict(self) -> dict:
        return {
            "operation_id": self.operation_id,
            "tool_id": self.tool_id,
            "purpose": self.purpose,
            "skipped": self.skipped,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidityReport":
        return cls(
            operation_id=str(data.get("operation_id", "")),
            tool_id=str(data.get("tool_id", "")),
            purpose=str(data.get("purpose", "")),
            skipped=bool(data.get("skipped", False)),
            checks=[ValidityCheck.from_dict(c) for c in data.get("checks", [])],
        )

    def blocking_errors(self) -> list[CamError]:
        return [
            CamError(
                code=c.code or INVALID_TOOLPATH,
                message=c.message or f"{c.name} failed",
                stage="operation_validity",
                operation_id=self.operation_id,
                tool_id=self.tool_id,
                severity=ErrorSeverity.ERROR,
                context={"check": c.name, "purpose": self.purpose},
            )
            for c in self.critical_failures
        ]


# ------------------------------------------------------------------ checker --
class OperationValidityChecker:
    """Evaluates the full check list for one planned operation."""

    def __init__(self, context: Optional[PlanningContext] = None,
                 setup: Optional[Setup] = None):
        self.context = context
        self.setup = setup
        self.machine = getattr(context, "machine", None)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _ok(name: str, message: str = "", code: str = "") -> ValidityCheck:
        return ValidityCheck(name=name, passed=True, message=message, code=code)

    @staticmethod
    def _fail(name: str, message: str, code: str, critical: bool = True) -> ValidityCheck:
        return ValidityCheck(name=name, passed=False, message=message,
                             code=code, critical=critical)

    @staticmethod
    def _skip(name: str, message: str) -> ValidityCheck:
        return ValidityCheck(name=name, passed=True, evaluated=False, message=message)

    def check(self, op: Any, toolpath: Any = None) -> ValidityReport:
        """Run every check that the available data allows us to evaluate."""
        feature = getattr(op, "feature", None)
        tool = getattr(op, "tool", None)
        purpose = getattr(op.purpose, "value", str(getattr(op, "purpose", "")))
        report = ValidityReport(
            operation_id=str(getattr(op, "id", "")),
            tool_id=str(getattr(tool, "id", "") or ""),
            purpose=purpose,
        )

        skipped_reason = None
        if toolpath is not None:
            skipped_reason = (toolpath.metadata or {}).get("skipped")
            if skipped_reason:
                report.skipped = True
                report.checks = [
                    self._skip(name, f"toolpath skipped: {skipped_reason}")
                    for name in CHECK_ORDER
                ]
                return report

        report.checks = [
            self._check_geometry(feature, toolpath),
            self._check_target_surface(feature, toolpath),
            self._check_tool_compatible(tool, feature, op, purpose),
            self._check_tool_access(feature),
            self._check_depth(feature),
            self._check_stock_intersection(tool, toolpath),
            self._check_reaches_target(feature, toolpath, purpose, op=op),
            self._check_compensation(tool, toolpath),
            self._check_feed(toolpath),
            self._check_rpm(toolpath),
            self._check_holder_clearance(tool, feature, toolpath),
            self._check_setup(),
        ]
        return report

    # -- 1. geometry exists -------------------------------------------------
    def _check_geometry(self, feature: Any, toolpath: Any) -> ValidityCheck:
        if feature is None and toolpath is None:
            return self._skip(GEOMETRY_EXISTS, "no feature or toolpath to inspect")
        if feature is not None:
            bmin = np.asarray(getattr(feature, "bounds_min", []), dtype=float)
            bmax = np.asarray(getattr(feature, "bounds_max", []), dtype=float)
            if bmin.size != 3 or bmax.size != 3:
                return self._fail(GEOMETRY_EXISTS,
                                  f"feature {getattr(feature, 'id', '?')} has no 3D bounds",
                                  INVALID_GEOMETRY)
            if not (np.all(np.isfinite(bmin)) and np.all(np.isfinite(bmax))):
                return self._fail(GEOMETRY_EXISTS,
                                  f"feature {feature.id} bounds are not finite",
                                  INVALID_GEOMETRY)
            extent = bmax - bmin
            diameter = float(getattr(feature, "diameter", 0.0) or 0.0)
            if float(np.max(extent)) <= LENGTH_EPS and diameter <= LENGTH_EPS:
                return self._fail(GEOMETRY_EXISTS,
                                  f"feature {feature.id} has degenerate geometry",
                                  INVALID_GEOMETRY)
            if float(getattr(feature, "depth", 0.0) or 0.0) < 0:
                return self._fail(GEOMETRY_EXISTS,
                                  f"feature {feature.id} has negative depth",
                                  INVALID_GEOMETRY)

        if toolpath is not None:
            segments = list(getattr(toolpath, "segments", []) or [])
            if not segments:
                return self._fail(GEOMETRY_EXISTS,
                                  f"toolpath {getattr(toolpath, 'operation_id', '?')} "
                                  "contains no motion",
                                  INVALID_TOOLPATH)
            for i, seg in enumerate(segments):
                start = np.asarray(seg.start, dtype=float)
                end = np.asarray(seg.end, dtype=float)
                if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                    return self._fail(GEOMETRY_EXISTS,
                                      f"segment {i} has non-finite endpoints",
                                      INVALID_TOOLPATH)
        return self._ok(GEOMETRY_EXISTS, "feature and toolpath geometry are finite and present")

    # -- 2. target surface exists ------------------------------------------
    def _check_target_surface(self, feature: Any, toolpath: Any) -> ValidityCheck:
        if toolpath is not None and (
            getattr(toolpath, "target_surface", None)
            or list(getattr(toolpath, "source_face_ids", []) or [])
        ):
            return self._ok(TARGET_SURFACE_EXISTS,
                            f"targets {getattr(toolpath, 'target_surface', None) or 'source faces'}")
        if feature is None:
            return self._fail(TARGET_SURFACE_EXISTS, "no feature to define a target surface",
                              INVALID_GEOMETRY)
        if list(getattr(feature, "face_indices", []) or []):
            return self._ok(TARGET_SURFACE_EXISTS,
                            f"faces {list(feature.face_indices)[:6]}")
        if (list(getattr(feature, "source_edges", []) or [])
                or getattr(feature, "boundary_loop", None) is not None
                or getattr(feature, "surface_type", None)
                or (getattr(feature, "notes", {}) or {}).get("source_geometry_reference")):
            return self._ok(TARGET_SURFACE_EXISTS, "boundary/surface geometry is present")
        return self._fail(TARGET_SURFACE_EXISTS,
                          f"feature {getattr(feature, 'id', '?')} has no CAD faces, "
                          "edges or surface to machine",
                          INVALID_GEOMETRY)

    # -- 3. tool compatible -------------------------------------------------
    def _check_tool_compatible(self, tool: Any, feature: Any, op: Any,
                               purpose: str) -> ValidityCheck:
        if tool is None:
            return self._fail(TOOL_COMPATIBLE, "operation has no tool", NO_COMPATIBLE_TOOL)
        if feature is None:
            return self._skip(TOOL_COMPATIBLE, "no feature to check the tool against")
        try:
            from .toolpaths.validation import PreToolpathGeometryValidator
            validator = PreToolpathGeometryValidator(
                context=self.context, setup=self.setup, shape=None)
            problems = validator.check_tool_feasibility(
                tool=tool, feature=feature, op_purpose=purpose,
                op_id=getattr(op, "id", None))
        except Exception as exc:  # pragma: no cover - defensive
            return self._fail(TOOL_COMPATIBLE,
                              f"tool feasibility could not be evaluated: {exc}",
                              NO_COMPATIBLE_TOOL, critical=False)
        if problems:
            return self._fail(TOOL_COMPATIBLE,
                              getattr(problems[0], "message", str(problems[0])),
                              getattr(problems[0], "code", NO_COMPATIBLE_TOOL))
        return self._ok(TOOL_COMPATIBLE, f"tool {tool.id} D{tool.diameter} fits the feature")

    # -- 4. tool access -----------------------------------------------------
    def _check_tool_access(self, feature: Any) -> ValidityCheck:
        if feature is None:
            return self._skip(TOOL_ACCESS_VALID, "no feature to inspect")
        accessibility = getattr(feature, "accessibility", None)
        if accessibility is None:
            return self._ok(TOOL_ACCESS_VALID, "no accessibility restriction declared")
        value = getattr(accessibility, "value", str(accessibility))
        if value == "inaccessible":
            return self._fail(TOOL_ACCESS_VALID,
                              f"feature {feature.id} is inaccessible along the tool axis",
                              FEATURE_INACCESSIBLE)
        if value == "needs_tilted_tool":
            machine = self.machine
            axes = getattr(machine, "axes", {}) or {}
            machine_type = getattr(machine, "machine_type", None)
            is_three_axis = len(axes) <= 3 or "three_axis" in str(
                getattr(machine_type, "value", machine_type)).lower()
            if is_three_axis:
                return self._fail(TOOL_ACCESS_VALID,
                                  f"feature {feature.id} needs a tilted tool; this machine "
                                  "has no tilt axis",
                                  FEATURE_INACCESSIBLE)
            return self._ok(TOOL_ACCESS_VALID, "tilted tool access available")
        return self._ok(TOOL_ACCESS_VALID, value)

    # -- 5. depth -----------------------------------------------------------
    def _check_depth(self, feature: Any) -> ValidityCheck:
        if feature is None:
            return self._skip(DEPTH_VALID, "no feature to measure")
        depth = getattr(feature, "depth", None)
        top_z = getattr(feature, "top_z", None)
        floor_z = getattr(feature, "floor_z", None)
        if depth is None or not np.isfinite(float(depth)) or float(depth) < 0:
            return self._fail(DEPTH_VALID,
                              f"feature {feature.id} declares invalid depth {depth}",
                              INVALID_GEOMETRY)
        if top_z is None or floor_z is None:
            return self._ok(DEPTH_VALID, f"depth {float(depth):.3f} mm is non-negative")
        top_z, floor_z = float(top_z), float(floor_z)
        if not (np.isfinite(top_z) and np.isfinite(floor_z)):
            return self._fail(DEPTH_VALID,
                              f"feature {feature.id} has non-finite Z limits",
                              INVALID_GEOMETRY)
        if floor_z > top_z + LENGTH_EPS:
            return self._fail(DEPTH_VALID,
                              f"feature {feature.id} floor Z {floor_z:.3f} lies above "
                              f"top Z {top_z:.3f}",
                              INVALID_GEOMETRY)
        machine = self.machine
        if machine is not None:
            z_axis = (getattr(machine, "axes", {}) or {}).get("Z")
            if z_axis is not None:
                lo = float(getattr(z_axis, "travel_min", -np.inf))
                hi = float(getattr(z_axis, "travel_max", np.inf))
                if top_z > hi + LENGTH_EPS or floor_z < lo - LENGTH_EPS:
                    return self._fail(DEPTH_VALID,
                                      f"feature Z range {floor_z:.3f}..{top_z:.3f} leaves "
                                      f"machine Z travel {lo:.1f}..{hi:.1f}",
                                      MACHINE_LIMIT_EXCEEDED)
        return self._ok(DEPTH_VALID,
                        f"depth {float(depth):.3f} mm within Z {floor_z:.3f}..{top_z:.3f}")

    # -- 6. toolpath intersects stock --------------------------------------
    def _check_stock_intersection(self, tool: Any, toolpath: Any) -> ValidityCheck:
        if toolpath is None:
            return self._skip(TOOLPATH_INTERSECTS_STOCK, "no toolpath generated yet")
        stock = getattr(self.setup, "stock", None) if self.setup is not None else None
        smin = getattr(stock, "bounds_min", None)
        smax = getattr(stock, "bounds_max", None)
        if smin is None or smax is None:
            return self._skip(TOOLPATH_INTERSECTS_STOCK, "no stock bounds configured")
        smin = np.asarray(smin, dtype=float)
        smax = np.asarray(smax, dtype=float)
        if not (np.all(np.isfinite(smin)) and np.all(np.isfinite(smax))):
            return self._fail(TOOLPATH_INTERSECTS_STOCK, "stock bounds are not finite",
                              INVALID_GEOMETRY)
        radius = float(getattr(tool, "diameter", 0.0) or 0.0) / 2.0
        lo, hi = smin - radius, smax + radius

        cutting = [s for s in getattr(toolpath, "segments", []) if s.is_cutting]
        if not cutting:
            return self._fail(TOOLPATH_INTERSECTS_STOCK,
                              "toolpath contains no cutting motion inside the stock",
                              INVALID_TOOLPATH)
        from .coords import CoordSpace
        for seg in cutting:
            if getattr(seg, "space", None) is not CoordSpace.SETUP:
                return self._skip(TOOLPATH_INTERSECTS_STOCK,
                                  f"segment space {seg.space} is not setup space")
            start = np.asarray(seg.start, dtype=float)
            end = np.asarray(seg.end, dtype=float)
            seg_lo = np.minimum(start, end)
            seg_hi = np.maximum(start, end)
            if np.all(seg_hi >= lo - STOCK_OVERLAP_EPS) and np.all(seg_lo <= hi + STOCK_OVERLAP_EPS):
                return self._ok(TOOLPATH_INTERSECTS_STOCK,
                                "cutting motion overlaps the stock envelope")
        return self._fail(TOOLPATH_INTERSECTS_STOCK,
                          "every cutting segment lies outside the stock envelope",
                          INVALID_TOOLPATH)

    # -- 7. toolpath reaches target ----------------------------------------
    def _check_reaches_target(self, feature: Any, toolpath: Any,
                              purpose: str, op: Any = None) -> ValidityCheck:
        if toolpath is None:
            return self._skip(TOOLPATH_REACHES_TARGET, "no toolpath generated yet")
        if feature is None:
            return self._skip(TOOLPATH_REACHES_TARGET, "no feature to reach")
        if purpose in _TOP_FACE_PURPOSES:
            return self._ok(TOOLPATH_REACHES_TARGET,
                            "target is the top face of the feature")
        cutting = [s for s in getattr(toolpath, "segments", []) if s.is_cutting]
        if not cutting:
            return self._fail(TOOLPATH_REACHES_TARGET,
                              "toolpath contains no cutting motion",
                              INVALID_TOOLPATH)
        from .coords import CoordSpace
        zs: list[float] = []
        for seg in cutting:
            if getattr(seg, "space", None) is not CoordSpace.SETUP:
                return self._skip(TOOLPATH_REACHES_TARGET,
                                  f"segment space {seg.space} is not setup space")
            zs.extend([float(np.asarray(seg.start, dtype=float)[2]),
                       float(np.asarray(seg.end, dtype=float)[2])])
        floor_z = getattr(op, "final_z", None) if op is not None else None
        if floor_z is None:
            floor_z = getattr(feature, "floor_z", None)
        if floor_z is None:
            return self._skip(TOOLPATH_REACHES_TARGET, "feature has no floor surface")
        deepest = min(zs)
        limit = float(floor_z) + TARGET_REACH_TOLERANCE_MM
        if deepest <= limit:
            return self._ok(TOOLPATH_REACHES_TARGET,
                            f"reaches Z {deepest:.3f} (target floor {float(floor_z):.3f})")
        return self._fail(TOOLPATH_REACHES_TARGET,
                          f"deepest cut Z {deepest:.3f} stops "
                          f"{deepest - float(floor_z):.3f} mm above the target floor "
                          f"{float(floor_z):.3f}",
                          INVALID_TOOLPATH)

    # -- 8. compensation ----------------------------------------------------
    def _check_compensation(self, tool: Any, toolpath: Any) -> ValidityCheck:
        if toolpath is None:
            return self._skip(COMPENSATION_VALID, "no toolpath generated yet")
        meta = getattr(toolpath, "metadata", None) or {}
        mode = str(meta.get("compensation_mode", "none") or "none").lower()
        representation = str(meta.get("toolpath_representation", "center_path")
                             or "center_path").lower()

        if mode == "geometry" and representation == "center_path":
            return self._fail(COMPENSATION_VALID,
                              "double compensation: center-path geometry with full-radius "
                              "control compensation",
                              INVALID_TOOLPATH)
        if representation == "part_boundary_path" and mode in ("none", "off", ""):
            return self._fail(COMPENSATION_VALID,
                              "uncompensated part contour: boundary path without G41/G42 "
                              "would gouge the part by one cutter radius",
                              INVALID_TOOLPATH)

        active = False
        tool_type = getattr(getattr(tool, "type", None), "value", None)
        no_comp = tool_type in {getattr(t, "value", t) for t in _NO_COMP_TOOL_TYPES}
        for i, seg in enumerate(getattr(toolpath, "segments", []) or []):
            comp = (seg.metadata or {}).get("cutter_comp") if seg.metadata else None
            if not comp:
                continue
            action = str(comp.get("action", "on")).lower()
            if action == "on":
                if active:
                    return self._fail(COMPENSATION_VALID,
                                      f"segment {i}: G41/G42 enabled while compensation is "
                                      "already active (missing G40)",
                                      INVALID_TOOLPATH)
                if no_comp:
                    return self._fail(COMPENSATION_VALID,
                                      f"segment {i}: tool {getattr(tool, 'id', '?')} "
                                      f"({tool_type}) cannot use cutter radius compensation",
                                      INVALID_TOOLPATH)
                d_reg = comp.get("d_register")
                if d_reg is not None and not isinstance(d_reg, str):
                    try:
                        numeric_d = float(d_reg)
                    except (TypeError, ValueError):
                        numeric_d = 0.0
                    if numeric_d < 0:
                        return self._fail(COMPENSATION_VALID,
                                          f"segment {i}: invalid D register {d_reg}",
                                          INVALID_TOOLPATH)
                elif isinstance(d_reg, str) and not d_reg.strip().lstrip("D").isdigit():
                    return self._fail(COMPENSATION_VALID,
                                      f"segment {i}: non-numeric D register {d_reg!r}",
                                      INVALID_TOOLPATH)
                if str(comp.get("side", "left")).lower() not in ("left", "right"):
                    return self._fail(COMPENSATION_VALID,
                                      f"segment {i}: invalid compensation side "
                                      f"'{comp.get('side')}'",
                                      INVALID_TOOLPATH)
                active = True
            elif action == "off":
                active = False
        if active:
            # The post processor cancels compensation at the end of every
            # operation (G40 teardown), so an open lifecycle in the toolpath is
            # reported but does not block emission.
            return ValidityCheck(
                name=COMPENSATION_VALID, passed=False, critical=False,
                message="toolpath ends with cutter compensation still active "
                        "(G40 is emitted by the post-processor teardown)",
                code=INVALID_TOOLPATH)
        return self._ok(COMPENSATION_VALID, f"compensation mode '{mode}' is consistent")

    # -- 9. feed ------------------------------------------------------------
    def _check_feed(self, toolpath: Any,
                    feed_limit: Optional[float] = None) -> ValidityCheck:
        if toolpath is None:
            return self._skip(FEED_VALID, "no toolpath generated yet")
        cutting = [s for s in getattr(toolpath, "segments", []) if s.is_cutting]
        if not cutting:
            return self._skip(FEED_VALID, "toolpath has no cutting motion")
        if feed_limit is None:
            feed_limit = _max_feed(self.machine, (getattr(toolpath, "metadata", None) or {}))
        missing = 0
        for i, seg in enumerate(cutting):
            feed = seg.feed
            if feed is None:
                missing += 1
                continue
            try:
                value = float(feed)
            except (TypeError, ValueError):
                return self._fail(FEED_VALID, f"segment {i} has a non-numeric feed {feed!r}",
                                  INVALID_CUTTING_PARAMETERS)
            if not np.isfinite(value) or value <= 0:
                return self._fail(FEED_VALID,
                                  f"segment {i} has invalid feed {feed}",
                                  INVALID_CUTTING_PARAMETERS)
            if feed_limit is not None and value > feed_limit + 1e-6:
                return self._fail(FEED_VALID,
                                  f"segment {i} feed {value:.1f} mm/min exceeds machine "
                                  f"maximum {feed_limit:.1f} mm/min",
                                  MACHINE_LIMIT_EXCEEDED)
        if missing:
            # A cutting move without its own feed word inherits the modal feed.
            # Legal across operations, so it is reported but never blocks.
            return ValidityCheck(
                name=FEED_VALID, passed=False, critical=False,
                message=f"{missing} cutting segment(s) rely on the modal feed",
                code=INVALID_CUTTING_PARAMETERS)
        return self._ok(FEED_VALID, "all cutting segments carry a legal feed")

    # -- 10. rpm ------------------------------------------------------------
    def _check_rpm(self, toolpath: Any,
                   spindle_limits: Optional[tuple] = None) -> ValidityCheck:
        if toolpath is None:
            return self._skip(RPM_VALID, "no toolpath generated yet")
        meta = getattr(toolpath, "metadata", None) or {}
        values: list[float] = []
        for seg in getattr(toolpath, "segments", []) or []:
            if seg.spindle is not None:
                values.append(float(seg.spindle))
        params = meta.get("cutting_parameters") or {}
        if params.get("spindle_rpm") is not None:
            values.append(float(params["spindle_rpm"]))
        if not values:
            return self._skip(RPM_VALID, "no spindle command declared")
        lo = hi = None
        if spindle_limits is not None:
            lo, hi = spindle_limits
        if lo is None or hi is None:
            machine_lo, machine_hi = _spindle_limits(self.machine, meta)
            lo = machine_lo if lo is None else lo
            hi = machine_hi if hi is None else hi
        for value in values:
            if not np.isfinite(value) or value <= 0:
                return self._fail(RPM_VALID, f"invalid spindle speed {value}",
                                  INVALID_CUTTING_PARAMETERS)
            if lo is not None and value < lo - 1e-6:
                return self._fail(RPM_VALID,
                                  f"spindle speed {value:.0f} rpm is below minimum {lo:.0f} rpm",
                                  MACHINE_LIMIT_EXCEEDED)
            if hi is not None and value > hi + 1e-6:
                return self._fail(RPM_VALID,
                                  f"spindle speed {value:.0f} rpm exceeds maximum {hi:.0f} rpm",
                                  MACHINE_LIMIT_EXCEEDED)
        return self._ok(RPM_VALID, f"spindle {values[0]:.0f} rpm inside machine range")

    # -- 11. holder clearance ----------------------------------------------
    def _check_holder_clearance(self, tool: Any, feature: Any,
                                toolpath: Any = None) -> ValidityCheck:
        if tool is None or feature is None:
            return self._skip(HOLDER_CLEARANCE_VALID, "no tool/feature to check reach against")
        depth = float(getattr(feature, "depth", 0.0) or 0.0)
        # An operation only has to reach the depth it actually cuts: a spot drill
        # never needs the full hole depth, a through drill needs breakthrough.
        actual = _actual_cut_depth(toolpath, feature)
        if actual is not None:
            depth = max(0.0, actual)
        flute = float(getattr(tool, "flute_length", 0.0) or 0.0)
        available = flute
        tool_type = getattr(getattr(tool, "type", None), "value", None)
        if tool_type in ("drill", "spot_drill", "center_drill"):
            # the conical tip of a drill cuts beyond the flute length
            point_angle = float(getattr(tool, "point_angle", 118.0) or 118.0)
            radius = float(getattr(tool, "diameter", 0.0) or 0.0) / 2.0
            half = np.radians(max(point_angle, 1.0) / 2.0)
            available += radius / np.tan(half) if half > 0 else 0.0
        if available + 1e-6 < depth:
            return self._fail(HOLDER_CLEARANCE_VALID,
                              f"tool {tool.id} can only reach {available:.2f} mm "
                              f"(flute {flute:.2f} mm) but the feature is {depth:.2f} mm deep",
                              TOOL_REACH_INSUFFICIENT)
        holder = getattr(tool, "holder", None)
        if holder is not None and getattr(feature, "diameter", None):
            holder_d = float(getattr(holder, "diameter", 0.0) or 0.0)
            feature_type = getattr(getattr(feature, "type", None), "value", str(getattr(feature, "type", "")))
            if feature_type in ("hole", "through_hole", "blind_hole", "bore",
                                "through_bore", "blind_bore") and holder_d > float(feature.diameter) + 1e-6:
                return self._fail(HOLDER_CLEARANCE_VALID,
                                  f"holder D{holder_d:.1f} collides in the "
                                  f"D{float(feature.diameter):.1f} bore",
                                  COLLISION_DETECTED)
        return self._ok(HOLDER_CLEARANCE_VALID,
                        f"reach {available:.2f} mm covers depth {depth:.2f} mm")

    # -- 12. setup ----------------------------------------------------------
    def _check_setup(self) -> ValidityCheck:
        setup = self.setup
        if setup is None:
            return self._skip(SETUP_VALID, "no setup supplied")
        transform = getattr(setup, "model_to_setup", None)
        if transform is None:
            return self._fail(SETUP_VALID, "setup has no model-to-setup transform",
                              INVALID_SETUP)
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return self._fail(SETUP_VALID, "model-to-setup transform must be a finite 4x4 matrix",
                              INVALID_SETUP)
        try:
            det = float(np.linalg.det(matrix[:3, :3]))
        except np.linalg.LinAlgError:
            det = 0.0
        if abs(det) < 1e-9:
            return self._fail(SETUP_VALID, "model-to-setup transform is singular",
                              INVALID_SETUP)
        if det <= 0:
            return self._fail(SETUP_VALID, "model-to-setup transform mirrors the part",
                              INVALID_SETUP)
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            return self._fail(SETUP_VALID, "model-to-setup rotation is not orthonormal",
                              INVALID_SETUP)
        stock = getattr(setup, "stock", None)
        if stock is not None:
            smin = getattr(stock, "bounds_min", None)
            smax = getattr(stock, "bounds_max", None)
            if smin is None or smax is None:
                return self._fail(SETUP_VALID, "setup stock has no bounds", INVALID_SETUP)
            smin = np.asarray(smin, dtype=float)
            smax = np.asarray(smax, dtype=float)
            if not (np.all(np.isfinite(smin)) and np.all(np.isfinite(smax))):
                return self._fail(SETUP_VALID, "setup stock bounds are not finite",
                                  INVALID_SETUP)
            if np.any(smax - smin <= 0):
                return self._fail(SETUP_VALID, "setup stock bounds are inverted or empty",
                                  INVALID_SETUP)
        return self._ok(SETUP_VALID, "setup transform and stock are well formed")


# ------------------------------------------------------------ post-time use --
def _actual_cut_depth(toolpath: Any, feature: Any) -> Optional[float]:
    """Deepest point the toolpath cuts below the feature's top face (setup space)."""
    if toolpath is None:
        return None
    top_z = getattr(feature, "top_z", None)
    if top_z is None:
        return None
    from .coords import CoordSpace
    zs: list[float] = []
    for seg in getattr(toolpath, "segments", []) or []:
        if not seg.is_cutting:
            continue
        if getattr(seg, "space", None) is not CoordSpace.SETUP:
            return None
        zs.append(float(np.asarray(seg.start, dtype=float)[2]))
        zs.append(float(np.asarray(seg.end, dtype=float)[2]))
    if not zs:
        return None
    return float(top_z) - min(zs)


def _max_feed(machine: Any, metadata: dict) -> Optional[float]:
    limit = None
    if machine is not None:
        feeds = [float(getattr(ax, "max_feed", 0.0) or 0.0)
                 for ax in (getattr(machine, "axes", {}) or {}).values()]
        feeds = [f for f in feeds if f > 0]
        if feeds:
            limit = max(feeds)
    params = metadata.get("cutting_parameters") or {}
    meta_limit = params.get("machine_max_feed")
    if meta_limit:
        try:
            meta_limit = float(meta_limit)
        except (TypeError, ValueError):
            meta_limit = 0.0
        if meta_limit > 0:
            limit = meta_limit if limit is None else min(limit, meta_limit)
    return limit


def _spindle_limits(machine: Any, metadata: dict) -> tuple[Optional[float], Optional[float]]:
    lo = hi = None
    if machine is not None:
        lo = getattr(machine, "spindle_min_rpm", None)
        hi = getattr(machine, "spindle_max_rpm", None)
        lo = float(lo) if lo is not None else None
        hi = float(hi) if hi is not None else None
    params = metadata.get("cutting_parameters") or {}
    meta_lo = params.get("machine_min_rpm")
    meta_hi = params.get("machine_max_rpm")
    try:
        meta_lo = float(meta_lo) if meta_lo else None
        meta_hi = float(meta_hi) if meta_hi else None
    except (TypeError, ValueError):
        meta_lo = meta_hi = None
    if meta_lo:
        lo = meta_lo if lo is None else max(lo, meta_lo)
    if meta_hi:
        hi = meta_hi if hi is None else min(hi, meta_hi)
    return lo, hi


def check_toolpath_for_post(toolpath: Any, machine: Any = None,
                            capability: Any = None) -> ValidityReport:
    """Toolpath-local checks (geometry, feed, rpm, compensation) at emission time.

    These run on every toolpath, stamped or not; the remaining checks were
    evaluated at generation time and travel with the toolpath's stamp.
    """
    machine = machine or getattr(toolpath, "machine", None)
    metadata = getattr(toolpath, "metadata", None) or {}
    report = ValidityReport(
        operation_id=str(getattr(toolpath, "operation_id", "")),
        tool_id=str(getattr(toolpath, "tool_id", "")),
        purpose=str(getattr(toolpath, "purpose", "")),
        skipped=bool(metadata.get("skipped"))
                or bool(getattr(toolpath, "is_empty_or_skipped", False)),
    )
    if report.skipped:
        report.checks = [OperationValidityChecker._skip(n, "toolpath skipped")
                         for n in (GEOMETRY_EXISTS, FEED_VALID, RPM_VALID, COMPENSATION_VALID)]
        return report

    probe = OperationValidityChecker(context=None, setup=None)
    probe.machine = machine

    # Controller-side limits are the authority once code is being emitted.
    feed_limit = None
    spindle_limits = None
    if capability is not None:
        cap_feed = getattr(capability, "max_feed_mm_min", None)
        feed_limit = float(cap_feed) if cap_feed else None
        cap_lo = getattr(capability, "min_spindle_rpm", None)
        cap_hi = getattr(capability, "max_spindle_rpm", None)
        spindle_limits = (float(cap_lo) if cap_lo else None,
                          float(cap_hi) if cap_hi else None)

    report.checks = [
        probe._check_geometry(None, toolpath),
        probe._check_feed(toolpath, feed_limit=feed_limit),
        probe._check_rpm(toolpath, spindle_limits=spindle_limits),
        probe._check_compensation(None, toolpath),
    ]
    return report


def stamped_report(toolpath: Any) -> Optional[ValidityReport]:
    """Validity report stamped onto a toolpath at generation time, if any."""
    metadata = getattr(toolpath, "metadata", None) or {}
    data = metadata.get("validity")
    if not isinstance(data, dict):
        return None
    return ValidityReport.from_dict(data)


def collect_blocking_failures(toolpaths: Iterable[Any], machine: Any = None,
                              capability: Any = None) -> list[dict]:
    """Critical failures across stamped + post-time checks, de-duplicated."""
    blocking: list[dict] = []
    seen: set[tuple] = set()

    def add(report: ValidityReport) -> None:
        for check in report.critical_failures:
            key = (report.operation_id, check.name)
            if key in seen:
                continue
            seen.add(key)
            blocking.append({
                "operation_id": report.operation_id,
                "tool_id": report.tool_id,
                "check": check.name,
                "code": check.code,
                "message": check.message,
            })

    for tp in toolpaths or []:
        stamped = stamped_report(tp)
        if stamped is not None and not stamped.skipped:
            add(stamped)
        add(check_toolpath_for_post(tp, machine=machine, capability=capability))
    return blocking


def assert_valid_for_post(toolpaths: Sequence[Any], machine: Any = None,
                          capability: Any = None) -> None:
    """Block G-code generation when any critical operation check failed."""
    failures = collect_blocking_failures(toolpaths, machine=machine, capability=capability)
    if not failures:
        return
    detail = "; ".join(
        f"{f['operation_id'] or 'toolpath'} [{f['check']}] {f['message']}"
        for f in failures[:8]
    )
    if len(failures) > 8:
        detail += f" (+{len(failures) - 8} more)"
    raise CamError(
        code=GCODE_VALIDATION_FAILED,
        message=(f"G-code generation blocked: {len(failures)} critical operation "
                 f"validity check(s) failed — {detail}"),
        stage="gcode_gate",
        operation_id=str(failures[0]["operation_id"] or "") or None,
        tool_id=str(failures[0]["tool_id"] or "") or None,
        severity=ErrorSeverity.ERROR,
        context={"blocking_failures": failures},
    )
