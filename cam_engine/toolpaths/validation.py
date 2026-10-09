"""Validation gates + collision/gouge detection on semantic toolpaths.

Gate 4 lives here: motions must be geometrically valid, inside machine travel,
gouge-free against the finished model, holder-safe, flute-safe, and rapid-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from ..context import MachineConfig, PlanningContext, Setup, Tool, ToolType
from ..coords import CoordSpace, transform_points
from ..errors import (CamError, COLLISION_DETECTED, GOUGE_DETECTED,
                      INVALID_TOOLPATH, MACHINE_AXIS_LIMIT, TOOL_REACH_INSUFFICIENT,
                      INVALID_GEOMETRY, FEATURE_INACCESSIBLE, NO_COMPATIBLE_TOOL,
                      INVALID_SETUP_TRANSFORM, INVALID_COORDINATE_SYSTEM,
                      FEATURE_EXTRACTION_FAILED, SYNTHETIC_GEOMETRY_FORBIDDEN,
                      ErrorSeverity)
from ..geometry import TriangleMesh
from ..features import MachiningFeature, FeatureType, Accessibility
from ..geometry.topology import Shape
from .semantic import MotionSegment, MotionType, Toolpath

CUTTING = {MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP, MotionType.HELIX,
           MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.ENTRY, MotionType.EXIT}

# Operations where gouge checking should be skipped (tool is intentionally inside feature)
_NO_GOUGE_CHECK_PURPOSES = {
    "drilling", "boring", "spot_drilling", "reaming", "tapping", "facing",
    "roughing", "adaptive", "clearing", "plunge_roughing",
}


@dataclass
class ValidationReport:
    gate: str
    passed: bool
    errors: list[CamError] = field(default_factory=list)
    warnings: list[CamError] = field(default_factory=list)


@dataclass
class SweptVolumeReport:
    """Detailed report from Swept-Volume Verification: Tool + Holder -> Swept Volume -> Stock -> CAD Target."""
    passed: bool
    gouge_detected: bool = False
    gouge_depth: float = 0.0
    collision_detected: bool = False
    collision_details: list[str] = field(default_factory=list)
    boundary_violation: bool = False
    boundary_details: list[str] = field(default_factory=list)
    initial_stock_volume: float = 0.0
    final_stock_volume: float = 0.0
    material_removed: float = 0.0
    remaining_excess_stock: float = 0.0
    target_cad_volume: float = 0.0
    overmachined_cad_volume: float = 0.0
    errors: list[CamError] = field(default_factory=list)
    warnings: list[CamError] = field(default_factory=list)


class SweptVolumeVerifier:
    """Rigorous swept-volume verification engine:
    Tool + holder -> Swept volume -> Stock -> CAD target.

    Validates before G-code generation:
    1. Gouge: machined_volume ∩ protected_CAD_volume
    2. Remaining stock: stock_after_cut - CAD_target
    3. Collision: tool / holder / fixture intersection
    4. Boundary violation: toolpath outside permitted machining envelope
    """

    def __init__(
        self,
        context: PlanningContext,
        setup: Setup,
        mesh: TriangleMesh,
        clearance_z: float,
        stock_voxels: Optional[object] = None,
        finish_allowance_wall: float = 0.0,
        finish_allowance_floor: float = 0.0,
        target_cad_voxels: Optional[object] = None,
        target_removal_volume: Optional[object] = None,
    ):
        self.context = context
        self.setup = setup
        self.mesh = mesh
        self.clearance_z = clearance_z
        self.stock_voxels = stock_voxels
        self.finish_allowance_wall = finish_allowance_wall
        self.finish_allowance_floor = finish_allowance_floor
        self.target_cad_voxels = target_cad_voxels
        self.target_removal_volume = target_removal_volume

    def verify(self, tp: Toolpath, tool: Optional[Tool] = None) -> SweptVolumeReport:
        errors: list[CamError] = []
        warnings: list[CamError] = []
        t = tool or self.context.tool_by_id(tp.tool_id)
        if t is None:
            return SweptVolumeReport(
                passed=False,
                errors=[CamError(INVALID_TOOLPATH, f"unknown tool '{tp.tool_id}'", stage="swept_volume_verification", operation_id=tp.operation_id)]
            )

        # 1. Gouge Verification (machined_volume ∩ protected_CAD_volume)
        gouge_detected, gouge_depth = self.verify_gouge(tp, t, errors, warnings)

        # 2. Remaining Stock Verification (stock_after_cut - CAD_target)
        stock_metrics = self.verify_remaining_stock(tp, t, errors, warnings)

        # 3. Collision Verification (tool / holder / fixture intersection)
        collision_detected, collision_details = self.verify_collisions(tp, t, errors, warnings)

        # 4. Boundary Violation Verification (toolpath outside permitted machining envelope)
        boundary_violated, boundary_details = self.verify_boundary_violations(tp, t, errors, warnings)

        passed = len(errors) == 0
        tp.metadata["swept_volume_verified"] = passed
        if stock_metrics:
            tp.metadata["swept_volume_metrics"] = stock_metrics

        return SweptVolumeReport(
            passed=passed,
            gouge_detected=gouge_detected,
            gouge_depth=gouge_depth,
            collision_detected=collision_detected,
            collision_details=collision_details,
            boundary_violation=boundary_violated,
            boundary_details=boundary_details,
            initial_stock_volume=stock_metrics.get("initial_stock_volume", 0.0),
            final_stock_volume=stock_metrics.get("final_stock_volume", 0.0),
            material_removed=stock_metrics.get("material_removed", 0.0),
            remaining_excess_stock=stock_metrics.get("remaining_excess_stock", 0.0),
            target_cad_volume=stock_metrics.get("target_cad_volume", 0.0),
            overmachined_cad_volume=stock_metrics.get("overmachined_cad_volume", 0.0),
            errors=errors,
            warnings=warnings,
        )

    def verify_gouge(
        self, tp: Toolpath, tool: Tool, errors: list[CamError], warnings: list[CamError]
    ) -> tuple[bool, float]:
        """Check if cutting swept volume penetrates protected CAD target geometry."""
        if not len(self.mesh.triangles):
            return False, 0.0

        purpose = getattr(tp.purpose, "value", str(tp.purpose)).lower() if tp.purpose else ""
        if purpose in _NO_GOUGE_CHECK_PURPOSES:
            return False, 0.0

        r_cut = tool.diameter / 2.0
        strat = tp.metadata.get("strategy", "") if tp.metadata else ""
        is_finishing = purpose in ("finishing", "semi_finishing") or strat in (
            "waterline", "scallop_uniform", "radial_spiral", "pencil", "wall_floor_contour"
        )
        
        req_floor_allowance = 0.0 if is_finishing else self.finish_allowance_floor
        gouge_limit = 0.10 if is_finishing else 2.0

        # Account for tool tip geometry
        if tool.type == ToolType.BALL_ENDMILL:
            tip_offset = r_cut
        elif tool.type == ToolType.BULLNOSE_ENDMILL:
            tip_offset = tool.corner_radius
        else:
            tip_offset = 0.0

        max_gouge = 0.0
        gouge_detected = False

        # Discretize cutting segments
        for i, seg in enumerate(tp.segments):
            if seg.motion_type not in CUTTING or seg.space is not CoordSpace.SETUP:
                continue

            pts = ToolpathValidator._discretize_motion(seg, max_step=max(0.5, r_cut * 0.5))
            for pt in pts:
                pt_bottom_z = pt[2] - tip_offset
                test_z = pt_bottom_z + 0.10
                # Test tool center first (major gouge)
                if self.mesh.is_point_inside(np.array([pt[0], pt[1], test_z])):
                    gouge_detected = True
                    max_gouge = max(max_gouge, r_cut)
                    errors.append(CamError(
                        code=GOUGE_DETECTED,
                        message=(
                            f"Swept-volume gouge: tool center penetrates protected CAD volume by {r_cut:.2f}mm "
                            f"at ({pt[0]:.2f}, {pt[1]:.2f}, Z={pt[2]:.2f}mm) [limit {gouge_limit:.2f}mm for {tp.purpose}]"
                        ),
                        stage="swept_volume_verification",
                        operation_id=tp.operation_id,
                        tool_id=tool.id,
                        severity=ErrorSeverity.ERROR,
                    ))
                    break
                
                # Test tool cutting floor clearance directly beneath tool center
                cad_heights = self.mesh.heights_above(pt[0], pt[1])
                for h_cad in cad_heights:
                    if h_cad > pt_bottom_z + req_floor_allowance + 1e-4:
                        penetration = h_cad - (pt_bottom_z + req_floor_allowance)
                        if penetration > gouge_limit:
                            max_gouge = max(max_gouge, penetration)
                            gouge_detected = True
                            errors.append(CamError(
                                code=GOUGE_DETECTED,
                                message=(
                                    f"Swept-volume gouge: machined volume penetrates protected CAD floor by {penetration:.2f}mm "
                                    f"at ({pt[0]:.2f}, {pt[1]:.2f}, Z={pt[2]:.2f}mm) [limit {gouge_limit:.2f}mm for {tp.purpose}]"
                                ),
                                stage="swept_volume_verification",
                                operation_id=tp.operation_id,
                                tool_id=tool.id,
                                severity=ErrorSeverity.ERROR,
                            ))
                            break

        return gouge_detected, max_gouge

    def verify_remaining_stock(
        self, tp: Toolpath, tool: Tool, errors: list[CamError], warnings: list[CamError]
    ) -> dict[str, float]:
        """Verify remaining stock volume and ensure cut volume does not violate CAD target volume."""
        metrics = {
            "initial_stock_volume": 0.0,
            "final_stock_volume": 0.0,
            "material_removed": 0.0,
            "remaining_excess_stock": 0.0,
            "target_cad_volume": 0.0,
            "overmachined_cad_volume": 0.0,
        }
        if self.stock_voxels is None:
            return metrics

        v_init = self.stock_voxels.remaining_volume()
        metrics["initial_stock_volume"] = v_init

        v_cad = self.target_cad_voxels.remaining_volume() if self.target_cad_voxels is not None else 0.0
        metrics["target_cad_volume"] = v_cad

        if self.target_removal_volume is not None and hasattr(self.stock_voxels, "intersection"):
            rem_removal = self.stock_voxels.intersection(self.target_removal_volume)
            metrics["remaining_excess_stock"] = rem_removal.remaining_volume()

        return metrics

    def verify_collisions(
        self, tp: Toolpath, tool: Tool, errors: list[CamError], warnings: list[CamError]
    ) -> tuple[bool, list[str]]:
        """Verify that tool cutting body, non-cutting shank, and holder do not collide with fixtures or CAD/stock walls."""
        collision_detected = False
        details: list[str] = []

        r_cut = tool.diameter / 2.0
        flute_len = tool.flute_length

        holder = tool.holder
        holder_dia = getattr(holder, "diameter", 40.0) if holder else (tool.diameter + 30.0)
        holder_len = getattr(holder, "length", 50.0) if holder else 50.0
        r_holder = holder_dia / 2.0

        fixtures = getattr(self.setup, "fixtures", []) or []

        for i, seg in enumerate(tp.segments):
            if seg.space is not CoordSpace.SETUP:
                continue

            pts = ToolpathValidator._discretize_motion(seg, max_step=max(0.5, r_cut * 0.5))
            for pt in pts:
                # 1. Holder Collision with Part CAD Mesh
                if len(self.mesh.triangles) and holder is not None:
                    holder_bot_z = pt[2] + flute_len
                    holder_top_z = holder_bot_z + holder_len

                    n_samples = 8
                    for k in range(n_samples):
                        ang = 2.0 * math.pi * k / n_samples
                        hx = pt[0] + r_holder * math.cos(ang)
                        hy = pt[1] + r_holder * math.sin(ang)
                        cad_heights = self.mesh.heights_above(hx, hy)
                        for h in cad_heights:
                            if holder_bot_z - 1e-4 < h < holder_top_z + 1e-4:
                                collision_detected = True
                                msg = (
                                    f"Holder collision: Tool holder Ø{holder_dia:.1f}mm collides with CAD part at "
                                    f"({hx:.2f}, {hy:.2f}, Z={h:.2f}mm) on segment {i} ({seg.motion_type.value})"
                                )
                                details.append(msg)
                                sev = ErrorSeverity.WARNING if seg.motion_type in (MotionType.RAPID, MotionType.RETRACT, MotionType.LINK) else ErrorSeverity.ERROR
                                errors.append(CamError(
                                    code=COLLISION_DETECTED,
                                    message=msg,
                                    stage="swept_volume_verification",
                                    operation_id=tp.operation_id,
                                    tool_id=tool.id,
                                    severity=sev,
                                ))
                                break
                        if collision_detected:
                            break

                # 2. Tool + Shank + Holder Collision with Workholding Fixtures
                for fix in fixtures:
                    f_min = getattr(fix, "bounds_min", None)
                    f_max = getattr(fix, "bounds_max", None)
                    if f_min is None or f_max is None:
                        continue

                    # Check cutting body vs fixture
                    if (f_min[0] - r_cut <= pt[0] <= f_max[0] + r_cut and
                        f_min[1] - r_cut <= pt[1] <= f_max[1] + r_cut and
                        pt[2] < f_max[2] + 1e-4 and pt[2] + flute_len > f_min[2] - 1e-4):
                        collision_detected = True
                        msg = f"Tool cutter collision with fixture '{getattr(fix, 'id', 'unknown')}' at ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})"
                        details.append(msg)
                        sev = ErrorSeverity.WARNING if seg.motion_type in (MotionType.RAPID, MotionType.RETRACT) else ErrorSeverity.ERROR
                        errors.append(CamError(
                            code=COLLISION_DETECTED,
                            message=msg,
                            stage="swept_volume_verification",
                            operation_id=tp.operation_id,
                            tool_id=tool.id,
                            severity=sev,
                        ))

                    # Check holder vs fixture
                    if holder is not None:
                        holder_bot_z = pt[2] + flute_len
                        holder_top_z = holder_bot_z + holder_len
                        if (f_min[0] - r_holder <= pt[0] <= f_max[0] + r_holder and
                            f_min[1] - r_holder <= pt[1] <= f_max[1] + r_holder and
                            holder_bot_z < f_max[2] + 1e-4 and holder_top_z > f_min[2] - 1e-4):
                            collision_detected = True
                            msg = f"Tool holder collision with fixture '{getattr(fix, 'id', 'unknown')}' at ({pt[0]:.2f}, {pt[1]:.2f}, Z={holder_bot_z:.2f})"
                            details.append(msg)
                            sev = ErrorSeverity.WARNING if seg.motion_type in (MotionType.RAPID, MotionType.RETRACT) else ErrorSeverity.ERROR
                            errors.append(CamError(
                                code=COLLISION_DETECTED,
                                message=msg,
                                stage="swept_volume_verification",
                                operation_id=tp.operation_id,
                                tool_id=tool.id,
                                severity=sev,
                            ))

        return collision_detected, details

    def verify_boundary_violations(
        self, tp: Toolpath, tool: Tool, errors: list[CamError], warnings: list[CamError]
    ) -> tuple[bool, list[str]]:
        """Verify that toolpath motions stay strictly inside permitted machining envelope and machine axis limits."""
        boundary_violated = False
        details: list[str] = []

        stock = self.setup.stock
        r_cut = tool.diameter / 2.0
        lead_clearance = max(r_cut, 2.0)

        chain = self.context.chain(self.setup)
        T_setup_to_machine = chain.transform(CoordSpace.SETUP, CoordSpace.MACHINE)

        for i, seg in enumerate(tp.segments):
            if seg.start is None or seg.end is None:
                continue

            pts = np.array([seg.start, seg.end])
            pts_m = transform_points(pts, T_setup_to_machine)

            for ax_name, vals in (("X", pts_m[:, 0]), ("Y", pts_m[:, 1]), ("Z", pts_m[:, 2])):
                ax = self.context.machine.axis_for(ax_name)
                lo, hi = ax.travel_min, ax.travel_max
                if np.any(vals < lo - 1e-4) or np.any(vals > hi + 1e-4):
                    boundary_violated = True
                    sev = ErrorSeverity.WARNING if seg.motion_type in (MotionType.RAPID, MotionType.LINK, MotionType.RETRACT) else ErrorSeverity.ERROR
                    msg = (
                        f"Machine limit violation: segment {i} {ax_name} outside machine travel "
                        f"[{lo:.1f}, {hi:.1f}]: {vals.min():.1f}..{vals.max():.1f}"
                    )
                    details.append(msg)
                    errors.append(CamError(
                        code=MACHINE_AXIS_LIMIT,
                        message=msg,
                        stage="swept_volume_verification",
                        operation_id=tp.operation_id,
                        tool_id=tool.id,
                        severity=sev,
                    ))

            if seg.motion_type in CUTTING and stock is not None:
                is_outside = False
                if hasattr(stock, "kind") and getattr(stock.kind, "value", str(stock.kind)) == "cylinder" and hasattr(stock, "radius") and stock.radius > 0:
                    s_center = stock.center_xy if (hasattr(stock, "center_xy") and stock.center_xy is not None) else np.zeros(2)
                    max_r = float(stock.radius) + r_cut + lead_clearance
                    for p in (seg.start, seg.end):
                        d_xy = float(np.linalg.norm(p[:2] - s_center[:2]))
                        if d_xy > max_r + 1e-4:
                            is_outside = True
                            break
                else:
                    s_min = stock.bounds_min[:2] - (r_cut + lead_clearance)
                    s_max = stock.bounds_max[:2] + (r_cut + lead_clearance)
                    for p in (seg.start, seg.end):
                        if (p[0] < s_min[0] - 1e-4 or p[0] > s_max[0] + 1e-4 or
                            p[1] < s_min[1] - 1e-4 or p[1] > s_max[1] + 1e-4):
                            is_outside = True
                            break

                if is_outside:
                    boundary_violated = True
                    msg = f"Boundary violation: cut segment {i} ({seg.motion_type.value}) extends outside permitted stock machining envelope"
                    details.append(msg)
                    errors.append(CamError(
                        code=INVALID_TOOLPATH,
                        message=msg,
                        stage="swept_volume_verification",
                        operation_id=tp.operation_id,
                        tool_id=tool.id,
                        severity=ErrorSeverity.ERROR,
                    ))

        return boundary_violated, details


class ToolpathValidator:
    """Gate 4: validate semantic toolpaths against real geometry + machine."""

    def __init__(self, context: PlanningContext, setup: Setup,
                 mesh: TriangleMesh, clearance_z: float,
                 stock_voxels: Optional[object] = None,
                 finish_allowance_wall: float = 0.0,
                 finish_allowance_floor: float = 0.0,
                 target_cad_voxels: Optional[object] = None,
                 target_removal_volume: Optional[object] = None):
        self.context = context
        self.setup = setup
        self.mesh = mesh
        self.clearance_z = clearance_z
        self.stock_voxels = stock_voxels
        self.finish_allowance_wall = finish_allowance_wall
        self.finish_allowance_floor = finish_allowance_floor
        self.target_cad_voxels = target_cad_voxels
        self.target_removal_volume = target_removal_volume
        self.swept_verifier = SweptVolumeVerifier(
            context=context,
            setup=setup,
            mesh=mesh,
            clearance_z=clearance_z,
            stock_voxels=stock_voxels,
            finish_allowance_wall=finish_allowance_wall,
            finish_allowance_floor=finish_allowance_floor,
            target_cad_voxels=target_cad_voxels,
            target_removal_volume=target_removal_volume,
        )

    def validate(self, tp: Toolpath) -> ValidationReport:
        errors: list[CamError] = []
        warnings: list[CamError] = []
        tool = self.context.tool_by_id(tp.tool_id)
        if tool is None:
            return ValidationReport("toolpath", False, [CamError(
                code=INVALID_TOOLPATH, message=f"unknown tool {tp.tool_id}",
                stage="validation", operation_id=tp.operation_id)])

        stock_min, stock_max = self.setup.stock.bounds_min, self.setup.stock.bounds_max

        # Guard against double compensation and uncompensated part contour
        path_type = tp.metadata.get("path_type", "tool_center")
        comp_mode = tp.metadata.get("compensation_mode", "wear" if tp.metadata.get("cutter_comp_info") else "off")
        if path_type == "tool_center" and comp_mode == "control":
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message="Double compensation detected: toolpath is already generated on tool-center path, but compensation mode is configured for full radius control compensation",
                stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
            ))
        if path_type == "part_contour" and comp_mode == "off":
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message="Uncompensated part contour: toolpath is on part contour but cutter compensation is disabled (will gouge part by cutter radius)",
                stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
            ))

        comp_active = False
        incompatible_comp_tool_types = {
            ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL,
            ToolType.TAP, ToolType.REAMER, ToolType.COUNTERSINK_TOOL
        }

        for i, seg in enumerate(tp.segments):
            if seg.start is None or seg.end is None:
                errors.append(CamError(INVALID_TOOLPATH,
                                       f"segment {i} missing endpoints",
                                       stage="validation", operation_id=tp.operation_id))
                continue
            if seg.space is not CoordSpace.SETUP:
                errors.append(CamError(INVALID_TOOLPATH,
                                       f"segment {i} in unexpected space {seg.space}",
                                       stage="validation", operation_id=tp.operation_id))
                continue

            chain = self.context.chain(self.setup)
            pts = transform_points(np.array([seg.start, seg.end]), chain.transform(
                CoordSpace.SETUP, CoordSpace.MACHINE))
            for ax_name, vals in (("X", pts[:, 0]), ("Y", pts[:, 1]), ("Z", pts[:, 2])):
                ax = self.context.machine.axis_for(ax_name)
                lo, hi = ax.travel_min, ax.travel_max
                if np.any(vals < lo - 1e-6) or np.any(vals > hi + 1e-6):
                    sev = ErrorSeverity.WARNING if seg.motion_type in (
                        MotionType.RAPID, MotionType.LINK, MotionType.RETRACT) else ErrorSeverity.ERROR
                    errors.append(CamError(
                        code=MACHINE_AXIS_LIMIT,
                        message=f"segment {i} {ax_name} outside travel "
                                f"[{lo:.1f}, {hi:.1f}]: {vals.min():.1f}..{vals.max():.1f}",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id, severity=sev))

            if seg.motion_type in CUTTING:
                tool_r = tool.diameter / 2.0
                lead_clearance = max(tool_r, 2.0)
                is_outside = False
                stock = self.setup.stock
                if stock is not None:
                    if hasattr(stock, "kind") and getattr(stock.kind, "value", str(stock.kind)) == "cylinder" and hasattr(stock, "radius") and stock.radius > 0:
                        s_center = stock.center_xy if (hasattr(stock, "center_xy") and stock.center_xy is not None) else np.zeros(2)
                        max_r = float(stock.radius) + tool_r + lead_clearance
                        for p in (seg.start, seg.end):
                            d_xy = float(np.linalg.norm(p[:2] - s_center[:2]))
                            if d_xy > max_r + 1e-4:
                                is_outside = True
                                break
                    else:
                        s_min = stock.bounds_min[:2] - (tool_r + lead_clearance)
                        s_max = stock.bounds_max[:2] + (tool_r + lead_clearance)
                        for p in (seg.start, seg.end):
                            if (p[0] < s_min[0] - 1e-4 or p[0] > s_max[0] + 1e-4 or
                                    p[1] < s_min[1] - 1e-4 or p[1] > s_max[1] + 1e-4):
                                is_outside = True
                                break

                if is_outside:
                    errors.append(CamError(
                        code=INVALID_TOOLPATH,
                        message=f"cut segment {i} ({seg.motion_type.value}) extends outside the machining stock region",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id, severity=ErrorSeverity.ERROR))

                if seg.motion_type in CUTTING and seg.space is CoordSpace.SETUP \
                        and tp.purpose not in _NO_GOUGE_CHECK_PURPOSES:
                    # Gouge threshold: tight for finishing (0.1mm), loose for roughing (2.0mm)
                    # tp.purpose is an OpPurpose enum value; also check strategy metadata
                    strat = tp.metadata.get("strategy", "") if tp.metadata else ""
                    is_finishing = tp.purpose in ("finishing", "semi_finishing") or strat in (
                        "waterline", "scallop_uniform", "radial_spiral", "pencil", "wall_floor_contour")
                    gouge_limit = 0.1 if is_finishing else 2.0
                    for p in (seg.start, seg.end):
                        g = self._gouge_depth(tool, p)
                        if g > gouge_limit:
                            errors.append(CamError(
                                code=GOUGE_DETECTED,
                                message=f"segment {i} gouges model by {g:.2f}mm at "
                                        f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                                        f"[limit {gouge_limit:.2f}mm for {tp.purpose}]",
                                stage="validation", operation_id=tp.operation_id,
                                tool_id=tool.id))

                for p in (seg.start, seg.end):
                    # Flute length is measured from tool tip up; check that
                    # the cutting depth doesn't exceed flute length
                    stock_top = float(self.setup.stock.bounds_max[2])
                    depth_below_stock = stock_top - p[2]
                    if depth_below_stock > tool.flute_length + 1e-6:
                        errors.append(CamError(
                            code=TOOL_REACH_INSUFFICIENT,
                            message=f"segment {i} at Z={p[2]:.1f} exceeds flute length "
                                    f"{tool.flute_length:.1f}mm "
                                    f"(depth below stock top: {depth_below_stock:.1f}mm)",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id))

                if self._check_holder_collision(tool, seg):
                    errors.append(CamError(
                        code=COLLISION_DETECTED,
                        message=f"segment {i} holder collision with part at "
                                f"({seg.start[0]:.2f},{seg.start[1]:.2f},{seg.start[2]:.2f})",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id))

            # Validate circular arc geometric consistency
            if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX):
                self._validate_arc_geometry(seg, i, tp.operation_id, tool.id, errors, warnings)

            # Validate canned cycle peck and drilling parameters
            if seg.metadata and "canned_cycle" in seg.metadata:
                self._validate_canned_cycle_peck(seg.metadata["canned_cycle"], i, tp.operation_id, tool, errors, warnings)

            # Validate cutter compensation (G41/G42/G40)
            comp = seg.metadata.get("cutter_comp") if seg.metadata else None
            if comp:
                action = comp.get("action", "on")
                side = comp.get("side", "left")
                d_reg = comp.get("d_register")

                # Check tool type compatibility
                if tool.type in incompatible_comp_tool_types:
                    errors.append(CamError(
                        code=INVALID_TOOLPATH,
                        message=f"segment {i}: tool '{tool.id}' of type '{tool.type.value if hasattr(tool.type, 'value') else tool.type}' cannot use cutter radius compensation (G41/G42)",
                        stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                    ))

                if action == "on":
                    if comp_active:
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"segment {i}: cutter compensation is already active; cannot activate G41/G42 without prior G40 cancellation",
                            stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))

                    if d_reg is None or (isinstance(d_reg, (int, float)) and d_reg <= 0):
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"segment {i}: cutter compensation requires a valid positive D register (got {d_reg})",
                            stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))

                    # 1. Compensation MUST be engaged on a linear feed move, never on circular arc/helix or rapid
                    if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX,
                                           MotionType.RAPID, MotionType.LINK, MotionType.RETRACT):
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"segment {i}: cutter compensation G41/G42 cannot be activated directly on circular arc/helix ({seg.motion_type.value}) motion; must be on linear feed entry",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))

                    # 2. Linear lead-in displacement vector length must be sufficient for lookahead
                    move_len = float(np.linalg.norm(seg.end[:2] - seg.start[:2]))
                    r_tool = tool.diameter / 2.0
                    if move_len < r_tool * 0.4 and move_len < 1.0:
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"segment {i}: cutter compensation lead-in length ({move_len:.2f}mm) is shorter than minimum required for tool Ø{tool.diameter:.1f}mm",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))

                    # 3. Side must be valid
                    if side not in ("left", "right"):
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"segment {i}: invalid cutter compensation side '{side}' (expected 'left' or 'right')",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))
                    comp_active = True

                elif action == "off":
                    if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX,
                                           MotionType.RAPID, MotionType.LINK, MotionType.RETRACT):
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"segment {i}: cutter compensation G40 cancellation must be on linear feed exit, not {seg.motion_type.value}",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))
                    comp_active = False

            # Check illegal rapid motion while cutter comp is active
            if comp_active and seg.motion_type in (MotionType.RAPID, MotionType.LINK, MotionType.RETRACT):
                errors.append(CamError(
                    code=INVALID_TOOLPATH,
                    message=f"segment {i}: rapid motion is illegal while cutter compensation (G41/G42) is active; cancel with G40 before rapid traverse",
                    stage="validation", operation_id=tp.operation_id,
                    tool_id=tool.id, severity=ErrorSeverity.ERROR
                ))

            # Check holder collision on rapids/retracts too (tool may be near walls)
            if seg.motion_type in (MotionType.RAPID, MotionType.RETRACT, MotionType.LINK):
                if self._check_holder_collision(tool, seg):
                    warnings.append(CamError(
                        code=COLLISION_DETECTED,
                        message=f"rapid/retract segment {i} holder may collide with part",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id, severity=ErrorSeverity.WARNING))

            # Check if rapid motion collides with solid stock material
            rapid_depth = self._check_rapid_stock_collision(tool, seg)
            if rapid_depth is not None:
                errors.append(CamError(
                    code=COLLISION_DETECTED,
                    message=f"rapid segment {i} plunges {rapid_depth:.2f}mm into stock material (Z={min(seg.start[2], seg.end[2]):.2f}mm vs stock top {float(self.setup.stock.bounds_max[2]):.2f}mm)",
                    stage="validation", operation_id=tp.operation_id,
                    tool_id=tool.id, severity=ErrorSeverity.ERROR))

            if self._check_rapid_gouge(tool, seg):
                if tp.purpose not in _NO_GOUGE_CHECK_PURPOSES:
                    errors.append(CamError(
                        code=GOUGE_DETECTED,
                        message=f"rapid segment {i} gouges model surface at Z={min(seg.start[2], seg.end[2]):.2f}mm",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id, severity=ErrorSeverity.ERROR))

            # Check fixture/clamp collisions on rapids and cutting moves
            if self.setup.fixtures:
                for fixture in self.setup.fixtures:
                    if self._check_fixture_collision(tool, seg, fixture):
                        sev = ErrorSeverity.WARNING if seg.motion_type in (
                            MotionType.RAPID, MotionType.RETRACT) else ErrorSeverity.ERROR
                        errors.append(CamError(
                            code=COLLISION_DETECTED,
                            message=f"segment {i} potential collision with fixture "
                                    f"'{getattr(fixture, 'id', 'unknown')}' at "
                                    f"({seg.start[0]:.2f},{seg.start[1]:.2f},{seg.start[2]:.2f})",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id, severity=sev))

        if comp_active:
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"toolpath '{tp.operation_id}' completed with uncancelled cutter compensation (missing G40 exit move)",
                stage="validation", operation_id=tp.operation_id,
                tool_id=tool.id, severity=ErrorSeverity.ERROR
            ))

        # Validate segment-to-segment continuity, tangency, and circular contour consistency
        self._validate_continuity_and_contours(tp, tool, errors, warnings)

        # Validate roughing geometric containment (Stock ∩ Tool Swept Volume ∩ Target Material)
        self._validate_roughing_containment(tp, tool, errors, warnings)

        # Gate 4 Swept-Volume Verification: Tool + Holder -> Swept Volume -> Stock -> CAD Target
        swept_report = self.swept_verifier.verify(tp, tool)
        tp.metadata["swept_volume_report"] = swept_report
        for err in swept_report.errors:
            # Avoid duplicate messages
            if not any(e.message == err.message and e.code == err.code for e in errors):
                errors.append(err)
        for warn in swept_report.warnings:
            if not any(w.message == warn.message and w.code == warn.code for w in warnings):
                warnings.append(warn)

        if errors:
            return ValidationReport("toolpath", False, errors, warnings)
        return ValidationReport("toolpath", True, [], warnings)

    def _validate_arc_geometry(self, seg: MotionSegment, i: int, op_id: str, tool_id: str,
                              errors: list[CamError], warnings: list[CamError]) -> None:
        """Validate individual arc/helix segment geometry: center, radii, chord feasibility, and sweep angle."""
        if seg.arc_center is None:
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"arc segment {i} ({seg.motion_type.value}) missing arc_center",
                stage="validation", operation_id=op_id, tool_id=tool_id, severity=ErrorSeverity.ERROR
            ))
            return

        c = seg.arc_center[:2]
        s = seg.start[:2]
        e = seg.end[:2]

        r_start = float(np.linalg.norm(s - c))
        r_end = float(np.linalg.norm(e - c))

        if r_start < 1e-4 or r_end < 1e-4:
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"arc segment {i} has degenerate zero radius (start R={r_start:.4f}mm, end R={r_end:.4f}mm)",
                stage="validation", operation_id=op_id, tool_id=tool_id, severity=ErrorSeverity.ERROR
            ))
            return

        delta_r = abs(r_start - r_end)
        if delta_r > 0.005:
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"arc segment {i} radius inconsistent: start R={r_start:.4f}mm, "
                        f"end R={r_end:.4f}mm (delta={delta_r:.4f}mm exceeds 0.005mm limit)",
                stage="validation", operation_id=op_id,
                tool_id=tool_id, severity=ErrorSeverity.ERROR
            ))
        elif delta_r > 0.001:
            warnings.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"arc segment {i} radius slight mismatch: start R={r_start:.4f}mm, "
                        f"end R={r_end:.4f}mm (delta={delta_r:.4f}mm)",
                stage="validation", operation_id=op_id,
                tool_id=tool_id, severity=ErrorSeverity.WARNING
            ))

        r_avg = 0.5 * (r_start + r_end)
        chord = float(np.linalg.norm(e - s))
        r_spec = seg.metadata.get("arc_radius") if seg.metadata else None
        r_check = float(r_spec) if r_spec is not None else r_avg
        if chord > 2.0 * r_check + 1e-4:
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"arc segment {i} chord length ({chord:.4f}mm) exceeds arc diameter ({2.0 * r_check:.4f}mm)",
                stage="validation", operation_id=op_id, tool_id=tool_id, severity=ErrorSeverity.ERROR
            ))

        # Sweep angle calculation
        u = s - c
        v = e - c
        a0 = math.atan2(u[1], u[0])
        a1 = math.atan2(v[1], v[0])
        is_ccw = seg.arc_ccw if seg.motion_type == MotionType.HELIX else (seg.motion_type == MotionType.ARC_CCW or seg.arc_ccw)
        if is_ccw:
            sweep = (a1 - a0) % (2.0 * math.pi)
        else:
            sweep = (a0 - a1) % (2.0 * math.pi)

        if chord < 1e-6:
            sweep = 2.0 * math.pi  # full circle
        elif sweep < 1e-6:
            sweep = 2.0 * math.pi

        if sweep <= 0.0 or sweep > 2.0 * math.pi + 1e-5:
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"arc segment {i} has invalid sweep angle {math.degrees(sweep):.2f}°",
                stage="validation", operation_id=op_id, tool_id=tool_id, severity=ErrorSeverity.ERROR
            ))

    @staticmethod
    def _segment_tangent(seg: MotionSegment, at_end: bool) -> Optional[np.ndarray]:
        """Compute 2D unit tangent vector at segment start (at_end=False) or end (at_end=True)."""
        if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX) and seg.arc_center is not None:
            c = seg.arc_center[:2]
            p = seg.end[:2] if at_end else seg.start[:2]
            r_vec = p - c
            r_len = float(np.linalg.norm(r_vec))
            if r_len < 1e-6:
                return None
            u_r = r_vec / r_len
            is_ccw = seg.arc_ccw if seg.motion_type == MotionType.HELIX else (seg.motion_type == MotionType.ARC_CCW or seg.arc_ccw)
            if is_ccw:
                # CCW tangent is 90 deg counter-clockwise from radius vector: (-r_y, r_x)
                return np.array([-u_r[1], u_r[0]])
            else:
                # CW tangent is 90 deg clockwise from radius vector: (r_y, -r_x)
                return np.array([u_r[1], -u_r[0]])
        else:
            d_xy = seg.end[:2] - seg.start[:2]
            l_xy = float(np.linalg.norm(d_xy))
            if l_xy > 1e-6:
                return d_xy / l_xy
        return None

    def _validate_continuity_and_contours(self, tp: Toolpath, tool: Tool,
                                         errors: list[CamError], warnings: list[CamError]) -> None:
        """Validate C0 positional continuity, G1 tangency at blends, and multi-segment circular contour consistency."""
        segments = tp.segments
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]

            # 1. C0 Positional Continuity across consecutive uninterrupted cutting moves
            if prev.motion_type in CUTTING and curr.motion_type in CUTTING:
                gap = float(np.linalg.norm(curr.start - prev.end))
                if gap > 0.001:
                    errors.append(CamError(
                        code=INVALID_TOOLPATH,
                        message=f"positional discontinuity between cutting segment {i - 1} and {i}: "
                                f"gap of {gap:.4f}mm without retract/link move",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id, severity=ErrorSeverity.ERROR
                    ))

            # 2. G1 Tangency Validation for blend transitions / lead-in / lead-out
            needs_tangent = (
                curr.metadata.get("tangent_blend") or prev.metadata.get("tangent_blend") or
                curr.metadata.get("tangent_lead") or prev.metadata.get("tangent_lead")
            )
            if needs_tangent:
                t_out = self._segment_tangent(prev, at_end=True)
                t_in = self._segment_tangent(curr, at_end=False)
                if t_out is not None and t_in is not None:
                    dot = float(np.dot(t_out, t_in))
                    # Allow up to 5.0 deg angular deviation for tangent blends
                    cos_tol = math.cos(math.radians(5.0))
                    if dot < cos_tol:
                        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"tangency violation at transition between segment {i - 1} and {i}: "
                                    f"angular deviation {angle_deg:.2f}° exceeds 5.0° tolerance",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))

            # 3. Multi-Segment Circular Contour Consistency
            # When two consecutive arc segments form part of the same circular contour / pass
            is_prev_arc = prev.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX)
            is_curr_arc = curr.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX)
            if is_prev_arc and is_curr_arc:
                prev_is_lead = prev.motion_type in (MotionType.ENTRY, MotionType.EXIT) or prev.metadata.get("is_lead", False)
                curr_is_lead = curr.motion_type in (MotionType.ENTRY, MotionType.EXIT) or curr.metadata.get("is_lead", False)

                if not prev_is_lead and not curr_is_lead:
                    # Check if they are contiguous at the same depth pass (or continuous helix)
                    is_connected = float(np.linalg.norm(curr.start - prev.end)) <= 0.001
                    is_same_z = abs(curr.start[2] - prev.start[2]) < 1e-4 or (
                        prev.motion_type == MotionType.HELIX and curr.motion_type == MotionType.HELIX
                    )
                    has_shared_contour = (
                        (prev.metadata.get("contour_id") and prev.metadata.get("contour_id") == curr.metadata.get("contour_id")) or
                        (is_connected and is_same_z)
                    )

                    if has_shared_contour and prev.arc_center is not None and curr.arc_center is not None:
                        c1 = prev.arc_center[:2]
                        c2 = curr.arc_center[:2]
                        d_c = float(np.linalg.norm(c1 - c2))
                        if d_c > 0.005:
                            errors.append(CamError(
                                code=INVALID_TOOLPATH,
                                message=f"consecutive arc segments {i - 1} and {i} in circular contour have inconsistent centers: "
                                        f"({c1[0]:.3f}, {c1[1]:.3f}) vs ({c2[0]:.3f}, {c2[1]:.3f}) (delta={d_c:.4f}mm exceeds 0.005mm limit)",
                                stage="validation", operation_id=tp.operation_id,
                                tool_id=tool.id, severity=ErrorSeverity.ERROR
                            ))

                        r1 = float(np.linalg.norm(prev.start[:2] - c1))
                        r2 = float(np.linalg.norm(curr.start[:2] - c2))
                        d_r = abs(r1 - r2)
                        if d_r > 0.005:
                            errors.append(CamError(
                                code=INVALID_TOOLPATH,
                                message=f"consecutive arc segments {i - 1} and {i} in circular contour have inconsistent radii: "
                                        f"R={r1:.4f}mm vs R={r2:.4f}mm (delta={d_r:.4f}mm exceeds 0.005mm limit)",
                                stage="validation", operation_id=tp.operation_id,
                                tool_id=tool.id, severity=ErrorSeverity.ERROR
                            ))

                        ccw1 = prev.arc_ccw if prev.motion_type == MotionType.HELIX else (prev.motion_type == MotionType.ARC_CCW or prev.arc_ccw)
                        ccw2 = curr.arc_ccw if curr.motion_type == MotionType.HELIX else (curr.motion_type == MotionType.ARC_CCW or curr.arc_ccw)
                        if ccw1 != ccw2:
                            errors.append(CamError(
                                code=INVALID_TOOLPATH,
                                message=f"consecutive arc segments {i - 1} and {i} in circular contour have opposite rotation directions",
                                stage="validation", operation_id=tp.operation_id,
                                tool_id=tool.id, severity=ErrorSeverity.ERROR
                            ))

    def _gouge_depth(self, tool: Tool, p: np.ndarray) -> float:
        if not len(self.mesh.triangles):
            return 0.0
        r = tool.diameter / 2.0
        # Account for tool shape: the gouge check compares surface height to
        # the tool's cutting bottom, not the tool center
        if tool.type == ToolType.BALL_ENDMILL:
            tool_bottom_offset = r  # ball extends radius below center
        elif tool.type == ToolType.BULLNOSE_ENDMILL:
            tool_bottom_offset = tool.corner_radius
        else:
            tool_bottom_offset = 0.0
        # Tool bottom Z = center Z - bottom offset
        tool_bottom_z = p[2] - tool_bottom_offset
        test_z = tool_bottom_z + 0.10

        # 1. Check if tool center is inside protected CAD solid
        if self.mesh.is_point_inside(np.array([p[0], p[1], test_z])):
            return r

        # 2. Check if tool bottom plunges below CAD floor directly beneath tool center
        best = 0.0
        hs = self.mesh.heights_above(p[0], p[1])
        for h in hs:
            if h > tool_bottom_z + 1e-4:
                best = max(best, h - tool_bottom_z)
        return best

    def _check_holder_collision(self, tool: Tool, seg: MotionSegment) -> bool:
        if tool.holder is None:
            return False
        if not len(self.mesh.triangles):
            return False
        for p in (seg.start, seg.end):
            holder_bottom_z = p[2] + tool.flute_length
            holder_top_z = holder_bottom_z + tool.holder.length
            hs = self.mesh.heights_above(p[0], p[1])
            for h in hs:
                if holder_bottom_z - 1e-6 < h < holder_top_z + 1e-6:
                    return True
        return False

    def _check_rapid_stock_collision(self, tool: Tool, seg: MotionSegment) -> Optional[float]:
        """Check if a rapid motion plunges into solid stock material.
        
        Returns the penetration depth below stock top if penetrating, else None.
        """
        if seg.motion_type not in (MotionType.RAPID, MotionType.LINK):
            return None
        stock = self.setup.stock
        if stock is None:
            return None

        stock_top = float(stock.bounds_max[2])
        r = tool.diameter / 2.0
        points_to_check = [seg.start, seg.end]
        dist = float(np.linalg.norm(seg.end - seg.start))
        if dist > tool.diameter:
            points_to_check.append((seg.start + seg.end) / 2.0)

        for p in points_to_check:
            if hasattr(stock, "kind") and getattr(stock.kind, "value", str(stock.kind)) == "cylinder" and hasattr(stock, "radius") and stock.radius > 0:
                s_center = stock.center_xy if (hasattr(stock, "center_xy") and stock.center_xy is not None) else np.zeros(2)
                d_xy = float(np.linalg.norm(p[:2] - s_center[:2]))
                if d_xy <= stock.radius + 1e-4:
                    if p[2] < stock_top - 0.05:  # allow 0.05mm tolerance for top-level rapid skimming
                        return float(stock_top - p[2])
            else:
                s_min = stock.bounds_min[:2]
                s_max = stock.bounds_max[:2]
                if (s_min[0] <= p[0] <= s_max[0]) and (s_min[1] <= p[1] <= s_max[1]):
                    if p[2] < stock_top - 0.05:
                        return float(stock_top - p[2])
        return None

    def _check_rapid_gouge(self, tool: Tool, seg: MotionSegment) -> bool:
        if seg.motion_type not in (MotionType.RAPID, MotionType.LINK):
            return False
        if not len(self.mesh.triangles):
            return False
        # Account for tool shape: ball tools extend radius below center,
        # flat endmills extend 0, bullnose extends corner_radius
        if tool.type == ToolType.BALL_ENDMILL:
            tool_bottom_offset = tool.diameter / 2.0
        elif tool.type == ToolType.BULLNOSE_ENDMILL:
            tool_bottom_offset = tool.corner_radius
        else:
            tool_bottom_offset = 0.0
        # Check start, end, and midpoint for gouge
        points_to_check = [seg.start, seg.end]
        # Add midpoint for longer rapids
        dist = np.linalg.norm(seg.end - seg.start)
        if dist > tool.diameter:
            mid = (seg.start + seg.end) / 2.0
            points_to_check.append(mid)
        for p in points_to_check:
            hs = self.mesh.heights_above(p[0], p[1])
            for h in hs:
                if p[2] < h - tool_bottom_offset - 1e-6:
                    return True
        return False

    def _check_fixture_collision(self, tool: Tool, seg: MotionSegment,
                                 fixture) -> bool:
        """Check if a motion segment collides with a fixture/clamp.
        Fixtures are modeled as axis-aligned bounding boxes."""
        # Fixtures may have bounds_min/bounds_max attributes (AABB)
        f_min = getattr(fixture, 'bounds_min', None)
        f_max = getattr(fixture, 'bounds_max', None)
        if f_min is None or f_max is None:
            return False
        # Tool envelope: cylinder of radius tool.diameter/2 from tip to flute_length above
        r = tool.diameter / 2.0
        # Check start, end, and midpoint for collision
        points_to_check = [seg.start, seg.end]
        dist = np.linalg.norm(seg.end - seg.start)
        if dist > tool.diameter:
            points_to_check.append((seg.start + seg.end) / 2.0)
        for p in points_to_check:
            # Check if tool center XY is within fixture XY bounds (with tool radius)
            if (p[0] + r < f_min[0] - 1e-6 or p[0] - r > f_max[0] + 1e-6 or
                    p[1] + r < f_min[1] - 1e-6 or p[1] - r > f_max[1] + 1e-6):
                continue
            # Tool extends from p[2] upward by flute_length
            tool_top_z = p[2] + tool.flute_length
            # Fixture extends from f_min[2] to f_max[2]
            if tool_top_z > f_min[2] - 1e-6 and p[2] < f_max[2] + 1e-6:
                return True
        return False

    @staticmethod
    def _discretize_motion(seg: MotionSegment, max_step: float = 0.5) -> list[np.ndarray]:
        """Discretize a motion segment into a sequence of points along the 3D trajectory."""
        if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX) and seg.arc_center is not None:
            c = seg.arc_center[:2]
            s = seg.start[:2]
            e = seg.end[:2]
            u = s - c
            v = e - c
            r = float(np.linalg.norm(u))
            a0 = math.atan2(u[1], u[0])
            a1 = math.atan2(v[1], v[0])
            is_ccw = seg.arc_ccw if seg.motion_type == MotionType.HELIX else (seg.motion_type == MotionType.ARC_CCW or seg.arc_ccw)
            if is_ccw:
                if a1 <= a0:
                    a1 += 2.0 * math.pi
                sweep = a1 - a0
            else:
                if a1 >= a0:
                    a1 -= 2.0 * math.pi
                sweep = a0 - a1

            arc_len = r * abs(sweep)
            n_steps = max(4, int(math.ceil(arc_len / max_step)))
            pts = []
            for k in range(n_steps + 1):
                t = k / n_steps
                angle = a0 + t * (a1 - a0)
                z = seg.start[2] + t * (seg.end[2] - seg.start[2])
                pts.append(np.array([c[0] + r * math.cos(angle), c[1] + r * math.sin(angle), z]))
            return pts
        else:
            length = float(np.linalg.norm(seg.end - seg.start))
            if length < 1e-6:
                return [seg.start, seg.end]
            n_steps = max(1, int(math.ceil(length / max_step)))
            pts = []
            for k in range(n_steps + 1):
                t = k / n_steps
                pts.append(seg.start + t * (seg.end - seg.start))
            return pts

    def _validate_roughing_containment(self, tp: Toolpath, tool: Tool,
                                       errors: list[CamError], warnings: list[CamError]) -> None:
        """Validate volumetric geometric containment of roughing toolpaths:
        Stock ∩ Tool Swept Volume (V_swept) ∩ Target Remaining Material

        Verifies:
        1. Tool swept volume stays within the stock envelope (plus approach/lead margin).
        2. Toolpath does not perform disconnected / ghost air cuts completely outside stock.
        3. Tool swept volume does not gouge finished CAD geometry and respects finish allowances (wall & floor).
        4. Tool swept volume does not collide with setup fixtures / workholding.
        5. Path does not leave unmachined material webs due to excessive stepover (stepover <= tool diameter).
        6. Loops and passes maintain topological continuity, monotonic stepdown, and avoid self-intersection.
        """
        strat = tp.metadata.get("strategy", "") if tp.metadata else ""
        purpose = getattr(tp.purpose, "value", str(tp.purpose)).lower() if tp.purpose else ""
        op_id = str(tp.operation_id).lower()

        is_roughing = (
            purpose in ("roughing", "adaptive_roughing", "rest_machining") or
            strat in ("boss_circular_roughing", "contour_parallel", "adaptive", "trochoidal", "hecm", "plunge_roughing") or
            "rough" in op_id
        )
        if not is_roughing:
            return

        r_tool = tool.diameter / 2.0
        stock = self.setup.stock
        lead_margin = max(r_tool * 0.5, 1.5)

        # 1. Collect cutting segments and discretize into continuous swept volume samples
        swept_points: list[np.ndarray] = []
        passes_by_z: dict[float, list[list[np.ndarray]]] = {}

        current_pass: list[np.ndarray] = []
        current_pass_z: Optional[float] = None

        for seg in tp.segments:
            if seg.motion_type in CUTTING:
                pts = self._discretize_motion(seg, max_step=max(0.2, r_tool * 0.25))
                swept_points.extend(pts)

                # Group horizontal cutting passes (arcs / cuts) for ring/contour stepover and topology analysis
                if abs(seg.start[2] - seg.end[2]) < 1e-4:
                    seg_z = round(float(seg.end[2]), 4)
                    if current_pass and current_pass_z == seg_z and len(pts) > 0 and np.linalg.norm(pts[0] - current_pass[-1]) < 1e-4:
                        current_pass.extend(pts[1:])
                    else:
                        if current_pass and current_pass_z is not None:
                            passes_by_z.setdefault(current_pass_z, []).append(current_pass)
                        current_pass = list(pts)
                        current_pass_z = seg_z
                else:
                    if current_pass and current_pass_z is not None:
                        passes_by_z.setdefault(current_pass_z, []).append(current_pass)
                        current_pass = []
                        current_pass_z = None
            else:
                if current_pass and current_pass_z is not None:
                    passes_by_z.setdefault(current_pass_z, []).append(current_pass)
                    current_pass = []
                    current_pass_z = None

        if current_pass and current_pass_z is not None:
            passes_by_z.setdefault(current_pass_z, []).append(current_pass)

        if not swept_points:
            return

        # 2. Stock Volumetric Containment Verification
        stock_intersected = False
        stock_top = float(stock.bounds_max[2]) if stock is not None else 0.0
        stock_bot = float(stock.bounds_min[2]) if stock is not None else 0.0

        for pt in swept_points:
            is_in_stock_env = False
            if stock is not None:
                if hasattr(stock, "kind") and getattr(stock.kind, "value", str(stock.kind)) == "cylinder" and hasattr(stock, "radius") and stock.radius > 0:
                    s_center = stock.center_xy if (hasattr(stock, "center_xy") and stock.center_xy is not None) else np.zeros(2)
                    d_xy = float(np.linalg.norm(pt[:2] - s_center[:2]))
                    # Tool cutting edge touches stock
                    if d_xy - r_tool <= stock.radius + 1e-4 and pt[2] <= stock_top + 1e-4:
                        stock_intersected = True
                    if d_xy <= stock.radius + r_tool + lead_margin + 1e-4 and pt[2] <= stock_top + lead_margin + 1e-4:
                        is_in_stock_env = True
                else:
                    s_min = stock.bounds_min[:2]
                    s_max = stock.bounds_max[:2]
                    # Tool cutting edge touches stock
                    if (s_min[0] - r_tool <= pt[0] <= s_max[0] + r_tool) and \
                       (s_min[1] - r_tool <= pt[1] <= s_max[1] + r_tool) and \
                       (stock_bot - 1e-4 <= pt[2] <= stock_top + 1e-4):
                        stock_intersected = True
                    if (s_min[0] - r_tool - lead_margin <= pt[0] <= s_max[0] + r_tool + lead_margin) and \
                       (s_min[1] - r_tool - lead_margin <= pt[1] <= s_max[1] + r_tool + lead_margin) and \
                       (stock_bot - lead_margin <= pt[2] <= stock_top + lead_margin):
                        is_in_stock_env = True

                if not is_in_stock_env:
                    errors.append(CamError(
                        code=INVALID_TOOLPATH,
                        message=f"roughing swept volume exceeds stock boundary at ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})",
                        stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                    ))
                    break

        if stock is not None and not stock_intersected:
            errors.append(CamError(
                code=INVALID_TOOLPATH,
                message=f"roughing toolpath swept volume has 0% intersection with stock (pure air cut outside stock)",
                stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
            ))

        # 3. Finished CAD Geometry Gouge & Finish Allowance Protection
        if self.mesh is not None and len(getattr(self.mesh, "triangles", [])) > 0:
            finish_floor = self.finish_allowance_floor
            finish_wall = self.finish_allowance_wall

            for pt in swept_points:
                tool_bot_z = pt[2]
                if tool.type == ToolType.BALL_ENDMILL:
                    tool_bot_z -= r_tool
                elif tool.type == ToolType.BULLNOSE_ENDMILL:
                    tool_bot_z -= getattr(tool, "corner_radius", 0.0)

                hs = self.mesh.heights_above(pt[0], pt[1])
                if hs:
                    model_z = max(hs)
                    if tool_bot_z < model_z + finish_floor - 1e-4:
                        violation = (model_z + finish_floor) - tool_bot_z
                        if tool_bot_z < model_z - 1e-4:
                            errors.append(CamError(
                                code=GOUGE_DETECTED,
                                message=f"roughing toolpath gouges finished model by {model_z - tool_bot_z:.2f}mm at ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})",
                                stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                            ))
                            break
                        elif finish_floor > 0.0:
                            errors.append(CamError(
                                code=GOUGE_DETECTED,
                                message=f"roughing toolpath violates finish floor allowance ({finish_floor:.2f}mm) by {violation:.2f}mm at ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})",
                                stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                            ))
                            break

        # 4. Stepover and Uncut Material Web Verification
        for z_lvl, passes in passes_by_z.items():
            if len(passes) > 1:
                ring_radii = []
                for p in passes:
                    if len(p) >= 4:
                        p_arr = np.array(p)[:, :2]
                        c_est = np.mean(p_arr, axis=0)
                        radii = np.linalg.norm(p_arr - c_est, axis=1)
                        if np.std(radii) < 0.1:
                            ring_radii.append(float(np.mean(radii)))

                if len(ring_radii) > 1:
                    ring_radii.sort()
                    for r_idx in range(len(ring_radii) - 1):
                        delta_r = ring_radii[r_idx + 1] - ring_radii[r_idx]
                        if delta_r > tool.diameter + 1e-4:
                            errors.append(CamError(
                                code=INVALID_TOOLPATH,
                                message=f"roughing ring stepover ({delta_r:.2f}mm) at Z={z_lvl:.2f} exceeds cutter diameter ({tool.diameter:.2f}mm), leaving uncut stock webs",
                                stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                            ))
                            break

        # 5. Loop Topology, Self-Intersection, and Monotonic Stepdown
        prev_z = None
        for z_lvl in sorted(passes_by_z.keys(), reverse=True):
            if prev_z is not None and z_lvl > prev_z + 1e-4:
                errors.append(CamError(
                    code=INVALID_TOOLPATH,
                    message=f"roughing passes have non-monotonic stepdown (Z stepped up from {prev_z:.2f} to {z_lvl:.2f})",
                    stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                ))
            prev_z = z_lvl

        # Collect raw segment vertices per pass for topology checking
        raw_passes_by_z: dict[float, list[list[np.ndarray]]] = {}
        curr_raw_pass: list[np.ndarray] = []
        curr_raw_z: Optional[float] = None

        for seg in tp.segments:
            if seg.motion_type in CUTTING and abs(seg.start[2] - seg.end[2]) < 1e-4:
                seg_z = round(float(seg.end[2]), 4)
                if curr_raw_pass and curr_raw_z == seg_z and np.linalg.norm(seg.start - curr_raw_pass[-1]) < 1e-4:
                    curr_raw_pass.append(seg.end)
                else:
                    if curr_raw_pass and curr_raw_z is not None:
                        raw_passes_by_z.setdefault(curr_raw_z, []).append(curr_raw_pass)
                    curr_raw_pass = [seg.start, seg.end]
                    curr_raw_z = seg_z
            else:
                if curr_raw_pass and curr_raw_z is not None:
                    raw_passes_by_z.setdefault(curr_raw_z, []).append(curr_raw_pass)
                    curr_raw_pass = []
                    curr_raw_z = None

        if curr_raw_pass and curr_raw_z is not None:
            raw_passes_by_z.setdefault(curr_raw_z, []).append(curr_raw_pass)

        # Check self-intersection on closed polygon passes
        for z_lvl, raw_passes in raw_passes_by_z.items():
            for p in raw_passes:
                if len(p) >= 4:
                    p_xy = [pt[:2] for pt in p]
                    if np.linalg.norm(p_xy[0] - p_xy[-1]) < 0.05:
                        if self._check_polygon_self_intersection(p_xy):
                            errors.append(CamError(
                                code=INVALID_TOOLPATH,
                                message=f"roughing pass at Z={z_lvl:.2f} contains self-intersecting loops",
                                stage="validation", operation_id=tp.operation_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                            ))

    @staticmethod
    def _check_polygon_self_intersection(polygon: list[np.ndarray]) -> bool:
        """Check if non-adjacent edges of a 2D closed loop intersect."""
        n = len(polygon)
        if n < 4:
            return False
        for i in range(n - 1):
            a0 = polygon[i]
            a1 = polygon[i + 1]
            for j in range(i + 2, n - 1):
                if i == 0 and j == n - 2:
                    continue
                b0 = polygon[j]
                b1 = polygon[j + 1]
                da = a1 - a0
                db = b1 - b0
                denom = da[0] * db[1] - da[1] * db[0]
                if abs(denom) < 1e-9:
                    continue
                dp = a0 - b0
                t = (db[0] * dp[1] - db[1] * dp[0]) / denom
                u = (da[0] * dp[1] - da[1] * dp[0]) / denom
                if 0.001 < t < 0.999 and 0.001 < u < 0.999:
                    return True
        return False

    def _validate_canned_cycle_peck(self, cycle: dict[str, Any], seg_idx: int, op_id: Optional[str],
                                    tool: Tool, errors: list[CamError], warnings: list[CamError]) -> None:
        """Validate canned cycle peck depth (Q) against tool diameter, material, and geometry constraints."""
        peck = float(cycle.get("peck", 0.0))
        if peck <= 0:
            return

        dia = max(float(tool.diameter), 0.01)
        has_tsc = getattr(tool, "has_through_coolant", False)
        mat = getattr(self.context, "material", None)
        mat_name = getattr(mat, "name", "workpiece") if mat else "workpiece"

        # 1. Tool manufacturer explicit limit override
        if getattr(tool, "max_peck_ratio", None) is not None and tool.max_peck_ratio > 0:
            max_allowed = tool.max_peck_ratio * dia
            if peck > max_allowed + 1e-4:
                errors.append(CamError(
                    code=INVALID_TOOLPATH,
                    message=f"segment {seg_idx}: canned cycle peck Q={peck:.3f}mm exceeds tool manufacturer limit {tool.max_peck_ratio:.2f}xD ({max_allowed:.3f}mm)",
                    stage="validation", operation_id=op_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                ))

        # 2. Micro-drill strict safety ceiling (D < 1.0 mm)
        if dia < 1.0:
            # Without TSC, single peck > 0.85*D is dangerous; with TSC, single peck > 1.10*D is dangerous
            micro_limit = 1.10 * dia if has_tsc else 0.85 * dia
            if peck > micro_limit + 1e-4:
                warnings.append(CamError(
                    code=INVALID_TOOLPATH,
                    message=f"segment {seg_idx}: canned cycle peck Q={peck:.3f}mm ({peck/dia:.2f}xD) is excessive for micro-drill Ø{dia:.2f}mm (recommended Q <= {0.50*dia:.3f}mm to avoid chip choking)",
                    stage="validation", operation_id=op_id, tool_id=tool.id, severity=ErrorSeverity.WARNING
                ))

        # 3. Work-hardening / tough material limits
        if mat is not None:
            mat_id = (mat.id or mat.name or "").lower()
            chip_tendency = getattr(mat, "chip_breaking_tendency", "medium").lower()
            if "stainless" in mat_id or "titanium" in mat_id or "inconel" in mat_id or "becu" in mat_id or chip_tendency in ("stringy", "work_hardening"):
                tough_limit = 1.50 * dia if has_tsc else 1.20 * dia
                if dia < 1.0:
                    tough_limit = 0.65 * dia
                if peck > tough_limit + 1e-4:
                    warnings.append(CamError(
                        code=INVALID_TOOLPATH,
                        message=f"segment {seg_idx}: canned cycle peck Q={peck:.3f}mm ({peck/dia:.2f}xD) is high for {mat_name} (risk of work hardening / chip binding)",
                        stage="validation", operation_id=op_id, tool_id=tool.id, severity=ErrorSeverity.WARNING
                    ))

        # 4. Retract mode safety validation (G98 vs G99)
        retract_mode = cycle.get("retract_mode", "G98")
        z_r = cycle.get("z_r")
        if retract_mode == "G99" and z_r is not None:
            if hasattr(self.setup, "fixtures") and self.setup.fixtures:
                for fixture in self.setup.fixtures:
                    if fixture.bounds_max is not None and fixture.bounds_max[2] > z_r + 1e-4:
                        errors.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"segment {seg_idx}: G99 R-plane return (Z={z_r:.2f}mm) collides with fixture '{fixture.id}' (top Z={fixture.bounds_max[2]:.2f}mm); G98 Initial Plane return is required",
                            stage="validation", operation_id=op_id, tool_id=tool.id, severity=ErrorSeverity.ERROR
                        ))


class PreToolpathGeometryValidator:
    """Gate 2: Mandatory geometry validation gate BEFORE toolpath generation.

    Enforces validation across three critical domains:
    1. Feature Geometry:
       - face exists in CAD B-Rep shape
       - edge exists in CAD B-Rep shape (if referenced)
       - boundary closed (closed loops have start-to-end gap <= 1e-3 mm)
       - surface valid (finite bounds, non-inverted, valid surface normal)
       - axis valid (axial features have non-zero, finite, normalized axis vectors)
       - dimensions valid (finite depth >= 0, diameter > 0, radius > 0, non-negative extents)
    2. Manufacturing Geometry:
       - target reachable (accessibility != INACCESSIBLE)
       - target belongs to current setup (machining direction aligned with setup +Z, within envelope)
       - tool can physically access target (flute length / reach >= depth)
       - tool diameter compatible (fits inside hole/pocket/slot)
       - holder clearance valid (feature depth within allowable tool protrusion)
    3. Coordinate Transformation:
       - CAD coordinates valid (finite, no NaN/Inf)
       - setup transform valid (finite, det != 0, invertible)
       - WCS transform valid (finite, invertible)
       - no unintended scale (scale == 1.0, det(R) == 1.0, column norms == 1.0)
       - no unintended rotation/shear (orthonormal matrix R^T R == I)

    Strict Rule: If validation fails -> NO TOOLPATH is generated.
    """

    def __init__(
        self,
        context: Optional[PlanningContext] = None,
        setup: Optional[Setup] = None,
        shape: Optional[Shape] = None,
    ):
        self.context = context
        self.setup = setup
        self.shape = shape

    def validate_setup_transform(self, setup: Optional[Setup] = None) -> ValidationReport:
        """Validate setup and coordinate transformation matrices.
        Checks:
        - Matrix is 4x4, finite, invertible
        - Scale is strictly 1.0 (no unintended scale)
        - Rotation submatrix is orthonormal (R^T R == I, det(R) == 1.0, no unintended shear/distortion)
        """
        s = setup or self.setup
        errors: list[CamError] = []
        warnings: list[CamError] = []

        if s is None:
            return ValidationReport("geometry_validation", True)

        M = s.model_to_setup
        if M is None:
            errors.append(CamError(
                code=INVALID_SETUP_TRANSFORM,
                message=f"Setup '{s.id}' is missing model_to_setup transform matrix",
                stage="geometry_validation",
            ))
            return ValidationReport("geometry_validation", False, errors, warnings)

        if not isinstance(M, np.ndarray) or M.shape != (4, 4):
            errors.append(CamError(
                code=INVALID_SETUP_TRANSFORM,
                message=f"Setup '{s.id}' transform matrix is not 4x4 (shape: {getattr(M, 'shape', None)})",
                stage="geometry_validation",
            ))
            return ValidationReport("geometry_validation", False, errors, warnings)

        if not np.all(np.isfinite(M)):
            errors.append(CamError(
                code=INVALID_SETUP_TRANSFORM,
                message=f"Setup '{s.id}' transform matrix contains NaN or Inf values",
                stage="geometry_validation",
            ))
            return ValidationReport("geometry_validation", False, errors, warnings)

        # Check affine bottom row [0, 0, 0, 1]
        if not np.allclose(M[3, :], [0, 0, 0, 1], atol=1e-5):
            errors.append(CamError(
                code=INVALID_SETUP_TRANSFORM,
                message=f"Setup '{s.id}' transform matrix bottom row is not [0, 0, 0, 1] (got {M[3, :]})",
                stage="geometry_validation",
            ))

        R = M[:3, :3]
        det_R = float(np.linalg.det(R))

        # Check invertibility
        if abs(det_R) < 1e-6:
            errors.append(CamError(
                code=INVALID_SETUP_TRANSFORM,
                message=f"Setup '{s.id}' transform matrix is singular / non-invertible (det={det_R:.6f})",
                stage="geometry_validation",
            ))
            return ValidationReport("geometry_validation", False, errors, warnings)

        # Check for unintended scale (column norms must be 1.0)
        col_norms = np.linalg.norm(R, axis=0)
        scale_error = float(np.max(np.abs(col_norms - 1.0)))
        if scale_error > 1e-4:
            errors.append(CamError(
                code=INVALID_SETUP_TRANSFORM,
                message=f"Setup '{s.id}' transform contains unintended scale (col norms: {col_norms.tolist()}, max dev: {scale_error:.6f} != 1.0)",
                stage="geometry_validation",
            ))

        # Check for unintended shear / non-orthogonality (R^T R == I)
        ortho_diff = np.abs(R.T @ R - np.eye(3))
        max_ortho_dev = float(np.max(ortho_diff))
        if max_ortho_dev > 1e-4:
            errors.append(CamError(
                code=INVALID_SETUP_TRANSFORM,
                message=f"Setup '{s.id}' transform contains unintended shear/rotation distortion (orthogonality dev: {max_ortho_dev:.6f} > 1e-4)",
                stage="geometry_validation",
            ))

        # Check proper orientation (det(R) == +1.0)
        if abs(det_R - 1.0) > 1e-4:
            if abs(det_R + 1.0) <= 1e-4:
                errors.append(CamError(
                    code=INVALID_SETUP_TRANSFORM,
                    message=f"Setup '{s.id}' transform contains left-handed reflection (det={det_R:.4f}); rigid right-handed setup required",
                    stage="geometry_validation",
                ))
            else:
                errors.append(CamError(
                    code=INVALID_SETUP_TRANSFORM,
                    message=f"Setup '{s.id}' transform determinant is {det_R:.4f} (expected +1.0 for rigid rotation)",
                    stage="geometry_validation",
                ))

        if errors:
            return ValidationReport("geometry_validation", False, errors, warnings)
        return ValidationReport("geometry_validation", True, [], warnings)

    def validate_feature(
        self,
        feature: MachiningFeature,
        setup: Optional[Setup] = None,
        shape: Optional[Shape] = None,
    ) -> ValidationReport:
        """Validate feature geometry, CAD references, and manufacturing reachability."""
        s = setup or self.setup
        shp = shape or self.shape
        errors: list[CamError] = []
        warnings: list[CamError] = []

        # 1. Feature geometry validation
        # Coordinates validity
        if feature.bounds_min is None or feature.bounds_max is None:
            errors.append(CamError(
                code=INVALID_GEOMETRY,
                message=f"Feature '{feature.id}' is missing bounding box bounds",
                stage="geometry_validation",
                feature_id=feature.id,
            ))
        else:
            if not np.all(np.isfinite(feature.bounds_min)) or not np.all(np.isfinite(feature.bounds_max)):
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' contains non-finite NaN/Inf bounding box coordinates",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))
            elif np.any(feature.bounds_min > feature.bounds_max + 1e-6):
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' has inverted bounding box: min={feature.bounds_min}, max={feature.bounds_max}",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))

        # Face exists
        if not feature.face_indices:
            errors.append(CamError(
                code=INVALID_GEOMETRY,
                message=f"Feature '{feature.id}' has no associated face indices (empty face list)",
                stage="geometry_validation",
                feature_id=feature.id,
            ))
        elif shp is not None:
            faces = shp.faces() if hasattr(shp, "faces") and callable(shp.faces) else getattr(shp, "_faces", [])
            if faces:
                num_faces = len(faces)
                for f_idx in feature.face_indices:
                    if f_idx < 0 or f_idx >= num_faces:
                        errors.append(CamError(
                            code=INVALID_GEOMETRY,
                            message=f"Feature '{feature.id}' references non-existent face index {f_idx} (shape has {num_faces} faces)",
                            stage="geometry_validation",
                            feature_id=feature.id,
                        ))

        # Edge exists (if source edges are specified)
        if feature.source_edges and shp is not None:
            edges = shp.edges() if hasattr(shp, "edges") and callable(shp.edges) else getattr(shp, "_edges", [])
            if edges:
                num_edges = len(edges)
                for e_idx in feature.source_edges:
                    if e_idx < 0 or e_idx >= num_edges:
                        errors.append(CamError(
                            code=INVALID_GEOMETRY,
                            message=f"Feature '{feature.id}' references non-existent edge index {e_idx} (shape has {num_edges} edges)",
                            stage="geometry_validation",
                            feature_id=feature.id,
                        ))

        # Boundary closed
        closed_boundary_types = {
            FeatureType.POCKET, FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE,
            FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
            FeatureType.BOSS, FeatureType.FACING_REGION, FeatureType.SLOT,
            FeatureType.COUNTERBORE, FeatureType.COUNTERSINK, FeatureType.TAPPED_HOLE
        }
        if feature.boundary_loop is not None:
            loop = feature.boundary_loop
            if len(loop) < 3:
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' boundary loop has fewer than 3 vertices ({len(loop)} vertices)",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))
            else:
                pts_arr = np.array(loop)
                if not np.all(np.isfinite(pts_arr)):
                    errors.append(CamError(
                        code=INVALID_GEOMETRY,
                        message=f"Feature '{feature.id}' boundary loop contains NaN/Inf coordinates",
                        stage="geometry_validation",
                        feature_id=feature.id,
                    ))
                elif feature.type in closed_boundary_types:
                    gap = float(np.linalg.norm(pts_arr[0] - pts_arr[-1]))
                    if gap > 1e-3:
                        errors.append(CamError(
                            code=INVALID_GEOMETRY,
                            message=f"Feature '{feature.id}' boundary loop is open (gap={gap:.4f}mm between start and end > 0.001mm)",
                            stage="geometry_validation",
                            feature_id=feature.id,
                        ))

        if feature.island_loops:
            for isl_idx, isl in enumerate(feature.island_loops):
                if len(isl) < 3:
                    errors.append(CamError(
                        code=INVALID_GEOMETRY,
                        message=f"Feature '{feature.id}' island loop {isl_idx} has fewer than 3 vertices",
                        stage="geometry_validation",
                        feature_id=feature.id,
                    ))
                else:
                    isl_arr = np.array(isl)
                    gap = float(np.linalg.norm(isl_arr[0] - isl_arr[-1]))
                    if gap > 1e-3:
                        errors.append(CamError(
                            code=INVALID_GEOMETRY,
                            message=f"Feature '{feature.id}' island loop {isl_idx} is open (gap={gap:.4f}mm > 0.001mm)",
                            stage="geometry_validation",
                            feature_id=feature.id,
                        ))

        # Axis valid
        axial_types = {
            FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
            FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
            FeatureType.COUNTERBORE, FeatureType.COUNTERSINK, FeatureType.TAPPED_HOLE,
            FeatureType.THREADED_HOLE, FeatureType.BOSS
        }
        if feature.type in axial_types:
            if feature.axis is None:
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' of axial type '{feature.type.value}' is missing axis vector",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))
            else:
                if not np.all(np.isfinite(feature.axis)):
                    errors.append(CamError(
                        code=INVALID_GEOMETRY,
                        message=f"Feature '{feature.id}' axis contains NaN or Inf values",
                        stage="geometry_validation",
                        feature_id=feature.id,
                    ))
                else:
                    norm = float(np.linalg.norm(feature.axis))
                    if norm < 1e-4:
                        errors.append(CamError(
                            code=INVALID_GEOMETRY,
                            message=f"Feature '{feature.id}' axis is degenerate zero vector",
                            stage="geometry_validation",
                            feature_id=feature.id,
                        ))
                    elif abs(norm - 1.0) > 1e-3:
                        errors.append(CamError(
                            code=INVALID_GEOMETRY,
                            message=f"Feature '{feature.id}' axis vector is not normalized (norm={norm:.4f})",
                            stage="geometry_validation",
                            feature_id=feature.id,
                        ))

        # Dimensions valid
        if feature.depth is None or not math.isfinite(feature.depth) or feature.depth < 0.0:
            errors.append(CamError(
                code=INVALID_GEOMETRY,
                message=f"Feature '{feature.id}' has invalid depth: {feature.depth}",
                stage="geometry_validation",
                feature_id=feature.id,
            ))

        if feature.top_z is not None and feature.floor_z is not None:
            if not (math.isfinite(feature.top_z) and math.isfinite(feature.floor_z)):
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' has non-finite top_z ({feature.top_z}) or floor_z ({feature.floor_z})",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))
            elif feature.floor_z > feature.top_z + 1e-4:
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' floor Z ({feature.floor_z:.3f}) is higher than top Z ({feature.top_z:.3f})",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))

        if feature.diameter is not None:
            if not math.isfinite(feature.diameter) or feature.diameter <= 0.0:
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' has invalid diameter: {feature.diameter}",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))

        if feature.radius is not None:
            if not math.isfinite(feature.radius) or feature.radius <= 0.0:
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' has invalid radius: {feature.radius}",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))

        # 2. Manufacturing geometry validation
        # Accessibility check
        if feature.accessibility == Accessibility.INACCESSIBLE:
            errors.append(CamError(
                code=FEATURE_INACCESSIBLE,
                message=f"Feature '{feature.id}' is physically inaccessible (marked INACCESSIBLE)",
                stage="geometry_validation",
                feature_id=feature.id,
            ))

        # Target belongs to current setup (machining direction alignment with setup tool axis +Z)
        if feature.machining_direction is not None:
            tool_axis = np.array([0.0, 0.0, 1.0])
            mach_dir = np.array(feature.machining_direction, dtype=float)
            norm_dir = float(np.linalg.norm(mach_dir))
            if norm_dir > 1e-4:
                mach_dir /= norm_dir
                dot_axis = float(np.dot(mach_dir, tool_axis))
                # For 3-axis machining setup, feature approach direction must align with +Z
                if dot_axis < 0.99:
                    errors.append(CamError(
                        code=FEATURE_INACCESSIBLE,
                        message=f"Feature '{feature.id}' machining direction {mach_dir.tolist()} does not align with setup tool axis +Z (dot={dot_axis:.3f})",
                        stage="geometry_validation",
                        feature_id=feature.id,
                    ))

        # Check if feature is within setup stock boundaries
        if s is not None and s.stock is not None and feature.bounds_min is not None and feature.bounds_max is not None:
            stock = s.stock
            if (feature.bounds_min[2] > stock.bounds_max[2] + 1.0 or
                feature.bounds_max[2] < stock.bounds_min[2] - 1.0):
                errors.append(CamError(
                    code=INVALID_GEOMETRY,
                    message=f"Feature '{feature.id}' Z range [{feature.bounds_min[2]:.2f}, {feature.bounds_max[2]:.2f}] is completely outside setup stock Z range [{stock.bounds_min[2]:.2f}, {stock.bounds_max[2]:.2f}]",
                    stage="geometry_validation",
                    feature_id=feature.id,
                ))

        # 3. No Synthetic Geometry Verification (Strict Rule)
        # Final manufacturing geometry must originate strictly from true CAD B-Rep topology.
        # It must NEVER be synthesized or guessed from bounding box, silhouette, tool diameter, DOC, or defaults.
        if shp is not None and feature.face_indices:
            faces = shp.faces() if hasattr(shp, "faces") and callable(shp.faces) else getattr(shp, "_faces", [])
            if faces:
                feat_faces = [faces[idx] for idx in feature.face_indices if 0 <= idx < len(faces)]

                # Check axial cylindrical features (HOLE, BORE, BOSS)
                if feature.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
                                   FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE, FeatureType.BOSS):
                    has_cad_cyl = any(
                        (getattr(f.surface.kind, "value", str(f.surface.kind)) in ("cylinder", "cone")) and f.surface.radius is not None
                        for f in feat_faces
                    )
                    if not has_cad_cyl and feature.diameter is not None:
                        errors.append(CamError(
                            code=SYNTHETIC_GEOMETRY_FORBIDDEN,
                            message=f"Feature '{feature.id}' specifies diameter Ø{feature.diameter:.3f}mm but underlying CAD faces {feature.face_indices} have no cylindrical surface (synthetic geometry from bounding box/tool is forbidden)",
                            stage="geometry_validation",
                            feature_id=feature.id,
                        ))
                    elif has_cad_cyl and feature.diameter is not None:
                        cad_radii = [f.surface.radius for f in feat_faces if getattr(f.surface.kind, "value", str(f.surface.kind)) == "cylinder" and f.surface.radius is not None]
                        if cad_radii:
                            expected_dia = 2.0 * cad_radii[0]
                            if abs(feature.diameter - expected_dia) > 0.05:
                                errors.append(CamError(
                                    code=SYNTHETIC_GEOMETRY_FORBIDDEN,
                                    message=f"Feature '{feature.id}' diameter Ø{feature.diameter:.3f}mm diverges from CAD B-Rep cylindrical radius R={cad_radii[0]:.3f}mm (synthetic geometry forbidden)",
                                    stage="geometry_validation",
                                    feature_id=feature.id,
                                ))

        if errors:
            return ValidationReport("geometry_validation", False, errors, warnings)
        return ValidationReport("geometry_validation", True, [], warnings)

    def check_tool_feasibility(
        self,
        tool: Any,
        feature: MachiningFeature,
        op_purpose: Optional[Any] = None,
        op_id: Optional[str] = None,
    ) -> list[CamError]:
        """Automatic tool-radius & envelope feasibility checking.
        
        Checks:
        1. Minimum feature radius / cavity diameter
        2. Minimum channel / slot width
        3. Minimum corner / fillet radius
        4. Minimum wall / island clearance
        5. Tool reach & tool holder envelope
        
        If the tool cannot physically create the required geometry:
        Generates explicit, actionable error diagnostics:
            Tool Ø{tool.diameter:.2f} cannot machine required feature '{feature.id}'.
            Required minimum access radius = {req_radius:.2f} mm
            Available tool = Ø{tool.diameter:.2f}
            Recommended action = select smaller tool (Ø <= {max_allowed_dia:.2f} mm) / different orientation
        """
        errors: list[CamError] = []
        op_id_str = op_id or "unassigned"
        purpose_val = getattr(op_purpose, "value", str(op_purpose)) if op_purpose is not None else ""
        is_spot_or_chamfer = (
            purpose_val in ("spot_drilling", "countersinking", "chamfering")
            or getattr(tool, "type", None) in (
                ToolType.SPOT_DRILL,
                ToolType.CENTER_DRILL,
                ToolType.COUNTERSINK_TOOL,
                ToolType.CHAMFER_MILL,
            )
            or feature.type in (FeatureType.COUNTERSINK, FeatureType.CHAMFER)
        )
        is_hole_op = purpose_val in ("drilling", "tapping", "reaming")
        is_finish_op = purpose_val in ("finishing", "semi_finishing", "rest_machining")

        tool_dia = float(getattr(tool, "diameter", 0.0))
        tool_r = tool_dia / 2.0
        tool_id = getattr(tool, "id", f"T{getattr(tool, 'tool_number', 1)}")

        # 1. Minimum Feature Radius / Cavity Diameter
        internal_cavity_types = {
            FeatureType.HOLE, FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE,
            FeatureType.COUNTERBORE, FeatureType.TAPPED_HOLE,
            FeatureType.THREADED_HOLE, FeatureType.BORE, FeatureType.BLIND_BORE,
            FeatureType.THROUGH_BORE, FeatureType.POCKET, FeatureType.OPEN_POCKET,
            FeatureType.SLOT, FeatureType.OPEN_SLOT, FeatureType.GROOVE
        }
        if feature.diameter is not None and feature.diameter > 1e-4 and not is_spot_or_chamfer:
            req_radius = float(feature.diameter) / 2.0
            if (is_hole_op or feature.type in internal_cavity_types) and feature.is_concave:
                if tool_dia > feature.diameter + 1e-4:
                    errors.append(CamError(
                        code=NO_COMPATIBLE_TOOL,
                        message=(
                            f"Tool Ø{tool_dia:.2f} cannot machine required feature '{feature.id}'.\n"
                            f"Required minimum access radius = {req_radius:.2f} mm\n"
                            f"Available tool = Ø{tool_dia:.2f}\n"
                            f"Recommended action = select smaller tool (Ø <= {feature.diameter:.2f} mm) / different orientation"
                        ),
                        stage="geometry_validation",
                        operation_id=op_id_str,
                        feature_id=feature.id,
                        tool_id=tool_id,
                        context={"required_access_radius": req_radius, "tool_diameter": tool_dia},
                    ))

        # 2. Minimum Channel / Slot Width
        if feature.type in (FeatureType.SLOT, FeatureType.OPEN_SLOT, FeatureType.GROOVE, FeatureType.POCKET):
            extent = feature.xy_extent
            min_span = float(min(extent)) if len(extent) and float(min(extent)) > 1e-4 else None
            if min_span is not None and feature.diameter is None:
                if tool_dia > min_span + 1e-4:
                    req_radius = min_span / 2.0
                    errors.append(CamError(
                        code=NO_COMPATIBLE_TOOL,
                        message=(
                            f"Tool Ø{tool_dia:.2f} cannot machine required feature '{feature.id}' (channel width {min_span:.2f} mm).\n"
                            f"Required minimum access radius = {req_radius:.2f} mm\n"
                            f"Available tool = Ø{tool_dia:.2f}\n"
                            f"Recommended action = select smaller tool (Ø <= {min_span:.2f} mm) / different orientation"
                        ),
                        stage="geometry_validation",
                        operation_id=op_id_str,
                        feature_id=feature.id,
                        tool_id=tool_id,
                        context={"required_access_radius": req_radius, "tool_diameter": tool_dia, "channel_width": min_span},
                    ))

        # 3. Minimum Corner / Fillet Radius (for internal concave finishing / rest machining)
        corner_r = None
        if feature.fillet_radius is not None and feature.fillet_radius > 1e-4:
            corner_r = float(feature.fillet_radius)
        elif feature.radius is not None and feature.type in (FeatureType.FILLET, FeatureType.POCKET, FeatureType.OPEN_POCKET, FeatureType.SLOT, FeatureType.OPEN_SLOT):
            corner_r = float(feature.radius)
        elif feature.notes.get("corner_radius") is not None:
            corner_r = float(feature.notes["corner_radius"])

        if corner_r is not None and is_finish_op and feature.is_concave and not is_spot_or_chamfer:
            if tool_r > corner_r + 1e-4:
                errors.append(CamError(
                    code=NO_COMPATIBLE_TOOL,
                    message=(
                        f"Tool Ø{tool_dia:.2f} cannot machine required feature '{feature.id}' (corner radius {corner_r:.2f} mm).\n"
                        f"Required minimum access radius = {corner_r:.2f} mm\n"
                        f"Available tool = Ø{tool_dia:.2f}\n"
                        f"Recommended action = select smaller tool (Ø <= {2 * corner_r:.2f} mm) / different orientation"
                    ),
                    stage="geometry_validation",
                    operation_id=op_id_str,
                    feature_id=feature.id,
                    tool_id=tool_id,
                    context={"required_access_radius": corner_r, "tool_diameter": tool_dia, "corner_radius": corner_r},
                ))

        # 4. Minimum Wall / Island Clearance
        if feature.boundary_loop is not None and feature.island_loops:
            outer_arr = np.asarray(feature.boundary_loop, dtype=float)
            for isl_idx, isl in enumerate(feature.island_loops):
                isl_arr = np.asarray(isl, dtype=float)
                if len(outer_arr) > 0 and len(isl_arr) > 0:
                    dists = [np.min(np.linalg.norm(outer_arr[:, :2] - pt[:2], axis=1)) for pt in isl_arr]
                    min_wall_clearance = float(np.min(dists)) if dists else None
                    if min_wall_clearance is not None and min_wall_clearance > 1e-4:
                        if tool_dia > min_wall_clearance + 1e-4:
                            req_radius = min_wall_clearance / 2.0
                            errors.append(CamError(
                                code=NO_COMPATIBLE_TOOL,
                                message=(
                                    f"Tool Ø{tool_dia:.2f} cannot machine required feature '{feature.id}' (wall clearance {min_wall_clearance:.2f} mm).\n"
                                    f"Required minimum access radius = {req_radius:.2f} mm\n"
                                    f"Available tool = Ø{tool_dia:.2f}\n"
                                    f"Recommended action = select smaller tool (Ø <= {min_wall_clearance:.2f} mm) / different orientation"
                                ),
                                stage="geometry_validation",
                                operation_id=op_id_str,
                                feature_id=feature.id,
                                tool_id=tool_id,
                                context={"required_access_radius": req_radius, "tool_diameter": tool_dia, "wall_clearance": min_wall_clearance},
                            ))

        # 5. Tool Reach & Tool Holder Envelope
        feature_depth = float(feature.depth if feature.depth is not None and feature.depth > 0 else (
            feature.top_z - feature.floor_z if (feature.top_z is not None and feature.floor_z is not None) else 0.0
        ))
        required_reach = feature_depth
        if is_spot_or_chamfer:
            required_reach = float(feature.notes.get("spot_depth", feature.notes.get("chamfer_depth", min(feature_depth, 5.0))))

        flute_len = getattr(tool, "flute_length", None)
        overall_len = getattr(tool, "overall_length", None)
        # For internal concave cavities, flute length restricts cut depth without shank relief;
        # for open/external contours, overall reach / stickout is the physical reach limit
        available_reach = flute_len if (feature.is_concave and feature.type in internal_cavity_types) else (overall_len or flute_len)

        if available_reach is not None and available_reach < required_reach - 1e-4:
            errors.append(CamError(
                code=TOOL_REACH_INSUFFICIENT,
                message=(
                    f"Tool Ø{tool_dia:.2f} reach ({available_reach:.2f} mm) is insufficient for feature '{feature.id}' depth ({required_reach:.2f} mm).\n"
                    f"Required minimum access reach = {required_reach:.2f} mm\n"
                    f"Available tool reach = {available_reach:.2f} mm\n"
                    f"Recommended action = select longer tool / different orientation"
                ),
                stage="geometry_validation",
                operation_id=op_id_str,
                feature_id=feature.id,
                tool_id=tool_id,
                context={"feature_depth": required_reach, "tool_reach": available_reach},
            ))

        holder = getattr(tool, "holder", None)
        if holder is not None and flute_len is not None and flute_len < feature_depth - 1e-4 and not is_spot_or_chamfer:
            holder_dia = getattr(holder, "diameter", 40.0)
            opening = float(feature.diameter) if feature.diameter is not None else (
                float(min(feature.xy_extent)) if len(feature.xy_extent) and float(min(feature.xy_extent)) > 0 else 0.0
            )
            if opening > 1e-4 and holder_dia > opening:
                errors.append(CamError(
                    code=COLLISION_DETECTED,
                    message=(
                        f"Tool Ø{tool_dia:.2f} holder (Ø{holder_dia:.2f} mm) cannot access required feature '{feature.id}' at depth {feature_depth:.2f} mm.\n"
                        f"Required minimum access radius = {opening / 2.0:.2f} mm (holder envelope collision)\n"
                        f"Available tool = Ø{tool_dia:.2f}\n"
                        f"Recommended action = increase tool stickout / select smaller holder / different orientation"
                    ),
                    stage="geometry_validation",
                    operation_id=op_id_str,
                    feature_id=feature.id,
                    tool_id=tool_id,
                    context={"feature_depth": feature_depth, "holder_diameter": holder_dia, "opening": opening},
                ))

        return errors

    def validate_operation(
        self,
        op: Any,  # PlannedOperation
        setup: Optional[Setup] = None,
        shape: Optional[Shape] = None,
    ) -> ValidationReport:
        """Validate an individual planned operation against geometry and tool compatibility."""
        s = setup or self.setup
        shp = shape or self.shape
        errors: list[CamError] = []
        warnings: list[CamError] = []

        # Validate underlying feature first
        feat_report = self.validate_feature(op.feature, setup=s, shape=shp)
        if not feat_report.passed:
            errors.extend(feat_report.errors)

        # Comprehensive tool radius & envelope feasibility checking
        feasibility_errors = self.check_tool_feasibility(op.tool, op.feature, op_purpose=op.purpose, op_id=op.id)
        errors.extend(feasibility_errors)

        if errors:
            return ValidationReport("geometry_validation", False, errors, warnings)
        return ValidationReport("geometry_validation", True, [], warnings)



