"""Validation gates + collision/gouge detection on semantic toolpaths.

Gate 4 lives here: motions must be geometrically valid, inside machine travel,
gouge-free against the finished model, holder-safe, flute-safe, and rapid-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..context import MachineConfig, PlanningContext, Setup, Tool, ToolType
from ..coords import CoordSpace, transform_points
from ..errors import (CamError, COLLISION_DETECTED, GOUGE_DETECTED,
                      INVALID_TOOLPATH, MACHINE_AXIS_LIMIT, TOOL_REACH_INSUFFICIENT,
                      ErrorSeverity)
from ..geometry import TriangleMesh
from .semantic import MotionSegment, MotionType, Toolpath

CUTTING = {MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP, MotionType.HELIX,
           MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.ENTRY, MotionType.EXIT}

# Operations where gouge checking should be skipped (tool is intentionally inside feature)
_NO_GOUGE_CHECK_PURPOSES = {"drilling", "boring", "spot_drilling", "reaming", "tapping"}


@dataclass
class ValidationReport:
    gate: str
    passed: bool
    errors: list[CamError] = field(default_factory=list)
    warnings: list[CamError] = field(default_factory=list)


class ToolpathValidator:
    """Gate 4: validate semantic toolpaths against real geometry + machine."""

    def __init__(self, context: PlanningContext, setup: Setup,
                 mesh: TriangleMesh, clearance_z: float):
        self.context = context
        self.setup = setup
        self.mesh = mesh
        self.clearance_z = clearance_z

    def validate(self, tp: Toolpath) -> ValidationReport:
        errors: list[CamError] = []
        warnings: list[CamError] = []
        tool = self.context.tool_by_id(tp.tool_id)
        if tool is None:
            return ValidationReport("toolpath", False, [CamError(
                code=INVALID_TOOLPATH, message=f"unknown tool {tp.tool_id}",
                stage="validation", operation_id=tp.operation_id)])

        stock_min, stock_max = self.setup.stock.bounds_min, self.setup.stock.bounds_max
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
                for p in (seg.start, seg.end):
                    if (p[0] < stock_min[0] - 1e-6 or p[0] > stock_max[0] + 1e-6 or
                            p[1] < stock_min[1] - 1e-6 or p[1] > stock_max[1] + 1e-6):
                        warnings.append(CamError(
                            code=INVALID_TOOLPATH,
                            message=f"cut segment {i} outside stock footprint "
                                    f"({p[0]:.1f},{p[1]:.1f})",
                            stage="validation", operation_id=tp.operation_id,
                            severity=ErrorSeverity.WARNING))

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

            # Check holder collision on rapids/retracts too (tool may be near walls)
            if seg.motion_type in (MotionType.RAPID, MotionType.RETRACT, MotionType.LINK):
                if self._check_holder_collision(tool, seg):
                    warnings.append(CamError(
                        code=COLLISION_DETECTED,
                        message=f"rapid/retract segment {i} holder may collide with part",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id, severity=ErrorSeverity.WARNING))

            if self._check_rapid_gouge(tool, seg):
                warnings.append(CamError(
                    code=GOUGE_DETECTED,
                    message=f"rapid segment {i} may gouge model",
                    stage="validation", operation_id=tp.operation_id,
                    severity=ErrorSeverity.WARNING))

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

        if errors:
            return ValidationReport("toolpath", False, errors, warnings)
        return ValidationReport("toolpath", True, [], warnings)

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
        best = 0.0
        n = 8
        for k in range(n):
            a = 2 * np.pi * k / n
            for frac in (0.0, 0.5, 1.0):
                rr = r * frac
                x = p[0] + rr * np.cos(a)
                y = p[1] + rr * np.sin(a)
                hs = self.mesh.heights_above(x, y)
                for h in hs:
                    if h > tool_bottom_z + 1e-6:
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
