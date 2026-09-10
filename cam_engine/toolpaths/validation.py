"""Validation gates + collision/gouge detection on semantic toolpaths.

Gate 4 lives here: motions must be geometrically valid, inside machine travel,
gouge-free against the finished model, holder-safe, flute-safe, and rapid-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..context import MachineConfig, PlanningContext, Setup, Tool
from ..coords import CoordSpace, transform_points
from ..errors import (CamError, COLLISION_DETECTED, GOUGE_DETECTED,
                      INVALID_TOOLPATH, MACHINE_AXIS_LIMIT, TOOL_REACH_INSUFFICIENT,
                      ErrorSeverity)
from ..geometry import TriangleMesh
from .semantic import MotionSegment, MotionType, Toolpath

CUTTING = {MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP, MotionType.HELIX,
           MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.ENTRY, MotionType.EXIT}


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
                    errors.append(CamError(
                        code=MACHINE_AXIS_LIMIT,
                        message=f"segment {i} {ax_name} outside travel "
                                f"[{lo:.1f}, {hi:.1f}]: {vals.min():.1f}..{vals.max():.1f}",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id))

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

                if seg.motion_type in CUTTING and seg.space is CoordSpace.SETUP:
                    for p in (seg.start, seg.end):
                        g = self._gouge_depth(tool, p)
                        if g > 0.05:
                            errors.append(CamError(
                                code=GOUGE_DETECTED,
                                message=f"segment {i} gouges model by {g:.2f}mm at "
                                        f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})",
                                stage="validation", operation_id=tp.operation_id,
                                tool_id=tool.id))

                for p in (seg.start, seg.end):
                    if p[2] < -tool.flute_length:
                        errors.append(CamError(
                            code=TOOL_REACH_INSUFFICIENT,
                            message=f"segment {i} at Z={p[2]:.1f} exceeds flute length "
                                    f"{tool.flute_length:.1f}mm",
                            stage="validation", operation_id=tp.operation_id,
                            tool_id=tool.id))

                if self._check_holder_collision(tool, seg):
                    errors.append(CamError(
                        code=COLLISION_DETECTED,
                        message=f"segment {i} holder collision with part at "
                                f"({seg.start[0]:.2f},{seg.start[1]:.2f},{seg.start[2]:.2f})",
                        stage="validation", operation_id=tp.operation_id,
                        tool_id=tool.id))

            if self._check_rapid_gouge(tool, seg):
                warnings.append(CamError(
                    code=GOUGE_DETECTED,
                    message=f"rapid segment {i} may gouge model",
                    stage="validation", operation_id=tp.operation_id,
                    severity=ErrorSeverity.WARNING))

        if errors:
            return ValidationReport("toolpath", False, errors, warnings)
        return ValidationReport("toolpath", True, [], warnings)

    def _gouge_depth(self, tool: Tool, p: np.ndarray) -> float:
        if not len(self.mesh.triangles):
            return 0.0
        r = tool.diameter / 2.0
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
                    if h < p[2] - 1e-6:
                        best = max(best, p[2] - h)
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
        for p in (seg.start, seg.end):
            hs = self.mesh.heights_above(p[0], p[1])
            for h in hs:
                if p[2] < h - tool.diameter / 2.0 - 1e-6:
                    return True
        return False
