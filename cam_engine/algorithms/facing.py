"""Facing toolpath algorithms ported and adapted from FreeCAD CAM PathBaseGenerator.

Implements:
1. Zigzag facing with continuous tangential arc links between rows
2. Inward/outward spiral facing for rectangular envelopes with constant stepover
"""

from __future__ import annotations
import math
from enum import Enum
from typing import NamedTuple, Optional
import numpy as np


class FacingPattern(str, Enum):
    ZIGZAG = "zigzag"
    SPIRAL = "spiral"
    ONE_WAY = "one_way"
    CONTOUR = "contour"


class FacingMotionSegment(NamedTuple):
    motion_type: str       # "cut", "arc_cw", "arc_ccw", "rapid", "entry", "exit"
    start: np.ndarray      # [x, y, z]
    end: np.ndarray        # [x, y, z]
    arc_center: Optional[np.ndarray] = None  # [cx, cy, cz] for arc moves


def generate_zigzag_facing(
    bounds_min: Sequence[float] | np.ndarray,
    bounds_max: Sequence[float] | np.ndarray,
    tool_diameter: float,
    stepover_mm: float,
    z_depth: float,
    angle_deg: float = 0.0,
    stock_margin_mm: float = 2.0,
    smooth_arc_links: bool = True,
) -> list[FacingMotionSegment]:
    """Generate high-efficiency zigzag facing toolpaths with smooth turnaround arcs.

    Inspired by FreeCAD Path `zigzag_facing.py`. Turnarounds between raster lines
    use 180° circular arcs to preserve machine momentum and avoid corner dwell marks.
    """
    bmin = np.asarray(bounds_min[:2], dtype=float)
    bmax = np.asarray(bounds_max[:2], dtype=float)
    r_tool = tool_diameter / 2.0
    stepover = max(0.5, min(stepover_mm, tool_diameter * 0.95))

    # Overhang tool beyond bounds by tool radius + margin to fully clean part edges
    x_start = bmin[0] - r_tool - stock_margin_mm
    x_end = bmax[0] + r_tool + stock_margin_mm
    y_min = bmin[1] - r_tool - stock_margin_mm
    y_max = bmax[1] + r_tool + stock_margin_mm

    y_span = max(y_max - y_min, 1.0)
    n_rows = max(1, int(math.ceil(y_span / stepover)) + 1)
    actual_step = y_span / max(1, n_rows - 1) if n_rows > 1 else y_span

    segments: list[FacingMotionSegment] = []
    prev_end: Optional[np.ndarray] = None

    for r in range(n_rows):
        y_curr = y_min + r * actual_step
        if r % 2 == 0:
            p0 = np.array([x_start, y_curr, z_depth], dtype=float)
            p1 = np.array([x_end, y_curr, z_depth], dtype=float)
        else:
            p0 = np.array([x_end, y_curr, z_depth], dtype=float)
            p1 = np.array([x_start, y_curr, z_depth], dtype=float)

        if prev_end is not None:
            if smooth_arc_links and abs(actual_step) > 1e-4:
                # 180° semi-circular transition arc connecting row endpoints
                arc_r = actual_step / 2.0
                arc_center = np.array([prev_end[0], prev_end[1] + arc_r, z_depth], dtype=float)
                is_cw = (r % 2 == 1)  # moving right-to-left -> arc turns CW
                motion = "arc_cw" if is_cw else "arc_ccw"
                segments.append(FacingMotionSegment(
                    motion_type=motion,
                    start=prev_end,
                    end=p0,
                    arc_center=arc_center,
                ))
            else:
                segments.append(FacingMotionSegment(
                    motion_type="cut",
                    start=prev_end,
                    end=p0,
                ))

        # Main linear cutting pass across face
        segments.append(FacingMotionSegment(
            motion_type="cut",
            start=p0,
            end=p1,
        ))
        prev_end = p1

    return segments


def generate_spiral_facing(
    bounds_min: Sequence[float] | np.ndarray,
    bounds_max: Sequence[float] | np.ndarray,
    tool_diameter: float,
    stepover_mm: float,
    z_depth: float,
    stock_margin_mm: float = 2.0,
) -> list[FacingMotionSegment]:
    """Generate inward rectangular spiral facing toolpaths with constant stepover.

    Inspired by FreeCAD Path `spiral_facing.py`. Starts from outer stock boundary
    and spirals inward toward center, ensuring constant climb/conventional cutting direction.
    """
    bmin = np.asarray(bounds_min[:2], dtype=float)
    bmax = np.asarray(bounds_max[:2], dtype=float)
    r_tool = tool_diameter / 2.0
    stepover = max(0.5, min(stepover_mm, tool_diameter * 0.85))

    x0 = bmin[0] - r_tool - stock_margin_mm
    x1 = bmax[0] + r_tool + stock_margin_mm
    y0 = bmin[1] - r_tool - stock_margin_mm
    y1 = bmax[1] + r_tool + stock_margin_mm

    segments: list[FacingMotionSegment] = []
    prev_pt = np.array([x0, y0, z_depth], dtype=float)

    while (x1 - x0 > r_tool * 0.5) and (y1 - y0 > r_tool * 0.5):
        # 4 corners of current outer rectangular loop (CCW)
        c1 = np.array([x1, y0, z_depth], dtype=float)
        c2 = np.array([x1, y1, z_depth], dtype=float)
        c3 = np.array([x0, y1, z_depth], dtype=float)
        c4 = np.array([x0 + stepover, y0 + stepover, z_depth], dtype=float)

        segments.append(FacingMotionSegment("cut", prev_pt, c1))
        segments.append(FacingMotionSegment("cut", c1, c2))
        segments.append(FacingMotionSegment("cut", c2, c3))
        segments.append(FacingMotionSegment("cut", c3, c4))

        x0 += stepover
        x1 -= stepover
        y0 += stepover
        y1 -= stepover
        prev_pt = c4

    # Final center clearing pass if span remains
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    center_pt = np.array([cx, cy, z_depth], dtype=float)
    if np.linalg.norm(prev_pt[:2] - center_pt[:2]) > 0.5:
        segments.append(FacingMotionSegment("cut", prev_pt, center_pt))

    return segments


def generate_circular_spiral_facing(
    center_xy: Sequence[float] | np.ndarray,
    outer_radius: float,
    tool_diameter: float,
    stepover_mm: float,
    z_depth: float,
    stock_margin_mm: float = 2.0,
    climb: bool = True,
) -> list[FacingMotionSegment]:
    """Generate inward circular concentric/spiral facing toolpaths with constant stepover.

    Starts from outer stock radius (extended by tool radius + stock margin) and performs
    circular arcs (CW for climb milling or CCW for conventional) stepping down in radius
    until reaching the center.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_tool = tool_diameter / 2.0
    stepover = max(0.5, min(stepover_mm, tool_diameter * 0.85))

    start_radius = outer_radius + r_tool + stock_margin_mm
    segments: list[FacingMotionSegment] = []

    radii: list[float] = []
    curr_r = start_radius
    while curr_r > 0.2:
        radii.append(curr_r)
        curr_r -= stepover
    if not radii or radii[-1] > 0.2:
        radii.append(0.0)

    prev_pt: Optional[np.ndarray] = None
    arc_motion = "arc_cw" if climb else "arc_ccw"

    for r in radii:
        if r <= 0.2:
            center_pt = np.array([cx, cy, z_depth], dtype=float)
            if prev_pt is not None:
                segments.append(FacingMotionSegment("cut", prev_pt, center_pt))
            break

        p0 = np.array([cx - r, cy, z_depth], dtype=float)
        p_mid = np.array([cx + r, cy, z_depth], dtype=float)
        center_3d = np.array([cx, cy, z_depth], dtype=float)

        if prev_pt is not None:
            segments.append(FacingMotionSegment("cut", prev_pt, p0))

        # First 180° arc: p0 -> p_mid
        segments.append(FacingMotionSegment(
            motion_type=arc_motion,
            start=p0,
            end=p_mid,
            arc_center=center_3d,
        ))
        # Second 180° arc: p_mid -> p0
        segments.append(FacingMotionSegment(
            motion_type=arc_motion,
            start=p_mid,
            end=p0,
            arc_center=center_3d,
        ))

        prev_pt = p0

    return segments

