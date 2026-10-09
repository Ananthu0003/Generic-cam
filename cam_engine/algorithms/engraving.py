"""Engraving and V-Carving toolpath algorithms ported and adapted from FreeCAD CAM PathEngrave & PathVcarve.

Supports:
1. 2D/3D profile & text line engraving at fixed depth
2. V-Carving with variable Z-depth calculated from local stroke width and V-bit included angle
"""

from __future__ import annotations
import math
from typing import NamedTuple, Optional, Sequence
import numpy as np


class EngraveMotion(NamedTuple):
    motion_type: str                  # "rapid", "plunge", "cut", "retract"
    start: np.ndarray                 # [x, y, z]
    end: np.ndarray                   # [x, y, z]
    feed: Optional[float] = None      # Feed rate mm/min


def generate_line_engraving(
    curves: Sequence[Sequence[Sequence[float]]],
    depth_z: float,
    retract_z: float,
    feed_cut: float = 800.0,
    feed_plunge: float = 300.0,
) -> list[EngraveMotion]:
    """Generate 2D line engraving motions following strokes or centerline curves."""
    motions: list[EngraveMotion] = []

    for stroke in curves:
        pts = [np.asarray(p, dtype=float) for p in stroke]
        if len(pts) < 2:
            continue

        p_start_top = np.array([pts[0][0], pts[0][1], retract_z])
        p_start_cut = np.array([pts[0][0], pts[0][1], depth_z])

        # Plunge into stroke start
        motions.append(EngraveMotion("plunge", p_start_top, p_start_cut, feed=feed_plunge))

        # Cut along stroke
        for i in range(len(pts) - 1):
            p0 = np.array([pts[i][0], pts[i][1], depth_z])
            p1 = np.array([pts[i + 1][0], pts[i + 1][1], depth_z])
            motions.append(EngraveMotion("cut", p0, p1, feed=feed_cut))

        # Retract at stroke end
        p_end_cut = np.array([pts[-1][0], pts[-1][1], depth_z])
        p_end_top = np.array([pts[-1][0], pts[-1][1], retract_z])
        motions.append(EngraveMotion("retract", p_end_cut, p_end_top))

    return motions


def generate_v_carve(
    centerline_pts: Sequence[Sequence[float]],
    stroke_widths: Sequence[float],
    v_bit_included_angle_deg: float = 60.0,
    max_depth_z: float = -5.0,
    retract_z: float = 5.0,
    top_z: float = 0.0,
    feed_cut: float = 600.0,
    feed_plunge: float = 250.0,
) -> list[EngraveMotion]:
    """Generate true V-carving with variable Z-depth matching local stroke widths.

    Inspired by FreeCAD Path `vcarve.py`. Depth is automatically governed by the
    half-angle of the V-cutter: Z = top_z - (width / (2 * tan(theta / 2))).
    """
    pts = [np.asarray(p[:2], dtype=float) for p in centerline_pts]
    widths = list(stroke_widths)
    if len(pts) < 2 or len(widths) != len(pts):
        return []

    half_angle_rad = math.radians(v_bit_included_angle_deg / 2.0)
    tan_half = max(0.01, math.tan(half_angle_rad))

    def depth_for_width(w: float) -> float:
        d = float(w) / (2.0 * tan_half)
        z = top_z - d
        return max(max_depth_z, z)

    motions: list[EngraveMotion] = []

    # Calculate 3D waypoints with variable Z
    pts_3d: list[np.ndarray] = []
    for p, w in zip(pts, widths):
        z = depth_for_width(w)
        pts_3d.append(np.array([p[0], p[1], z]))

    p_start_top = np.array([pts_3d[0][0], pts_3d[0][1], retract_z])
    motions.append(EngraveMotion("plunge", p_start_top, pts_3d[0], feed=feed_plunge))

    for i in range(len(pts_3d) - 1):
        motions.append(EngraveMotion("cut", pts_3d[i], pts_3d[i + 1], feed=feed_cut))

    p_end_top = np.array([pts_3d[-1][0], pts_3d[-1][1], retract_z])
    motions.append(EngraveMotion("retract", pts_3d[-1], p_end_top))

    return motions
