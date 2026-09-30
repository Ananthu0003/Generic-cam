"""Arc geometry mathematics for toolpaths.

This module owns every arc decision: center correction, I/J offset rounding,
validity evaluation against controller formatting, linearization fallback and
polyline-to-arc fitting. It lives on the CAM engine side so that a
post-processor only has to *print* an arc that was already resolved.

Nothing here is controller syntax; the output is plain geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .semantic import MotionSegment, MotionType


def project_center_to_bisector(start: np.ndarray, end: np.ndarray,
                               center: np.ndarray) -> np.ndarray:
    """Project candidate arc center onto the perpendicular bisector of chord [start, end].

    This guarantees distance(start, center) == distance(end, center) exactly
    in IEEE 754 floating point arithmetic.
    """
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    chord_len_sq = dx * dx + dy * dy
    if chord_len_sq < 1e-12:
        # Full circle (start == end), center is already equidistant to start and end
        return center.copy()

    chord_len = math.sqrt(chord_len_sq)
    # Midpoint of chord
    mx = (start[0] + end[0]) * 0.5
    my = (start[1] + end[1]) * 0.5

    # Unit normal to chord
    nx = -dy / chord_len
    ny = dx / chord_len

    # Vector from midpoint to candidate center
    vmx = float(center[0] - mx)
    vmy = float(center[1] - my)

    # Height along normal
    h = vmx * nx + vmy * ny

    # Exact equidistant center
    proj_x = mx + h * nx
    proj_y = my + h * ny
    z = center[2] if len(center) > 2 else start[2]
    return np.array([proj_x, proj_y, z], dtype=float)


def semicircle_center_offset(start: np.ndarray, end: np.ndarray, ccw: bool) -> tuple[float, float]:
    """Compute I/J arc center offsets from start/end when arc_center is unknown.

    For a 2-point arc, the center lies on the perpendicular bisector.
    We assume a semicircle (180-degree arc) as the most reasonable default,
    placing the center at the midpoint of the chord offset by radius/2
    perpendicular to the chord in the direction dictated by ccw.
    """
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    chord = math.hypot(dx, dy)
    if chord < 1e-12:
        return 0.0, 0.0
    # Perpendicular direction (rotated 90 degrees)
    # For CCW: center is to the left of the chord direction
    # For CW: center is to the right
    if ccw:
        perp_x, perp_y = -dy / chord, dx / chord
    else:
        perp_x, perp_y = dy / chord, -dx / chord
    # For a semicircle, radius = chord/2, center is at midpoint + radius * perp
    radius = chord / 2.0
    cx = (start[0] + end[0]) / 2.0 + perp_x * radius
    cy = (start[1] + end[1]) / 2.0 + perp_y * radius
    # I/J are relative to start point
    return cx - start[0], cy - start[1]


def format_arc_offsets(start: np.ndarray, end: np.ndarray,
                       center: Optional[np.ndarray], ccw: bool,
                       precision: int) -> tuple[float, float]:
    """Compute and optimize I/J offsets so controller-side rounding keeps the radii equal.

    The returned offsets are the exact values a post should print: the center is
    pre-corrected here, not by the post.
    """
    if center is not None:
        proj_center = project_center_to_bisector(start, end, center)
        i_raw = float(proj_center[0] - start[0])
        j_raw = float(proj_center[1] - start[1])
    else:
        i_raw, j_raw = semicircle_center_offset(start, end, ccw)

    # Optimize for rounding precision
    p = precision
    step = 10.0 ** (-p)

    xs = round(float(start[0]), p)
    ys = round(float(start[1]), p)
    xe = round(float(end[0]), p)
    ye = round(float(end[1]), p)

    i_base = round(float(i_raw), p)
    j_base = round(float(j_raw), p)

    best_i = i_base
    best_j = j_base

    def radius_delta(cand_i: float, cand_j: float) -> float:
        r1 = math.hypot(cand_i, cand_j)
        cx = xs + cand_i
        cy = ys + cand_j
        r2 = math.hypot(xe - cx, ye - cy)
        return abs(r1 - r2)

    best_delta = radius_delta(i_base, j_base)

    # Micro-adjust by +/- 1 LSB to minimize post-rounded controller radial error
    if best_delta > 1e-4:
        for di in (-step, 0.0, step):
            for dj in (-step, 0.0, step):
                ci = round(i_base + di, p)
                cj = round(j_base + dj, p)
                delta = radius_delta(ci, cj)
                if delta < best_delta:
                    best_delta = delta
                    best_i = ci
                    best_j = cj

    return best_i, best_j


@dataclass
class ArcEvaluation:
    """Result of evaluating an arc against controller rounding, in NC values."""

    valid: bool
    i_offset: float
    j_offset: float
    radius_start: float
    radius_end: float
    delta_radius: float
    chord: float
    sweep: float
    start_rounded: tuple[float, float, float]
    end_rounded: tuple[float, float, float]
    center_rounded: tuple[float, float]
    linearize_hint: str = ""


def evaluate_arc(start: np.ndarray, end: np.ndarray, center: Optional[np.ndarray],
                 ccw: bool, precision: int, interpolation_tolerance: float) -> ArcEvaluation:
    """Validate an arc after the exact rounding the controller will apply."""
    i_off, j_off = format_arc_offsets(start, end, center, ccw, precision)

    p = precision
    xs = round(float(start[0]), p)
    ys = round(float(start[1]), p)
    zs = round(float(start[2]), p)
    xe = round(float(end[0]), p)
    ye = round(float(end[1]), p)
    ze = round(float(end[2]), p)
    i_nc = round(float(i_off), p)
    j_nc = round(float(j_off), p)

    cx_nc = xs + i_nc
    cy_nc = ys + j_nc

    r_start_nc = math.hypot(i_nc, j_nc)
    r_end_nc = math.hypot(xe - cx_nc, ye - cy_nc)
    delta_r_nc = abs(r_start_nc - r_end_nc)
    chord_nc = math.hypot(xe - xs, ye - ys)

    ux = xs - cx_nc
    uy = ys - cy_nc
    vx = xe - cx_nc
    vy = ye - cy_nc
    a0 = math.atan2(uy, ux)
    a1 = math.atan2(vy, vx)

    if ccw:
        sweep = (a1 - a0) % (2.0 * math.pi)
    else:
        sweep = (a0 - a1) % (2.0 * math.pi)

    if chord_nc < 1e-5 or sweep < 1e-5:
        sweep = 2.0 * math.pi

    is_valid_arc = (
        r_start_nc > 1e-4 and
        delta_r_nc <= interpolation_tolerance and
        chord_nc <= 2.0 * r_start_nc + interpolation_tolerance + 1e-4 and
        0.0 < sweep <= 2.0 * math.pi + 1e-4
    )

    return ArcEvaluation(
        valid=is_valid_arc,
        i_offset=i_nc,
        j_offset=j_nc,
        radius_start=r_start_nc,
        radius_end=r_end_nc,
        delta_radius=delta_r_nc,
        chord=chord_nc,
        sweep=sweep,
        start_rounded=(xs, ys, zs),
        end_rounded=(xe, ye, ze),
        center_rounded=(cx_nc, cy_nc),
        linearize_hint=f"dR={delta_r_nc:.4f}mm",
    )


def linearize_arc(evaluation: ArcEvaluation, start: np.ndarray, end: np.ndarray,
                  ccw: bool, chord_tolerance: float) -> list[np.ndarray]:
    """Expand an arc that cannot be represented into chord-limited linear points."""
    xs, ys, _ = evaluation.start_rounded
    cx_nc, cy_nc = evaluation.center_rounded
    a0 = math.atan2(evaluation.start_rounded[1] - cy_nc, evaluation.start_rounded[0] - cx_nc)
    sweep = evaluation.sweep
    is_ccw = ccw

    r_avg = max(evaluation.radius_start, 0.1)
    tol = max(chord_tolerance, 1e-4)
    cos_val = max(-1.0, min(1.0, 1.0 - tol / r_avg))
    max_dtheta = 2.0 * math.acos(cos_val)
    if max_dtheta < 1e-4:
        max_dtheta = math.radians(5.0)

    n_steps = max(2, int(math.ceil(sweep / max(max_dtheta, 1e-9))))
    points: list[np.ndarray] = []
    start_pos = np.asarray(start, dtype=float)
    end_pos = np.asarray(end, dtype=float)
    for k in range(1, n_steps + 1):
        t = k / n_steps
        ang = a0 + t * (sweep if is_ccw else -sweep)
        if k == n_steps:
            p_k = end_pos.copy()
        else:
            xk = cx_nc + evaluation.radius_start * math.cos(ang)
            yk = cy_nc + evaluation.radius_start * math.sin(ang)
            zk = start_pos[2] + t * (end_pos[2] - start_pos[2])
            p_k = np.array([xk, yk, zk])
        points.append(p_k)
    return points


def fit_arc(points: list[np.ndarray], center_tolerance: float = 0.002) -> tuple[bool, Optional[np.ndarray], bool]:
    """Algebraic Kåsa circle fit + bisector projection; returns (fitted, center, ccw)."""
    if len(points) < 3:
        return False, None, False

    p0 = points[0]
    p_mid = points[len(points) // 2]

    # Arc fitting is only valid within a flat plane (XY plane)
    for p in points:
        if abs(p[2] - p0[2]) > 1e-4:
            return False, None, False

    n = len(points)
    x = np.array([float(p[0]) for p in points])
    y = np.array([float(p[1]) for p in points])
    z = x * x + y * y

    m_xx = float(np.sum(x * x))
    m_xy = float(np.sum(x * y))
    m_yy = float(np.sum(y * y))
    m_x = float(np.sum(x))
    m_y = float(np.sum(y))
    m_xz = float(np.sum(x * z))
    m_yz = float(np.sum(y * z))
    m_z = float(np.sum(z))

    M = np.array([
        [m_xx, m_xy, m_x],
        [m_xy, m_yy, m_y],
        [m_x,  m_y,  float(n)]
    ], dtype=float)
    rhs = np.array([m_xz, m_yz, m_z], dtype=float)

    try:
        sol = np.linalg.solve(M, rhs)
        ux = float(sol[0] / 2.0)
        uy = float(sol[1] / 2.0)
    except np.linalg.LinAlgError:
        return False, None, False

    init_center_3d = np.array([ux, uy, p0[2]], dtype=float)

    exact_center_3d = project_center_to_bisector(p0, points[-1], init_center_3d)
    exact_center_2d = exact_center_3d[:2]

    r = float(np.linalg.norm(p0[:2] - exact_center_2d))
    if r < 1e-3 or r > 1e6:
        return False, None, False

    for p in points:
        dist = abs(float(np.linalg.norm(p[:2] - exact_center_2d)) - r)
        if dist > center_tolerance:
            return False, None, False

    v0 = p0[:2] - exact_center_2d
    vm = p_mid[:2] - exact_center_2d
    cross_z = v0[0] * vm[1] - v0[1] * vm[0]
    ccw = cross_z > 0

    return True, exact_center_3d, ccw


def fit_arcs(segments: list[MotionSegment]) -> list[MotionSegment]:
    """Merge runs of collinear-cut segments into a single arc when one fits.

    Engine-side toolpath refinement; the post never rewrites toolpaths.
    """
    if len(segments) < 3:
        return list(segments)

    result: list[MotionSegment] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.motion_type != MotionType.CUT or seg.feed is None:
            result.append(seg)
            i += 1
            continue

        pts = [seg.start.copy(), seg.end.copy()]
        j = i + 1
        while j < len(segments):
            next_seg = segments[j]
            if next_seg.motion_type != MotionType.CUT:
                break
            pts.append(next_seg.end.copy())
            if len(pts) >= 4:
                fitted, center, ccw = fit_arc(pts)
                if fitted:
                    j += 1
                    continue
                else:
                    pts.pop()
                    break
            else:
                j += 1

        if len(pts) >= 4:
            fitted, center, ccw = fit_arc(pts)
            if fitted:
                arc_seg = MotionSegment(
                    motion_type=MotionType.ARC_CCW if ccw else MotionType.ARC_CW,
                    start=pts[0],
                    end=pts[-1],
                    space=seg.space,
                    feed=seg.feed,
                    spindle=seg.spindle,
                    tool_id=seg.tool_id,
                    operation_id=seg.operation_id,
                    feature_id=seg.feature_id,
                    arc_center=center,
                    arc_ccw=ccw,
                )
                result.append(arc_seg)
                i = j
                continue

        result.append(seg)
        i += 1

    return result
