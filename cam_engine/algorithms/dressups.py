"""Toolpath dressup modifiers ported and adapted from FreeCAD CAM PathDressup.

Implements:
1. Tangential lead-in and lead-out circular arcs with cutter compensation support
2. Progressive linear and helical ramp entries
3. Dogbone internal corner overcut relief for mating parts
"""

from __future__ import annotations
import math
from typing import NamedTuple, Optional, Sequence
import numpy as np


class LeadArc(NamedTuple):
    entry_approach_point: np.ndarray    # Point in open air before lead-in
    lead_in_start: np.ndarray           # Start of circular lead-in arc
    lead_in_arc_center: np.ndarray      # Arc center for G2/G3
    lead_in_is_ccw: bool                # True for G3, False for G2
    cut_start_point: np.ndarray         # Point on part contour where cutting begins
    cut_end_point: np.ndarray           # Point on part contour where cutting ends
    lead_out_end: np.ndarray            # End of circular lead-out arc
    lead_out_arc_center: np.ndarray     # Arc center for G2/G3
    lead_out_is_ccw: bool               # True for G3, False for G2
    exit_departure_point: np.ndarray    # Retract point in open air after lead-out


def generate_lead_in_out(
    start_pt: Sequence[float] | np.ndarray,
    next_pt: Sequence[float] | np.ndarray,
    end_pt: Sequence[float] | np.ndarray,
    prev_end_pt: Sequence[float] | np.ndarray,
    tool_radius: float,
    lead_radius: Optional[float] = None,
    linear_lead_len: Optional[float] = None,
    side: str = "left",   # "left" (G41 climb) or "right" (G42 conventional)
) -> LeadArc:
    """Calculate tangential lead-in and lead-out arcs for smooth contour engagement.

    Inspired by FreeCAD Path `leadinout.py`. Enters the material via a 90° tangential arc
    so the cutter engages gradually without chatter or entrance gouging.
    """
    p_start = np.asarray(start_pt, dtype=float)
    p_next = np.asarray(next_pt, dtype=float)
    p_end = np.asarray(end_pt, dtype=float)
    p_prev_end = np.asarray(prev_end_pt, dtype=float)

    z = float(p_start[2])
    r_lead = float(lead_radius if lead_radius is not None else max(tool_radius * 0.8, 1.0))
    d_lin = float(linear_lead_len if linear_lead_len is not None else max(tool_radius * 1.2, 2.0))

    # Unit forward cutting vector at start
    v_cut = p_next[:2] - p_start[:2]
    norm_cut = np.linalg.norm(v_cut)
    v_cut = v_cut / norm_cut if norm_cut > 1e-9 else np.array([1.0, 0.0])

    # Perpendicular normal vector (pointing to the left of the cut direction)
    v_perp = np.array([-v_cut[1], v_cut[0]])
    if side == "right":
        v_perp = -v_perp

    # 90° Tangential Lead-In:
    # Arc center is placed perpendicularly to the cut vector at start
    c_in = p_start[:2] + v_perp * r_lead
    # Lead-in arc starts opposite the cut direction from the center
    p_in_start = c_in - v_cut * r_lead
    # Linear approach point further back for cutter compensation (G41) activation
    p_approach = p_in_start - v_cut * d_lin

    # Unit forward cutting vector at end
    v_exit = p_end[:2] - p_prev_end[:2]
    norm_exit = np.linalg.norm(v_exit)
    v_exit = v_exit / norm_exit if norm_exit > 1e-9 else v_cut
    v_perp_exit = np.array([-v_exit[1], v_exit[0]])
    if side == "right":
        v_perp_exit = -v_perp_exit

    # 90° Tangential Lead-Out:
    c_out = p_end[:2] + v_perp_exit * r_lead
    p_out_end = c_out + v_exit * r_lead
    p_depart = p_out_end + v_exit * d_lin

    is_ccw = (side == "right")

    return LeadArc(
        entry_approach_point=np.array([p_approach[0], p_approach[1], z]),
        lead_in_start=np.array([p_in_start[0], p_in_start[1], z]),
        lead_in_arc_center=np.array([c_in[0], c_in[1], z]),
        lead_in_is_ccw=is_ccw,
        cut_start_point=p_start,
        cut_end_point=p_end,
        lead_out_end=np.array([p_out_end[0], p_out_end[1], z]),
        lead_out_arc_center=np.array([c_out[0], c_out[1], z]),
        lead_out_is_ccw=not is_ccw,
        exit_departure_point=np.array([p_depart[0], p_depart[1], z]),
    )


def generate_dogbone_corners(
    polygon: Sequence[Sequence[float]],
    tool_radius: float,
    relief_factor: float = 1.05,
) -> list[np.ndarray]:
    """Insert dogbone corner overcuts on internal concave corners for assembly clearance.

    Inspired by FreeCAD Path `dogboneII.py`. Extends the cutter along the corner bisector
    so a rectangular mating part with sharp 90° corners can fit into an internal milled pocket.
    """
    pts = [np.asarray(p[:2], dtype=float) for p in polygon]
    n = len(pts)
    if n < 3:
        return [np.asarray(p, dtype=float) for p in polygon]

    r_eff = tool_radius * relief_factor
    out: list[np.ndarray] = []

    for i in range(n):
        p_prev = pts[(i - 1 + n) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        out.append(p_curr)

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        u1 = v1 / n1
        u2 = v2 / n2

        # 2D cross product to detect internal (concave) corners on CCW polygon
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        if cross > 0.1:  # Concave left turn in CCW loop
            # Bisector vector pointing into the corner
            bisector = -(u1 - u2)
            b_norm = np.linalg.norm(bisector)
            if b_norm > 1e-6:
                bisector_u = bisector / b_norm
                # Overcut distance required to clear the tool radius at the corner
                cos_half_angle = np.dot(-u1, u2)
                half_angle = math.acos(max(-1.0, min(1.0, cos_half_angle))) / 2.0
                overcut_dist = (r_eff / max(math.sin(half_angle), 0.1)) - tool_radius
                if overcut_dist > 0.05:
                    dogbone_pt = p_curr + bisector_u * overcut_dist
                    out.append(dogbone_pt)
                    out.append(p_curr)  # Return back to corner vertex

    return out


def generate_ramp_entry(
    start_pt: Sequence[float] | np.ndarray,
    target_z: float,
    ramp_angle_deg: float = 10.0,
    max_stepdown: float = 2.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate linear zigzag ramp segments descending safely to target cut depth.

    Inspired by FreeCAD Path `ramp_entry.py`. Replaces harsh vertical plunges
    with smooth ramping motion at the material's recommended ramp angle.
    """
    p0 = np.asarray(start_pt, dtype=float)
    z_start = float(p0[2])
    dz_total = abs(z_start - target_z)
    if dz_total < 1e-4:
        return []

    ramp_rad = math.radians(max(1.0, min(ramp_angle_deg, 45.0)))
    step_z = min(dz_total, max(0.5, max_stepdown))
    ramp_len = step_z / math.tan(ramp_rad)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    z_curr = z_start
    direction = 1.0  # +X then -X

    p_curr = p0.copy()
    while z_curr > target_z + 1e-4:
        z_next = max(target_z, z_curr - step_z)
        p_next = np.array([p_curr[0] + direction * ramp_len, p_curr[1], z_next])
        segments.append((p_curr.copy(), p_next.copy()))
        p_curr = p_next
        z_curr = z_next
        direction = -direction

    return segments


class HoldingTab(NamedTuple):
    center_distance: float   # Distance along path in mm
    width: float             # Width of tab in mm
    height: float            # Height of tab above floor_z in mm


def generate_holding_tabs(
    contour_points: Sequence[Sequence[float]] | np.ndarray,
    tab_count: int = 4,
    tab_width: float = 5.0,
    tab_height: float = 2.0,
    floor_z: float = 0.0,
    tab_shape: str = "trapezoidal",  # "trapezoidal" or "triangular"
) -> list[np.ndarray]:
    """Generate toolpath with elevated holding tabs (bridges) to retain cut parts.

    Inspired by FreeCAD Path `Tags.py`. When cutting complete perimeter profiles,
    elevates the toolpath over `tab_count` bridge regions so the part stays firmly
    clamped to the stock until manually snapped or trimmed.
    """
    pts = [np.asarray(p, dtype=float) for p in contour_points]
    if len(pts) < 2 or tab_count <= 0:
        return pts

    # Calculate segment lengths and cumulative distances
    seg_lens = []
    cum_lens = [0.0]
    for i in range(len(pts) - 1):
        d = float(np.linalg.norm(pts[i + 1][:2] - pts[i][:2]))
        seg_lens.append(d)
        cum_lens.append(cum_lens[-1] + d)

    total_len = cum_lens[-1]
    if total_len < tab_width * 2.0:
        return pts

    # Position tabs evenly along the perimeter
    spacing = total_len / tab_count
    tab_intervals: list[tuple[float, float]] = []  # (start_dist, end_dist)
    for t in range(tab_count):
        center = (t + 0.5) * spacing
        t_start = max(0.0, center - tab_width / 2.0)
        t_end = min(total_len, center + tab_width / 2.0)
        tab_intervals.append((t_start, t_end))

    tab_z = floor_z + tab_height
    out_pts: list[np.ndarray] = []

    def get_pt_at_dist(d: float) -> np.ndarray:
        for idx in range(len(seg_lens)):
            if cum_lens[idx] <= d <= cum_lens[idx + 1] or idx == len(seg_lens) - 1:
                seg_d = seg_lens[idx]
                if seg_d < 1e-9:
                    return pts[idx].copy()
                frac = max(0.0, min(1.0, (d - cum_lens[idx]) / seg_d))
                p = pts[idx] + frac * (pts[idx + 1] - pts[idx])
                return p
        return pts[-1].copy()

    # Walk along segments and insert elevated tab waypoints
    curr_d = 0.0
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        s_start = cum_lens[i]
        s_end = cum_lens[i + 1]

        # Add base start point
        if not out_pts:
            out_pts.append(np.array([p0[0], p0[1], floor_z]))

        # Check tabs that overlap this segment
        tabs_in_seg = [tab for tab in tab_intervals if not (tab[1] <= s_start or tab[0] >= s_end)]

        if not tabs_in_seg:
            out_pts.append(np.array([p1[0], p1[1], floor_z]))
        else:
            for tab_s, tab_e in tabs_in_seg:
                # Up ramp to tab
                if tab_s > s_start:
                    p_tab_start_xy = get_pt_at_dist(tab_s)
                    out_pts.append(np.array([p_tab_start_xy[0], p_tab_start_xy[1], floor_z]))
                p_tab_start_up = get_pt_at_dist(tab_s)
                out_pts.append(np.array([p_tab_start_up[0], p_tab_start_up[1], tab_z]))

                if tab_shape == "triangular":
                    # Mid peak
                    t_mid = 0.5 * (tab_s + tab_e)
                    p_mid = get_pt_at_dist(t_mid)
                    out_pts.append(np.array([p_mid[0], p_mid[1], tab_z]))
                else:
                    # Flat bridge across tab
                    p_tab_end_up = get_pt_at_dist(tab_e)
                    out_pts.append(np.array([p_tab_end_up[0], p_tab_end_up[1], tab_z]))

                # Down ramp from tab
                p_tab_end_dn = get_pt_at_dist(tab_e)
                out_pts.append(np.array([p_tab_end_dn[0], p_tab_end_dn[1], floor_z]))

            # Finally add the segment end point
            out_pts.append(np.array([p1[0], p1[1], floor_z]))

    return out_pts


def generate_linear_array(
    segments: Sequence[Sequence[Sequence[float]] | tuple[np.ndarray, np.ndarray]],
    count_x: int = 1,
    count_y: int = 1,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Replicate a toolpath pattern across a regular 2D grid array.

    Inspired by FreeCAD Path `Array.py` (Linear Array). Replicates cutting patterns
    for multi-cavity fixtures or repeated part features.
    """
    out_segments: list[tuple[np.ndarray, np.ndarray]] = []

    for ix in range(max(1, count_x)):
        dx = ix * offset_x
        for iy in range(max(1, count_y)):
            dy = iy * offset_y
            shift = np.array([dx, dy, 0.0])

            for seg in segments:
                p0 = np.asarray(seg[0], dtype=float) + shift
                p1 = np.asarray(seg[1], dtype=float) + shift
                out_segments.append((p0, p1))

    return out_segments


def generate_polar_array(
    segments: Sequence[Sequence[Sequence[float]] | tuple[np.ndarray, np.ndarray]],
    count: int = 4,
    total_angle_deg: float = 360.0,
    center_xy: Sequence[float] | np.ndarray = (0.0, 0.0),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Replicate a toolpath pattern rotated in a circular/polar pattern.

    Inspired by FreeCAD Path `Array.py` (Polar Array). Rotates feature toolpaths
    around a specified center axis (e.g., bolt circle pockets, turbine slots).
    """
    out_segments: list[tuple[np.ndarray, np.ndarray]] = []
    cx, cy = float(center_xy[0]), float(center_xy[1])
    n = max(1, count)
    step_ang = total_angle_deg / n if abs(total_angle_deg - 360.0) < 1e-4 else (total_angle_deg / max(1, n - 1))

    for i in range(n):
        ang_rad = math.radians(i * step_ang)
        cos_a = math.cos(ang_rad)
        sin_a = math.sin(ang_rad)

        def rotate_pt(pt: np.ndarray) -> np.ndarray:
            x_rel = pt[0] - cx
            y_rel = pt[1] - cy
            x_rot = cx + (x_rel * cos_a - y_rel * sin_a)
            y_rot = cy + (x_rel * sin_a + y_rel * cos_a)
            return np.array([x_rot, y_rot, pt[2]])

        for seg in segments:
            p0 = rotate_pt(np.asarray(seg[0], dtype=float))
            p1 = rotate_pt(np.asarray(seg[1], dtype=float))
            out_segments.append((p0, p1))

    return out_segments


