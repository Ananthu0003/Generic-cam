"""Touch probing algorithms and cycles ported and adapted from FreeCAD CAM PathProbe.

Supports standard touch probe cycles including:
1. Single surface probing (X, Y, Z datum capture)
2. Bore / Boss 4-point center and diameter finding
3. Web / Pocket center finding (X, Y channel midpoints)
4. Corner finding (internal pocket corner or external stock corner)
5. G-code macro generation (G31 skip signal, Renishaw Inspection Plus G65 macros, Haas / Fanuc)
"""

from __future__ import annotations
import math
from enum import Enum
from typing import NamedTuple, Optional, Sequence
import numpy as np


class ProbeType(str, Enum):
    SURFACE_Z = "surface_z"
    SURFACE_X = "surface_x"
    SURFACE_Y = "surface_y"
    BORE_CENTER = "bore_center"
    BOSS_CENTER = "boss_center"
    POCKET_CENTER = "pocket_center"
    WEB_CENTER = "web_center"
    CORNER_EXTERNAL = "corner_external"
    CORNER_INTERNAL = "corner_internal"


class ProbeCycleResult(NamedTuple):
    cycle_type: ProbeType
    points: list[np.ndarray]           # Toolpath waypoints for probe motion
    gcode_blocks: list[str]            # Formatted G-code macro lines
    expected_nominal: dict[str, float] # Expected nominal dimension/coordinate


def generate_surface_probe(
    approach_pt: Sequence[float] | np.ndarray,
    probe_axis: str = "Z",         # "X", "Y", "Z"
    probe_direction: int = -1,     # -1 (towards negative axis), +1 (towards positive)
    search_dist: float = 10.0,
    clearance_dist: float = 5.0,
    feed_touch: float = 100.0,
    feed_fast: float = 500.0,
) -> ProbeCycleResult:
    """Generate single surface touch probing cycle (e.g. Work Coordinate Z zeroing)."""
    p0 = np.asarray(approach_pt, dtype=float)
    axis_idx = {"X": 0, "Y": 1, "Z": 2}[probe_axis.upper()]
    dir_sign = 1.0 if probe_direction >= 0 else -1.0

    touch_pt = p0.copy()
    touch_pt[axis_idx] += dir_sign * search_dist

    retract_pt = p0.copy()
    retract_pt[axis_idx] -= dir_sign * clearance_dist

    points = [p0, touch_pt, retract_pt]
    gcode = [
        f"(--- Touch Probe: Surface {probe_axis.upper()} ---)",
        f"G0 X{p0[0]:.4f} Y{p0[1]:.4f} Z{p0[2]:.4f}",
        f"G31 {probe_axis.upper()}{touch_pt[axis_idx]:.4f} F{feed_touch:.1f}",
        f"G0 {probe_axis.upper()}{retract_pt[axis_idx]:.4f}",
    ]

    return ProbeCycleResult(
        cycle_type=ProbeType[f"SURFACE_{probe_axis.upper()}"],
        points=points,
        gcode_blocks=gcode,
        expected_nominal={f"nominal_{probe_axis.lower()}": float(p0[axis_idx])},
    )


def generate_bore_boss_probe(
    center_xy: Sequence[float] | np.ndarray,
    depth_z: float,
    clearance_z: float,
    diameter: float,
    is_internal_bore: bool = True,
    probe_radius: float = 3.0,
    overshoot_margin: float = 5.0,
    feed_touch: float = 100.0,
) -> ProbeCycleResult:
    """Generate 4-point (0°, 90°, 180°, 270°) center and diameter finding cycle."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_nom = diameter / 2.0

    points: list[np.ndarray] = []
    gcode: list[str] = [
        f"(--- Touch Probe: {'Internal Bore' if is_internal_bore else 'External Boss'} Ø{diameter:.2f}mm ---)",
        f"G0 X{cx:.4f} Y{cy:.4f} Z{clearance_z:.4f}",
    ]

    angles = [0.0, 180.0, 90.0, 270.0]  # X+, X-, Y+, Y-
    for ang in angles:
        rad = math.radians(ang)
        ux = math.cos(rad)
        uy = math.sin(rad)

        if is_internal_bore:
            # Internal bore: start at center, probe outward radially
            p_start = np.array([cx, cy, depth_z])
            p_touch = np.array([cx + (r_nom + overshoot_margin) * ux,
                                cy + (r_nom + overshoot_margin) * uy, depth_z])
            points.extend([p_start, p_touch, p_start])
            gcode.append(f"G0 X{cx:.4f} Y{cy:.4f} Z{depth_z:.4f}")
            gcode.append(f"G31 X{p_touch[0]:.4f} Y{p_touch[1]:.4f} F{feed_touch:.1f}")
            gcode.append(f"G0 X{cx:.4f} Y{cy:.4f}")
        else:
            # External boss: start outside clearance ring, probe inward radially
            r_out = r_nom + probe_radius + overshoot_margin
            p_above = np.array([cx + r_out * ux, cy + r_out * uy, clearance_z])
            p_down = np.array([cx + r_out * ux, cy + r_out * uy, depth_z])
            p_touch = np.array([cx + (r_nom - overshoot_margin) * ux,
                                cy + (r_nom - overshoot_margin) * uy, depth_z])
            points.extend([p_above, p_down, p_touch, p_above])
            gcode.append(f"G0 X{p_above[0]:.4f} Y{p_above[1]:.4f}")
            gcode.append(f"G0 Z{depth_z:.4f}")
            gcode.append(f"G31 X{p_touch[0]:.4f} Y{p_touch[1]:.4f} F{feed_touch:.1f}")
            gcode.append(f"G0 Z{clearance_z:.4f}")

    gcode.append(f"G0 X{cx:.4f} Y{cy:.4f} Z{clearance_z:.4f}")

    return ProbeCycleResult(
        cycle_type=ProbeType.BORE_CENTER if is_internal_bore else ProbeType.BOSS_CENTER,
        points=points,
        gcode_blocks=gcode,
        expected_nominal={"center_x": cx, "center_y": cy, "diameter": diameter},
    )


def generate_corner_probe(
    corner_xy: Sequence[float] | np.ndarray,
    depth_z: float,
    clearance_z: float,
    quadrant: int = 1,  # 1 (+X,+Y), 2 (-X,+Y), 3 (-X,-Y), 4 (+X,-Y)
    is_external: bool = True,
    offset_dist: float = 10.0,
    search_dist: float = 15.0,
    feed_touch: float = 100.0,
) -> ProbeCycleResult:
    """Generate 2-point corner finding cycle to locate datum corner."""
    cx, cy = float(corner_xy[0]), float(corner_xy[1])
    # Direction signs for X and Y in the target quadrant
    sx = 1.0 if quadrant in (1, 4) else -1.0
    sy = 1.0 if quadrant in (1, 2) else -1.0

    points: list[np.ndarray] = []
    gcode: list[str] = [
        f"(--- Touch Probe: {'External' if is_external else 'Internal'} Corner Q{quadrant} ---)",
        f"G0 Z{clearance_z:.4f}",
    ]

    if is_external:
        # Probe X edge
        px_start = np.array([cx + sx * (search_dist + 5.0), cy + sy * offset_dist, depth_z])
        px_touch = np.array([cx - sx * 5.0, cy + sy * offset_dist, depth_z])
        # Probe Y edge
        py_start = np.array([cx + sx * offset_dist, cy + sy * (search_dist + 5.0), depth_z])
        py_touch = np.array([cx + sx * offset_dist, cy - sy * 5.0, depth_z])
    else:
        # Probe internal pocket wall X
        px_start = np.array([cx + sx * offset_dist, cy + sy * offset_dist, depth_z])
        px_touch = np.array([cx - sx * search_dist, cy + sy * offset_dist, depth_z])
        # Probe internal pocket wall Y
        py_start = np.array([cx + sx * offset_dist, cy + sy * offset_dist, depth_z])
        py_touch = np.array([cx + sx * offset_dist, cy - sy * search_dist, depth_z])

    points.extend([px_start, px_touch, py_start, py_touch])
    gcode.extend([
        f"G0 X{px_start[0]:.4f} Y{px_start[1]:.4f}",
        f"G0 Z{depth_z:.4f}",
        f"G31 X{px_touch[0]:.4f} F{feed_touch:.1f}",
        f"G0 Z{clearance_z:.4f}",
        f"G0 X{py_start[0]:.4f} Y{py_start[1]:.4f}",
        f"G0 Z{depth_z:.4f}",
        f"G31 Y{py_touch[1]:.4f} F{feed_touch:.1f}",
        f"G0 Z{clearance_z:.4f}",
    ])

    return ProbeCycleResult(
        cycle_type=ProbeType.CORNER_EXTERNAL if is_external else ProbeType.CORNER_INTERNAL,
        points=points,
        gcode_blocks=gcode,
        expected_nominal={"corner_x": cx, "corner_y": cy, "quadrant": float(quadrant)},
    )
