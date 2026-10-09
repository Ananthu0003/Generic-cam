"""Advanced CAM algorithms inspired by FreeCAD CAM / Path Workbench.

Provides:
- Zigzag and Spiral facing with smooth arc links and boundary containment
- Lead-in and Lead-out tangential circular arcs with cutter radius compensation (G41/G42)
- Progressive ramp entry (linear & helical)
- Dogbone internal corner relief overcuts
"""

from .facing import generate_zigzag_facing, generate_spiral_facing, generate_circular_spiral_facing, FacingPattern
from .dressups import (
    generate_lead_in_out,
    generate_ramp_entry,
    generate_dogbone_corners,
    generate_holding_tabs,
    HoldingTab,
    generate_linear_array,
    generate_polar_array,
)
from .probing import generate_surface_probe, generate_bore_boss_probe, generate_corner_probe, ProbeType
from .engraving import generate_line_engraving, generate_v_carve

__all__ = [
    "generate_zigzag_facing",
    "generate_spiral_facing",
    "generate_circular_spiral_facing",
    "FacingPattern",
    "generate_lead_in_out",
    "generate_ramp_entry",
    "generate_dogbone_corners",
    "generate_holding_tabs",
    "HoldingTab",
    "generate_linear_array",
    "generate_polar_array",
    "generate_surface_probe",
    "generate_bore_boss_probe",
    "generate_corner_probe",
    "ProbeType",
    "generate_line_engraving",
    "generate_v_carve",
]


