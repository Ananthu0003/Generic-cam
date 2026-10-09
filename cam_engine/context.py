"""Planning context: the single source of truth for the complete machining state.

Every downstream component consumes this context. No component invents values:
missing required data produces an explicit error, never a fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .coords import (
    CoordSpace,
    CoordinateSystemChain,
    make_transform,
    transform_points,
    transform_direction,
    transform_bounding_box,
    validate_rigid,
    _invert,
)
from .errors import (CamError, MISSING_MACHINE_DATA, MISSING_STOCK_GEOMETRY,
                     INVALID_SETUP)
from .geometry import ImportedModel, Shape, TriangleMesh, VoxelStock
from .units import Units


# ---------------------------------------------------------------- machine ---
class MachineType(str, Enum):
    THREE_AXIS_VERTICAL = "3axis_vertical"
    THREE_AXIS_HORIZONTAL = "3axis_horizontal"
    FOUR_AXIS = "4axis"
    FIVE_AXIS = "5axis"


@dataclass
class MachineAxis:
    name: str                 # X / Y / Z / A / B / C
    travel_min: float         # mm, machine coordinates
    travel_max: float
    rapid_rate: float         # mm/min
    max_feed: float           # mm/min


@dataclass
class MachineConfig:
    id: str
    name: str
    machine_type: MachineType
    axes: dict[str, MachineAxis]
    spindle_min_rpm: float
    spindle_max_rpm: float
    max_spindle_power_kw: float
    controller: str = "grbl"           # selects default post processor
    units: Units = Units.MM
    tool_change_position: Optional[np.ndarray] = None   # machine coords; required
    work_offsets: dict[str, np.ndarray] = field(default_factory=dict)  # "G54" -> machine coords of work origin
    safe_retract_height: Optional[float] = None         # mm above stock top, machine-consistent
    hsm_enabled: bool = True                            # high-speed lookahead / AICC / G187
    hsm_roughing_tolerance_mm: float = 0.05             # tolerance for roughing passes
    hsm_finishing_tolerance_mm: float = 0.005           # tolerance for finishing passes
    enable_cutter_comp: bool = True                     # G41/G42 wear/control compensation
    has_atc: bool = True                                # machine has automatic tool changer
    tool_change_strategy: Optional[str] = None          # "atc" | "manual" | "comment_only" | "custom" (None => derived from has_atc & controller)
    tool_change_command: str = "M6"                     # standard tool change command ("M6", "M0", etc.)
    tool_change_format: Optional[str] = None            # custom template (e.g., "T{tool_num:02d} M6")
    machine_retract_z: Optional[float] = None           # absolute machine Z safe return (e.g. 0.0 for G53/G28)
    home_command_z: Optional[str] = None                # custom machine Z homing command (e.g. "G53 G00 Z0.0")
    home_command_xy: Optional[str] = None               # custom machine XY homing command (e.g. "G53 G00 X0.0 Y0.0")


    def validate(self) -> None:
        problems = []
        if not self.axes:
            problems.append("no axes defined")
        for axis_name in ("X", "Y", "Z"):
            if axis_name not in self.axes:
                problems.append(f"missing axis {axis_name}")
        if self.spindle_max_rpm <= self.spindle_min_rpm or self.spindle_min_rpm < 0:
            problems.append("invalid spindle range")
        if self.tool_change_position is None:
            problems.append("tool_change_position not configured")
        if self.safe_retract_height is None:
            problems.append("safe_retract_height not configured")
        if not self.work_offsets:
            problems.append("no work offsets configured")
        if problems:
            raise CamError(
                code=MISSING_MACHINE_DATA,
                message="machine configuration incomplete: " + "; ".join(problems),
                stage="context",
            )

    def axis_for(self, name: str) -> MachineAxis:
        if name not in self.axes:
            raise CamError(MISSING_MACHINE_DATA, f"machine has no axis {name}", stage="context")
        return self.axes[name]


# ------------------------------------------------------------------ tools ---
class ToolType(str, Enum):
    FLAT_ENDMILL = "flat_endmill"
    BALL_ENDMILL = "ball_endmill"
    BULLNOSE_ENDMILL = "bullnose_endmill"
    DRILL = "drill"
    SPOT_DRILL = "spot_drill"              # dedicated NC spot drill
    CENTER_DRILL = "center_drill"          # center drill (conical pilot)
    FACE_MILL = "face_mill"
    BORING_BAR = "boring_bar"
    CHAMFER_MILL = "chamfer_mill"
    COUNTERSINK_TOOL = "countersink_tool"  # dedicated countersink / spotfacing tool
    TAP = "tap"
    REAMER = "reamer"
    THREAD_MILL = "thread_mill"
    GROOVE_CUTTER = "groove_cutter"
    V_BIT = "v_bit"                        # V-carving / engraving bit
    DOVETAIL_CUTTER = "dovetail_cutter"    # Dovetail & undercut milling cutter
    SLITTING_SAW = "slitting_saw"          # Slitting / slotting saw
    TOUCH_PROBE = "touch_probe"            # Electronic touch probe (Renishaw / Blum)



class DepthReference(str, Enum):
    """Reference datum plane for feature depth measurement."""
    TOP_FACE = "TOP_FACE"         # Depth measured from the feature's top face / finished face (default)
    STOCK_TOP = "STOCK_TOP"       # Depth measured from raw stock top surface
    WCS_ORIGIN = "WCS_ORIGIN"     # Depth absolute relative to WCS Z=0


class DrillDepthType(str, Enum):
    """Semantic definition of what the drawing depth dimension physically represents."""
    CYLINDRICAL_DEPTH = "CYLINDRICAL_DEPTH"   # Full-diameter usable cylindrical hole depth (shoulder depth)
    TIP_DEPTH = "TIP_DEPTH"                   # Drill tip/apex stopping depth (drawing depth = tip depth)
    TOTAL_HOLE_DEPTH = "TOTAL_HOLE_DEPTH"     # Total penetration from reference to tip apex
    THROUGH_BREAKTHROUGH = "THROUGH_BREAKTHROUGH" # Full diameter through-hole with breakthrough allowance


@dataclass
class DrillingDepthSemantics:
    """Rigorous, auditable drill depth semantics model linking CAD drawing requirements to CNC motion."""
    depth_reference: DepthReference = DepthReference.TOP_FACE
    depth_type: DrillDepthType = DrillDepthType.CYLINDRICAL_DEPTH
    target_depth: float = 0.0                  # Drawing depth dimension (e.g. 1.75mm or 1.27mm)
    reference_z: float = 0.0                   # Reference plane Z in setup coordinates (e.g. 3.22 or 0.0)
    tool_point_angle_deg: float = 118.0        # Conical point angle in degrees (e.g. 118.0)
    tool_tip_geometry: str = "conical"         # Conical, split_point, faceted, flat
    tool_tip_length: float = 0.0               # Axial length of tool tip cone (e.g. 0.174mm)
    breakthrough_margin: float = 0.0           # Extra breakthrough for through holes (e.g. 0.50mm)
    programmed_tip_depth: float = 0.0          # Calculated cycle / G-code Z endpoint for drill apex
    achieved_cylindrical_depth: float = 0.0    # Usable full-diameter depth produced
    total_hole_depth: float = 0.0              # Total distance from reference_z to drill apex
    remaining_web_thickness: Optional[float] = None # Web thickness if opposing hole or bottom web exists
    derivation_reason: str = ""                # Provenance description

    def to_dict(self) -> dict:
        return {
            "depth_reference": self.depth_reference.value if hasattr(self.depth_reference, "value") else str(self.depth_reference),
            "depth_type": self.depth_type.value if hasattr(self.depth_type, "value") else str(self.depth_type),
            "target_depth": self.target_depth,
            "reference_z": self.reference_z,
            "tool_point_angle_deg": self.tool_point_angle_deg,
            "tool_tip_geometry": self.tool_tip_geometry,
            "tool_tip_length": self.tool_tip_length,
            "breakthrough_margin": self.breakthrough_margin,
            "programmed_tip_depth": self.programmed_tip_depth,
            "achieved_cylindrical_depth": self.achieved_cylindrical_depth,
            "total_hole_depth": self.total_hole_depth,
            "remaining_web_thickness": self.remaining_web_thickness,
            "derivation_reason": self.derivation_reason,
        }


@dataclass
class ToolHolder:
    """Simplified holder envelope: cylinder above the flute for collision checks."""

    diameter: float       # mm
    length: float         # mm, from gauge line upward


@dataclass
class Tool:
    id: str
    tool_number: int              # machine tool-change number (from tool library)
    type: ToolType
    diameter: float               # mm, cutting diameter
    flute_length: float           # mm
    overall_length: float         # mm, gauge line to tip
    corner_radius: float = 0.0    # bullnose corner radius
    tip_radius: float = 0.0       # ball nose: diameter/2; drill: 0
    point_angle: float = 118.0    # tip point angle (e.g. 90, 118, 120, 135, 140, 142 deg)
    tip_geometry: str = "conical" # "conical", "split_point", "flat", "faceted", "brad_point"
    drill_type: str = "twist"     # "twist", "parabolic", "split_point", "micro_drill", "gun_drill", "flat_bottom"
    has_through_coolant: bool = False  # Through-spindle coolant (TSC) drill capability
    max_peck_ratio: Optional[float] = None  # Tool manufacturer maximum allowable single peck ratio (Q / D)
    flutes: int = 2
    material: str = "carbide"     # tool material key into cutting-data tables
    can_plunge: bool = False      # center-cutting capability
    can_spot_drill: bool = False  # explicit capability flag for spot drilling
    holder: Optional[ToolHolder] = None
    length_offset_h: Optional[int] = None   # machine H offset register (config, not assumed)

    diameter_offset_d: Optional[int] = None  # machine D offset register (config, not assumed)
    compensation_mode: Optional[str] = None  # default mode: "none", "wear", "geometry"
    operations: list[str] = field(default_factory=list)  # explicit operation purposes supported


    @property
    def effective_h(self) -> int:
        """Independent length offset register H (never assumed H == T)."""
        return self.length_offset_h if (self.length_offset_h is not None and self.length_offset_h > 0) else self.tool_number

    @property
    def effective_d(self) -> int:
        """Independent diameter offset register D (never assumed D == T or D == H)."""
        return self.diameter_offset_d if (self.diameter_offset_d is not None and self.diameter_offset_d > 0) else self.tool_number

    def get_length_offset_h(self) -> int:
        return self.effective_h

    def get_diameter_offset_d(self) -> int:
        return self.effective_d

    @property
    def tip_length(self) -> float:

        """Calculate the axial tool-tip / point length dynamically from actual tool definition.

        For standard drills / spot drills:
            L_tip = (D / 2) / tan(point_angle / 2)
        For ball nose mills:
            L_tip = D / 2 (or tip_radius)
        For flat endmills / 180° drills / flat tip geometry:
            L_tip = 0.0
        """
        return self.compute_tip_length(self.diameter)

    def compute_tip_length(self, diameter: Optional[float] = None) -> float:
        """Compute tip length for a given effective diameter or the full tool diameter."""
        eff_d = float(diameter if diameter is not None else self.diameter)
        if self.type in (ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL, ToolType.COUNTERSINK_TOOL):
            if self.tip_geometry == "flat" or self.point_angle >= 179.9:
                return 0.0
            half_angle_rad = math.radians(self.point_angle / 2.0)
            if half_angle_rad > 1e-4:
                return float((eff_d / 2.0) / math.tan(half_angle_rad))
            return 0.0
        elif self.type is ToolType.BALL_ENDMILL:
            return float(self.tip_radius if self.tip_radius > 0 else eff_d / 2.0)
        elif self.type is ToolType.BULLNOSE_ENDMILL:
            return float(self.corner_radius)
        return 0.0

    def compute_drill_depth_semantics(
        self,
        start_z: float,
        target_depth: float,
        depth_type: DrillDepthType | str = DrillDepthType.CYLINDRICAL_DEPTH,
        depth_reference: DepthReference | str = DepthReference.TOP_FACE,
        breakthrough: float = 0.0,
        remaining_web: Optional[float] = None,
        is_through: bool = False,
    ) -> DrillingDepthSemantics:
        """Derive authoritative drilling depth semantics from drawing target and tool tip geometry."""
        if isinstance(depth_reference, DepthReference):
            d_ref = depth_reference
        else:
            d_ref_str = str(depth_reference).upper()
            if "STOCK" in d_ref_str:
                d_ref = DepthReference.STOCK_TOP
            elif "WCS" in d_ref_str or "ORIGIN" in d_ref_str:
                d_ref = DepthReference.WCS_ORIGIN
            else:
                d_ref = DepthReference.TOP_FACE

        if isinstance(depth_type, DrillDepthType):
            d_type = depth_type
        else:
            d_type_str = str(depth_type).upper()
            if is_through or "THROUGH" in d_type_str:
                d_type = DrillDepthType.THROUGH_BREAKTHROUGH
            elif d_type_str in ("TIP", "TIP_DEPTH"):
                d_type = DrillDepthType.TIP_DEPTH
            elif d_type_str in ("TOTAL", "TOTAL_HOLE_DEPTH"):
                d_type = DrillDepthType.TOTAL_HOLE_DEPTH
            else:
                d_type = DrillDepthType.CYLINDRICAL_DEPTH

        tip_len = float(self.tip_length)
        pt_angle = float(getattr(self, "point_angle", 118.0) or 118.0)
        tip_geom = str(getattr(self, "tip_geometry", "conical") or "conical")
        target_d = float(target_depth)
        ref_z = float(start_z)
        bt = float(breakthrough)

        if d_type is DrillDepthType.THROUGH_BREAKTHROUGH:
            achieved_cyl = target_d
            total_penetration = target_d + tip_len + bt
            prog_z = ref_z - total_penetration
            reason = f"Through hole: target={target_d:.3f}mm + tip={tip_len:.3f}mm + breakthrough={bt:.3f}mm"
        elif d_type is DrillDepthType.CYLINDRICAL_DEPTH:
            achieved_cyl = target_d
            total_penetration = target_d + tip_len + bt
            prog_z = ref_z - total_penetration
            reason = f"Full cylindrical depth target: {target_d:.3f}mm; tip programmed to {prog_z:.3f} to clear conical apex ({tip_len:.3f}mm)"
        elif d_type is DrillDepthType.TIP_DEPTH:
            total_penetration = target_d + bt
            prog_z = ref_z - total_penetration
            achieved_cyl = max(0.0, total_penetration - tip_len)
            reason = f"Drawing tip/apex depth target: {target_d:.3f}mm; resulting full-diameter cylindrical depth is {achieved_cyl:.3f}mm"
        else:
            total_penetration = target_d + bt
            prog_z = ref_z - total_penetration
            achieved_cyl = max(0.0, total_penetration - tip_len)
            reason = f"Total hole depth target: {target_d:.3f}mm"

        return DrillingDepthSemantics(
            depth_reference=d_ref,
            depth_type=d_type,
            target_depth=target_d,
            reference_z=ref_z,
            tool_point_angle_deg=pt_angle,
            tool_tip_geometry=tip_geom,
            tool_tip_length=tip_len,
            breakthrough_margin=bt,
            programmed_tip_depth=prog_z,
            achieved_cylindrical_depth=achieved_cyl,
            total_hole_depth=total_penetration,
            remaining_web_thickness=remaining_web,
            derivation_reason=reason,
        )

    def compute_programmed_depth(self, start_z: float, cylindrical_depth: float,
                                 depth_reference: str = "shoulder",
                                 breakthrough: float = 0.0) -> float:
        """Dynamically compute the required programmed cycle Z depth from actual tool-tip geometry."""
        sem = self.compute_drill_depth_semantics(
            start_z=start_z,
            target_depth=cylindrical_depth,
            depth_type=depth_reference,
            breakthrough=breakthrough,
        )
        return float(sem.programmed_tip_depth)

    def is_capable_of(self, purpose: str) -> bool:
        """Data-driven check if this tool can perform the specified machining operation purpose."""
        p = purpose.lower()
        if p in [op.lower() for op in self.operations]:
            return True
        if p in ("spot_drilling", "spot_drill"):
            if self.type in (ToolType.SPOT_DRILL, ToolType.CENTER_DRILL):
                return True
            if self.can_spot_drill:
                return True
            if self.type is ToolType.CHAMFER_MILL and self.can_plunge:
                return True
            if "spot" in self.id.lower() or "center" in self.id.lower():
                return True
            return False
        if p in ("drilling", "drill"):
            return self.type in (ToolType.DRILL, ToolType.SPOT_DRILL, ToolType.CENTER_DRILL)
        if p in ("facing", "face"):
            return self.type in (ToolType.FACE_MILL, ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL)
        if p in ("roughing", "rough"):
            return self.type in (ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL, ToolType.FACE_MILL)
        if p in ("finishing", "finish"):
            return self.type in (ToolType.FLAT_ENDMILL, ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL, ToolType.CHAMFER_MILL, ToolType.GROOVE_CUTTER)
        if p in ("chamfering", "chamfer"):
            return self.type in (ToolType.CHAMFER_MILL, ToolType.COUNTERSINK_TOOL, ToolType.SPOT_DRILL)
        if p in ("countersinking", "countersink"):
            return self.type in (ToolType.COUNTERSINK_TOOL, ToolType.CHAMFER_MILL, ToolType.SPOT_DRILL)
        if p in ("boring", "bore"):
            return self.type is ToolType.BORING_BAR
        if p in ("reaming", "ream"):
            return self.type is ToolType.REAMER
        if p in ("tapping", "tap"):
            return self.type is ToolType.TAP
        if p in ("thread_milling", "thread_mill"):
            return self.type is ToolType.THREAD_MILL
        if p in ("grooving", "groove"):
            return self.type in (ToolType.GROOVE_CUTTER, ToolType.FLAT_ENDMILL)
        return False

    def validate(self) -> None:
        problems = []
        if self.diameter <= 0:
            problems.append("diameter must be positive")
        if self.flute_length <= 0:
            problems.append("flute_length must be positive")
        if self.overall_length < self.flute_length:
            problems.append("overall_length < flute_length")
        if self.length_offset_h is not None and self.length_offset_h <= 0:
            problems.append("length_offset_h must be a positive integer")
        if self.diameter_offset_d is not None and self.diameter_offset_d <= 0:
            problems.append("diameter_offset_d must be a positive integer")
        if self.type is ToolType.DRILL and self.tip_radius != 0.0:
            problems.append("drill tip_radius must be 0 (point tip handled by point angle)")
        if self.type is ToolType.BALL_ENDMILL and abs(self.tip_radius - self.diameter / 2) > 1e-9:
            problems.append("ball_endmill tip_radius must equal diameter/2")
        if problems:
            raise CamError(code="INVALID_TOOL", message="; ".join(problems),
                           stage="context", tool_id=self.id)


# -------------------------------------------------------------- materials ---
@dataclass
class Material:
    """Workpiece material with cutting-data basis. Values are domain data, not
    model-specific constants: they come from a material database entry."""

    id: str
    name: str
    hardness_hb: float
    specific_cutting_force_kn_mm2: float   # Kc for feed/power calculations
    v_carbide: float                       # m/min carbide cutting speed basis
    v_hss: float                           # m/min HSS basis
    max_hm_ratio: float = 0.5              # max depth-of-cut / tool-diameter ratio (roughing)
    max_ae_ratio: float = 0.7              # max radial engagement ratio
    needs_coolant: bool = True
    chip_breaking_tendency: str = "medium" # "short_chipping", "medium", "stringy", "work_hardening"


DEFAULT_MATERIALS: dict[str, Material] = {
    "aluminum_6061": Material(
        id="aluminum_6061", name="Aluminum 6061-T6", hardness_hb=95,
        specific_cutting_force_kn_mm2=0.70, v_carbide=450.0, v_hss=120.0,
        max_hm_ratio=1.0, max_ae_ratio=0.75, needs_coolant=True),
    "aluminum_7075": Material(
        id="aluminum_7075", name="Aluminum 7075-T6 (High Strength Aerospace)", hardness_hb=150,
        specific_cutting_force_kn_mm2=0.85, v_carbide=400.0, v_hss=100.0,
        max_hm_ratio=0.9, max_ae_ratio=0.70, needs_coolant=True),
    "becu": Material(
        id="becu", name="Beryllium Copper (C17200 / C17300)", hardness_hb=220,
        specific_cutting_force_kn_mm2=1.55, v_carbide=120.0, v_hss=25.0,
        max_hm_ratio=0.60, max_ae_ratio=0.65, needs_coolant=True),
    "copper_110": Material(
        id="copper_110", name="Copper 110 Pure Electrolytic (C11000)", hardness_hb=85,
        specific_cutting_force_kn_mm2=0.95, v_carbide=220.0, v_hss=60.0,
        max_hm_ratio=0.80, max_ae_ratio=0.75, needs_coolant=True),
    "brass_360": Material(
        id="brass_360", name="Brass 360 Free Machining (C36000)", hardness_hb=80,
        specific_cutting_force_kn_mm2=0.65, v_carbide=320.0, v_hss=90.0,
        max_hm_ratio=1.0, max_ae_ratio=0.75, needs_coolant=False),
    "bronze_660": Material(
        id="bronze_660", name="Bronze 660 Bearing Bronze (SAE 660 / C93200)", hardness_hb=65,
        specific_cutting_force_kn_mm2=0.80, v_carbide=240.0, v_hss=70.0,
        max_hm_ratio=0.90, max_ae_ratio=0.75, needs_coolant=False),
    "steel_1018": Material(
        id="steel_1018", name="Steel 1018 Low Carbon", hardness_hb=126,
        specific_cutting_force_kn_mm2=1.50, v_carbide=180.0, v_hss=35.0,
        max_hm_ratio=0.75, max_ae_ratio=0.70, needs_coolant=True),
    "steel_4140": Material(
        id="steel_4140", name="Steel 4140 Pre-Hard", hardness_hb=280,
        specific_cutting_force_kn_mm2=2.10, v_carbide=110.0, v_hss=20.0,
        max_hm_ratio=0.50, max_ae_ratio=0.60, needs_coolant=True),
    "tool_steel_d2": Material(
        id="tool_steel_d2", name="Tool Steel D2 (High Carbon / Chromium)", hardness_hb=230,
        specific_cutting_force_kn_mm2=2.50, v_carbide=80.0, v_hss=15.0,
        max_hm_ratio=0.45, max_ae_ratio=0.55, needs_coolant=True),
    "tool_steel_o1": Material(
        id="tool_steel_o1", name="Tool Steel O1 Oil-Hardening", hardness_hb=210,
        specific_cutting_force_kn_mm2=2.20, v_carbide=90.0, v_hss=18.0,
        max_hm_ratio=0.50, max_ae_ratio=0.60, needs_coolant=True),
    "stainless_304": Material(
        id="stainless_304", name="Stainless 304 Austenitic", hardness_hb=170,
        specific_cutting_force_kn_mm2=2.00, v_carbide=95.0, v_hss=18.0,
        max_hm_ratio=0.50, max_ae_ratio=0.55, needs_coolant=True),
    "stainless_316": Material(
        id="stainless_316", name="Stainless 316 Marine Grade", hardness_hb=180,
        specific_cutting_force_kn_mm2=2.10, v_carbide=90.0, v_hss=16.0,
        max_hm_ratio=0.50, max_ae_ratio=0.55, needs_coolant=True),
    "stainless_17_4ph": Material(
        id="stainless_17_4ph", name="Stainless 17-4 PH (Condition H1025)", hardness_hb=330,
        specific_cutting_force_kn_mm2=2.40, v_carbide=85.0, v_hss=15.0,
        max_hm_ratio=0.45, max_ae_ratio=0.50, needs_coolant=True),
    "titanium_gr5": Material(
        id="titanium_gr5", name="Titanium Ti-6Al-4V (Grade 5)", hardness_hb=330,
        specific_cutting_force_kn_mm2=2.40, v_carbide=60.0, v_hss=15.0,
        max_hm_ratio=0.40, max_ae_ratio=0.50, needs_coolant=True),
    "inconel_718": Material(
        id="inconel_718", name="Inconel 718 Nickel Superalloy", hardness_hb=360,
        specific_cutting_force_kn_mm2=3.10, v_carbide=35.0, v_hss=8.0,
        max_hm_ratio=0.35, max_ae_ratio=0.45, needs_coolant=True),
    "cast_iron_gray": Material(
        id="cast_iron_gray", name="Cast Iron Class 30 (Gray Iron)", hardness_hb=200,
        specific_cutting_force_kn_mm2=1.30, v_carbide=180.0, v_hss=35.0,
        max_hm_ratio=0.70, max_ae_ratio=0.65, needs_coolant=False),
    "delrin_acetal": Material(
        id="delrin_acetal", name="Delrin / Acetal POM-C", hardness_hb=20,
        specific_cutting_force_kn_mm2=0.35, v_carbide=500.0, v_hss=160.0,
        max_hm_ratio=1.20, max_ae_ratio=0.80, needs_coolant=False),
    "peek": Material(
        id="peek", name="PEEK High Performance Polymer", hardness_hb=30,
        specific_cutting_force_kn_mm2=0.45, v_carbide=300.0, v_hss=90.0,
        max_hm_ratio=1.00, max_ae_ratio=0.75, needs_coolant=False),
}

# Drawing/Spec Alias Lookup Table
MATERIAL_ALIASES: dict[str, str] = {
    # Beryllium Copper
    "becu": "becu",
    "be_cu": "becu",
    "be-cu": "becu",
    "beryllium_copper": "becu",
    "beryllium copper": "becu",
    "beryllium-copper": "becu",
    "c17200": "becu",
    "c17300": "becu",
    "c172": "becu",
    "c173": "becu",
    "alloy 25": "becu",
    "alloy_25": "becu",
    "cube": "becu",
    "cu_be": "becu",
    "becu_c17200": "becu",

    # Aluminum 6061
    "6061": "aluminum_6061",
    "6061-t6": "aluminum_6061",
    "6061_t6": "aluminum_6061",
    "al 6061": "aluminum_6061",
    "al_6061": "aluminum_6061",
    "al-6061": "aluminum_6061",
    "aluminum": "aluminum_6061",
    "aluminium": "aluminum_6061",
    "aluminum 6061": "aluminum_6061",
    "aluminum_6061": "aluminum_6061",
    "aluminum_6061_t6": "aluminum_6061",

    # Aluminum 7075
    "7075": "aluminum_7075",
    "7075-t6": "aluminum_7075",
    "7075_t6": "aluminum_7075",
    "al 7075": "aluminum_7075",
    "al_7075": "aluminum_7075",
    "aluminum 7075": "aluminum_7075",
    "aluminum_7075": "aluminum_7075",

    # Copper 110
    "copper": "copper_110",
    "copper_110": "copper_110",
    "c11000": "copper_110",
    "c110": "copper_110",
    "pure_copper": "copper_110",

    # Brass 360
    "brass": "brass_360",
    "brass_360": "brass_360",
    "c36000": "brass_360",
    "c360": "brass_360",
    "free_cutting_brass": "brass_360",

    # Bronze 660
    "bronze": "bronze_660",
    "bronze_660": "bronze_660",
    "sae_660": "bronze_660",
    "c93200": "bronze_660",

    # Steel 1018
    "steel": "steel_1018",
    "mild_steel": "steel_1018",
    "mild steel": "steel_1018",
    "1018": "steel_1018",
    "steel_1018": "steel_1018",
    "crs": "steel_1018",
    "crs_1018": "steel_1018",

    # Steel 4140
    "4140": "steel_4140",
    "steel_4140": "steel_4140",
    "chromoly": "steel_4140",
    "4140_ht": "steel_4140",
    "4140_prehard": "steel_4140",

    # Tool Steels
    "d2": "tool_steel_d2",
    "tool_steel_d2": "tool_steel_d2",
    "o1": "tool_steel_o1",
    "tool_steel_o1": "tool_steel_o1",

    # Stainless Steels
    "304": "stainless_304",
    "ss304": "stainless_304",
    "ss_304": "stainless_304",
    "stainless_304": "stainless_304",
    "stainless steel 304": "stainless_304",
    "316": "stainless_316",
    "ss316": "stainless_316",
    "ss_316": "stainless_316",
    "stainless_316": "stainless_316",
    "17-4": "stainless_17_4ph",
    "17-4ph": "stainless_17_4ph",
    "17_4ph": "stainless_17_4ph",
    "17_4_ph": "stainless_17_4ph",
    "stainless_17_4ph": "stainless_17_4ph",

    # Titanium
    "titanium": "titanium_gr5",
    "ti": "titanium_gr5",
    "ti_6al_4v": "titanium_gr5",
    "ti6al4v": "titanium_gr5",
    "ti-6al-4v": "titanium_gr5",
    "titanium_gr5": "titanium_gr5",
    "grade_5_titanium": "titanium_gr5",

    # Inconel
    "inconel": "inconel_718",
    "inconel_718": "inconel_718",
    "inconel 718": "inconel_718",
    "inco_718": "inconel_718",

    # Cast Iron
    "cast_iron": "cast_iron_gray",
    "gray_iron": "cast_iron_gray",
    "cast_iron_gray": "cast_iron_gray",

    # Plastics
    "delrin": "delrin_acetal",
    "acetal": "delrin_acetal",
    "pom": "delrin_acetal",
    "pom_c": "delrin_acetal",
    "delrin_acetal": "delrin_acetal",
    "acetal_delrin": "delrin_acetal",
    "peek": "peek",
}


def get_material(identifier: str) -> Material:
    """Resolve a material instance from an ID, name, or drawing spec alias.

    Raises:
        CamError: If material is empty, unresolved, or unknown.
    """
    if not identifier or not str(identifier).strip():
        raise CamError(
            code="INVALID_MATERIAL",
            message="Workpiece material identifier is required and cannot be empty.",
            stage="context",
        )

    raw = str(identifier).strip()
    # 1. Exact match in DEFAULT_MATERIALS
    if raw in DEFAULT_MATERIALS:
        return DEFAULT_MATERIALS[raw]

    # 2. Normalized lowercase lookup in DEFAULT_MATERIALS
    cleaned = raw.lower().replace("-", "_").replace(" ", "_")
    if cleaned in DEFAULT_MATERIALS:
        return DEFAULT_MATERIALS[cleaned]

    # 3. Alias lookup
    lookup_key = raw.lower()
    if lookup_key in MATERIAL_ALIASES:
        target_id = MATERIAL_ALIASES[lookup_key]
        if target_id in DEFAULT_MATERIALS:
            return DEFAULT_MATERIALS[target_id]

    if cleaned in MATERIAL_ALIASES:
        target_id = MATERIAL_ALIASES[cleaned]
        if target_id in DEFAULT_MATERIALS:
            return DEFAULT_MATERIALS[target_id]

    valid_keys = list(DEFAULT_MATERIALS.keys())
    raise CamError(
        code="INVALID_MATERIAL",
        message=f"Unknown or unresolved material '{identifier}'. Supported materials include: {valid_keys}",
        stage="context",
    )


def list_materials() -> list[Material]:
    """Return all configured materials in the material database."""
    return list(DEFAULT_MATERIALS.values())


# ------------------------------------------------------------------ stock ---
class StockKind(str, Enum):
    BOX = "box"
    MESH = "mesh"       # arbitrary solid from tessellation
    CYLINDER = "cylinder"  # cylindrical stock for turning or round parts


@dataclass
class Stock:
    """Real setup-defined stock. No fallback dimensions exist anywhere."""

    kind: StockKind
    bounds_min: np.ndarray          # setup space
    bounds_max: np.ndarray
    mesh: Optional[TriangleMesh] = None   # for MESH kind
    cylinder_axis: Optional[np.ndarray] = None  # for CYLINDER kind: axis direction (0,0,1)
    cylinder_center: Optional[np.ndarray] = None  # for CYLINDER kind: center point (x,y)
    cylinder_radius: Optional[float] = None  # for CYLINDER kind: radius in mm

    def validate(self) -> None:
        if np.any(self.bounds_max - self.bounds_min <= 1e-9):
            raise CamError(MISSING_STOCK_GEOMETRY,
                           "stock has degenerate bounds", stage="context")
        if self.kind is StockKind.MESH and self.mesh is None:
            raise CamError(MISSING_STOCK_GEOMETRY,
                           "mesh stock has no mesh", stage="context")
        if self.kind is StockKind.CYLINDER:
            if self.cylinder_radius is None or self.cylinder_radius <= 0:
                raise CamError(MISSING_STOCK_GEOMETRY,
                               "cylinder stock requires positive radius", stage="context")

    def top_z(self) -> float:
        return float(self.bounds_max[2])

    @property
    def radius(self) -> Optional[float]:
        if self.cylinder_radius is not None and self.cylinder_radius > 0:
            return float(self.cylinder_radius)
        if self.bounds_min is not None and self.bounds_max is not None:
            return float(0.5 * min(self.bounds_max[0] - self.bounds_min[0], self.bounds_max[1] - self.bounds_min[1]))
        return None

    @property
    def center_xy(self) -> Optional[np.ndarray]:
        if self.cylinder_center is not None:
            return self.cylinder_center
        if self.bounds_min is not None and self.bounds_max is not None:
            return 0.5 * (self.bounds_min[:2] + self.bounds_max[:2])
        return None

    def dimension_description(self) -> str:
        """Format stock dimensions for reports and setup sheets."""
        if self.kind is StockKind.CYLINDER and self.cylinder_radius is not None:
            dia = self.cylinder_radius * 2.0
            height = float(self.bounds_max[2] - self.bounds_min[2])
            return f"Ø{dia:.1f} x {height:.1f} mm (Cylindrical Bar)"
        dims = self.bounds_max - self.bounds_min
        return f"{dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm (Rectangular Block)"

    @classmethod
    def from_cylinder(cls, center_xy: np.ndarray, z_min: float, z_max: float,
                      radius: float, axis: Optional[np.ndarray] = None) -> "Stock":
        """Create cylindrical stock from center, height, and radius."""
        c = np.asarray(center_xy, dtype=float)
        bmin = np.array([c[0] - radius, c[1] - radius, z_min])
        bmax = np.array([c[0] + radius, c[1] + radius, z_max])
        return cls(
            kind=StockKind.CYLINDER,
            bounds_min=bmin,
            bounds_max=bmax,
            cylinder_axis=axis if axis is not None else np.array([0.0, 0.0, 1.0]),
            cylinder_center=c,
            cylinder_radius=radius,
        )


# ---------------------------------------------------------------- fixtures ---
@dataclass
class Fixture:
    """Clamping or holding fixture geometry (vise jaws, toe clamps, fixture plates)."""
    id: str
    name: str
    bounds_min: np.ndarray          # setup space [xmin, ymin, zmin]
    bounds_max: np.ndarray          # setup space [xmax, ymax, zmax]
    kind: str = "vise_jaw"          # "vise_jaw", "toe_clamp", "fixture_plate", "custom"
    mesh: Optional[TriangleMesh] = None


# ----------------------------------------------------------------- setups ---
class WorkOffset(str, Enum):
    G54 = "G54"
    G55 = "G55"
    G56 = "G56"
    G57 = "G57"
    G58 = "G58"
    G59 = "G59"


@dataclass
class SetupClearanceInfo:
    """Rigorous, auditable provenance of a setup's safe clearance plane.
    
    Derives clearance from stock, fixtures, tool assembly, model orientation, and machine limits.
    """
    setup_id: str
    work_offset: str
    z_stock_max: float
    z_model_max: float
    z_fixture_max: float
    highest_obstacle_z: float
    safe_retract_margin: float
    tool_clearance_margin: float
    clearance_z: float
    approach_margin: float = 1.0
    approach_z: float = 0.0
    retract_margin: float = 2.0
    retract_z: float = 0.0
    machine_travel_max_z: Optional[float] = None
    machine_limit_ok: bool = True
    derivation_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "setup_id": self.setup_id,
            "work_offset": self.work_offset,
            "z_stock_max": self.z_stock_max,
            "z_model_max": self.z_model_max,
            "z_fixture_max": self.z_fixture_max,
            "highest_obstacle_z": self.highest_obstacle_z,
            "safe_retract_margin": self.safe_retract_margin,
            "tool_clearance_margin": self.tool_clearance_margin,
            "clearance_z": self.clearance_z,
            "approach_margin": self.approach_margin,
            "approach_z": self.approach_z,
            "retract_margin": self.retract_margin,
            "retract_z": self.retract_z,
            "machine_travel_max_z": self.machine_travel_max_z,
            "machine_limit_ok": self.machine_limit_ok,
            "derivation_notes": self.derivation_notes,
        }


def derive_setup_clearance_plane(
    setup: Setup,
    mesh: Optional[TriangleMesh] = None,
    machine: Optional[MachineConfig] = None,
    tools: Optional[list[Tool]] = None,
    model_shape: Optional[Shape] = None,
    extra_tool_margin: float = 0.0,
) -> SetupClearanceInfo:
    """Derive setup-specific safe clearance plane from stock, model, fixtures, tool assembly, and machine limits."""
    # 1. Stock top in setup coordinates
    z_stock_max = float('-inf')
    if getattr(setup, "stock", None) is not None:
        z_stock_max = float(setup.stock.bounds_max[2])

    # 2. Model top in setup coordinates
    z_model_max = float('-inf')
    if mesh is not None and getattr(mesh, "triangles", None) is not None and len(mesh.triangles):
        z_model_max = float(mesh.triangles[:, :, 2].max())
    elif model_shape is not None:
        try:
            _, bmax = model_shape.bounding_box()
            z_model_max = float(bmax[2])
        except Exception:
            pass

    # 3. Fixtures top in setup coordinates
    z_fixture_max = float('-inf')
    fixtures = getattr(setup, "fixtures", None) or []
    for f in fixtures:
        f_max = getattr(f, "bounds_max", None)
        if f_max is not None:
            z_fixture_max = max(z_fixture_max, float(f_max[2]))
        elif getattr(f, "triangles", None) is not None and len(f.triangles):
            z_fixture_max = max(z_fixture_max, float(f.triangles[:, :, 2].max()))

    # 4. Highest obstacle in setup environment
    valid_obstacles = [z for z in (z_stock_max, z_model_max, z_fixture_max) if z > float('-inf')]
    highest_obstacle_z = max(valid_obstacles) if valid_obstacles else 0.0

    # 5. Safe retract margin
    safe_retract = 10.0
    if machine is not None and getattr(machine, "safe_retract_height", None) is not None:
        safe_retract = float(machine.safe_retract_height)

    # 6. Tool assembly margin
    tool_margin = float(extra_tool_margin)
    if tools:
        max_cr = max((getattr(t, "corner_radius", 0.0) or 0.0 for t in tools), default=0.0)
        tool_margin = max(tool_margin, max_cr)

    clearance_z = highest_obstacle_z + safe_retract + tool_margin

    # 7. Setup-level approach & retract planes
    approach_margin = float(getattr(setup, "approach_margin", 1.0) or 1.0)
    retract_margin = float(getattr(setup, "retract_margin", 2.0) or 2.0)
    z_stock_top = z_stock_max if z_stock_max > float('-inf') else highest_obstacle_z
    approach_z = z_stock_top + approach_margin
    retract_z = highest_obstacle_z + retract_margin

    # 8. Machine travel limits check
    z_limit_setup = None
    machine_ok = True
    if machine is not None and hasattr(machine, "axes") and "Z" in machine.axes:
        z_axis = machine.axes["Z"]
        wo_str = setup.work_offset.value if hasattr(setup.work_offset, "value") else str(setup.work_offset)
        wo_vec = machine.work_offsets.get(wo_str, np.zeros(3)) if machine.work_offsets else np.zeros(3)
        z_limit_setup = float(z_axis.travel_max - wo_vec[2])
        if clearance_z > z_limit_setup:
            machine_ok = False

    # 9. Human-readable derivation notes
    stock_desc = f"stock={z_stock_max:.2f}" if z_stock_max > float('-inf') else "stock=none"
    model_desc = f"model={z_model_max:.2f}" if z_model_max > float('-inf') else "model=none"
    fix_desc = f", fixtures={z_fixture_max:.2f}" if z_fixture_max > float('-inf') else ""
    tool_desc = f", tool_margin={tool_margin:.2f}" if tool_margin > 0 else ""
    derivation_notes = f"max({stock_desc}, {model_desc}{fix_desc}) + retract={safe_retract:.2f}{tool_desc} -> Z{clearance_z:.2f}"

    wo_name = setup.work_offset.value if hasattr(setup.work_offset, "value") else str(setup.work_offset)
    return SetupClearanceInfo(
        setup_id=setup.id,
        work_offset=wo_name,
        z_stock_max=z_stock_max,
        z_model_max=z_model_max,
        z_fixture_max=z_fixture_max,
        highest_obstacle_z=highest_obstacle_z,
        safe_retract_margin=safe_retract,
        tool_clearance_margin=tool_margin,
        clearance_z=clearance_z,
        approach_margin=approach_margin,
        approach_z=approach_z,
        retract_margin=retract_margin,
        retract_z=retract_z,
        machine_travel_max_z=z_limit_setup,
        machine_limit_ok=machine_ok,
        derivation_notes=derivation_notes,
    )


@dataclass
class Setup:
    """One machining setup: orientation of model+stock in setup coordinates,
    work offset selection, clamping constraints, and canonical transformation pipeline.

    Coordinate Transformation Architecture:
        Canonical CAD coordinates (Model Space)
                   ↓
            Setup transform (model_to_setup)
                   ↓
        Machine/setup coordinates (Setup / Work Space)

    Every setup operates on the single authoritative canonical part coordinate system.
    Geometry is never independently regenerated from scratch per setup, ensuring the
    same physical surface maintains exact CAD identity and tolerances across G54/G55/etc.
    """

    id: str = ""
    name: str = ""
    model_to_setup: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    work_offset: WorkOffset = WorkOffset.G54
    stock: Optional[Stock] = None
    fixtures: list[Fixture | TriangleMesh] = field(default_factory=list)
    fixture_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    comment: str = ""
    approach_margin: float = 1.0        # Safe approach standoff distance above material top (mm)
    retract_margin: float = 2.0         # Safe local retract standoff distance above cut (mm)
    features: list[Any] = field(default_factory=list)

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "",
        model_to_setup: Optional[np.ndarray] = None,
        work_offset: Optional[WorkOffset | str] = None,
        stock: Optional[Stock] = None,
        fixtures: Optional[list[Fixture | TriangleMesh]] = None,
        comment: str = "",
        approach_margin: float = 1.0,
        retract_margin: float = 2.0,
        fixture_transform: Optional[np.ndarray] = None,
        features: Optional[list[Any]] = None,
        *,
        setup_id: Optional[str] = None,
        WCS: Optional[WorkOffset | str] = None,
        wcs: Optional[WorkOffset | str] = None,
        origin: Optional[np.ndarray | list[float]] = None,
        rotation: Optional[np.ndarray | list[list[float]]] = None,
        translation: Optional[np.ndarray | list[float]] = None,
        orientation: Optional[np.ndarray | list[float]] = None,
        model_to_setup_transform: Optional[np.ndarray] = None,
        setup_transform: Optional[np.ndarray] = None,
    ):
        actual_id = setup_id or id or "setup_001"
        self.id = str(actual_id)
        self.name = str(name or actual_id)

        raw_wcs = WCS or wcs or work_offset or WorkOffset.G54
        if isinstance(raw_wcs, str):
            try:
                self.work_offset = WorkOffset(raw_wcs.upper())
            except ValueError:
                self.work_offset = WorkOffset.G54
        elif isinstance(raw_wcs, WorkOffset):
            self.work_offset = raw_wcs
        else:
            self.work_offset = WorkOffset.G54

        M = model_to_setup_transform if model_to_setup_transform is not None else (
            setup_transform if setup_transform is not None else model_to_setup
        )
        if M is not None:
            self.model_to_setup = np.asarray(M, dtype=float)
        elif rotation is not None or translation is not None or origin is not None:
            t_vec = translation if translation is not None else origin
            self.model_to_setup = make_transform(rotation=rotation, translation=t_vec)
        else:
            self.model_to_setup = np.eye(4, dtype=float)

        self.stock = stock
        self.fixtures = fixtures if fixtures is not None else []
        self.fixture_transform = (
            np.asarray(fixture_transform, dtype=float)
            if fixture_transform is not None
            else np.eye(4, dtype=float)
        )
        self.comment = str(comment or "")
        self.approach_margin = float(approach_margin)
        self.retract_margin = float(retract_margin)
        self.features = features if features is not None else []

    @property
    def setup_id(self) -> str:
        """Alias for setup id."""
        return self.id

    @setup_id.setter
    def setup_id(self, val: str) -> None:
        self.id = str(val)

    @property
    def WCS(self) -> str:
        """Work Coordinate System identifier string (e.g. 'G54', 'G55')."""
        return self.work_offset.value if hasattr(self.work_offset, "value") else str(self.work_offset)

    @property
    def wcs(self) -> str:
        """Work Coordinate System identifier string (e.g. 'G54', 'G55')."""
        return self.WCS

    @property
    def model_to_setup_transform(self) -> np.ndarray:
        """4x4 rigid transformation matrix from Canonical CAD space to Setup space."""
        return self.model_to_setup

    @property
    def setup_transform(self) -> np.ndarray:
        """4x4 rigid transformation matrix from Canonical CAD space to Setup space."""
        return self.model_to_setup

    @property
    def setup_to_model_transform(self) -> np.ndarray:
        """4x4 rigid transformation matrix from Setup space to Canonical CAD space."""
        return _invert(self.model_to_setup)

    @property
    def setup_to_canonical_transform(self) -> np.ndarray:
        """4x4 rigid transformation matrix from Setup space to Canonical CAD space."""
        return _invert(self.model_to_setup)

    @property
    def rotation(self) -> np.ndarray:
        """3x3 orthonormal rotation matrix component of setup transform."""
        return self.model_to_setup[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        """3D translation vector [tx, ty, tz] component of setup transform."""
        return self.model_to_setup[:3, 3].copy()

    @property
    def origin(self) -> np.ndarray:
        """3D origin point of the setup coordinate system in canonical CAD space."""
        return self.model_to_setup[:3, 3].copy()

    @property
    def orientation(self) -> np.ndarray:
        """Tool approach unit orientation vector in setup coordinates."""
        # Setup-space tool axis is canonically [0, 0, 1] (Z-up)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    @property
    def canonical_orientation(self) -> np.ndarray:
        """Tool approach unit orientation vector mapped back to Canonical CAD space."""
        # R^T @ [0, 0, 1] gives the machining tool vector relative to canonical CAD body
        return self.model_to_setup[:3, :3].T @ np.array([0.0, 0.0, 1.0], dtype=float)

    @property
    def stock_top_z(self) -> float:
        if self.stock is not None and hasattr(self.stock, "bounds_max") and self.stock.bounds_max is not None:
            return float(self.stock.bounds_max[2])
        return 0.0

    @property
    def approach_z(self) -> float:
        """Setup-relative default approach plane (stock top + approach margin)."""
        return self.stock_top_z + self.approach_margin

    @property
    def retract_z(self) -> float:
        """Setup-relative default retract plane (stock top + retract margin)."""
        return self.stock_top_z + self.retract_margin

    def transform_point(self, pt: np.ndarray) -> np.ndarray:
        """Transform point(s) from Canonical CAD coordinates into Setup coordinates."""
        return transform_points(pt, self.model_to_setup)

    def transform_direction(self, vec: np.ndarray) -> np.ndarray:
        """Transform direction vector from Canonical CAD coordinates into Setup coordinates."""
        return transform_direction(vec, self.model_to_setup)

    def transform_bounding_box(self, bmin: np.ndarray, bmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform bounding box from Canonical CAD coordinates into Setup coordinates."""
        return transform_bounding_box(bmin, bmax, self.model_to_setup)

    def transform_feature(self, feat: Any, new_id: Optional[str] = None) -> Any:
        """Transform a canonical CAD MachiningFeature into this setup's coordinate system."""
        if hasattr(feat, "transformed"):
            return feat.transformed(self.model_to_setup, dst_space=CoordSpace.SETUP, new_id=new_id)
        return feat

    def inverse_transform_point(self, pt: np.ndarray) -> np.ndarray:
        """Transform point(s) from Setup coordinates back into Canonical CAD coordinates."""
        return transform_points(pt, self.setup_to_model_transform)

    def inverse_transform_direction(self, vec: np.ndarray) -> np.ndarray:
        """Transform direction vector from Setup coordinates back into Canonical CAD coordinates."""
        return transform_direction(vec, self.setup_to_model_transform)

    def inverse_transform_feature(self, feat: Any, new_id: Optional[str] = None) -> Any:
        """Transform a setup MachiningFeature back into Canonical CAD coordinates."""
        if hasattr(feat, "transformed"):
            return feat.transformed(self.setup_to_model_transform, dst_space=CoordSpace.MODEL, new_id=new_id)
        return feat

    def transform_shape(self, shape: Shape) -> Shape:
        """Transform canonical CAD Shape into Setup coordinates without regenerating B-Rep topology."""
        return shape.transformed(self.model_to_setup)

    def transform_mesh(self, mesh: TriangleMesh) -> TriangleMesh:
        """Transform TriangleMesh from canonical CAD coordinates into Setup coordinates."""
        return mesh.transformed(self.model_to_setup)

    def compute_clearance(
        self,
        mesh: Optional[TriangleMesh] = None,
        machine: Optional[MachineConfig] = None,
        tools: Optional[list[Tool]] = None,
        model_shape: Optional[Shape] = None,
        extra_tool_margin: float = 0.0,
    ) -> SetupClearanceInfo:
        """Compute the setup-specific safe clearance plane from all environment constraints."""
        return derive_setup_clearance_plane(
            self, mesh=mesh, machine=machine, tools=tools,
            model_shape=model_shape, extra_tool_margin=extra_tool_margin,
        )


# ------------------------------------------------------- Finished Geometry Hierarchy --

class TargetStrategy(str, Enum):
    NOMINAL = "nominal"
    TOLERANCE_MIDPOINT = "tolerance_midpoint"
    MAXIMUM_MATERIAL_CONDITION = "mmc"
    LEAST_MATERIAL_CONDITION = "lmc"
    WEAR_BIASED = "wear_biased"


class TargetGeometryType(str, Enum):
    SURFACE = "surface"
    BOUNDARY = "boundary"
    CURVE = "curve"
    DEPTH = "depth"


@dataclass
class TargetSurface:
    """Exact CAD target surface definition (Plane, Cylinder, Cone, Torus, Blend/Fillet, Freeform/NURBS)."""
    surface_kind: str = "plane"           # plane | cylinder | cone | torus | fillet | blend | freeform | other
    face_id: int | str = -1
    point_on: Optional[np.ndarray] = None # surface anchor point / center
    normal: Optional[np.ndarray] = None   # surface unit normal (for planar surfaces)
    axis: Optional[np.ndarray] = None     # axis of revolution (cylinders, cones, tori)
    radius: Optional[float] = None        # radius (cylinders, fillets, tori)
    semi_angle_rad: Optional[float] = None# cone half-angle
    bounds_min: Optional[np.ndarray] = None
    bounds_max: Optional[np.ndarray] = None
    description: str = ""

    def distance_to_surface(self, point: np.ndarray) -> float:
        """Calculate signed or unsigned orthogonal distance from point to target surface."""
        p = np.asarray(point, dtype=float)
        kind = str(self.surface_kind).lower()
        if "plane" in kind and self.normal is not None and self.point_on is not None:
            n = self.normal / np.linalg.norm(self.normal)
            return float(np.dot(p - self.point_on, n))
        elif "cylinder" in kind and self.axis is not None and self.point_on is not None and self.radius is not None:
            a = self.axis / np.linalg.norm(self.axis)
            v = p - self.point_on
            proj = np.dot(v, a) * a
            rad_vec = v - proj
            return float(np.linalg.norm(rad_vec) - self.radius)
        return 0.0


@dataclass
class TargetBoundary:
    """Exact CAD target 2D/3D closed boundary wire/loop (pockets, islands, bosses, profiles)."""
    boundary_type: str = "closed_loop"    # closed_loop | island | pocket_wall | boss_perimeter
    points: list[np.ndarray] = field(default_factory=list) # ordered vertices forming exact loop
    face_id: int | str = -1
    z_level: Optional[float] = None
    is_closed: bool = True
    description: str = ""


@dataclass
class TargetCurve:
    """Exact CAD target 3D curve/edge (chamfers, 3D contours, fillet guides, wire paths)."""
    curve_kind: str = "line"              # line | circle | ellipse | bspline | other
    edge_id: int | str = -1
    points: list[np.ndarray] = field(default_factory=list)
    start_point: Optional[np.ndarray] = None
    end_point: Optional[np.ndarray] = None
    radius: Optional[float] = None
    center: Optional[np.ndarray] = None
    description: str = ""


@dataclass
class TargetDepth:
    """Exact CAD target axial depth level (drilled hole bottoms, spot depths, boring floors)."""
    depth_reference: DepthReference = DepthReference.TOP_FACE
    nominal_depth: float = 0.0
    reference_z: float = 0.0
    target_z: float = 0.0
    depth_type: str = "cylindrical_depth" # cylindrical_depth | tip_depth | total_hole_depth | floor_level
    feature_id: str = ""
    description: str = ""


@dataclass
class NominalGeometry:
    """Unaltered CAD/B-Rep nominal geometry directly extracted from the CAD model."""
    dimension: float
    dimension_type: str = "diameter"    # diameter | depth | width | floor_z | top_z | radius | contour
    canonical_reference_id: str = ""    # Face/Edge/Feature ID
    description: str = ""

    def __repr__(self) -> str:
        return f"NominalGeometry({self.dimension_type}={self.dimension:.4f}mm, ref={self.canonical_reference_id})"


@dataclass
class ToleranceModel:
    """Explicit drawing or specification tolerance envelope applied to nominal geometry."""
    upper_tol_mm: float = 0.05
    lower_tol_mm: float = -0.05
    tolerance_class: Optional[str] = None   # e.g., "H7", "ISO 2768-m", "bilateral"
    surface_finish_ra_um: Optional[float] = None
    allowed_scallop_mm: Optional[float] = None

    @property
    def tolerance_band_mm(self) -> float:
        return float(self.upper_tol_mm - self.lower_tol_mm)

    def upper_limit(self, nominal: float) -> float:
        return float(nominal + self.upper_tol_mm)

    def lower_limit(self, nominal: float) -> float:
        return float(nominal + self.lower_tol_mm)

    def midpoint(self, nominal: float) -> float:
        return float(nominal + 0.5 * (self.upper_tol_mm + self.lower_tol_mm))

    def is_in_tolerance(self, nominal: float, actual: float, tol_epsilon: float = 1e-6) -> bool:
        return (self.lower_limit(nominal) - tol_epsilon) <= actual <= (self.upper_limit(nominal) + tol_epsilon)


@dataclass
class MachiningAllowance:
    """Deliberate stock allowance or process bias applied to toolpaths (separate from nominal CAD)."""
    radial_stock_mm: float = 0.0          # Wall finish stock allowance left for subsequent passes
    axial_stock_mm: float = 0.0           # Floor finish stock allowance
    wear_offset_bias_mm: float = 0.0      # Deflection/wear bias offset
    sparkout_passes: int = 0              # Zero-depth spring passes
    purpose: str = "finishing"            # roughing | semi_finishing | finishing | pre_grind


@dataclass
class FinalManufacturingTarget:
    """The exact target programmed for CAM toolpaths, derived from CAD nominal, tolerance strategy, and allowance.
    
    The final machining operation terminates on the exact target CAD surface/curve/boundary/depth
    appropriate to the feature's geometry type:
    - Plane     -> TargetSurface (planar surface)
    - Cylinder  -> TargetSurface (cylindrical surface)
    - Cone      -> TargetSurface (conical surface)
    - Torus     -> TargetSurface (toroidal surface)
    - Fillet    -> TargetSurface / TargetCurve (blend surface / edge)
    - Freeform  -> TargetSurface (NURBS / freeform surface)
    - Boundary  -> TargetBoundary (closed contour / loop)
    - Depth     -> TargetDepth (axial termination)
    """
    nominal: NominalGeometry
    tolerance: Optional[ToleranceModel] = None
    allowance: Optional[MachiningAllowance] = None
    strategy: TargetStrategy = TargetStrategy.NOMINAL
    target_dimension: float = 0.0
    derivation_notes: str = ""
    target_surface: Optional[TargetSurface] = None
    target_boundary: Optional[TargetBoundary] = None
    target_curve: Optional[TargetCurve] = None
    target_depth: Optional[TargetDepth] = None

    @classmethod
    def derive(
        cls,
        nominal: NominalGeometry | float,
        dimension_type: str = "diameter",
        tolerance: Optional[ToleranceModel] = None,
        allowance: Optional[MachiningAllowance] = None,
        strategy: TargetStrategy = TargetStrategy.NOMINAL,
        canonical_reference_id: str = "",
    ) -> FinalManufacturingTarget:
        """Deterministically derive the manufacturing target from CAD nominal, tolerance, and allowance."""
        if isinstance(nominal, (int, float)):
            nom_obj = NominalGeometry(
                dimension=float(nominal),
                dimension_type=dimension_type,
                canonical_reference_id=canonical_reference_id,
                description=f"Nominal {dimension_type} {nominal:.4f}mm",
            )
        else:
            nom_obj = nominal

        base_dim = nom_obj.dimension
        notes: list[str] = []

        # 1. Apply tolerance positioning strategy
        if tolerance is not None and strategy == TargetStrategy.TOLERANCE_MIDPOINT:
            target = tolerance.midpoint(base_dim)
            if abs(target - base_dim) > 1e-6:
                notes.append(f"Shifted to tolerance midpoint {target:.4f}mm [{tolerance.lower_tol_mm:+.3f}/{tolerance.upper_tol_mm:+.3f}]")
        elif tolerance is not None and strategy == TargetStrategy.MAXIMUM_MATERIAL_CONDITION:
            if nom_obj.dimension_type in ("hole_diameter", "internal_diameter", "pocket_width"):
                target = tolerance.lower_limit(base_dim)
            else:
                target = tolerance.upper_limit(base_dim)
            notes.append(f"Targeted MMC {target:.4f}mm")
        elif tolerance is not None and strategy == TargetStrategy.LEAST_MATERIAL_CONDITION:
            if nom_obj.dimension_type in ("hole_diameter", "internal_diameter", "pocket_width"):
                target = tolerance.upper_limit(base_dim)
            else:
                target = tolerance.lower_limit(base_dim)
            notes.append(f"Targeted LMC {target:.4f}mm")
        else:
            target = base_dim
            notes.append(f"Targeted nominal CAD dimension {base_dim:.4f}mm")

        # 2. Apply machining allowance
        if allowance is not None:
            if nom_obj.dimension_type in ("floor_z", "depth"):
                if allowance.axial_stock_mm > 0:
                    target += allowance.axial_stock_mm
                    notes.append(f"+{allowance.axial_stock_mm:.3f}mm axial allowance ({allowance.purpose})")
            else:
                if allowance.radial_stock_mm > 0:
                    target += allowance.radial_stock_mm
                    notes.append(f"+{allowance.radial_stock_mm:.3f}mm radial allowance ({allowance.purpose})")
            if abs(allowance.wear_offset_bias_mm) > 0:
                target += allowance.wear_offset_bias_mm
                notes.append(f"{allowance.wear_offset_bias_mm:+.4f}mm wear/deflection bias")

        return cls(
            nominal=nom_obj,
            tolerance=tolerance,
            allowance=allowance,
            strategy=strategy,
            target_dimension=round(target, 6),
            derivation_notes=" | ".join(notes),
        )


@dataclass
class SimulatedMachinedResult:
    """The measured/simulated outcome comparing simulated cut geometry vs nominal and target."""
    target: FinalManufacturingTarget
    measured_dimension: float
    deviation_from_target_mm: float
    deviation_from_nominal_mm: float
    in_tolerance: bool
    tolerance_consumed_pct: float
    inspection_notes: str = ""

    @classmethod
    def evaluate(
        cls,
        target: FinalManufacturingTarget,
        measured_dimension: float,
    ) -> SimulatedMachinedResult:
        meas = float(measured_dimension)
        nom = target.nominal.dimension
        tgt = target.target_dimension
        dev_tgt = meas - tgt
        dev_nom = meas - nom

        in_tol = True
        pct_consumed = 0.0
        notes: list[str] = []

        if target.tolerance is not None:
            tol = target.tolerance
            in_tol = tol.is_in_tolerance(nom, meas)
            band = tol.tolerance_band_mm
            if band > 1e-9:
                mid = tol.midpoint(nom)
                pct_consumed = min(100.0, max(0.0, (abs(meas - mid) / (band * 0.5)) * 100.0))
            if not in_tol:
                notes.append(f"OUT OF TOLERANCE: measured {meas:.4f}mm outside [{tol.lower_limit(nom):.4f}, {tol.upper_limit(nom):.4f}]")
            else:
                notes.append(f"IN TOLERANCE ({pct_consumed:.1f}% consumed)")
        else:
            notes.append(f"Deviation from nominal: {dev_nom:+.4f}mm")

        return cls(
            target=target,
            measured_dimension=round(meas, 6),
            deviation_from_target_mm=round(dev_tgt, 6),
            deviation_from_nominal_mm=round(dev_nom, 6),
            in_tolerance=in_tol,
            tolerance_consumed_pct=round(pct_consumed, 1),
            inspection_notes=" | ".join(notes),
        )


# ------------------------------------------------------- finish requirements --
@dataclass
class FinishRequirement:
    """Manufacturing requirement for a face or feature; user- or drawing-driven."""

    tolerance_mm: float = 0.1           # dimensional tolerance
    surface_finish_um: float = 6.3      # Ra
    allowed_scallop_mm: float = 0.05    # for 3D surfaces


# ----------------------------------------------------------------- context ---
@dataclass
class PlanningContext:
    """Complete immutable-ish planning state handed to every pipeline stage."""

    model: ImportedModel
    setups: list[Setup]
    machine: MachineConfig
    tools: list[Tool]
    material: Material
    voxel_resolution: float = 0.0      # 0 => auto from stock size
    requirements: dict[str, FinishRequirement] = field(default_factory=dict)  # face index -> req

    def validate(self) -> None:
        self.machine.validate()
        if not self.tools:
            raise CamError(code=MISSING_MACHINE_DATA,
                           message="tool library is empty", stage="context")
        for tool in self.tools:
            tool.validate()
        if not self.setups:
            raise CamError(INVALID_SETUP, "no machining setups defined", stage="context")
        for setup in self.setups:
            CoordinateSystemChain(
                model_to_setup=setup.model_to_setup,
                setup_to_work=np.eye(4),          # setup == work origin by definition here
                work_to_machine=self._work_offset_transform(setup),
            ).validate_chain()
            setup.stock.validate()

    def _work_offset_transform(self, setup: Setup) -> np.ndarray:
        """setup -> machine: derived from the configured work offset position."""
        off = self.machine.work_offsets.get(setup.work_offset.value)
        if off is None:
            raise CamError(
                code="INVALID_WORK_OFFSET",
                message=f"work offset {setup.work_offset.value} not configured on machine",
                stage="context",
            )
        # The work offset stores the machine-coordinates location of the setup
        # origin; the transform maps setup coords into machine coords.
        t = make_transform(translation=off)
        return t

    def tool_by_id(self, tool_id: str) -> Optional[Tool]:
        for t in self.tools:
            if t.id == tool_id:
                return t
        return None

    def model_shape_in_setup(self, setup: Setup) -> Shape:
        return self.model.shape.transformed(setup.model_to_setup)

    def chain(self, setup: Setup) -> CoordinateSystemChain:
        return CoordinateSystemChain(
            model_to_setup=setup.model_to_setup,
            setup_to_work=np.eye(4),
            work_to_machine=self._work_offset_transform(setup),
        )

    def compute_setup_clearance(
        self,
        setup: Setup,
        mesh: Optional[TriangleMesh] = None,
        extra_tool_margin: float = 0.0,
    ) -> SetupClearanceInfo:
        """Compute the setup-specific safe clearance plane using the full planning context."""
        shape = self.model_shape_in_setup(setup) if self.model and getattr(self.model, "shape", None) else None
        return derive_setup_clearance_plane(
            setup=setup,
            mesh=mesh,
            machine=self.machine,
            tools=self.tools,
            model_shape=shape,
            extra_tool_margin=extra_tool_margin,
        )
