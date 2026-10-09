"""Canonical machine program: the complete contract between CAM engine and post.

Architecture
------------

    strategies -> Toolpath[] (machining intent, geometry, provenance)
                       |
                       |  build_canonical_program()   <-- every decision made here:
                       |     tool / H / D registers, feeds, spindle, compensation,
                       |     coolant, WCS, clearance planes, canned-cycle parameters,
                       |     arc validity & linearization, teardown, program number
                       v
              CanonicalProgram { operations: [CanonicalToolpath] }
                       |
                       |  post.translate()            <-- syntax only:
                       |     tool change, G43, G41/G42, G0/G1/G2/G3, spindle,
                       |     feed, coolant, WCS, cycle syntax, machine commands
                       v
                    G-code

A post-processor never invents geometry, registers, feeds, planes or defaults:
every field of a canonical command is already resolved by the engine, so a bug in
post logic can only corrupt *spelling*, never the toolpath.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from ..toolpaths.arcs import evaluate_arc, fit_arcs, linearize_arc
from ..toolpaths.semantic import MotionSegment, MotionType, Toolpath


# --------------------------------------------------------------------- text --
def fmt_number(val: float, precision: int = 3) -> str:
    """Format a coordinate for a controller (trailing zeros stripped, no negative zero)."""
    if round(val, precision) == 0.0 or abs(val) < 1e-9:
        val = 0.0
    s = f"{val:.{precision}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def fmt_feed(val: float, precision: int = 1) -> str:
    s = f"{val:.{precision}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def format_tool_description(tool_id: str, tool_meta: Optional[dict] = None) -> str:
    """Human readable tool dimensions, built from tool data only."""
    if not tool_meta:
        return tool_id

    parts = []
    ttype = tool_meta.get("type")
    if ttype:
        type_name = str(ttype).replace("_", " ").title()
        parts.append(type_name)

    dia = tool_meta.get("diameter")
    if dia is not None and dia > 0:
        dia_val = f"{dia:.3f}".rstrip("0").rstrip(".") if (dia < 1.0 or round(dia, 2) != round(dia, 3)) else f"{dia:.2f}".rstrip("0").rstrip(".")
        parts.append(f"\u00d8{dia_val}mm")

    cr = tool_meta.get("corner_radius")
    if cr is not None and cr > 0:
        parts.append(f"CR: {cr:.2f}mm")

    fl = tool_meta.get("flute_length")
    if fl is not None and fl > 0:
        parts.append(f"FL: {fl:.1f}mm")

    oal = tool_meta.get("overall_length")
    if oal is not None and oal > 0:
        parts.append(f"TL: {oal:.1f}mm")

    flutes = tool_meta.get("flutes")
    if flutes is not None and flutes > 0:
        parts.append(f"{flutes}F")

    return " | ".join(parts) if parts else tool_id


# ---------------------------------------------------------------- commands --
@dataclass(frozen=True)
class Comment:
    """A comment line. The post owns the comment syntax, the engine the text."""
    text: str
    kind: str = "audit"      # "audit" (operation/cycle notes) | "motion" (arc remediation)



@dataclass(frozen=True)
class BlankLine:
    """Separator line."""


@dataclass(frozen=True)
class ToolChange:
    """Exchange tools. Registers, description and retract plane are resolved."""
    tool_id: str
    tool_number: int
    length_offset_h: int
    description: str
    spindle_rpm: Optional[float] = None
    retract_z: Optional[float] = None
    stage_next_tool_number: Optional[int] = None


@dataclass(frozen=True)
class SafetyBlock:
    """Per-operation modal safety block; the post spells its own codes."""
    work_offset: str
    tool_number: int = 1
    h_offset: int = 1


@dataclass(frozen=True)
class CompOn:
    side: str            # "left" -> G41, "right" -> G42
    d_register: int


@dataclass(frozen=True)
class CompOff:
    pass


@dataclass(frozen=True)
class Rapid:
    target: np.ndarray
    axes: Optional[tuple[str, ...]] = None   # None -> modal axis suppression, tuple -> only these axes



@dataclass(frozen=True)
class Linear:
    target: np.ndarray
    feed: float


@dataclass(frozen=True)
class Arc:
    """A validated arc: center already corrected, I/J already rounded."""
    start: np.ndarray
    target: np.ndarray
    center: np.ndarray
    ccw: bool
    feed: float
    i_offset: float
    j_offset: float


@dataclass(frozen=True)
class Spindle:
    rpm: float
    direction: str = "cw"
    source: str = "segment"    # "operation" (op-level speed line) | "segment"


@dataclass(frozen=True)
class SpindleStop:
    pass


@dataclass(frozen=True)
class Coolant:
    kind: str = "flood"    # flood | mist | through_spindle


@dataclass(frozen=True)
class CoolantOff:
    pass


@dataclass(frozen=True)
class Dwell:
    seconds: float


@dataclass(frozen=True)
class Cycle:
    """A fully resolved canned cycle: no post-side defaults remain."""
    cycle_type: str
    x: float
    y: float
    depth_z: float
    r_plane: float
    feed: float
    peck: float = 0.0
    dwell: float = 0.0
    shift: float = 0.0
    retract_mode: str = "G98"
    initial_plane: Optional[float] = None
    surface_z: float = 0.0
    clearance_plane: float = 0.0
    safe_distance: float = 0.0
    spindle_rpm: Optional[float] = None
    comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class CycleCancel:
    pass


@dataclass(frozen=True)
class MachineCommand:
    """A controller-specific command selected by the engine (HSM mode, staging...)."""
    name: str
    params: tuple[tuple[str, Any], ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.params:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class SafeReturn:
    axis: str          # "z" | "xy"
    value: float


@dataclass(frozen=True)
class ProgramEnd:
    pass


CanonicalCommand = Union[
    Comment, BlankLine, ToolChange, SafetyBlock, CompOn, CompOff, Rapid, Linear, Arc,
    Spindle, SpindleStop, Coolant, CoolantOff, Dwell, Cycle, CycleCancel,
    MachineCommand, SafeReturn, ProgramEnd,
]


# ------------------------------------------------------------------ program --
@dataclass
class CanonicalToolpath:
    """One operation, fully resolved: what to do, in order, with no gaps."""
    operation_id: str
    feature_id: Optional[str]
    purpose: str
    tool_id: str
    tool_number: int
    work_offset: str
    commands: tuple[CanonicalCommand, ...] = ()


@dataclass
class ProgramStats:
    rapid_dist: float = 0.0
    cut_dist: float = 0.0
    rapid_time: float = 0.0
    cut_time: float = 0.0
    tool_changes: list[str] = field(default_factory=list)


@dataclass
class CanonicalProgram:
    """A complete, resolved program: the only input a post-processor gets."""
    program_name: str
    program_number: int
    work_offset: str
    tools_info: dict[str, dict] = field(default_factory=dict)
    operations: tuple[CanonicalToolpath, ...] = ()
    footer_commands: tuple[CanonicalCommand, ...] = ()
    stats: ProgramStats = field(default_factory=ProgramStats)


# ------------------------------------------------------------------ options --
@dataclass
class BuildOptions:
    """Every decision the engine makes before a post is allowed to see data."""

    # program identity
    program_name: str = "VEXCAM"
    program_number: Optional[Union[int, str]] = None
    base_program_number: Optional[int] = None
    used_program_numbers: Optional[set[int]] = None
    setup_id: Optional[str] = None
    setup_index: Optional[int] = None
    job_id: Optional[str] = None
    work_offset: str = "G54"

    # numeric formatting (rounding contract shared with the controller)
    precision: int = 3
    feed_precision: int = 1

    # tool registers
    renumber_tools: bool = True
    tool_map: dict[str, int] = field(default_factory=dict)
    h_offset_map: dict[str, int] = field(default_factory=dict)
    d_offset_map: dict[str, int] = field(default_factory=dict)
    h_offset_policy: str = "match_tool_number"
    d_offset_policy: str = "match_tool_number"
    enable_tool_length_comp: bool = True

    # clearance / retract
    setup_clearance_z: Optional[float] = None
    fixture_clearance_z: Optional[float] = None
    safe_retract_z: Optional[float] = None
    safe_retract_fallback: float = 20.0
    tool_change_retract_z: Optional[float] = 100.0
    tool_change_retract_policy: str = "below"   # below | always | when_established | never
    tool_change_retract_fallback: float = 20.0  # used when tool_change_retract_z is None
    tool_change_spindle_restart: bool = False   # dialect restarts the spindle inside the tool call
    tool_change_rpm_rule: str = "spindle"       # spindle | spindle_or_tool
    footer_include_comp_off: bool = True
    footer_include_cycle_cancel: bool = True
    footer_hsm_reset_when: str = "never"        # never | always | mode_set
    footer_clear_offsets: bool = False          # dialect needs an extra G49 G40 teardown line
    stage_next_tool: bool = False
    offset_memory_type: str = "type_c_separate"

    # cycles
    supported_cycles: frozenset[str] = frozenset(
        {"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"}
    )

    # arcs
    arc_interpolation_tolerance: float = 0.005
    arc_linearization_chord_tolerance: float = 0.002
    fit_arcs_enabled: bool = True

    # spindle
    default_tool_change_rpm: Optional[float] = None
    hsm_enabled: bool = True

    # statistics
    default_rapid_feed: float = 3000.0


# ------------------------------------------------------------- audit block --
def format_operation_audit_block(
    tp: Toolpath,
    tool_num: int,
    tool_desc: str,
    tool_meta: Optional[dict],
    *,
    resolve_diameter_offset: Any,
    renumber_tools: bool = True,
    offset_memory_type: str = "type_c_separate",
    precision: int = 3,
    feed_precision: int = 1,
) -> list[str]:
    """Auditable comment block explaining how RPM, feeds and engagement derive."""
    fmt = lambda v: fmt_number(v, precision)
    fmt_f = lambda v: fmt_feed(v, feed_precision)
    lines: list[str] = []
    op_id = tp.operation_id or "op"
    purpose = str(tp.purpose).upper()
    lines.append(f"--- Operation: {op_id} [{purpose}] Tool: T{tool_num:02d} - {tool_desc} ---")

    cp = tp.metadata.get("cutting_parameters")
    if cp:
        mat_name = cp.get("material_name") or tp.metadata.get("material_name")
        mat_hb = cp.get("material_hardness_hb")
        if mat_name:
            hb_str = f" [HB {mat_hb:.0f}]" if mat_hb and mat_hb > 0 else ""
            lines.append(f"--- Material: {mat_name}{hb_str} ---")

        target_vc = cp.get("target_cutting_speed_m_min") or cp.get("cutting_speed_m_min")
        effective_vc = cp.get("effective_cutting_speed_m_min")
        rpm = cp.get("spindle_rpm")
        calc_rpm = cp.get("calculated_rpm") or cp.get("target_rpm")
        rpm_clamped = cp.get("rpm_clamped")
        max_rpm = cp.get("machine_max_rpm")
        min_rpm = cp.get("machine_min_rpm")
        tool_dia = cp.get("tool_diameter_mm") or (tool_meta.get("diameter") if tool_meta else None)

        if (effective_vc is None or effective_vc == 0.0) and tool_dia and rpm:
            effective_vc = (np.pi * float(tool_dia) * float(rpm)) / 1000.0
        if (target_vc is None or target_vc == 0.0) and effective_vc:
            target_vc = effective_vc

        if rpm is not None and rpm > 0:
            if rpm_clamped and calc_rpm and calc_rpm > 0 and abs(calc_rpm - rpm) > 1.0:
                limit_val = max_rpm if calc_rpm > rpm else min_rpm
                limit_str = f"{limit_val:.0f} RPM" if limit_val else f"{rpm:.0f} RPM"
                lines.append(
                    f"--- Speed: Target Vc = {target_vc:.1f} m/min (Calc: S{int(round(calc_rpm))} RPM) | "
                    f"Machine Limit: {limit_str} -> Applied S{int(round(rpm))} RPM (Effective Vc = {effective_vc:.1f} m/min) ---"
                )
            else:
                eff_str = f" (Effective Vc = {effective_vc:.1f} m/min)" if (effective_vc and effective_vc > 0) else ""
                vc_str = f"Vc = {target_vc:.1f} m/min -> " if (target_vc and target_vc > 0) else ""
                lines.append(f"--- Speed: {vc_str}S{int(round(rpm))} RPM{eff_str} ---")

        fz = cp.get("feed_per_tooth_mm") or cp.get("chip_load_mm")
        flutes = cp.get("flute_count") or (tool_meta.get("flutes") if tool_meta else None) or 1
        f = cp.get("feed_rate")
        feed_clamped = cp.get("feed_clamped")
        max_feed = cp.get("machine_max_feed")
        plunge_f = cp.get("plunge_feed")
        ramp_f = cp.get("ramp_feed")

        if fz is not None and fz > 0 and f is not None and f > 0:
            feed_details = []
            if plunge_f and plunge_f > 0 and abs(plunge_f - f) > 1e-3:
                feed_details.append(f"Plunge: F{fmt_f(plunge_f)}")
            if ramp_f and ramp_f > 0 and abs(ramp_f - f) > 1e-3:
                feed_details.append(f"Ramp: F{fmt_f(ramp_f)}")
            detail_str = f" [{', '.join(feed_details)}]" if feed_details else ""
            clamp_feed_str = f" [Machine Max Clamp: {max_feed:.0f}]" if feed_clamped else ""
            lines.append(f"--- Feeds: fz = {fz:.4f} mm/tooth x {flutes}F -> F{fmt_f(f)} mm/min{detail_str}{clamp_feed_str} ---")

        actual_ap = cp.get("actual_depth_of_cut_mm")
        max_ap = cp.get("max_depth_of_cut_mm") or cp.get("depth_of_cut_mm")
        stock_removal = cp.get("stock_removal_z_mm")
        ap_val = actual_ap if (actual_ap is not None and actual_ap > 0) else max_ap

        ae = cp.get("stepover_mm")
        power = cp.get("power_kw")
        torque = cp.get("torque_nm")
        eng_parts = []
        if ap_val is not None and ap_val > 0:
            if max_ap is not None and max_ap > 0 and abs(ap_val - max_ap) > 1e-3:
                if stock_removal is not None and stock_removal > 0 and abs(stock_removal - ap_val) < 1e-3:
                    eng_parts.append(f"ap = {ap_val:.2f}mm [Actual Stock: {stock_removal:.2f}mm, Max Tool DOC: {max_ap:.2f}mm]")
                elif stock_removal is not None and stock_removal > 0:
                    eng_parts.append(f"ap = {ap_val:.2f}mm [Total Stock: {stock_removal:.2f}mm, Max Tool DOC: {max_ap:.2f}mm]")
                else:
                    eng_parts.append(f"ap = {ap_val:.2f}mm [Max Tool DOC: {max_ap:.2f}mm]")
            else:
                eng_parts.append(f"ap = {ap_val:.2f}mm")
        if ae is not None and ae > 0:
            eng_parts.append(f"ae = {ae:.2f}mm")
        if power is not None and power > 0:
            eng_parts.append(f"Power: {power:.2f}kW")
        if torque is not None and torque > 0:
            eng_parts.append(f"Torque: {torque:.2f}Nm")
        if eng_parts:
            lines.append(f"--- Engagement: {', '.join(eng_parts)} ---")

    cl_info = tp.metadata.get("clearance_info")
    if cl_info:
        cl_z = cl_info.get("clearance_z", tp.metadata.get("clearance_z"))
        notes = cl_info.get("derivation_notes", "")
        lines.append(f"--- Setup Clearance: Z{fmt(cl_z)} [{notes}] ---")
    elif tp.metadata.get("clearance_z") is not None:
        cl_z = tp.metadata["clearance_z"]
        lines.append(f"--- Setup Clearance: Z{fmt(cl_z)} ---")

    cc_info = tp.metadata.get("cutter_comp_info")
    if cc_info:
        explicit_d = None
        if tool_meta and tool_meta.get("diameter_offset_d") is not None:
            explicit_d = tool_meta["diameter_offset_d"]
        elif cc_info.get("diameter_offset_d") is not None:
            explicit_d = cc_info["diameter_offset_d"]
        elif not renumber_tools and cc_info.get("d_register") is not None:
            explicit_d = cc_info["d_register"]

        d_reg = resolve_diameter_offset(
            tool_id=tp.tool_id,
            tool_num=tool_num,
            diameter_offset_d=explicit_d,
            tool_meta=tool_meta,
        )
        side_str = str(cc_info.get("side", "left")).lower()
        side_desc = "Left (Climb)" if side_str == "left" else ("Right (Conventional)" if side_str == "right" else "Off")
        g_code = "G41" if side_str == "left" else ("G42" if side_str == "right" else "G40")
        mode = str(cc_info.get("mode", "wear")).lower()
        lead = cc_info.get("lead_len", 0.0)
        t_dia = tool_meta.get("diameter", 0.0) if tool_meta else 0.0
        r_tool = t_dia / 2.0
        mem_type = offset_memory_type or "type_c_separate"

        if mode == "wear":
            if mem_type == "type_c_separate":
                reg_info = f"CNC D{d_reg} [Fanuc Type C: Geom=0.000mm, Wear=0.000mm (Wear Delta Only)]"
            elif mem_type == "type_a_combined":
                reg_info = f"CNC D{d_reg} [Haas/Combined: Offset=0.000mm (Wear Delta Only)]"
            else:
                reg_info = f"CNC D{d_reg} [Geom=0.000mm, Wear=0.000mm (Wear Delta Only)]"
            mode_desc = f"Wear (Tool Center Path | {reg_info})"
        elif mode == "control":
            if mem_type == "type_c_separate":
                reg_info = f"CNC D{d_reg} [Fanuc Type C: Geom=R{r_tool:.3f}mm, Wear=0.000mm (Full Cutter Radius)]"
            elif mem_type == "type_a_combined":
                reg_info = f"CNC D{d_reg} [Haas/Combined: Radius=R{r_tool:.3f}mm]"
            else:
                reg_info = f"CNC D{d_reg} [Geom=R{r_tool:.3f}mm, Wear=0.000mm]"
            mode_desc = f"Control (Part Contour Path | {reg_info})"
        elif mode == "computer":
            mode_desc = "Computer (Tool Center Path, No CNC G41/G42 Code)"
        else:
            mode_desc = f"{mode.title()} (Tool Center)"

        lines.append(f"--- Cutter Comp: {g_code} D{d_reg} [Tool: T{tool_num:02d} \u00d8{t_dia:.2f}mm, Mode: {mode_desc}, Side: {side_desc}, Lead: {lead:.2f}mm] ---")

    return lines


# ------------------------------------------------------------ program number --
def resolve_program_number(
    program_name: Optional[str] = None,
    setup_id: Optional[str] = None,
    setup_index: Optional[int] = None,
    job_id: Optional[str] = None,
    explicit_number: Optional[Union[int, str]] = None,
    base_number: Optional[int] = None,
    used_numbers: Optional[set[int]] = None,
) -> int:
    """Derive a unique, collision-free CNC program number adhering to machine conventions.

    Rules & Priority:
    1. Explicit Number: If provided (e.g. 1001, "O1002"), parse and validate within [1, 99999].
    2. Configured Base Number: If base_number is given, add setup offset (e.g. base=5000, setup 2 -> 5002).
    3. Deterministic Job/Part Block + Setup:
       - Computes a deterministic 32-bit CRC hash from the job/part identifier.
       - Selects a unique 100-block in standard user range 1000..8800 (leaving 9000..9999 for OEM/canned macros).
       - Adds setup offset derived from setup_index or setup_id digits (e.g. setup_001 -> +1, setup_002 -> +2).
    4. Collision Registry: If used_numbers is provided, resolves collisions dynamically.
    """
    if explicit_number is not None:
        try:
            cleaned = str(explicit_number).strip().lstrip("Oo")
            val = int(cleaned)
            if 1 <= val <= 99999:
                if used_numbers is not None:
                    while val in used_numbers:
                        val += 1
                        if val >= 90000:
                            val = 1001
                    used_numbers.add(val)
                return val
        except ValueError:
            pass

    # Determine setup offset
    offset = 1
    if setup_index is not None:
        offset = setup_index if setup_index > 0 else (setup_index + 1)
    elif setup_id:
        digits = "".join(filter(str.isdigit, str(setup_id)))
        if digits:
            d_val = int(digits)
            # If formatted like OP10, OP20 -> normalize to 1, 2 if divisible by 10
            if d_val >= 10 and d_val % 10 == 0:
                offset = d_val // 10
            else:
                offset = d_val
    elif program_name:
        m = re.search(r"(?:setup[-_]?|OP[-_]?)(\d+)", str(program_name), re.IGNORECASE)
        if m:
            d_val = int(m.group(1))
            if d_val >= 10 and d_val % 10 == 0:
                offset = d_val // 10
            else:
                offset = d_val

    if base_number is not None and base_number > 0:
        num = base_number + (offset % 100)
    else:
        # Extract base job seed string
        seed_str = str(job_id or "").strip()
        if not seed_str and program_name:
            # Strip setup suffixes like _setup_001, _OP10, _G54 from program name to isolate part identifier
            cleaned_prog = re.sub(r"[-_](?:setup[-_]?\d+|OP[-_]?\d+|G5[4-9]|G59\.\d+).*", "", str(program_name), flags=re.IGNORECASE)
            seed_str = cleaned_prog.strip()
        if not seed_str:
            seed_str = "VEXCAM_PART"

        crc = zlib.crc32(seed_str.encode("utf-8")) & 0xFFFFFFFF
        # Pick 100-block between 1000 and 8800 (79 blocks: 1000, 1100, ..., 8800)
        job_block = 1000 + ((crc % 79) * 100)
        num = job_block + (offset % 100)

    # Collision avoidance against session registry
    if used_numbers is not None:
        while num in used_numbers:
            num += 1
            if num >= 9000:
                num = 1001
        used_numbers.add(num)

    return num


# ------------------------------------------------------------------ builder --
class ProgramBuilder:
    """Resolves toolpaths into a CanonicalProgram.

    Every decision that used to live inside the post-processor happens here:
    registers, feeds, spindle, compensation, coolant, clearance planes,
    canned-cycle parameters, arc validity/linearization, teardown and stats.
    """

    def __init__(self, options: BuildOptions):
        self.o = options

        # --- semantic state (what must be emitted next) --------------------
        self.current_tool_id: Optional[str] = None
        self.current_spindle: Optional[float] = None
        self.spindle_active = False
        self.coolant_active = False
        self.comp_active = False
        self.cycle_active = False
        self.cycle_op_id: Optional[str] = None
        self.cycle_r_plane = 0.0
        self.current_hsm_mode: Optional[str] = None
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.established = False
        self.last_feed: Optional[float] = None

        self.tools_info: dict[str, dict] = {}
        self.stats = ProgramStats()

        self.setup_clearance_z = options.setup_clearance_z
        self.fixture_clearance_z = options.fixture_clearance_z
        self.safe_retract_z = options.safe_retract_z

    # ------------------------------------------------------------ registers --
    def resolve_tool_number(self, tool_id: Optional[str], default_num: int = 1) -> int:
        o = self.o
        if not tool_id:
            return default_num
        if tool_id in o.tool_map:
            return o.tool_map[tool_id]
        if o.renumber_tools:
            next_num = len(o.tool_map) + 1
            o.tool_map[tool_id] = next_num
            return next_num
        try:
            extracted = "".join(filter(str.isdigit, tool_id))
            return int(extracted) if extracted else default_num
        except ValueError:
            return default_num

    def resolve_length_offset(
        self,
        tool_id: Optional[str],
        tool_num: int,
        length_offset_h: Optional[int] = None,
        tool_meta: Optional[dict] = None,
    ) -> int:
        o = self.o
        if tool_id and tool_id in o.h_offset_map:
            return o.h_offset_map[tool_id]
        if length_offset_h is not None and length_offset_h > 0:
            return length_offset_h
        if tool_meta and tool_meta.get("length_offset_h") is not None:
            h_meta = tool_meta["length_offset_h"]
            if h_meta is not None and h_meta > 0:
                return h_meta
        if o.h_offset_policy == "strict_explicit":
            raise ValueError(
                f"Tool '{tool_id}' (T{tool_num:02d}) has no explicit H length offset defined in tool library or post mapping, "
                "and post-processor is configured with strict_explicit H-offset policy."
            )
        return tool_num

    def resolve_diameter_offset(
        self,
        tool_id: Optional[str],
        tool_num: int,
        diameter_offset_d: Optional[int] = None,
        tool_meta: Optional[dict] = None,
    ) -> int:
        o = self.o
        if tool_id and tool_id in o.d_offset_map:
            return o.d_offset_map[tool_id]

        if tool_meta and tool_meta.get("diameter_offset_d") is not None:
            d_meta = tool_meta["diameter_offset_d"]
            if d_meta is not None and d_meta > 0:
                return d_meta
        if diameter_offset_d is not None and diameter_offset_d > 0 and not o.renumber_tools:
            return diameter_offset_d

        if o.d_offset_policy == "strict_explicit":
            raise ValueError(
                f"Tool '{tool_id}' (T{tool_num:02d}) has no explicit D diameter/radius offset defined in tool library or post mapping, "
                "and post-processor is configured with strict_explicit D-offset policy."
            )

        return tool_num

    # ------------------------------------------------------------- clearance --
    def safe_retract(self, fallback: Optional[float] = None) -> float:
        candidates = [
            z for z in (self.setup_clearance_z, self.fixture_clearance_z, self.safe_retract_z)
            if z is not None
        ]
        if candidates:
            return max(candidates)
        if fallback is not None:
            return fallback
        return self.o.safe_retract_fallback

    def _derive_clearances(self, toolpaths: Sequence[Toolpath]) -> None:
        if self.setup_clearance_z is not None or not toolpaths:
            return
        derived: list[float] = []
        for tp in toolpaths:
            if tp.metadata:
                if tp.metadata.get("setup_clearance_z") is not None:
                    derived.append(float(tp.metadata["setup_clearance_z"]))
                elif tp.metadata.get("clearance_z") is not None:
                    derived.append(float(tp.metadata["clearance_z"]))
                if tp.metadata.get("fixture_clearance_z") is not None:
                    derived.append(float(tp.metadata["fixture_clearance_z"]))
            for s in tp.segments:
                if s.metadata:
                    cc = s.metadata.get("canned_cycle")
                    if cc and cc.get("initial_z") is not None:
                        derived.append(float(cc["initial_z"]))
        if derived:
            self.setup_clearance_z = max(derived)

    # ------------------------------------------------------------------ data --
    def _collect_tools(self, toolpaths: Sequence[Toolpath], tools: Optional[Sequence[Any]]) -> None:
        tools_info: dict[str, dict] = {}
        if tools:
            for t in tools:
                if hasattr(t, "id"):
                    tools_info[str(t.id)] = {
                        "id": getattr(t, "id", None),
                        "tool_number": getattr(t, "tool_number", None),
                        "type": str(getattr(t, "type", "")),
                        "diameter": getattr(t, "diameter", None),
                        "corner_radius": getattr(t, "corner_radius", None),
                        "flute_length": getattr(t, "flute_length", None),
                        "overall_length": getattr(t, "overall_length", None),
                        "flutes": getattr(t, "flutes", None),
                    }
                elif isinstance(t, dict) and "id" in t:
                    tools_info[str(t["id"])] = t

        for tp in toolpaths:
            if tp.tool_id and tp.tool_id not in tools_info:
                meta = tp.metadata.get("tool_info") or tp.metadata.get("tool")
                if not meta:
                    for s in tp.segments:
                        if s.metadata.get("tool_info"):
                            meta = s.metadata["tool_info"]
                            break
                        elif s.metadata.get("tool"):
                            meta = s.metadata["tool"]
                            break
                if meta:
                    tools_info[tp.tool_id] = meta

        if self.o.renumber_tools and not self.o.tool_map:
            for tp in toolpaths:
                if tp.tool_id and tp.tool_id not in self.o.tool_map:
                    self.resolve_tool_number(tp.tool_id)

        for tool_id, meta in list(tools_info.items()):
            if isinstance(meta, dict):
                meta.setdefault("tool_number", self.resolve_tool_number(tool_id))

        self.tools_info = tools_info


    # ---------------------------------------------------------- tool change --
    def _tool_change_commands(
        self,
        tool_id: str,
        tool_num: int,
        h_offset: int,
        tool_meta: Optional[dict],
        target_op_rpm: Optional[float],
        explicit_h: Optional[int] = None,
) -> list[CanonicalCommand]:
        """Teardown + tool exchange, in the controller-safe order the engine chose."""
        cmds: list[CanonicalCommand] = []
        if self.current_tool_id == tool_id:
            return cmds

        if self.comp_active:
            cmds.append(CompOff())
            self.comp_active = False
        if self.spindle_active:
            cmds.append(SpindleStop())
            self.spindle_active = False

        retract = self._retract_plane_for_tool_change()
        if retract is not None:
            target = np.array([self.current_pos[0], self.current_pos[1], float(retract)])
            cmds.append(Rapid(target, axes=("z",)))
            self.current_pos = target.copy()

        rpm = self._tool_change_rpm(tool_meta, target_op_rpm)
        cmds.append(ToolChange(
            tool_id=tool_id,
            tool_number=tool_num,
            length_offset_h=explicit_h if explicit_h is not None else h_offset,
            description=format_tool_description(tool_id, tool_meta),
            spindle_rpm=rpm if self.o.tool_change_spindle_restart else None,
        ))
        self.current_tool_id = tool_id
        if rpm is not None and self.o.tool_change_spindle_restart:
            self.current_spindle = rpm
            self.spindle_active = True
        else:
            self.current_spindle = None
            self.spindle_active = False
        self.established = False
        if tool_id not in self.stats.tool_changes:
            self.stats.tool_changes.append(tool_id)
        return cmds

    def _retract_plane_for_tool_change(self) -> Optional[float]:
        o = self.o
        plane = o.tool_change_retract_z
        if plane is None:
            plane = self.safe_retract(fallback=o.tool_change_retract_fallback)
        if o.tool_change_retract_policy == "never":
            return None
        if o.tool_change_retract_policy == "always":
            return float(plane)
        if o.tool_change_retract_policy == "when_established":
            return float(plane) if self.established else None
        # default: "below" -- retract only when the tool is under the retract plane
        if self.current_pos[2] < float(plane) - 1e-6:
            return float(plane)
        return None

    def _tool_change_rpm(self, tool_meta: Optional[dict], target_op_rpm: Optional[float]) -> Optional[float]:
        rule = self.o.tool_change_rpm_rule
        if rule == "spindle_or_tool":
            if target_op_rpm and target_op_rpm > 0:
                return float(target_op_rpm)
            if tool_meta and tool_meta.get("spindle_rpm"):
                return float(tool_meta["spindle_rpm"])
            cp = tool_meta.get("cutting_parameters") if tool_meta else None
            if isinstance(cp, dict) and cp.get("spindle_rpm"):
                return float(cp["spindle_rpm"])
        if target_op_rpm and target_op_rpm > 0:
            return float(target_op_rpm)
        if tool_meta and tool_meta.get("spindle_rpm"):
            return float(tool_meta["spindle_rpm"])
        if self.current_spindle and self.current_spindle > 0:
            return float(self.current_spindle)
        return self.o.default_tool_change_rpm

    # ------------------------------------------------------------ programs --
    def build(self, toolpaths: Sequence[Toolpath], tools: Optional[Sequence[Any]] = None) -> CanonicalProgram:
        o = self.o
        self._derive_clearances(toolpaths)
        self._collect_tools(toolpaths, tools)

        setup_id = o.setup_id
        if not setup_id:
            for tp in toolpaths:
                if tp.metadata and tp.metadata.get("setup_id"):
                    setup_id = tp.metadata["setup_id"]
                    break

        program_number = resolve_program_number(
            program_name=o.program_name,
            setup_id=setup_id,
            setup_index=o.setup_index,
            job_id=o.job_id,
            explicit_number=o.program_number,
            base_number=o.base_program_number,
            used_numbers=o.used_program_numbers,
        )

        operations: list[CanonicalToolpath] = []
        for tp in toolpaths:
            if getattr(tp, "is_empty_or_skipped", False) or (tp.metadata and tp.metadata.get("skipped")):
                continue
            operations.append(self._build_operation(tp, toolpaths))

        footer = self._build_footer()
        return CanonicalProgram(
            program_name=o.program_name,
            program_number=program_number,
            work_offset=o.work_offset,
            tools_info=self.tools_info,
            operations=tuple(operations),
            footer_commands=tuple(footer),
            stats=self.stats,
        )

    def _build_footer(self) -> list[CanonicalCommand]:
        o = self.o
        cmds: list[CanonicalCommand] = []
        if self.cycle_active and o.footer_include_cycle_cancel:
            cmds.append(CycleCancel())
            self.cycle_active = False
            self.cycle_op_id = None
        if self.comp_active and o.footer_include_comp_off:
            cmds.append(CompOff())
            self.comp_active = False
        if self.coolant_active:
            cmds.append(CoolantOff())
            self.coolant_active = False
        if self.spindle_active:
            cmds.append(SpindleStop())
            self.spindle_active = False
        hsm_reset = (
            o.footer_hsm_reset_when == "always"
            or (o.footer_hsm_reset_when == "mode_set" and self.current_hsm_mode is not None)
        )
        if hsm_reset:
            cmds.append(MachineCommand("hsm_reset"))
        if o.footer_clear_offsets:
            cmds.append(MachineCommand("clear_offsets"))
        cmds.append(SafeReturn("z", self.safe_retract()))
        cmds.append(SafeReturn("xy", 0.0))
        cmds.append(ProgramEnd())
        return cmds

    def _build_operation(self, tp: Toolpath, all_toolpaths: Sequence[Toolpath]) -> CanonicalToolpath:
        o = self.o
        cmds: list[CanonicalCommand] = []

        t_num = self.resolve_tool_number(tp.tool_id)
        t_meta = self.tools_info.get(tp.tool_id)
        t_desc = format_tool_description(tp.tool_id, t_meta)
        h_offset = self.resolve_length_offset(tp.tool_id, t_num, tool_meta=t_meta)
        wcs = tp.metadata.get("work_offset", o.work_offset)

        # prior-operation modal teardown
        if self.cycle_active:
            cmds.append(CycleCancel())
            self.cycle_active = False
            self.cycle_op_id = None
        if self.comp_active:
            cmds.append(CompOff())
            self.comp_active = False

        # audit comments
        cmds.append(BlankLine())
        for text in format_operation_audit_block(
            tp, t_num, t_desc, t_meta,
            resolve_diameter_offset=self.resolve_diameter_offset,
            renumber_tools=o.renumber_tools,
            offset_memory_type=o.offset_memory_type,
            precision=o.precision,
            feed_precision=o.feed_precision,
        ):
            cmds.append(Comment(text))

        self.established = False

        cp = tp.metadata.get("cutting_parameters") if tp.metadata else None
        target_op_rpm = float(cp["spindle_rpm"]) if (cp and cp.get("spindle_rpm")) else None

        tool_changed = False
        if tp.tool_id and tp.tool_id != self.current_tool_id:
            tool_changed = True
            h_explicit = None
            for s in tp.segments:
                if s.motion_type is MotionType.TOOL_CHANGE and s.metadata.get("length_offset_h") is not None:
                    h_explicit = s.metadata["length_offset_h"]
                    break
            meta_for_tc = dict(t_meta) if t_meta else {}
            if target_op_rpm:
                meta_for_tc["spindle_rpm"] = target_op_rpm
            cmds.extend(self._tool_change_commands(
                tp.tool_id, t_num, h_offset, meta_for_tc, target_op_rpm, explicit_h=h_explicit,
            ))

            if o.__dict__.get("stage_next_tool"):
                current_idx = all_toolpaths.index(tp)
                next_tool_id = None
                for rem_tp in all_toolpaths[current_idx + 1:]:
                    if rem_tp.tool_id and rem_tp.tool_id != tp.tool_id:
                        next_tool_id = rem_tp.tool_id
                        break
                if next_tool_id:
                    cmds.append(MachineCommand("tool_stage", (("tool_number", self.resolve_tool_number(next_tool_id)),)))

        # If the first segment has an unsupported cycle, emit rapid to its XY position now
        # so the first motion in the operation establishes X,Y,Z at the new location
        first_seg = tp.segments[0] if tp.segments else None
        first_cycle = first_seg.metadata.get("canned_cycle") if (first_seg and first_seg.metadata) else None
        if first_cycle and first_cycle.get("type", "G83") not in o.supported_cycles:
            x = first_cycle.get("x", first_seg.end[0])
            y = first_cycle.get("y", first_seg.end[1])
            z_r = first_cycle.get("z_r", first_seg.start[2])
            rapid_target = np.array([float(x), float(y), float(z_r)], dtype=float)
            cmds.append(Rapid(rapid_target))
            self.current_pos = rapid_target.copy()
            self.established = True

        cmds.append(SafetyBlock(str(wcs), tool_number=t_num, h_offset=h_offset))

        if target_op_rpm and target_op_rpm > 0:
            if not self.spindle_active or self.current_spindle != target_op_rpm:
                cmds.append(Spindle(float(target_op_rpm), source="operation"))
                self.current_spindle = target_op_rpm
                self.spindle_active = True

        if o.hsm_enabled:
            mode = (tp.metadata or {}).get("hsm_mode")
            if not mode:
                if tp.purpose in ("roughing", "facing", "rest_machining"):
                    mode = "roughing"
                elif tp.purpose in ("semi_finishing", "chamfering", "countersinking", "grooving"):
                    mode = "medium"
                else:
                    mode = "finishing"
            if mode != self.current_hsm_mode:
                self.current_hsm_mode = mode
                cmds.append(MachineCommand("hsm_mode", (("mode", mode),)))

        segments = fit_arcs(tp.segments) if o.fit_arcs_enabled else list(tp.segments)
        for seg in segments:
            cmds.extend(self._segment_commands(seg))

        return CanonicalToolpath(
            operation_id=tp.operation_id or "op",
            feature_id=tp.feature_id,
            purpose=str(tp.purpose),
            tool_id=tp.tool_id,
            tool_number=t_num,
            work_offset=str(wcs),
            commands=tuple(cmds),
        )


    # ------------------------------------------------------------- segments --
    def _segment_commands(self, seg: MotionSegment) -> list[CanonicalCommand]:
        o = self.o
        cmds: list[CanonicalCommand] = []
        dist = seg.length()

        if seg.motion_type == MotionType.SPINDLE_START:
            if seg.spindle is not None and seg.spindle > 0:
                if seg.spindle != self.current_spindle or not self.spindle_active:
                    cmds.append(Spindle(float(seg.spindle)))
                    self.current_spindle = seg.spindle
                    self.spindle_active = True
            return cmds
        if seg.motion_type == MotionType.SPINDLE_STOP:
            if self.spindle_active:
                cmds.append(SpindleStop())
                self.spindle_active = False
            return cmds

        # spindle speed piggybacked on motion segments
        if seg.spindle is not None and seg.spindle > 0:
            if seg.spindle != self.current_spindle or not self.spindle_active:
                cmds.append(Spindle(float(seg.spindle)))
                self.current_spindle = seg.spindle
                self.spindle_active = True

        if seg.motion_type == MotionType.TOOL_CHANGE:
            if self.cycle_active:
                cmds.append(CycleCancel())
                self.cycle_active = False
                self.cycle_op_id = None
            t_num = 1
            if seg.tool_id and seg.tool_id.startswith("T"):
                try:
                    t_num = int("".join(filter(str.isdigit, seg.tool_id)))
                except ValueError:
                    t_num = 1
            seg_h = seg.metadata.get("length_offset_h") if seg.metadata else None
            tool_id = seg.tool_id or "Tool1"
            actual = self.resolve_tool_number(tool_id, t_num)
            h_offset = self.resolve_length_offset(tool_id, actual, seg_h, None)
            cmds.extend(self._tool_change_commands(tool_id, actual, h_offset, None, None, explicit_h=seg_h))
            return cmds

        if seg.motion_type == MotionType.COOLANT_ON:
            if not self.coolant_active:
                coolant_type = seg.metadata.get("coolant_type", "flood") if seg.metadata else "flood"
                cmds.append(Coolant(coolant_type or "flood"))
                self.coolant_active = True
            return cmds
        if seg.motion_type == MotionType.COOLANT_OFF:
            if self.coolant_active:
                cmds.append(CoolantOff())
                self.coolant_active = False
            return cmds

        if seg.motion_type == MotionType.DWELL:
            cmds.append(Dwell(float(seg.dwell_seconds or 0.0)))
            self.stats.cut_time += seg.dwell_seconds
            return cmds

        comp = seg.metadata.get("cutter_comp") if seg.metadata else None
        if comp:
            cmds.extend(self._compensation_commands(seg, comp))

        cycle = seg.metadata.get("canned_cycle") if seg.metadata else None
        if cycle and cycle.get("type", "G83") in o.supported_cycles:
            cmds.extend(self._cycle_commands(seg, cycle))
            return cmds
        elif cycle:
            # Unsupported cycle: emit rapid to cycle's initial position (X, Y, Z_R)
            # so the controller starts at the correct XY before explicit plunge motions
            x = cycle.get("x", seg.end[0])
            y = cycle.get("y", seg.end[1])
            z_r = cycle.get("z_r", seg.start[2])
            rapid_target = np.array([float(x), float(y), float(z_r)], dtype=float)
            cmds.append(Rapid(rapid_target))
            self.current_pos = rapid_target.copy()
            self.established = True
            # Fall through to let PLUNGE/CUT handler emit the explicit plunge motion

        if self.cycle_active and self.cycle_op_id == seg.operation_id and \
                seg.motion_type in (MotionType.PLUNGE, MotionType.DWELL):
            return cmds

        if seg.motion_type in (MotionType.RAPID, MotionType.LINK, MotionType.RETRACT):
            if self.cycle_active and self.cycle_op_id == seg.operation_id and \
                    seg.motion_type in (MotionType.RETRACT, MotionType.RAPID) and \
                    seg.end[2] <= self.cycle_r_plane + 1e-6:
                return cmds
            if self.cycle_active:
                cmds.append(CycleCancel())
                self.cycle_active = False
                self.cycle_op_id = None
            if seg.motion_type is MotionType.RETRACT and seg.feed is not None and seg.feed > 0:
                feed = float(seg.feed)
                self.last_feed = feed
                cmds.append(Linear(seg.end.copy(), feed))
                self.stats.cut_dist += dist
                self.stats.cut_time += (dist / max(feed, 10.0)) * 60.0
            else:
                cmds.append(Rapid(seg.end.copy()))
                self.stats.rapid_dist += dist
                self.stats.rapid_time += (dist / max(o.default_rapid_feed, 100.0)) * 60.0
            self.current_pos = seg.end.copy()
            self.established = True
            return cmds

        if seg.motion_type in (MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
                               MotionType.ENTRY, MotionType.EXIT):
            if self.cycle_active:
                cmds.append(CycleCancel())
                self.cycle_active = False
                self.cycle_op_id = None
            feed = seg.feed or self.last_feed or 500.0
            self.last_feed = feed
            cmds.append(Linear(seg.end.copy(), feed))
            self.stats.cut_dist += dist
            self.stats.cut_time += (dist / max(feed, 10.0)) * 60.0
            self.current_pos = seg.end.copy()
            self.established = True
            return cmds

        if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX):
            cmds.extend(self._arc_commands(seg, dist, is_helix=(seg.motion_type == MotionType.HELIX)))
            return cmds

        return cmds

    def _compensation_commands(self, seg: MotionSegment, comp: dict) -> list[CanonicalCommand]:
        o = self.o
        cmds: list[CanonicalCommand] = []
        action = comp.get("action", "on")
        side = comp.get("side", "left")
        d_reg = comp.get("d_register", 0)
        t_meta = self.tools_info.get(self.current_tool_id) if self.tools_info else None
        if not t_meta and seg.metadata and seg.metadata.get("tool_info"):
            t_meta = seg.metadata["tool_info"]
        if action == "on" and not self.comp_active:
            explicit_d = None
            if t_meta and t_meta.get("diameter_offset_d") is not None:
                explicit_d = t_meta["diameter_offset_d"]
            elif d_reg > 0 and not o.renumber_tools:
                explicit_d = d_reg
            actual_t = self.resolve_tool_number(self.current_tool_id)
            d_reg_resolved = self.resolve_diameter_offset(self.current_tool_id, actual_t, explicit_d, t_meta)
            cmds.append(CompOn(side, d_reg_resolved))
            self.comp_active = True
            self.comp_side = side
        elif action == "off" and self.comp_active:
            cmds.append(CompOff())
            self.comp_active = False
            self.comp_side = None
        return cmds

    def _cycle_commands(self, seg: MotionSegment, cycle: dict) -> list[CanonicalCommand]:
        o = self.o
        cycle_type = cycle.get("type", "G83")
        x = cycle.get("x", seg.end[0])
        y = cycle.get("y", seg.end[1])
        z_r = cycle.get("z_r", seg.start[2])
        z_depth = cycle.get("z_depth", seg.end[2])
        peck = cycle.get("peck", 0.0)
        dwell = cycle.get("dwell", 0.0)
        feed = cycle.get("feed", self.last_feed or 500.0)
        comments = self._cycle_audit_comments(cycle, cycle_type, z_depth, z_r)
        rpm = None
        if cycle_type == "G84":
            rpm = float(seg.spindle or self.current_spindle or 0)

        cmds: list[CanonicalCommand] = [Cycle(
            cycle_type=str(cycle_type),
            x=float(x),
            y=float(y),
            depth_z=float(z_depth),
            r_plane=float(z_r),
            feed=float(feed),
            peck=float(peck or 0.0),
            dwell=float(dwell or 0.0),
            shift=float(cycle.get("shift", 0.0) or 0.0),
            retract_mode=str(cycle.get("retract_mode", "G98") or "G98"),
            initial_plane=(float(cycle["initial_z"]) if cycle.get("initial_z") is not None else None),
            surface_z=float(cycle.get("surface_z", 0.0) or 0.0),
            clearance_plane=self.safe_retract(),
            safe_distance=max(0.5, abs(float(z_r))),
            spindle_rpm=rpm,
            comments=tuple(comments),
        )]

        self.last_feed = feed
        self.current_pos = np.array([float(x), float(y), float(z_r)])
        self.cycle_active = True
        self.cycle_op_id = seg.operation_id
        self.cycle_r_plane = float(z_r)
        self.established = True
        return cmds

    @staticmethod
    def _cycle_audit_comments(cycle: dict, cycle_type: Any, z_depth: float, z_r: float) -> list[str]:
        depth_ref = cycle.get("depth_reference")
        depth_type = cycle.get("depth_type", "")
        depth_datum = cycle.get("depth_ref_datum", "TOP_FACE")
        target_depth = cycle.get("target_depth")
        tip_len = cycle.get("tip_length")
        cyl_depth = cycle.get("cylindrical_depth")
        rem_web = cycle.get("remaining_web")
        pt_angle = cycle.get("point_angle_deg")
        tip_geom = cycle.get("tip_geometry")
        peck_val = cycle.get("peck")
        ld_ratio = cycle.get("ld_ratio")
        peck_ratio = cycle.get("peck_ratio")
        mat_name = cycle.get("material_name")

        out: list[str] = []
        if depth_ref and tip_len is not None and cyl_depth is not None:
            if not depth_type:
                depth_type = "TIP_DEPTH" if depth_ref == "tip" else ("CYLINDRICAL_DEPTH" if depth_ref in ("shoulder", "full_diameter") else "THROUGH_BREAKTHROUGH")
            ref_label = "Tip Depth" if (depth_ref == "tip" or depth_type == "TIP_DEPTH") else ("Shoulder (Full Dia)" if (depth_ref in ("shoulder", "full_diameter") or depth_type == "CYLINDRICAL_DEPTH") else "Through Breakthrough")
            tip_travel = cycle.get("tip_travel")
            travel_str = f", Tip Travel={tip_travel:.3f}mm" if tip_travel is not None else ""
            target_str = f", Target={target_depth:.3f}mm" if target_depth is not None else ""
            web_str = f", Remaining Web: {rem_web:.3f}mm" if (rem_web is not None and rem_web > 0) else ""
            geom_str = f" [Point Angle: {pt_angle:.1f}deg, Geom: {tip_geom}]" if pt_angle is not None and tip_geom is not None else (f" [Point Angle: {pt_angle:.1f}deg]" if pt_angle is not None else "")
            out.append(f"--- Drill Depth Semantics: Ref={ref_label} (Datum: {depth_datum}, Type: {depth_type}), Cycle Z={z_depth:.3f}{target_str}, Tip Len={tip_len:.3f}mm{geom_str}, Cyl Depth={cyl_depth:.3f}mm{travel_str}{web_str} ---")

        if peck_val and peck_val > 0 and (ld_ratio is not None or peck_ratio is not None):
            p_ratio_str = f" ({peck_ratio:.2f}xD)" if peck_ratio is not None else ""
            ld_str = f", L/D={ld_ratio:.2f}" if ld_ratio is not None else ""
            mat_str = f", Mat={mat_name}" if mat_name else ""
            out.append(f"--- Drill Peck Strategy: Q={peck_val:.3f}mm{p_ratio_str}, Cycle={cycle_type}{ld_str}{mat_str} ---")

        retract_mode = cycle.get("retract_mode", "G98")
        retract_reason = cycle.get("retract_reason", "")
        if retract_mode:
            retract_desc = "Initial Plane" if retract_mode == "G98" else "R-Plane"
            r_str = f" [{retract_desc} Z{z_r:.3f}]" if z_r is not None else ""
            init_val = cycle.get("initial_z")
            init_str = f", Initial Z={init_val:.3f}" if (init_val is not None and retract_mode == "G98") else ""
            reason_str = f", Reason: {retract_reason}" if retract_reason else ""
            out.append(f"--- Drill Retract Strategy: Mode={retract_mode}{r_str}{init_str}{reason_str} ---")
        return out

    def _arc_commands(self, seg: MotionSegment, dist: float, is_helix: bool) -> list[CanonicalCommand]:
        o = self.o
        cmds: list[CanonicalCommand] = []
        if self.cycle_active:
            cmds.append(CycleCancel())
            self.cycle_active = False
            self.cycle_op_id = None

        start_pos = self.current_pos.copy() if self.established else np.asarray(seg.start, dtype=float)
        end_pos = np.asarray(seg.end, dtype=float)
        is_ccw = seg.arc_ccw if is_helix else (seg.motion_type == MotionType.ARC_CCW or seg.arc_ccw)
        feed = seg.feed or self.last_feed or 500.0
        self.last_feed = feed
        self.stats.cut_dist += dist
        self.stats.cut_time += (dist / max(feed, 10.0)) * 60.0

        ev = evaluate_arc(start_pos, end_pos, seg.arc_center, is_ccw, o.precision, o.arc_interpolation_tolerance)
        if ev.valid:
            center = np.array([ev.center_rounded[0], ev.center_rounded[1], start_pos[2]])
            cmds.append(Arc(
                start=start_pos.copy(),
                target=end_pos.copy(),
                center=center,
                ccw=bool(is_ccw),
                feed=feed,
                i_offset=ev.i_offset,
                j_offset=ev.j_offset,
            ))
        else:
            cmds.append(Comment(
                f"--- Post Arc Linearized: {ev.linearize_hint} exceeds tol={o.arc_interpolation_tolerance:.4f}mm, "
                f"sweep={math.degrees(ev.sweep):.1f}deg ---",
                kind="motion",
            ))
            for point in linearize_arc(ev, start_pos, end_pos, is_ccw, o.arc_linearization_chord_tolerance):
                cmds.append(Linear(point, feed))

        self.current_pos = end_pos.copy()
        self.established = True
        return cmds


def state_footer_commands(
    *,
    options: BuildOptions,
    cycle_active: bool,
    comp_active: bool,
    coolant_active: bool,
    spindle_active: bool,
    hsm_mode: Optional[str],
    safe_z: float,
) -> list[CanonicalCommand]:
    """Canonical footer for a post that only knows its own modal state.

    Used by ``footer()`` when called without a program (direct API use).
    """
    cmds: list[CanonicalCommand] = []
    if cycle_active and options.footer_include_cycle_cancel:
        cmds.append(CycleCancel())
    if comp_active and options.footer_include_comp_off:
        cmds.append(CompOff())
    if coolant_active:
        cmds.append(CoolantOff())
    if spindle_active:
        cmds.append(SpindleStop())
    hsm_reset = (
        options.footer_hsm_reset_when == "always"
        or (options.footer_hsm_reset_when == "mode_set" and hsm_mode is not None)
    )
    if hsm_reset:
        cmds.append(MachineCommand("hsm_reset"))
    if options.footer_clear_offsets:
        cmds.append(MachineCommand("clear_offsets"))
    cmds.append(SafeReturn("z", safe_z))
    cmds.append(SafeReturn("xy", 0.0))
    cmds.append(ProgramEnd())
    return cmds


def build_canonical_program(
    toolpaths: Sequence[Toolpath],
    options: BuildOptions,
    tools: Optional[Sequence[Any]] = None,
) -> CanonicalProgram:
    """Resolve toolpaths into the controller-independent canonical program."""
    return ProgramBuilder(options).build(toolpaths, tools)
