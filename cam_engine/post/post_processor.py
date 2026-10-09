"""Post-processor engine translating semantic toolpaths into controller-specific G-code.

Supports:
- GRBL / Standard ISO G-code
- Fanuc (standard 3-axis mill with line numbers, safe blocks, canned returns)
- Haas (standard NGC format)
- LinuxCNC (RS274/NGC format)
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

from ..toolpaths.semantic import MotionSegment, MotionType, Toolpath
from ..validity import assert_valid_for_post
from .gcode_parser import GCodeGeometryValidator, NCGeometryValidationReport
from .controller_models import (
    ControllerCapabilityModel,
    MachineProfile,
    ToolChangePosition,
    WcsBehavior,
    CannedCycleBehavior,
    SpindleFeedLimits,
    SafetyRetractStrategy,
    FANUC_CAPABILITY_MODEL,
    HAAS_CAPABILITY_MODEL,
    GRBL_CAPABILITY_MODEL,
    LINUXCNC_CAPABILITY_MODEL,
    HEIDENHAIN_CAPABILITY_MODEL,
    SIEMENS_CAPABILITY_MODEL,
    SINUMERIK_CAPABILITY_MODEL,
    FANUC_MACHINE_PROFILE,
    HAAS_MACHINE_PROFILE,
    GRBL_MACHINE_PROFILE,
    LINUXCNC_MACHINE_PROFILE,
    HEIDENHAIN_MACHINE_PROFILE,
    SIEMENS_MACHINE_PROFILE,
    get_controller_capability_model,
    get_machine_profile,
)
from .syntax_validator import (
    ControllerSyntaxValidator,
    ControllerSyntaxValidationReport,
)
from ..validation import (
    FinalValidationGate,
    FinalValidationResult,
    ValidationOutcome,
    ValidationPipeline,
    PipelineResult,
)
from .canonical import (
    build_canonical_program,
    BuildOptions,
    CanonicalProgram,
    CanonicalToolpath,
    CanonicalCommand,
    Comment,
    BlankLine,
    ToolChange,
    SafetyBlock,
    CompOn,
    CompOff,
    Rapid,
    Linear,
    Arc,
    Spindle,
    SpindleStop,
    Coolant,
    CoolantOff,
    Dwell,
    Cycle,
    CycleCancel,
    MachineCommand,
    SafeReturn,
    ProgramEnd,
    format_tool_description,
    resolve_program_number as canonical_resolve_program_number,
)


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


@dataclass
class PostResult:
    """The result of post-processing toolpaths into G-code."""
    gcode: str
    line_count: int
    rapid_time_seconds: float
    cut_time_seconds: float
    total_time_seconds: float
    total_rapid_dist_mm: float
    total_cut_dist_mm: float
    tool_changes: list[str]
    work_offset: str = "G54"
    program_number: int = 1001
    geometry_validation: Optional[NCGeometryValidationReport] = None
    syntax_validation: Optional[ControllerSyntaxValidationReport] = None
    final_validation: Optional[FinalValidationResult] = None


class BasePostProcessor:
    """Base modal G-code post-processor."""

    def __init__(
        self,
        precision: int = 3,
        feed_precision: int = 1,
        work_offset: str = "G54",
        default_rapid_feed: float = 3000.0,
        renumber_tools: bool = True,
        tool_map: Optional[dict[str, int]] = None,
        h_offset_map: Optional[dict[str, int]] = None,
        d_offset_map: Optional[dict[str, int]] = None,
        h_offset_policy: str = "match_tool_number",
        d_offset_policy: str = "match_tool_number",
        enable_tool_length_comp: bool = True,
        tool_change_mode: Optional[str] = None,
        tool_change_strategy: Optional[str] = None,
        tool_change_command: Optional[str] = None,
        tool_change_format: Optional[str] = None,
        stage_next_tool: bool = False,
        program_number: Optional[Union[int, str]] = None,
        base_program_number: Optional[int] = None,
        used_program_numbers: Optional[set[int]] = None,
        arc_interpolation_tolerance: float = 0.005,
        arc_linearization_chord_tolerance: float = 0.002,
        capability_model: Optional[ControllerCapabilityModel] = None,
        setup_clearance_z: Optional[float] = None,
        safe_retract_z: Optional[float] = None,
        fixture_clearance_z: Optional[float] = None,
        home_command_z: Optional[str] = None,
        home_command_xy: Optional[str] = None,
        program_end_command: Optional[str] = None,
        machine_profile: Optional[Union[MachineProfile, str]] = None,
    ):
        self.precision = precision
        self.feed_precision = feed_precision
        self.work_offset = work_offset
        self.default_rapid_feed = default_rapid_feed
        self.renumber_tools = renumber_tools
        self.tool_map: dict[str, int] = dict(tool_map) if tool_map is not None else {}
        self.h_offset_map: dict[str, int] = dict(h_offset_map) if h_offset_map is not None else {}
        self.d_offset_map: dict[str, int] = dict(d_offset_map) if d_offset_map is not None else {}
        self.h_offset_policy = h_offset_policy
        self.d_offset_policy = d_offset_policy
        self.enable_tool_length_comp = enable_tool_length_comp

        # Load machine profile
        if isinstance(machine_profile, str):
            self.machine_profile = get_machine_profile(machine_profile)
        elif isinstance(machine_profile, MachineProfile):
            self.machine_profile = machine_profile
        else:
            # Fallback to capability model based profile
            self.machine_profile = get_machine_profile(
                capability_model.controller_id if capability_model else "grbl"
            )

        self.capability_model: ControllerCapabilityModel = self.machine_profile.controller

        # Safe return and clearance geometry hierarchy (from machine profile)
        self.setup_clearance_z = setup_clearance_z
        self.safe_retract_z = safe_retract_z
        self.fixture_clearance_z = fixture_clearance_z
        self.home_command_z = home_command_z
        self.home_command_xy = home_command_xy
        self.program_end_command = program_end_command or self.machine_profile.safety.program_end_command

        # Tool change settings from machine profile (profile provides the base defaults)
        self.tool_change_mode = self.machine_profile.tool_change_strategy
        self.tool_change_strategy = self.machine_profile.tool_change_strategy
        self.tool_change_command = self.machine_profile.tool_change_command
        self.tool_change_position = self.machine_profile.tool_change
        self.tool_change_format = None  # Can be overridden per-machine

        self.stage_next_tool = False

        # Apply caller-supplied overrides (these take precedence over machine profile defaults)
        _mode_overridden = False
        if tool_change_mode is not None:
            self.tool_change_mode = tool_change_mode
            self.tool_change_strategy = tool_change_mode
            _mode_overridden = True
        if tool_change_strategy is not None:
            self.tool_change_strategy = tool_change_strategy
            if tool_change_mode is None:
                self.tool_change_mode = tool_change_strategy
            _mode_overridden = True
        if tool_change_command is not None:
            self.tool_change_command = tool_change_command
        elif _mode_overridden:
            # Derive the command from the overridden strategy when no explicit command is given,
            # so the command stays consistent with the new mode (e.g. atc→M6, manual→M0).
            eff_strat = (self.tool_change_strategy or "atc").lower()
            if eff_strat in ("manual", "m0", "stop"):
                self.tool_change_command = "M0"
            elif eff_strat in ("atc", "m6"):
                self.tool_change_command = "M6"
        if tool_change_format is not None:
            self.tool_change_format = tool_change_format
        if stage_next_tool:
            self.stage_next_tool = True
        self.program_number = program_number
        self.base_program_number = base_program_number
        self.used_program_numbers: set[int] = set()
        self.arc_interpolation_tolerance = float(arc_interpolation_tolerance)
        self.arc_linearization_chord_tolerance = float(arc_linearization_chord_tolerance)
        self.tools_info: dict[str, dict] = {}

        # Modal tracking
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_feed: Optional[float] = None
        self.current_spindle: Optional[float] = None
        self.current_motion_mode: Optional[str] = None
        self.spindle_active = False
        self.coolant_active = False
        self.current_tool_id: Optional[str] = None
        self.is_pos_established = False

        # Canned cycle state
        self.canned_cycle_active = False
        self._cycle_op_id: Optional[str] = None
        self._cycle_r_plane: float = 0.0

        # Safety retract Z from machine profile
        self.retract_z: float = self.machine_profile.tool_change.z

        # Cutter compensation state
        self.comp_active = False
        self.comp_side: Optional[str] = None
        self.hsm_enabled = self.machine_profile.hsm_default_mode != "none"
        self.current_hsm_mode: Optional[str] = None

        # Statistics
        self.rapid_dist = 0.0
        self.cut_dist = 0.0
        self.rapid_time = 0.0
        self.cut_time = 0.0
        self.used_tools: list[str] = []

        # Feed/spindle limits from machine profile
        self.max_spindle_rpm = self.machine_profile.max_spindle_rpm
        self.max_feed_mm_min = self.machine_profile.max_feed_mm_min
        self.max_rapid_mm_min = self.machine_profile.max_rapid_mm_min
        self.default_rapid_feed = self.machine_profile.default_rapid_feed_mm_min

    def _build_options(self) -> BuildOptions:
        """Construct BuildOptions from this post processor's configuration."""
        overrides = self._build_options_overrides()
        return BuildOptions(
            program_name=getattr(self, "_program_name", "VEXCAM"),
            program_number=self.program_number,
            base_program_number=self.base_program_number,
            used_program_numbers=self.used_program_numbers,
            setup_id=None,  # derived from toolpaths in builder
            setup_index=None,
            job_id=None,
            work_offset=self.work_offset,
            precision=self.precision,
            feed_precision=self.feed_precision,
            renumber_tools=self.renumber_tools,
            tool_map=self.tool_map,
            h_offset_map=self.h_offset_map,
            d_offset_map=self.d_offset_map,
            h_offset_policy=self.h_offset_policy,
            d_offset_policy=self.d_offset_policy,
            enable_tool_length_comp=self.enable_tool_length_comp,
            setup_clearance_z=self.setup_clearance_z,
            fixture_clearance_z=self.fixture_clearance_z,
            safe_retract_z=self.safe_retract_z,
            safe_retract_fallback=overrides.get("safe_retract_fallback", self.machine_profile.tool_change.z),
            tool_change_retract_z=overrides.get("tool_change_retract_z", self.machine_profile.tool_change.z),
            tool_change_retract_policy=overrides.get("tool_change_retract_policy", self.machine_profile.tool_change.retract_policy),
            tool_change_retract_fallback=overrides.get("tool_change_retract_fallback", self.machine_profile.tool_change.z),
            tool_change_spindle_restart=overrides.get("tool_change_spindle_restart", self.machine_profile.controller.tool_change.requires_spindle_stop),
            tool_change_rpm_rule=overrides.get("tool_change_rpm_rule", "spindle_or_tool" if self.machine_profile.controller.tool_change.requires_spindle_stop else "spindle"),
            footer_include_comp_off=overrides.get("footer_include_comp_off", True),
            footer_include_cycle_cancel=overrides.get("footer_include_cycle_cancel", True),
            footer_hsm_reset_when=overrides.get("footer_hsm_reset_when", "mode_set" if self.hsm_enabled else "never"),
            footer_clear_offsets=overrides.get("footer_clear_offsets", self.machine_profile.controller.controller_id == "fanuc"),
            stage_next_tool=self.stage_next_tool,
            offset_memory_type=overrides.get(
                "offset_memory_type",
                getattr(getattr(self.capability_model, "cutter_comp", None), "offset_memory_type", "type_c_separate")
            ),
            supported_cycles=frozenset(self.machine_profile.supported_canned_cycles) or frozenset(self.capability_model.supported_canned_cycles),
            arc_interpolation_tolerance=self.arc_interpolation_tolerance,
            arc_linearization_chord_tolerance=self.arc_linearization_chord_tolerance,
            fit_arcs_enabled=True,
            default_tool_change_rpm=overrides.get("default_tool_change_rpm", self.machine_profile.controller.tool_change.requires_spindle_stop and self.machine_profile.spindle_feed_limits.max_spindle_rpm),
            hsm_enabled=self.hsm_enabled,
            default_rapid_feed=self.default_rapid_feed,
        )

    def _build_options_overrides(self) -> dict:
        """Override in subclasses for controller-specific BuildOptions."""
        return {}

    def _get_safe_retract_z(self, fallback: Optional[float] = None) -> float:
        """Derive the highest safe clearance/retract Z level from setup, fixture, or machine settings."""
        candidates = [
            z for z in (self.setup_clearance_z, self.fixture_clearance_z, self.safe_retract_z)
            if z is not None
        ]
        if candidates:
            return max(candidates)
        if fallback is not None:
            return fallback
        return self.machine_profile.tool_change.z

    @property
    def supported_cycles(self) -> set[str]:
        """Supported canned cycles from machine profile."""
        return self.machine_profile.supported_cycles

    def supports_cycle(self, cycle_type: str) -> bool:
        return cycle_type in self.supported_cycles

    def _resolve_tool_number(self, tool_id: Optional[str], default_num: int = 1) -> int:
        """Resolve the effective sequential machine tool number for a given tool identifier."""
        if not tool_id:
            return default_num
        if tool_id in self.tool_map:
            return self.tool_map[tool_id]
        if self.renumber_tools:
            next_num = len(self.tool_map) + 1
            self.tool_map[tool_id] = next_num
            return next_num
        try:
            extracted = "".join(filter(str.isdigit, tool_id))
            return int(extracted) if extracted else default_num
        except ValueError:
            return default_num

    def _resolve_length_offset(
        self,
        tool_id: Optional[str],
        tool_num: int,
        length_offset_h: Optional[int] = None,
        tool_meta: Optional[dict] = None,
    ) -> int:
        """Derive the H offset register from Tool metadata, post configuration, or controller convention."""
        if tool_id and tool_id in self.h_offset_map:
            return self.h_offset_map[tool_id]
        if length_offset_h is not None and length_offset_h > 0:
            return length_offset_h
        if tool_meta and tool_meta.get("length_offset_h") is not None:
            h_meta = tool_meta["length_offset_h"]
            if h_meta is not None and h_meta > 0:
                return h_meta
        if self.h_offset_policy == "strict_explicit":
            raise ValueError(
                f"Tool '{tool_id}' (T{tool_num:02d}) has no explicit H length offset defined in tool library or post mapping, "
                "and post-processor is configured with strict_explicit H-offset policy."
            )
        return tool_num

    def _resolve_diameter_offset(
        self,
        tool_id: Optional[str],
        tool_num: int,
        diameter_offset_d: Optional[int] = None,
        tool_meta: Optional[dict] = None,
    ) -> int:
        """Derive the D offset register from Tool metadata, post configuration, or controller convention.
        
        Single deterministic source of truth for both G-code NC blocks (G41/G42 D...) and audit comments.
        """
        # 1. Post-processor explicit D offset map has highest precedence
        if tool_id and tool_id in self.d_offset_map:
            return self.d_offset_map[tool_id]

        # 2. Tool-level explicit diameter_offset_d configured in tool library / metadata
        if tool_meta and tool_meta.get("diameter_offset_d") is not None:
            d_meta = tool_meta["diameter_offset_d"]
            if d_meta is not None and d_meta > 0:
                return d_meta
        if diameter_offset_d is not None and diameter_offset_d > 0 and not self.renumber_tools:
            return diameter_offset_d

        # 3. If postprocessor has a strict explicit policy, error out
        if getattr(self, "d_offset_policy", "match_tool_number") == "strict_explicit":
            raise ValueError(
                f"Tool '{tool_id}' (T{tool_num:02d}) has no explicit D diameter/radius offset defined in tool library or post mapping, "
                "and post-processor is configured with strict_explicit D-offset policy."
            )

        # 4. Standard controller convention: D register matches active tool number (T01 -> D1, T02 -> D2, T03 -> D3)
        return tool_num

    def _format_tool_description(self, tool_id: str, tool_meta: Optional[dict] = None) -> str:
        """Dynamically and generically format tool dimensions into a human-readable string."""
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
            parts.append(f"Ø{dia_val}mm")

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

    def _format_tools_header_block(
        self,
        tools_info: Optional[dict[str, dict]] = None,
        style: str = "semicolon",
    ) -> list[str]:
        """Generic program header tool table summary block."""
        if not tools_info:
            return []

        lines = []
        sorted_tools = sorted(
            tools_info.items(),
            key=lambda item: self._resolve_tool_number(item[0])
        )

        if style == "parentheses":
            lines.append("(==========================================)")
            lines.append("(TOOLS USED IN THIS PROGRAM:)")
            for tid, tmeta in sorted_tools:
                t_num = self._resolve_tool_number(tid)
                desc = self._format_tool_description(tid, tmeta)
                lines.append(f"(T{t_num:02d}: {desc})")
            lines.append("(==========================================)")
        else:
            lines.append("; ==========================================")
            lines.append("; TOOLS USED IN THIS PROGRAM:")
            for tid, tmeta in sorted_tools:
                t_num = self._resolve_tool_number(tid)
                desc = self._format_tool_description(tid, tmeta)
                lines.append(f"; T{t_num:02d}: {desc}")
            lines.append("; ==========================================")
        return lines

    def _format_operation_audit_block(
        self,
        tp: Toolpath,
        tool_num: int,
        tool_desc: str,
        tool_meta: Optional[dict] = None,
    ) -> list[str]:
        """Format an auditable CAM operation header explaining how RPM, feed rate,
        chip load, cutting speed, and engagement were derived from material, tool, and machine limits."""
        lines = []
        op_id = tp.operation_id or "op"
        purpose = str(tp.purpose).upper()
        lines.append(f"\n(--- Operation: {op_id} [{purpose}] Tool: T{tool_num:02d} - {tool_desc} ---)")

        cp = tp.metadata.get("cutting_parameters")
        if cp:
            mat_name = cp.get("material_name") or tp.metadata.get("material_name")
            mat_hb = cp.get("material_hardness_hb")
            if mat_name:
                hb_str = f" [HB {mat_hb:.0f}]" if mat_hb and mat_hb > 0 else ""
                lines.append(f"(--- Material: {mat_name}{hb_str} ---)")

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
                        f"(--- Speed: Target Vc = {target_vc:.1f} m/min (Calc: S{int(round(calc_rpm))} RPM) | "
                        f"Machine Limit: {limit_str} -> Applied S{int(round(rpm))} RPM (Effective Vc = {effective_vc:.1f} m/min) ---)"
                    )
                else:
                    eff_str = f" (Effective Vc = {effective_vc:.1f} m/min)" if (effective_vc and effective_vc > 0) else ""
                    vc_str = f"Vc = {target_vc:.1f} m/min -> " if (target_vc and target_vc > 0) else ""
                    lines.append(f"(--- Speed: {vc_str}S{int(round(rpm))} RPM{eff_str} ---)")

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
                    feed_details.append(f"Plunge: F{self._fmt_f(plunge_f)}")
                if ramp_f and ramp_f > 0 and abs(ramp_f - f) > 1e-3:
                    feed_details.append(f"Ramp: F{self._fmt_f(ramp_f)}")
                detail_str = f" [{', '.join(feed_details)}]" if feed_details else ""
                clamp_feed_str = f" [Machine Max Clamp: {max_feed:.0f}]" if feed_clamped else ""
                lines.append(f"(--- Feeds: fz = {fz:.4f} mm/tooth x {flutes}F -> F{self._fmt_f(f)} mm/min{detail_str}{clamp_feed_str} ---)")

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
                lines.append(f"(--- Engagement: {', '.join(eng_parts)} ---)")

        cl_info = tp.metadata.get("clearance_info")
        if cl_info:
            cl_z = cl_info.get("clearance_z", tp.metadata.get("clearance_z"))
            notes = cl_info.get("derivation_notes", "")
            lines.append(f"(--- Setup Clearance: Z{self._fmt(cl_z)} [{notes}] ---)")
        elif tp.metadata.get("clearance_z") is not None:
            cl_z = tp.metadata["clearance_z"]
            lines.append(f"(--- Setup Clearance: Z{self._fmt(cl_z)} ---)")

        cc_info = tp.metadata.get("cutter_comp_info")
        if cc_info:
            explicit_d = None
            if tool_meta and tool_meta.get("diameter_offset_d") is not None:
                explicit_d = tool_meta["diameter_offset_d"]
            elif cc_info.get("diameter_offset_d") is not None:
                explicit_d = cc_info.get("diameter_offset_d")
            elif not self.renumber_tools and cc_info.get("d_register") is not None:
                explicit_d = cc_info.get("d_register")

            d_reg = self._resolve_diameter_offset(
                tool_id=tp.tool_id or self.current_tool_id,
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

            mem_type = getattr(self, "offset_memory_type", None)
            if not mem_type and hasattr(self, "capability_model") and self.capability_model:
                mem_type = getattr(self.capability_model.cutter_comp, "offset_memory_type", "type_c_separate")
            mem_type = mem_type or "type_c_separate"

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

            lines.append(f"(--- Cutter Comp: {g_code} D{d_reg} [Tool: T{tool_num:02d} Ø{t_dia:.2f}mm, Mode: {mode_desc}, Side: {side_desc}, Lead: {lead:.2f}mm] ---)")

        return lines

    def _fmt(self, val: float) -> str:
        s = f"{val:.{self.precision}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    def _fmt_f(self, val: float) -> str:
        s = f"{val:.{self.feed_precision}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    def _fmt_cycle_dwell(self, dwell_seconds: float) -> str:
        """Format dwell P value for canned cycles. Override in subclasses for ms."""
        return f"P{self._fmt(dwell_seconds)}"

    def format_coord(self, axis: str, val: float) -> str:
        return f"{axis}{self._fmt(val)}"

    def start_toolpath(self, tp: Toolpath) -> list[str]:
        """Hook called before toolpath segments. Used for HSM modes (G187/G05.1)."""
        return []

    def operation_safety_preamble(
        self,
        tp: Toolpath,
        tool_num: int,
        h_offset: int,
        wcs: str,
    ) -> list[str]:
        """Emit deterministic machine safety / modal initialization block for an operation."""
        return [f"{wcs} G90 G17 G21 G94 G40 G80"]

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        if program_number is not None:
            try:
                prog_num = int(str(program_number).strip().lstrip("Oo"))
            except ValueError:
                prog_num = 1001
        else:
            prog_num = resolve_program_number(program_name=program_name, explicit_number=self.program_number, base_number=self.base_program_number, used_numbers=self.used_program_numbers)
        lines = [
            f"; Program: {program_name} (O{prog_num:04d})",
            "(Generated by VexCAM Engine)",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="semicolon"))
        lines.append(f"{self.work_offset} G90 G17 G21 G94 G40 G49 G80")
        return lines

    def footer(self) -> list[str]:
        lines = []
        if self.canned_cycle_active:
            lines.extend(self.cycle_cancel())
            self.canned_cycle_active = False
            self._cycle_op_id = None
        if self.comp_active:
            lines.extend(self.cutter_compensation_off())
            self.comp_active = False
            self.comp_side = None
        if self.coolant_active:
            lines.append("M9")
            self.coolant_active = False
        if self.spindle_active:
            lines.append("M5")
            self.spindle_active = False
        safe_z = self._get_safe_retract_z()
        if self.home_command_z:
            lines.append(self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}"))
        else:
            lines.append(f"G0 Z{self._fmt(safe_z)}")
        if self.home_command_xy:
            lines.append(self.home_command_xy)
        else:
            lines.append("G0 X0 Y0")
        end_cmd = self.program_end_command or getattr(getattr(self.capability_model, "safe_return", None), "program_end_command", "M30")
        lines.append(end_cmd)
        return lines

    def _render_tool_change_command(self, tool_num: int, desc: str, h_offset: int) -> list[str]:
        """Format the controller-specific tool-change command block based on mode, command, or template."""
        if self.tool_change_format:
            try:
                cmd = self.tool_change_format.format(
                    tool_num=tool_num,
                    h_offset=h_offset,
                    desc=desc,
                    tool_num_2d=f"{tool_num:02d}",
                    h_offset_2d=f"{h_offset:02d}",
                )
                return [cmd]
            except Exception:
                return [f"T{tool_num} {self.tool_change_command} (Tool: T{tool_num:02d} - {desc})"]

        strat = (self.tool_change_strategy or self.tool_change_mode or "atc").lower()
        if self.tool_change_command:
            cmd_code = self.tool_change_command
        else:
            cmd_code = "M6" if strat == "atc" else "M0"

        if strat in ("manual", "m0", "stop"):
            return [
                f"; Tool change: T{tool_num:02d} ({desc})",
                f"T{tool_num} {cmd_code} (Change to T{tool_num:02d} - {desc})",
            ]
        elif strat in ("comment_only", "none"):
            comment_char = ";" if (self.capability_model and self.capability_model.block_format.comment_style == "semicolon") else ""
            if comment_char == ";":
                return [f"; T{tool_num:02d} - {desc}"]
            return [f"(T{tool_num:02d} - {desc})"]
        else:  # "atc" / "m6"
            return [f"T{tool_num} {cmd_code} (Tool: T{tool_num:02d} - {desc})"]

    def tool_change(self, tool_id: str, tool_num: int = 1,
                    length_offset_h: Optional[int] = None,
                    tool_meta: Optional[dict] = None) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)
        h = self._resolve_length_offset(tool_id, actual_tool_num, length_offset_h, tool_meta)
        if self.current_tool_id != tool_id:
            if self.comp_active:
                lines.extend(self.cutter_compensation_off())
                self.comp_active = False
                self.comp_side = None
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            # Retract Z before tool change for safety
            if self.current_pos[2] < self.retract_z - 1e-6:
                lines.append(f"G0 {self.format_coord('Z', self.retract_z)}")
                self.current_pos[2] = self.retract_z
            lines.extend(self._render_tool_change_command(actual_tool_num, desc, h))
            if self.enable_tool_length_comp:
                lines.append(f"G43 H{h}")
            self.current_tool_id = tool_id
            self.is_pos_established = False
            if tool_id not in self.used_tools:
                self.used_tools.append(tool_id)
        return lines

    def cutter_compensation_on(self, side: str = "left", d_register: int = 0, tool_meta: Optional[dict] = None) -> list[str]:
        """Enable cutter compensation. side: 'left' (G41) or 'right' (G42).
        d_register: offset register number (matches tool number convention)."""
        g = "G41" if side == "left" else "G42"
        actual_t_num = self._resolve_tool_number(self.current_tool_id)
        if tool_meta is None and hasattr(self, "tools_info") and self.tools_info:
            tool_meta = self.tools_info.get(self.current_tool_id)

        explicit_d = None
        if tool_meta and tool_meta.get("diameter_offset_d") is not None:
            explicit_d = tool_meta["diameter_offset_d"]
        elif d_register > 0 and not self.renumber_tools:
            explicit_d = d_register

        d_reg = self._resolve_diameter_offset(
            tool_id=self.current_tool_id,
            tool_num=actual_t_num,
            diameter_offset_d=explicit_d,
            tool_meta=tool_meta,
        )
        return [f"{g} D{d_reg}"]

    def cutter_compensation_off(self) -> list[str]:
        """Cancel cutter compensation."""
        return ["G40"]


    def cycle_cancel(self) -> list[str]:
        """Cancel active canned cycle (e.g. G80 on standard ISO / MCALL on Siemens)."""
        return ["G80"]

    def retract_mode(self, mode: str = "initial_level") -> list[str]:
        """Set canned cycle retract mode. mode: 'initial_level' (G98) or 'r_level' (G99)."""
        if mode == "r_level":
            return ["G99"]
        return ["G98"]

    def process_segment(self, seg: MotionSegment) -> list[str]:
        lines = []
        dist = seg.length()

        # Handle spindle start/stop explicitly
        if seg.motion_type == MotionType.SPINDLE_START:
            if seg.spindle is not None and seg.spindle > 0:
                if seg.spindle != self.current_spindle or not self.spindle_active:
                    lines.append(f"S{int(seg.spindle)} M3")
                    self.current_spindle = seg.spindle
                    self.spindle_active = True
            return lines
        if seg.motion_type == MotionType.SPINDLE_STOP:
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            return lines

        # Handle spindle speed changes or start (piggybacked on cutting segments)
        if seg.spindle is not None and seg.spindle > 0:
            if seg.spindle != self.current_spindle or not self.spindle_active:
                lines.append(f"S{int(seg.spindle)} M3")
                self.current_spindle = seg.spindle
                self.spindle_active = True

        # Handle tool change
        if seg.motion_type == MotionType.TOOL_CHANGE:
            if self.canned_cycle_active:
                lines.extend(self.cycle_cancel())
                self.canned_cycle_active = False
                self._cycle_op_id = None
            t_num = 1
            if seg.tool_id and seg.tool_id.startswith("T"):
                try:
                    t_num = int("".join(filter(str.isdigit, seg.tool_id)))
                except ValueError:
                    t_num = 1
            h_offset = seg.metadata.get("length_offset_h")
            lines.extend(self.tool_change(seg.tool_id or "Tool1", t_num, h_offset))
            return lines

        # Handle coolant
        if seg.motion_type == MotionType.COOLANT_ON:
            if not self.coolant_active:
                # M8 = flood coolant (default), M7 = mist, M50 = through-spindle
                coolant_type = seg.metadata.get("coolant_type", "flood") if seg.metadata else "flood"
                if coolant_type == "mist":
                    lines.append("M7")
                elif coolant_type == "through_spindle":
                    lines.append("M50")
                else:
                    lines.append("M8")
                self.coolant_active = True
            return lines
        if seg.motion_type == MotionType.COOLANT_OFF:
            if self.coolant_active:
                lines.append("M9")
                self.coolant_active = False
            return lines

        # Handle Dwell
        if seg.motion_type == MotionType.DWELL:
            lines.append(f"G4 P{self._fmt(seg.dwell_seconds)}")
            self.cut_time += seg.dwell_seconds
            return lines

        # Handle Cutter Compensation (G41/G42/G40)
        comp = seg.metadata.get("cutter_comp") if seg.metadata else None
        if comp:
            action = comp.get("action", "on")
            side = comp.get("side", "left")
            d_reg = comp.get("d_register", 0)
            t_meta = self.tools_info.get(self.current_tool_id) if hasattr(self, "tools_info") and self.tools_info else None
            if not t_meta and seg.metadata and seg.metadata.get("tool_info"):
                t_meta = seg.metadata["tool_info"]
            if action == "on" and not self.comp_active:
                lines.extend(self.cutter_compensation_on(side, d_reg, tool_meta=t_meta))
                self.comp_active = True
                self.comp_side = side
            elif action == "off" and self.comp_active:
                lines.extend(self.cutter_compensation_off())
                self.comp_active = False
                self.comp_side = None

        # Handle canned cycle detection for drilling

        if seg.metadata.get("canned_cycle") and self.supports_cycle(
                seg.metadata["canned_cycle"].get("type", "G83")):
            cycle = seg.metadata["canned_cycle"]
            cycle_type = cycle.get("type", "G83")
            x = cycle.get("x", seg.end[0])
            y = cycle.get("y", seg.end[1])
            z_r = cycle.get("z_r", seg.start[2])
            z_depth = cycle.get("z_depth", seg.end[2])
            peck = cycle.get("peck", 0.0)
            dwell = cycle.get("dwell", 0.0)
            feed = cycle.get("feed", self.current_feed or 500.0)

            if feed != self.current_feed:
                lines.append(f"F{self._fmt_f(feed)}")
                self.current_feed = feed

            # Emit audit preamble for drill depth and peck semantics if available
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

            if depth_ref and tip_len is not None and cyl_depth is not None:
                if not depth_type:
                    depth_type = "TIP_DEPTH" if depth_ref == "tip" else ("CYLINDRICAL_DEPTH" if depth_ref in ("shoulder", "full_diameter") else "THROUGH_BREAKTHROUGH")
                ref_label = "Tip Depth" if (depth_ref == "tip" or depth_type == "TIP_DEPTH") else ("Shoulder (Full Dia)" if (depth_ref in ("shoulder", "full_diameter") or depth_type == "CYLINDRICAL_DEPTH") else "Through Breakthrough")
                tip_travel = cycle.get("tip_travel")
                travel_str = f", Tip Travel={tip_travel:.3f}mm" if tip_travel is not None else ""
                target_str = f", Target={target_depth:.3f}mm" if target_depth is not None else ""
                web_str = f", Remaining Web: {rem_web:.3f}mm" if (rem_web is not None and rem_web > 0) else ""
                geom_str = f" [Point Angle: {pt_angle:.1f}deg, Geom: {tip_geom}]" if pt_angle is not None and tip_geom is not None else (f" [Point Angle: {pt_angle:.1f}deg]" if pt_angle is not None else "")
                lines.append(f"(--- Drill Depth Semantics: Ref={ref_label} (Datum: {depth_datum}, Type: {depth_type}), Cycle Z={z_depth:.3f}{target_str}, Tip Len={tip_len:.3f}mm{geom_str}, Cyl Depth={cyl_depth:.3f}mm{travel_str}{web_str} ---)")

            if peck_val and peck_val > 0 and (ld_ratio is not None or peck_ratio is not None):
                p_ratio_str = f" ({peck_ratio:.2f}xD)" if peck_ratio is not None else ""
                ld_str = f", L/D={ld_ratio:.2f}" if ld_ratio is not None else ""
                mat_str = f", Mat={mat_name}" if mat_name else ""
                lines.append(f"(--- Drill Peck Strategy: Q={peck_val:.3f}mm{p_ratio_str}, Cycle={cycle_type}{ld_str}{mat_str} ---)")

            retract_mode = cycle.get("retract_mode", "G98")
            retract_reason = cycle.get("retract_reason", "")
            if retract_mode:
                retract_desc = "Initial Plane" if retract_mode == "G98" else "R-Plane"
                r_str = f" [{retract_desc} Z{z_r:.3f}]" if z_r is not None else ""
                init_val = cycle.get("initial_z")
                init_str = f", Initial Z={init_val:.3f}" if (init_val is not None and retract_mode == "G98") else ""
                reason_str = f", Reason: {retract_reason}" if retract_reason else ""
                lines.append(f"(--- Drill Retract Strategy: Mode={retract_mode}{r_str}{init_str}{reason_str} ---)")

            if cycle_type == "G84":
                rpm = seg.spindle or self.current_spindle or 0
                if rpm > 0:
                    lines.append(f"M29 S{int(rpm)}")

            if cycle_type == "G83":
                lines.append(f"{retract_mode} G83 {self.format_coord('X', x)} {self.format_coord('Y', y)} "
                           f"{self.format_coord('Z', z_depth)} {self.format_coord('R', z_r)} "
                           f"{self.format_coord('Q', peck)} F{self._fmt_f(feed)}")
            elif cycle_type == "G73":
                lines.append(f"{retract_mode} G73 {self.format_coord('X', x)} {self.format_coord('Y', y)} "
                           f"{self.format_coord('Z', z_depth)} {self.format_coord('R', z_r)} "
                           f"{self.format_coord('Q', peck)} F{self._fmt_f(feed)}")
            elif cycle_type == "G81":
                lines.append(f"{retract_mode} G81 {self.format_coord('X', x)} {self.format_coord('Y', y)} "
                           f"{self.format_coord('Z', z_depth)} {self.format_coord('R', z_r)} "
                           f"F{self._fmt_f(feed)}")
            elif cycle_type == "G82":
                lines.append(f"{retract_mode} G82 {self.format_coord('X', x)} {self.format_coord('Y', y)} "
                           f"{self.format_coord('Z', z_depth)} {self.format_coord('R', z_r)} "
                           f"{self._fmt_cycle_dwell(dwell)} F{self._fmt_f(feed)}")
            elif cycle_type == "G85":
                lines.append(f"{retract_mode} G85 {self.format_coord('X', x)} {self.format_coord('Y', y)} "
                           f"{self.format_coord('Z', z_depth)} {self.format_coord('R', z_r)} "
                           f"F{self._fmt_f(feed)}")
            elif cycle_type == "G86":
                lines.append(f"{retract_mode} G86 {self.format_coord('X', x)} {self.format_coord('Y', y)} "
                           f"{self.format_coord('Z', z_depth)} {self.format_coord('R', z_r)} "
                           f"F{self._fmt_f(feed)}")
            elif cycle_type == "G76":
                shift = cycle.get("shift", 0.0)
                lines.append(f"{retract_mode} G76 {self.format_coord('X', x)} {self.format_coord('Y', y)} "
                           f"{self.format_coord('Z', z_depth)} {self.format_coord('R', z_r)} "
                           f"{self.format_coord('Q', shift)} F{self._fmt_f(feed)}")
            else:
                lines.append(f"{retract_mode} {cycle_type} {self.format_coord('X', x)} "
                           f"{self.format_coord('Y', y)} {self.format_coord('Z', z_depth)} "
                           f"{self.format_coord('R', z_r)} F{self._fmt_f(feed)}")

            self.current_pos = np.array([x, y, z_r])
            self.canned_cycle_active = True
            self._cycle_op_id = seg.operation_id
            self._cycle_r_plane = z_r
            self.is_pos_established = True
            return lines

        if self.canned_cycle_active and self._cycle_op_id == seg.operation_id and \
                seg.motion_type in (MotionType.PLUNGE, MotionType.DWELL):
            return lines

        # Handle Rapid motions
        if seg.motion_type in (MotionType.RAPID, MotionType.LINK, MotionType.RETRACT):
            if self.canned_cycle_active and self._cycle_op_id == seg.operation_id and \
                    seg.motion_type in (MotionType.RETRACT, MotionType.RAPID) and \
                    seg.end[2] <= self._cycle_r_plane + 1e-6:
                return lines
            if self.canned_cycle_active:
                lines.extend(self.cycle_cancel())
                self.canned_cycle_active = False
                self._cycle_op_id = None
            # RETRACT with a feed rate means feed-out (reaming/tapping), emit as G1
            if seg.motion_type is MotionType.RETRACT and seg.feed is not None and seg.feed > 0:
                feed = seg.feed
                self.cut_dist += dist
                self.cut_time += (dist / max(feed, 10.0)) * 60.0
                feed_cmd = ""
                if feed != self.current_feed:
                    feed_cmd = f" F{self._fmt_f(feed)}"
                    self.current_feed = feed
                cmd = self._motion_line("G1", seg.end, extra=feed_cmd)
                if cmd:
                    lines.append(cmd)
                self.current_pos = seg.end.copy()
                return lines
            self.rapid_dist += dist
            self.rapid_time += (dist / max(self.default_rapid_feed, 100.0)) * 60.0
            cmd = self._motion_line("G0", seg.end)
            if cmd:
                lines.append(cmd)
            self.current_pos = seg.end.copy()
            return lines

        # Handle linear cutting moves (HELIX handled separately as G2/G3)
        if seg.motion_type in (MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
                               MotionType.ENTRY, MotionType.EXIT):
            if self.canned_cycle_active:
                lines.extend(self.cycle_cancel())
                self.canned_cycle_active = False
                self._cycle_op_id = None
            feed = seg.feed or self.current_feed or 500.0
            self.cut_dist += dist
            self.cut_time += (dist / max(feed, 10.0)) * 60.0
            feed_cmd = ""
            if feed != self.current_feed:
                feed_cmd = f" F{self._fmt_f(feed)}"
                self.current_feed = feed

            cmd = self._motion_line("G1", seg.end, extra=feed_cmd)
            if cmd:
                lines.append(cmd)
            self.current_pos = seg.end.copy()
            return lines

        # Handle circular and helical interpolation
        if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX):
            return self._emit_arc(seg, dist, is_helix=(seg.motion_type == MotionType.HELIX))

        return lines

    def _emit_arc(self, seg: MotionSegment, dist: float, is_helix: bool) -> list[str]:
        """Mathematically validate arc geometry (start, end, center, radii, sweep, plane) before emission.

        Requires |R_start - R_end| <= arc_interpolation_tolerance on post-formatted controller values.
        If valid: emits G2/G3 with precise I/J offsets.
        If invalid: safely linearizes the arc into G1 moves with chord error <= arc_linearization_chord_tolerance.
        """
        lines = []
        if self.canned_cycle_active:
            lines.extend(self.cycle_cancel())
            self.canned_cycle_active = False
            self._cycle_op_id = None

        start_pos = self.current_pos if self.is_pos_established else seg.start
        end_pos = seg.end
        is_ccw = seg.arc_ccw if is_helix else (seg.motion_type == MotionType.ARC_CCW or seg.arc_ccw)
        code = "G3" if is_ccw else "G2"

        feed = seg.feed or self.current_feed or 500.0
        self.cut_dist += dist
        self.cut_time += (dist / max(feed, 10.0)) * 60.0

        feed_cmd = ""
        if feed != self.current_feed:
            feed_cmd = f" F{self._fmt_f(feed)}"
            self.current_feed = feed

        # 1. Compute and optimize I/J offsets
        i_off, j_off = self._format_arc_ij(start_pos, end_pos, seg.arc_center, is_ccw)

        # 2. Evaluate post-formatted / rounded NC coordinates
        p = self.precision
        xs = round(float(start_pos[0]), p)
        ys = round(float(start_pos[1]), p)
        xe = round(float(end_pos[0]), p)
        ye = round(float(end_pos[1]), p)
        i_nc = round(float(i_off), p)
        j_nc = round(float(j_off), p)

        cx_nc = xs + i_nc
        cy_nc = ys + j_nc

        r_start_nc = math.hypot(i_nc, j_nc)
        r_end_nc = math.hypot(xe - cx_nc, ye - cy_nc)
        delta_r_nc = abs(r_start_nc - r_end_nc)
        chord_nc = math.hypot(xe - xs, ye - ys)

        # Vector angles
        ux = xs - cx_nc
        uy = ys - cy_nc
        vx = xe - cx_nc
        vy = ye - cy_nc
        a0 = math.atan2(uy, ux)
        a1 = math.atan2(vy, vx)

        if is_ccw:
            sweep = (a1 - a0) % (2.0 * math.pi)
        else:
            sweep = (a0 - a1) % (2.0 * math.pi)

        if chord_nc < 1e-5 or sweep < 1e-5:
            sweep = 2.0 * math.pi

        # Mathematical Validation Check
        is_valid_arc = (
            r_start_nc > 1e-4 and
            delta_r_nc <= self.arc_interpolation_tolerance and
            chord_nc <= 2.0 * r_start_nc + self.arc_interpolation_tolerance + 1e-4 and
            0.0 < sweep <= 2.0 * math.pi + 1e-4
        )

        if is_valid_arc:
            ij_str = f" I{self._fmt(i_nc)} J{self._fmt(j_nc)}"
            cmd = self._motion_line(code, seg.end, extra=f"{ij_str}{feed_cmd}")
            if cmd:
                lines.append(cmd)
            self.current_pos = seg.end.copy()
            return lines

        # Remediate / Fallback: Safely validated linearization
        lines.append(
            f"(--- Post Arc Linearized: dR={delta_r_nc:.4f}mm exceeds tol={self.arc_interpolation_tolerance:.4f}mm, sweep={math.degrees(sweep):.1f}deg ---)"
        )
        r_avg = max(r_start_nc, 0.1)
        chord_tol = max(self.arc_linearization_chord_tolerance, 1e-4)
        cos_val = max(-1.0, min(1.0, 1.0 - chord_tol / r_avg))
        max_dtheta = 2.0 * math.acos(cos_val)
        if max_dtheta < 1e-4:
            max_dtheta = math.radians(5.0)

        n_steps = max(2, int(math.ceil(sweep / max_dtheta)))
        for k in range(1, n_steps + 1):
            t = k / n_steps
            ang = a0 + t * (sweep if is_ccw else -sweep)
            if k == n_steps:
                p_k = seg.end.copy()
            else:
                xk = cx_nc + r_start_nc * math.cos(ang)
                yk = cy_nc + r_start_nc * math.sin(ang)
                zk = start_pos[2] + t * (end_pos[2] - start_pos[2])
                p_k = np.array([xk, yk, zk])

            cmd = self._motion_line("G1", p_k, extra=feed_cmd if k == 1 else "")
            if cmd:
                lines.append(cmd)
            self.current_pos = p_k.copy()

        return lines

    def _motion_line(self, mode: str, target: np.ndarray, extra: str = "") -> Optional[str]:
        parts = []
        if mode != self.current_motion_mode or not self.is_pos_established:
            parts.append(mode)
            self.current_motion_mode = mode

        # Establish complete positioning state on operation start
        if not self.is_pos_established:
            parts.append(self.format_coord("X", target[0]))
            parts.append(self.format_coord("Y", target[1]))
            parts.append(self.format_coord("Z", target[2]))
            self.is_pos_established = True
        else:
            if abs(target[0] - self.current_pos[0]) > 1e-4:
                parts.append(self.format_coord("X", target[0]))
            if abs(target[1] - self.current_pos[1]) > 1e-4:
                parts.append(self.format_coord("Y", target[1]))
            if abs(target[2] - self.current_pos[2]) > 1e-4:
                parts.append(self.format_coord("Z", target[2]))

        if not parts and not extra:
            return None
        if not parts and extra:
            parts.append(mode)

        res = " ".join(parts)
        if extra:
            res += extra
        return res

    @staticmethod
    def _project_center_to_bisector(start: np.ndarray, end: np.ndarray,
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

    def _format_arc_ij(self, start: np.ndarray, end: np.ndarray,
                       center: Optional[np.ndarray], ccw: bool) -> tuple[float, float]:
        """Compute and optimize I/J offsets so post-formatting radius delta is minimized."""
        if center is not None:
            proj_center = self._project_center_to_bisector(start, end, center)
            i_raw = float(proj_center[0] - start[0])
            j_raw = float(proj_center[1] - start[1])
        else:
            i_raw, j_raw = self._compute_arc_center_offset(start, end, ccw)

        # Optimize for rounding precision
        p = self.precision
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

    def _compute_arc_center_offset(self, start: np.ndarray, end: np.ndarray,
                                    ccw: bool) -> tuple[float, float]:
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

    def post_process(
            self,
            toolpaths: list[Toolpath],
            program_name: Union[str, list[Any]] = "VEXCAM",
            program_number: Optional[Union[int, str]] = None,
            setup_id: Optional[str] = None,
            setup_index: Optional[int] = None,
            job_id: Optional[str] = None,
            validate_geometry: bool = True,
            validate_syntax: bool = True,
            strict_validation: bool = False,
            require_validity: bool = True,
            tools: Optional[list[Any]] = None,
        ) -> PostResult:
            """Translate a sequence of Toolpath objects into G-code.

            With ``require_validity`` (the default) every operation must have
            passed its validity checks first; a single critical failure raises and
            no G-code is emitted.
            """
            if isinstance(program_name, (list, tuple)):
                if tools is None:
                    tools = list(program_name)
                program_name = "VEXCAM"

            # --- operation validity gate: no questionable G-code ---------------
            if require_validity:
                assert_valid_for_post(
                    toolpaths,
                    capability=getattr(self, "capability_model", None),
                )

            self._program_name = str(program_name)
            # Reset renderer modal state for fresh program
            self.current_pos = np.array([0.0, 0.0, 0.0])
            self.current_feed = None
            self.current_spindle = None
            self.current_motion_mode = None
            self.spindle_active = False
            self.coolant_active = False
            self.current_tool_id = None
            self.canned_cycle_active = False
            self._cycle_op_id = None
            self._cycle_r_plane = 0.0
            self.is_pos_established = False
            options = self._build_options()
            # Override options with explicit parameters (fallback to instance attributes)
            options.program_number = program_number if program_number is not None else self.program_number
            options.setup_id = setup_id
            options.setup_index = setup_index
            options.job_id = job_id

            program = build_canonical_program(toolpaths, options, tools)
            output_lines = self.translate(program)
            full_gcode = "\n".join(output_lines)

            geo_report: Optional[NCGeometryValidationReport] = None
            if validate_geometry and toolpaths:
                geo_validator = GCodeGeometryValidator()
                geo_report = geo_validator.validate_toolpaths_vs_gcode(
                    gcode=full_gcode,
                    toolpaths=toolpaths,
                    strict=strict_validation,
                )

            syntax_report: Optional[ControllerSyntaxValidationReport] = None
            if validate_syntax:
                syntax_validator = ControllerSyntaxValidator(self.capability_model)
                syntax_report = syntax_validator.validate(full_gcode)
                if strict_validation and not syntax_report.is_valid:
                    err_msg = "; ".join(syntax_report.errors)
                    raise ValueError(f"Controller syntax validation failed for '{self.capability_model.name}': {err_msg}")

            # --- Final Validation Gate: G-code vs Canonical Toolpath ---
            final_validation_report: Optional[FinalValidationResult] = None
            if validate_geometry and toolpaths:
                # Run final validation gate comparing canonical toolpaths with generated G-code
                final_gate = FinalValidationGate(
                    position_tolerance_mm=0.05,
                    arc_center_tolerance_mm=0.01,
                    feed_tolerance_pct=5.0,
                    spindle_tolerance_rpm=100.0,
                    require_full_match=True,
                )
                final_validation_report = final_gate.validate(
                    canonical_toolpaths=toolpaths,
                    gcode=full_gcode,
                    machine_profile=self.machine_profile,
                )
                # In strict mode, fail the post-process if final validation fails
                if strict_validation and not final_validation_report.passed:
                    err_msg = "; ".join(final_validation_report.errors)
                    raise ValueError(
                        f"Final validation gate failed for '{self.capability_model.name}': {err_msg}"
                    )

            return PostResult(
                gcode=full_gcode,
                line_count=len(output_lines),
                rapid_time_seconds=program.stats.rapid_time,
                cut_time_seconds=program.stats.cut_time,
                total_time_seconds=program.stats.rapid_time + program.stats.cut_time,
                total_rapid_dist_mm=program.stats.rapid_dist,
                total_cut_dist_mm=program.stats.cut_dist,
                tool_changes=program.stats.tool_changes,
                work_offset=self.work_offset,
                program_number=program.program_number,
                geometry_validation=geo_report,
                syntax_validation=syntax_report,
                final_validation=final_validation_report,
            )

    # ------------------------------------------------------------------ translate --
    def translate(self, program: CanonicalProgram) -> list[str]:
        """Render a CanonicalProgram into G-code lines."""
        lines: list[str] = []
        lines.extend(self.header(program.program_name, tools_info=program.tools_info, program_number=program.program_number))
        for op in program.operations:
            lines.extend(self._render_operation(op))
        lines.extend(self._render_footer(program.footer_commands))
        return lines

    def _render_operation(self, op: CanonicalToolpath) -> list[str]:
        lines: list[str] = []
        self.is_pos_established = False
        self.current_motion_mode = None
        for cmd in op.commands:
            lines.extend(self._render_command(cmd))
        return lines

    def _render_footer(self, commands: Sequence[CanonicalCommand]) -> list[str]:
        lines: list[str] = []
        for cmd in commands:
            lines.extend(self._render_command(cmd))
        return lines

    def _render_command(self, cmd: CanonicalCommand) -> list[str]:
        if isinstance(cmd, Comment):
            return self._render_comment(cmd)
        if isinstance(cmd, BlankLine):
            return self._render_blank_line(cmd)
        if isinstance(cmd, ToolChange):
            return self._render_tool_change(cmd)
        if isinstance(cmd, SafetyBlock):
            return self._render_safety_block(cmd)
        if isinstance(cmd, CompOn):
            return self._render_comp_on(cmd)
        if isinstance(cmd, CompOff):
            return self._render_comp_off(cmd)
        if isinstance(cmd, Rapid):
            return self._render_rapid(cmd)
        if isinstance(cmd, Linear):
            return self._render_linear(cmd)
        if isinstance(cmd, Arc):
            return self._render_arc(cmd)
        if isinstance(cmd, Spindle):
            return self._render_spindle(cmd)
        if isinstance(cmd, SpindleStop):
            return self._render_spindle_stop(cmd)
        if isinstance(cmd, Coolant):
            return self._render_coolant(cmd)
        if isinstance(cmd, CoolantOff):
            return self._render_coolant_off(cmd)
        if isinstance(cmd, Dwell):
            return self._render_dwell(cmd)
        if isinstance(cmd, Cycle):
            return self._render_cycle(cmd)
        if isinstance(cmd, CycleCancel):
            return self._render_cycle_cancel(cmd)
        if isinstance(cmd, MachineCommand):
            return self._render_machine_command(cmd)
        if isinstance(cmd, SafeReturn):
            return self._render_safe_return(cmd)
        if isinstance(cmd, ProgramEnd):
            return self._render_program_end(cmd)
        return []

    # Individual renderers - base ISO implementations
    def _render_comment(self, cmd: Comment) -> list[str]:
        # Heidenhain conversational skips "motion" kind comments (arc linearization notes)
        if getattr(self, "conversational", False) and cmd.kind == "motion":
            return []
        return [f"({cmd.text})"]

    def _render_blank_line(self, cmd: BlankLine) -> list[str]:
        return [""]

    def _render_tool_change(self, cmd: ToolChange) -> list[str]:
        lines = []
        lines.extend(self._render_tool_change_command(cmd.tool_number, cmd.description, cmd.length_offset_h))
        if self.enable_tool_length_comp:
            lines.append(f"{self.machine_profile.tool_length_comp_default} H{cmd.length_offset_h}")
        if cmd.spindle_rpm is not None:
            rpm_int = int(round(cmd.spindle_rpm))
            lines.append(f"S{rpm_int} M3")
            self.current_spindle = cmd.spindle_rpm
            self.spindle_active = True
        return lines

    def _render_tool_change_command(self, tool_num: int, desc: str, h_offset: int) -> list[str]:
        if self.tool_change_format:
            try:
                return [self.tool_change_format.format(
                    tool_num=tool_num,
                    h_offset=h_offset,
                    desc=desc,
                    tool_num_2d=f"{tool_num:02d}",
                    h_offset_2d=f"{h_offset:02d}",
                )]
            except Exception:
                pass
        strat = (self.tool_change_strategy or "atc").lower()
        cmd_code = self.tool_change_command or self.machine_profile.tool_change_command
        if strat in ("manual", "m0", "stop"):
            return [f"; Tool change: T{tool_num:02d} ({desc})", f"T{tool_num} {cmd_code} (Change to T{tool_num:02d} - {desc})"]
        elif strat in ("comment_only", "none"):
            comment_char = ";" if (self.capability_model and self.capability_model.block_format.comment_style == "semicolon") else ""
            if comment_char == ";":
                return [f"; T{tool_num:02d} - {desc}"]
            return [f"(T{tool_num:02d} - {desc})"]
        else:
            return [f"T{tool_num} {cmd_code} (Tool: T{tool_num:02d} - {desc})"]

    def _render_safety_block(self, cmd: SafetyBlock) -> list[str]:
        # Delegate to subclass's operation_safety_preamble for controller-specific codes
        return self.operation_safety_preamble(None, cmd.tool_number, cmd.h_offset, cmd.work_offset)

    def _render_comp_on(self, cmd: CompOn) -> list[str]:
        g = "G41" if cmd.side == "left" else "G42"
        return [f"{g} D{cmd.d_register}"]

    def _render_comp_off(self, cmd: CompOff) -> list[str]:
        return ["G40"]

    def _render_rapid(self, cmd: Rapid) -> list[str]:
        return self._render_motion("G0", cmd.target, cmd.axes)

    def _render_linear(self, cmd: Linear) -> list[str]:
        extra = ""
        if cmd.feed != self.current_feed:
            extra = f" F{self._fmt_f(cmd.feed)}"
            self.current_feed = cmd.feed
        return self._render_motion("G1", cmd.target, None, extra)

    def _render_arc(self, cmd: Arc) -> list[str]:
        extra = f" I{self._fmt(cmd.i_offset)} J{self._fmt(cmd.j_offset)}"
        if cmd.feed != self.current_feed:
            extra += f" F{self._fmt_f(cmd.feed)}"
            self.current_feed = cmd.feed
        code = "G3" if cmd.ccw else "G2"
        return self._render_motion(code, cmd.target, None, extra)

    def _render_spindle(self, cmd: Spindle) -> list[str]:
        # Heidenhain conversational skips operation-level spindle (already in tool call)
        if getattr(self, "conversational", False) and cmd.source == "operation":
            return []
        self.current_spindle = cmd.rpm
        self.spindle_active = True
        return [f"S{int(round(cmd.rpm))} M3"]

    def _render_spindle_stop(self, cmd: SpindleStop) -> list[str]:
        self.spindle_active = False
        return ["M5"]

    def _render_coolant(self, cmd: Coolant) -> list[str]:
        self.coolant_active = True
        if cmd.kind == "mist":
            return ["M7"]
        elif cmd.kind == "through_spindle":
            return ["M50"]
        return ["M8"]

    def _render_coolant_off(self, cmd: CoolantOff) -> list[str]:
        self.coolant_active = False
        return ["M9"]

    def _render_dwell(self, cmd: Dwell) -> list[str]:
        self.cut_time += cmd.seconds
        return [f"G4 P{self._fmt(cmd.seconds)}"]

    def _render_cycle(self, cmd: Cycle) -> list[str]:
        lines = []
        # Standalone feed line if changed (ISO style)
        if cmd.feed != self.current_feed:
            lines.append(f"F{self._fmt_f(cmd.feed)}")
            self.current_feed = cmd.feed
        # Audit comments
        for comment_text in cmd.comments:
            lines.append(f"({comment_text})")
        # G84 tapping: M29
        if cmd.spindle_rpm and cmd.spindle_rpm > 0:
            lines.append(f"M29 S{int(round(cmd.spindle_rpm))}")
        # Cycle line
        cycle_line = f"{cmd.retract_mode} {cmd.cycle_type} "
        cycle_line += f"{self.format_coord('X', cmd.x)} {self.format_coord('Y', cmd.y)} "
        cycle_line += f"{self.format_coord('Z', cmd.depth_z)} {self.format_coord('R', cmd.r_plane)}"
        if cmd.cycle_type in ("G83", "G73") and cmd.peck > 0:
            cycle_line += f" {self.format_coord('Q', cmd.peck)}"
        elif cmd.cycle_type == "G76" and cmd.shift != 0:
            cycle_line += f" {self.format_coord('Q', cmd.shift)}"
        elif cmd.cycle_type == "G82" and cmd.dwell > 0:
            cycle_line += f" {self._fmt_cycle_dwell(cmd.dwell)}"
        cycle_line += f" F{self._fmt_f(cmd.feed)}"
        lines.append(cycle_line)
        # Update state
        self.current_pos = np.array([cmd.x, cmd.y, cmd.r_plane])
        self.canned_cycle_active = True
        self._cycle_op_id = ""  # operation_id not directly available
        self._cycle_r_plane = cmd.r_plane
        self.is_pos_established = True
        return lines

    def _render_cycle_cancel(self, cmd: CycleCancel) -> list[str]:
        self.canned_cycle_active = False
        self._cycle_op_id = None
        return ["G80"]

    def _render_machine_command(self, cmd: MachineCommand) -> list[str]:
        if cmd.name == "hsm_mode":
            mode = dict(cmd.params).get("mode", "")
            if self.hsm_enabled and mode:
                pattern = self.machine_profile.lookahead_on_pattern
                if "{mode}" in pattern:
                    return [pattern.format(mode=mode)]
                elif "{tol}" in pattern:
                    tol = 0.05 if mode == "roughing" else (0.02 if mode == "medium" else 0.005)
                    mode_code = 1 if mode == "roughing" else (2 if mode == "medium" else 3)
                    return [pattern.format(tol=tol, mode=mode_code)]
                return [pattern]
        elif cmd.name == "hsm_reset":
            if self.hsm_enabled:
                pattern = self.machine_profile.lookahead_off_pattern
                return [pattern]
        elif cmd.name == "tool_stage":
            tool_num = dict(cmd.params).get("tool_number", 0)
            return [f"T{tool_num} (Stage next tool in carousel)"]
        elif cmd.name == "clear_offsets":
            return [self.machine_profile.clear_offsets_command]
        return []

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        if cmd.axis == "z":
            if self.home_command_z:
                safe_z = self._get_safe_retract_z()
                return [self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}")]
            return [self.machine_profile.safety.z_retract_command.format(safe_z=cmd.value)]
        elif cmd.axis == "xy":
            if self.home_command_xy:
                return [self.home_command_xy]
            return [self.machine_profile.safety.xy_home_command]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        end_cmd = self.program_end_command or self.machine_profile.program_end_command
        return [end_cmd]

    # Motion helper with modal suppression
    def _render_motion(self, mode: str, target: np.ndarray, axes: Optional[tuple[str, ...]], extra: str = "") -> list[str]:
        parts = []
        if mode != self.current_motion_mode or not self.is_pos_established:
            parts.append(mode)
            self.current_motion_mode = mode
        if not self.is_pos_established:
            parts.append(self.format_coord("X", target[0]))
            parts.append(self.format_coord("Y", target[1]))
            parts.append(self.format_coord("Z", target[2]))
            self.is_pos_established = True
        else:
            # On subsequent moves, respect axes restriction
            if axes is None or "x" in axes:
                if abs(target[0] - self.current_pos[0]) > 1e-4:
                    parts.append(self.format_coord("X", target[0]))
            if axes is None or "y" in axes:
                if abs(target[1] - self.current_pos[1]) > 1e-4:
                    parts.append(self.format_coord("Y", target[1]))
            if axes is None or "z" in axes:
                if abs(target[2] - self.current_pos[2]) > 1e-4:
                    parts.append(self.format_coord("Z", target[2]))
        if not parts and not extra:
            return []
        if not parts and extra:
            parts.append(mode)
        res = " ".join(parts)
        if extra:
            res += extra
        self.current_pos = target.copy()
        return [res]

    def _arc_fitting(self, segments: list[MotionSegment]) -> list[MotionSegment]:
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

            arc_start = i
            pts = [seg.start.copy(), seg.end.copy()]
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                if next_seg.motion_type != MotionType.CUT:
                    break
                pts.append(next_seg.end.copy())
                if len(pts) >= 4:
                    fitted, center, ccw = self._fit_arc(pts)
                    if fitted:
                        j += 1
                        continue
                    else:
                        pts.pop()
                        break
                else:
                    j += 1

            if len(pts) >= 4:
                fitted, center, ccw = self._fit_arc(pts)
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

    def _fit_arc(self, points: list[np.ndarray]) -> tuple[bool, Optional[np.ndarray], bool]:
        if len(points) < 3:
            return False, None, False

        p0 = points[0]
        p_mid = points[len(points) // 2]
        p_end = points[-1]

        # Arc fitting is only valid within a flat plane (XY plane)
        for p in points:
            if abs(p[2] - p0[2]) > 1e-4:
                return False, None, False

        # Multi-point algebraic Kåsa circle fit across all points in the sequence
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

        # Project center onto perpendicular bisector of chord [p0, p_end]
        exact_center_3d = self._project_center_to_bisector(p0, p_end, init_center_3d)
        exact_center_2d = exact_center_3d[:2]

        r = float(np.linalg.norm(p0[:2] - exact_center_2d))
        if r < 1e-3 or r > 1e6:
            return False, None, False

        # Validate that all intermediate points lie strictly on this arc
        for p in points:
            dist = abs(float(np.linalg.norm(p[:2] - exact_center_2d)) - r)
            if dist > 0.002:  # strict tolerance for arc fitting
                return False, None, False

        v0 = p0[:2] - exact_center_2d
        vm = p_mid[:2] - exact_center_2d
        cross_z = v0[0] * vm[1] - v0[1] * vm[0]
        ccw = cross_z > 0

        return True, exact_center_3d, ccw


class GrblPostProcessor(BasePostProcessor):
    """GRBL-compatible post processor. Standard GRBL v1.1 does not support
    canned cycles (G81-G89) or G98/G99. All drilling must use explicit motion."""

    supported_cycles: set[str] = set()  # GRBL has no canned cycle support

    def __init__(self, tool_change_mode: str = "manual", **kwargs):
        cap_model = kwargs.pop("capability_model", None) or GRBL_CAPABILITY_MODEL
        super().__init__(tool_change_mode=tool_change_mode, capability_model=cap_model, **kwargs)
        self.hsm_enabled = False

    def _build_options_overrides(self) -> dict:
        return {
            "footer_include_comp_off": False,
            "footer_include_cycle_cancel": False,
            "tool_change_retract_policy": "when_established",
            "tool_change_retract_z": None,
            "tool_change_retract_fallback": 20.0,
        }

    def operation_safety_preamble(
        self,
        tp: Toolpath,
        tool_num: int,
        h_offset: int,
        wcs: str,
    ) -> list[str]:
        """GRBL deterministic modal safety line (G90 G94 G17 G21 G40)."""
        return [f"{wcs} G90 G17 G21 G94 G40"]

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        if program_number is not None:
            try:
                prog_num = int(str(program_number).strip().lstrip("Oo"))
            except ValueError:
                prog_num = 1001
        else:
            prog_num = resolve_program_number(program_name=program_name, explicit_number=self.program_number, base_number=self.base_program_number, used_numbers=self.used_program_numbers)
        lines = [
            f"; Program: {program_name} (O{prog_num:04d})",
            "; Generated by VexCAM",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="semicolon"))
        lines.extend([
            "G90 G94 G17 G21 G40 G49 G80",
            f"{self.work_offset}",
        ])
        return lines

    def tool_change(self, tool_id: str, tool_num: int = 1,
                    length_offset_h: Optional[int] = None,
                    tool_meta: Optional[dict] = None) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)
        h = self._resolve_length_offset(tool_id, actual_tool_num, length_offset_h, tool_meta)
        if self.current_tool_id != tool_id:
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            # Retract Z before tool change for safety
            if self.is_pos_established:
                safe_z = self._get_safe_retract_z()
                lines.append(f"G0 Z{self._fmt(safe_z)}")
            lines.extend(self._render_tool_change_command(actual_tool_num, desc, h))
            if self.enable_tool_length_comp:
                lines.append(f"G43 H{h}")
            self.current_tool_id = tool_id
            self.is_pos_established = False
            if tool_id not in self.used_tools:
                self.used_tools.append(tool_id)
        return lines

    def footer(self) -> list[str]:
        lines = []
        if self.canned_cycle_active:
            lines.append("G80")
            self.canned_cycle_active = False
            self._cycle_op_id = None
        if self.coolant_active:
            lines.append("M9")
            self.coolant_active = False
        if self.spindle_active:
            lines.append("M5")
            self.spindle_active = False
        safe_z = self._get_safe_retract_z()
        if self.home_command_z:
            lines.append(self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}"))
        else:
            lines.append(f"G0 Z{self._fmt(safe_z)}")
        if self.home_command_xy:
            lines.append(self.home_command_xy)
        else:
            lines.append("G0 X0 Y0")
        end_cmd = self.program_end_command or getattr(getattr(self.capability_model, "safe_return", None), "program_end_command", "M2")
        lines.append(end_cmd)
        return lines


class FanucPostProcessor(BasePostProcessor):
    """Fanuc 3-axis CNC mill post processor with line numbering and AICC lookahead."""

    def __init__(self, line_increment: int = 10, **kwargs):
        cap_model = kwargs.pop("capability_model", None) or FANUC_CAPABILITY_MODEL
        super().__init__(capability_model=cap_model, **kwargs)
        self.line_num = line_increment
        self.line_increment = line_increment

    def _build_options_overrides(self) -> dict:
        return {
            "footer_hsm_reset_when": "always",
            "footer_clear_offsets": True,
        }

    def operation_safety_preamble(
        self,
        tp: Toolpath,
        tool_num: int,
        h_offset: int,
        wcs: str,
    ) -> list[str]:
        """Fanuc deterministic modal safety line."""
        return [f"G00 G90 G17 G21 G94 G40 G80 {wcs}"]

    def process_segment(self, seg: MotionSegment) -> list[str]:
        """Override dwell to emit P in milliseconds for Fanuc."""
        if seg.motion_type == MotionType.DWELL:
            # Fanuc G4 expects P in milliseconds (integer)
            p_ms = int(round(seg.dwell_seconds * 1000.0))
            self.cut_time += seg.dwell_seconds
            return [f"G4 P{p_ms}"]
        return super().process_segment(seg)

    def _fmt_cycle_dwell(self, dwell_seconds: float) -> str:
        """Fanuc canned cycle dwell P in milliseconds."""
        p_ms = int(round(dwell_seconds * 1000.0))
        return f"P{p_ms}"

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        if program_number is not None:
            try:
                prog_num = int(str(program_number).strip().lstrip("Oo"))
            except ValueError:
                prog_num = 1001
        else:
            prog_num = resolve_program_number(program_name=program_name, explicit_number=self.program_number, base_number=self.base_program_number, used_numbers=self.used_program_numbers)
        lines = [
            "%",
            f"O{prog_num:04d} ({program_name})",
            "(FANUC 3-AXIS MILL POST)",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="parentheses"))
        lines.extend([
            "G00 G17 G21 G40 G49 G80 G90",
            f"{self.work_offset}",
        ])
        if self.hsm_enabled:
            lines.append("G05.1 Q1 (AICC Lookahead ON)")
        return lines

    def footer(self) -> list[str]:
        lines = []
        if self.canned_cycle_active:
            lines.append("G80")
            self.canned_cycle_active = False
            self._cycle_op_id = None
        if self.comp_active:
            lines.extend(self.cutter_compensation_off())
            self.comp_active = False
            self.comp_side = None
        if self.coolant_active:
            lines.append("M09")
            self.coolant_active = False
        if self.spindle_active:
            lines.append("M05")
            self.spindle_active = False
        if self.hsm_enabled:
            lines.append("G05.1 Q0 (AICC Lookahead OFF)")
        lines.append("G49 G40")
        if self.home_command_z:
            safe_z = self._get_safe_retract_z()
            lines.append(self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}"))
        else:
            lines.append("G91 G28 Z0.0")
        if self.home_command_xy:
            lines.append(self.home_command_xy)
        else:
            lines.append("G28 X0.0 Y0.0")
        end_cmd = self.program_end_command or getattr(getattr(self.capability_model, "safe_return", None), "program_end_command", "M30")
        lines.extend([
            end_cmd,
            "%",
        ])
        return lines

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        end_cmd = self.program_end_command or getattr(getattr(self.capability_model, "safe_return", None), "program_end_command", "M30")
        return [end_cmd, "%"]

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        if cmd.axis == "z":
            if self.home_command_z:
                safe_z = self._get_safe_retract_z()
                return [self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}")]
            return ["G91 G28 Z0.0"]
        elif cmd.axis == "xy":
            if self.home_command_xy:
                return [self.home_command_xy]
            return ["G28 X0.0 Y0.0"]
        return []


class HaasPostProcessor(BasePostProcessor):
    """Haas VF-series CNC mill post processor with G187 smoothing modes."""

    def __init__(self, **kwargs):
        cap_model = kwargs.pop("capability_model", None) or HAAS_CAPABILITY_MODEL
        super().__init__(capability_model=cap_model, **kwargs)

    def _build_options_overrides(self) -> dict:
        return {
            "footer_hsm_reset_when": "mode_set",
            "footer_clear_offsets": True,
        }

    def operation_safety_preamble(
        self,
        tp: Toolpath,
        tool_num: int,
        h_offset: int,
        wcs: str,
    ) -> list[str]:
        """Haas deterministic modal safety line."""
        return [f"G00 G90 G17 G21 G94 G40 G80 {wcs}"]

    def start_toolpath(self, tp: Toolpath) -> list[str]:
        if not self.hsm_enabled:
            return []
        mode = tp.metadata.get("hsm_mode") if tp.metadata else None
        if not mode:
            if tp.purpose in ("roughing", "facing", "rest_machining"):
                mode = "roughing"
            elif tp.purpose in ("semi_finishing", "chamfering", "countersinking", "grooving"):
                mode = "medium"
            else:
                mode = "finishing"
        if mode != self.current_hsm_mode:
            self.current_hsm_mode = mode
            p_code = 1 if mode == "roughing" else (2 if mode == "medium" else 3)
            return [f"G187 P{p_code} (Haas High-Speed Smoothing {mode.upper()})"]
        return []

    def process_segment(self, seg: MotionSegment) -> list[str]:
        """Override dwell to emit P in milliseconds for Haas."""
        if seg.motion_type == MotionType.DWELL:
            p_ms = int(round(seg.dwell_seconds * 1000.0))
            self.cut_time += seg.dwell_seconds
            return [f"G4 P{p_ms}"]
        return super().process_segment(seg)

    def _fmt_cycle_dwell(self, dwell_seconds: float) -> str:
        """Haas canned cycle dwell P in milliseconds."""
        p_ms = int(round(dwell_seconds * 1000.0))
        return f"P{p_ms}"

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        if program_number is not None:
            try:
                prog_num = int(str(program_number).strip().lstrip("Oo"))
            except ValueError:
                prog_num = 1001
        else:
            prog_num = resolve_program_number(program_name=program_name, explicit_number=self.program_number, base_number=self.base_program_number, used_numbers=self.used_program_numbers)
        lines = [
            "%",
            f"O{prog_num:05d} ({program_name})",
            "(HAAS CNC MILL)",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="parentheses"))
        lines.extend([
            "G00 G17 G21 G40 G49 G80 G90",
            f"{self.work_offset}",
        ])
        return lines

    def footer(self) -> list[str]:
        lines = []
        if self.canned_cycle_active:
            lines.append("G80")
            self.canned_cycle_active = False
            self._cycle_op_id = None
        if self.comp_active:
            lines.extend(self.cutter_compensation_off())
            self.comp_active = False
            self.comp_side = None
        if self.coolant_active:
            lines.append("M09")
            self.coolant_active = False
        if self.spindle_active:
            lines.append("M05")
            self.spindle_active = False
        if self.hsm_enabled and self.current_hsm_mode is not None:
            lines.append("G187 (Restore Haas default smoothing)")
        lines.append("G49 G40")
        if self.home_command_z:
            safe_z = self._get_safe_retract_z()
            lines.append(self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}"))
        else:
            lines.append("G53 G00 Z0.0")
        if self.home_command_xy:
            lines.append(self.home_command_xy)
        else:
            lines.append("G53 G00 X0.0 Y0.0")
        end_cmd = self.program_end_command or getattr(getattr(self.capability_model, "safe_return", None), "program_end_command", "M30")
        lines.extend([
            end_cmd,
            "%",
        ])
        return lines

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        end_cmd = self.program_end_command or getattr(getattr(self.capability_model, "safe_return", None), "program_end_command", "M30")
        return [end_cmd, "%"]

    def _render_machine_command(self, cmd: MachineCommand) -> list[str]:
        if cmd.name == "hsm_mode":
            mode = dict(cmd.params).get("mode", "")
            if self.hsm_enabled and mode:
                p_code = 1 if mode == "roughing" else (2 if mode == "medium" else 3)
                return [f"G187 P{p_code} (Haas High-Speed Smoothing {mode.upper()})"]
        elif cmd.name == "hsm_reset":
            if self.hsm_enabled:
                return ["G187 P0 (Haas Smoothing OFF)"]
        elif cmd.name == "tool_stage":
            tool_num = dict(cmd.params).get("tool_number", 0)
            return [f"T{tool_num} (Stage next tool in carousel)"]
        elif cmd.name == "clear_offsets":
            return ["G49 G40"]
        return []

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        if cmd.axis == "z":
            if self.home_command_z:
                safe_z = self._get_safe_retract_z()
                return [self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}")]
            return ["G53 G00 Z0.0"]
        elif cmd.axis == "xy":
            if self.home_command_xy:
                return [self.home_command_xy]
            return ["G53 G00 X0.0 Y0.0"]
        return []


class LinuxCncPostProcessor(BasePostProcessor):
    """LinuxCNC / AXIS post processor (strict RS274/NGC)."""

    def __init__(self, **kwargs):
        cap_model = kwargs.pop("capability_model", None) or LINUXCNC_CAPABILITY_MODEL
        super().__init__(capability_model=cap_model, **kwargs)
        self.hsm_enabled = False

    def _build_options_overrides(self) -> dict:
        return {
            "safe_retract_fallback": 25.0,
        }

    def operation_safety_preamble(
        self,
        tp: Toolpath,
        tool_num: int,
        h_offset: int,
        wcs: str,
    ) -> list[str]:
        """LinuxCNC deterministic modal safety line."""
        return [f"G90 G94 G17 G21 G40 G49 G80 {wcs}"]

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        if program_number is not None:
            try:
                prog_num = int(str(program_number).strip().lstrip("Oo"))
            except ValueError:
                prog_num = 1001
        else:
            prog_num = resolve_program_number(program_name=program_name, explicit_number=self.program_number, base_number=self.base_program_number, used_numbers=self.used_program_numbers)
        lines = [
            f"(LinuxCNC Program: {program_name} - O{prog_num:04d})",
            "G21 (Units in mm)",
            "G90 (Absolute distance mode)",
            "G94 (Units per minute feed rate)",
            "G17 (XY Plane)",
            f"{self.work_offset}",
            "G64 P0.01 Q0.005 (Naive CAM tolerance / path blending)",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="parentheses"))
        return lines

    def footer(self) -> list[str]:
        lines = []
        if self.canned_cycle_active:
            lines.append("G80")
            self.canned_cycle_active = False
            self._cycle_op_id = None
        if self.comp_active:
            lines.extend(self.cutter_compensation_off())
            self.comp_active = False
            self.comp_side = None
        if self.coolant_active:
            lines.append("M9")
            self.coolant_active = False
        if self.spindle_active:
            lines.append("M5")
            self.spindle_active = False
        safe_z = self._get_safe_retract_z(fallback=25.0)
        lines.append("G49 G40")
        if self.home_command_z:
            lines.append(self.home_command_z.format(safe_z=f"{safe_z:.{self.precision}f}"))
        else:
            lines.append(f"G0 Z{self._fmt(safe_z)}")
        if self.home_command_xy:
            lines.append(self.home_command_xy)
        end_cmd = self.program_end_command or getattr(getattr(self.capability_model, "safe_return", None), "program_end_command", "M2")
        lines.append(end_cmd)
        return lines


class HeidenhainPostProcessor(BasePostProcessor):
    """Heidenhain CNC milling post processor supporting Conversational Klartext and ISO mode."""

    def __init__(self, conversational: bool = True, datum_number: int = 1, **kwargs):
        cap_model = kwargs.pop("capability_model", None) or HEIDENHAIN_CAPABILITY_MODEL
        super().__init__(capability_model=cap_model, **kwargs)
        self.conversational = conversational
        self.datum_number = datum_number
        self.hsm_enabled = True

    def _build_options_overrides(self) -> dict:
        return {
            "footer_hsm_reset_when": "never",  # Heidenhain uses custom footer
            "footer_include_cycle_cancel": False,
            "footer_include_comp_off": False,
            "tool_change_spindle_restart": True,
            "tool_change_rpm_rule": "spindle_or_tool",
            "tool_change_retract_policy": "always",
            "tool_change_retract_z": None,
            "tool_change_retract_fallback": 50.0,
            "safe_retract_fallback": 50.0,
        }

    def _fmt_hnc(self, val: float) -> str:
        """Format signed coordinate for Heidenhain Klartext (e.g. +10.000 or -5.500)."""
        rounded = round(val, self.precision)
        if abs(rounded) < 1e-9:
            sign = "+"
            s = f"{0.0:.{self.precision}f}"
        elif rounded > 0:
            sign = "+"
            s = f"{rounded:.{self.precision}f}"
        else:
            sign = "-"
            s = f"{abs(rounded):.{self.precision}f}"
        s_clean = s.rstrip("0").rstrip(".") if "." in s else s
        return f"{sign}{s_clean}"

    def format_coord(self, axis: str, val: float) -> str:
        if self.conversational:
            return f"{axis}{self._fmt_hnc(val)}"
        return f"{axis}{self._fmt(val)}"

    def operation_safety_preamble(
        self,
        tp: Toolpath,
        tool_num: int,
        h_offset: int,
        wcs: str,
    ) -> list[str]:
        if not self.conversational:
            return [f"{wcs} G90 G17 G21 G94 G40 G80"]
        return [f"; --- Setup Datum: {wcs} ---"]

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", program_name).upper()
        if not clean_name:
            clean_name = "VEXCAM"

        if not self.conversational:
            # ISO mode
            lines = [
                f"% {clean_name} G71 *",
                f"(Program: {clean_name})",
            ]
            if tools_info:
                lines.extend(self._format_tools_header_block(tools_info, style="parentheses"))
            lines.extend([
                "G00 G17 G21 G40 G49 G80 G90",
                f"{self.work_offset}",
            ])
            return lines

        # Conversational Klartext mode
        lines = [
            f"BEGIN PGM {clean_name} MM",
            f"; Program: {clean_name}",
            "; Generated by VexCAM Post-Processor",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="semicolon"))

        # Datum setting (Cycle 247)
        lines.extend([
            f"CYCL DEF 247 DATUM SETTING ~",
            f"  Q339={self.datum_number} ;DATUM NUMBER",
        ])

        if self.hsm_enabled:
            lines.extend([
                "CYCL DEF 32.0 TOLERANCE",
                "CYCL DEF 32.1 T0.01",
            ])
        return lines

    def cycle_cancel(self) -> list[str]:
        return []

    def tool_change(
        self,
        tool_id: str,
        tool_num: int = 1,
        length_offset_h: Optional[int] = None,
        tool_meta: Optional[dict] = None,
    ) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)
        h = self._resolve_length_offset(tool_id, actual_tool_num, length_offset_h, tool_meta)

        if self.current_tool_id != tool_id:
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False

            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.conversational:
                lines.append(f"L Z{self._fmt_hnc(safe_z)} R0 FMAX")
            else:
                lines.append(f"G0 Z{self._fmt(safe_z)}")

            rpm = 3000.0
            if self.current_spindle and self.current_spindle > 0:
                rpm = self.current_spindle
            elif tool_meta and tool_meta.get("spindle_rpm"):
                rpm = float(tool_meta["spindle_rpm"])
            elif tool_meta and isinstance(tool_meta.get("cutting_parameters"), dict) and tool_meta["cutting_parameters"].get("spindle_rpm"):
                rpm = float(tool_meta["cutting_parameters"]["spindle_rpm"])

            rpm_int = int(round(rpm))
            if self.conversational:
                lines.append(f"TOOL CALL {actual_tool_num} Z S{rpm_int} (Tool: T{actual_tool_num:02d} - {desc})")
                lines.append("M3")
            else:
                lines.extend(self._render_tool_change_command(actual_tool_num, desc, h))
                if self.enable_tool_length_comp:
                    lines.append(f"G43 H{h}")
                lines.append(f"S{rpm_int} M3")

            self.spindle_active = True
            self.current_tool_id = tool_id
            self.is_pos_established = False
            if tool_id not in self.used_tools:
                self.used_tools.append(tool_id)
        return lines

    def cutter_compensation_on(self, side: str = "left", d_register: int = 0, tool_meta: Optional[dict] = None) -> list[str]:
        if not self.conversational:
            return super().cutter_compensation_on(side, d_register, tool_meta)
        self.comp_active = True
        self.comp_side = "RL" if side == "left" else "RR"
        return []

    def cutter_compensation_off(self) -> list[str]:
        if not self.conversational:
            return super().cutter_compensation_off()
        self.comp_active = False
        self.comp_side = None
        return []

    def _motion_line(self, mode: str, target: np.ndarray, extra: str = "") -> Optional[str]:
        if not self.conversational:
            return super()._motion_line(mode, target, extra)

        comp_str = "R0"
        if self.comp_active and self.comp_side:
            side_s = str(self.comp_side).lower()
            comp_str = "RL" if side_s in ("rl", "left") else ("RR" if side_s in ("rr", "right") else "R0")

        if mode == "G0":
            # Rapid in Klartext: L X+... Y+... R0 FMAX
            coords = []
            if not self.is_pos_established:
                coords.append(self.format_coord("X", target[0]))
                coords.append(self.format_coord("Y", target[1]))
                coords.append(self.format_coord("Z", target[2]))
                self.is_pos_established = True
            else:
                if abs(target[0] - self.current_pos[0]) > 1e-6:
                    coords.append(self.format_coord("X", target[0]))
                if abs(target[1] - self.current_pos[1]) > 1e-6:
                    coords.append(self.format_coord("Y", target[1]))
                if abs(target[2] - self.current_pos[2]) > 1e-6:
                    coords.append(self.format_coord("Z", target[2]))

            if not coords:
                return None
            return f"L {' '.join(coords)} {comp_str} FMAX"

        elif mode == "G1":
            # Feed in Klartext: L X+... Y+... RL/RR/R0 F...
            coords = []
            if not self.is_pos_established:
                coords.append(self.format_coord("X", target[0]))
                coords.append(self.format_coord("Y", target[1]))
                coords.append(self.format_coord("Z", target[2]))
                self.is_pos_established = True
            else:
                if abs(target[0] - self.current_pos[0]) > 1e-6:
                    coords.append(self.format_coord("X", target[0]))
                if abs(target[1] - self.current_pos[1]) > 1e-6:
                    coords.append(self.format_coord("Y", target[1]))
                if abs(target[2] - self.current_pos[2]) > 1e-6:
                    coords.append(self.format_coord("Z", target[2]))

            if not coords:
                return None
            f_val = self.current_feed or 500.0
            return f"L {' '.join(coords)} {comp_str} F{self._fmt_f(f_val)}"

        return super()._motion_line(mode, target, extra)

    def _emit_arc(self, seg: MotionSegment, dist: float, is_helix: bool) -> list[str]:
        if not self.conversational:
            return super()._emit_arc(seg, dist, is_helix)

        lines = []
        start_pos = self.current_pos if self.is_pos_established else seg.start
        end_pos = seg.end
        is_ccw = seg.arc_ccw if is_helix else (seg.motion_type == MotionType.ARC_CCW or seg.arc_ccw)
        feed = seg.feed or self.current_feed or 500.0
        self.cut_dist += dist
        self.cut_time += (dist / max(feed, 10.0)) * 60.0

        if feed != self.current_feed:
            self.current_feed = feed

        # Format arc I/J offsets and center
        i_off, j_off = self._format_arc_ij(start_pos, end_pos, seg.arc_center, is_ccw)
        cx = start_pos[0] + i_off
        cy = start_pos[1] + j_off

        # Circle center CC
        lines.append(f"CC X{self._fmt_hnc(cx)} Y{self._fmt_hnc(cy)}")

        # Arc circular motion C
        dr = "DR+" if is_ccw else "DR-"
        comp_str = "R0"
        if self.comp_active and self.comp_side:
            side_s = str(self.comp_side).lower()
            comp_str = "RL" if side_s in ("rl", "left") else ("RR" if side_s in ("rr", "right") else "R0")
        z_end_str = f" Z{self._fmt_hnc(end_pos[2])}" if abs(end_pos[2] - start_pos[2]) > 1e-6 else ""
        lines.append(f"C X{self._fmt_hnc(end_pos[0])} Y{self._fmt_hnc(end_pos[1])}{z_end_str} {dr} {comp_str} F{self._fmt_f(feed)}")

        self.current_pos = end_pos.copy()
        self.is_pos_established = True
        return lines

    def process_segment(self, seg: MotionSegment) -> list[str]:
        if not self.conversational:
            return super().process_segment(seg)

        # Canned cycle detection for drilling in Klartext
        if seg.metadata.get("canned_cycle") and self.supports_cycle(
                seg.metadata["canned_cycle"].get("type", "G83")):
            cycle = seg.metadata["canned_cycle"]
            cycle_type = cycle.get("type", "G83")
            x = cycle.get("x", seg.end[0])
            y = cycle.get("y", seg.end[1])
            z_r = cycle.get("z_r", seg.start[2])
            z_depth = cycle.get("z_depth", seg.end[2])
            peck = cycle.get("peck", 0.0)
            dwell = cycle.get("dwell", 0.0)
            feed = cycle.get("feed", self.current_feed or 500.0)
            safe_z = self._get_safe_retract_z(fallback=50.0)

            setup_clearance = max(0.5, abs(z_r - 0.0))
            plunge_feed = int(round(feed))

            lines = []
            if cycle_type in ("G83", "G73", "205") and peck > 0:
                lines.extend([
                    "CYCL DEF 205 UNIVERSAL PECKING ~",
                    f"  Q200={setup_clearance:.3f} ;SET-UP CLEARANCE ~",
                    f"  Q201={z_depth:.3f} ;DEPTH ~",
                    f"  Q206={plunge_feed} ;FEED RATE FOR PLNGING ~",
                    f"  Q202={peck:.3f} ;PLUNGING DEPTH ~",
                    f"  Q203=+0.000 ;SURFACE COORDINATE ~",
                    f"  Q204={safe_z:.3f} ;2ND SET-UP CLEARANCE ~",
                    "  Q212=0 ;DECREMENT ~",
                    "  Q205=0 ;MIN. PLUNGING DEPTH ~",
                    "  Q258=0.2 ;UPPER ADVANCED STOP DIST ~",
                    "  Q259=0.2 ;LOWER ADVANCED STOP DIST ~",
                    "  Q257=0 ;DEPTH FOR CHIP BREAKING ~",
                    "  Q256=0.5 ;DIST. FOR CHIP BREAKING ~",
                    f"  Q211={dwell:.2f} ;DWELL TIME AT DEPTH",
                    f"L X{self._fmt_hnc(x)} Y{self._fmt_hnc(y)} R0 FMAX M99",
                ])
            else:
                lines.extend([
                    "CYCL DEF 200 DRILLING ~",
                    f"  Q200={setup_clearance:.3f} ;SET-UP CLEARANCE ~",
                    f"  Q201={z_depth:.3f} ;DEPTH ~",
                    f"  Q206={plunge_feed} ;FEED RATE FOR PLNGING ~",
                    f"  Q202={abs(z_depth):.3f} ;PLUNGING DEPTH ~",
                    "  Q210=0 ;DWELL TIME AT TOP ~",
                    f"  Q203=+0.000 ;SURFACE COORDINATE ~",
                    f"  Q204={safe_z:.3f} ;2ND SET-UP CLEARANCE ~",
                    f"  Q211={dwell:.2f} ;DWELL TIME AT DEPTH",
                    f"L X{self._fmt_hnc(x)} Y{self._fmt_hnc(y)} R0 FMAX M99",
                ])

            self.current_pos = np.array([x, y, z_r])
            self.canned_cycle_active = True
            self._cycle_op_id = seg.operation_id
            self._cycle_r_plane = z_r
            self.is_pos_established = True
            return lines

        if seg.motion_type == MotionType.DWELL:
            self.cut_time += seg.dwell_seconds
            return [
                "CYCL DEF 9.0 DWELL",
                f"CYCL DEF 9.1 DWELL {seg.dwell_seconds:.2f}",
            ]

        return super().process_segment(seg)

    def footer(self) -> list[str]:
        if not self.conversational:
            return super().footer()

        lines = []
        if self.coolant_active:
            lines.append("M9")
            self.coolant_active = False
        if self.spindle_active:
            lines.append("M5")
            self.spindle_active = False

        safe_z = self._get_safe_retract_z(fallback=50.0)
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", getattr(self, "_program_name", "VEXCAM")).upper()
        lines.extend([
            f"L Z{self._fmt_hnc(safe_z)} R0 FMAX M9",
            "L Z+0 R0 FMAX M91",
            "L X+0 Y+0 R0 FMAX M91",
            "M30",
            f"END PGM {clean_name} MM",
        ])
        return lines

    # --- Heidenhain Conversational Renderers ---
    def _render_tool_change(self, cmd: ToolChange) -> list[str]:
        if not self.conversational:
            return super()._render_tool_change(cmd)
        actual_tool_num = self._resolve_tool_number(cmd.tool_id, cmd.tool_number)
        # In conversational mode, the tool call includes spindle speed
        rpm_int = int(round(cmd.spindle_rpm)) if cmd.spindle_rpm else 3000
        if self.current_spindle and self.current_spindle > 0:
            rpm_int = int(round(self.current_spindle))
        elif cmd.spindle_rpm and cmd.spindle_rpm > 0:
            rpm_int = int(round(cmd.spindle_rpm))
        lines = [
            f"TOOL CALL {actual_tool_num} Z S{rpm_int} (Tool: T{actual_tool_num:02d} - {cmd.description})",
            "M3",
        ]
        self.spindle_active = True
        self.current_tool_id = cmd.tool_id
        self.is_pos_established = False
        if cmd.tool_id not in self.used_tools:
            self.used_tools.append(cmd.tool_id)
        return lines

    def _render_cycle(self, cmd: Cycle) -> list[str]:
        if not self.conversational:
            return super()._render_cycle(cmd)
        lines = []
        cycle_type = cmd.cycle_type
        x = cmd.x
        y = cmd.y
        z_r = cmd.r_plane
        z_depth = cmd.depth_z
        peck = cmd.peck
        dwell = cmd.dwell
        feed = cmd.feed
        safe_z = self._get_safe_retract_z(fallback=50.0)
        setup_clearance = max(0.5, abs(z_r - 0.0))
        plunge_feed = int(round(feed))
        if cycle_type in ("G83", "G73", "205") and peck > 0:
            lines.extend([
                "CYCL DEF 205 UNIVERSAL PECKING ~",
                f"  Q200={setup_clearance:.3f} ;SET-UP CLEARANCE ~",
                f"  Q201={z_depth:.3f} ;DEPTH ~",
                f"  Q206={plunge_feed} ;FEED RATE FOR PLNGING ~",
                f"  Q202={peck:.3f} ;PLUNGING DEPTH ~",
                f"  Q203=+0.000 ;SURFACE COORDINATE ~",
                f"  Q204={safe_z:.3f} ;2ND SET-UP CLEARANCE ~",
                "  Q212=0 ;DECREMENT ~",
                "  Q205=0 ;MIN. PLUNGING DEPTH ~",
                "  Q258=0.2 ;UPPER ADVANCED STOP DIST ~",
                "  Q259=0.2 ;LOWER ADVANCED STOP DIST ~",
                "  Q257=0 ;DEPTH FOR CHIP BREAKING ~",
                "  Q256=0.5 ;DIST. FOR CHIP BREAKING ~",
                f"  Q211={dwell:.2f} ;DWELL TIME AT DEPTH",
                f"L X{self._fmt_hnc(x)} Y{self._fmt_hnc(y)} R0 FMAX M99",
            ])
        else:
            lines.extend([
                "CYCL DEF 200 DRILLING ~",
                f"  Q200={setup_clearance:.3f} ;SET-UP CLEARANCE ~",
                f"  Q201={z_depth:.3f} ;DEPTH ~",
                f"  Q206={plunge_feed} ;FEED RATE FOR PLNGING ~",
                f"  Q202={abs(z_depth):.3f} ;PLUNGING DEPTH ~",
                "  Q210=0 ;DWELL TIME AT TOP ~",
                f"  Q203=+0.000 ;SURFACE COORDINATE ~",
                f"  Q204={safe_z:.3f} ;2ND SET-UP CLEARANCE ~",
                f"  Q211={dwell:.2f} ;DWELL TIME AT DEPTH",
                f"L X{self._fmt_hnc(x)} Y{self._fmt_hnc(y)} R0 FMAX M99",
            ])
        self.current_pos = np.array([x, y, z_r])
        self.canned_cycle_active = True
        self._cycle_op_id = ""  # operation_id not directly available
        self._cycle_r_plane = z_r
        self.is_pos_established = True
        return lines

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        if not self.conversational:
            return super()._render_safe_return(cmd)
        safe_z = self._get_safe_retract_z(fallback=50.0)
        if cmd.axis == "z":
            return [
                f"L Z{self._fmt_hnc(safe_z)} R0 FMAX M9",
                "L Z+0 R0 FMAX M91",
            ]
        elif cmd.axis == "xy":
            return ["L X+0 Y+0 R0 FMAX M91"]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        if not self.conversational:
            return super()._render_program_end(cmd)
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", getattr(self, "_program_name", "VEXCAM")).upper()
        return [
            "M30",
            f"END PGM {clean_name} MM",
        ]

    def _render_header(self, program_name: str, tools_info: Optional[dict[str, dict]], program_number: Optional[Union[int, str]]) -> list[str]:
        if not self.conversational:
            return super()._render_header(program_name, tools_info, program_number)
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", program_name).upper()
        if not clean_name:
            clean_name = "VEXCAM"
        lines = [
            f"BEGIN PGM {clean_name} MM",
            f"; Program: {clean_name}",
            "; Generated by VexCAM Post-Processor",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="semicolon"))
        lines.extend([
            f"CYCL DEF 247 DATUM SETTING ~",
            f"  Q339={self.datum_number} ;DATUM NUMBER",
        ])
        if self.hsm_enabled:
            lines.extend([
                "CYCL DEF 32.0 TOLERANCE",
                "CYCL DEF 32.1 T0.01",
            ])
        return lines

    def _render_footer(self, commands: Sequence[CanonicalCommand]) -> list[str]:
        if not self.conversational:
            return super()._render_footer(commands)
        lines = []
        for cmd in commands:
            if isinstance(cmd, CoolantOff):
                lines.append("M9")
            elif isinstance(cmd, SpindleStop):
                lines.append("M5")
            elif isinstance(cmd, SafeReturn):
                lines.extend(self._render_safe_return(cmd))
            elif isinstance(cmd, ProgramEnd):
                lines.extend(self._render_program_end(cmd))
        return lines

    def _render_comp_on(self, cmd: CompOn) -> list[str]:
        if not self.conversational:
            return super()._render_comp_on(cmd)
        self.comp_active = True
        self.comp_side = "RL" if cmd.side == "left" else "RR"
        return []

    def _render_comp_off(self, cmd: CompOff) -> list[str]:
        if not self.conversational:
            return super()._render_comp_off(cmd)
        self.comp_active = False
        self.comp_side = None
        return []

    def _render_motion(self, mode: str, target: np.ndarray, axes: Optional[tuple[str, ...]], extra: str = "") -> list[str]:
        if not self.conversational:
            return super()._render_motion(mode, target, axes, extra)
        comp_str = "R0"
        if self.comp_active and self.comp_side:
            comp_str = self.comp_side
        if mode == "G0":
            coords = []
            if not self.is_pos_established:
                coords.append(self.format_coord("X", target[0]))
                coords.append(self.format_coord("Y", target[1]))
                coords.append(self.format_coord("Z", target[2]))
                self.is_pos_established = True
            else:
                if abs(target[0] - self.current_pos[0]) > 1e-6:
                    coords.append(self.format_coord("X", target[0]))
                if abs(target[1] - self.current_pos[1]) > 1e-6:
                    coords.append(self.format_coord("Y", target[1]))
                if abs(target[2] - self.current_pos[2]) > 1e-6:
                    coords.append(self.format_coord("Z", target[2]))
            if not coords:
                return []
            return [f"L {' '.join(coords)} {comp_str} FMAX"]
        elif mode == "G1":
            coords = []
            if not self.is_pos_established:
                coords.append(self.format_coord("X", target[0]))
                coords.append(self.format_coord("Y", target[1]))
                coords.append(self.format_coord("Z", target[2]))
                self.is_pos_established = True
            else:
                if abs(target[0] - self.current_pos[0]) > 1e-6:
                    coords.append(self.format_coord("X", target[0]))
                if abs(target[1] - self.current_pos[1]) > 1e-6:
                    coords.append(self.format_coord("Y", target[1]))
                if abs(target[2] - self.current_pos[2]) > 1e-6:
                    coords.append(self.format_coord("Z", target[2]))
            if not coords:
                return []
            f_val = self.current_feed or 500.0
            return [f"L {' '.join(coords)} {comp_str} F{self._fmt_f(f_val)}"]
        return super()._render_motion(mode, target, axes, extra)

    def _render_arc(self, cmd: Arc) -> list[str]:
        if not self.conversational:
            return super()._render_arc(cmd)
        lines = []
        start_pos = self.current_pos if self.is_pos_established else cmd.start
        end_pos = cmd.target
        is_ccw = cmd.ccw
        feed = cmd.feed
        dist = float(np.linalg.norm(end_pos[:2] - start_pos[:2]))
        self.cut_dist += dist
        self.cut_time += (dist / max(feed, 10.0)) * 60.0
        if feed != self.current_feed:
            self.current_feed = feed
        # Use pre-computed center from canonical Arc command (engine-side precision)
        cx, cy = cmd.center[0], cmd.center[1]
        lines.append(f"CC X{self._fmt_hnc(cx)} Y{self._fmt_hnc(cy)}")
        dr = "DR+" if is_ccw else "DR-"
        comp_str = "R0"
        if self.comp_active and self.comp_side:
            comp_str = self.comp_side
        z_end_str = f" Z{self._fmt_hnc(end_pos[2])}" if abs(end_pos[2] - start_pos[2]) > 1e-6 else ""
        lines.append(f"C X{self._fmt_hnc(end_pos[0])} Y{self._fmt_hnc(end_pos[1])}{z_end_str} {dr} {comp_str} F{self._fmt_f(feed)}")
        self.current_pos = end_pos.copy()
        self.is_pos_established = True
        return lines

    def _render_dwell(self, cmd: Dwell) -> list[str]:
        if not self.conversational:
            return super()._render_dwell(cmd)
        self.cut_time += cmd.seconds
        return [
            "CYCL DEF 9.0 DWELL",
            f"CYCL DEF 9.1 DWELL {cmd.seconds:.2f}",
        ]

    def _render_machine_command(self, cmd: MachineCommand) -> list[str]:
        if not self.conversational:
            return super()._render_machine_command(cmd)
        # Heidenhain uses CYCL DEF 32 for HSM, not G05.1
        if cmd.name == "hsm_mode":
            return []  # Already emitted in header via CYCL DEF 32.0/32.1
        elif cmd.name == "hsm_reset":
            return []  # Not needed in Klartext
        return super()._render_machine_command(cmd)


class SiemensPostProcessor(BasePostProcessor):
    """Siemens Sinumerik (840D/828D) post processor supporting CYCLE81..86, D-registers, and CYCLE832."""

    supported_cycles: set[str] = {
        "G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76",
        "CYCLE81", "CYCLE82", "CYCLE83", "CYCLE84", "CYCLE85", "CYCLE86",
    }

    def __init__(self, **kwargs):
        cap_model = kwargs.pop("capability_model", None) or SIEMENS_CAPABILITY_MODEL
        super().__init__(capability_model=cap_model, **kwargs)
        self.hsm_enabled = True

    def _build_options_overrides(self) -> dict:
        return {
            "footer_hsm_reset_when": "mode_set",
            "tool_change_spindle_restart": True,
            "tool_change_rpm_rule": "spindle",
            "tool_change_retract_policy": "when_established",
            "tool_change_retract_z": None,
            "tool_change_retract_fallback": 50.0,
            "safe_retract_fallback": 50.0,
            "footer_include_cycle_cancel": True,  # uses MCALL via render_cycle_cancel
            "footer_include_comp_off": True,
        }

    def cycle_cancel(self) -> list[str]:
        """Cancel Siemens modal cycle with MCALL."""
        return ["MCALL"]

    def operation_safety_preamble(
        self,
        tp: Toolpath,
        tool_num: int,
        h_offset: int,
        wcs: str,
    ) -> list[str]:
        """Siemens Sinumerik deterministic modal safety line."""
        return [f"G90 G94 G17 G71 G40 G601 {wcs}"]

    def start_toolpath(self, tp: Toolpath) -> list[str]:
        if not self.hsm_enabled:
            return []
        mode = tp.metadata.get("hsm_mode") if tp.metadata else None
        if not mode:
            if tp.purpose in ("roughing", "facing", "rest_machining"):
                mode = "roughing"
            elif tp.purpose in ("semi_finishing", "chamfering", "countersinking", "grooving"):
                mode = "medium"
            else:
                mode = "finishing"
        if mode != self.current_hsm_mode:
            self.current_hsm_mode = mode
            tol = 0.05 if mode == "roughing" else (0.02 if mode == "medium" else 0.005)
            mode_code = 1 if mode == "roughing" else (2 if mode == "medium" else 3)
            return [f"CYCLE832({tol:.4f}, {mode_code}, 1) ; High-Speed Settings {mode.upper()}"]
        return []

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", program_name).upper()
        if not clean_name:
            clean_name = "VEXCAM"

        lines = [
            f"; %_N_{clean_name}_MPF",
            "; SIEMENS SINUMERIK 840D / 828D POST",
            "; Generated by VexCAM Engine",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="semicolon"))
        lines.extend([
            "G90 G94 G17 G71 G40 G601",
            f"{self.work_offset}",
        ])
        if self.hsm_enabled:
            lines.append("CYCLE832(0.0100, 1, 1) ; High-Speed Settings Active")
        return lines

    def tool_change(
        self,
        tool_id: str,
        tool_num: int = 1,
        length_offset_h: Optional[int] = None,
        tool_meta: Optional[dict] = None,
    ) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)

        if self.current_tool_id != tool_id:
            if self.canned_cycle_active:
                lines.append("MCALL")
                self.canned_cycle_active = False
                self._cycle_op_id = None
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False

            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.is_pos_established:
                lines.append(f"G0 Z{self._fmt(safe_z)}")

            # Siemens standard tool call: T1 M6 followed by D1 (active cutting edge & tool length)
            lines.append(f"T{actual_tool_num} M6 (Tool: T{actual_tool_num:02d} - {desc})")
            lines.append("D1")

            rpm = int(round(self.current_spindle or 3000.0))
            lines.append(f"S{rpm} M3")
            self.spindle_active = True

            self.current_tool_id = tool_id
            self.is_pos_established = False
            if tool_id not in self.used_tools:
                self.used_tools.append(tool_id)
        return lines

    def cutter_compensation_on(self, side: str = "left", d_register: int = 0, tool_meta: Optional[dict] = None) -> list[str]:
        g = "G41" if side == "left" else "G42"
        # Siemens activates cutter comp with D1 (or active D register)
        d_val = 1
        if d_register > 0:
            d_val = min(9, d_register)
        return [f"{g} D{d_val}"]

    def cutter_compensation_off(self) -> list[str]:
        return ["G40"]

    def process_segment(self, seg: MotionSegment) -> list[str]:
        # Handle Siemens canned cycles
        if seg.metadata.get("canned_cycle") and self.supports_cycle(
                seg.metadata["canned_cycle"].get("type", "G83")):
            cycle = seg.metadata["canned_cycle"]
            cycle_type = cycle.get("type", "G83")
            x = cycle.get("x", seg.end[0])
            y = cycle.get("y", seg.end[1])
            z_r = cycle.get("z_r", seg.start[2])
            z_depth = cycle.get("z_depth", seg.end[2])
            peck = cycle.get("peck", 0.0)
            dwell = cycle.get("dwell", 0.0)
            feed = cycle.get("feed", self.current_feed or 500.0)
            safe_z = self._get_safe_retract_z(fallback=50.0)

            if feed != self.current_feed:
                self.current_feed = feed

            lines = []
            rtp = safe_z
            rfp = z_r
            sdis = max(1.0, abs(z_r - 0.0)) if abs(z_r) < 1e-4 else 2.0
            dp = z_depth

            if cycle_type in ("G83", "G73", "CYCLE83") and peck > 0:
                # CYCLE83(RTP, RFP, SDIS, DP, DPR, FDEP, FDPR, DAM, DTB, DTS, FRF, VARI, _MDEP, _VRT, _DTD, _DIS1)
                fdep = z_depth + (abs(z_depth - z_r) * 0.5) if abs(z_depth - z_r) > 2.0 else z_depth
                lines.append(
                    f"CYCLE83({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {fdep:.3f}, , {peck:.3f}, {dwell:.2f}, 0.0, 1.0, 0)"
                )
                lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
            elif cycle_type in ("G82", "CYCLE82") and dwell > 0:
                # CYCLE82(RTP, RFP, SDIS, DP, DPR, DTB)
                lines.append(
                    f"CYCLE82({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f})"
                )
                lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
            elif cycle_type in ("G85", "CYCLE85"):
                # CYCLE85(RTP, RFP, SDIS, DP, DPR, DTB, FFR, RFF)
                lines.append(
                    f"CYCLE85({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f}, {feed:.1f}, {feed:.1f})"
                )
                lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
            elif cycle_type in ("G86", "CYCLE86"):
                # CYCLE86(RTP, RFP, SDIS, DP, DPR, DTB, SDIR, RPA, RPO, RPAP, POSS)
                lines.append(
                    f"CYCLE86({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f}, 3)"
                )
                lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
            else:
                # CYCLE81(RTP, RFP, SDIS, DP)
                lines.append(
                    f"CYCLE81({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f})"
                )
                lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")

            self.current_pos = np.array([x, y, z_r])
            self.canned_cycle_active = True
            self._cycle_op_id = seg.operation_id
            self._cycle_r_plane = z_r
            self.is_pos_established = True
            return lines

        if seg.motion_type == MotionType.DWELL:
            self.cut_time += seg.dwell_seconds
            return [f"G4 F{seg.dwell_seconds:.2f}"]

        return super().process_segment(seg)

    def footer(self) -> list[str]:
        lines = []
        if self.canned_cycle_active:
            lines.append("MCALL")
            self.canned_cycle_active = False
            self._cycle_op_id = None
        if self.comp_active:
            lines.extend(self.cutter_compensation_off())
            self.comp_active = False
            self.comp_side = None
        if self.coolant_active:
            lines.append("M9")
            self.coolant_active = False
        if self.spindle_active:
            lines.append("M5")
            self.spindle_active = False
        if self.hsm_enabled and self.current_hsm_mode is not None:
            lines.append("CYCLE832() ; Reset High-Speed Settings")

        lines.extend([
            "SUPA G0 Z0.0 D0",
            "SUPA G0 X0.0 Y0.0",
            "M30",
        ])
        return lines

    # --- Siemens Renderers ---
    def _render_tool_change(self, cmd: ToolChange) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(cmd.tool_id, cmd.tool_number)
        desc = cmd.description
        h = self._resolve_length_offset(cmd.tool_id, actual_tool_num, cmd.length_offset_h, None)
        if self.current_tool_id != cmd.tool_id:
            if self.canned_cycle_active:
                lines.append("MCALL")
                self.canned_cycle_active = False
                self._cycle_op_id = None
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.is_pos_established:
                lines.append(f"G0 Z{self._fmt(safe_z)}")
            lines.append(f"T{actual_tool_num} M6 (Tool: T{actual_tool_num:02d} - {desc})")
            lines.append("D1")
            target_rpm = cmd.spindle_rpm if cmd.spindle_rpm is not None else (self.current_spindle or 3000.0)
            rpm = int(round(target_rpm))
            lines.append(f"S{rpm} M3")
            self.current_spindle = target_rpm
            self.spindle_active = True
            self.current_tool_id = cmd.tool_id
            self.is_pos_established = False
            if cmd.tool_id not in self.used_tools:
                self.used_tools.append(cmd.tool_id)
        return lines

    def _render_cycle(self, cmd: Cycle) -> list[str]:
        lines = []
        cycle_type = cmd.cycle_type
        x = cmd.x
        y = cmd.y
        z_r = cmd.r_plane
        z_depth = cmd.depth_z
        peck = cmd.peck
        dwell = cmd.dwell
        feed = cmd.feed
        safe_z = self._get_safe_retract_z(fallback=50.0)
        if feed != self.current_feed:
            self.current_feed = feed
        rtp = safe_z
        rfp = z_r
        sdis = max(1.0, abs(z_r - 0.0)) if abs(z_r) < 1e-4 else 2.0
        dp = z_depth
        if cycle_type in ("G83", "G73", "CYCLE83") and peck > 0:
            fdep = z_depth + (abs(z_depth - z_r) * 0.5) if abs(z_depth - z_r) > 2.0 else z_depth
            lines.append(
                f"CYCLE83({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {fdep:.3f}, , {peck:.3f}, {dwell:.2f}, 0.0, 1.0, 0)"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        elif cycle_type in ("G82", "CYCLE82") and dwell > 0:
            lines.append(
                f"CYCLE82({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f})"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        elif cycle_type in ("G85", "CYCLE85"):
            lines.append(
                f"CYCLE85({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f}, {feed:.1f}, {feed:.1f})"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        elif cycle_type in ("G86", "CYCLE86"):
            lines.append(
                f"CYCLE86({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f}, 3)"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        else:
            lines.append(
                f"CYCLE81({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f})"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        self.current_pos = np.array([x, y, z_r])
        self.canned_cycle_active = True
        self._cycle_op_id = ""
        self._cycle_r_plane = z_r
        self.is_pos_established = True
        return lines

    def _render_cycle_cancel(self, cmd: CycleCancel) -> list[str]:
        self.canned_cycle_active = False
        self._cycle_op_id = None
        return ["MCALL"]

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        if cmd.axis == "z":
            return ["SUPA G0 Z0.0 D0"]
        elif cmd.axis == "xy":
            return ["SUPA G0 X0.0 Y0.0"]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        return ["M30"]

    def _render_machine_command(self, cmd: MachineCommand) -> list[str]:
        if cmd.name == "hsm_mode":
            mode = dict(cmd.params).get("mode", "")
            if self.hsm_enabled and mode:
                tol = 0.05 if mode == "roughing" else (0.02 if mode == "medium" else 0.005)
                mode_code = 1 if mode == "roughing" else (2 if mode == "medium" else 3)
                return [f"CYCLE832({tol:.4f}, {mode_code}, 1) ; High-Speed Settings {mode.upper()}"]
        elif cmd.name == "hsm_reset":
            if self.hsm_enabled and self.current_hsm_mode is not None:
                return ["CYCLE832() ; Reset High-Speed Settings"]
        return []


class Mach3Mach4PostProcessor(BasePostProcessor):
    """Mach3 and Mach4 CNC controller post-processor."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_program_end = "M30"

    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", program_name).upper() or "VEXCAM"
        lines = [
            f"(Program: {clean_name})",
            "(Mach3 / Mach4 CNC Controller)",
            "(Generated by VexCAM Engine)",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="parentheses"))
        lines.extend([
            "G90 G94 G91.1 G40 G49 G80",
            "G17 G21",
            f"{self.work_offset}",
        ])
        return lines

    def tool_change(self, tool_id: str, tool_num: int = 1, length_offset_h: Optional[int] = None, tool_meta: Optional[dict] = None) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        h_offset = self._resolve_h_offset(tool_id, length_offset_h, actual_tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)
        if self.current_tool_id != tool_id:
            if self.canned_cycle_active:
                lines.append("G80")
                self.canned_cycle_active = False
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.is_pos_established:
                lines.append(f"G0 Z{self._fmt(safe_z)}")
            lines.append(f"T{actual_tool_num} M6 (Tool: T{actual_tool_num:02d} - {desc})")
            lines.append(f"G43 H{h_offset}")
            rpm = int(round(self.current_spindle or 3000.0))
            lines.append(f"S{rpm} M3")
            self.spindle_active = True
            self.current_tool_id = tool_id
            self.is_pos_established = False
            if tool_id not in self.used_tools:
                self.used_tools.append(tool_id)
        return lines

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        safe_z = self._get_safe_retract_z(fallback=50.0)
        if cmd.axis == "z":
            return [f"G0 Z{self._fmt(safe_z)}"]
        elif cmd.axis == "xy":
            return ["G0 X0.0 Y0.0"]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        return ["M30"]


class CentroidPostProcessor(BasePostProcessor):
    """Centroid CNC controller post-processor."""
    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", program_name).upper() or "VEXCAM"
        lines = [
            f"; Program: {clean_name}",
            "; Centroid CNC Controller",
            "; Generated by VexCAM Engine",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="semicolon"))
        lines.extend([
            "G90 G94 G17 G21 G40 G49 G80",
            f"{self.work_offset}",
        ])
        return lines

    def tool_change(self, tool_id: str, tool_num: int = 1, length_offset_h: Optional[int] = None, tool_meta: Optional[dict] = None) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        h_offset = self._resolve_h_offset(tool_id, length_offset_h, actual_tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)
        if self.current_tool_id != tool_id:
            if self.canned_cycle_active:
                lines.append("G80")
                self.canned_cycle_active = False
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.is_pos_established:
                lines.append(f"G0 Z{self._fmt(safe_z)}")
            lines.append(f"T{actual_tool_num} M6 ; Tool: T{actual_tool_num:02d} - {desc}")
            lines.append(f"G43 H{h_offset}")
            rpm = int(round(self.current_spindle or 3000.0))
            lines.append(f"S{rpm} M3")
            self.spindle_active = True
            self.current_tool_id = tool_id
            self.is_pos_established = False
            if tool_id not in self.used_tools:
                self.used_tools.append(tool_id)
        return lines

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        safe_z = self._get_safe_retract_z(fallback=50.0)
        if cmd.axis == "z":
            return [f"G0 Z{self._fmt(safe_z)}"]
        elif cmd.axis == "xy":
            return ["G0 X0.0 Y0.0"]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        return ["M30"]


class MassoPostProcessor(BasePostProcessor):
    """MASSO G3 CNC controller post-processor."""
    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", program_name).upper() or "VEXCAM"
        lines = [
            f"(Program: {clean_name})",
            "(MASSO G3 CNC Controller)",
            "(Generated by VexCAM Engine)",
        ]
        if tools_info:
            lines.extend(self._format_tools_header_block(tools_info, style="parentheses"))
        lines.extend([
            "G90 G94 G17 G21 G40 G49 G80",
            f"{self.work_offset}",
        ])
        return lines

    def tool_change(self, tool_id: str, tool_num: int = 1, length_offset_h: Optional[int] = None, tool_meta: Optional[dict] = None) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)
        if self.current_tool_id != tool_id:
            if self.canned_cycle_active:
                lines.append("G80")
                self.canned_cycle_active = False
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.is_pos_established:
                lines.append(f"G0 Z{self._fmt(safe_z)}")
            lines.append(f"T{actual_tool_num} M6 (Tool: T{actual_tool_num:02d} - {desc})")
            rpm = int(round(self.current_spindle or 3000.0))
            lines.append(f"S{rpm} M3")
            self.spindle_active = True
            self.current_tool_id = tool_id
            self.is_pos_established = False
            if tool_id not in self.used_tools:
                self.used_tools.append(tool_id)
        return lines

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        safe_z = self._get_safe_retract_z(fallback=50.0)
        if cmd.axis == "z":
            return [f"G0 Z{self._fmt(safe_z)}"]
        elif cmd.axis == "xy":
            return ["G0 X0.0 Y0.0"]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        return ["M30"]


class MarlinPostProcessor(BasePostProcessor):
    """Marlin 3D/CNC firmware post-processor."""
    def header(self, program_name: str = "VEXCAM", tools_info: Optional[dict[str, dict]] = None, program_number: Optional[Union[int, str]] = None) -> list[str]:
        clean_name = re.sub(r"[^A-Za-z0-9_]", "_", program_name).upper() or "VEXCAM"
        lines = [
            f"; Program: {clean_name}",
            "; Marlin CNC Firmware",
            "; Generated by VexCAM Engine",
            "G90 G21",
        ]
        return lines

    def tool_change(self, tool_id: str, tool_num: int = 1, length_offset_h: Optional[int] = None, tool_meta: Optional[dict] = None) -> list[str]:
        actual_tool_num = self._resolve_tool_number(tool_id, tool_num)
        desc = self._format_tool_description(tool_id, tool_meta)
        lines = []
        if self.current_tool_id != tool_id:
            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.is_pos_established:
                lines.append(f"G0 Z{self._fmt(safe_z)}")
            lines.append(f"M0 (Change to T{actual_tool_num} - {desc})")
            rpm = int(round(self.current_spindle or 3000.0))
            lines.append(f"M3 S{rpm}")
            self.spindle_active = True
            self.current_tool_id = tool_id
        return lines

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        safe_z = self._get_safe_retract_z(fallback=50.0)
        if cmd.axis == "z":
            return [f"G0 Z{self._fmt(safe_z)}"]
        elif cmd.axis == "xy":
            return ["G0 X0.0 Y0.0"]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        return ["M5", "M2"]


SmoothiePostProcessor = MarlinPostProcessor
UccncPostProcessor = Mach3Mach4PostProcessor
SinumerikPostProcessor = SiemensPostProcessor

def get_post_processor(
    controller: str = "grbl",
    work_offset: Union[str, Any] = "G54",
    renumber_tools: bool = True,
    tool_map: Optional[dict[str, int]] = None,
    h_offset_map: Optional[dict[str, int]] = None,
    d_offset_map: Optional[dict[str, int]] = None,
    h_offset_policy: str = "match_tool_number",
    d_offset_policy: str = "match_tool_number",
    enable_tool_length_comp: bool = True,
    tool_change_mode: Optional[str] = None,
    tool_change_strategy: Optional[str] = None,
    tool_change_command: Optional[str] = None,
    tool_change_format: Optional[str] = None,
    stage_next_tool: bool = False,
    program_number: Optional[Union[int, str]] = None,
    base_program_number: Optional[int] = None,
    used_program_numbers: Optional[set[int]] = None,
    arc_interpolation_tolerance: float = 0.005,
    arc_linearization_chord_tolerance: float = 0.002,
    capability_model: Optional[ControllerCapabilityModel] = None,
    setup_clearance_z: Optional[float] = None,
    safe_retract_z: Optional[float] = None,
    fixture_clearance_z: Optional[float] = None,
    home_command_z: Optional[str] = None,
    home_command_xy: Optional[str] = None,
    program_end_command: Optional[str] = None,
    machine: Optional[Any] = None,
) -> BasePostProcessor:
    """Factory helper to obtain a post-processor configured from machine/controller profiles."""
    if not isinstance(work_offset, str):
        machine = work_offset
        work_offset = "G54"

    if machine is not None:
        if capability_model is None and hasattr(machine, "controller"):
            capability_model = get_controller_capability_model(getattr(machine, "controller", "grbl"))
        if tool_change_mode is None and tool_change_strategy is None:
            strat = getattr(machine, "tool_change_strategy", None)
            if strat is None:
                strat = "atc" if getattr(machine, "has_atc", True) else "manual"
            tool_change_mode = strat
            tool_change_strategy = strat
        if tool_change_command is None:
            tool_change_command = getattr(machine, "tool_change_command", None)
        if tool_change_format is None:
            tool_change_format = getattr(machine, "tool_change_format", None)
        if safe_retract_z is None:
            safe_retract_z = getattr(machine, "safe_retract_height", None) or getattr(machine, "machine_retract_z", None)
        if home_command_z is None:
            home_command_z = getattr(machine, "home_command_z", None)
        if home_command_xy is None:
            home_command_xy = getattr(machine, "home_command_xy", None)

    c = controller.lower().strip()
    kwargs = {
        "work_offset": work_offset,
        "renumber_tools": renumber_tools,
        "tool_map": tool_map,
        "h_offset_map": h_offset_map,
        "d_offset_map": d_offset_map,
        "h_offset_policy": h_offset_policy,
        "d_offset_policy": d_offset_policy,
        "enable_tool_length_comp": enable_tool_length_comp,
        "stage_next_tool": stage_next_tool,
        "program_number": program_number,
        "base_program_number": base_program_number,
        "used_program_numbers": used_program_numbers,
        "arc_interpolation_tolerance": arc_interpolation_tolerance,
        "arc_linearization_chord_tolerance": arc_linearization_chord_tolerance,
        "setup_clearance_z": setup_clearance_z,
        "safe_retract_z": safe_retract_z,
        "fixture_clearance_z": fixture_clearance_z,
        "home_command_z": home_command_z,
        "home_command_xy": home_command_xy,
        "program_end_command": program_end_command,
    }
    if capability_model is not None:
        kwargs["capability_model"] = capability_model
    if tool_change_mode is not None or tool_change_strategy is not None:
        kwargs["tool_change_mode"] = tool_change_strategy or tool_change_mode
        kwargs["tool_change_strategy"] = tool_change_strategy or tool_change_mode
    if tool_change_command is not None:
        kwargs["tool_change_command"] = tool_change_command
    if tool_change_format is not None:
        kwargs["tool_change_format"] = tool_change_format

    if "heidenhain" in c or "klartext" in c or "tnc" in c:
        return HeidenhainPostProcessor(**kwargs)
    elif "siemens" in c or "sinumerik" in c or "840d" in c or "828d" in c:
        return SiemensPostProcessor(**kwargs)
    elif "fanuc" in c:
        return FanucPostProcessor(**kwargs)
    elif "haas" in c:
        return HaasPostProcessor(**kwargs)
    elif "mach3" in c or "mach4" in c or "mach" in c:
        return Mach3Mach4PostProcessor(**kwargs)
    elif "centroid" in c:
        return CentroidPostProcessor(**kwargs)
    elif "masso" in c:
        return MassoPostProcessor(**kwargs)
    elif "uccnc" in c:
        return UccncPostProcessor(**kwargs)
    elif "marlin" in c:
        return MarlinPostProcessor(**kwargs)
    elif "smoothie" in c:
        return SmoothiePostProcessor(**kwargs)
    elif "linux" in c or "ngc" in c:
        return LinuxCncPostProcessor(**kwargs)
    else:
        return GrblPostProcessor(**kwargs)


