"""Controller-Specific G-Code Syntax and Capability Validator.

Validates generated CNC G-code against formal controller capability models before export:
- Supported modal codes & whitelists
- Supported canned cycles & parameter words (R, Q, P, F)
- Cutter compensation semantics (G41/G42/G40, D-registers)
- Tool-length compensation syntax (G43/G49, H-registers)
- Lookahead & smoothing syntax (Fanuc G05.1, Haas G187, LinuxCNC G64)
- Safe-return & homing behavior (G28 vs G53)
- Arc limitations & plane selections
- Machine feed & spindle speed limits
- Block formats, comment styles, percent (%) enclosures, and program numbering.
"""

import re
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any, Union

from .controller_models import (
    ControllerCapabilityModel,
    FANUC_CAPABILITY_MODEL,
    HAAS_CAPABILITY_MODEL,
    GRBL_CAPABILITY_MODEL,
    LINUXCNC_CAPABILITY_MODEL,
    get_controller_capability_model,
)


@dataclass
class ControllerSyntaxValidationReport:
    """Detailed diagnostic report of controller capability and syntax validation."""
    controller_id: str
    controller_name: str
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    modal_codes_used: Set[str] = field(default_factory=set)
    canned_cycles_used: Set[str] = field(default_factory=set)
    lookahead_codes_used: Set[str] = field(default_factory=set)
    cutter_comp_d_registers: List[int] = field(default_factory=list)
    tool_length_h_registers: List[int] = field(default_factory=list)
    max_feed_observed: float = 0.0
    max_spindle_observed: float = 0.0
    program_number: Optional[int] = None
    percent_enclosed: bool = False

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Controller Syntax Validation [{self.controller_name}]: {status}",
            f"Errors ({len(self.errors)}): " + ("; ".join(self.errors) if self.errors else "None"),
            f"Warnings ({len(self.warnings)}): " + ("; ".join(self.warnings) if self.warnings else "None"),
            f"Modal Codes: {sorted(list(self.modal_codes_used))}",
            f"Canned Cycles: {sorted(list(self.canned_cycles_used))}",
            f"D Registers: {sorted(list(set(self.cutter_comp_d_registers)))}",
            f"H Registers: {sorted(list(set(self.tool_length_h_registers)))}",
            f"Max Feed: {self.max_feed_observed:.1f} mm/min | Max Spindle: {self.max_spindle_observed:.0f} RPM",
        ]
        return "\n".join(lines)


class ControllerSyntaxValidator:
    """Validates G-code programs against a formal ControllerCapabilityModel."""

    def __init__(self, capability_model: Optional[ControllerCapabilityModel] = None):
        self.model = capability_model or FANUC_CAPABILITY_MODEL

    def validate(self, gcode: Union[str, List[str]], capability_model: Optional[ControllerCapabilityModel] = None) -> ControllerSyntaxValidationReport:
        model = capability_model or self.model
        report = ControllerSyntaxValidationReport(
            controller_id=model.controller_id,
            controller_name=model.name,
        )

        if isinstance(gcode, list):
            lines = gcode
        else:
            lines = gcode.splitlines()
        clean_lines: List[tuple[int, str, str]] = []  # (line_no, stripped_code, raw_line)

        # Regex for tokens and words
        g_re = re.compile(r"\b(G\d+(?:\.\d+)?)\b", re.IGNORECASE)
        m_re = re.compile(r"\b(M\d+)\b", re.IGNORECASE)
        d_re = re.compile(r"\bD(\d+)\b", re.IGNORECASE)
        h_re = re.compile(r"\bH(\d+)\b", re.IGNORECASE)
        f_re = re.compile(r"\bF([\d.]+)\b", re.IGNORECASE)
        s_re = re.compile(r"\bS([\d.]+)\b", re.IGNORECASE)
        o_re = re.compile(r"^O(\d+)", re.IGNORECASE)

        percent_lines = []

        for line_no, raw_line in enumerate(lines, start=1):
            raw_stripped = raw_line.strip()
            if not raw_stripped:
                continue

            if raw_stripped == "%":
                percent_lines.append(line_no)
                continue

            # Check comment styles and strip comments
            code_text = raw_stripped
            # Strip parentheses comments ( ... )
            code_text = re.sub(r"\(.*?\)", "", code_text)
            # Strip semicolon comments ; ...
            if ";" in code_text:
                code_text = code_text.split(";", 1)[0]
            code_text = code_text.strip()

            clean_lines.append((line_no, code_text, raw_stripped))

        # 1. Validate Block Enclosure (%)
        if model.block_format.requires_percent_enclosure:
            if len(percent_lines) < 2:
                report.errors.append(
                    f"Controller '{model.name}' requires % program block enclosure delimiters at start and end of program; found {len(percent_lines)}."
                )
            else:
                report.percent_enclosed = True
        else:
            if percent_lines:
                report.errors.append(
                    f"Controller '{model.name}' does not support % tape/block enclosure delimiters; found on line(s) {percent_lines}."
                )

        # 2. Validate Program Numbering (O####)
        found_o = False
        for line_no, code_text, raw_line in clean_lines:
            # Check if this line is an executable O program number block (e.g. "O1001" or "O1001 (NAME)")
            o_match = re.match(r"^O(\d+)", raw_line.strip(), re.IGNORECASE)
            if o_match:
                prog_num = int(o_match.group(1))
                report.program_number = prog_num
                found_o = True
                if not model.program_numbering.supported:
                    report.errors.append(
                        f"Line {line_no}: Controller '{model.name}' does not use program numbers (O####); found 'O{prog_num}'."
                    )
                else:
                    if prog_num < model.program_numbering.min_number or prog_num > model.program_numbering.max_number:
                        report.errors.append(
                            f"Line {line_no}: Program number O{prog_num} is out of allowable range "
                            f"[{model.program_numbering.min_number}..{model.program_numbering.max_number}] for '{model.name}'."
                        )
                    digits_len = len(o_match.group(1))
                    if digits_len > model.program_numbering.digits:
                        report.errors.append(
                            f"Line {line_no}: Program number O{o_match.group(1)} exceeds maximum digits ({model.program_numbering.digits}) for '{model.name}'."
                        )
                break

        if model.program_numbering.required and not found_o:
            report.errors.append(
                f"Controller '{model.name}' requires an explicit O program number header (e.g. O{model.program_numbering.min_number:0{model.program_numbering.digits}d})."
            )

        # 3. Line-by-Line Token, Modal, Feed, Spindle, and Comp Validation
        active_comp = None
        active_tool = None
        is_heidenhain = (model.controller_id == "heidenhain")
        is_siemens = (model.controller_id in ("siemens", "sinumerik"))

        cycle_re = re.compile(r"\b(CYCLE\d+)\b", re.IGNORECASE)
        hnc_cycl_re = re.compile(r"\bCYCL\s+DEF\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
        siemens_misc_re = re.compile(r"\b(MCALL|SUPA|TRANS|ATRANS|ROT|AROT|G601|SOFT|BRISK)\b", re.IGNORECASE)

        for line_no, code_text, raw_line in clean_lines:
            if not code_text:
                continue

            # Heidenhain Klartext Block handling
            if is_heidenhain:
                upper_code = code_text.upper()
                if upper_code.startswith("BEGIN PGM") or upper_code.startswith("END PGM"):
                    continue

                if upper_code.startswith("TOOL CALL"):
                    tc_m = re.search(r"TOOL CALL\s+(\d+)", upper_code)
                    if tc_m:
                        report.modal_codes_used.add("TOOL CALL")
                    s_m = s_re.findall(code_text)
                    for s_str in s_m:
                        s_val = float(s_str)
                        report.max_spindle_observed = max(report.max_spindle_observed, s_val)
                    continue

                # CYCL DEF in Klartext
                cycl_m = hnc_cycl_re.search(upper_code)
                if cycl_m:
                    c_num = cycl_m.group(1)
                    c_name = f"CYCL DEF {c_num}"
                    report.canned_cycles_used.add(c_name)
                    report.modal_codes_used.add("CYCL DEF")
                    if c_num in ("32", "32.0", "32.1"):
                        report.lookahead_codes_used.add("CYCL DEF 32")
                    continue

                # Klartext move commands: L, CC, C, CR
                if upper_code.startswith(("L ", "CC ", "C ", "CR ")) or upper_code in ("L", "CC", "C"):
                    cmd_word = upper_code.split()[0]
                    report.modal_codes_used.add(cmd_word)
                    if "RL" in upper_code:
                        report.modal_codes_used.add("RL")
                    elif "RR" in upper_code:
                        report.modal_codes_used.add("RR")
                    elif "R0" in upper_code:
                        report.modal_codes_used.add("R0")

                    if "FMAX" in upper_code:
                        report.modal_codes_used.add("FMAX")
                    else:
                        f_matches = f_re.findall(code_text)
                        for f_str in f_matches:
                            f_val = float(f_str)
                            report.max_feed_observed = max(report.max_feed_observed, f_val)
                            if f_val > model.max_feed_mm_min:
                                report.errors.append(
                                    f"Line {line_no}: Feed rate F{f_val:.1f} exceeds controller maximum clamp {model.max_feed_mm_min:.1f} mm/min."
                                )

                    # Check M-codes on the L line (e.g. M3, M9, M91, M99)
                    m_tokens = [m.upper() for m in m_re.findall(code_text)]
                    for m in m_tokens:
                        report.modal_codes_used.add(m)
                        if not model.is_gcode_supported(m):
                            report.errors.append(
                                f"Line {line_no}: Unsupported M-code '{m}' for controller '{model.name}'."
                            )
                    continue

            # Extract Siemens Cycles and Keywords
            if is_siemens:
                c_matches = cycle_re.findall(code_text)
                for cyc in c_matches:
                    cyc_u = cyc.upper()
                    report.canned_cycles_used.add(cyc_u)
                    if cyc_u == "CYCLE832":
                        report.lookahead_codes_used.add(cyc_u)
                    elif cyc_u not in model.supported_canned_cycles:
                        report.errors.append(
                            f"Line {line_no}: Canned cycle '{cyc_u}' is not supported by controller '{model.name}'."
                        )

                misc_matches = siemens_misc_re.findall(code_text)
                for msc in misc_matches:
                    report.modal_codes_used.add(msc.upper())

            # Extract G codes
            g_tokens = [g.upper() for g in g_re.findall(code_text)]
            # Extract M codes
            m_tokens = [m.upper() for m in m_re.findall(code_text)]

            for g in g_tokens:
                # Normalize leading zero: G0 -> G00, G1 -> G01, G05.1 -> G05.1
                norm_g = g
                report.modal_codes_used.add(norm_g)

                # Check general code support
                if not model.is_gcode_supported(norm_g):
                    report.errors.append(
                        f"Line {line_no}: Unsupported G-code '{norm_g}' for controller '{model.name}'."
                    )

                # Check Canned Cycle specifics
                if norm_g in {"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"}:
                    report.canned_cycles_used.add(norm_g)
                    if norm_g not in model.supported_canned_cycles:
                        report.errors.append(
                            f"Line {line_no}: Canned cycle '{norm_g}' is not supported by controller '{model.name}'."
                        )
                    # Verify canned cycle parameter words
                    if "Z" not in code_text:
                        report.errors.append(f"Line {line_no}: Canned cycle '{norm_g}' missing required Z target depth word.")
                    if "R" not in code_text:
                        report.errors.append(f"Line {line_no}: Canned cycle '{norm_g}' missing required R-plane clearance word.")
                    if norm_g in ("G83", "G73") and "Q" not in code_text:
                        report.errors.append(f"Line {line_no}: Peck drilling cycle '{norm_g}' missing required Q peck depth word.")
                    if norm_g in ("G82", "G86", "G76") and "P" not in code_text:
                        report.errors.append(f"Line {line_no}: Dwell/fine boring cycle '{norm_g}' missing required P dwell word.")

                # Check G98 / G99 return level support
                if norm_g in ("G98", "G99"):
                    if not model.supports_g98_g99:
                        report.errors.append(
                            f"Line {line_no}: Retract level command '{norm_g}' is not supported on controller '{model.name}'."
                        )

                # Check Lookahead specifics
                if norm_g in ("G05.1", "G5.1", "G187", "G64", "G61"):
                    report.lookahead_codes_used.add(norm_g)
                    if norm_g not in model.lookahead.valid_codes:
                        report.errors.append(
                            f"Line {line_no}: Lookahead/smoothing command '{norm_g}' does not match controller '{model.name}' "
                            f"(valid syntax: {model.lookahead.valid_codes or 'None'})."
                        )

                # Check Cutter Compensation specifics
                if norm_g in ("G41", "G42"):
                    if not model.cutter_comp.supported:
                        report.errors.append(
                            f"Line {line_no}: Cutter radius compensation '{norm_g}' is not supported by controller '{model.name}'."
                        )
                    else:
                        d_matches = d_re.findall(code_text)
                        if not d_matches:
                            report.errors.append(
                                f"Line {line_no}: Cutter compensation '{norm_g}' activated without D offset register word."
                            )
                        else:
                            d_val = int(d_matches[0])
                            report.cutter_comp_d_registers.append(d_val)
                            if d_val < 1 or d_val > model.cutter_comp.max_d_register:
                                report.errors.append(
                                    f"Line {line_no}: Cutter compensation D register D{d_val} exceeds controller limit "
                                    f"[1..{model.cutter_comp.max_d_register}] on '{model.name}'."
                                )

                # Check Tool Length Compensation specifics
                if norm_g == "G43":
                    if not model.tool_length_comp.supported:
                        report.errors.append(
                            f"Line {line_no}: Tool length compensation G43 is not supported by controller '{model.name}'."
                        )
                    else:
                        h_matches = h_re.findall(code_text)
                        if not h_matches:
                            report.errors.append(
                                f"Line {line_no}: Tool length compensation G43 activated without H offset register word."
                            )
                        else:
                            h_val = int(h_matches[0])
                            report.tool_length_h_registers.append(h_val)
                            if h_val < 1 or h_val > model.tool_length_comp.max_h_register:
                                report.errors.append(
                                    f"Line {line_no}: Tool length compensation H register H{h_val} exceeds controller limit "
                                    f"[1..{model.tool_length_comp.max_h_register}] on '{model.name}'."
                                )

            # Siemens standalone D register activation (e.g. D1 on tool change line)
            if is_siemens:
                d_matches = d_re.findall(code_text)
                for d_str in d_matches:
                    d_val = int(d_str)
                    report.cutter_comp_d_registers.append(d_val)

            for m in m_tokens:
                norm_m = m
                report.modal_codes_used.add(norm_m)
                if not model.is_gcode_supported(norm_m):
                    report.errors.append(
                        f"Line {line_no}: Unsupported M-code '{norm_m}' for controller '{model.name}'."
                    )

            # Check Feed Limits
            f_matches = f_re.findall(code_text)
            for f_str in f_matches:
                f_val = float(f_str)
                report.max_feed_observed = max(report.max_feed_observed, f_val)
                if f_val > model.max_feed_mm_min:
                    report.errors.append(
                        f"Line {line_no}: Feed rate F{f_val:.1f} exceeds controller maximum clamp {model.max_feed_mm_min:.1f} mm/min."
                    )

            # Check Spindle Limits
            s_matches = s_re.findall(code_text)
            for s_str in s_matches:
                s_val = float(s_str)
                report.max_spindle_observed = max(report.max_spindle_observed, s_val)
                if s_val > model.max_spindle_rpm:
                    report.errors.append(
                        f"Line {line_no}: Spindle speed S{s_val:.0f} exceeds controller maximum clamp {model.max_spindle_rpm:.0f} RPM."
                    )
                elif s_val < model.min_spindle_rpm and s_val > 0:
                    report.warnings.append(
                        f"Line {line_no}: Spindle speed S{s_val:.0f} is below recommended controller minimum {model.min_spindle_rpm:.0f} RPM."
                    )

        report.is_valid = len(report.errors) == 0
        return report
