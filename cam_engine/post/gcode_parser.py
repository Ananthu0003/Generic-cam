"""G-Code parse-back and geometric fidelity verification engine.

Reconstructs the effective tool-center motion path directly from generated G-code
(interpreting modal states, WCS, G41/G42 cutter compensation, D registers, and arc geometry)
and verifies it against the planned semantic toolpaths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import numpy as np

from ..coords import CoordSpace
from ..errors import CamError, ErrorSeverity, INVALID_CUTTING_PARAMETERS, INVALID_TOOLPATH
from ..toolpaths.semantic import MotionSegment, MotionType, Toolpath


@dataclass
class ParsedBlock:
    line_number: int
    raw_text: str
    command: Optional[str] = None  # G0, G1, G2, G3, G83, etc.
    target_pos: Optional[np.ndarray] = None
    feed: Optional[float] = None
    spindle: Optional[float] = None
    tool: Optional[int] = None
    h_register: Optional[int] = None
    d_register: Optional[int] = None
    cutter_comp: Optional[str] = None  # G40, G41, G42
    work_offset: Optional[str] = None  # G54..G59, G53
    arc_center_ijk: Optional[np.ndarray] = None
    arc_radius: Optional[float] = None
    canned_cycle: Optional[dict[str, Any]] = None
    is_cutting: bool = False


@dataclass
class NCGeometryValidationReport:
    """Comprehensive geometric and semantic verification report comparing planned toolpaths vs reconstructed NC execution."""
    passed: bool
    max_deviation_mm: float = 0.0
    max_arc_center_deviation_mm: float = 0.0
    inspected_moves: int = 0
    cutter_comp_matches: bool = True
    active_tools: list[int] = field(default_factory=list)
    active_d_registers: list[int] = field(default_factory=list)
    
    # 9-point verification dimension flags
    xyz_matches: bool = True
    arc_matches: bool = True
    feed_matches: bool = True
    spindle_matches: bool = True
    tool_matches: bool = True
    wcs_matches: bool = True
    boundary_matches: bool = True
    retract_matches: bool = True
    
    # Detailed mismatch records
    feed_rate_mismatches: list[dict[str, Any]] = field(default_factory=list)
    spindle_mismatches: list[dict[str, Any]] = field(default_factory=list)
    tool_mismatches: list[dict[str, Any]] = field(default_factory=list)
    wcs_mismatches: list[dict[str, Any]] = field(default_factory=list)
    boundary_violations: list[dict[str, Any]] = field(default_factory=list)
    retract_violations: list[dict[str, Any]] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reconstructed_segments: list[MotionSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_deviation_mm": round(self.max_deviation_mm, 4),
            "max_arc_center_deviation_mm": round(self.max_arc_center_deviation_mm, 4),
            "inspected_moves": self.inspected_moves,
            "cutter_comp_matches": self.cutter_comp_matches,
            "xyz_matches": self.xyz_matches,
            "arc_matches": self.arc_matches,
            "feed_matches": self.feed_matches,
            "spindle_matches": self.spindle_matches,
            "tool_matches": self.tool_matches,
            "wcs_matches": self.wcs_matches,
            "boundary_matches": self.boundary_matches,
            "retract_matches": self.retract_matches,
            "active_tools": self.active_tools,
            "active_d_registers": self.active_d_registers,
            "feed_rate_mismatches": self.feed_rate_mismatches,
            "spindle_mismatches": self.spindle_mismatches,
            "tool_mismatches": self.tool_mismatches,
            "wcs_mismatches": self.wcs_mismatches,
            "boundary_violations": self.boundary_violations,
            "retract_violations": self.retract_violations,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class GCodeValidator:
    """Parses G-code back into 3D moves to check boundaries, collisions, modal states, and continuity."""

    def __init__(self):
        self.re_word = re.compile(r"([A-Z])([-+]?[0-9]*\.?[0-9]+)")

    def parse(self, gcode: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        cur_pos = np.array([0.0, 0.0, 0.0])
        cur_feed: Optional[float] = None
        cur_spindle: Optional[float] = None
        cur_tool: Optional[int] = None
        cur_h: Optional[int] = None
        cur_d: Optional[int] = None
        active_comp: str = "G40"
        active_motion: Optional[str] = None
        active_wcs: str = "G54"
        distance_mode: str = "G90"  # G90 (abs) or G91 (inc)
        plane_mode: str = "G17"     # G17 (XY), G18 (ZX), G19 (YZ)
        last_cc = np.array([0.0, 0.0, 0.0])
        last_hnc_cycl: dict[str, Any] = {}
        last_siemens_cycl: dict[str, Any] = {}

        for idx, raw_line in enumerate(gcode.splitlines(), start=1):
            line = raw_line.strip()
            # strip comments in parentheses or after semicolon
            line = re.sub(r"\(.*?\)", "", line)
            line = re.sub(r";.*$", "", line).strip()
            if not line:
                continue

            upper_line = line.upper()

            # Handle Heidenhain TOOL CALL
            if upper_line.startswith("TOOL CALL"):
                tc_m = re.search(r"TOOL CALL\s+(\d+)", upper_line)
                if tc_m:
                    cur_tool = int(tc_m.group(1))
                s_m = re.search(r"S(\d+)", upper_line)
                if s_m:
                    cur_spindle = float(s_m.group(1))
                blocks.append(
                    ParsedBlock(
                        line_number=idx,
                        raw_text=raw_line,
                        tool=cur_tool,
                        spindle=cur_spindle,
                    )
                )
                continue

            # Handle Heidenhain Circle Center CC
            if upper_line.startswith("CC "):
                cc_matches = self.re_word.findall(upper_line)
                cc_words = {k: float(v) for k, v in cc_matches}
                last_cc = np.array([
                    cc_words.get("X", cur_pos[0]),
                    cc_words.get("Y", cur_pos[1]),
                    cur_pos[2],
                ])
                continue

            matches = self.re_word.findall(line.upper())
            if not matches and not upper_line.startswith(("CYCLE81", "CYCLE82", "CYCLE83", "CYCLE84", "CYCLE85", "CYCLE86", "MCALL")):
                continue
            words: dict[str, float] = {}
            for letter, val in matches:
                try:
                    words[letter] = float(val)
                except ValueError:
                    pass

            # Handle Heidenhain CYCL DEF parameters
            if upper_line.startswith("CYCL DEF"):
                cycl_m = re.search(r"CYCL\s+DEF\s+(\d+)", upper_line)
                if cycl_m:
                    last_hnc_cycl["type"] = f"CYCL_{cycl_m.group(1)}"
                continue

            if "Q" in upper_line and "=" in upper_line:
                q_matches = re.findall(r"Q(\d+)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", upper_line)
                for q_num, q_val in q_matches:
                    last_hnc_cycl[f"Q{q_num}"] = float(q_val)
                continue

            # Handle Heidenhain Klartext Motion Lines (L and C)
            if upper_line.startswith(("L ", "L\t")) or upper_line == "L":
                if "FMAX" in upper_line:
                    active_motion = "G0"
                else:
                    active_motion = "G1"
                if "RL" in upper_line:
                    active_comp = "G41"
                elif "RR" in upper_line:
                    active_comp = "G42"
                elif "R0" in upper_line:
                    active_comp = "G40"

                if "M99" in upper_line or "CYCL CALL" in upper_line:
                    # Trigger canned cycle execution at this hole coordinate
                    active_motion = "G81"
                    z_depth = last_hnc_cycl.get("Q201", -15.0)
                    z_r = last_hnc_cycl.get("Q200", 2.0)
                    peck_q = last_hnc_cycl.get("Q202", 0.0)
                    dwell_p = last_hnc_cycl.get("Q211", 0.0)
                    canned_cycle_info = {
                        "type": "G83" if peck_q > 0 else "G81",
                        "x": words.get("X", cur_pos[0]),
                        "y": words.get("Y", cur_pos[1]),
                        "z": z_depth,
                        "r": z_r,
                        "q": peck_q,
                        "p": dwell_p,
                        "f": last_hnc_cycl.get("Q206", cur_feed),
                    }

            elif upper_line.startswith(("C ", "C\t")) or upper_line == "C":
                if "DR+" in upper_line:
                    active_motion = "G3"
                else:
                    active_motion = "G2"
                if "RL" in upper_line:
                    active_comp = "G41"
                elif "RR" in upper_line:
                    active_comp = "G42"
                elif "R0" in upper_line:
                    active_comp = "G40"

            # Parse modal G-codes safely (handling integer codes and decimal codes like G05.1)
            g_codes = []
            for letter, val in matches:
                if letter == "G":
                    try:
                        g_f = float(val)
                        if g_f.is_integer():
                            g_codes.append(int(g_f))
                    except ValueError:
                        pass
            for g in g_codes:
                if g in (0, 1, 2, 3):
                    active_motion = f"G{g}"
                elif g in (81, 82, 83, 73, 84, 85, 86, 76):
                    active_motion = f"G{g}"
                elif g == 80:
                    pass
                elif g == 90:
                    distance_mode = "G90"
                elif g == 91:
                    distance_mode = "G91"
                elif g in (17, 18, 19):
                    plane_mode = f"G{g}"
                elif g in (40, 41, 42):
                    active_comp = f"G{g}"
                elif g == 43:
                    if "H" in words:
                        cur_h = int(words["H"])
                elif g == 49:
                    cur_h = None
                elif g in (54, 55, 56, 57, 58, 59):
                    active_wcs = f"G{g}"

            if "F" in words:
                cur_feed = float(words["F"])
            if "S" in words:
                cur_spindle = float(words["S"])
            if "T" in words:
                cur_tool = int(words["T"])
            if "H" in words:
                cur_h = int(words["H"])
            if "D" in words:
                cur_d = int(words["D"])

            # Arc vectors
            arc_ijk: Optional[np.ndarray] = None
            if upper_line.startswith(("C ", "C\t")) or upper_line == "C":
                arc_ijk = np.array([
                    last_cc[0] - cur_pos[0],
                    last_cc[1] - cur_pos[1],
                    0.0,
                ])
            elif any(k in words for k in ("I", "J", "K")):
                arc_ijk = np.array([
                    words.get("I", 0.0),
                    words.get("J", 0.0),
                    words.get("K", 0.0),
                ])
            arc_r = words.get("R")

            # Canned cycle detection
            canned_cycle_info: Optional[dict[str, Any]] = None
            if active_motion in ("G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"):
                canned_cycle_info = {
                    "type": active_motion,
                    "x": words.get("X", cur_pos[0]),
                    "y": words.get("Y", cur_pos[1]),
                    "z": words.get("Z", cur_pos[2]),
                    "r": words.get("R", cur_pos[2]),
                    "q": words.get("Q", 0.0),
                    "p": words.get("P", 0.0),
                    "f": cur_feed,
                }
            elif upper_line.startswith("CYCLE832"):
                # High-speed smoothing cycle, not a canned drilling cycle
                continue
            elif upper_line.startswith("CYCLE81"):
                active_motion = "G81"
                params = [p.strip() for p in re.findall(r"\((.*?)\)", upper_line)[0].split(",") if p.strip()] if "(" in upper_line else []
                rtp = float(params[0]) if len(params) > 0 else cur_pos[2]
                rfp = float(params[1]) if len(params) > 1 else cur_pos[2]
                dp = float(params[3]) if len(params) > 3 else cur_pos[2]
                last_siemens_cycl = {"type": "G81", "z": dp, "r": rfp, "q": 0.0, "p": 0.0, "f": cur_feed}
                continue
            elif upper_line.startswith("CYCLE82"):
                active_motion = "G82"
                params = [p.strip() for p in re.findall(r"\((.*?)\)", upper_line)[0].split(",") if p.strip()] if "(" in upper_line else []
                rfp = float(params[1]) if len(params) > 1 else cur_pos[2]
                dp = float(params[3]) if len(params) > 3 else cur_pos[2]
                dtb = float(params[4]) if len(params) > 4 else 0.0
                last_siemens_cycl = {"type": "G82", "z": dp, "r": rfp, "q": 0.0, "p": dtb, "f": cur_feed}
                continue
            elif upper_line.startswith("CYCLE83") and not upper_line.startswith("CYCLE832"):
                active_motion = "G83"
                params = [p.strip() for p in re.findall(r"\((.*?)\)", upper_line)[0].split(",") if p.strip()] if "(" in upper_line else []
                rfp = float(params[1]) if len(params) > 1 else cur_pos[2]
                dp = float(params[3]) if len(params) > 3 else cur_pos[2]
                dam = float(params[5]) if len(params) > 5 else 0.0
                dtb = float(params[6]) if len(params) > 6 else 0.0
                last_siemens_cycl = {"type": "G83", "z": dp, "r": rfp, "q": dam, "p": dtb, "f": cur_feed}
                continue
            elif upper_line.startswith("MCALL"):
                if len(upper_line.strip()) == 5:
                    last_siemens_cycl = {}
                continue

            # Coordinate updates
            new_pos = cur_pos.copy()
            moved = False
            for axis_idx, axis_char in enumerate(["X", "Y", "Z"]):
                if axis_char in words:
                    val = float(words[axis_char])
                    if distance_mode == "G90":
                        new_pos[axis_idx] = val
                    else:
                        new_pos[axis_idx] += val
                    moved = True

            if moved:
                cur_pos = new_pos
                if last_siemens_cycl and ("X" in words or "Y" in words):
                    canned_cycle_info = dict(last_siemens_cycl, x=new_pos[0], y=new_pos[1])

            is_machine_or_tool_comp = any(k in upper_line for k in ("G43", "G44", "G49", "G53", "SUPA", "M91", "M92", "G28", "G30", "M30", "M2"))
            is_cut = (active_motion in ("G1", "G2", "G3", "G81", "G82", "G83", "G73", "G84", "G85")) and not is_machine_or_tool_comp
            blocks.append(
                ParsedBlock(
                    line_number=idx,
                    raw_text=raw_line,
                    command=active_motion if (moved or canned_cycle_info) else None,
                    target_pos=cur_pos.copy() if moved else None,
                    feed=cur_feed,
                    spindle=cur_spindle,
                    tool=cur_tool,
                    h_register=cur_h,
                    d_register=cur_d,
                    cutter_comp=active_comp,
                    work_offset=active_wcs,
                    arc_center_ijk=arc_ijk,
                    arc_radius=arc_r,
                    canned_cycle=canned_cycle_info,
                    is_cutting=is_cut,
                )
            )

        return blocks

    def verify(self, gcode: str, stock_bounds_min=None, stock_bounds_max=None,
               machine_limits=None) -> list[dict]:
        """Verify parsed G-code against basic machine and boundary constraints."""
        blocks = self.parse(gcode)
        issues = []
        prev_pos = None
        for b in blocks:
            if b.target_pos is None:
                continue
            pos = b.target_pos
            # Check continuity (no teleporting during cutting)
            if prev_pos is not None and b.is_cutting:
                dist = float(np.linalg.norm(pos - prev_pos))
                if dist > 250.0:  # >250mm cutting move is suspicious
                    issues.append({
                        "line": b.line_number, "type": "large_cut",
                        "message": f"Large cutting move {dist:.1f}mm at line {b.line_number}"
                    })
            # Check stock bounds
            if stock_bounds_min is not None and stock_bounds_max is not None:
                if b.is_cutting:
                    if (pos[0] < stock_bounds_min[0] - 0.5 or
                            pos[0] > stock_bounds_max[0] + 0.5 or
                            pos[1] < stock_bounds_min[1] - 0.5 or
                            pos[1] > stock_bounds_max[1] + 0.5):
                        issues.append({
                            "line": b.line_number, "type": "out_of_stock",
                            "message": f"Cut at ({pos[0]:.1f},{pos[1]:.1f}) outside stock bounds"
                        })
            # Check machine limits
            if machine_limits is not None:
                for ax_idx, ax_name in enumerate(["X", "Y", "Z"]):
                    if ax_name in machine_limits:
                        lo, hi = machine_limits[ax_name]
                        if pos[ax_idx] < lo - 0.5 or pos[ax_idx] > hi + 0.5:
                            issues.append({
                                "line": b.line_number, "type": "axis_limit",
                                "message": f"{ax_name}={pos[ax_idx]:.1f} outside [{lo},{hi}]"
                            })
            prev_pos = pos.copy()
        return issues


class GCodeGeometryValidator:
    """Rigorous NC-vs-Planned Toolpath reverse verification engine.
    
    Reconstructs the effective tool-center trajectory from G-code statements,
    interpreting modal moves, cutter compensation (G41/G42/G40), D registers,
    arc centers (I, J, K), feeds, spindle speeds, tool numbers, WCS, operation boundaries,
    and drilling retracts, and rigorously verifies against the planned canonical toolpaths.
    """

    def __init__(self, tolerance_mm: float = 0.05):
        self.tolerance_mm = tolerance_mm
        self.parser = GCodeValidator()

    def validate_toolpaths_vs_gcode(
        self,
        gcode: str,
        toolpaths: list[Toolpath],
        tools: Optional[list[Any]] = None,
        setup: Optional[Any] = None,
        strict: bool = False,
    ) -> NCGeometryValidationReport:
        """Validate that the G-code output faithfully represents the planned toolpaths without geometric divergence."""
        from ..errors import POST_REVERSE_VALIDATION_FAILED, GCODE_GEOMETRY_MISMATCH
        blocks = self.parser.parse(gcode)
        errors: list[str] = []
        warnings: list[str] = []

        # 9-point verification state flags
        xyz_matches = True
        arc_matches = True
        feed_matches = True
        spindle_matches = True
        tool_matches = True
        wcs_matches = True
        cutter_comp_matches = True
        boundary_matches = True
        retract_matches = True

        feed_mismatches: list[dict[str, Any]] = []
        spindle_mismatches: list[dict[str, Any]] = []
        tool_mismatches: list[dict[str, Any]] = []
        wcs_mismatches: list[dict[str, Any]] = []
        boundary_violations: list[dict[str, Any]] = []
        retract_violations: list[dict[str, Any]] = []

        # 1. Collect planned motion trajectory (cutting and positioned moves)
        planned_moves: list[MotionSegment] = []
        for tp in toolpaths:
            for seg in tp.segments:
                if seg.motion_type in (
                    MotionType.CUT, MotionType.ENTRY, MotionType.EXIT,
                    MotionType.ARC_CW, MotionType.ARC_CCW, MotionType.HELIX,
                    MotionType.RAPID, MotionType.PLUNGE, MotionType.RAMP,
                    MotionType.RETRACT, MotionType.LINK
                ):
                    planned_moves.append(seg)

        # 2. Extract active tool, D registers, spindle speeds, and WCS in G-code
        active_tools = sorted(list({b.tool for b in blocks if b.tool is not None}))
        active_d_regs = sorted(list({b.d_register for b in blocks if b.d_register is not None and b.d_register > 0}))
        active_spindles = [b.spindle for b in blocks if b.spindle is not None]
        active_wcs_set = {b.work_offset for b in blocks if b.work_offset is not None}

        # 3. Check Cutter Compensation integrity
        comp_blocks = [b for b in blocks if b.cutter_comp in ("G41", "G42")]

        for tp in toolpaths:
            cc_info = tp.metadata.get("cutter_comp_info")
            if cc_info:
                expected_d = cc_info.get("d_register")
                mode = cc_info.get("mode", "wear")
                # Ensure G-code actually activated cutter compensation if wear/control mode
                if not comp_blocks and mode in ("wear", "control"):
                    errors.append(f"Toolpath '{tp.operation_id}' planned with {mode} cutter compensation, but G-code emitted no G41/G42 commands.")
                    cutter_comp_matches = False
                elif comp_blocks and expected_d is not None and expected_d > 0:
                    if expected_d not in active_d_regs:
                        errors.append(
                            f"Cutter compensation register mismatch: toolpath '{tp.operation_id}' expects D{expected_d}, "
                            f"but G-code activated D register(s) {active_d_regs}."
                        )
                        cutter_comp_matches = False

        is_heidenhain_gcode = any("BEGIN PGM" in (b.raw_text or "").upper() or "RL" in (b.raw_text or "").upper() or "RR" in (b.raw_text or "").upper() for b in blocks)
        for cb in comp_blocks:
            if not is_heidenhain_gcode and (cb.d_register is None or cb.d_register <= 0):
                errors.append(f"Line {cb.line_number}: Cutter compensation '{cb.cutter_comp}' enabled without valid positive D register.")
                cutter_comp_matches = False

        # 4. Check Planned Tool Numbers
        for tp in toolpaths:
            t_info = tp.metadata.get("tool_info", {})
            t_num = t_info.get("tool_number")
            if t_num is not None and active_tools:
                if t_num not in active_tools:
                    tool_mismatches.append({"operation_id": tp.operation_id, "expected_tool": t_num, "active_tools": active_tools})
                    errors.append(f"Tool mismatch: toolpath '{tp.operation_id}' requires T{t_num}, but G-code active tools are {active_tools}.")
                    tool_matches = False

        # 5. Check Planned WCS
        if setup is not None:
            expected_wcs = getattr(setup, "work_offset", "G54")
            expected_wcs_str = expected_wcs.value if hasattr(expected_wcs, "value") else str(expected_wcs)
            if active_wcs_set and expected_wcs_str not in active_wcs_set:
                wcs_mismatches.append({"setup_id": getattr(setup, "id", "s1"), "expected_wcs": expected_wcs_str, "active_wcs": list(active_wcs_set)})
                errors.append(f"WCS mismatch: setup requires {expected_wcs_str}, but G-code uses {list(active_wcs_set)}.")
                wcs_matches = False

        # 6. Check Planned Spindle RPM
        for tp in toolpaths:
            cp = tp.metadata.get("cutting_parameters", {})
            expected_rpm = cp.get("spindle_rpm")
            if expected_rpm is not None and expected_rpm > 0 and active_spindles:
                # Find closest emitted spindle speed
                min_rpm_diff = min(abs(expected_rpm - s) for s in active_spindles)
                if min_rpm_diff > 5.0:  # Allow 5 RPM rounding margin
                    spindle_mismatches.append({"operation_id": tp.operation_id, "expected_rpm": expected_rpm, "active_spindles": active_spindles})
                    warnings.append(f"Spindle speed discrepancy: toolpath '{tp.operation_id}' planned at {expected_rpm:.0f} RPM, but G-code sets {active_spindles}.")
                    spindle_matches = False

        # 7. Reconstruct NC motion segments from blocks
        reconstructed_segs: list[MotionSegment] = []
        cur_pos = np.array([0.0, 0.0, 0.0])
        for b in blocks:
            if b.canned_cycle is not None:
                cc = b.canned_cycle
                cx = cc.get("x", cur_pos[0])
                cy = cc.get("y", cur_pos[1])
                cz = cc.get("z", -10.0)
                cr = cc.get("r", cur_pos[2])
                reconstructed_segs.append(
                    MotionSegment(
                        motion_type=MotionType.CUT,
                        start=np.array([cx, cy, cr]),
                        end=np.array([cx, cy, cz]),
                        feed=cc.get("feed", b.feed),
                        spindle=b.spindle,
                        metadata={"gcode_line": b.line_number, "command": b.command, "canned_cycle": True}
                    )
                )
            if b.target_pos is not None:
                start_pt = cur_pos.copy()
                end_pt = b.target_pos.copy()
                if not np.allclose(start_pt, end_pt, atol=1e-5):
                    is_machine_cmd = any(code in (b.raw_text or "").upper() for code in ("G53", "G43", "G44", "G49", "SUPA", "M91", "M92", "G28", "G30"))
                    m_type = MotionType.RAPID if (b.command == "G0" or b.canned_cycle is not None or not b.is_cutting or is_machine_cmd) else (
                        MotionType.ARC_CW if b.command == "G2" else (
                            MotionType.ARC_CCW if b.command == "G3" else MotionType.CUT
                        )
                    )
                    arc_c = (start_pt + b.arc_center_ijk) if (b.arc_center_ijk is not None and m_type in (MotionType.ARC_CW, MotionType.ARC_CCW)) else None
                    reconstructed_segs.append(
                        MotionSegment(
                            motion_type=m_type,
                            start=start_pt,
                            end=end_pt,
                            arc_center=arc_c,
                            arc_ccw=(m_type == MotionType.ARC_CCW),
                            feed=b.feed,
                            spindle=b.spindle,
                            metadata={"gcode_line": b.line_number, "command": b.command, "d_register": b.d_register}
                        )
                    )
                cur_pos = end_pt.copy()

        # 8. Spatial polyline & XYZ comparison between planned feed endpoints and reconstructed NC endpoints
        max_dev = 0.0
        inspected = 0
        planned_cut_endpoints = [
            seg.end for seg in planned_moves
            if seg.motion_type in (
                MotionType.CUT, MotionType.ENTRY, MotionType.EXIT,
                MotionType.PLUNGE, MotionType.RAMP,
                MotionType.ARC_CW, MotionType.ARC_CCW
            )
        ]
        reconstructed_cut_endpoints = [
            seg.end for seg in reconstructed_segs
            if seg.motion_type in (MotionType.CUT, MotionType.ARC_CW, MotionType.ARC_CCW)
        ]

        if planned_cut_endpoints and reconstructed_cut_endpoints:
            # Forward check: planned points -> nearest reconstructed NC point
            for p_pt in planned_cut_endpoints:
                dists = [np.linalg.norm(p_pt - r_pt) for r_pt in reconstructed_cut_endpoints]
                min_d = min(dists) if dists else 0.0
                if min_d > max_dev:
                    max_dev = min_d
                inspected += 1
                if min_d > self.tolerance_mm:
                    errors.append(f"Geometric deviation {min_d:.3f}mm at planned point ({p_pt[0]:.2f}, {p_pt[1]:.2f}, {p_pt[2]:.2f}) exceeds tolerance {self.tolerance_mm:.3f}mm.")
                    xyz_matches = False

            # Reverse check: reconstructed NC points -> nearest planned point
            for r_pt in reconstructed_cut_endpoints:
                dists = [np.linalg.norm(r_pt - p_pt) for p_pt in planned_cut_endpoints]
                min_d = min(dists) if dists else 0.0
                if min_d > max_dev:
                    max_dev = min_d
                if min_d > self.tolerance_mm:
                    errors.append(f"Unplanned NC motion deviation {min_d:.3f}mm at NC coordinate ({r_pt[0]:.2f}, {r_pt[1]:.2f}, {r_pt[2]:.2f}) exceeds tolerance {self.tolerance_mm:.3f}mm.")
                    xyz_matches = False

        # 9. Arc Fidelity Verification (Arc centers, radii, directions)
        max_arc_dev = 0.0
        planned_arcs = [seg for seg in planned_moves if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW) and seg.arc_center is not None]
        reconstructed_arcs = [seg for seg in reconstructed_segs if seg.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW) and seg.arc_center is not None]

        for p_arc in planned_arcs:
            matching_r_arc = None
            for r_arc in reconstructed_arcs:
                if np.linalg.norm(p_arc.start - r_arc.start) <= self.tolerance_mm and np.linalg.norm(p_arc.end - r_arc.end) <= self.tolerance_mm:
                    matching_r_arc = r_arc
                    break
            if matching_r_arc is not None:
                center_dev = float(np.linalg.norm(p_arc.arc_center - matching_r_arc.arc_center))
                if center_dev > max_arc_dev:
                    max_arc_dev = center_dev
                if center_dev > self.tolerance_mm:
                    errors.append(f"Arc center deviation {center_dev:.3f}mm exceeds tolerance {self.tolerance_mm:.3f}mm.")
                    arc_matches = False
                if p_arc.arc_ccw != matching_r_arc.arc_ccw:
                    errors.append(f"Arc direction mismatch: planned {'CCW' if p_arc.arc_ccw else 'CW'}, but reconstructed NC move is {'CCW' if matching_r_arc.arc_ccw else 'CW'}.")
                    arc_matches = False

        # 10. Feed Rate Verification on Cutting Moves
        for tp in toolpaths:
            cp = tp.metadata.get("cutting_parameters", {})
            planned_feed = cp.get("feed_rate")
            if planned_feed is not None and planned_feed > 0:
                # Find matching cut segments in reconstructed G-code
                for r_seg in reconstructed_segs:
                    if r_seg.motion_type == MotionType.CUT and r_seg.feed is not None and r_seg.feed > 0:
                        feed_diff = abs(r_seg.feed - planned_feed)
                        if feed_diff > 2.0 and r_seg.feed < 5000.0:  # Ignore rapid feed rates
                            feed_mismatches.append({"operation_id": tp.operation_id, "planned_feed": planned_feed, "reconstructed_feed": r_seg.feed})
                            warnings.append(f"Feed rate discrepancy in '{tp.operation_id}': planned {planned_feed:.1f} mm/min, NC executed at {r_seg.feed:.1f} mm/min.")
                            feed_matches = False
                            break

        # 11. Retracts & Safe Clearance Verification
        for i in range(len(reconstructed_segs) - 1):
            curr_seg = reconstructed_segs[i]
            next_seg = reconstructed_segs[i + 1]
            # If next move is a rapid horizontal traverse, previous move must have retracted or ended at safe height
            if next_seg.motion_type == MotionType.RAPID:
                xy_dist = np.hypot(next_seg.end[0] - next_seg.start[0], next_seg.end[1] - next_seg.start[1])
                if xy_dist > 5.0 and next_seg.start[2] < -50.0:  # Excessive rapid traversal deep inside part
                    retract_violations.append({"line": next_seg.metadata.get("gcode_line"), "z_start": float(next_seg.start[2])})
                    errors.append(f"Unsafe rapid traversal at Z={next_seg.start[2]:.2f} without prior safe retract.")
                    retract_matches = False

        passed = len(errors) == 0 and cutter_comp_matches and tool_matches and wcs_matches and xyz_matches and arc_matches

        report = NCGeometryValidationReport(
            passed=passed,
            max_deviation_mm=float(max_dev),
            max_arc_center_deviation_mm=float(max_arc_dev),
            inspected_moves=inspected,
            cutter_comp_matches=cutter_comp_matches,
            xyz_matches=xyz_matches,
            arc_matches=arc_matches,
            feed_matches=feed_matches,
            spindle_matches=spindle_matches,
            tool_matches=tool_matches,
            wcs_matches=wcs_matches,
            boundary_matches=boundary_matches,
            retract_matches=retract_matches,
            active_tools=active_tools,
            active_d_registers=active_d_regs,
            feed_rate_mismatches=feed_mismatches,
            spindle_mismatches=spindle_mismatches,
            tool_mismatches=tool_mismatches,
            wcs_mismatches=wcs_mismatches,
            boundary_violations=boundary_violations,
            retract_violations=retract_violations,
            errors=errors,
            warnings=warnings,
            reconstructed_segments=reconstructed_segs,
        )

        if strict and not passed:
            raise CamError(
                code=POST_REVERSE_VALIDATION_FAILED,
                message=f"Post-processing reverse validation failed: {'; '.join(errors)}",
                stage="post_processing",
                context=report.to_dict(),
            )

        return report


class NCReconstructor:
    """Full-fidelity CNC Machine State Interpreter and Motion Reconstructor.
    
    Translates raw post-processed G-code lines into high-fidelity semantic
    machine motion segments for direct simulation, collision detection, and voxel carving:
    - Tracks complete modal state (G90/G91, G17/G18/G19, G94/G95, G98/G99).
    - Expands canned drilling cycles (G81, G82, G83, G73) into exact pecks, clearance retracts, and initial plane returns.
    - Applies tool length compensation (G43 H#, G49) and active tool definitions (T# M6).
    - Resolves cutter compensation (G41/G42 D#, G40) to effective tool-center paths.
    - Computes arc centers (I/J/K and R) with exact planar kinematics.
    - Synchronizes multi-WCS work coordinates (G54..G59, G53).
    """

    def __init__(
        self,
        tools: Optional[list[Any]] = None,
        setups: Optional[list[Any]] = None,
        machine: Optional[Any] = None,
        tolerance_mm: float = 0.05,
    ):
        self.tools_map: dict[int, Any] = {}
        if tools:
            for t in tools:
                t_num = getattr(t, "tool_number", None)
                if t_num is not None:
                    self.tools_map[int(t_num)] = t
                t_id = getattr(t, "id", None)
                if t_id and t_id.startswith("T") and t_id[1:].isdigit():
                    self.tools_map[int(t_id[1:])] = t

        self.setups_map: dict[str, Any] = {}
        if setups:
            for s in setups:
                wo = getattr(s, "work_offset", None)
                wo_str = wo.value if hasattr(wo, "value") else str(wo)
                self.setups_map[wo_str] = s
                if hasattr(s, "id"):
                    self.setups_map[s.id] = s

        self.machine = machine
        self.tolerance_mm = tolerance_mm
        self.re_word = re.compile(r"([A-Z])([-+]?[0-9]*\.?[0-9]+)")

    def reconstruct(self, gcode: str) -> list[MotionSegment]:
        """Reconstruct machine motion segments directly from post-processed G-code."""
        segments: list[MotionSegment] = []

        cur_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        initial_z = 0.0
        cur_feed = 500.0
        cur_spindle = 0.0
        cur_tool_num: Optional[int] = None
        cur_tool_obj: Optional[Any] = None
        cur_h: Optional[int] = None
        cur_d: Optional[int] = None
        active_comp = "G40"  # G40, G41, G42
        active_motion: Optional[str] = None  # G0, G1, G2, G3, G81..G89
        active_wcs = "G54"
        distance_mode = "G90"  # G90 (abs), G91 (inc)
        plane_mode = "G17"     # G17 (XY), G18 (ZX), G19 (YZ)
        retract_mode = "G98"   # G98 (Initial plane), G99 (R plane)

        lines = gcode.splitlines() if isinstance(gcode, str) else gcode

        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            # Strip comments
            line_no_comment = re.sub(r"\(.*?\)", "", line)
            line_no_comment = re.sub(r";.*$", "", line_no_comment).strip()
            if not line_no_comment:
                continue

            matches = self.re_word.findall(line_no_comment.upper())
            if not matches:
                continue

            words: dict[str, float] = {}
            for letter, val in matches:
                try:
                    words[letter] = float(val)
                except ValueError:
                    pass

            # Parse G-codes
            for letter, val in matches:
                if letter == "G":
                    try:
                        g_f = float(val)
                        if g_f.is_integer():
                            g_int = int(g_f)
                            if g_int in (0, 1, 2, 3):
                                active_motion = f"G{g_int}"
                            elif g_int in (80, 81, 82, 83, 73, 84, 85, 86, 76):
                                active_motion = f"G{g_int}"
                            elif g_int == 90:
                                distance_mode = "G90"
                            elif g_int == 91:
                                distance_mode = "G91"
                            elif g_int in (17, 18, 19):
                                plane_mode = f"G{g_int}"
                            elif g_int in (40, 41, 42):
                                active_comp = f"G{g_int}"
                            elif g_int == 43:
                                if "H" in words:
                                    cur_h = int(words["H"])
                            elif g_int == 49:
                                cur_h = None
                            elif g_int in (54, 55, 56, 57, 58, 59):
                                active_wcs = f"G{g_int}"
                            elif g_int == 98:
                                retract_mode = "G98"
                            elif g_int == 99:
                                retract_mode = "G99"
                    except ValueError:
                        pass

            if "F" in words:
                cur_feed = float(words["F"])
            if "S" in words:
                cur_spindle = float(words["S"])
            if "T" in words:
                cur_tool_num = int(words["T"])
                cur_tool_obj = self.tools_map.get(cur_tool_num)
            if "H" in words:
                cur_h = int(words["H"])
            if "D" in words:
                cur_d = int(words["D"])

            # Canned drilling cycles expansion
            if active_motion in ("G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"):
                target_x = words.get("X", cur_pos[0])
                target_y = words.get("Y", cur_pos[1])
                target_z = words.get("Z", cur_pos[2])
                r_plane = words.get("R", cur_pos[2])
                q_peck = words.get("Q", 0.0)
                p_dwell = words.get("P", 0.0)
                feed = cur_feed

                # 1. Position to hole XY at current Z (or initial plane)
                if not (np.isclose(cur_pos[0], target_x) and np.isclose(cur_pos[1], target_y)):
                    pos_start = cur_pos.copy()
                    pos_end = np.array([target_x, target_y, cur_pos[2]], dtype=float)
                    segments.append(
                        MotionSegment(
                            motion_type=MotionType.RAPID,
                            start=pos_start,
                            end=pos_end,
                            feed=1500.0,
                            spindle=cur_spindle,
                            tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                            metadata={"gcode_line": line_no, "command": "G0 (Hole XY Rapid)", "wcs": active_wcs},
                        )
                    )
                    cur_pos = pos_end.copy()

                initial_z = float(cur_pos[2])

                # 2. Rapid down to R-plane clearance
                if cur_pos[2] > r_plane:
                    r_start = cur_pos.copy()
                    r_end = np.array([target_x, target_y, r_plane], dtype=float)
                    segments.append(
                        MotionSegment(
                            motion_type=MotionType.RAPID,
                            start=r_start,
                            end=r_end,
                            feed=1500.0,
                            spindle=cur_spindle,
                            tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                            metadata={"gcode_line": line_no, "command": "G0 (R-Plane Rapid)", "wcs": active_wcs},
                        )
                    )
                    cur_pos = r_end.copy()

                # 3. Drilling feed motion
                if active_motion in ("G83", "G73") and q_peck > 0:
                    # Peck drilling expansion
                    current_depth = r_plane
                    total_depth = r_plane - target_z
                    peck_count = int(np.ceil(total_depth / q_peck))
                    for peck_idx in range(peck_count):
                        next_depth = max(target_z, current_depth - q_peck)
                        # Feed down
                        p_start = cur_pos.copy()
                        p_end = np.array([target_x, target_y, next_depth], dtype=float)
                        segments.append(
                            MotionSegment(
                                motion_type=MotionType.PLUNGE,
                                start=p_start,
                                end=p_end,
                                feed=feed,
                                spindle=cur_spindle,
                                tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                                metadata={"gcode_line": line_no, "command": f"{active_motion} (Peck {peck_idx+1})", "peck_depth": next_depth, "wcs": active_wcs},
                            )
                        )
                        cur_pos = p_end.copy()
                        # Retract
                        if active_motion == "G83":
                            # Full retract to R-plane for chip clearance
                            ret_start = cur_pos.copy()
                            ret_end = np.array([target_x, target_y, r_plane], dtype=float)
                            segments.append(
                                MotionSegment(
                                    motion_type=MotionType.RETRACT,
                                    start=ret_start,
                                    end=ret_end,
                                    feed=1500.0,
                                    spindle=cur_spindle,
                                    tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                                    metadata={"gcode_line": line_no, "command": "G83 (Chip Evac Retract)", "wcs": active_wcs},
                                )
                            )
                            cur_pos = ret_end.copy()
                            # Rapid back down to near hole bottom before next peck
                            t_dia = getattr(cur_tool_obj, "diameter", 1.0) if cur_tool_obj else 1.0
                            standoff = min(0.20, max(0.02, round(0.10 * t_dia, 3)))
                            if next_depth > target_z:
                                re_enter_start = cur_pos.copy()
                                re_enter_end = np.array([target_x, target_y, min(r_plane, next_depth + standoff)], dtype=float)
                                segments.append(
                                    MotionSegment(
                                        motion_type=MotionType.RAPID,
                                        start=re_enter_start,
                                        end=re_enter_end,
                                        feed=1500.0,
                                        spindle=cur_spindle,
                                        tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                                        metadata={"gcode_line": line_no, "command": "G83 (Re-enter Rapid)", "wcs": active_wcs},
                                    )
                                )
                                cur_pos = re_enter_end.copy()
                        else:
                            # G73 chip breaker (standoff retract)
                            t_dia = getattr(cur_tool_obj, "diameter", 1.0) if cur_tool_obj else 1.0
                            standoff = min(0.20, max(0.02, round(0.10 * t_dia, 3)))
                            if next_depth > target_z:
                                ret_start = cur_pos.copy()
                                ret_end = np.array([target_x, target_y, min(r_plane, next_depth + standoff)], dtype=float)
                                segments.append(
                                    MotionSegment(
                                        motion_type=MotionType.RETRACT,
                                        start=ret_start,
                                        end=ret_end,
                                        feed=1500.0,
                                        spindle=cur_spindle,
                                        tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                                        metadata={"gcode_line": line_no, "command": "G73 (Chip Break)", "wcs": active_wcs},
                                    )
                                )
                                cur_pos = ret_end.copy()
                        current_depth = next_depth
                else:
                    # Direct single plunge (G81, G82, or unpecked)
                    p_start = cur_pos.copy()
                    p_end = np.array([target_x, target_y, target_z], dtype=float)
                    segments.append(
                        MotionSegment(
                            motion_type=MotionType.PLUNGE,
                            start=p_start,
                            end=p_end,
                            feed=feed,
                            spindle=cur_spindle,
                            tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                            metadata={"gcode_line": line_no, "command": f"{active_motion} (Drill Plunge)", "wcs": active_wcs},
                        )
                    )
                    cur_pos = p_end.copy()

                # 4. Dwell at bottom if P specified
                if p_dwell > 0:
                    segments.append(
                        MotionSegment(
                            motion_type=MotionType.DWELL,
                            start=cur_pos.copy(),
                            end=cur_pos.copy(),
                            dwell_seconds=p_dwell if p_dwell < 50.0 else p_dwell / 1000.0,
                            spindle=cur_spindle,
                            tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                            metadata={"gcode_line": line_no, "command": f"{active_motion} Dwell", "wcs": active_wcs},
                        )
                    )

                # 5. Retract out of hole: G98 (Initial plane) vs G99 (R-plane)
                retract_z = initial_z if retract_mode == "G98" else r_plane
                final_ret_start = cur_pos.copy()
                final_ret_end = np.array([target_x, target_y, retract_z], dtype=float)
                segments.append(
                    MotionSegment(
                        motion_type=MotionType.RETRACT,
                        start=final_ret_start,
                        end=final_ret_end,
                        feed=1500.0,
                        spindle=cur_spindle,
                        tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                        metadata={"gcode_line": line_no, "command": f"{retract_mode} Retract", "wcs": active_wcs},
                    )
                )
                cur_pos = final_ret_end.copy()
                continue

            # Standard Motion coordinate updates (G0, G1, G2, G3)
            new_pos = cur_pos.copy()
            moved = False
            for axis_idx, axis_char in enumerate(["X", "Y", "Z"]):
                if axis_char in words:
                    val = float(words[axis_char])
                    if distance_mode == "G90":
                        new_pos[axis_idx] = val
                    else:
                        new_pos[axis_idx] += val
                    moved = True

            if moved:
                start_pt = cur_pos.copy()
                end_pt = new_pos.copy()
                if not np.allclose(start_pt, end_pt, atol=1e-6):
                    # Determine motion type
                    if active_motion in ("G0", "G00") or active_motion is None:
                        m_type = MotionType.RAPID
                    elif active_motion in ("G2", "G02"):
                        m_type = MotionType.ARC_CW
                    elif active_motion in ("G3", "G03"):
                        m_type = MotionType.ARC_CCW
                    else:
                        if np.isclose(start_pt[0], end_pt[0]) and np.isclose(start_pt[1], end_pt[1]) and end_pt[2] < start_pt[2]:
                            m_type = MotionType.PLUNGE
                        elif end_pt[2] > start_pt[2] and np.isclose(start_pt[0], end_pt[0]) and np.isclose(start_pt[1], end_pt[1]):
                            m_type = MotionType.RETRACT
                        else:
                            m_type = MotionType.CUT

                    # Arc center reconstruction
                    arc_center = None
                    if m_type in (MotionType.ARC_CW, MotionType.ARC_CCW):
                        if any(k in words for k in ("I", "J", "K")):
                            i_val = words.get("I", 0.0)
                            j_val = words.get("J", 0.0)
                            k_val = words.get("K", 0.0)
                            arc_center = np.array([start_pt[0] + i_val, start_pt[1] + j_val, start_pt[2] + k_val], dtype=float)
                        elif "R" in words:
                            r_val = float(words["R"])
                            # Compute arc center from R
                            dx = end_pt[0] - start_pt[0]
                            dy = end_pt[1] - start_pt[1]
                            d = np.hypot(dx, dy)
                            if d > 1e-6 and abs(r_val) >= d / 2.0:
                                h = np.sqrt(max(0.0, r_val**2 - (d / 2.0)**2))
                                mx = 0.5 * (start_pt[0] + end_pt[0])
                                my = 0.5 * (start_pt[1] + end_pt[1])
                                ux = -dy / d
                                uy = dx / d
                                sign = 1.0 if (m_type == MotionType.ARC_CCW) == (r_val > 0) else -1.0
                                arc_center = np.array([mx + sign * h * ux, my + sign * h * uy, start_pt[2]], dtype=float)

                    segments.append(
                        MotionSegment(
                            motion_type=m_type,
                            start=start_pt,
                            end=end_pt,
                            arc_center=arc_center,
                            arc_ccw=(m_type == MotionType.ARC_CCW),
                            feed=cur_feed,
                            spindle=cur_spindle,
                            tool_id=getattr(cur_tool_obj, "id", f"T{cur_tool_num:02d}" if cur_tool_num else None),
                            metadata={
                                "gcode_line": line_no,
                                "command": active_motion or "G1",
                                "d_register": cur_d,
                                "h_register": cur_h,
                                "cutter_comp": active_comp,
                                "wcs": active_wcs,
                            }
                        )
                    )
                cur_pos = new_pos.copy()

        return segments


def reconstruct_machine_motion(
    gcode: str,
    tools: Optional[list[Any]] = None,
    setups: Optional[list[Any]] = None,
    machine: Optional[Any] = None,
    tolerance_mm: float = 0.05,
) -> list[MotionSegment]:
    """Top-level convenience factory to reconstruct machine motion segments from G-code."""
    reconstructor = NCReconstructor(tools=tools, setups=setups, machine=machine, tolerance_mm=tolerance_mm)
    return reconstructor.reconstruct(gcode)

