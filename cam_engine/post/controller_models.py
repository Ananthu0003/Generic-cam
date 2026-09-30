"""Controller Capability Model.

Defines comprehensive CNC machine controller syntax, modal groups, canned cycles,
cutter compensation semantics, tool-length compensation syntax, lookahead syntax,
safe-return behaviors, arc limitations, block formatting, and program numbering rules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set, Dict, Any, List


@dataclass
class CutterCompCapability:
    supported: bool = True
    requires_d_register: bool = True
    max_d_register: int = 99
    valid_commands: Set[str] = field(default_factory=lambda: {"G40", "G41", "G42"})
    supported_modes: Set[str] = field(default_factory=lambda: {"wear", "control", "computer"})
    min_lead_in_ratio: float = 0.8  # Minimum linear lead move length as ratio of cutter radius
    offset_memory_type: str = "type_c_separate"  # "type_c_separate" (Fanuc Type C: separate Geom and Wear), "type_a_combined" (Haas / Fanuc Type A), "siemens_d_register"
    default_mode: str = "wear"  # "wear" | "control" | "computer"
    geom_value_format: str = "radius"  # "radius" or "diameter"


@dataclass
class ToolLengthCompCapability:
    supported: bool = True
    command: str = "G43"
    cancel_command: str = "G49"
    requires_h_register: bool = True
    max_h_register: int = 99
    valid_commands: Set[str] = field(default_factory=lambda: {"G43", "G44", "G49"})


@dataclass
class LookaheadCapability:
    supported: bool = True
    syntax_type: str = "aicc"  # "aicc" (Fanuc G05.1 Q1), "hsm" (Haas G187), "naive_cam" (LinuxCNC G64), "none" (GRBL)
    valid_codes: Set[str] = field(default_factory=lambda: {"G05.1"})
    on_pattern: str = "G05.1 Q1"
    off_pattern: str = "G05.1 Q0"


@dataclass
class SafeReturnCapability:
    syntax_type: str = "g28_incremental"  # "g28_incremental" (Fanuc), "g53_machine_coords" (Haas), "g0_retract" (GRBL/LinuxCNC), "custom"
    valid_codes: Set[str] = field(default_factory=lambda: {"G28", "G91"})
    z_retract_command: str = "G91 G28 Z0.0"
    xy_home_command: str = "G28 X0.0 Y0.0"
    program_end_command: str = "M30"      # "M30", "M2"
    retract_to_setup_clearance_first: bool = True


@dataclass
class ArcCapability:
    requires_ij_center: bool = True
    supports_r_word: bool = True
    min_radius_mm: float = 0.005
    max_radius_mm: float = 100000.0
    valid_planes: Set[str] = field(default_factory=lambda: {"G17", "G18", "G19"})
    valid_commands: Set[str] = field(default_factory=lambda: {"G2", "G02", "G3", "G03"})


@dataclass
class BlockFormatCapability:
    requires_percent_enclosure: bool = True
    comment_style: str = "parentheses"  # "parentheses" ( ... ) or "semicolon" ; ...
    allow_line_numbers: bool = True
    max_line_length: int = 256
    line_endings: str = "\n"


@dataclass
class ProgramNumberingCapability:
    supported: bool = True
    required: bool = True
    style: str = "O_4digit"  # "O_4digit" (O1001), "O_5digit" (O01001), "none"
    prefix: str = "O"
    min_number: int = 1
    max_number: int = 9999
    digits: int = 4


class ToolChangeStrategy(str, Enum):
    ATC = "atc"                    # Automatic tool changer (Txx M6)
    MANUAL = "manual"              # Manual operator tool change with program stop (Txx M0 / M00)
    COMMENT_ONLY = "comment_only"  # Tool info in comment only (no M-code)
    CUSTOM = "custom"              # Custom formatting template


@dataclass
class ToolChangeCapability:
    strategy: str = "atc"          # "atc", "manual", "comment_only", "custom"
    command: str = "M6"            # "M6", "M0", "M00", "M1", "M01"
    template: Optional[str] = None # custom format string e.g. "M6 T{tool_num}"
    supports_preselect: bool = True # allows T{next} staging in carousel
    requires_spindle_stop: bool = True
    requires_coolant_stop: bool = True
    retract_z_first: bool = True


@dataclass
class ControllerCapabilityModel:
    """Formal capability and syntax specification for a CNC machine controller."""
    controller_id: str
    name: str
    supported_modal_groups: Dict[str, Set[str]] = field(default_factory=dict)
    supported_canned_cycles: Set[str] = field(default_factory=set)
    supports_g98_g99: bool = True
    cutter_comp: CutterCompCapability = field(default_factory=CutterCompCapability)
    tool_length_comp: ToolLengthCompCapability = field(default_factory=ToolLengthCompCapability)
    lookahead: LookaheadCapability = field(default_factory=LookaheadCapability)
    safe_return: SafeReturnCapability = field(default_factory=SafeReturnCapability)
    arc: ArcCapability = field(default_factory=ArcCapability)
    block_format: BlockFormatCapability = field(default_factory=BlockFormatCapability)
    program_numbering: ProgramNumberingCapability = field(default_factory=ProgramNumberingCapability)
    tool_change: ToolChangeCapability = field(default_factory=ToolChangeCapability)
    max_feed_mm_min: float = 30000.0
    max_spindle_rpm: float = 24000.0
    min_spindle_rpm: float = 50.0

    def is_gcode_supported(self, code: str) -> bool:
        """Check whether a G-code or M-code token is recognized and supported by this controller."""
        norm_code = code.upper().strip()
        # Check across all modal groups
        for group, codes in self.supported_modal_groups.items():
            if norm_code in codes:
                return True
        # Check canned cycles
        if norm_code in self.supported_canned_cycles:
            return True
        # Check lookahead codes
        if norm_code in self.lookahead.valid_codes:
            return True
        # Check cutter comp codes
        if norm_code in self.cutter_comp.valid_commands:
            return True
        # Check tool length comp codes
        if norm_code in self.tool_length_comp.valid_commands:
            return True
        # Check safe return codes
        if norm_code in self.safe_return.valid_codes:
            return True
        # Check arc codes
        if norm_code in self.arc.valid_commands:
            return True
        return False


# ---------------------------------------------------------------------------
# Predefined Controller Capability Models
# ---------------------------------------------------------------------------

FANUC_CAPABILITY_MODEL = ControllerCapabilityModel(
    controller_id="fanuc",
    name="Fanuc 3-Axis Mill (0i/30i/31i Series)",
    supported_modal_groups={
        "motion": {"G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G4"},
        "plane": {"G17", "G18", "G19"},
        "distance": {"G90", "G91"},
        "feed_mode": {"G93", "G94", "G95"},
        "units": {"G20", "G21"},
        "wcs": {"G54", "G55", "G56", "G57", "G58", "G59", "G59.1", "G59.2", "G59.3"},
        "spindle_misc": {"M3", "M03", "M4", "M04", "M5", "M05"},
        "coolant": {"M7", "M07", "M8", "M08", "M9", "M09", "M50"},
        "tool_change": {"M6", "M06", "M0", "M00", "M1", "M01"},
        "prog_end": {"M2", "M02", "M30"},
        "canned_cycle_return": {"G98", "G99"},
        "canned_cycle_cancel": {"G80"},
        "exact_stop": {"G09", "G61", "G64"},
    },
    supported_canned_cycles={"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"},
    supports_g98_g99=True,
    cutter_comp=CutterCompCapability(
        supported=True,
        requires_d_register=True,
        max_d_register=99,
        valid_commands={"G40", "G41", "G42"},
        supported_modes={"wear", "control", "computer"},
    ),
    tool_length_comp=ToolLengthCompCapability(
        supported=True,
        command="G43",
        cancel_command="G49",
        requires_h_register=True,
        max_h_register=99,
    ),
    lookahead=LookaheadCapability(
        supported=True,
        syntax_type="aicc",
        valid_codes={"G05.1", "G5.1"},
        on_pattern="G05.1 Q1",
        off_pattern="G05.1 Q0",
    ),
    safe_return=SafeReturnCapability(
        syntax_type="g28_incremental",
        valid_codes={"G28", "G91", "G30"},
        z_retract_command="G91 G28 Z0.0",
        xy_home_command="G28 X0.0 Y0.0",
    ),
    arc=ArcCapability(
        requires_ij_center=True,
        supports_r_word=True,
        min_radius_mm=0.005,
        max_radius_mm=100000.0,
    ),
    block_format=BlockFormatCapability(
        requires_percent_enclosure=True,
        comment_style="parentheses",
        allow_line_numbers=True,
    ),
    program_numbering=ProgramNumberingCapability(
        supported=True,
        style="O_4digit",
        prefix="O",
        min_number=1,
        max_number=9999,
        digits=4,
    ),
    tool_change=ToolChangeCapability(
        strategy="atc",
        command="M6",
    ),
    max_feed_mm_min=30000.0,
    max_spindle_rpm=24000.0,
)


HAAS_CAPABILITY_MODEL = ControllerCapabilityModel(
    controller_id="haas",
    name="Haas NextGen / VF-Series CNC Mill",
    supported_modal_groups={
        "motion": {"G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G4"},
        "plane": {"G17", "G18", "G19"},
        "distance": {"G90", "G91"},
        "feed_mode": {"G93", "G94", "G95"},
        "units": {"G20", "G21"},
        "wcs": {"G54", "G55", "G56", "G57", "G58", "G59", "G154", "G59.1", "G59.2", "G59.3"},
        "machine_coords": {"G53"},
        "spindle_misc": {"M3", "M03", "M4", "M04", "M5", "M05"},
        "coolant": {"M7", "M07", "M8", "M08", "M9", "M09", "M50", "M88"},
        "tool_change": {"M6", "M06", "M0", "M00", "M1", "M01"},
        "prog_end": {"M2", "M02", "M30"},
        "canned_cycle_return": {"G98", "G99"},
        "canned_cycle_cancel": {"G80"},
    },
    supported_canned_cycles={"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"},
    supports_g98_g99=True,
    cutter_comp=CutterCompCapability(
        supported=True,
        requires_d_register=True,
        max_d_register=200,
        valid_commands={"G40", "G41", "G42"},
        supported_modes={"wear", "control", "computer"},
        offset_memory_type="type_a_combined",
    ),
    tool_length_comp=ToolLengthCompCapability(
        supported=True,
        command="G43",
        cancel_command="G49",
        requires_h_register=True,
        max_h_register=200,
    ),
    lookahead=LookaheadCapability(
        supported=True,
        syntax_type="hsm",
        valid_codes={"G187"},
        on_pattern="G187 P{mode}",
        off_pattern="G187",
    ),
    safe_return=SafeReturnCapability(
        syntax_type="g53_machine_coords",
        valid_codes={"G53", "G0", "G00"},
        z_retract_command="G53 G00 Z0.0",
        xy_home_command="G53 G00 X0.0 Y0.0",
    ),
    arc=ArcCapability(
        requires_ij_center=True,
        supports_r_word=True,
        min_radius_mm=0.005,
        max_radius_mm=100000.0,
    ),
    block_format=BlockFormatCapability(
        requires_percent_enclosure=True,
        comment_style="parentheses",
        allow_line_numbers=True,
    ),
    program_numbering=ProgramNumberingCapability(
        supported=True,
        style="O_5digit",
        prefix="O",
        min_number=1,
        max_number=99999,
        digits=5,
    ),
    tool_change=ToolChangeCapability(
        strategy="atc",
        command="M6",
    ),
    max_feed_mm_min=30000.0,
    max_spindle_rpm=24000.0,
)


GRBL_CAPABILITY_MODEL = ControllerCapabilityModel(
    controller_id="grbl",
    name="GRBL v1.1 Open-Source CNC Controller",
    supported_modal_groups={
        "motion": {"G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G4"},
        "plane": {"G17", "G18", "G19"},
        "distance": {"G90", "G91"},
        "feed_mode": {"G93", "G94"},
        "units": {"G20", "G21"},
        "wcs": {"G54", "G55", "G56", "G57", "G58", "G59"},
        "spindle_misc": {"M3", "M03", "M4", "M04", "M5", "M05"},
        "coolant": {"M7", "M07", "M8", "M08", "M9", "M09"},
        "tool_change": {"M6", "M06", "M0", "M00", "M1", "M01"},
        "prog_end": {"M2", "M02", "M30"},
        "canned_cycle_cancel": {"G80"},
        "modal_defaults": {"G40", "G49"},
    },
    supported_canned_cycles=set(),  # Standard GRBL has NO canned cycle support (G81-G89 forbidden)
    supports_g98_g99=False,        # G98/G99 unsupported in GRBL
    cutter_comp=CutterCompCapability(
        supported=False,           # Standard GRBL does not support runtime G41/G42
        requires_d_register=False,
        max_d_register=0,
        valid_commands={"G40"},   # Only G40 cancellation accepted in preamble
        supported_modes={"computer"},
    ),
    tool_length_comp=ToolLengthCompCapability(
        supported=True,
        command="G43",
        cancel_command="G49",
        requires_h_register=True,
        max_h_register=99,
        valid_commands={"G43", "G49", "G43.1"},
    ),
    lookahead=LookaheadCapability(
        supported=False,           # GRBL manages lookahead in hardware planner; no NC lookahead codes
        syntax_type="none",
        valid_codes=set(),
        on_pattern="",
        off_pattern="",
    ),
    safe_return=SafeReturnCapability(
        syntax_type="g0_retract",
        valid_codes={"G0", "G00"},
        z_retract_command="G0 Z{safe_z}",
        xy_home_command="G0 X0 Y0",
        program_end_command="M2",
        retract_to_setup_clearance_first=True,
    ),
    arc=ArcCapability(
        requires_ij_center=True,
        supports_r_word=True,
        min_radius_mm=0.01,
        max_radius_mm=10000.0,
    ),
    block_format=BlockFormatCapability(
        requires_percent_enclosure=False,
        comment_style="semicolon",
        allow_line_numbers=False,
    ),
    program_numbering=ProgramNumberingCapability(
        supported=False,
        required=False,
        style="none",
        prefix="",
        min_number=0,
        max_number=0,
        digits=0,
    ),
    tool_change=ToolChangeCapability(
        strategy="manual",
        command="M0",
    ),
    max_feed_mm_min=10000.0,
    max_spindle_rpm=24000.0,
)


LINUXCNC_CAPABILITY_MODEL = ControllerCapabilityModel(
    controller_id="linuxcnc",
    name="LinuxCNC / AXIS (RS274/NGC Standard)",
    supported_modal_groups={
        "motion": {"G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G4"},
        "plane": {"G17", "G18", "G19"},
        "distance": {"G90", "G91"},
        "feed_mode": {"G93", "G94", "G95"},
        "units": {"G20", "G21"},
        "wcs": {"G54", "G55", "G56", "G57", "G58", "G59", "G59.1", "G59.2", "G59.3"},
        "spindle_misc": {"M3", "M03", "M4", "M04", "M5", "M05"},
        "coolant": {"M7", "M07", "M8", "M08", "M9", "M09"},
        "tool_change": {"M6", "M06", "M0", "M00", "M1", "M01"},
        "prog_end": {"M2", "M02", "M30"},
        "canned_cycle_return": {"G98", "G99"},
        "canned_cycle_cancel": {"G80"},
    },
    supported_canned_cycles={"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"},
    supports_g98_g99=True,
    cutter_comp=CutterCompCapability(
        supported=True,
        requires_d_register=True,
        max_d_register=99,
        valid_commands={"G40", "G41", "G42", "G41.1", "G42.1"},
        supported_modes={"wear", "control", "computer"},
    ),
    tool_length_comp=ToolLengthCompCapability(
        supported=True,
        command="G43",
        cancel_command="G49",
        requires_h_register=True,
        max_h_register=99,
        valid_commands={"G43", "G49", "G43.1"},
    ),
    lookahead=LookaheadCapability(
        supported=True,
        syntax_type="naive_cam",
        valid_codes={"G64", "G61", "G61.1"},
        on_pattern="G64 P{tol} Q{tol}",
        off_pattern="G64",
    ),
    safe_return=SafeReturnCapability(
        syntax_type="g0_retract",
        valid_codes={"G0", "G00", "G53", "G28"},
        z_retract_command="G0 Z{safe_z}",
        xy_home_command="G0 X0 Y0",
        program_end_command="M2",
        retract_to_setup_clearance_first=True,
    ),
    arc=ArcCapability(
        requires_ij_center=True,
        supports_r_word=True,
        min_radius_mm=0.005,
        max_radius_mm=100000.0,
    ),
    block_format=BlockFormatCapability(
        requires_percent_enclosure=False,
        comment_style="parentheses",
        allow_line_numbers=True,
    ),
    program_numbering=ProgramNumberingCapability(
        supported=True,
        required=False,
        style="O_4digit",
        prefix="O",
        min_number=1,
        max_number=9999,
        digits=4,
    ),
    tool_change=ToolChangeCapability(
        strategy="atc",
        command="M6",
    ),
    max_feed_mm_min=20000.0,
    max_spindle_rpm=24000.0,
)


CONTROLLER_CAPABILITY_REGISTRY: Dict[str, ControllerCapabilityModel] = {
    "fanuc": FANUC_CAPABILITY_MODEL,
    "haas": HAAS_CAPABILITY_MODEL,
    "grbl": GRBL_CAPABILITY_MODEL,
    "linuxcnc": LINUXCNC_CAPABILITY_MODEL,
    "linux": LINUXCNC_CAPABILITY_MODEL,
    "ngc": LINUXCNC_CAPABILITY_MODEL,
}


HEIDENHAIN_CAPABILITY_MODEL = ControllerCapabilityModel(
    controller_id="heidenhain",
    name="Heidenhain TNC Series (Conversational Klartext / ISO)",
    supported_modal_groups={
        "motion": {"L", "C", "CC", "CR", "CT", "RND", "CHF", "G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G4"},
        "plane": {"PLANE SPATIAL", "PLANE VECTOR", "PLANE POINTS", "PLANE RELATIV", "G17", "G18", "G19"},
        "distance": {"G90", "G91", "IX", "IY", "IZ", "IV"},
        "feed_mode": {"FMAX", "FAUTO", "FU", "FZ", "G94", "G95"},
        "units": {"MM", "INCH", "G70", "G71", "G20", "G21"},
        "wcs": {"CYCL DEF 247", "CYCL DEF 7", "TRANS DATUM", "G54", "G55", "G56", "G57", "G58", "G59"},
        "machine_coords": {"M91", "M92", "G53"},
        "spindle_misc": {"M3", "M03", "M4", "M04", "M5", "M05", "M13", "M14"},
        "coolant": {"M7", "M07", "M8", "M08", "M9", "M09", "M13"},
        "tool_change": {"TOOL CALL", "TOOL DEF", "M6", "M06", "M0", "M00", "M1", "M01"},
        "prog_end": {"M2", "M02", "M30", "END PGM"},
        "canned_cycle_cancel": {"CYCL DEF 0", "G80"},
    },
    supported_canned_cycles={
        "200", "201", "202", "203", "204", "205", "206", "207", "208", "209",
        "CYCL DEF 200", "CYCL DEF 203", "CYCL DEF 205", "CYCL DEF 208", "CYCL DEF 32", "CYCL DEF 247",
        "G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76",
    },
    supports_g98_g99=False,
    cutter_comp=CutterCompCapability(
        supported=True,
        requires_d_register=False,
        max_d_register=0,
        valid_commands={"RL", "RR", "R0", "G40", "G41", "G42"},
        supported_modes={"wear", "control", "computer"},
        offset_memory_type="heidenhain_tool_table",
    ),
    tool_length_comp=ToolLengthCompCapability(
        supported=True,
        command="TOOL CALL",
        cancel_command="",
        requires_h_register=False,
        max_h_register=0,
        valid_commands={"TOOL CALL", "DL", "DR", "G43", "G49"},
    ),
    lookahead=LookaheadCapability(
        supported=True,
        syntax_type="cycl_def_32",
        valid_codes={"CYCL DEF 32", "CYCL DEF 32.0", "CYCL DEF 32.1"},
        on_pattern="CYCL DEF 32.0 TOLERANCE ~ CYCL DEF 32.1 T{tol}",
        off_pattern="CYCL DEF 32.0 TOLERANCE ~ CYCL DEF 32.1 T0",
    ),
    safe_return=SafeReturnCapability(
        syntax_type="m91_machine_coords",
        valid_codes={"M91", "M92", "G53", "L", "FMAX"},
        z_retract_command="L Z+0 R0 FMAX M91",
        xy_home_command="L X+0 Y+0 R0 FMAX M91",
        program_end_command="M30",
    ),
    arc=ArcCapability(
        requires_ij_center=True,
        supports_r_word=True,
        min_radius_mm=0.005,
        max_radius_mm=100000.0,
    ),
    block_format=BlockFormatCapability(
        requires_percent_enclosure=False,
        comment_style="semicolon",
        allow_line_numbers=False,
    ),
    program_numbering=ProgramNumberingCapability(
        supported=True,
        required=False,
        style="named_or_number",
        prefix="",
        min_number=0,
        max_number=99999999,
        digits=8,
    ),
    tool_change=ToolChangeCapability(
        strategy="atc",
        command="TOOL CALL",
    ),
    max_feed_mm_min=40000.0,
    max_spindle_rpm=30000.0,
)


SIEMENS_CAPABILITY_MODEL = ControllerCapabilityModel(
    controller_id="siemens",
    name="Siemens Sinumerik (840D/828D)",
    supported_modal_groups={
        "motion": {"G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G4", "CIP"},
        "plane": {"G17", "G18", "G19"},
        "distance": {"G90", "G91"},
        "feed_mode": {"G93", "G94", "G95", "G96", "G97"},
        "units": {"G70", "G71", "G700", "G710", "G20", "G21"},
        "wcs": {"G54", "G55", "G56", "G57", "G58", "G59", "G500", "TRANS", "ATRANS", "ROT", "AROT", "SCALE", "MIRROR"},
        "machine_coords": {"SUPA", "G53", "G153"},
        "spindle_misc": {"M3", "M03", "M4", "M04", "M5", "M05", "SPOS"},
        "coolant": {"M7", "M07", "M8", "M08", "M9", "M09"},
        "tool_change": {"M6", "M06", "M0", "M00", "M1", "M01"},
        "prog_end": {"M2", "M02", "M30"},
        "canned_cycle_cancel": {"MCALL", "G80"},
        "exact_stop": {"G60", "G601", "G602", "G603", "G64", "G641", "G642", "G643", "G644", "G645", "SOFT", "BRISK"},
    },
    supported_canned_cycles={
        "CYCLE81", "CYCLE82", "CYCLE83", "CYCLE84", "CYCLE85", "CYCLE86", "CYCLE840", "CYCLE800", "CYCLE832", "HOLES1", "HOLES2",
        "G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76",
    },
    supports_g98_g99=False,
    cutter_comp=CutterCompCapability(
        supported=True,
        requires_d_register=True,
        max_d_register=9,
        valid_commands={"G40", "G41", "G42"},
        supported_modes={"wear", "control", "computer"},
        offset_memory_type="siemens_d_register",
    ),
    tool_length_comp=ToolLengthCompCapability(
        supported=True,
        command="D1",
        cancel_command="D0",
        requires_h_register=False,
        max_h_register=9,
        valid_commands={"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D0", "G43", "G49"},
    ),
    lookahead=LookaheadCapability(
        supported=True,
        syntax_type="cycle832",
        valid_codes={"CYCLE832", "FFWON", "COMPCAD", "COMPSURF"},
        on_pattern="CYCLE832({tol}, 1, 1)",
        off_pattern="CYCLE832()",
    ),
    safe_return=SafeReturnCapability(
        syntax_type="supa_machine_coords",
        valid_codes={"SUPA", "G53", "G0", "G00", "D0"},
        z_retract_command="SUPA G0 Z0.0 D0",
        xy_home_command="SUPA G0 X0.0 Y0.0",
        program_end_command="M30",
    ),
    arc=ArcCapability(
        requires_ij_center=True,
        supports_r_word=True,
        min_radius_mm=0.005,
        max_radius_mm=100000.0,
    ),
    block_format=BlockFormatCapability(
        requires_percent_enclosure=False,
        comment_style="semicolon",
        allow_line_numbers=True,
    ),
    program_numbering=ProgramNumberingCapability(
        supported=False,
        required=False,
        style="none",
        prefix="",
        min_number=0,
        max_number=0,
        digits=0,
    ),
    tool_change=ToolChangeCapability(
        strategy="atc",
        command="M6",
    ),
    max_feed_mm_min=40000.0,
    max_spindle_rpm=30000.0,
)

SINUMERIK_CAPABILITY_MODEL = SIEMENS_CAPABILITY_MODEL

CONTROLLER_CAPABILITY_REGISTRY["heidenhain"] = HEIDENHAIN_CAPABILITY_MODEL
CONTROLLER_CAPABILITY_REGISTRY["klartext"] = HEIDENHAIN_CAPABILITY_MODEL
CONTROLLER_CAPABILITY_REGISTRY["tnc"] = HEIDENHAIN_CAPABILITY_MODEL
CONTROLLER_CAPABILITY_REGISTRY["siemens"] = SIEMENS_CAPABILITY_MODEL
CONTROLLER_CAPABILITY_REGISTRY["sinumerik"] = SIEMENS_CAPABILITY_MODEL
CONTROLLER_CAPABILITY_REGISTRY["840d"] = SIEMENS_CAPABILITY_MODEL
CONTROLLER_CAPABILITY_REGISTRY["828d"] = SIEMENS_CAPABILITY_MODEL


@dataclass
class ToolChangePosition:
    """Defines where the machine moves for tool changes."""
    x: Optional[float] = None
    y: Optional[float] = None
    z: float = 100.0          # Safe Z for tool change
    retract_policy: str = "always"  # "always" | "when_established" | "below" | "never"


@dataclass
class WcsBehavior:
    """Work coordinate system behavior settings."""
    default_wcs: str = "G54"
    allow_multiple: bool = True
    require_explicit: bool = False  # Require WCS on each operation


@dataclass
class CannedCycleBehavior:
    """Canned cycle behavior settings."""
    supported_cycles: frozenset[str] = frozenset()
    supports_g98_g99: bool = True
    default_return_mode: str = "G98"  # "G98" (initial) | "G99" (R-plane)
    peck_drilling_type: str = "G83"   # Default peck cycle type


@dataclass
class SpindleFeedLimits:
    """Spindle and feed rate limits for the machine."""
    max_spindle_rpm: float = 24000.0
    min_spindle_rpm: float = 50.0
    max_feed_mm_min: float = 30000.0
    max_rapid_mm_min: float = 30000.0
    default_rapid_feed_mm_min: float = 3000.0


@dataclass
class SafetyRetractStrategy:
    """Safety retract / safe return strategy."""
    syntax_type: str = "g28_incremental"  # "g28_incremental" | "g53_machine_coords" | "supa_machine_coords" | "g0_retract" | "custom"
    z_retract_command: str = "G91 G28 Z0.0"
    xy_home_command: str = "G28 X0.0 Y0.0"
    program_end_command: str = "M30"
    retract_to_setup_clearance_first: bool = True


@dataclass
class MachineProfile:
    """Complete machine profile combining controller capabilities with machine-specific settings.

    This replaces hardcoded values in post-processors with machine-driven configuration.
    """
    name: str
    controller: ControllerCapabilityModel
    tool_change: ToolChangePosition = field(default_factory=ToolChangePosition)
    wcs: WcsBehavior = field(default_factory=WcsBehavior)
    canned_cycles: CannedCycleBehavior = field(default_factory=CannedCycleBehavior)
    spindle_feed_limits: SpindleFeedLimits = field(default_factory=SpindleFeedLimits)
    safety: SafetyRetractStrategy = field(default_factory=SafetyRetractStrategy)
    hsm_default_mode: str = "roughing"  # "roughing" | "medium" | "finishing" | "none"
    cutter_comp_default_mode: str = "wear"
    tool_length_comp_default: str = "G43"
    enable_tool_length_comp: bool = True
    default_rapid_feed_mm_min: float = 3000.0
    coolant_default: str = "flood"  # "flood" | "mist" | "through_spindle" | "none"
    clear_offsets_command: str = "G49 G40"  # Controller-specific clear offsets command

    @property
    def controller_id(self) -> str:
        return self.controller.controller_id

    @property
    def supported_canned_cycles(self) -> frozenset[str]:
        return self.canned_cycles.supported_cycles or frozenset(self.controller.supported_canned_cycles)

    @property
    def supported_cycles(self) -> set[str]:
        return set(self.supported_canned_cycles)

    @property
    def tool_change_strategy(self) -> str:
        return self.controller.tool_change.strategy

    @property
    def tool_change_command(self) -> str:
        return self.controller.tool_change.command

    @property
    def lookahead_on_pattern(self) -> str:
        return self.controller.lookahead.on_pattern

    @property
    def lookahead_off_pattern(self) -> str:
        return self.controller.lookahead.off_pattern

    @property
    def safe_return_z_command(self) -> str:
        return self.safety.z_retract_command

    @property
    def safe_return_xy_command(self) -> str:
        return self.safety.xy_home_command

    @property
    def program_end_command(self) -> str:
        return self.safety.program_end_command

    @property
    def cutter_comp_modes(self) -> set[str]:
        return self.controller.cutter_comp.supported_modes

    @property
    def offset_memory_type(self) -> str:
        return self.controller.cutter_comp.offset_memory_type

    @property
    def max_spindle_rpm(self) -> float:
        return min(self.spindle_feed_limits.max_spindle_rpm, self.controller.max_spindle_rpm)

    @property
    def max_feed_mm_min(self) -> float:
        return min(self.spindle_feed_limits.max_feed_mm_min, self.controller.max_feed_mm_min)

    @property
    def max_rapid_mm_min(self) -> float:
        return min(self.spindle_feed_limits.max_rapid_mm_min, self.controller.max_feed_mm_min)

    def get_safe_z(self, setup_clearance: Optional[float] = None, fixture_clearance: Optional[float] = None) -> float:
        """Calculate safe retract Z from profile and setup."""
        candidates = [self.tool_change.z]
        if setup_clearance is not None:
            candidates.append(setup_clearance)
        if fixture_clearance is not None:
            candidates.append(fixture_clearance)
        return max(candidates) if candidates else self.tool_change.z

    def get_tool_change_retract_z(self, is_pos_established: bool = True) -> Optional[float]:
        """Get tool change retract Z based on policy."""
        if self.tool_change.retract_policy == "never":
            return None
        if self.tool_change.retract_policy == "always":
            return self.tool_change.z
        if self.tool_change.retract_policy == "when_established":
            return self.tool_change.z if is_pos_established else None
        # "below" - retract only when below the retract plane
        return self.tool_change.z


# ---------------------------------------------------------------------------
# Predefined Machine Profiles
# ---------------------------------------------------------------------------

# Fanuc 3-axis mill with ATC
FANUC_MACHINE_PROFILE = MachineProfile(
    name="Fanuc 3-Axis Mill (Generic ATC)",
    controller=FANUC_CAPABILITY_MODEL,
    tool_change=ToolChangePosition(z=100.0, retract_policy="below"),
    wcs=WcsBehavior(default_wcs="G54"),
    canned_cycles=CannedCycleBehavior(
        supported_cycles=frozenset({"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"}),
        supports_g98_g99=True,
        default_return_mode="G98",
        peck_drilling_type="G83",
    ),
    spindle_feed_limits=SpindleFeedLimits(max_spindle_rpm=24000.0, max_feed_mm_min=30000.0),
    safety=SafetyRetractStrategy(
        syntax_type="g28_incremental",
        z_retract_command="G91 G28 Z0.0",
        xy_home_command="G28 X0.0 Y0.0",
        program_end_command="M30",
    ),
    hsm_default_mode="roughing",
    clear_offsets_command="G49 G40",
)

# Haas VF-series with ATC
HAAS_MACHINE_PROFILE = MachineProfile(
    name="Haas VF-Series (Generic ATC)",
    controller=HAAS_CAPABILITY_MODEL,
    tool_change=ToolChangePosition(z=100.0, retract_policy="always"),
    wcs=WcsBehavior(default_wcs="G54"),
    canned_cycles=CannedCycleBehavior(
        supported_cycles=frozenset({"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"}),
        supports_g98_g99=True,
        default_return_mode="G98",
        peck_drilling_type="G83",
    ),
    spindle_feed_limits=SpindleFeedLimits(max_spindle_rpm=24000.0, max_feed_mm_min=30000.0),
    safety=SafetyRetractStrategy(
        syntax_type="g53_machine_coords",
        z_retract_command="G53 G00 Z0.0",
        xy_home_command="G53 G00 X0.0 Y0.0",
        program_end_command="M30",
    ),
    hsm_default_mode="roughing",
)

# GRBL (manual tool change, no canned cycles)
GRBL_MACHINE_PROFILE = MachineProfile(
    name="GRBL v1.1 (Manual Tool Change)",
    controller=GRBL_CAPABILITY_MODEL,
    tool_change=ToolChangePosition(z=20.0, retract_policy="when_established"),
    wcs=WcsBehavior(default_wcs="G54"),
    canned_cycles=CannedCycleBehavior(
        supported_cycles=frozenset(),
        supports_g98_g99=False,
    ),
    spindle_feed_limits=SpindleFeedLimits(max_spindle_rpm=24000.0, max_feed_mm_min=10000.0, max_rapid_mm_min=10000.0),
    safety=SafetyRetractStrategy(
        syntax_type="g0_retract",
        z_retract_command="G0 Z{safe_z}",
        xy_home_command="G0 X0 Y0",
        program_end_command="M2",
    ),
    hsm_default_mode="none",
    enable_tool_length_comp=True,
)

# LinuxCNC
LINUXCNC_MACHINE_PROFILE = MachineProfile(
    name="LinuxCNC / AXIS (RS274/NGC)",
    controller=LINUXCNC_CAPABILITY_MODEL,
    tool_change=ToolChangePosition(z=50.0, retract_policy="always"),
    wcs=WcsBehavior(default_wcs="G54"),
    canned_cycles=CannedCycleBehavior(
        supported_cycles=frozenset({"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"}),
        supports_g98_g99=True,
        default_return_mode="G98",
    ),
    spindle_feed_limits=SpindleFeedLimits(max_spindle_rpm=24000.0, max_feed_mm_min=20000.0),
    safety=SafetyRetractStrategy(
        syntax_type="g0_retract",
        z_retract_command="G0 Z{safe_z}",
        xy_home_command="G0 X0 Y0",
        program_end_command="M2",
    ),
)

# Heidenhain Conversational
HEIDENHAIN_MACHINE_PROFILE = MachineProfile(
    name="Heidenhain TNC (Conversational Klartext)",
    controller=HEIDENHAIN_CAPABILITY_MODEL,
    tool_change=ToolChangePosition(z=50.0, retract_policy="always"),
    wcs=WcsBehavior(default_wcs="CYCL DEF 247"),
    canned_cycles=CannedCycleBehavior(
        supported_cycles=frozenset({"G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76", "200", "203", "205", "208", "CYCL DEF 200", "CYCL DEF 203", "CYCL DEF 205", "CYCL DEF 208"}),
        supports_g98_g99=False,
        peck_drilling_type="205",
    ),
    spindle_feed_limits=SpindleFeedLimits(max_spindle_rpm=24000.0, max_feed_mm_min=30000.0),
    safety=SafetyRetractStrategy(
        syntax_type="custom",
        z_retract_command="L Z{safe_z} R0 FMAX M9",
        xy_home_command="L X+0 Y+0 R0 FMAX M91",
        program_end_command="M30",
    ),
    hsm_default_mode="roughing",
    cutter_comp_default_mode="wear",
)

# Siemens Sinumerik
SIEMENS_MACHINE_PROFILE = MachineProfile(
    name="Siemens Sinumerik 840D/828D",
    controller=SIEMENS_CAPABILITY_MODEL,
    tool_change=ToolChangePosition(z=50.0, retract_policy="when_established"),
    wcs=WcsBehavior(default_wcs="G54"),
    canned_cycles=CannedCycleBehavior(
        supported_cycles=frozenset({"CYCLE81", "CYCLE82", "CYCLE83", "CYCLE84", "CYCLE85", "CYCLE86", "CYCLE832", "G81", "G82", "G83", "G73", "G84", "G85", "G86", "G76"}),
        supports_g98_g99=False,
        peck_drilling_type="CYCLE83",
    ),
    spindle_feed_limits=SpindleFeedLimits(max_spindle_rpm=30000.0, max_feed_mm_min=40000.0, max_rapid_mm_min=40000.0),
    safety=SafetyRetractStrategy(
        syntax_type="supa_machine_coords",
        z_retract_command="SUPA G0 Z0.0 D0",
        xy_home_command="SUPA G0 X0.0 Y0.0",
        program_end_command="M30",
    ),
    hsm_default_mode="roughing",
)

MACHINE_PROFILE_REGISTRY: Dict[str, MachineProfile] = {
    "fanuc": FANUC_MACHINE_PROFILE,
    "haas": HAAS_MACHINE_PROFILE,
    "grbl": GRBL_MACHINE_PROFILE,
    "linuxcnc": LINUXCNC_MACHINE_PROFILE,
    "linux": LINUXCNC_MACHINE_PROFILE,
    "heidenhain": HEIDENHAIN_MACHINE_PROFILE,
    "klartext": HEIDENHAIN_MACHINE_PROFILE,
    "tnc": HEIDENHAIN_MACHINE_PROFILE,
    "siemens": SIEMENS_MACHINE_PROFILE,
    "sinumerik": SIEMENS_MACHINE_PROFILE,
    "840d": SIEMENS_MACHINE_PROFILE,
    "828d": SIEMENS_MACHINE_PROFILE,
}


def get_machine_profile(controller_name: str) -> MachineProfile:
    """Retrieve a machine profile by controller name."""
    key = controller_name.lower().strip()
    for reg_key, profile in MACHINE_PROFILE_REGISTRY.items():
        if reg_key in key:
            return profile
    # Default fallback
    return GRBL_MACHINE_PROFILE


def get_controller_capability_model(controller_name: str) -> ControllerCapabilityModel:
    """Retrieve the formal capability model for a controller by name."""
    key = controller_name.lower().strip()
    for reg_key, model in CONTROLLER_CAPABILITY_REGISTRY.items():
        if reg_key in key:
            return model
    return GRBL_CAPABILITY_MODEL
