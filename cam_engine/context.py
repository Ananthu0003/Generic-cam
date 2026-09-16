"""Planning context: the single source of truth for the complete machining state.

Every downstream component consumes this context. No component invents values:
missing required data produces an explicit error, never a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .coords import CoordSpace, CoordinateSystemChain, make_transform
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
    FACE_MILL = "face_mill"
    BORING_BAR = "boring_bar"
    CHAMFER_MILL = "chamfer_mill"
    COUNTERSINK_TOOL = "countersink_tool"  # dedicated countersink / spotfacing tool
    TAP = "tap"
    REAMER = "reamer"
    THREAD_MILL = "thread_mill"
    GROOVE_CUTTER = "groove_cutter"


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
    flutes: int = 2
    material: str = "carbide"     # tool material key into cutting-data tables
    can_plunge: bool = False      # center-cutting capability
    holder: Optional[ToolHolder] = None
    length_offset_h: Optional[int] = None   # machine H offset register (config, not assumed)
    diameter_offset_d: Optional[int] = None  # machine D offset register

    def validate(self) -> None:
        problems = []
        if self.diameter <= 0:
            problems.append("diameter must be positive")
        if self.flute_length <= 0:
            problems.append("flute_length must be positive")
        if self.overall_length < self.flute_length:
            problems.append("overall_length < flute_length")
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


DEFAULT_MATERIALS: dict[str, Material] = {
    "aluminum_6061": Material(
        id="aluminum_6061", name="Aluminum 6061-T6", hardness_hb=95,
        specific_cutting_force_kn_mm2=0.7, v_carbide=450.0, v_hss=120.0,
        max_hm_ratio=1.0, max_ae_ratio=0.75, needs_coolant=True),
    "steel_1018": Material(
        id="steel_1018", name="Steel 1018 Low Carbon", hardness_hb=126,
        specific_cutting_force_kn_mm2=1.5, v_carbide=180.0, v_hss=35.0,
        max_hm_ratio=0.75, max_ae_ratio=0.7, needs_coolant=True),
    "steel_4140": Material(
        id="steel_4140", name="Steel 4140 Pre-Hard", hardness_hb=280,
        specific_cutting_force_kn_mm2=2.1, v_carbide=110.0, v_hss=20.0,
        max_hm_ratio=0.5, max_ae_ratio=0.6, needs_coolant=True),
    "stainless_304": Material(
        id="stainless_304", name="Stainless 304", hardness_hb=170,
        specific_cutting_force_kn_mm2=2.0, v_carbide=95.0, v_hss=18.0,
        max_hm_ratio=0.5, max_ae_ratio=0.55, needs_coolant=True),
    "brass_360": Material(
        id="brass_360", name="Brass 360 Free Machining", hardness_hb=80,
        specific_cutting_force_kn_mm2=0.65, v_carbide=320.0, v_hss=90.0,
        max_hm_ratio=1.0, max_ae_ratio=0.75, needs_coolant=False),
    "delrin_acetal": Material(
        id="delrin_acetal", name="Delrin Acetal", hardness_hb=20,
        specific_cutting_force_kn_mm2=0.35, v_carbide=500.0, v_hss=160.0,
        max_hm_ratio=1.2, max_ae_ratio=0.8, needs_coolant=False),
}


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
class Setup:
    """One machining setup: orientation of model+stock in setup coordinates,
    work offset selection, and clamping constraints."""

    id: str
    name: str
    model_to_setup: np.ndarray          # rigid transform, validated
    work_offset: WorkOffset
    stock: Stock
    fixtures: list[Fixture | TriangleMesh] = field(default_factory=list)  # clamping geometry, setup space
    comment: str = ""


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
