"""Semantic toolpath representation: controller-independent machining intent.

Segments carry full provenance (operation, feature, tool, coordinate space) so
post-processing, validation, simulation, and the UI can all trace back.
No G-code strings ever appear here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from ..coords import CoordSpace


class MotionType(str, Enum):
    RAPID = "rapid"
    CUT = "cut"
    PLUNGE = "plunge"
    RAMP = "ramp"
    HELIX = "helix"
    ARC_CW = "arc_cw"
    ARC_CCW = "arc_ccw"
    ENTRY = "entry"
    EXIT = "exit"
    RETRACT = "retract"
    LINK = "link"
    DWELL = "dwell"
    TOOL_CHANGE = "tool_change"
    SPINDLE_START = "spindle_start"
    SPINDLE_STOP = "spindle_stop"
    COOLANT_ON = "coolant_on"
    COOLANT_OFF = "coolant_off"


class CompensationSide(str, Enum):
    """Cutter radius compensation offset side relative to direction of tool motion."""
    LEFT = "left"      # G41 (climb milling)
    RIGHT = "right"    # G42 (conventional milling)
    OFF = "off"        # G40 (cancelled / none)


class ToolpathRepresentation(str, Enum):
    """The authoritative geometric representation contract of toolpath coordinate points.
    
    Explicit policy:
    - CENTER_PATH (TOOL_CENTER): Points represent the tool centerline trajectory.
      Normally no geometric cutter-radius offset applied by the machine controller.
      May use WEAR compensation for minor wear delta adjustments (CNC D = 0.0 + delta).
    - PART_BOUNDARY_PATH (NOMINAL_BOUNDARY): Points lie directly on the nominal CAD part boundary.
      Requires the CNC controller to apply full cutter-radius compensation (CNC D = Tool Radius).
    - MACHINE_COMPENSATED: Points are fully machine-compensated; no further controller offset.
    """
    CENTER_PATH = "center_path"
    PART_BOUNDARY_PATH = "part_boundary_path"
    TOOL_CENTER = "tool_center"
    NOMINAL_BOUNDARY = "nominal_boundary"
    MACHINE_COMPENSATED = "machine_compensated"


# Backward compatibility aliases
ToolpathGeometryMode = ToolpathRepresentation
PathGeometryType = ToolpathRepresentation


class CompensationMode(str, Enum):
    """Explicit controller cutter radius compensation mode.
    
    Modes:
    - NONE: No G41/G42 emitted in G-code (CAM calculates tool center path; G40 only).
    - WEAR: Toolpath is programmed as CENTER_PATH; G41/G42 emitted, CNC D-register holds 0.0 + wear delta.
    - GEOMETRY: Toolpath is programmed as PART_BOUNDARY_PATH; G41/G42 emitted, CNC D-register holds full tool radius.
    """
    NONE = "none"
    WEAR = "wear"
    GEOMETRY = "geometry"
    OFF = "none"
    COMPUTER = "none"
    CONTROL = "geometry"


# Backward compatibility alias
CompensationType = CompensationMode


@dataclass
class CutterCompensationPolicy:
    """Explicit cutter compensation policy mapping representation, mode, and independent T/H/D registers."""

    representation: ToolpathRepresentation = ToolpathRepresentation.CENTER_PATH
    mode: CompensationMode = CompensationMode.NONE
    side: CompensationSide = CompensationSide.LEFT
    tool_number: int = 1
    length_offset_h: Optional[int] = None
    diameter_offset_d: Optional[int] = None

    @property
    def emits_g41_g42(self) -> bool:
        """Never emit G41/G42 simply because an operation is finishing. Only emit for WEAR or GEOMETRY mode."""
        return self.mode in (CompensationMode.WEAR, CompensationMode.GEOMETRY, CompensationMode.CONTROL)

    @property
    def effective_h(self) -> int:
        """Independent length offset register H (never assumed H == T)."""
        return self.length_offset_h if (self.length_offset_h is not None and self.length_offset_h > 0) else self.tool_number

    @property
    def effective_d(self) -> int:
        """Independent diameter offset register D (never assumed D == T or D == H)."""
        return self.diameter_offset_d if (self.diameter_offset_d is not None and self.diameter_offset_d > 0) else self.tool_number



@dataclass
class MotionSegment:
    """One semantic motion segment in a defined coordinate space with full operation-to-geometry traceability."""

    motion_type: MotionType
    start: np.ndarray                  # (3,) in `space`
    end: np.ndarray                    # (3,)
    space: CoordSpace = CoordSpace.SETUP
    feed: Optional[float] = None       # mm/min (cut/plunge/ramp/helix)
    spindle: Optional[float] = None    # rpm
    tool_id: Optional[str] = None
    operation_id: Optional[str] = None
    feature_id: Optional[str] = None
    segment_id: Optional[str] = None
    source_face_ids: list[int] = field(default_factory=list)
    source_edge_ids: list[int] = field(default_factory=list)
    geometry_mode: Optional[ToolpathGeometryMode] = None
    target_surface: Optional[str] = None
    arc_center: Optional[np.ndarray] = None   # arc center (space coords) for arcs/helix
    arc_ccw: bool = True
    dwell_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def is_cutting(self) -> bool:
        """Return True if this motion is material-removing (cut, plunge, ramp, helix, arc, entry, exit, or canned cycle)."""
        return self.motion_type in (
            MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
            MotionType.HELIX, MotionType.ENTRY, MotionType.EXIT,
            MotionType.ARC_CW, MotionType.ARC_CCW,
        ) or bool(self.metadata.get("canned_cycle"))

    def length(self) -> float:
        if self.motion_type in (MotionType.ARC_CW, MotionType.ARC_CCW):
            return self._arc_length()
        return float(np.linalg.norm(self.end - self.start))

    def _arc_length(self) -> float:
        c = self.arc_center
        if c is None:
            return float(np.linalg.norm(self.end - self.start))
        r0 = np.linalg.norm(self.start[:2] - c[:2])
        r1 = np.linalg.norm(self.end[:2] - c[:2])
        r = 0.5 * (r0 + r1)
        a0 = np.arctan2(self.start[1] - c[1], self.start[0] - c[0])
        a1 = np.arctan2(self.end[1] - c[1], self.end[0] - c[0])
        if self.arc_ccw:
            sweep = (a1 - a0) % (2 * np.pi)
        else:
            sweep = (a0 - a1) % (2 * np.pi)
        # full circle when start==end
        if np.allclose(self.start, self.end, atol=1e-9):
            sweep = 2 * np.pi
        return float(r * sweep)

    def transformed(self, transform: np.ndarray, dst_space: CoordSpace = CoordSpace.WORK) -> MotionSegment:
        """Transform this segment by a 4x4 matrix into a new coordinate space."""
        from ..coords import transform_points
        t = np.asarray(transform, dtype=float)
        new_start = transform_points(self.start.reshape(1, 3), t).reshape(3)
        new_end = transform_points(self.end.reshape(1, 3), t).reshape(3)
        new_center = None
        new_ccw = self.arc_ccw
        if self.arc_center is not None:
            new_center = transform_points(self.arc_center.reshape(1, 3), t).reshape(3)
            if np.linalg.det(t[:3, :3]) < 0:
                new_ccw = not self.arc_ccw

        new_meta = dict(self.metadata)
        if "canned_cycle" in new_meta:
            cc = dict(new_meta["canned_cycle"])
            p_cyc = np.array([cc.get("x", self.end[0]), cc.get("y", self.end[1]), cc.get("z_depth", self.end[2])])
            p_cyc_tr = transform_points(p_cyc.reshape(1, 3), t).reshape(3)
            cc["x"] = float(p_cyc_tr[0])
            cc["y"] = float(p_cyc_tr[1])
            cc["z_depth"] = float(p_cyc_tr[2])
            if "z_r" in cc:
                p_r = np.array([cc["x"], cc["y"], cc["z_r"]])
                p_r_tr = transform_points(p_r.reshape(1, 3), t).reshape(3)
                cc["z_r"] = float(p_r_tr[2])
            new_meta["canned_cycle"] = cc

        return MotionSegment(
            motion_type=self.motion_type,
            start=new_start,
            end=new_end,
            space=dst_space,
            feed=self.feed,
            spindle=self.spindle,
            tool_id=self.tool_id,
            operation_id=self.operation_id,
            feature_id=self.feature_id,
            segment_id=self.segment_id,
            source_face_ids=list(self.source_face_ids),
            source_edge_ids=list(self.source_edge_ids),
            geometry_mode=self.geometry_mode,
            target_surface=self.target_surface,
            arc_center=new_center,
            arc_ccw=new_ccw,
            dwell_seconds=self.dwell_seconds,
            metadata=new_meta,
        )


@dataclass
class Toolpath:
    """A validated semantic toolpath for one operation with explicit geometry representation contract."""

    operation_id: str
    feature_id: str
    tool_id: str
    purpose: str
    segments: list[MotionSegment] = field(default_factory=list)
    space: CoordSpace = CoordSpace.SETUP
    geometry_mode: ToolpathGeometryMode = ToolpathGeometryMode.TOOL_CENTER
    source_face_ids: list[int] = field(default_factory=list)
    source_edge_ids: list[int] = field(default_factory=list)
    target_surface: Optional[str] = None
    design_geometry: Optional[Any] = None
    manufacturing_geometry: Optional[Any] = None
    metadata: dict = field(default_factory=dict)

    def add(self, seg: MotionSegment) -> None:
        if seg.segment_id is None:
            seg.segment_id = f"{self.operation_id}_seg_{len(self.segments) + 1:04d}"
        if seg.operation_id is None:
            seg.operation_id = self.operation_id
        if seg.feature_id is None:
            seg.feature_id = self.feature_id
        if not seg.source_face_ids and self.source_face_ids:
            seg.source_face_ids = list(self.source_face_ids)
        if not seg.source_edge_ids and self.source_edge_ids:
            seg.source_edge_ids = list(self.source_edge_ids)
        if seg.geometry_mode is None:
            seg.geometry_mode = self.geometry_mode
        if seg.target_surface is None and self.target_surface:
            seg.target_surface = self.target_surface
        self.segments.append(seg)

    @property
    def has_cutting_motion(self) -> bool:
        """Return True if the toolpath contains at least one cutting or material-removing segment."""
        return any(s.is_cutting for s in self.segments)

    @property
    def is_empty_or_skipped(self) -> bool:
        """Return True if the toolpath was skipped or contains zero segments."""
        return bool(self.metadata.get("skipped")) or len(self.segments) == 0

    def cutting_length(self) -> float:
        cutting = {MotionType.CUT, MotionType.PLUNGE, MotionType.RAMP,
                   MotionType.HELIX, MotionType.ENTRY, MotionType.EXIT,
                   MotionType.ARC_CW, MotionType.ARC_CCW}
        return sum(s.length() for s in self.segments if s.motion_type in cutting)

    def rapid_length(self) -> float:
        return sum(s.length() for s in self.segments
                   if s.motion_type in (MotionType.RAPID, MotionType.LINK, MotionType.RETRACT))

    def segment_count(self) -> int:
        return len(self.segments)

    def transformed(self, transform: np.ndarray, dst_space: CoordSpace = CoordSpace.WORK) -> Toolpath:
        """Transform all segments in this toolpath by a 4x4 matrix into a new coordinate space."""
        new_segments = [seg.transformed(transform, dst_space=dst_space) for seg in self.segments]
        new_design = self.design_geometry.transformed(transform, dst_space) if self.design_geometry else None
        new_mfg = self.manufacturing_geometry.transformed(transform, dst_space) if self.manufacturing_geometry else None
        return Toolpath(
            operation_id=self.operation_id,
            feature_id=self.feature_id,
            tool_id=self.tool_id,
            purpose=self.purpose,
            segments=new_segments,
            space=dst_space,
            geometry_mode=self.geometry_mode,
            source_face_ids=list(self.source_face_ids),
            source_edge_ids=list(self.source_edge_ids),
            target_surface=self.target_surface,
            design_geometry=new_design,
            manufacturing_geometry=new_mfg,
            metadata=dict(self.metadata),
        )



# =============================================================================
# Canonical Segment Factory Constructors
# =============================================================================

def Rapid(
    start: np.ndarray | list[float],
    end: np.ndarray | list[float],
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Rapid motion segment."""
    return MotionSegment(
        motion_type=MotionType.RAPID,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        space=space,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=dict(metadata) if metadata else {},
    )


def LinearCut(
    start: np.ndarray | list[float],
    end: np.ndarray | list[float],
    feed: Optional[float] = None,
    spindle: Optional[float] = None,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Linear cutting motion segment."""
    return MotionSegment(
        motion_type=MotionType.CUT,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        space=space,
        feed=feed,
        spindle=spindle,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=dict(metadata) if metadata else {},
    )


def CircularCut(
    start: np.ndarray | list[float],
    end: np.ndarray | list[float],
    arc_center: np.ndarray | list[float],
    arc_ccw: bool = True,
    feed: Optional[float] = None,
    spindle: Optional[float] = None,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Circular cutting arc segment."""
    m_type = MotionType.ARC_CCW if arc_ccw else MotionType.ARC_CW
    return MotionSegment(
        motion_type=m_type,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        space=space,
        feed=feed,
        spindle=spindle,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        arc_center=np.asarray(arc_center, dtype=float),
        arc_ccw=arc_ccw,
        metadata=dict(metadata) if metadata else {},
    )


def HelicalCut(
    start: np.ndarray | list[float],
    end: np.ndarray | list[float],
    arc_center: np.ndarray | list[float],
    arc_ccw: bool = True,
    feed: Optional[float] = None,
    spindle: Optional[float] = None,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Helical interpolation cutting segment."""
    return MotionSegment(
        motion_type=MotionType.HELIX,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        space=space,
        feed=feed,
        spindle=spindle,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        arc_center=np.asarray(arc_center, dtype=float),
        arc_ccw=arc_ccw,
        metadata=dict(metadata) if metadata else {},
    )


def Retract(
    start: np.ndarray | list[float],
    end: np.ndarray | list[float],
    feed: Optional[float] = None,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Retract motion segment."""
    return MotionSegment(
        motion_type=MotionType.RETRACT,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        space=space,
        feed=feed,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=dict(metadata) if metadata else {},
    )


def Plunge(
    start: np.ndarray | list[float],
    end: np.ndarray | list[float],
    feed: Optional[float] = None,
    spindle: Optional[float] = None,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Plunge cutting motion segment."""
    return MotionSegment(
        motion_type=MotionType.PLUNGE,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        space=space,
        feed=feed,
        spindle=spindle,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=dict(metadata) if metadata else {},
    )


def Ramp(
    start: np.ndarray | list[float],
    end: np.ndarray | list[float],
    feed: Optional[float] = None,
    spindle: Optional[float] = None,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Ramp cutting motion segment."""
    return MotionSegment(
        motion_type=MotionType.RAMP,
        start=np.asarray(start, dtype=float),
        end=np.asarray(end, dtype=float),
        space=space,
        feed=feed,
        spindle=spindle,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=dict(metadata) if metadata else {},
    )


def ToolChange(
    tool_id: str,
    tool_number: int = 1,
    length_offset_h: Optional[int] = None,
    diameter_offset_d: Optional[int] = None,
    position: Optional[np.ndarray | list[float]] = None,
    space: CoordSpace = CoordSpace.SETUP,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Tool Change command segment."""
    p = np.asarray(position if position is not None else [0.0, 0.0, 0.0], dtype=float)
    meta = dict(metadata) if metadata else {}
    meta["tool_number"] = tool_number
    if length_offset_h is not None:
        meta["length_offset_h"] = length_offset_h
    if diameter_offset_d is not None:
        meta["diameter_offset_d"] = diameter_offset_d
    return MotionSegment(
        motion_type=MotionType.TOOL_CHANGE,
        start=p.copy(),
        end=p.copy(),
        space=space,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=meta,
    )


def Spindle(
    rpm: float,
    direction: str = "cw",
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Spindle command segment."""
    m_type = MotionType.SPINDLE_START if rpm > 0 else MotionType.SPINDLE_STOP
    meta = dict(metadata) if metadata else {}
    meta["direction"] = direction
    return MotionSegment(
        motion_type=m_type,
        start=np.zeros(3),
        end=np.zeros(3),
        space=space,
        spindle=rpm,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=meta,
    )


def Coolant(
    mode: str = "flood",
    on: bool = True,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Coolant command segment."""
    m_type = MotionType.COOLANT_ON if on else MotionType.COOLANT_OFF
    meta = dict(metadata) if metadata else {}
    meta["coolant_type"] = mode
    return MotionSegment(
        motion_type=m_type,
        start=np.zeros(3),
        end=np.zeros(3),
        space=space,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=meta,
    )


def Dwell(
    seconds: float,
    space: CoordSpace = CoordSpace.SETUP,
    tool_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> MotionSegment:
    """Canonical Dwell command segment."""
    return MotionSegment(
        motion_type=MotionType.DWELL,
        start=np.zeros(3),
        end=np.zeros(3),
        space=space,
        dwell_seconds=seconds,
        tool_id=tool_id,
        operation_id=operation_id,
        feature_id=feature_id,
        metadata=dict(metadata) if metadata else {},
    )

