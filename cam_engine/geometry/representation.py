"""Design Geometry vs Manufacturing Representation.

Maintains strict separation between:
1. Design Geometry: Pure, nominal CAD surfaces, boundaries, and curves (invariant, from CAD/B-Rep).
2. Manufacturing Geometry: Toolpath boundaries (with allowances), Tool-center paths (offset by tool radius),
   Stock boundaries, and Swept cutting envelopes.

Never mixes nominal CAD dimensions with tool radii or random offset paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Union
import numpy as np

from ..coords import (
    CoordSpace,
    transform_points,
    transform_direction,
    transform_bounding_box,
)


# =============================================================================
# 1. DESIGN GEOMETRY (Nominal CAD Entity Representation)
# =============================================================================

class CadSurfaceKind(str, Enum):
    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    BSPLINE = "bspline"
    FREEFORM = "freeform"
    OTHER = "other"


@dataclass
class NominalCadSurface:
    """Exact nominal CAD surface representation extracted directly from B-Rep topology."""

    kind: CadSurfaceKind
    face_ids: list[int] = field(default_factory=list)
    diameter: Optional[float] = None          # Nominal CAD diameter (e.g. 6.00 mm for Ø6 cylinder)
    radius: Optional[float] = None            # Nominal CAD radius (e.g. 3.00 mm)
    axis: Optional[np.ndarray] = None         # Surface axis direction vector (3,)
    center: Optional[np.ndarray] = None       # Surface origin / 3D center point (3,)
    normal: Optional[np.ndarray] = None       # Surface normal vector (for planar surfaces) (3,)
    axial_start: Optional[float] = None       # Nominal CAD axial start along axis / Z
    axial_end: Optional[float] = None         # Nominal CAD axial end along axis / Z
    depth: Optional[float] = None             # Nominal CAD depth (axial_start - axial_end)
    fillet_radius: Optional[float] = None     # Nominal CAD fillet / blend radius
    semi_angle: Optional[float] = None        # Cone semi-angle in radians
    ref_radius: Optional[float] = None        # Cone reference radius
    is_internal: Optional[bool] = None        # True for hole/pocket/bore, False for boss/shaft
    source_reference: Optional[str] = None    # Provenance (e.g. "Face#12:CYLINDER:Ø6.000")
    tolerances: Optional[Any] = None          # Nominal tolerance specification

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> NominalCadSurface:
        """Rigidly transform nominal CAD surface into a target coordinate space."""
        t = np.asarray(transform_matrix, dtype=float)
        new_axis = transform_direction(self.axis, t) if self.axis is not None else None
        new_normal = transform_direction(self.normal, t) if self.normal is not None else None
        new_center = transform_points(self.center.reshape(1, 3), t)[0] if self.center is not None else None

        new_axial_start = self.axial_start
        new_axial_end = self.axial_end
        if self.center is not None and (self.axial_start is not None or self.axial_end is not None):
            if self.axial_start is not None:
                p_start = np.array([[self.center[0], self.center[1], self.axial_start]])
                new_axial_start = float(transform_points(p_start, t)[0, 2])
            if self.axial_end is not None:
                p_end = np.array([[self.center[0], self.center[1], self.axial_end]])
                new_axial_end = float(transform_points(p_end, t)[0, 2])

        return NominalCadSurface(
            kind=self.kind,
            face_ids=list(self.face_ids),
            diameter=self.diameter,
            radius=self.radius,
            axis=new_axis,
            center=new_center,
            normal=new_normal,
            axial_start=new_axial_start,
            axial_end=new_axial_end,
            depth=self.depth,
            fillet_radius=self.fillet_radius,
            semi_angle=self.semi_angle,
            ref_radius=self.ref_radius,
            is_internal=self.is_internal,
            source_reference=self.source_reference,
            tolerances=self.tolerances,
        )


@dataclass
class NominalCadBoundary:
    """Nominal CAD wire boundary loop (outer perimeter and inner islands)."""

    outer_loop: np.ndarray                    # (N, 3) or (N, 2) nominal boundary vertices
    island_loops: list[np.ndarray] = field(default_factory=list) # Inner island wire loops
    edge_ids: list[int] = field(default_factory=list)
    z_level: Optional[float] = None
    is_closed: bool = True

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> NominalCadBoundary:
        t = np.asarray(transform_matrix, dtype=float)
        new_outer = transform_points(self.outer_loop, t) if len(self.outer_loop) > 0 else self.outer_loop.copy()
        new_islands = [transform_points(loop, t) for loop in self.island_loops]
        new_z = None
        if self.z_level is not None:
            if len(new_outer) > 0 and new_outer.ndim == 2 and new_outer.shape[1] >= 3:
                new_z = float(new_outer[0, 2])
            else:
                pt_z = np.array([0.0, 0.0, float(self.z_level)])
                new_z = float((pt_z @ t[:3, :3].T + t[:3, 3])[2])

        return NominalCadBoundary(
            outer_loop=new_outer,
            island_loops=new_islands,
            edge_ids=list(self.edge_ids),
            z_level=new_z,
            is_closed=self.is_closed,
        )


@dataclass
class NominalCadCurve:
    """Nominal CAD edge / 3D curve representation."""

    curve_kind: str                           # "line", "circle", "ellipse", "bspline", etc.
    edge_id: int
    points: np.ndarray                        # (N, 3) discretized CAD points
    center: Optional[np.ndarray] = None       # Center for circular/elliptical curves
    axis: Optional[np.ndarray] = None
    radius: Optional[float] = None
    closed: bool = False

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> NominalCadCurve:
        t = np.asarray(transform_matrix, dtype=float)
        new_pts = transform_points(self.points, t) if len(self.points) > 0 else self.points.copy()
        new_center = transform_points(self.center.reshape(1, 3), t)[0] if self.center is not None else None
        new_axis = transform_direction(self.axis, t) if self.axis is not None else None
        return NominalCadCurve(
            curve_kind=self.curve_kind,
            edge_id=self.edge_id,
            points=new_pts,
            center=new_center,
            axis=new_axis,
            radius=self.radius,
            closed=self.closed,
        )


@dataclass
class DesignGeometry:
    """Pure nominal CAD design geometry representation.
    
    Contains only unadulterated CAD features: nominal surfaces, boundaries, and dimensions.
    Contains NO tooling geometry, NO machining allowances, and NO stepdown passes.
    """

    surfaces: list[NominalCadSurface] = field(default_factory=list)
    boundaries: list[NominalCadBoundary] = field(default_factory=list)
    curves: list[NominalCadCurve] = field(default_factory=list)
    primary_surface: Optional[NominalCadSurface] = None
    primary_boundary: Optional[NominalCadBoundary] = None
    nominal_dimensions: dict[str, float] = field(default_factory=dict)
    space: CoordSpace = CoordSpace.MODEL

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> DesignGeometry:
        """Transform nominal design geometry into another coordinate space."""
        t = np.asarray(transform_matrix, dtype=float)
        return DesignGeometry(
            surfaces=[s.transformed(t, dst_space) for s in self.surfaces],
            boundaries=[b.transformed(t, dst_space) for b in self.boundaries],
            curves=[c.transformed(t, dst_space) for c in self.curves],
            primary_surface=self.primary_surface.transformed(t, dst_space) if self.primary_surface else None,
            primary_boundary=self.primary_boundary.transformed(t, dst_space) if self.primary_boundary else None,
            nominal_dimensions=dict(self.nominal_dimensions),
            space=dst_space,
        )

    def summary(self) -> str:
        s_kind = self.primary_surface.kind.value if self.primary_surface else "unknown"
        dims = ", ".join(f"{k}={v:.3f}" for k, v in self.nominal_dimensions.items())
        return f"DesignGeometry(surface={s_kind}, dims=[{dims}], space={self.space.value})"


# =============================================================================
# 2. MANUFACTURING GEOMETRY (Toolpath, Allowances, Offset, & Envelope)
# =============================================================================

@dataclass
class ToolpathBoundary:
    """Part boundary offset by manufacturing allowances (roughing/finishing allowances).
    
    Represents the target boundary the cutter must finish to, before tool radius offsetting.
    """

    contour: np.ndarray                       # (N, 3) or (N, 2) boundary with machining allowance applied
    island_contours: list[np.ndarray] = field(default_factory=list)
    z_level: float = 0.0
    radial_allowance: float = 0.0             # Stock remaining on walls (mm)
    axial_allowance: float = 0.0              # Stock remaining on floor (mm)
    is_internal: bool = True                  # Internal pocket/hole vs external boss
    source_design_boundary: Optional[NominalCadBoundary] = None
    boundary_radius: Optional[float] = None   # For circular features: CAD radius ± radial_allowance

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> ToolpathBoundary:
        t = np.asarray(transform_matrix, dtype=float)
        new_contour = transform_points(self.contour, t) if len(self.contour) > 0 else self.contour.copy()
        new_islands = [transform_points(c, t) for c in self.island_contours]
        new_z = None
        if self.z_level is not None:
            if len(new_contour) > 0 and new_contour.ndim == 2 and new_contour.shape[1] >= 3:
                new_z = float(new_contour[0, 2])
            else:
                pt_z = np.array([0.0, 0.0, float(self.z_level)])
                new_z = float((pt_z @ t[:3, :3].T + t[:3, 3])[2])
        elif len(new_contour) > 0 and new_contour.ndim == 2 and new_contour.shape[1] >= 3:
            new_z = float(new_contour[0, 2])

        return ToolpathBoundary(
            contour=new_contour,
            island_contours=new_islands,
            z_level=new_z,
            radial_allowance=self.radial_allowance,
            axial_allowance=self.axial_allowance,
            is_internal=self.is_internal,
            source_design_boundary=self.source_design_boundary.transformed(t, dst_space) if self.source_design_boundary else None,
            boundary_radius=self.boundary_radius,
        )


@dataclass
class ToolCenterPath:
    """Tool-center path coordinates derived by offsetting ToolpathBoundary by cutter radius."""

    points: np.ndarray                        # (N, 3) Tool centerline coordinates
    tool_radius: float                        # Tool nominal radius applied in CAM
    tool_corner_radius: float = 0.0           # Bull-nose / corner radius
    compensation_type: str = "computer"       # "computer", "control", "wear", "off"
    compensation_side: str = "left"           # "left" (G41 climb), "right" (G42 conv), "off"
    center_radius: Optional[float] = None     # For circular path: boundary_radius ∓ tool_radius
    center_arc_center: Optional[np.ndarray] = None
    is_helical: bool = False
    helical_pitch: float = 0.0
    source_toolpath_boundary: Optional[ToolpathBoundary] = None

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> ToolCenterPath:
        t = np.asarray(transform_matrix, dtype=float)
        new_pts = transform_points(self.points, t) if len(self.points) > 0 else self.points.copy()
        new_center = transform_points(self.center_arc_center.reshape(1, 3), t)[0] if self.center_arc_center is not None else None
        return ToolCenterPath(
            points=new_pts,
            tool_radius=self.tool_radius,
            tool_corner_radius=self.tool_corner_radius,
            compensation_type=self.compensation_type,
            compensation_side=self.compensation_side,
            center_radius=self.center_radius,
            center_arc_center=new_center,
            is_helical=self.is_helical,
            helical_pitch=self.helical_pitch,
            source_toolpath_boundary=self.source_toolpath_boundary.transformed(t, dst_space) if self.source_toolpath_boundary else None,
        )


@dataclass
class StockBoundary:
    """In-process or initial stock workpiece boundary envelope."""

    bounds_min: np.ndarray                    # (3,)
    bounds_max: np.ndarray                    # (3,)
    stock_contour: Optional[np.ndarray] = None# (N, 3) 2D/3D polygon perimeter
    remaining_thickness: Optional[float] = None
    is_in_process: bool = False

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> StockBoundary:
        t = np.asarray(transform_matrix, dtype=float)
        bmin, bmax = transform_bounding_box(self.bounds_min, self.bounds_max, t)
        new_contour = transform_points(self.stock_contour, t) if self.stock_contour is not None else None
        return StockBoundary(
            bounds_min=bmin,
            bounds_max=bmax,
            stock_contour=new_contour,
            remaining_thickness=self.remaining_thickness,
            is_in_process=self.is_in_process,
        )


@dataclass
class CuttingEnvelope:
    """Swept geometric cutting envelope of the tool along the ToolCenterPath.
    
    Used for collision checking, gouge validation, and stock removal verification.
    """

    swept_radius: float                       # Outer cutting radius = tool_radius
    tool_length: float                        # Flute length / gauge length
    min_z: float                              # Bottom-most cutting depth
    max_z: float                              # Top entry level
    bounds_min: np.ndarray                    # 3D bounding box of cutting envelope
    bounds_max: np.ndarray
    swept_center_path: Optional[np.ndarray] = None # (N, 3) Tool centerline points

    def contains_point_xy_z(self, x: float, y: float, z: float, tolerance: float = 1e-4) -> bool:
        """Check if 3D point is inside the swept cutting envelope."""
        if z < self.min_z - tolerance or z > self.max_z + tolerance:
            return False
        if x < self.bounds_min[0] - tolerance or x > self.bounds_max[0] + tolerance:
            return False
        if y < self.bounds_min[1] - tolerance or y > self.bounds_max[1] + tolerance:
            return False
        if self.swept_center_path is None or len(self.swept_center_path) == 0:
            return True
        # Check minimum distance from (x, y) to centerline segments
        p_xy = np.array([x, y])
        pts_xy = self.swept_center_path[:, :2]
        dists = np.linalg.norm(pts_xy - p_xy, axis=1)
        return float(np.min(dists)) <= (self.swept_radius + tolerance)


@dataclass
class ManufacturingGeometry:
    """Authoritative manufacturing representation separating machining artifacts from nominal CAD geometry.
    
    Explicitly tracks:
    1. Toolpath boundary (nominal CAD boundary + manufacturing allowances)
    2. Tool-center path (boundary offset by cutter radius)
    3. Stock boundary (in-process workpiece envelope)
    4. Cutting envelope (swept cutting tool volume)
    5. Provenance back to DesignGeometry
    """

    feature_id: str
    operation_id: Optional[str] = None
    tool_id: Optional[str] = None
    design_geometry: Optional[DesignGeometry] = None
    toolpath_boundary: Optional[ToolpathBoundary] = None
    tool_center_path: Optional[ToolCenterPath] = None
    stock_boundary: Optional[StockBoundary] = None
    cutting_envelope: Optional[CuttingEnvelope] = None
    manufacturing_params: dict = field(default_factory=dict)
    space: CoordSpace = CoordSpace.SETUP

    def transformed(self, transform_matrix: np.ndarray, dst_space: CoordSpace = CoordSpace.SETUP) -> ManufacturingGeometry:
        t = np.asarray(transform_matrix, dtype=float)
        return ManufacturingGeometry(
            feature_id=self.feature_id,
            operation_id=self.operation_id,
            tool_id=self.tool_id,
            design_geometry=self.design_geometry.transformed(t, dst_space) if self.design_geometry else None,
            toolpath_boundary=self.toolpath_boundary.transformed(t, dst_space) if self.toolpath_boundary else None,
            tool_center_path=self.tool_center_path.transformed(t, dst_space) if self.tool_center_path else None,
            stock_boundary=self.stock_boundary.transformed(t, dst_space) if self.stock_boundary else None,
            cutting_envelope=self.cutting_envelope, # Envelope bounds can be recomputed
            manufacturing_params=dict(self.manufacturing_params),
            space=dst_space,
        )

    def validate_against_design(self, tolerance: float = 0.01) -> bool:
        """Verify that manufacturing tool-center offset does not violate nominal design geometry."""
        if not self.design_geometry or not self.design_geometry.primary_surface:
            return True
        cad_surf = self.design_geometry.primary_surface
        if cad_surf.kind == CadSurfaceKind.CYLINDER and self.tool_center_path and self.tool_center_path.center_radius is not None:
            cad_r = cad_surf.radius or (cad_surf.diameter / 2.0 if cad_surf.diameter else 0.0)
            t_r = self.tool_center_path.tool_radius
            c_r = self.tool_center_path.center_radius
            is_int = cad_surf.is_internal if cad_surf.is_internal is not None else True
            
            # For internal cylinder (hole/bore): cutter outer edge = c_r + t_r must NOT exceed CAD nominal radius
            if is_int:
                cutter_outer_r = c_r + t_r
                if cutter_outer_r > cad_r + tolerance:
                    return False
            else:
                # For external boss: cutter inner edge = c_r - t_r must NOT gouge CAD boss radius
                cutter_inner_r = c_r - t_r
                if cutter_inner_r < cad_r - tolerance:
                    return False
        return True


# =============================================================================
# 3. MANUFACTURING OFFSET CALCULATOR
# =============================================================================

class ManufacturingOffsetCalculator:
    """Calculates deterministic manufacturing representations from nominal design geometry.
    
    Implements the explicit manufacturing transformation pipeline:
        Nominal CAD Geometry (DesignGeometry)
                ↓
        Allowance Application (ToolpathBoundary)
                ↓
        Cutter Radius Offset (ToolCenterPath)
                ↓
        Swept Cutting Volume (CuttingEnvelope)
    """

    @staticmethod
    def calculate_cylindrical_offset(
        cad_surface: NominalCadSurface,
        tool_radius: float,
        radial_allowance: float = 0.0,
        is_internal: Optional[bool] = None,
    ) -> tuple[float, float, float]:
        """Calculate exact (nominal_radius, boundary_radius, tool_center_radius).
        
        Example for CAD Ø6 hole (cad_r=3.0, allowance=0.2, tool_r=1.5):
            nominal_radius     = 3.00 mm
            boundary_radius    = 3.00 - 0.20 = 2.80 mm (stock left on wall)
            tool_center_radius = 2.80 - 1.50 = 1.30 mm (tool centerline path)
            
        Example for CAD Ø8 boss (cad_r=4.0, allowance=0.5, tool_r=2.0):
            nominal_radius     = 4.00 mm
            boundary_radius    = 4.00 + 0.50 = 4.50 mm (stock left on boss)
            tool_center_radius = 4.50 + 2.00 = 6.50 mm (tool centerline path outside)
        """
        cad_r = cad_surface.radius
        if cad_r is None and cad_surface.diameter is not None:
            cad_r = cad_surface.diameter / 2.0
        if cad_r is None:
            raise ValueError(f"CAD surface {cad_surface.source_reference} has no diameter/radius defined.")

        internal = cad_surface.is_internal if is_internal is None else is_internal
        if internal is None:
            internal = True

        if internal:
            # Internal feature: boundary is inside CAD surface by radial_allowance
            boundary_r = max(0.0, cad_r - radial_allowance)
            # Tool center is inside boundary by tool_radius
            tool_center_r = max(0.0, boundary_r - tool_radius)
        else:
            # External feature: boundary is outside CAD surface by radial_allowance
            boundary_r = cad_r + radial_allowance
            # Tool center is outside boundary by tool_radius
            tool_center_r = boundary_r + tool_radius

        return float(cad_r), float(boundary_r), float(tool_center_r)

    @staticmethod
    def build_circular_tool_center_path(
        center_xy: np.ndarray,
        nominal_surface: NominalCadSurface,
        tool_radius: float,
        radial_allowance: float = 0.0,
        z_start: float = 0.0,
        z_end: float = 0.0,
        is_internal: bool = True,
        num_points: int = 36,
        compensation_type: str = "computer",
        compensation_side: str = "left",
    ) -> ToolCenterPath:
        """Construct exact ToolCenterPath for circular/helical operations from nominal CAD cylinder."""
        cad_r, bound_r, center_r = ManufacturingOffsetCalculator.calculate_cylindrical_offset(
            nominal_surface,
            tool_radius=tool_radius,
            radial_allowance=radial_allowance,
            is_internal=is_internal,
        )

        angles = np.linspace(0, 2 * np.pi, num_points + 1)
        z_levels = np.linspace(z_start, z_end, num_points + 1)
        
        pts = np.zeros((num_points + 1, 3))
        pts[:, 0] = center_xy[0] + center_r * np.cos(angles)
        pts[:, 1] = center_xy[1] + center_r * np.sin(angles)
        pts[:, 2] = z_levels

        bound_pts = np.zeros((num_points + 1, 3))
        bound_pts[:, 0] = center_xy[0] + bound_r * np.cos(angles)
        bound_pts[:, 1] = center_xy[1] + bound_r * np.sin(angles)
        bound_pts[:, 2] = z_levels

        boundary = ToolpathBoundary(
            contour=bound_pts,
            z_level=z_end,
            radial_allowance=radial_allowance,
            axial_allowance=0.0,
            is_internal=is_internal,
            boundary_radius=bound_r,
        )

        return ToolCenterPath(
            points=pts,
            tool_radius=tool_radius,
            compensation_type=compensation_type,
            compensation_side=compensation_side,
            center_radius=center_r,
            center_arc_center=np.array([center_xy[0], center_xy[1], z_end]),
            is_helical=(abs(z_start - z_end) > 1e-6),
            helical_pitch=abs(z_start - z_end),
            source_toolpath_boundary=boundary,
        )

    @staticmethod
    def calculate_cutting_envelope(
        tool_center_path: ToolCenterPath,
        tool_length: float = 50.0,
    ) -> CuttingEnvelope:
        """Compute the swept 3D cutting envelope along the tool center path."""
        pts = tool_center_path.points
        t_r = tool_center_path.tool_radius
        if len(pts) == 0:
            return CuttingEnvelope(
                swept_radius=t_r,
                tool_length=tool_length,
                min_z=0.0,
                max_z=0.0,
                bounds_min=np.zeros(3),
                bounds_max=np.zeros(3),
            )

        min_xy = np.min(pts[:, :2], axis=0) - t_r
        max_xy = np.max(pts[:, :2], axis=0) + t_r
        min_z = float(np.min(pts[:, 2]))
        max_z = float(np.max(pts[:, 2])) + tool_length

        bmin = np.array([min_xy[0], min_xy[1], min_z])
        bmax = np.array([max_xy[0], max_xy[1], max_z])

        return CuttingEnvelope(
            swept_radius=t_r,
            tool_length=tool_length,
            min_z=min_z,
            max_z=max_z,
            bounds_min=bmin,
            bounds_max=bmax,
            swept_center_path=pts.copy(),
        )

    @classmethod
    def build_manufacturing_geometry(
        cls,
        design_geometry: DesignGeometry,
        tool_radius: float,
        radial_allowance: float = 0.0,
        axial_allowance: float = 0.0,
        feature_id: str = "",
        operation_id: Optional[str] = None,
        tool_id: Optional[str] = None,
        tool_length: float = 50.0,
        space: CoordSpace = CoordSpace.SETUP,
    ) -> ManufacturingGeometry:
        """Assemble a complete ManufacturingGeometry instance derived from DesignGeometry."""
        cad_surf = design_geometry.primary_surface
        tcp = None
        tpb = None
        envelope = None

        if cad_surf and cad_surf.kind == CadSurfaceKind.CYLINDER and cad_surf.center is not None:
            z_start = cad_surf.axial_start if cad_surf.axial_start is not None else 0.0
            z_end = cad_surf.axial_end if cad_surf.axial_end is not None else 0.0
            is_int = cad_surf.is_internal if cad_surf.is_internal is not None else True
            tcp = cls.build_circular_tool_center_path(
                center_xy=cad_surf.center[:2],
                nominal_surface=cad_surf,
                tool_radius=tool_radius,
                radial_allowance=radial_allowance,
                z_start=z_start,
                z_end=z_end,
                is_internal=is_int,
            )
            tpb = tcp.source_toolpath_boundary
            envelope = cls.calculate_cutting_envelope(tcp, tool_length=tool_length)

        elif design_geometry.primary_boundary and len(design_geometry.primary_boundary.outer_loop) > 0:
            # 2D contour polygon boundary
            contour = design_geometry.primary_boundary.outer_loop.copy()
            z_lvl = float(contour[0, 2]) if contour.shape[1] == 3 else 0.0
            tpb = ToolpathBoundary(
                contour=contour,
                z_level=z_lvl,
                radial_allowance=radial_allowance,
                axial_allowance=axial_allowance,
                is_internal=True,
                source_design_boundary=design_geometry.primary_boundary,
            )

        return ManufacturingGeometry(
            feature_id=feature_id,
            operation_id=operation_id,
            tool_id=tool_id,
            design_geometry=design_geometry,
            toolpath_boundary=tpb,
            tool_center_path=tcp,
            cutting_envelope=envelope,
            manufacturing_params={
                "radial_allowance": radial_allowance,
                "axial_allowance": axial_allowance,
                "tool_radius": tool_radius,
                "tool_length": tool_length,
            },
            space=space,
        )
