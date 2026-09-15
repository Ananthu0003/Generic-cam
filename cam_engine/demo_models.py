"""Procedural CAD models and sample geometries for testing and instant UI demonstration."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .context import Tool, ToolType, MachineConfig, MachineAxis, MachineType, Units, WorkOffset, Setup
from .geometry.topology import Face, SurfaceData, SurfaceKind, CurveData, CurveKind, Edge, Shape
from .geometry.mesh import TriangleMesh
from .features import MachiningFeature, FeatureType, Accessibility


@dataclass
class DemoCADModel:
    name: str
    description: str
    bounds_min: list[float]
    bounds_max: list[float]
    vertices: list[float]      # Flat [x,y,z, x,y,z, ...]
    normals: list[float]       # Flat [nx,ny,nz, ...]
    indices: list[int]         # Triangle index triplets [i0, i1, i2, ...]
    faces_metadata: list[dict] # per-face metadata (type, center, normal)


def create_prismatic_bracket_shape():
    """Creates a true B-Rep solid CAD model of the sample CNC milling test part:
    - Base: 100 x 60 x 25 mm
    - Top Facing face at Z = 25 mm
    - Rectangular Pocket: 40 x 30 mm, depth 10 mm (floor at Z = 15)
    - Through Hole: 12 mm diameter at (70, 30), depth 25 mm (Z = 0)
    - Step feature on the right side: 20 mm wide, 8 mm stepdown (Z = 17)
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from .geometry.topology import Shape
    from .geometry.step_import import ImportedModel

    base = BRepPrimAPI_MakeBox(100.0, 60.0, 25.0).Shape()
    step_cut = BRepPrimAPI_MakeBox(gp_Pnt(80.0, -1.0, 17.0), gp_Pnt(101.0, 61.0, 26.0)).Shape()
    s1 = BRepAlgoAPI_Cut(base, step_cut).Shape()
    pocket_cut = BRepPrimAPI_MakeBox(gp_Pnt(15.0, 15.0, 15.0), gp_Pnt(55.0, 45.0, 26.0)).Shape()
    s2 = BRepAlgoAPI_Cut(s1, pocket_cut).Shape()
    hole_cut = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(70.0, 30.0, -1.0), gp_Dir(0.0, 0.0, 1.0)), 6.0, 27.0).Shape()
    s3 = BRepAlgoAPI_Cut(s2, hole_cut).Shape()
    shape = Shape(s3)
    return ImportedModel(
        shape=shape,
        declared_units="mm",
        original_bounds=shape.bounding_box(),
    )


def create_prismatic_bracket_mesh() -> DemoCADModel:
    """Creates a sample CNC milling test part:
    - Base: 100 x 60 x 25 mm
    - Top Facing face at Z = 25 mm
    - Rectangular Pocket: 40 x 30 mm, depth 10 mm (floor at Z = 15)
    - Through Hole: 12 mm diameter at (75, 30), depth 25 mm (Z = 0)
    - Step feature on the right side: 20 mm wide, 8 mm stepdown (Z = 17)
    """
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    indices: list[int] = []
    faces_meta: list[dict] = []

    def add_quad(p0, p1, p2, p3, n, face_idx: int, kind: str = "wall"):
        base_idx = len(vertices)
        for p in (p0, p1, p2, p3):
            vertices.append(list(p))
            normals.append(list(n))
        # 2 triangles (0, 1, 2) and (0, 2, 3)
        indices.extend([base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3])

    def add_tri(p0, p1, p2, n):
        base_idx = len(vertices)
        for p in (p0, p1, p2):
            vertices.append(list(p))
            normals.append(list(n))
        indices.extend([base_idx, base_idx + 1, base_idx + 2])

    face_id = 0

    # 1. Bottom Face (Z = 0)
    add_quad([0, 0, 0], [100, 0, 0], [100, 60, 0], [0, 60, 0], [0, 0, -1], face_id, "bottom")
    faces_meta.append({"id": face_id, "kind": "bottom", "normal": [0, 0, -1], "z": 0})
    face_id += 1

    # 2. Outer side walls
    # Front wall (Y = 0)
    add_quad([0, 0, 0], [0, 0, 25], [100, 0, 25], [100, 0, 0], [0, -1, 0], face_id, "outer_wall")
    faces_meta.append({"id": face_id, "kind": "wall", "normal": [0, -1, 0]})
    face_id += 1

    # Back wall (Y = 60)
    add_quad([100, 60, 0], [100, 60, 25], [0, 60, 25], [0, 60, 0], [0, 1, 0], face_id, "outer_wall")
    faces_meta.append({"id": face_id, "kind": "wall", "normal": [0, 1, 0]})
    face_id += 1

    # Left wall (X = 0)
    add_quad([0, 60, 0], [0, 60, 25], [0, 0, 25], [0, 0, 0], [-1, 0, 0], face_id, "outer_wall")
    faces_meta.append({"id": face_id, "kind": "wall", "normal": [-1, 0, 0]})
    face_id += 1

    # Right wall (X = 100)
    add_quad([100, 0, 0], [100, 0, 17], [100, 60, 17], [100, 60, 0], [1, 0, 0], face_id, "outer_wall")
    faces_meta.append({"id": face_id, "kind": "wall", "normal": [1, 0, 0]})
    face_id += 1

    # 3. Top Face surrounding the pocket (Z = 25)
    # Divided into segments around pocket [15..55, 15..45]
    # Left of pocket [0..15, 0..60]
    add_quad([0, 0, 25], [15, 0, 25], [15, 60, 25], [0, 60, 25], [0, 0, 1], face_id, "top")
    # Bottom of pocket [15..55, 0..15]
    add_quad([15, 0, 25], [55, 0, 25], [55, 15, 25], [15, 15, 25], [0, 0, 1], face_id, "top")
    # Top of pocket [15..55, 45..60]
    add_quad([15, 45, 25], [55, 45, 25], [55, 60, 25], [15, 60, 25], [0, 0, 1], face_id, "top")
    # Between pocket and step [55..80, 0..60]
    add_quad([55, 0, 25], [80, 0, 25], [80, 60, 25], [55, 60, 25], [0, 0, 1], face_id, "top")
    faces_meta.append({"id": face_id, "kind": "top", "normal": [0, 0, 1], "z": 25})
    face_id += 1

    # 4. Step face on right side [80..100, 0..60] at Z = 17
    add_quad([80, 0, 17], [100, 0, 17], [100, 60, 17], [80, 60, 17], [0, 0, 1], face_id, "step_floor")
    faces_meta.append({"id": face_id, "kind": "step_floor", "normal": [0, 0, 1], "z": 17})
    face_id += 1

    # Step vertical wall at X = 80 [Z 17..25]
    add_quad([80, 0, 17], [80, 60, 17], [80, 60, 25], [80, 0, 25], [1, 0, 0], face_id, "step_wall")
    faces_meta.append({"id": face_id, "kind": "wall", "normal": [1, 0, 0]})
    face_id += 1

    # 5. Pocket (40x30 mm, depth 10mm -> floor Z = 15)
    # Floor: [15..55, 15..45] at Z = 15
    add_quad([15, 15, 15], [55, 15, 15], [55, 45, 15], [15, 45, 15], [0, 0, 1], face_id, "pocket_floor")
    faces_meta.append({"id": face_id, "kind": "pocket_floor", "normal": [0, 0, 1], "z": 15})
    face_id += 1

    # Pocket walls (Z 15..25)
    # Front inner wall (Y = 15)
    add_quad([15, 15, 15], [15, 15, 25], [55, 15, 25], [55, 15, 15], [0, 1, 0], face_id, "pocket_wall")
    # Back inner wall (Y = 45)
    add_quad([55, 45, 15], [55, 45, 25], [15, 45, 25], [15, 45, 15], [0, -1, 0], face_id, "pocket_wall")
    # Left inner wall (X = 15)
    add_quad([15, 45, 15], [15, 45, 25], [15, 15, 25], [15, 15, 15], [1, 0, 0], face_id, "pocket_wall")
    # Right inner wall (X = 55)
    add_quad([55, 15, 15], [55, 15, 25], [55, 45, 25], [55, 45, 15], [-1, 0, 0], face_id, "pocket_wall")
    faces_meta.append({"id": face_id, "kind": "wall", "normal": [0, 0, 0]})
    face_id += 1

    # 6. Through Hole cylinder at center (70, 30), radius 6 mm (D = 12 mm), Z: 0..25
    cx, cy, r = 70.0, 30.0, 6.0
    segments = 24
    for i in range(segments):
        a0 = 2 * np.pi * i / segments
        a1 = 2 * np.pi * (i + 1) / segments
        x0, y0 = cx + r * np.cos(a0), cy + r * np.sin(a0)
        x1, y1 = cx + r * np.cos(a1), cy + r * np.sin(a1)
        # Inward-pointing normal for hole cylinder
        nx = -np.cos(0.5 * (a0 + a1))
        ny = -np.sin(0.5 * (a0 + a1))
        add_quad([x0, y0, 0], [x0, y0, 25], [x1, y1, 25], [x1, y1, 0], [nx, ny, 0], face_id, "hole_wall")
    faces_meta.append({"id": face_id, "kind": "hole_cylinder", "radius": r, "center": [cx, cy]})

    flat_verts = [coord for pt in vertices for coord in pt]
    flat_normals = [n for norm in normals for n in norm]

    return DemoCADModel(
        name="Prismatic CAM Test Bracket",
        description="Standard 100x60x25mm aerospace bracket with top facing, 40x30mm pocket, Ø12mm bore, and 8mm stepdown.",
        bounds_min=[0.0, 0.0, 0.0],
        bounds_max=[100.0, 60.0, 25.0],
        vertices=flat_verts,
        normals=flat_normals,
        indices=indices,
        faces_metadata=faces_meta,
    )


def get_default_tools() -> list[Tool]:
    """Returns standard CNC tool catalog with comprehensive workshop tooling."""
    return [
        Tool(
            id="T01",
            tool_number=1,
            type=ToolType.FLAT_ENDMILL,
            diameter=12.7,
            corner_radius=0.0,
            flute_length=35.0,
            overall_length=75.0,
            flutes=3,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T02",
            tool_number=2,
            type=ToolType.FLAT_ENDMILL,
            diameter=6.35,
            corner_radius=0.0,
            flute_length=25.0,
            overall_length=60.0,
            flutes=4,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T03",
            tool_number=3,
            type=ToolType.FLAT_ENDMILL,
            diameter=3.0,
            corner_radius=0.0,
            flute_length=15.0,
            overall_length=50.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T04",
            tool_number=4,
            type=ToolType.FLAT_ENDMILL,
            diameter=2.0,
            corner_radius=0.0,
            flute_length=10.0,
            overall_length=45.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T05",
            tool_number=5,
            type=ToolType.DRILL,
            diameter=12.0,
            corner_radius=0.0,
            flute_length=50.0,
            overall_length=85.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T06",
            tool_number=6,
            type=ToolType.DRILL,
            diameter=8.0,
            corner_radius=0.0,
            flute_length=40.0,
            overall_length=75.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T07",
            tool_number=7,
            type=ToolType.DRILL,
            diameter=6.0,
            corner_radius=0.0,
            flute_length=35.0,
            overall_length=70.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T08",
            tool_number=8,
            type=ToolType.DRILL,
            diameter=5.0,
            corner_radius=0.0,
            flute_length=30.0,
            overall_length=65.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T09",
            tool_number=9,
            type=ToolType.DRILL,
            diameter=3.0,
            corner_radius=0.0,
            flute_length=20.0,
            overall_length=55.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T10",
            tool_number=10,
            type=ToolType.DRILL,
            diameter=2.5,
            corner_radius=0.0,
            flute_length=18.0,
            overall_length=50.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        Tool(
            id="T11",
            tool_number=11,
            type=ToolType.FACE_MILL,
            diameter=50.0,
            corner_radius=0.8,
            flute_length=10.0,
            overall_length=90.0,
            flutes=4,
            material="carbide",
            can_plunge=False,
        ),
        Tool(
            id="T12",
            tool_number=12,
            type=ToolType.CHAMFER_MILL,
            diameter=6.0,
            corner_radius=0.0,
            flute_length=12.0,
            overall_length=60.0,
            flutes=2,
            material="carbide",
            can_plunge=False,
        ),
        Tool(
            id="T13",
            tool_number=13,
            type=ToolType.TAP,
            diameter=6.0,
            corner_radius=0.0,
            flute_length=25.0,
            overall_length=65.0,
            flutes=2,
            material="hss",
            can_plunge=True,
        ),
        Tool(
            id="T14",
            tool_number=14,
            type=ToolType.REAMER,
            diameter=8.0,
            corner_radius=0.0,
            flute_length=30.0,
            overall_length=70.0,
            flutes=6,
            material="carbide",
            can_plunge=False,
        ),
        Tool(
            id="T15",
            tool_number=15,
            type=ToolType.THREAD_MILL,
            diameter=4.0,
            corner_radius=0.0,
            flute_length=15.0,
            overall_length=55.0,
            flutes=4,
            material="carbide",
            can_plunge=False,
        ),
        Tool(
            id="T16",
            tool_number=16,
            type=ToolType.GROOVE_CUTTER,
            diameter=3.0,
            corner_radius=0.0,
            flute_length=10.0,
            overall_length=50.0,
            flutes=2,
            material="carbide",
            can_plunge=True,
        ),
        # T17: 90° countersink — covers M3-M10 clearance holes
        Tool(
            id="T17",
            tool_number=17,
            type=ToolType.COUNTERSINK_TOOL,
            diameter=12.0,       # major diameter of conical tool
            corner_radius=0.0,
            flute_length=8.0,
            overall_length=55.0,
            flutes=6,
            material="carbide",
            can_plunge=True,
            tip_radius=0.0,
        ),
        # T18: boring bar — 16mm minimum bore diameter
        Tool(
            id="T18",
            tool_number=18,
            type=ToolType.BORING_BAR,
            diameter=16.0,       # adjustable; this is the nominal bore diameter
            corner_radius=0.0,
            flute_length=60.0,
            overall_length=120.0,
            flutes=1,            # single-point
            material="carbide",
            can_plunge=True,
        ),
    ]


def get_default_machine() -> MachineConfig:
    """Returns a default 3-axis CNC vertical machining center."""
    return MachineConfig(
        id="vmc-3axis",
        name="Precision 3-Axis VMC (Standard ISO)",
        machine_type=MachineType.THREE_AXIS_VERTICAL,
        axes={
            "X": MachineAxis(name="X", travel_min=-500.0, travel_max=500.0, rapid_rate=15000.0, max_feed=5000.0),
            "Y": MachineAxis(name="Y", travel_min=-300.0, travel_max=300.0, rapid_rate=15000.0, max_feed=5000.0),
            "Z": MachineAxis(name="Z", travel_min=-400.0, travel_max=50.0, rapid_rate=10000.0, max_feed=3000.0),
        },
        spindle_min_rpm=100.0,
        spindle_max_rpm=15000.0,
        max_spindle_power_kw=7.5,
        controller="grbl",
        units=Units.MM,
        tool_change_position=np.array([0.0, 0.0, 40.0]),
        work_offsets={"G54": np.array([0.0, 0.0, 0.0])},
        safe_retract_height=10.0,
    )
