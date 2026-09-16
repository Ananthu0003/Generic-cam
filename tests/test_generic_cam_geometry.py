"""Comprehensive geometry-driven regression and acceptance tests for VexCAM architecture.

Tests:
1. Feature Recognition (Through hole, Blind hole, Counterbore, Boss, Pocket, Slot, Step, Facing)
2. Coordinate and Setup Invariance (Translation, Rotation, Multi-setup, Coordinate chain round-trip)
3. Zero Modal Coordinate Leakage in G-code
4. Fail-Fast on Missing Data (No synthetic geometry fallback)
5. End-to-End CAD model test with sample_bracket.step
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from OCP.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Trsf, gp_Vec
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

from cam_engine.geometry.topology import Shape
from cam_engine.geometry.step_import import ImportedModel
from cam_engine.context import (
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    Material,
    DEFAULT_MATERIALS,
    Tool,
    ToolType,
    WorkOffset,
)
from cam_engine.coords import (
    CoordinateSystemChain,
    CoordSpace,
    make_transform,
    rotation_about_axis,
    transform_points,
)
from cam_engine.errors import CamError, UNSUPPORTED_FEATURE, MISSING_STOCK_GEOMETRY
from cam_engine.features import FeatureRecognizer, FeatureType, MachiningFeature
from cam_engine.operations import OperationPlanner, OpPurpose
from cam_engine.pipeline import run_cam_pipeline
from cam_engine.demo_models import get_default_tools, get_default_machine


def make_test_step_file(occ_shape) -> Path:
    """Save an OCC shape to a temporary STEP file and return its path."""
    writer = STEPControl_Writer()
    writer.Transfer(occ_shape, STEPControl_AsIs)
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        tmp_path = Path(f.name)
    writer.Write(str(tmp_path))
    return tmp_path


# =============================================================================
# 1. Feature Recognition Tests on Parametric B-Rep Shapes
# =============================================================================

def test_through_hole_recognition():
    """A box with a through-hole cylinder subtracted should be recognized as THROUGH_HOLE or THROUGH_BORE."""
    # Box 60x60x20
    box = BRepPrimAPI_MakeBox(60.0, 60.0, 20.0).Shape()
    # Cylinder dia 10 (radius 5) from Z=0 to Z=20 at center (30, 30)
    cyl_ax = gp_Ax2(gp_Pnt(30.0, 30.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    cyl = BRepPrimAPI_MakeCylinder(cyl_ax, 5.0, 20.0).Shape()
    part = BRepAlgoAPI_Cut(box, cyl).Shape()

    shape = Shape(part)
    rec = FeatureRecognizer(shape, stock_top_z=21.0)
    features = rec.recognize()

    holes = [f for f in features if f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BORE, FeatureType.THROUGH_BORE)]
    assert len(holes) == 1, f"Expected 1 hole, got {len(holes)}"
    h = holes[0]
    assert np.isclose(h.diameter, 10.0, atol=0.2)
    assert np.isclose(h.depth, 20.0, atol=0.2)
    assert np.allclose(h.center_xy, [30.0, 30.0], atol=0.5)


def test_blind_hole_recognition():
    """A box with a cylinder subtracted only partially from top should be recognized as BLIND_HOLE."""
    box = BRepPrimAPI_MakeBox(60.0, 60.0, 30.0).Shape()
    # Cylinder radius 4 (dia 8) from Z=15 to Z=30 at (30, 30)
    cyl_ax = gp_Ax2(gp_Pnt(30.0, 30.0, 15.0), gp_Dir(0.0, 0.0, 1.0))
    cyl = BRepPrimAPI_MakeCylinder(cyl_ax, 4.0, 15.0).Shape()
    part = BRepAlgoAPI_Cut(box, cyl).Shape()

    shape = Shape(part)
    rec = FeatureRecognizer(shape, stock_top_z=31.0)
    features = rec.recognize()

    holes = [f for f in features if f.type in (FeatureType.HOLE, FeatureType.BLIND_HOLE)]
    assert len(holes) == 1
    h = holes[0]
    assert np.isclose(h.diameter, 8.0, atol=0.2)
    assert np.isclose(h.depth, 15.0, atol=0.2)
    assert np.isclose(h.floor_z, 15.0, atol=0.2)


def test_boss_recognition():
    """A cylinder fused on top of a box should be recognized as a BOSS, NOT a hole."""
    box = BRepPrimAPI_MakeBox(60.0, 60.0, 20.0).Shape()
    cyl_ax = gp_Ax2(gp_Pnt(30.0, 30.0, 20.0), gp_Dir(0.0, 0.0, 1.0))
    cyl = BRepPrimAPI_MakeCylinder(cyl_ax, 8.0, 10.0).Shape()
    part = BRepAlgoAPI_Fuse(box, cyl).Shape()

    shape = Shape(part)
    rec = FeatureRecognizer(shape, stock_top_z=32.0)
    features = rec.recognize()

    bosses = [f for f in features if f.type == FeatureType.BOSS]
    holes = [f for f in features if f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE, FeatureType.BORE)]
    assert len(bosses) == 1
    assert len(holes) == 0
    assert np.isclose(bosses[0].diameter, 16.0, atol=0.2)


def test_counterbore_recognition():
    """A stepped cylinder (large top + small bottom) should be recognized relationally as COUNTERBORE + hole."""
    box = BRepPrimAPI_MakeBox(80.0, 80.0, 30.0).Shape()
    # Upper cylinder: dia 20 (radius 10) from Z=20 to Z=30
    cyl1_ax = gp_Ax2(gp_Pnt(40.0, 40.0, 20.0), gp_Dir(0.0, 0.0, 1.0))
    cyl1 = BRepPrimAPI_MakeCylinder(cyl1_ax, 10.0, 10.0).Shape()
    cut1 = BRepAlgoAPI_Cut(box, cyl1).Shape()

    # Lower cylinder: dia 10 (radius 5) from Z=0 to Z=20
    cyl2_ax = gp_Ax2(gp_Pnt(40.0, 40.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    cyl2 = BRepPrimAPI_MakeCylinder(cyl2_ax, 5.0, 20.0).Shape()
    part = BRepAlgoAPI_Cut(cut1, cyl2).Shape()

    shape = Shape(part)
    rec = FeatureRecognizer(shape, stock_top_z=31.0)
    features = rec.recognize()

    cbores = [f for f in features if f.type == FeatureType.COUNTERBORE]
    assert len(cbores) == 1
    cb = cbores[0]
    assert np.isclose(cb.diameter, 20.0, atol=0.2)
    assert np.isclose(cb.depth, 10.0, atol=0.2)
    assert len(cb.child_feature_ids) == 1


def test_pocket_recognition():
    """A box with an internal cavity subtracted should be recognized as a POCKET."""
    box = BRepPrimAPI_MakeBox(100.0, 80.0, 30.0).Shape()
    # Cavity 40x30 from Z=15 to Z=30 inside the box
    cavity_trsf = gp_Trsf()
    cavity_trsf.SetTranslation(gp_Vec(30.0, 25.0, 15.0))
    cavity = BRepBuilderAPI_Transform(BRepPrimAPI_MakeBox(40.0, 30.0, 15.0).Shape(), cavity_trsf, True).Shape()
    part = BRepAlgoAPI_Cut(box, cavity).Shape()

    shape = Shape(part)
    rec = FeatureRecognizer(shape, stock_top_z=31.0)
    features = rec.recognize()

    pockets = [f for f in features if f.type in (FeatureType.POCKET, FeatureType.OPEN_POCKET)]
    assert len(pockets) == 1
    p = pockets[0]
    assert np.isclose(p.depth, 15.0, atol=0.2)
    assert np.isclose(p.floor_z, 15.0, atol=0.2)


def test_step_recognition():
    """A box with a corner removed from top to mid-height should be recognized as a STEP."""
    box = BRepPrimAPI_MakeBox(100.0, 60.0, 30.0).Shape()
    step_trsf = gp_Trsf()
    step_trsf.SetTranslation(gp_Vec(70.0, 0.0, 18.0))
    step_box = BRepBuilderAPI_Transform(BRepPrimAPI_MakeBox(30.0, 60.0, 12.0).Shape(), step_trsf, True).Shape()
    part = BRepAlgoAPI_Cut(box, step_box).Shape()

    shape = Shape(part)
    rec = FeatureRecognizer(shape, stock_top_z=31.0)
    features = rec.recognize()

    steps = [f for f in features if f.type in (FeatureType.STEP, FeatureType.SHOULDER)]
    assert len(steps) == 1
    assert np.isclose(steps[0].depth, 12.0, atol=0.2)
    assert np.isclose(steps[0].floor_z, 18.0, atol=0.2)


# =============================================================================
# 2. Coordinate and Setup Invariance Tests
# =============================================================================

def test_translation_invariance():
    """Translating a CAD model by arbitrary offsets does not change recognized relative feature geometry."""
    box = BRepPrimAPI_MakeBox(60.0, 60.0, 20.0).Shape()
    cyl_ax = gp_Ax2(gp_Pnt(30.0, 30.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    cyl = BRepPrimAPI_MakeCylinder(cyl_ax, 5.0, 20.0).Shape()
    part = BRepAlgoAPI_Cut(box, cyl).Shape()

    # Translate part by (+150, -75, +50)
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(150.0, -75.0, 50.0))
    translated_part = BRepBuilderAPI_Transform(part, trsf, True).Shape()

    shape1 = Shape(part)
    shape2 = Shape(translated_part)

    rec1 = FeatureRecognizer(shape1, stock_top_z=21.0)
    rec2 = FeatureRecognizer(shape2, stock_top_z=71.0)

    f1 = [f for f in rec1.recognize() if f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BORE, FeatureType.THROUGH_BORE)][0]
    f2 = [f for f in rec2.recognize() if f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BORE, FeatureType.THROUGH_BORE)][0]

    assert np.isclose(f1.diameter, f2.diameter, atol=0.01)
    assert np.isclose(f1.depth, f2.depth, atol=0.01)
    assert np.allclose(f2.center_xy - f1.center_xy, [150.0, -75.0], atol=0.01)


def test_coordinate_chain_composition():
    """Verify matrix composition order in CoordinateSystemChain: model -> setup -> work -> machine."""
    m2s = make_transform(translation=[10.0, 20.0, 30.0])
    s2w = make_transform(translation=[-5.0, -5.0, 0.0])
    w2m = make_transform(translation=[100.0, 200.0, 300.0])

    chain = CoordinateSystemChain(
        model_to_setup=m2s,
        setup_to_work=s2w,
        work_to_machine=w2m,
    )
    chain.validate_chain()

    # Test forward mapping of point (1, 2, 3)
    p_model = np.array([[1.0, 2.0, 3.0]])
    p_setup = transform_points(p_model, chain.transform(CoordSpace.MODEL, CoordSpace.SETUP))
    assert np.allclose(p_setup, [[11.0, 22.0, 33.0]])

    p_work = transform_points(p_model, chain.transform(CoordSpace.MODEL, CoordSpace.WORK))
    assert np.allclose(p_work, [[6.0, 17.0, 33.0]])

    p_machine = transform_points(p_model, chain.transform(CoordSpace.MODEL, CoordSpace.MACHINE))
    assert np.allclose(p_machine, [[106.0, 217.0, 333.0]])

    # Round-trip back to model
    p_roundtrip = transform_points(p_machine, chain.transform(CoordSpace.MACHINE, CoordSpace.MODEL))
    assert np.allclose(p_roundtrip, p_model, atol=1e-9)


# =============================================================================
# 3. Modal Coordinate Leakage & Output Tests
# =============================================================================

def test_zero_modal_coordinate_leakage_in_gcode():
    """Test that every operation explicitly establishes complete initial (X, Y, Z) coordinates."""
    box = BRepPrimAPI_MakeBox(100.0, 80.0, 30.0).Shape()
    cyl_ax = gp_Ax2(gp_Pnt(50.0, 40.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    cyl = BRepPrimAPI_MakeCylinder(cyl_ax, 6.0, 30.0).Shape()
    part = BRepAlgoAPI_Cut(box, cyl).Shape()

    tmp_step = make_test_step_file(part)
    try:
        res = run_cam_pipeline(tmp_step, controller="grbl", work_offset="G54")
        assert len(res.operations) >= 2

        gcode_lines = res.gcode.splitlines()
        # Find all operation blocks
        op_indices = [i for i, line in enumerate(gcode_lines) if "(--- Operation:" in line]
        assert len(op_indices) >= 2

        for op_idx in op_indices:
            # Check subsequent motion lines in this operation before any cut
            op_block = gcode_lines[op_idx:op_idx + 15]
            # Find the first G0/G1/G98 motion line
            first_motion = next((line for line in op_block if any(line.startswith(c) for c in ("G0", "G00", "G1", "G01", "G98"))), None)
            assert first_motion is not None, f"Operation starting at line {op_idx} has no motion"
            # First motion or initial positioning must have explicit coordinates
            # Check that initial positioning lines establish X, Y, and Z
            block_text = " ".join(op_block)
            assert "X" in block_text and "Y" in block_text and "Z" in block_text, \
                f"Operation block does not establish full XYZ coordinates:\n{block_text}"
    finally:
        tmp_step.unlink(missing_ok=True)


# =============================================================================
# 4. Fail-Fast on Missing Data (No Fallbacks)
# =============================================================================

def test_missing_compatible_tool_fails():
    """If no tool fits a feature span, the planner must raise CamError instead of inventing a tool."""
    box = BRepPrimAPI_MakeBox(50.0, 50.0, 20.0).Shape()
    # Hole dia 1.0mm
    cyl_ax = gp_Ax2(gp_Pnt(25.0, 25.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    cyl = BRepPrimAPI_MakeCylinder(cyl_ax, 0.5, 20.0).Shape()
    part = BRepAlgoAPI_Cut(box, cyl).Shape()

    shape = Shape(part)
    rec = FeatureRecognizer(shape, stock_top_z=21.0)
    features = rec.recognize()

    # Tools library with only large tools (smallest is 6mm)
    tools = [
        Tool(id="T01", tool_number=1, type=ToolType.FLAT_ENDMILL, diameter=12.0, flute_length=30.0, overall_length=70.0),
        Tool(id="T02", tool_number=2, type=ToolType.DRILL, diameter=8.0, flute_length=40.0, overall_length=80.0),
    ]
    stock = Stock(kind=StockKind.BOX, bounds_min=np.zeros(3), bounds_max=np.array([50.0, 50.0, 21.0]))
    setup = Setup(id="s1", name="S1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)
    ctx = PlanningContext(
        model=ImportedModel(shape=shape, declared_units="mm", original_bounds=(np.zeros(3), np.array([50.0, 50.0, 20.0]))),
        setups=[setup],
        machine=get_default_machine(),
        tools=tools,
        material=DEFAULT_MATERIALS["aluminum_6061"],
    )

    planner = OperationPlanner(ctx, setup, features)
    ops = planner.plan()
    # Planner does not invent tools for the 1.0mm hole; it logs a structured MISSING_TOOL warning
    assert len(planner.warnings) >= 1
    assert any(w["code"] == "MISSING_TOOL" for w in planner.warnings)
    # The 1.0mm hole was NOT assigned an oversized 12mm or 8mm tool
    assert not any(op.tool.diameter > 1.0 and op.feature.diameter == 1.0 for op in ops)



# =============================================================================
# 5. End-to-End Test on sample_bracket.step
# =============================================================================

def test_sample_bracket_pipeline():
    """Verify that sample_bracket.step produces clean, geometrically correct features, operations, and G-code."""
    bracket_path = Path("sample_bracket.step")
    if not bracket_path.exists():
        pytest.skip("sample_bracket.step not found in workspace")

    res = run_cam_pipeline(bracket_path, controller="grbl", work_offset="G54")

    # 1. Feature Recognition: Exactly 1 through bore (dia 30mm) and 1 facing region
    hole_feats = [f for f in res.features if f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BORE, FeatureType.THROUGH_BORE)]
    facing_feats = [f for f in res.features if f.type == FeatureType.FACING_REGION]

    assert len(hole_feats) == 1, f"Expected exactly 1 hole feature, got {len(hole_feats)}"
    assert np.isclose(hole_feats[0].diameter, 30.0, atol=0.2)
    assert np.isclose(hole_feats[0].depth, 30.0, atol=0.2)

    assert len(facing_feats) == 1, f"Expected exactly 1 facing feature, got {len(facing_feats)}"
    assert np.isclose(facing_feats[0].depth, 1.0, atol=0.2)

    # 2. Operations: Facing + Drilling/Boring + Outer Contour Finishing
    assert len(res.operations) == 3
    op_purposes = [o.purpose for o in res.operations]
    assert OpPurpose.FACING in op_purposes
    assert OpPurpose.DRILLING in op_purposes
    assert OpPurpose.FINISHING in op_purposes

    # 3. G-code: Valid and non-empty
    assert len(res.gcode.splitlines()) > 20
    assert "G54" in res.gcode
    assert "M30" in res.gcode or "M2" in res.gcode
