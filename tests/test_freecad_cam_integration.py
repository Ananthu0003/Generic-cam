"""Test suite verifying integration of FreeCAD CAM algorithms and post-processors."""

import pytest
import numpy as np
from cam_engine.algorithms.facing import generate_zigzag_facing, generate_spiral_facing, FacingPattern
from cam_engine.algorithms.dressups import generate_lead_in_out, generate_dogbone_corners, generate_ramp_entry
from cam_engine.post import (
    get_post_processor,
    Mach3Mach4PostProcessor,
    CentroidPostProcessor,
    MassoPostProcessor,
    MarlinPostProcessor,
    SmoothiePostProcessor,
    UccncPostProcessor,
)
from cam_engine.toolpaths.semantic import Toolpath, MotionSegment, MotionType


def test_zigzag_facing_generates_smooth_turnaround_arcs():
    """Verify FreeCAD-inspired zigzag facing creates smooth 180-deg connecting arcs."""
    bmin = [-20.0, -10.0]
    bmax = [20.0, 10.0]
    segs = generate_zigzag_facing(
        bounds_min=bmin,
        bounds_max=bmax,
        tool_diameter=12.0,
        stepover_mm=8.0,
        z_depth=0.0,
        smooth_arc_links=True,
    )
    assert len(segs) > 3
    # Check that arc links exist between rows
    arc_moves = [s for s in segs if s.motion_type in ("arc_cw", "arc_ccw")]
    assert len(arc_moves) >= 1
    for arc in arc_moves:
        assert arc.arc_center is not None
        assert arc.start[2] == pytest.approx(0.0)
        assert arc.end[2] == pytest.approx(0.0)


def test_spiral_facing_generates_inward_clearing():
    """Verify FreeCAD-inspired spiral facing generates inward rectangular spiral."""
    bmin = [-30.0, -15.0]
    bmax = [30.0, 15.0]
    segs = generate_spiral_facing(
        bounds_min=bmin,
        bounds_max=bmax,
        tool_diameter=10.0,
        stepover_mm=5.0,
        z_depth=-1.0,
    )
    assert len(segs) >= 4
    for s in segs:
        assert s.motion_type == "cut"
        assert s.start[2] == pytest.approx(-1.0)
        assert s.end[2] == pytest.approx(-1.0)


def test_lead_in_out_calculates_tangential_arcs():
    """Verify FreeCAD-inspired leadinout generates smooth 90-degree tangential entry/exit arcs."""
    p_start = np.array([0.0, 0.0, -5.0])
    p_next = np.array([20.0, 0.0, -5.0])
    p_end = np.array([20.0, 10.0, -5.0])
    p_prev_end = np.array([0.0, 10.0, -5.0])

    lead = generate_lead_in_out(
        start_pt=p_start,
        next_pt=p_next,
        end_pt=p_end,
        prev_end_pt=p_prev_end,
        tool_radius=5.0,
        side="left",
    )
    assert lead.entry_approach_point[2] == pytest.approx(-5.0)
    assert lead.lead_in_arc_center is not None
    assert lead.lead_out_arc_center is not None
    # Entry approach should be positioned before the lead-in start along the cut vector
    assert lead.entry_approach_point[0] < lead.lead_in_start[0]


def test_dogbone_internal_corner_relief():
    """Verify FreeCAD-inspired dogbone overcuts internal 90-deg corners."""
    square_pocket = [
        [0.0, 0.0],
        [50.0, 0.0],
        [50.0, 30.0],
        [0.0, 30.0],
    ]
    dogbone_pts = generate_dogbone_corners(
        polygon=square_pocket,
        tool_radius=3.0,
        relief_factor=1.05,
    )
    # The 4 internal corners should each have dogbone overcut vertices inserted
    assert len(dogbone_pts) > len(square_pocket)


def test_ramp_entry_generation():
    """Verify FreeCAD-inspired progressive ramp entry replaces vertical plunge."""
    start_pt = [10.0, 10.0, 2.0]
    target_z = -4.0
    ramps = generate_ramp_entry(start_pt, target_z, ramp_angle_deg=15.0, max_stepdown=2.0)
    assert len(ramps) >= 2
    assert ramps[0][0][2] == pytest.approx(2.0)
    assert ramps[-1][1][2] == pytest.approx(-4.0)


def test_freecad_post_processors_instantiation_and_export():
    """Verify all FreeCAD controller post-processors export clean syntax."""
    tp = Toolpath(
        operation_id="op_test",
        feature_id="feat_1",
        tool_id="T01",
        purpose="finishing",
    )
    tp.add(MotionSegment(
        motion_type=MotionType.RAPID,
        start=np.array([0.0, 0.0, 10.0]),
        end=np.array([10.0, 10.0, 10.0]),
        tool_id="T01",
        operation_id="op_test",
    ))
    tp.add(MotionSegment(
        motion_type=MotionType.CUT,
        start=np.array([10.0, 10.0, 10.0]),
        end=np.array([10.0, 10.0, 0.0]),
        feed=1000.0,
        spindle=5000.0,
        tool_id="T01",
        operation_id="op_test",
    ))

    controllers = ["mach3", "mach4", "centroid", "masso", "marlin", "smoothie", "uccnc"]
    for c in controllers:
        post = get_post_processor(controller=c, work_offset="G54")
        assert post is not None
        res = post.post_process([tp], program_name=f"TEST_{c.upper()}")
        assert res.gcode is not None
        assert len(res.gcode) > 0
        assert "M3" in res.gcode or "M03" in res.gcode
        assert "G1" in res.gcode or "G01" in res.gcode


def test_holding_tabs_generation():
    """Verify FreeCAD-inspired Tags / Holding tabs elevate Z above floor."""
    from cam_engine.algorithms.dressups import generate_holding_tabs
    contour = [
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 50.0],
        [0.0, 50.0],
        [0.0, 0.0],
    ]
    tabbed_pts = generate_holding_tabs(
        contour_points=contour,
        tab_count=4,
        tab_width=6.0,
        tab_height=2.5,
        floor_z=-10.0,
        tab_shape="trapezoidal",
    )
    assert len(tabbed_pts) > len(contour)
    # Check that some points reach floor_z + tab_height (-7.5)
    elevated_pts = [p for p in tabbed_pts if p[2] == pytest.approx(-7.5)]
    assert len(elevated_pts) >= 4


def test_touch_probing_cycles():
    """Verify FreeCAD-inspired PathProbe generates surface, bore, and corner probing cycles."""
    from cam_engine.algorithms.probing import (
        generate_surface_probe,
        generate_bore_boss_probe,
        generate_corner_probe,
        ProbeType,
    )
    # Surface Z
    surf = generate_surface_probe(approach_pt=[10.0, 20.0, 5.0], probe_axis="Z", probe_direction=-1)
    assert surf.cycle_type == ProbeType.SURFACE_Z
    assert any("G31 Z" in line for line in surf.gcode_blocks)

    # Bore 4-point
    bore = generate_bore_boss_probe(center_xy=[50.0, 50.0], depth_z=-5.0, clearance_z=10.0, diameter=25.0)
    assert bore.cycle_type == ProbeType.BORE_CENTER
    assert len(bore.points) == 12  # 4 directions * 3 points

    # Corner external
    corner = generate_corner_probe(corner_xy=[0.0, 0.0], depth_z=-3.0, clearance_z=5.0, quadrant=1)
    assert corner.cycle_type == ProbeType.CORNER_EXTERNAL
    assert len(corner.points) == 4


def test_engraving_and_vcarve():
    """Verify FreeCAD-inspired line engraving and variable depth V-carving."""
    from cam_engine.algorithms.engraving import generate_line_engraving, generate_v_carve
    
    # 2D line engraving
    strokes = [
        [[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]],
    ]
    motions = generate_line_engraving(strokes, depth_z=-0.5, retract_z=2.0)
    assert len(motions) == 4  # plunge + 2 cuts + retract
    assert motions[0].motion_type == "plunge"
    assert motions[-1].motion_type == "retract"

    # V-carving with variable stroke width
    centerline = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
    widths = [2.0, 4.0, 2.0]  # wider in middle -> deeper cut
    v_motions = generate_v_carve(centerline, widths, v_bit_included_angle_deg=90.0, top_z=0.0)
    assert len(v_motions) == 4  # plunge + 2 cuts + retract
    # Middle cut should go deeper (at width 4.0, 90-deg V-bit: depth = -4.0 / (2 * tan(45)) = -2.0)
    assert v_motions[1].end[2] == pytest.approx(-2.0)


def test_tool_type_freecad_extensions():
    """Verify extended ToolType enum includes specialized tools."""
    from cam_engine.context import ToolType
    assert ToolType.V_BIT.value == "v_bit"
    assert ToolType.DOVETAIL_CUTTER.value == "dovetail_cutter"
    assert ToolType.SLITTING_SAW.value == "slitting_saw"
    assert ToolType.TOUCH_PROBE.value == "touch_probe"


def test_linear_and_polar_array_dressups():
    """Verify FreeCAD-inspired linear and polar toolpath array replication."""
    from cam_engine.algorithms.dressups import generate_linear_array, generate_polar_array
    base_segment = [(np.array([0.0, 0.0, -1.0]), np.array([10.0, 0.0, -1.0]))]

    # 2x3 linear grid
    linear_segs = generate_linear_array(base_segment, count_x=2, count_y=3, offset_x=20.0, offset_y=15.0)
    assert len(linear_segs) == 6
    assert linear_segs[0][0][0] == pytest.approx(0.0)
    assert linear_segs[-1][0][0] == pytest.approx(20.0)
    assert linear_segs[-1][0][1] == pytest.approx(30.0)

    # 4-point polar circular array around (0,0)
    polar_segs = generate_polar_array(base_segment, count=4, total_angle_deg=360.0, center_xy=[0.0, 0.0])
    assert len(polar_segs) == 4
    # 90-degree rotated segment should point along +Y
    assert polar_segs[1][1][0] == pytest.approx(0.0, abs=1e-5)
    assert polar_segs[1][1][1] == pytest.approx(10.0, abs=1e-5)


