"""Setup sheet and tool list generation for shop floor documentation.

Generates formatted text reports for CNC operators with setup information,
tool lists, operation summaries, and estimated cycle times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .context import PlanningContext, Setup, Tool, ToolType
from .features import MachiningFeature
from .operations import PlannedOperation, OpPurpose
from .toolpaths.semantic import Toolpath
from .post.post_processor import PostResult


@dataclass
class SetupSheet:
    """Complete setup sheet for shop floor documentation."""
    program_name: str
    part_name: str
    material: str
    machine: str
    controller: str
    units: str
    setup_name: str
    work_offset: str
    stock_dimensions: str
    stock_material: str
    tool_list: list[dict]
    operations: list[dict]
    estimated_cycle_time: float
    generated_at: str
    notes: list[str]


def generate_setup_sheet(
    program_name: str,
    part_name: str,
    planning_ctx: PlanningContext,
    setup: Setup,
    features: list[MachiningFeature],
    operations: list[PlannedOperation],
    toolpaths: list[Toolpath],
    post_result: Optional[PostResult] = None,
) -> SetupSheet:
    """Generate a complete setup sheet for the given setup."""
    stock = setup.stock
    stock_dims = stock.bounds_max - stock.bounds_min

    tool_list = _build_tool_list(planning_ctx.tools, operations, toolpaths)
    op_list = _build_operation_list(operations, toolpaths)
    total_time = post_result.total_time_seconds if post_result else sum(
        op.get("estimated_time", 0) for op in op_list
    )

    notes = []
    if stock.kind.value == "cylinder":
        notes.append("Cylindrical stock - ensure proper chucking")
    if any(op.purpose == OpPurpose.DRILLING for op in operations):
        notes.append("Verify drill center locations before machining")
    if planning_ctx.material.needs_coolant:
        notes.append("Coolant required for this material")

    return SetupSheet(
        program_name=program_name,
        part_name=part_name,
        material=planning_ctx.material.name,
        machine=planning_ctx.machine.name,
        controller=planning_ctx.machine.controller,
        units="mm",
        setup_name=setup.name,
        work_offset=setup.work_offset.value,
        stock_dimensions=f"{stock_dims[0]:.1f} x {stock_dims[1]:.1f} x {stock_dims[2]:.1f} mm",
        stock_material=planning_ctx.material.name,
        tool_list=tool_list,
        operations=op_list,
        estimated_cycle_time=total_time,
        generated_at=datetime.now().isoformat(),
        notes=notes,
    )


def format_setup_sheet(sheet: SetupSheet) -> str:
    """Format a SetupSheet into a human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SETUP SHEET - GENERIC CAM STUDIO")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Program Name:    {sheet.program_name}")
    lines.append(f"Part Name:       {sheet.part_name}")
    lines.append(f"Generated:       {sheet.generated_at}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("MACHINE & MATERIAL")
    lines.append("-" * 70)
    lines.append(f"Machine:         {sheet.machine}")
    lines.append(f"Controller:      {sheet.controller}")
    lines.append(f"Units:           {sheet.units}")
    lines.append(f"Material:        {sheet.material}")
    lines.append(f"Stock Size:      {sheet.stock_dimensions}")
    lines.append(f"Stock Material:  {sheet.stock_material}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SETUP INFORMATION")
    lines.append("-" * 70)
    lines.append(f"Setup Name:      {sheet.setup_name}")
    lines.append(f"Work Offset:     {sheet.work_offset}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOOL LIST")
    lines.append("-" * 70)
    lines.append(f"{'T#':<5} {'Type':<20} {'Dia (mm)':<10} {'Flute (mm)':<12} {'Description'}")
    lines.append("-" * 70)
    for tool in sheet.tool_list:
        lines.append(
            f"{tool['tool_number']:<5} {tool['type']:<20} {tool['diameter']:<10.2f} "
            f"{tool['flute_length']:<12.1f} {tool['description']}"
        )
    lines.append("")
    lines.append("-" * 70)
    lines.append("OPERATIONS")
    lines.append("-" * 70)
    lines.append(f"{'Op#':<8} {'Purpose':<15} {'Tool':<8} {'Feed':<10} {'RPM':<8} {'Time (s)'}")
    lines.append("-" * 70)
    for op in sheet.operations:
        lines.append(
            f"{op['id']:<8} {op['purpose']:<15} {op['tool']:<8} "
            f"{op['feed_rate']:<10.0f} {op['spindle_rpm']:<8.0f} {op.get('estimated_time', 0):.1f}"
        )
    lines.append("")
    lines.append(f"Estimated Total Cycle Time: {sheet.estimated_cycle_time:.1f} seconds "
                 f"({sheet.estimated_cycle_time / 60:.1f} minutes)")
    lines.append("")
    if sheet.notes:
        lines.append("-" * 70)
        lines.append("NOTES")
        lines.append("-" * 70)
        for note in sheet.notes:
            lines.append(f"  - {note}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _build_tool_list(tools, operations, toolpaths) -> list[dict]:
    used_tool_ids = set()
    for op in operations:
        used_tool_ids.add(op.tool.id)
    for tp in toolpaths:
        used_tool_ids.add(tp.tool_id)

    tool_list = []
    for tool in tools:
        if tool.id in used_tool_ids:
            type_name = tool.type.value.replace("_", " ").title()
            desc = f"{type_name} - {tool.flutes} flute"
            if tool.material:
                desc += f", {tool.material}"
            tool_list.append({
                "tool_number": tool.tool_number,
                "id": tool.id,
                "type": type_name,
                "diameter": tool.diameter,
                "flute_length": tool.flute_length,
                "overall_length": tool.overall_length,
                "flutes": tool.flutes,
                "description": desc,
            })
    return tool_list


def _build_operation_list(operations, toolpaths) -> list[dict]:
    tp_map = {tp.operation_id: tp for tp in toolpaths}
    op_list = []
    for op in operations:
        tp = tp_map.get(op.id)
        est_time = 0.0
        if tp:
            cut_len = tp.cutting_length()
            rapid_len = tp.rapid_length()
            feed = op.params.feed_rate
            rapid_feed = 5000.0
            est_time = (cut_len / max(feed, 10.0) + rapid_len / max(rapid_feed, 100.0)) * 60.0

        op_list.append({
            "id": op.id,
            "purpose": op.purpose.value,
            "feature_id": op.feature.id,
            "tool": op.tool.id,
            "tool_number": op.tool.tool_number,
            "feed_rate": op.params.feed_rate,
            "spindle_rpm": op.params.spindle_rpm,
            "stepover_mm": op.params.stepover_mm,
            "stepdown_mm": op.params.depth_of_cut_mm,
            "estimated_time": est_time,
        })
    return op_list
