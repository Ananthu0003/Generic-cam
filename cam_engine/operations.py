"""Operation planning: derive the machining sequence from geometry and
manufacturing requirements. No fixed sequence is assumed; operations exist only
when the geometry and stock state demand them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .context import PlanningContext, Setup, Tool, ToolType
from .errors import CamError, FEATURE_INACCESSIBLE, UNSUPPORTED_FEATURE
from .features import (Accessibility, FeatureRecognizer, FeatureType,
                       MachiningFeature)
from .machinability import CuttingParameters, MachinabilityAnalyzer


class OpPurpose(str, Enum):
    FACING = "facing"
    ROUGHING = "roughing"
    REST_MACHINING = "rest_machining"
    SEMI_FINISHING = "semi_finishing"
    FINISHING = "finishing"
    DRILLING = "drilling"
    BORING = "boring"
    SPOT_DRILLING = "spot_drilling"
    REAMING = "reaming"
    TAPPING = "tapping"
    COUNTERSINKING = "countersinking"  # countersink / spotfacing with conical tool
    CHAMFERING = "chamfering"
    THREAD_MILLING = "thread_milling"
    GROOVING = "grooving"


@dataclass
class PlannedOperation:
    """One planned machining operation (semantic, pre-toolpath)."""

    id: str
    purpose: OpPurpose
    feature: MachiningFeature
    tool: Tool
    params: CuttingParameters
    depends_on: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


class OperationPlanner:
    """Plans operations per setup from recognized features + stock state."""

    def __init__(self, context: PlanningContext, setup: Setup,
                 features: list[MachiningFeature]):
        self.context = context
        self.setup = setup
        self.features = features
        self.analyzer = MachinabilityAnalyzer(context.machine, context.material)
        self._counter: dict[str, int] = {}
        self.warnings: list[dict] = []
        self.unmachined_features: list[dict] = []

    def _next_id(self, prefix: str) -> str:
        n = self._counter.get(prefix, 0) + 1
        self._counter[prefix] = n
        return f"{prefix}_{n:03d}"

    def plan(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        ops += self._plan_facing()
        ops += self._plan_spot_drilling()
        ops += self._plan_roughing()
        ops += self._plan_rest_machining(ops)
        ops += self._plan_semi_finishing(ops)
        ops += self._plan_finishing()
        ops += self._plan_chamfering()
        ops += self._plan_drilling()
        ops += self._plan_boring()
        ops += self._plan_reaming()
        ops += self._plan_tapping()
        ops += self._plan_thread_milling()
        ops += self._plan_grooving()
        ops += self._plan_countersinking()
        # Topological sort: reorder by depends_on dependencies
        return self._topo_sort(ops)

    @staticmethod
    def _topo_sort(ops: list[PlannedOperation]) -> list[PlannedOperation]:
        """Topological sort of operations by depends_on relationships.
        Operations with no dependencies come first. Within the same
        dependency level, preserve the original planner ordering."""
        if not ops:
            return ops
        # Build op lookup by id and by feature id
        op_by_id = {o.id: o for o in ops}
        # Map feature_id -> list of op ids that depend on it
        dep_graph: dict[str, list[str]] = {o.id: [] for o in ops}
        in_degree: dict[str, int] = {o.id: 0 for o in ops}
        for o in ops:
            for dep_id in (o.depends_on or []):
                # dep_id could be a feature id or an operation id
                # Find the operation that produces this dependency
                producer = None
                for candidate in ops:
                    if candidate.id == dep_id or candidate.feature.id == dep_id:
                        producer = candidate.id
                        break
                if producer and producer != o.id:
                    dep_graph[producer].append(o.id)
                    in_degree[o.id] += 1
        # Kahn's algorithm
        queue = [oid for oid, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            # Stable: preserve original order within same level
            queue.sort(key=lambda oid: ops.index(op_by_id[oid]))
            oid = queue.pop(0)
            result.append(op_by_id[oid])
            for child in dep_graph[oid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        # If cycle detected, append remaining in original order
        if len(result) < len(ops):
            seen = {o.id for o in result}
            for o in ops:
                if o.id not in seen:
                    result.append(o)
        return result

    def _plan_facing(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type is not FeatureType.FACING_REGION:
                continue
            excess = f.notes.get("excess_height", 0.0)
            if excess <= 1e-4:
                continue
            tool = self._pick_facing_tool(f)
            if tool is None:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.FACING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_facing"), purpose=OpPurpose.FACING,
                feature=f, tool=tool, params=params,
                notes={"excess_height": excess}))
        return ops

    def _pick_facing_tool(self, f: MachiningFeature) -> Optional[Tool]:
        candidates = [t for t in self.context.tools
                      if t.type in ToolType_mill()]
        if not candidates:
            self.warnings.append({
                "code": "MISSING_TOOL",
                "feature_id": f.id,
                "feature_type": "facing_region",
                "message": f"facing region {f.id}: no milling tool available in library",
                "suggested_tool": {
                    "type": "face_mill",
                    "diameter": 50.0,
                    "flute_length": 15.0,
                    "overall_length": 60.0,
                    "flutes": 4,
                }
            })
            return None
        # Prefer face mill or larger flat endmill
        face_mills = [t for t in candidates if t.type is ToolType.FACE_MILL]
        pool = face_mills or candidates
        return max(pool, key=lambda t: t.diameter)

    def _plan_roughing(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        roughable = (
            FeatureType.POCKET, FeatureType.OPEN_POCKET,
            FeatureType.SLOT, FeatureType.OPEN_SLOT,
            FeatureType.STEP, FeatureType.SHOULDER,
        )
        for f in self.features:
            if f.type not in roughable:
                continue
            if f.accessibility is Accessibility.INACCESSIBLE:
                self.warnings.append({
                    "code": "FEATURE_INACCESSIBLE",
                    "feature_id": f.id,
                    "feature_type": f.type.value if hasattr(f.type, "value") else str(f.type),
                    "message": f"feature {f.id} ({f.type.value}) inaccessible along tool axis",
                })
                continue
            tool = self._pick_roughing_tool(f)
            if tool is None:
                continue
            params = self.analyzer.analyze(tool, f, OpPurpose.ROUGHING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_rough"), purpose=OpPurpose.ROUGHING,
                feature=f, tool=tool, params=params))
        return ops

    def _pick_roughing_tool(self, f: MachiningFeature) -> Optional[Tool]:
        extent = f.xy_extent
        min_span = float(min(extent))
        endmills = [t for t in self.context.tools if t.type in (ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL)]
        fit = [t for t in endmills if t.diameter < min_span]
        if not fit:
            all_mills = endmills or [t for t in self.context.tools if t.type in ToolType_mill()]
            if not all_mills:
                self.warnings.append({
                    "code": "MISSING_TOOL",
                    "feature_id": f.id,
                    "feature_type": f.type.value if hasattr(f.type, "value") else str(f.type),
                    "message": f"feature {f.id}: no milling tool available in library",
                    "suggested_tool": {
                        "type": "flat_endmill",
                        "diameter": round(max(min_span * 0.7, 0.5), 2),
                        "flute_length": round(max((f.depth or 5.0) * 1.25, 5.0), 1),
                        "overall_length": 50.0,
                        "flutes": 3,
                    }
                })
                return None
            return min(all_mills, key=lambda t: t.diameter)
        return max(fit, key=lambda t: t.diameter)

    def _plan_rest_machining(self, prior: list[PlannedOperation]) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        rough_ops = [o for o in prior if o.purpose is OpPurpose.ROUGHING]
        for rough in rough_ops:
            f = rough.feature
            extent = f.xy_extent
            min_span = float(min(extent))
            endmills = [t for t in self.context.tools if t.type in (ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL)]
            smaller = [t for t in endmills
                       if t.diameter < rough.tool.diameter - 1e-6
                       and t.diameter < min_span]
            if not smaller:
                continue
            rest_tool = max(smaller, key=lambda t: t.diameter)
            params = self.analyzer.analyze(rest_tool, f, OpPurpose.REST_MACHINING.value,
                                           self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_rest"), purpose=OpPurpose.REST_MACHINING,
                feature=f, tool=rest_tool, params=params,
                depends_on=[rough.id],
                notes={"parent_rough_tool": rough.tool.id}))
        return ops

    def _plan_semi_finishing(self, prior: list[PlannedOperation]) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        rough_ops = [o for o in prior if o.purpose is OpPurpose.ROUGHING]
        for rough in rough_ops:
            f = rough.feature
            endmills = [t for t in self.context.tools if t.type in (ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL)]
            finish_tools = [t for t in endmills if t.diameter < rough.tool.diameter]
            if not finish_tools:
                continue
            wall_stock = rough.params.stepover_mm * 0.5
            if wall_stock > 1.5 * min(t.diameter for t in finish_tools):
                tool = max(finish_tools, key=lambda t: t.diameter)
                params = self.analyzer.analyze(tool, f, OpPurpose.SEMI_FINISHING.value,
                                               self.setup)
                ops.append(PlannedOperation(
                    id=self._next_id("op_semi"), purpose=OpPurpose.SEMI_FINISHING,
                    feature=f, tool=tool, params=params,
                    depends_on=[rough.id]))
        return ops

    def _plan_finishing(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        finishable = (
            FeatureType.POCKET, FeatureType.OPEN_POCKET,
            FeatureType.SLOT, FeatureType.OPEN_SLOT,
            FeatureType.STEP, FeatureType.SHOULDER,
            FeatureType.FREEFORM_SURFACE, FeatureType.CONTOUR,
        )
        for f in self.features:
            if f.type in finishable:
                if f.accessibility is Accessibility.INACCESSIBLE:
                    continue
                extent = f.xy_extent
                if f.type is FeatureType.CONTOUR and float(min(extent)) < 1e-6:
                    continue
                tool = self._pick_finishing_tool(f)
                params = self.analyzer.analyze(tool, f, OpPurpose.FINISHING.value,
                                               self.setup)
                ops.append(PlannedOperation(
                    id=self._next_id("op_finish"), purpose=OpPurpose.FINISHING,
                    feature=f, tool=tool, params=params))
        return ops

    def _pick_finishing_tool(self, f: MachiningFeature) -> Tool:
        if f.type is FeatureType.FREEFORM_SURFACE:
            balls = [t for t in self.context.tools if t.type is ToolType.BALL_ENDMILL]
            pool = balls or [t for t in self.context.tools if t.type in ToolType_mill()]
        elif f.type is FeatureType.CHAMFER:
            chams = [t for t in self.context.tools if t.type is ToolType.CHAMFER_MILL]
            pool = chams or [t for t in self.context.tools if t.type in ToolType_mill()]
        elif f.type is FeatureType.FACING_REGION:
            pool = [t for t in self.context.tools if t.type in ToolType_mill()]
        else:
            endmills = [t for t in self.context.tools if t.type in (ToolType.FLAT_ENDMILL, ToolType.BULLNOSE_ENDMILL)]
            pool = endmills or [t for t in self.context.tools if t.type in ToolType_mill()]
        extent = f.xy_extent
        min_span = float(min(extent)) if len(extent) else 1e9
        fit = [t for t in pool if t.diameter < min_span]
        pool2 = fit or pool
        # For 2.5D features (pockets, steps, shoulders, contours, etc.), prefer largest tool that fits
        # for fewer passes and better surface finish
        if f.type in (FeatureType.POCKET, FeatureType.OPEN_POCKET,
                      FeatureType.SLOT, FeatureType.OPEN_SLOT,
                      FeatureType.STEP, FeatureType.SHOULDER,
                      FeatureType.FACING_REGION, FeatureType.CONTOUR,
                      FeatureType.PLANAR_WALL, FeatureType.BOSS):
            return max(pool2, key=lambda t: t.diameter)
        return min(pool2, key=lambda t: t.diameter)

    def _plan_spot_drilling(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        hole_types = (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
        for f in self.features:
            if f.type not in hole_types or f.diameter is None:
                continue
            spot_drills = [t for t in self.context.tools
                           if t.type is ToolType.DRILL and t.diameter <= f.diameter]
            if not spot_drills:
                continue
            tool = min(spot_drills, key=lambda t: t.diameter)
            params = self.analyzer.analyze(tool, f, OpPurpose.SPOT_DRILLING.value, self.setup)
            spot_depth = min(tool.diameter * 0.3, f.depth * 0.1)
            ops.append(PlannedOperation(
                id=self._next_id("op_spot"), purpose=OpPurpose.SPOT_DRILLING,
                feature=f, tool=tool, params=params,
                notes={"spot_depth": spot_depth}))
        return ops

    def _plan_drilling(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        hole_types = (
            FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
            FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE,
            FeatureType.COUNTERBORE,
        )
        for f in self.features:
            if f.type not in hole_types or f.diameter is None:
                continue

            drills = [t for t in self.context.tools if t.type is ToolType.DRILL]
            exact = [d for d in drills if abs(d.diameter - f.diameter) <= 0.05 or abs(d.diameter - round(f.diameter, 2)) < 1e-4]
            if exact and f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                tool = exact[0]
                params = self.analyzer.analyze(tool, f, OpPurpose.DRILLING.value, self.setup)
                ops.append(PlannedOperation(
                    id=self._next_id("op_drill"), purpose=OpPurpose.DRILLING,
                    feature=f, tool=tool, params=params))
                continue

            # If no exact drill, or feature is bore/counterbore -> Plan helical milling or boring
            mills = [t for t in self.context.tools
                     if t.type in ToolType_mill() and t.diameter < f.diameter]
            if not mills:
                is_std_hole = f.type in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
                sugg_type = "drill" if is_std_hole else "flat_endmill"
                if is_std_hole:
                    sugg_dia = round(f.diameter, 2)
                else:
                    sugg_dia = round(max(f.diameter * 0.7, 0.2), 2)
                    if sugg_dia >= f.diameter:
                        sugg_dia = round(f.diameter * 0.5, 2)

                self.warnings.append({
                    "code": "MISSING_TOOL",
                    "feature_id": f.id,
                    "feature_type": f.type.value if hasattr(f.type, "value") else str(f.type),
                    "feature_diameter": round(f.diameter, 2),
                    "depth": round(f.depth, 2) if f.depth else 5.0,
                    "message": f"Feature {f.id} (dia {f.diameter:.2f}mm): no {'drill or endmill' if is_std_hole else 'fitting endmill'} in active library",
                    "suggested_tool": {
                        "type": sugg_type,
                        "diameter": sugg_dia,
                        "flute_length": round(max((f.depth or 5.0) * 1.25, 5.0), 1),
                        "overall_length": round(max((f.depth or 5.0) * 2.0, 30.0), 1),
                        "flutes": 2,
                    }
                })
                self.unmachined_features.append({
                    "id": f.id,
                    "diameter": round(f.diameter, 2),
                    "type": f.type.value if hasattr(f.type, "value") else str(f.type)
                })
                continue
            tool = max(mills, key=lambda t: t.diameter)
            params = self.analyzer.analyze(tool, f, OpPurpose.DRILLING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_drill"), purpose=OpPurpose.DRILLING,
                feature=f, tool=tool, params=params,
                notes={"method": "helical_interpolation", "target_diameter": f.diameter}))
        return ops

    def _plan_boring(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        bore_types = (
            FeatureType.BORE, FeatureType.THROUGH_BORE,
            FeatureType.BLIND_BORE, FeatureType.COUNTERBORE,
        )
        for f in self.features:
            if f.type not in bore_types or f.diameter is None:
                continue
            boring_bars = [t for t in self.context.tools if t.type is ToolType.BORING_BAR]
            if not boring_bars:
                continue
            exact = [b for b in boring_bars if abs(b.diameter - f.diameter) < 0.05]
            if not exact:
                continue
            tool = exact[0]
            params = self.analyzer.analyze(tool, f, OpPurpose.BORING.value, self.setup)
            # Boring must follow whatever drilled the pre-bore pilot hole.
            # The topo_sort resolves feature-id references: any prior op whose
            # feature.id == dep_id is ordered before this boring op.
            deps = [f.id]  # resolves to the drilling/helical op on this feature
            if f.parent_feature_id:
                # Also depend on the counterbore parent's ops
                deps.append(f.parent_feature_id)
            ops.append(PlannedOperation(
                id=self._next_id("op_bore"), purpose=OpPurpose.BORING,
                feature=f, tool=tool, params=params, depends_on=deps))
        return ops

    def _plan_reaming(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type not in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BORE) or f.diameter is None:
                continue
            reamers = [t for t in self.context.tools
                       if t.type is ToolType.REAMER or "ream" in t.id.lower()]
            exact = [r for r in reamers if abs(r.diameter - f.diameter) < 0.05]
            if not exact:
                continue
            tool = exact[0]
            params = self.analyzer.analyze(tool, f, OpPurpose.REAMING.value, self.setup)
            # Reaming requires a pre-drilled hole — declare dependency on the feature
            ops.append(PlannedOperation(
                id=self._next_id("op_ream"), purpose=OpPurpose.REAMING,
                feature=f, tool=tool, params=params, depends_on=[f.id]))
        return ops

    def _plan_tapping(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        # Standard metric thread pitch lookup (diameter -> pitch in mm)
        _metric_pitch = {
            2.0: 0.4, 2.5: 0.45, 3.0: 0.5, 3.5: 0.6, 4.0: 0.7,
            5.0: 0.8, 6.0: 1.0, 8.0: 1.25, 10.0: 1.5, 12.0: 1.75,
            14.0: 2.0, 16.0: 2.0, 18.0: 2.5, 20.0: 2.5, 22.0: 2.5,
            24.0: 3.0, 27.0: 3.0, 30.0: 3.5, 33.0: 3.5, 36.0: 4.0,
        }
        for f in self.features:
            # Accept TAPPED_HOLE feature type or notes["tapped"] for backward compatibility
            if f.type is not FeatureType.TAPPED_HOLE and not f.notes.get("tapped"):
                continue
            if f.diameter is None:
                continue
            taps = [t for t in self.context.tools if t.type is ToolType.TAP or "tap" in t.id.lower()]
            exact = [t for t in taps if abs(t.diameter - f.diameter) < 0.1]
            if not exact:
                continue
            tool = exact[0]
            params = self.analyzer.analyze(tool, f, OpPurpose.TAPPING.value, self.setup)
            # Compute tap feed per revolution from thread pitch
            pitch = _metric_pitch.get(f.diameter, f.diameter * 0.15)
            tap_feed_mm_per_rev = pitch
            # Tapping requires a pre-drilled hole — declare dependency on the feature
            ops.append(PlannedOperation(
                id=self._next_id("op_tap"), purpose=OpPurpose.TAPPING,
                feature=f, tool=tool, params=params, depends_on=[f.id],
                notes={"tap_feed_mm_per_rev": tap_feed_mm_per_rev}))
        return ops

    def _plan_thread_milling(self) -> list[PlannedOperation]:
        """Plan thread milling for TAPPED_HOLE features when thread mill tools are available.
        Thread milling is an alternative to tapping that works for large diameters
        or when rigid tapping is not available."""
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type is not FeatureType.TAPPED_HOLE and not f.notes.get("tapped"):
                continue
            if f.diameter is None:
                continue
            # Only plan thread milling if thread mill tools are available
            thread_mills = [t for t in self.context.tools
                            if t.type is ToolType.THREAD_MILL or "thread" in t.id.lower()]
            if not thread_mills:
                continue
            # Find a thread mill that fits the hole diameter
            fitting = [t for t in thread_mills if t.diameter < f.diameter]
            if not fitting:
                continue
            tool = min(fitting, key=lambda t: f.diameter - t.diameter)
            params = self.analyzer.analyze(tool, f, OpPurpose.THREAD_MILLING.value, self.setup)
            # Thread pitch from standard metric table
            _metric_pitch = {
                2.0: 0.4, 2.5: 0.45, 3.0: 0.5, 3.5: 0.6, 4.0: 0.7,
                5.0: 0.8, 6.0: 1.0, 8.0: 1.25, 10.0: 1.5, 12.0: 1.75,
                14.0: 2.0, 16.0: 2.0, 18.0: 2.5, 20.0: 2.5, 22.0: 2.5,
                24.0: 3.0, 27.0: 3.0, 30.0: 3.5, 33.0: 3.5, 36.0: 4.0,
            }
            pitch = _metric_pitch.get(f.diameter, f.diameter * 0.15)
            # Thread milling requires a pre-drilled hole
            ops.append(PlannedOperation(
                id=self._next_id("op_threadmill"), purpose=OpPurpose.THREAD_MILLING,
                feature=f, tool=tool, params=params, depends_on=[f.id],
                notes={"pitch": pitch, "thread_depth": pitch * 1.2}))
        return ops

    def _plan_grooving(self) -> list[PlannedOperation]:
        """Plan grooving operations for features that require groove cutting.
        Currently matches BORE/THROUGH_BORE features when groove cutters are available."""
        ops: list[PlannedOperation] = []
        for f in self.features:
            # Groove features: undercuts or explicit groove annotations
            is_groove = (f.type in (FeatureType.BORE, FeatureType.THROUGH_BORE)
                         and f.notes.get("groove"))
            if not is_groove:
                continue
            if f.diameter is None:
                continue
            groove_tools = [t for t in self.context.tools
                            if t.type is ToolType.GROOVE_CUTTER or "groove" in t.id.lower()]
            if not groove_tools:
                continue
            # Find a groove cutter that fits
            fitting = [t for t in groove_tools if t.diameter <= f.diameter]
            if not fitting:
                continue
            tool = max(fitting, key=lambda t: t.diameter)
            params = self.analyzer.analyze(tool, f, OpPurpose.GROOVING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_groove"), purpose=OpPurpose.GROOVING,
                feature=f, tool=tool, params=params))
        return ops

    def _plan_chamfering(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type is not FeatureType.CHAMFER:
                continue
            chams = [t for t in self.context.tools if t.type is ToolType.CHAMFER_MILL]
            tool = chams[0] if chams else self._pick_finishing_tool(f)
            params = self.analyzer.analyze(tool, f, OpPurpose.CHAMFERING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_chamfer"), purpose=OpPurpose.CHAMFERING,
                feature=f, tool=tool, params=params))
        return ops

    def _plan_countersinking(self) -> list[PlannedOperation]:
        """Plan countersinking operations for COUNTERSINK features.
        Prefers a dedicated COUNTERSINK_TOOL; falls back to CHAMFER_MILL.
        The operation depends on the parent hole feature's drilling op (resolved
        via feature ID by the topo_sort)."""
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type is not FeatureType.COUNTERSINK:
                continue
            if f.diameter is None:
                continue
            # Prefer dedicated countersink tool, fall back to chamfer mill
            csink_tools = [t for t in self.context.tools
                           if t.type is ToolType.COUNTERSINK_TOOL]
            chamfer_tools = [t for t in self.context.tools
                             if t.type is ToolType.CHAMFER_MILL]
            pool = csink_tools or chamfer_tools
            if not pool:
                continue  # no suitable tool — skip this feature
            # Pick the smallest tool whose diameter covers the countersink major diameter
            fitting = [t for t in pool if t.diameter >= f.diameter - 0.1]
            tool = min(fitting, key=lambda t: t.diameter) if fitting else min(pool, key=lambda t: t.diameter)
            params = self.analyzer.analyze(tool, f, OpPurpose.COUNTERSINKING.value, self.setup)
            # Countersinking must follow drilling of the parent hole
            deps = [f.parent_feature_id] if f.parent_feature_id else []
            semi_angle_deg = f.notes.get("semi_angle_deg", 45.0)
            ops.append(PlannedOperation(
                id=self._next_id("op_csink"), purpose=OpPurpose.COUNTERSINKING,
                feature=f, tool=tool, params=params, depends_on=deps,
                notes={"countersink_angle_deg": 90.0 - float(semi_angle_deg)}))
        return ops


def ToolType_mill():
    """Milling-capable tool types."""
    from .context import ToolType
    return (ToolType.FLAT_ENDMILL, ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL, ToolType.FACE_MILL)
