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
    CHAMFERING = "chamfering"


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
        return ops

    def _plan_facing(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type is not FeatureType.FACING_REGION:
                continue
            excess = f.notes.get("excess_height", 0.0)
            if excess <= 1e-4:
                continue
            tool = self._pick_facing_tool(f)
            params = self.analyzer.analyze(tool, f, OpPurpose.FACING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_facing"), purpose=OpPurpose.FACING,
                feature=f, tool=tool, params=params,
                notes={"excess_height": excess}))
        return ops

    def _pick_facing_tool(self, f: MachiningFeature) -> Tool:
        span = float(np.max(f.xy_extent))
        candidates = [t for t in self.context.tools
                      if t.type in ToolType_mill()]
        if not candidates:
            raise CamError(
                code=UNSUPPORTED_FEATURE,
                message=f"facing region {f.id}: no milling tool available in library",
                stage="operation_planning", feature_id=f.id)
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
                raise CamError(
                    code=FEATURE_INACCESSIBLE,
                    message=f"feature {f.id} ({f.type.value}) inaccessible along tool axis",
                    stage="operation_planning", feature_id=f.id)
            tool = self._pick_roughing_tool(f)
            params = self.analyzer.analyze(tool, f, OpPurpose.ROUGHING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_rough"), purpose=OpPurpose.ROUGHING,
                feature=f, tool=tool, params=params))
        return ops

    def _pick_roughing_tool(self, f: MachiningFeature) -> Tool:
        extent = f.xy_extent
        min_span = float(min(extent))
        fit = [t for t in self.context.tools
               if t.type in ToolType_mill()
               and t.diameter < min_span]
        if not fit:
            # Check if any milling tool exists
            all_mills = [t for t in self.context.tools if t.type in ToolType_mill()]
            if not all_mills:
                raise CamError(
                    code=UNSUPPORTED_FEATURE,
                    message=f"feature {f.id}: no milling tool available in library",
                    stage="operation_planning", feature_id=f.id)
            return min(all_mills, key=lambda t: t.diameter)
        return max(fit, key=lambda t: t.diameter)

    def _plan_rest_machining(self, prior: list[PlannedOperation]) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        rough_ops = [o for o in prior if o.purpose is OpPurpose.ROUGHING]
        for rough in rough_ops:
            f = rough.feature
            extent = f.xy_extent
            min_span = float(min(extent))
            smaller = [t for t in self.context.tools
                       if t.type in ToolType_mill()
                       and t.diameter < rough.tool.diameter - 1e-6
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
            finish_tools = [t for t in self.context.tools
                            if t.type in ToolType_mill() and t.diameter < rough.tool.diameter]
            if not finish_tools:
                continue
            wall_stock = rough.params.depth_of_cut_mm * 0.5
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
        else:
            pool = [t for t in self.context.tools if t.type in ToolType_mill()]
        extent = f.xy_extent
        min_span = float(min(extent)) if len(extent) else 1e9
        fit = [t for t in pool if t.diameter < min_span]
        pool2 = fit or pool
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
            exact = [d for d in drills if abs(d.diameter - f.diameter) < 1e-4]
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
                raise CamError(
                    code=UNSUPPORTED_FEATURE,
                    message=f"feature {f.id} dia {f.diameter:.2f}mm: no drill or endmill fits",
                    stage="operation_planning", feature_id=f.id)
            tool = max(mills, key=lambda t: t.diameter)
            params = self.analyzer.analyze(tool, f, OpPurpose.DRILLING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_drill"), purpose=OpPurpose.DRILLING,
                feature=f, tool=tool, params=params,
                notes={"method": "helical_interpolation", "target_diameter": f.diameter}))
        return ops

    def _plan_boring(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type not in (FeatureType.BORE, FeatureType.COUNTERBORE) or f.diameter is None:
                continue
            boring_bars = [t for t in self.context.tools if t.type is ToolType.BORING_BAR]
            if not boring_bars:
                continue
            exact = [b for b in boring_bars if abs(b.diameter - f.diameter) < 1.0]
            if not exact:
                continue
            tool = exact[0]
            params = self.analyzer.analyze(tool, f, OpPurpose.BORING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_bore"), purpose=OpPurpose.BORING,
                feature=f, tool=tool, params=params))
        return ops

    def _plan_reaming(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if f.type not in (FeatureType.HOLE, FeatureType.THROUGH_HOLE, FeatureType.BORE) or f.diameter is None:
                continue
            reamers = [t for t in self.context.tools
                       if t.type is ToolType.DRILL and "ream" in t.id.lower()]
            exact = [r for r in reamers if abs(r.diameter - f.diameter) < 0.05]
            if not exact:
                continue
            tool = exact[0]
            params = self.analyzer.analyze(tool, f, OpPurpose.REAMING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_ream"), purpose=OpPurpose.REAMING,
                feature=f, tool=tool, params=params))
        return ops

    def _plan_tapping(self) -> list[PlannedOperation]:
        ops: list[PlannedOperation] = []
        for f in self.features:
            if not f.notes.get("tapped") or f.diameter is None:
                continue
            taps = [t for t in self.context.tools if t.type is ToolType.DRILL and "tap" in t.id.lower()]
            exact = [t for t in taps if abs(t.diameter - f.diameter) < 0.1]
            if not exact:
                continue
            tool = exact[0]
            params = self.analyzer.analyze(tool, f, OpPurpose.TAPPING.value, self.setup)
            ops.append(PlannedOperation(
                id=self._next_id("op_tap"), purpose=OpPurpose.TAPPING,
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


def ToolType_mill():
    """Milling-capable tool types."""
    from .context import ToolType
    return (ToolType.FLAT_ENDMILL, ToolType.BALL_ENDMILL, ToolType.BULLNOSE_ENDMILL, ToolType.FACE_MILL)
