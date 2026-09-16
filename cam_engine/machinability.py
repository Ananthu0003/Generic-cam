"""Machinability analysis + capability-driven tool selection.

Cutting parameters derive from material properties, tool geometry, machine
limits, and operation requirements - never fixed global constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .context import MachineConfig, Material, Setup, Tool, ToolType
from .errors import (CamError, FEATURE_INACCESSIBLE, INVALID_CUTTING_PARAMETERS,
                     NO_COMPATIBLE_TOOL, TOOL_REACH_INSUFFICIENT)
from .features import Accessibility, FeatureType, MachiningFeature


@dataclass
class CuttingParameters:
    """Validated cutting parameters for one operation."""

    spindle_rpm: float
    feed_rate: float          # mm/min XY cutting feed
    plunge_feed: float        # mm/min vertical feed
    ramp_feed: float          # mm/min ramping entry feed
    feed_per_tooth_mm: float
    depth_of_cut_mm: float    # axial per pass (stepdown)
    stepover_mm: float        # radial engagement per pass
    coolant: bool
    effective_diameter_mm: float   # for ball/bullnose stepover calc
    engagement_angle_deg: float    # average radial engagement
    power_kw: float                # estimated cutting power
    torque_nm: float               # estimated torque at spindle


@dataclass
class ToolCapability:
    tool: Tool
    score: float
    reasons: list[str]


class MachinabilityAnalyzer:
    """Determines valid cutting-parameter ranges and ranks candidate tools."""

    def __init__(self, machine: MachineConfig, material: Material):
        self.machine = machine
        self.material = material

    # -------------------------------------------------- parameter ranges --
    def surface_speed_to_rpm(self, v_m_per_min: float, diameter_mm: float) -> float:
        if diameter_mm <= 0:
            raise CamError(INVALID_CUTTING_PARAMETERS, "tool diameter <= 0", stage="machinability")
        return 1000.0 * v_m_per_min / (np.pi * diameter_mm)

    def analyze(self, tool: Tool, feature: MachiningFeature,
                op_purpose: str, setup: Setup) -> CuttingParameters:
        """Derive cutting parameters from material/tool/machine/operation data."""
        if tool.diameter <= 0:
            raise CamError(INVALID_CUTTING_PARAMETERS,
                           f"tool {tool.id} has invalid diameter",
                           stage="machinability", tool_id=tool.id)

        # --- cutting speed basis by tool material and operation purpose ---
        v_basis = self.material.v_carbide if tool.material == "carbide" else self.material.v_hss
        if op_purpose == "finishing":
            v = v_basis * 1.15
        elif op_purpose == "semi_finishing":
            v = v_basis * 1.05
        else:
            v = v_basis
        rpm = self.surface_speed_to_rpm(v, tool.diameter)
        rpm = float(np.clip(rpm, self.machine.spindle_min_rpm, self.machine.spindle_max_rpm))

        # --- feed per tooth from tool geometry + material hardness ---
        # base chip load scales with diameter and decreases with hardness (industrial carbide basis)
        fz_base = 0.018 * tool.diameter * (100.0 / max(self.material.hardness_hb, 10.0)) ** 0.35
        if op_purpose == "finishing":
            fz_base *= 0.60
        elif op_purpose == "semi_finishing":
            fz_base *= 0.80
        elif op_purpose == "roughing":
            fz_base *= 1.15
        fz = float(np.clip(fz_base, 0.005, 0.40))

        flutes = max(1, tool.flutes)
        feed = rpm * flutes * fz

        # --- engagement limits from material data ---
        max_hm = self.material.max_hm_ratio * tool.diameter
        max_ae = self.material.max_ae_ratio * tool.diameter

        stepdown = max_hm
        if op_purpose == "roughing":
            stepdown = max_hm
        elif op_purpose in ("semi_finishing", "finishing"):
            stepdown = min(max_hm, max(0.4 * tool.diameter, 0.5))

        stepover = max_ae
        if op_purpose == "roughing":
            stepover = max_ae
        elif op_purpose == "semi_finishing":
            stepover = 0.45 * tool.diameter
        elif op_purpose == "finishing":
            stepover = self._finish_stepover(tool)

        # clamp feed to machine axis capability
        max_feed = min(ax.max_feed for ax in self.machine.axes.values())
        feed = float(min(feed, max_feed))
        plunge = float(min(feed * 0.40, max_feed))
        ramp = float(min(feed * 0.70, max_feed))

        # engagement angle for slotting approx: ae = D*sin(theta/2) -> theta = 2*asin(ae/D)
        ae_ratio = min(1.0, stepover / tool.diameter)
        engagement = float(np.degrees(2.0 * np.arcsin(np.clip(ae_ratio, 0.0, 1.0))))

        # power/torque estimate: MRR = ap * ae * feed/60 (mm3/s)
        mrr = stepdown * stepover * feed / 60.0
        # power_kW = MRR_mm3s * Kc_kNmm2 / 1000  (since kN*mm/s = kW)
        power = mrr * self.material.specific_cutting_force_kn_mm2 / 1000.0
        power = float(min(power, self.machine.max_spindle_power_kw))
        torque = power * 9550.0 / max(rpm, 1.0)

        eff_dia = self._effective_diameter(tool, stepdown)
        # Coolant when the material requires it (drilling included) and the
        # spindle is cutting. Explicit boolean, no precedence tricks.
        coolant = bool(self.material.needs_coolant)

        return CuttingParameters(
            spindle_rpm=rpm, feed_rate=feed, plunge_feed=plunge, ramp_feed=ramp,
            feed_per_tooth_mm=fz, depth_of_cut_mm=stepdown, stepover_mm=stepover,
            coolant=bool(coolant), effective_diameter_mm=eff_dia,
            engagement_angle_deg=engagement, power_kw=power, torque_nm=torque,
        )

    def _effective_diameter(self, tool: Tool, doc: float) -> float:
        if tool.type is ToolType.BALL_ENDMILL:
            d, r = tool.diameter, tool.tip_radius
            if r <= 0:
                return d
            doc = min(doc, r)
            # chordal contact diameter at axial depth doc: 2*sqrt(r^2-(r-doc)^2)
            return float(2.0 * np.sqrt(max(r * r - (r - doc) ** 2, 1e-9)))
        if tool.type is ToolType.BULLNOSE_ENDMILL and tool.corner_radius > 0:
            r = tool.corner_radius
            doc = min(doc, r)
            d_eff = tool.diameter - 2 * r + 2 * np.sqrt(max(r * r - (r - doc) ** 2, 0.0))
            return float(d_eff)
        return tool.diameter

    def _finish_stepover(self, tool: Tool) -> float:
        # derived from allowed scallop via finish requirement carried on feature;
        # caller may override. Base: small fraction of diameter.
        return 0.15 * tool.diameter if tool.type is not ToolType.BALL_ENDMILL \
            else 0.25 * tool.diameter

    # ------------------------------------------------------ tool ranking --
    def rank_tools(self, feature: MachiningFeature, purpose: str,
                   tools: list[Tool]) -> list[ToolCapability]:
        """Rank candidate tools for a feature+purpose. Incompatible tools get
        excluded with reasons; the caller decides how to report zero candidates."""
        caps: list[ToolCapability] = []
        for tool in tools:
            reasons: list[str] = []
            score = 100.0

            # type compatibility
            if feature.type is FeatureType.HOLE or feature.type is FeatureType.BORE:
                if tool.type is not ToolType.DRILL and purpose == "drilling":
                    reasons.append("drilling requires drill-type tool")
                    score -= 60
            elif purpose == "roughing":
                if tool.type in (ToolType.DRILL, ToolType.BORING_BAR):
                    reasons.append("roughing needs milling tool")
                    score -= 60
            elif purpose == "finishing":
                if tool.type is ToolType.DRILL:
                    reasons.append("drill cannot finish walls")
                    score -= 60

            # diameter vs feature size
            if feature.diameter is not None:
                d = feature.diameter
                if feature.type in (FeatureType.HOLE, FeatureType.BORE):
                    if tool.diameter > d - 1e-6:
                        reasons.append(f"tool dia {tool.diameter:.2f} exceeds hole dia {d:.2f}")
                        score -= 80
                    elif tool.type is not ToolType.DRILL:
                        # milling a hole: prefer largest that fits
                        score += 20 * (tool.diameter / d)
                else:
                    # boss/external: smaller tool fits more detail
                    if tool.diameter > 0.5 * d:
                        reasons.append("tool too large for boss detail")
                        score -= 30
            # pocket: tool must fit into smallest interior feature later (rest machining
            # handles what this tool cannot reach) - large tools score higher for roughing
            if purpose == "roughing" and feature.type in (FeatureType.POCKET, FeatureType.SLOT):
                extent = feature.xy_extent
                if tool.diameter > 0.8 * float(min(extent)):
                    reasons.append("tool too large for pocket interior")
                    score -= 50

            if purpose in ("semi_finishing", "finishing"):
                # smaller tools reach finer detail
                score += 15.0 / max(tool.diameter, 0.5)

            # reach: flute length must cover depth + clearance
            reach_needed = feature.depth + 1.0
            if tool.flute_length < reach_needed:
                reasons.append(f"flute {tool.flute_length:.1f} < needed {reach_needed:.1f}")
                score -= 70

            # machine compatibility: spindle range only (diameter handled above)
            if tool.diameter > 50.0:
                reasons.append("unusually large tool")
                score -= 20

            if score > 40.0:
                caps.append(ToolCapability(tool=tool, score=score, reasons=reasons))
        caps.sort(key=lambda c: -c.score)
        return caps

    def select_tool(self, feature: MachiningFeature, purpose: str,
                    tools: list[Tool]) -> Tool:
        caps = self.rank_tools(feature, purpose, tools)
        if not caps:
            raise CamError(
                code=NO_COMPATIBLE_TOOL,
                message=f"no compatible tool for {feature.type.value} feature "
                        f"({feature.id}) for purpose '{purpose}'",
                stage="tool_selection", feature_id=feature.id,
                context={"candidates": [
                    {"tool": t.tool.id, "reasons": t.reasons} for t in caps]},
            )
        return caps[0].tool

    def check_reach(self, tool: Tool, feature: MachiningFeature) -> None:
        """Explicit reach validation: flute must span depth; holder must clear."""
        needed = feature.depth + 0.5
        if tool.flute_length < needed:
            raise CamError(
                code=TOOL_REACH_INSUFFICIENT,
                message=f"tool {tool.id} flute {tool.flute_length:.2f}mm insufficient "
                        f"for depth {feature.depth:.2f}mm",
                stage="tool_selection", tool_id=tool.id, feature_id=feature.id)
        if tool.holder is not None:
            holder_d = tool.holder.diameter
            if feature.diameter is not None and feature.type in (FeatureType.HOLE, FeatureType.BORE):
                if holder_d > feature.diameter + 1e-6:
                    raise CamError(
                        code=FEATURE_INACCESSIBLE,
                        message=f"holder dia {holder_d:.2f} collides in hole dia "
                                f"{feature.diameter:.2f}",
                        stage="tool_selection", tool_id=tool.id, feature_id=feature.id)
