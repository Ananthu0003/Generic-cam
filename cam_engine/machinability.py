"""Machinability analysis + capability-driven tool selection.

Cutting parameters derive from material properties, tool geometry, machine
limits, and operation requirements - never fixed global constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .context import MachineConfig, Material, Setup, Tool, ToolType
from .errors import (CamError, FEATURE_INACCESSIBLE, INVALID_CUTTING_PARAMETERS,
                     NO_COMPATIBLE_TOOL, TOOL_REACH_INSUFFICIENT)
from .features import Accessibility, FeatureType, MachiningFeature


@dataclass
class CuttingParameters:
    """Validated cutting parameters for one operation with complete derivation traceability."""

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

    # Traceability & auditability fields
    cutting_speed_m_min: float = 0.0
    target_cutting_speed_m_min: float = 0.0
    effective_cutting_speed_m_min: float = 0.0
    target_rpm: float = 0.0
    calculated_rpm: float = 0.0
    rpm_clamped: bool = False
    target_feed_rate: float = 0.0
    feed_clamped: bool = False
    material_name: str = ""
    material_hardness_hb: float = 0.0
    tool_id: str = ""
    tool_type: str = ""
    tool_material: str = ""
    tool_diameter_mm: float = 0.0
    flute_count: int = 0
    machine_max_rpm: float = 0.0
    machine_min_rpm: float = 0.0
    machine_max_feed: float = 0.0
    machine_max_power_kw: float = 0.0


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

    def calculate_drilling_peck_strategy(self, tool: Tool, feature: Optional[MachiningFeature] = None,
                                         coolant_active: bool = True) -> dict[str, Any]:
        """Compute comprehensive, physics- and material-informed drilling peck strategy.

        Considers:
        1. Tool diameter (D) with strict micro-drilling physics (D < 1.0mm)
        2. Workpiece material machinability, hardness, chip ductility, and work-hardening tendency
        3. Hole aspect ratio (L/D) and chip jamming risk at depth
        4. Tool geometry (point angle, drill type, parabolic vs standard twist, TSC)
        5. Coolant mode (through-spindle vs flood vs dry)
        6. Machine and controller cycle semantics (G83 deep-hole full retract vs G73 chip-break)
        """
        dia = max(float(tool.diameter), 0.05)
        if feature is not None:
            hole_depth = float(feature.depth if (feature.depth and feature.depth > 0) else (
                feature.top_z - feature.floor_z if (feature.top_z is not None and feature.floor_z is not None) else dia * 3.0
            ))
        else:
            hole_depth = dia * 3.0
        ld_ratio = max(hole_depth / dia, 0.1)

        # 1. Base diameter multiplier k_dia
        if dia < 0.8:
            k_dia = 0.35   # Micro-drilling: conservative peck to prevent flute packing & breakage
            max_peck_limit_ratio = 0.50
        elif dia < 1.5:
            k_dia = 0.60   # Miniature drilling
            max_peck_limit_ratio = 0.80
        elif dia < 3.0:
            k_dia = 1.00   # Small drilling
            max_peck_limit_ratio = 1.50
        elif dia <= 12.0:
            k_dia = 1.80   # Standard drilling
            max_peck_limit_ratio = 3.00
        else:
            k_dia = 1.20   # Large drilling (power/thrust constrained)
            max_peck_limit_ratio = 2.00

        # 2. Workpiece material factor k_mat
        kc = self.material.specific_cutting_force_kn_mm2
        hb = self.material.hardness_hb
        mat_id = (self.material.id or self.material.name or "").lower()
        chip_tendency = getattr(self.material, "chip_breaking_tendency", "medium").lower()

        if "stainless" in mat_id or "304" in mat_id or "316" in mat_id or chip_tendency == "stringy":
            k_mat = 0.45
        elif "titanium" in mat_id or "inconel" in mat_id or "ti-6" in mat_id:
            k_mat = 0.35
        elif "becu" in mat_id or "beryllium" in mat_id or chip_tendency == "work_hardening":
            k_mat = 0.65
        elif kc <= 0.90:  # Aluminum alloys, brass, soft non-ferrous
            k_mat = 1.00
        elif kc <= 1.60:  # Mild carbon steels (1018)
            k_mat = 0.75
        elif kc <= 2.30:  # Alloy steels (4140), tool steels
            k_mat = 0.60
        elif kc <= 3.20:  # Tough high-strength steels
            k_mat = 0.45
        else:
            k_mat = 0.30

        if hb > 220:
            k_mat *= max(0.40, (220.0 / hb) ** 0.4)

        # 3. Aspect ratio / depth factor k_depth
        if ld_ratio <= 1.5:
            k_depth = 1.00
        elif ld_ratio <= 3.0:
            k_depth = 0.85
        elif ld_ratio <= 5.0:
            k_depth = 0.70
        elif ld_ratio <= 8.0:
            k_depth = 0.50
        else:
            k_depth = 0.35

        # 4. Tool & coolant technology factor k_tech
        has_tsc = getattr(tool, "has_through_coolant", False)
        drill_type = getattr(tool, "drill_type", "twist").lower()

        k_tech = 1.10 if tool.material == "carbide" else 0.90
        if has_tsc and coolant_active:
            k_tech *= 1.35
            max_peck_limit_ratio *= 1.30
        elif drill_type == "parabolic":
            k_tech *= 1.20
        
        if not coolant_active and self.material.needs_coolant:
            k_tech *= 0.50

        # Tool manufacturer explicit limit override
        if getattr(tool, "max_peck_ratio", None) is not None and tool.max_peck_ratio > 0:
            max_peck_limit_ratio = min(max_peck_limit_ratio, float(tool.max_peck_ratio))

        # Nominal first peck calculation
        nominal_peck = dia * k_dia * k_mat * k_depth * k_tech
        min_peck_floor = max(0.10 * dia, 0.03)  # Absolute minimum machine resolution floor
        max_peck_allowed = min(max_peck_limit_ratio * dia, max(hole_depth, 0.1))

        first_peck = float(np.clip(nominal_peck, min_peck_floor, max_peck_allowed))
        first_peck = round(first_peck, 3)

        # 5. Controller cycle selection & progressive reduction
        is_micro = dia < 1.0
        is_tough_or_stringy = (
            k_mat <= 0.65 or
            chip_tendency in ("stringy", "work_hardening") or
            "stainless" in mat_id or "titanium" in mat_id or "inconel" in mat_id or "becu" in mat_id
        )
        if ld_ratio <= 1.5 and not is_micro and not is_tough_or_stringy:
            cycle_type = "G73"
            peck_reduction = 0.0
            subsequent_peck = first_peck
            min_peck = first_peck
        else:
            cycle_type = "G83"
            if ld_ratio > 4.0 and not is_micro:
                peck_reduction = round(max(0.12 * first_peck, 0.02), 3)
                min_peck = round(max(0.35 * first_peck, min_peck_floor), 3)
                subsequent_peck = round(max(min_peck, first_peck - peck_reduction), 3)
            elif ld_ratio > 5.0 and is_micro:
                peck_reduction = round(max(0.10 * first_peck, 0.01), 3)
                min_peck = round(max(0.40 * first_peck, min_peck_floor), 3)
                subsequent_peck = round(max(min_peck, first_peck - peck_reduction), 3)
            else:
                peck_reduction = 0.0
                subsequent_peck = first_peck
                min_peck = first_peck

        return {
            "first_peck": first_peck,
            "subsequent_peck": subsequent_peck,
            "peck_reduction": peck_reduction,
            "min_peck": min_peck,
            "cycle_type": cycle_type,
            "max_allowable_peck": round(max_peck_allowed, 3),
            "peck_ratio": round(first_peck / dia, 3),
            "ld_ratio": round(ld_ratio, 2),
            "material_id": self.material.id,
            "hole_depth": round(hole_depth, 3),
        }

    def calculate_drilling_peck(self, tool: Tool, feature: Optional[MachiningFeature] = None) -> float:
        """Calculate primary drilling peck depth (Q) using the rule-based strategy."""
        strat = self.calculate_drilling_peck_strategy(tool, feature, coolant_active=self.material.needs_coolant)
        return float(strat["first_peck"])

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
        target_rpm = self.surface_speed_to_rpm(v, tool.diameter)
        rpm_clamped = target_rpm < self.machine.spindle_min_rpm or target_rpm > self.machine.spindle_max_rpm
        rpm = float(np.clip(target_rpm, self.machine.spindle_min_rpm, self.machine.spindle_max_rpm))
        effective_v = (np.pi * tool.diameter * rpm) / 1000.0

        # --- feed per tooth from tool geometry + material hardness ---
        # base chip load scales with diameter and decreases with hardness (industrial carbide basis)
        fz_base = 0.018 * tool.diameter * (100.0 / max(self.material.hardness_hb, 10.0)) ** 0.35
        if op_purpose == "finishing":
            fz_base *= 0.60
        elif op_purpose == "semi_finishing":
            fz_base *= 0.80
        elif op_purpose == "roughing":
            fz_base *= 1.15
        elif op_purpose in ("drilling", "spot_drilling"):
            fz_base = 0.012 * tool.diameter * (100.0 / max(self.material.hardness_hb, 10.0)) ** 0.35
            if tool.diameter < 1.0:
                fz_base *= float(np.clip(np.sqrt(tool.diameter / 1.0), 0.5, 1.0))
        fz = float(np.clip(fz_base, 0.002, 0.40))

        flutes = max(1, tool.flutes)
        target_feed = rpm * flutes * fz

        # --- engagement limits from material data ---
        max_hm = self.material.max_hm_ratio * tool.diameter
        max_ae = self.material.max_ae_ratio * tool.diameter

        stepdown = max_hm
        if op_purpose == "roughing":
            stepdown = max_hm
        elif op_purpose in ("semi_finishing", "finishing"):
            stepdown = min(max_hm, max(0.4 * tool.diameter, 0.5))
        elif op_purpose in ("drilling", "spot_drilling"):
            stepdown = self.calculate_drilling_peck(tool, feature)

        stepover = max_ae
        if op_purpose == "roughing":
            stepover = max_ae
        elif op_purpose == "semi_finishing":
            stepover = 0.45 * tool.diameter
        elif op_purpose == "finishing":
            stepover = self._finish_stepover(tool)
        elif op_purpose in ("drilling", "spot_drilling"):
            stepover = tool.diameter

        # clamp feed to machine axis capability
        max_feed = min((ax.max_feed for ax in self.machine.axes.values()), default=5000.0)
        feed_clamped = target_feed > max_feed
        feed = float(min(target_feed, max_feed))
        if op_purpose in ("drilling", "spot_drilling"):
            plunge = feed
            ramp = feed
        else:
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
            spindle_rpm=rpm,
            feed_rate=feed,
            plunge_feed=plunge,
            ramp_feed=ramp,
            feed_per_tooth_mm=fz,
            depth_of_cut_mm=stepdown,
            stepover_mm=stepover,
            coolant=bool(coolant),
            effective_diameter_mm=eff_dia,
            engagement_angle_deg=engagement,
            power_kw=power,
            torque_nm=torque,
            cutting_speed_m_min=effective_v,
            target_cutting_speed_m_min=v,
            effective_cutting_speed_m_min=effective_v,
            target_rpm=target_rpm,
            calculated_rpm=target_rpm,
            rpm_clamped=rpm_clamped,
            target_feed_rate=target_feed,
            feed_clamped=feed_clamped,
            material_name=self.material.name,
            material_hardness_hb=self.material.hardness_hb,
            tool_id=tool.id,
            tool_type=tool.type.value if hasattr(tool.type, "value") else str(tool.type),
            tool_material=tool.material,
            tool_diameter_mm=tool.diameter,
            flute_count=flutes,
            machine_max_rpm=self.machine.spindle_max_rpm,
            machine_min_rpm=self.machine.spindle_min_rpm,
            machine_max_feed=max_feed,
            machine_max_power_kw=self.machine.max_spindle_power_kw,
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
