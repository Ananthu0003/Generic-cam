"""Validation pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import numpy as np

from ..toolpaths.semantic import Toolpath
from .stock import BoundingBox
from .geometry import ToolpathGeometryValidator, GeometryValidationResult, validate_toolpath_geometry
from .stock import StockSimulator, StockSimulationResult, simulate_stock_removal
from .cad_deviation import CadDeviationAnalyzer, CadDeviationResult, analyze_cad_deviation
from .machine import MachineControllerValidator, MachineValidationResult, validate_for_machine
from .final import FinalValidationGate, FinalValidationResult, run_final_validation, ValidationOutcome


class PipelineStage(Enum):
    """Validation pipeline stages."""
    GEOMETRY = "geometry"
    STOCK = "stock"
    CAD_DEVIATION = "cad_deviation"
    MACHINE = "machine"
    FINAL = "final"


@dataclass
class PipelineResult:
    """Complete validation pipeline result."""
    overall_passed: bool
    overall_outcome: str
    stage_results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_passed": self.overall_passed,
            "overall_outcome": self.overall_outcome,
            "stage_results": {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.stage_results.items()},
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ValidationPipeline:
    """Orchestrates the complete CAM validation pipeline."""

    # Tolerances
    geometry_tolerance_mm: float = 0.01
    arc_tolerance_mm: float = 0.005
    cad_tolerance_mm: float = 0.05
    final_position_tolerance_mm: float = 0.05
    final_arc_tolerance_mm: float = 0.01
    final_feed_tolerance_pct: float = 5.0
    final_spindle_tolerance_rpm: float = 100.0
    stock_voxel_mm: float = 0.5

    # Gating behavior
    stop_on_failure: bool = True
    require_final_pass: bool = True

    def run(
        self,
        toolpaths: list,
        gcode: str,
        machine_profile: str = "fanuc",
        stock_bounds: Optional[BoundingBox] = None,
        part_bounds: Optional[BoundingBox] = None,
        cad_faces: Optional[list] = None,
        tool_diameters: Optional[dict[str, float]] = None,
    ) -> PipelineResult:
        """Run complete validation pipeline.

        Pipeline order:
        1. Geometry validation (toolpath internal consistency)
        2. Stock simulation (material removal verification)
        3. CAD deviation analysis (planned vs CAD)
        4. Machine/controller validation (limits, capabilities)
        5. Final validation gate (G-code vs canonical toolpath)
        """
        stage_results: dict[str, Any] = {}
        all_errors: list[str] = []
        all_warnings: list[str] = []

        # Stage 1: Geometry Validation
        geom_result = validate_toolpath_geometry(
            toolpaths,
            tolerance_mm=self.geometry_tolerance_mm,
            arc_tolerance_mm=self.arc_tolerance_mm,
        )
        stage_results["geometry"] = geom_result
        if not geom_result.passed:
            all_errors.extend(geom_result.errors)
        all_warnings.extend(geom_result.warnings)
        if self.stop_on_failure and not geom_result.passed:
            return PipelineResult(
                overall_passed=False,
                overall_outcome="FAIL",
                stage_results=stage_results,
                errors=all_errors,
                warnings=all_warnings,
            )

        # Stage 2: Stock Simulation
        if stock_bounds is not None:
            stock_result = simulate_stock_removal(
                toolpaths,
                stock_bounds,
                part_bounds,
                voxel_size_mm=self.stock_voxel_mm,
            )
            stage_results["stock"] = stock_result
            if not stock_result.passed:
                all_errors.extend(stock_result.errors)
            all_warnings.extend(stock_result.warnings)
            if self.stop_on_failure and not stock_result.passed:
                return PipelineResult(
                    overall_passed=False,
                    overall_outcome="FAIL",
                    stage_results=stage_results,
                    errors=all_errors,
                    warnings=all_warnings,
                )

        # Stage 3: CAD Deviation
        if cad_faces is not None:
            cad_result = analyze_cad_deviation(
                cad_faces,
                toolpaths,
                tool_diameters or {},
                tolerance_mm=self.cad_tolerance_mm,
            )
            stage_results["cad_deviation"] = cad_result
            if not cad_result.passed:
                all_errors.extend(cad_result.errors)
            all_warnings.extend(cad_result.warnings)
            if self.stop_on_failure and not cad_result.passed:
                return PipelineResult(
                    overall_passed=False,
                    overall_outcome="FAIL",
                    stage_results=stage_results,
                    errors=all_errors,
                    warnings=all_warnings,
                )

        # Stage 4: Machine/Controller Validation
        machine_result = validate_for_machine(toolpaths, machine_profile)
        stage_results["machine"] = machine_result
        if not machine_result.passed:
            all_errors.extend(machine_result.errors)
        all_warnings.extend(machine_result.warnings)
        if self.stop_on_failure and not machine_result.passed:
            return PipelineResult(
                overall_passed=False,
                overall_outcome="FAIL",
                stage_results=stage_results,
                errors=all_errors,
                warnings=all_warnings,
            )

        # Stage 5: Final Validation Gate (G-code vs Toolpath)
        # Note: requires G-code to be passed in
        # This stage is typically run after post-processing
        # We'll skip it here if gcode is not available
        # The final gate is typically run separately after post-processing

        overall_passed = len(all_errors) == 0
        return PipelineResult(
            overall_passed=overall_passed,
            overall_outcome="PASS" if overall_passed else "FAIL",
            stage_results=stage_results,
            errors=all_errors,
            warnings=all_warnings,
        )

    def run_final_gate(
        self,
        canonical_toolpaths: list,
        gcode: str,
        machine_profile: str,
    ) -> FinalValidationResult:
        """Run the final validation gate separately (after post-processing)."""
        gate = FinalValidationGate(
            position_tolerance_mm=self.final_position_tolerance_mm,
            arc_center_tolerance_mm=self.final_arc_tolerance_mm,
            feed_tolerance_pct=self.final_feed_tolerance_pct,
            spindle_tolerance_rpm=self.final_spindle_tolerance_rpm,
        )
        return gate.validate(canonical_toolpaths, gcode, get_machine_profile(machine_profile))

    def run_complete(
        self,
        toolpaths: list,
        gcode: str,
        machine_profile: str,
        stock_bounds: Optional[BoundingBox] = None,
        part_bounds: Optional[BoundingBox] = None,
        cad_faces: Optional[list] = None,
        tool_diameters: Optional[dict[str, float]] = None,
    ) -> PipelineResult:
        """Run complete pipeline including final gate."""
        # Pre-post stages
        pre_result = self.run(
            toolpaths=toolpaths,
            gcode=gcode,
            machine_profile=machine_profile,
            stock_bounds=stock_bounds,
            part_bounds=part_bounds,
            cad_faces=cad_faces,
            tool_diameters=tool_diameters,
        )

        if not pre_result.overall_passed and self.stop_on_failure:
            return pre_result

        # Final gate (always run if gcode provided)
        if gcode:
            from .controller_models import get_machine_profile
            final_result = self.run_final_gate(
                canonical_toolpaths=toolpaths,
                gcode=gcode,
                machine_profile=machine_profile,
            )
            pre_result.stage_results["final"] = final_result
            if not final_result.passed and self.require_final_pass:
                pre_result.overall_passed = False
                pre_result.overall_outcome = "FAIL"
                pre_result.errors.extend(final_result.errors)

        return pre_result


def run_validation_pipeline(
    toolpaths: list,
    gcode: str,
    machine_profile: str,
    stock_bounds: Optional[BoundingBox] = None,
    part_bounds: Optional[BoundingBox] = None,
    cad_faces: Optional[list] = None,
    tool_diameters: Optional[dict[str, float]] = None,
    **kwargs,
) -> PipelineResult:
    """Convenience function to run complete validation pipeline."""
    pipeline = ValidationPipeline(**kwargs)
    return pipeline.run_complete(
        toolpaths=toolpaths,
        gcode=gcode,
        machine_profile=machine_profile,
        stock_bounds=stock_bounds,
        part_bounds=part_bounds,
        cad_faces=cad_faces,
        tool_diameters=tool_diameters,
    )