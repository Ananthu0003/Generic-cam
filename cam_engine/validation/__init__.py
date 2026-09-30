"""CAM Validation Pipeline.

Comprehensive validation gates ensuring geometric fidelity from CAD through G-code.
"""

from .geometry import (
    ToolpathGeometryValidator,
    GeometryValidationResult,
    validate_toolpath_geometry,
)
from .stock import (
    StockSimulator,
    StockSimulationResult,
    simulate_stock_removal,
)
from .cad_deviation import (
    CadDeviationAnalyzer,
    CadDeviationResult,
    analyze_cad_deviation,
)
from .machine import (
    MachineControllerValidator,
    MachineValidationResult,
    validate_for_machine,
)
from .final import (
    FinalValidationGate,
    FinalValidationResult,
    run_final_validation,
    ValidationOutcome,
)
from .pipeline import (
    ValidationPipeline,
    PipelineStage,
    PipelineResult,
    run_validation_pipeline,
)

__all__ = [
    "ToolpathGeometryValidator",
    "GeometryValidationResult",
    "validate_toolpath_geometry",
    "StockSimulator",
    "StockSimulationResult",
    "simulate_stock_removal",
    "CadDeviationAnalyzer",
    "CadDeviationResult",
    "analyze_cad_deviation",
    "MachineControllerValidator",
    "MachineValidationResult",
    "validate_for_machine",
    "FinalValidationGate",
    "FinalValidationResult",
    "run_final_validation",
    "ValidationOutcome",
    "ValidationPipeline",
    "PipelineStage",
    "PipelineResult",
    "run_validation_pipeline",
]