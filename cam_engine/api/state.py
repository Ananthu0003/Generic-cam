"""Session state models for VexCAM API.

This module owns only the session data classes (``SetupConfig``,
``SessionState``).  Session *retrieval* and TTL eviction now live in
``dependencies.py``.  The ``_get_session`` name is re-exported from there
for backwards-compatibility with older callers.
"""

from __future__ import annotations

from typing import Optional

from ..demo_models import (
    create_prismatic_bracket_shape,
    create_prismatic_bracket_mesh,
    get_default_tools,
    get_default_machine,
    DemoCADModel,
)
from ..geometry.step_import import ImportedModel
from ..features import MachiningFeature
from ..operations import PlannedOperation
from ..toolpaths.semantic import Toolpath
from ..context import Tool

from .schemas import StockConfig


class SetupConfig:
    """Lightweight mutable config object representing one fixturing setup."""

    def __init__(
        self,
        id: str,
        name: str,
        work_offset: str = "G54",
        rotation_deg: Optional[list[float]] = None,
        stock_cfg: Optional[StockConfig] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.work_offset = work_offset
        self.rotation_deg = rotation_deg or [0.0, 0.0, 0.0]
        self.stock_cfg = stock_cfg or StockConfig(work_offset=work_offset)
        self.feature_ids: list[str] = []
        self.operations_count: int = 0


class SessionState:
    """Per-session mutable planning state."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.model_name: str = "Demo Prismatic Bracket"
        self.raw_mesh: Optional[DemoCADModel] = None
        self.imported_model: Optional[ImportedModel] = None
        self.features: list[MachiningFeature] = []
        self.tools: list[Tool] = get_default_tools()
        self.machine = get_default_machine()
        self.planned_ops: list[PlannedOperation] = []
        self.toolpaths: list[Toolpath] = []
        self.stock_config = StockConfig()
        self.setups: list[SetupConfig] = [
            SetupConfig(
                id="setup_001",
                name="Setup 1 - Top Milling",
                work_offset="G54",
                rotation_deg=[0.0, 0.0, 0.0],
            ),
        ]
        self.active_setup_id: str = "setup_001"
        self.load_demo()

    def load_demo(self) -> None:
        """Populate state with the built-in prismatic bracket demo model."""
        from .services.geometry_service import shape_to_demo_cad_model

        try:
            self.imported_model = create_prismatic_bracket_shape()
            self.model_name = "Prismatic CAM Test Bracket"
            self.raw_mesh = shape_to_demo_cad_model(
                self.model_name,
                self.imported_model.shape,
                "Standard 100x60x25mm aerospace bracket with top facing, "
                "40x30mm pocket, Ø12mm bore, and 8mm stepdown.",
            )
        except Exception:
            self.raw_mesh = create_prismatic_bracket_mesh()
            self.model_name = self.raw_mesh.name
            self.imported_model = None

        self.features = []
        self.planned_ops = []
        self.toolpaths = []
        self.setups = [
            SetupConfig(
                id="setup_001",
                name="Setup 1 - Top Milling",
                work_offset="G54",
                rotation_deg=[0.0, 0.0, 0.0],
            ),
        ]
        self.active_setup_id = "setup_001"


__all__ = ["SetupConfig", "SessionState"]
