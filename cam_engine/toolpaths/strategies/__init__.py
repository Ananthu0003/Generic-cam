"""Toolpath generation strategies package (engagement-aware, geometry-driven)."""

from .base import StrategyContext
from .boundaries import MachiningRegion
from .engine import StrategyEngine

__all__ = ["StrategyEngine", "StrategyContext", "MachiningRegion"]
