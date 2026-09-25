"""StrategyEngine facade and operation dispatcher."""

from __future__ import annotations

from ...errors import CamError, UNSUPPORTED_FEATURE
from ...features import FeatureType
from ...operations import OpPurpose, PlannedOperation
from ..semantic import Toolpath
from .base import BaseStrategy
from .boundaries import BoundariesMixin
from .finishing import FinishingStrategyMixin
from .hole import HoleStrategyMixin
from .primitives import PrimitivesMixin
from .roughing import RoughingStrategyMixin


class StrategyEngine(
    RoughingStrategyMixin,
    FinishingStrategyMixin,
    HoleStrategyMixin,
    BoundariesMixin,
    PrimitivesMixin,
    BaseStrategy,
):
    """Generates semantic toolpaths for planned operations."""

    def generate(self, op: PlannedOperation) -> Toolpath:
        if op.purpose is OpPurpose.FACING:
            return self._facing(op)
        if op.purpose is OpPurpose.ROUGHING:
            if op.notes.get("method") == "adaptive":
                return self._adaptive_roughing(op)
            if op.notes.get("method") == "helical_interpolation" or op.feature.type in (FeatureType.BORE, FeatureType.THROUGH_BORE, FeatureType.BLIND_BORE):
                return self._helical_interpolation(op)
            return self._roughing(op)
        if op.purpose is OpPurpose.REST_MACHINING:
            return self._rest_machining(op)
        if op.purpose is OpPurpose.SEMI_FINISHING:
            return self._semi_finishing(op)
        if op.purpose is OpPurpose.FINISHING:
            return self._finishing(op)
        if op.purpose is OpPurpose.DRILLING:
            return self._drilling(op)
        if op.purpose is OpPurpose.BORING:
            return self._boring(op)
        if op.purpose is OpPurpose.SPOT_DRILLING:
            return self._spot_drilling(op)
        if op.purpose is OpPurpose.REAMING:
            return self._reaming(op)
        if op.purpose is OpPurpose.TAPPING:
            return self._tapping(op)
        if op.purpose is OpPurpose.CHAMFERING:
            return self._chamfering(op)
        if op.purpose is OpPurpose.COUNTERSINKING:
            return self._countersinking(op)
        if op.purpose is OpPurpose.THREAD_MILLING:
            return self._thread_milling(op)
        if op.purpose is OpPurpose.GROOVING:
            return self._grooving(op)
        raise CamError(UNSUPPORTED_FEATURE,
                       f"no strategy for purpose {op.purpose}", stage="toolpath",
                       operation_id=op.id)
