from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from bfcl_env.multi_turn_checker import response_checker, state_checker


@dataclass
class ValidationResult:
    state_ok: bool
    response_ok: bool

    @property
    def reward(self) -> float:
        return float(self.state_ok and self.response_ok)


class DualOutcomeValidator:
    """Two-dimensional outcome validator used by self-play triggering."""

    def validate(
        self,
        model_instances: Dict[str, Any],
        gt_instances: Dict[str, Any],
        model_results: List[Any],
        gt_results: List[Any],
        turn_index: int,
    ) -> ValidationResult:
        try:
            state_ok = bool(state_checker(model_instances, gt_instances).get("valid", False))
        except Exception:
            state_ok = False

        try:
            response_ok = bool(response_checker(model_results, gt_results, turn_index).get("valid", False))
        except Exception:
            response_ok = False

        return ValidationResult(state_ok=state_ok, response_ok=response_ok)
