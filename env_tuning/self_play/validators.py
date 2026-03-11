from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from bfcl_env.multi_turn_checker import response_checker, state_checker


@dataclass
class ValidationScore:
    state_score: int
    response_score: int

    @property
    def binary_reward(self) -> int:
        return self.state_score * self.response_score


class DualOutcomeValidator:
    """Outcome-based two-dimensional validator (state + response)."""

    def validate(
        self,
        model_instances: Dict[str, Any],
        gt_instances: Dict[str, Any],
        model_results: List[Any],
        gt_results: List[Any],
        turn_index: int,
    ) -> ValidationScore:
        state_ok = self._validate_state(model_instances, gt_instances)
        response_ok = self._validate_response(model_results, gt_results, turn_index)
        return ValidationScore(state_score=int(state_ok), response_score=int(response_ok))

    def progress_reward(self, turn_scores: List[ValidationScore]) -> float:
        if not turn_scores:
            return 0.0
        return sum(score.binary_reward for score in turn_scores) / len(turn_scores)

    def _validate_state(self, model_instances: Dict[str, Any], gt_instances: Dict[str, Any]) -> bool:
        try:
            return bool(state_checker(model_instances, gt_instances).get("valid", False))
        except Exception:
            return False

    def _validate_response(self, model_results: List[Any], gt_results: List[Any], turn_index: int) -> bool:
        try:
            return bool(response_checker(model_results, gt_results, turn_index).get("valid", False))
        except Exception:
            return False
