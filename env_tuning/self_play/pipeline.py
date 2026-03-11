from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .anchor_selector import AnchorSelector, TrajectoryCandidate
from .ast_diagnostics import LogicDivergence, find_first_logic_divergence


@dataclass
class CounterfactualSample:
    task_signature: str
    divergence: LogicDivergence
    preferred_trajectory: List[Any]
    rejected_trajectory: List[Any]


class SelfPlayCoordinator:
    """Coordinator for anchor selection and AST-based hindsight diagnostics."""

    def __init__(self, selector: Optional[AnchorSelector] = None):
        self.selector = selector or AnchorSelector()

    def build_counterfactual_sample(
        self,
        task_signature: str,
        failed_candidate: TrajectoryCandidate,
        batch_candidates: List[TrajectoryCandidate],
        curriculum_anchor: Optional[TrajectoryCandidate] = None,
    ) -> Optional[CounterfactualSample]:
        anchor = self.selector.select_anchor(task_signature, batch_candidates, curriculum_anchor)
        if anchor is None:
            return None

        divergence = find_first_logic_divergence(anchor.trajectory, failed_candidate.trajectory)
        if divergence is None:
            return None

        self.selector.memory.add(anchor)

        return CounterfactualSample(
            task_signature=task_signature,
            divergence=divergence,
            preferred_trajectory=anchor.trajectory,
            rejected_trajectory=failed_candidate.trajectory,
        )

    def build_grpo_advantage_hints(self, sample: CounterfactualSample) -> Dict[str, Any]:
        """Build optimization hints consumable by a GRPO trainer."""
        divergence = sample.divergence
        return {
            "task_signature": sample.task_signature,
            "positive_node": {
                "turn_index": divergence.turn_index,
                "action_index": divergence.action_index,
                "action": divergence.success_action,
            },
            "negative_node": {
                "turn_index": divergence.turn_index,
                "action_index": divergence.action_index,
                "action": divergence.failed_action,
            },
            "divergence_type": divergence.divergence_type,
            "reason": divergence.reason,
        }
