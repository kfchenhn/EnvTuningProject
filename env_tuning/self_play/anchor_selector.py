from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryCandidate:
    task_signature: str
    reward: float
    trajectory: List[Any]
    metadata: Dict[str, Any]


class AnchorMemory:
    """Global replay pool for successful historical anchors."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._buffer: Dict[str, List[TrajectoryCandidate]] = {}

    def add(self, candidate: TrajectoryCandidate) -> None:
        if candidate.reward <= 0:
            return
        bucket = self._buffer.setdefault(candidate.task_signature, [])
        bucket.append(candidate)
        bucket.sort(key=lambda item: item.reward, reverse=True)
        self._buffer[candidate.task_signature] = bucket[: self.capacity]

    def get_best(self, task_signature: str) -> Optional[TrajectoryCandidate]:
        bucket = self._buffer.get(task_signature, [])
        return bucket[0] if bucket else None


class AnchorSelector:
    """Three-level waterfall anchor selector: peer -> memory -> curriculum."""

    def __init__(self, memory: Optional[AnchorMemory] = None):
        self.memory = memory or AnchorMemory()

    def select_anchor(
        self,
        task_signature: str,
        batch_candidates: List[TrajectoryCandidate],
        curriculum_anchor: Optional[TrajectoryCandidate] = None,
    ) -> Optional[TrajectoryCandidate]:
        peer_anchor = self._select_peer_anchor(batch_candidates)
        if peer_anchor is not None:
            return peer_anchor

        history_anchor = self.memory.get_best(task_signature)
        if history_anchor is not None:
            return history_anchor

        return curriculum_anchor

    def _select_peer_anchor(self, batch_candidates: List[TrajectoryCandidate]) -> Optional[TrajectoryCandidate]:
        successful = [candidate for candidate in batch_candidates if candidate.reward > 0]
        if not successful:
            return None
        return max(successful, key=lambda item: item.reward)
