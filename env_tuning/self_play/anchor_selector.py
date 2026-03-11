from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrajectoryRecord:
    """A compact trajectory descriptor used by self-play anchor search."""

    task_signature: str
    trajectory_text: str
    progress_reward: float
    success: bool
    source: str = "peer"
    metadata: Dict[str, str] = field(default_factory=dict)


class AnchorSelector:
    """Three-level fallback anchor selector.

    Priority:
    1) peer anchor from current mini-batch
    2) historical anchor from replay memory
    3) curriculum anchor injected by env hints
    """

    def __init__(self, max_history_per_task: int = 64):
        self.max_history_per_task = max_history_per_task
        self._history: Dict[str, List[TrajectoryRecord]] = {}

    def add_history(self, record: TrajectoryRecord) -> None:
        if not record.success:
            return
        bucket = self._history.setdefault(record.task_signature, [])
        bucket.append(record)
        if len(bucket) > self.max_history_per_task:
            del bucket[0 : len(bucket) - self.max_history_per_task]

    def choose_anchor(
        self,
        task_signature: str,
        peer_records: List[TrajectoryRecord],
        curriculum_anchor: Optional[TrajectoryRecord] = None,
    ) -> Tuple[Optional[TrajectoryRecord], str]:
        """Select best available anchor and return (anchor, source)."""
        peer_success = [r for r in peer_records if r.success and r.task_signature == task_signature]
        if peer_success:
            best = max(peer_success, key=lambda r: r.progress_reward)
            return best, "peer"

        history = self._history.get(task_signature, [])
        if history:
            best = max(history, key=lambda r: r.progress_reward)
            return best, "history"

        if curriculum_anchor is not None:
            return curriculum_anchor, "curriculum"

        return None, "none"
