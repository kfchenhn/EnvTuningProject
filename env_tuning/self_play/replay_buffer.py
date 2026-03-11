from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from .data_models import Trajectory


class SelfPlayReplayBuffer:
    """Task-keyed replay buffer for retrieving historical successful anchors."""

    def __init__(self, maxlen_per_task: int = 256):
        self.maxlen_per_task = maxlen_per_task
        self._storage: Dict[str, Deque[Trajectory]] = defaultdict(lambda: deque(maxlen=self.maxlen_per_task))

    def add(self, trajectory: Trajectory) -> None:
        self._storage[trajectory.task_signature].append(trajectory)

    def latest_success(self, task_signature: str) -> Optional[Trajectory]:
        bucket: List[Trajectory] = list(self._storage.get(task_signature, []))
        for traj in reversed(bucket):
            if traj.reward > 0:
                return traj
        return None

    def size(self, task_signature: str) -> int:
        return len(self._storage.get(task_signature, []))
