from typing import Iterable, Optional

from .data_models import Trajectory
from .replay_buffer import SelfPlayReplayBuffer


class AnchorSelector:
    """Three-level waterfall anchor selector.

    Priority:
      1) Peer anchor in current batch
      2) Historical anchor from replay buffer
      3) Curriculum-induced anchor (provided by caller)
    """

    def __init__(self, replay_buffer: SelfPlayReplayBuffer):
        self.replay_buffer = replay_buffer

    def select(
        self,
        failed_trajectory: Trajectory,
        batch_candidates: Iterable[Trajectory],
        curriculum_anchor: Optional[Trajectory] = None,
    ) -> Optional[Trajectory]:
        # Priority 1: peer anchors in same batch.
        for candidate in batch_candidates:
            if (
                candidate.task_signature == failed_trajectory.task_signature
                and candidate.turn_index == failed_trajectory.turn_index
                and candidate.reward > failed_trajectory.reward
            ):
                return candidate

        # Priority 2: historical anchors from replay buffer.
        replay_anchor = self.replay_buffer.latest_success(failed_trajectory.task_signature)
        if replay_anchor is not None:
            return replay_anchor

        # Priority 3: curriculum-induced anchor fallback.
        return curriculum_anchor
