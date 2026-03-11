from typing import Iterable, Optional

from .data_models import Trajectory
from .replay_buffer import SelfPlayReplayBuffer


class AnchorSelector:
    """三级瀑布流锚点选择器。

    优先级：
    1) 同批次同任务的成功轨迹（最小分布偏移）；
    2) 回放池历史成功轨迹（防遗忘）；
    3) 课程诱导锚点（冷启动兜底）。
    """

    def __init__(self, replay_buffer: SelfPlayReplayBuffer):
        self.replay_buffer = replay_buffer

    def select(
        self,
        failed_trajectory: Trajectory,
        batch_candidates: Iterable[Trajectory],
        curriculum_anchor: Optional[Trajectory] = None,
    ) -> Optional[Trajectory]:
        # Priority-1：同批次锚点
        for candidate in batch_candidates:
            if candidate is None:
                continue
            if (
                candidate.task_signature == failed_trajectory.task_signature
                and candidate.turn_index == failed_trajectory.turn_index
                and candidate.reward > failed_trajectory.reward
            ):
                return candidate

        # Priority-2：历史锚点
        replay_anchor = self.replay_buffer.latest_success(failed_trajectory.task_signature)
        if replay_anchor is not None:
            return replay_anchor

        # Priority-3：课程锚点
        return curriculum_anchor
