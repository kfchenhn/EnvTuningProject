from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from .data_models import Trajectory


class SelfPlayReplayBuffer:
    """按任务签名分桶的经验回放池。

    设计目标：
    1) 为失败轨迹提供“历史成功锚点”；
    2) 限制每个任务的缓存长度，避免内存无限增长。
    """

    def __init__(self, maxlen_per_task: int = 256):
        self.maxlen_per_task = maxlen_per_task
        self._storage: Dict[str, Deque[Trajectory]] = defaultdict(lambda: deque(maxlen=self.maxlen_per_task))

    def add(self, trajectory: Trajectory) -> None:
        """写入一条轨迹。"""
        self._storage[trajectory.task_signature].append(trajectory)

    def latest_success(self, task_signature: str) -> Optional[Trajectory]:
        """返回最近一条 reward>0 的成功轨迹。"""
        bucket: List[Trajectory] = list(self._storage.get(task_signature, []))
        for traj in reversed(bucket):
            if traj.reward > 0:
                return traj
        return None

    def size(self, task_signature: str) -> int:
        """获取某任务签名下缓存的轨迹数量。"""
        return len(self._storage.get(task_signature, []))
