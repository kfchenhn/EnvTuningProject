from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrajectoryRecord:
    """自博弈锚点选择使用的轻量轨迹描述对象。"""

    task_signature: str
    trajectory_text: str
    progress_reward: float
    success: bool
    source: str = "peer"
    metadata: Dict[str, str] = field(default_factory=dict)


class AnchorSelector:
    """三级降级锚点选择器。

    Priority:
    1) peer anchor from current mini-batch
    2) historical anchor from replay memory
    3) curriculum anchor injected by env hints
    """

    def __init__(self, max_history_per_task: int = 64):
        # 每个 task_signature 的历史成功轨迹最多保留多少条。
        self.max_history_per_task = max_history_per_task
        # 回放池：按任务签名分桶保存成功轨迹。
        self._history: Dict[str, List[TrajectoryRecord]] = {}

    def add_history(self, record: TrajectoryRecord) -> None:
        # 仅成功轨迹才可作为锚点候选。
        if not record.success:
            return
        bucket = self._history.setdefault(record.task_signature, [])
        bucket.append(record)
        # 控制内存占用，采用 FIFO 丢弃最旧样本。
        if len(bucket) > self.max_history_per_task:
            del bucket[0 : len(bucket) - self.max_history_per_task]

    def choose_anchor(
        self,
        task_signature: str,
        peer_records: List[TrajectoryRecord],
        curriculum_anchor: Optional[TrajectoryRecord] = None,
    ) -> Tuple[Optional[TrajectoryRecord], str]:
        """按 peer > history > curriculum 的顺序选择锚点。"""
        # 1) 同 batch 同任务成功样本优先，避免跨批次分布漂移。
        peer_success = [r for r in peer_records if r.success and r.task_signature == task_signature]
        if peer_success:
            best = max(peer_success, key=lambda r: r.progress_reward)
            return best, "peer"

        # 2) 若批内无成功样本，则回放池检索历史最佳成功轨迹。
        history = self._history.get(task_signature, [])
        if history:
            best = max(history, key=lambda r: r.progress_reward)
            return best, "history"

        # 3) 最后降级到课程诱导锚点（例如通道B挽救得到的成功样本）。
        if curriculum_anchor is not None:
            return curriculum_anchor, "curriculum"

        # 没有任何可用锚点。
        return None, "none"
