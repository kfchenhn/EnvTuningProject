from dataclasses import dataclass
from typing import Optional


@dataclass
class DualChannelState:
    """双通道调度状态。

    - stage: 当前课程阶段（1~4）
    - channel_b_hint_prob: 通道B（在线提示）注入概率
    """

    stage: int = 1
    channel_b_hint_prob: float = 1.0


class DualChannelScheduler:
    """通道B退火调度器。

    规则：
    - 阶段1/2：保持强提示（概率1.0）
    - 阶段3：指数退火到 min_prob
    - 阶段4：完全关闭（概率0）
    """

    def __init__(self, decay_rate: float = 0.95, min_prob: float = 0.05):
        self.decay_rate = decay_rate
        self.min_prob = min_prob

    def step(self, state: DualChannelState, validation_score: Optional[float] = None) -> DualChannelState:
        # validation_score 预留给后续“按验证集表现自适应退火”的扩展。
        if state.stage < 3:
            state.channel_b_hint_prob = 1.0
            return state

        if state.stage == 3:
            state.channel_b_hint_prob = max(self.min_prob, state.channel_b_hint_prob * self.decay_rate)
            return state

        state.channel_b_hint_prob = 0.0
        return state
