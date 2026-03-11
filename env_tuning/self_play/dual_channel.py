from dataclasses import dataclass
from typing import Optional


@dataclass
class DualChannelState:
    """Keeps runtime schedule for online hinting (channel-B) and offline internalization (channel-A)."""

    stage: int = 1
    channel_b_hint_prob: float = 1.0


class DualChannelScheduler:
    """Simple scheduler that exponentially decays channel-B intervention in stage 3."""

    def __init__(self, decay_rate: float = 0.95, min_prob: float = 0.05):
        self.decay_rate = decay_rate
        self.min_prob = min_prob

    def step(self, state: DualChannelState, validation_score: Optional[float] = None) -> DualChannelState:
        if state.stage < 3:
            state.channel_b_hint_prob = 1.0
            return state

        if state.stage == 3:
            state.channel_b_hint_prob = max(self.min_prob, state.channel_b_hint_prob * self.decay_rate)
            return state

        # stage 4 fully disables channel-B hints.
        state.channel_b_hint_prob = 0.0
        return state
