from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallNode:
    """Normalized representation of a single tool call."""

    tool_name: str
    arguments: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Trajectory:
    """Trajectory container used by self-play diagnostics."""

    task_signature: str
    turn_index: int
    calls: List[ToolCallNode]
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergencePoint:
    """First meaningful mismatch between anchor/failure trajectory."""

    index: int
    divergence_type: str
    anchor_call: Optional[ToolCallNode]
    failed_call: Optional[ToolCallNode]
    reason: str


@dataclass
class CounterfactualSample:
    """Structured pair used for preference optimization."""

    task_signature: str
    turn_index: int
    anchor: Trajectory
    failed: Trajectory
    divergence: DivergencePoint
    diagnosis: str
