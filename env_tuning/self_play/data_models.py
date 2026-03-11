from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallNode:
    """标准化后的单个工具调用节点。

    说明：这里用“近似 AST 节点”的抽象形式保存工具名、参数和依赖，
    便于后续进行结构级比较，而不是脆弱的纯文本比较。
    """

    tool_name: str
    arguments: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Trajectory:
    """自博弈轨迹容器。"""

    task_signature: str
    turn_index: int
    calls: List[ToolCallNode]
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergencePoint:
    """第一逻辑分歧点。

    - strategy: 工具选择不同
    - parameter: 工具相同但核心参数不同
    - length: 调用长度不同（某一方提前结束）
    """

    index: int
    divergence_type: str
    anchor_call: Optional[ToolCallNode]
    failed_call: Optional[ToolCallNode]
    reason: str


@dataclass
class CounterfactualSample:
    """反事实偏好样本（成功锚点 vs 失败轨迹）。"""

    task_signature: str
    turn_index: int
    anchor: Trajectory
    failed: Trajectory
    divergence: DivergencePoint
    diagnosis: str
