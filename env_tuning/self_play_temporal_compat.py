from __future__ import annotations

#######新增（开始）#######
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from verl.workers.rollout.schemas import AsyncRolloutRequest

logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """抽象语法树中的完整节点：保留函数名、参数值与依赖信息。"""

    tool_name: str
    tool_category: str
    arguments: Any
    normalized_arguments: str
    tool_call_id: str
    dependent_tool_call_ids: tuple[str, ...]


@dataclass
class DiagnosisResult:
    """失败轨迹与锚点轨迹的诊断结果。"""

    first_divergence_turn: int
    divergence_type: str
    masked_feedback: str


@dataclass
class AnchorSelectionDecision:
    """锚点选择输出。"""

    anchor_request: Optional["AsyncRolloutRequest"]
    priority: int
    reason: str


class ASTLogicalDiagnoser:
    """基于 AST 的后见逻辑诊断器。"""

    def build_ast(self, req: "AsyncRolloutRequest") -> List[ASTNode]:
        nodes: List[ASTNode] = []
        observed_tool_markers: List[str] = []
        for msg in req.messages:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tool_call in msg.tool_calls:
                tool_name = getattr(tool_call.function, "name", "") or "unknown"
                args = self._safe_json_load(getattr(tool_call.function, "arguments", "{}"))
                dependent_ids = self._dependent_tool_call_ids(args, observed_tool_markers)
                nodes.append(
                    ASTNode(
                        tool_name=tool_name,
                        tool_category=self._tool_category(tool_name),
                        arguments=args,
                        normalized_arguments=self._normalize_arguments(args),
                        tool_call_id=str(getattr(tool_call, "id", "") or ""),
                        dependent_tool_call_ids=dependent_ids,
                    )
                )
                observed_tool_markers.append(str(getattr(tool_call, "id", "")))
        return nodes

    def diagnose(self, failed_req: "AsyncRolloutRequest", anchor_req: "AsyncRolloutRequest") -> Optional[DiagnosisResult]:
        failed_ast = self.build_ast(failed_req)
        anchor_ast = self.build_ast(anchor_req)
        if not failed_ast and not anchor_ast:
            return None

        idx = self._first_divergence_idx(failed_ast, anchor_ast)
        divergence_type = self._divergence_type(failed_ast, anchor_ast, idx)
        feedback = self._build_masked_feedback(divergence_type, idx + 1)
        return DiagnosisResult(first_divergence_turn=idx, divergence_type=divergence_type, masked_feedback=feedback)

    def _first_divergence_idx(self, left: Sequence[ASTNode], right: Sequence[ASTNode]) -> int:
        for i, (lhs, rhs) in enumerate(zip(left, right)):
            if lhs != rhs:
                return i
        return min(len(left), len(right))

    def _divergence_type(self, failed_ast: Sequence[ASTNode], anchor_ast: Sequence[ASTNode], idx: int) -> str:
        if idx >= len(failed_ast) or idx >= len(anchor_ast):
            return "Strategy Divergence"
        if failed_ast[idx].tool_name != anchor_ast[idx].tool_name or failed_ast[idx].tool_category != anchor_ast[idx].tool_category:
            return "Strategy Divergence"
        return "Parameter Divergence"

    def _build_masked_feedback(self, divergence_type: str, turn_no: int) -> str:
        if divergence_type == "Strategy Divergence":
            return (
                f"在第 {turn_no} 步出现【策略分歧】：当前动作类别与可行逻辑链不一致。"
                "请先执行【信息检索类】动作以补齐前置状态，再继续后续状态修改。"
            )
        return (
            f"在第 {turn_no} 步出现【参数分歧】：动作类别正确但参数语义/类型不匹配。"
            "请检查函数参数取值、实体标识符、数据类型与前序返回值引用关系。"
        )

    def _tool_category(self, tool_name: str) -> str:
        lower = tool_name.lower()
        retrieve_markers = ["get", "search", "query", "list", "fetch", "read"]
        return "信息检索类" if any(k in lower for k in retrieve_markers) else "状态修改类"

    def _normalize_arguments(self, args: Any) -> str:
        """稳定化参数序列化，便于跨轨迹逐节点比对。"""
        try:
            return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(args)

    def _dependent_tool_call_ids(self, args: Any, markers: Sequence[str]) -> tuple[str, ...]:
        if not isinstance(args, dict):
            return ()
        serialized = json.dumps(args, ensure_ascii=False)
        return tuple(marker for marker in markers if marker and marker in serialized)

    def _safe_json_load(self, raw: Any) -> Any:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {"__raw_arguments__": str(raw)}


class SelfPlayAnchorSelector:
    """自博弈锚点智能选择器（批次同伴 -> 历史自我 -> GT-shadow）。"""

    def __init__(self) -> None:
        self._history_success_pool: Dict[str, "AsyncRolloutRequest"] = {}

    def is_success(self, req: "AsyncRolloutRequest") -> bool:
        rewards = req.reward_scores or {}
        turn_rewards = rewards.get("user_turn_rewards", []) if isinstance(rewards, dict) else []
        if isinstance(turn_rewards, list) and any(x == 1 for x in turn_rewards):
            return True

        for value in rewards.values() if isinstance(rewards, dict) else []:
            if isinstance(value, (int, float)) and value > 0.5:
                return True
            if isinstance(value, list) and any(isinstance(v, (int, float)) and v > 0.5 for v in value):
                return True
        return False

    def update_history(self, requests: Sequence["AsyncRolloutRequest"]) -> None:
        for req in requests:
            if self.is_success(req):
                self._history_success_pool[self._signature(req)] = req

    def select_anchor(self, candidate_group: Sequence["AsyncRolloutRequest"], failed_req: "AsyncRolloutRequest") -> AnchorSelectionDecision:
        # 优先级一：批次内同伴锚点（排除当前失败轨迹自身）
        peer_success = [req for req in candidate_group if req is not failed_req and self.is_success(req)]
        if peer_success:
            anchor = max(peer_success, key=self._shadow_topology_score)
            return AnchorSelectionDecision(anchor_request=anchor, priority=1, reason="in-batch peer")

        # 优先级二：历史自我锚点
        signature = self._signature(failed_req)
        if signature in self._history_success_pool:
            return AnchorSelectionDecision(anchor_request=self._history_success_pool[signature], priority=2, reason="historical self")

        # 优先级三：GT-shadow 局部锚点（从同组其他失败轨迹中选拓扑覆盖率最高者）
        failed_peers = [req for req in candidate_group if req is not failed_req]
        if failed_peers:
            best_failed = max(failed_peers, key=self._shadow_topology_score)
            return AnchorSelectionDecision(anchor_request=best_failed, priority=3, reason="gt-shadow partial")

        return AnchorSelectionDecision(anchor_request=None, priority=0, reason="no anchor")

    def _shadow_topology_score(self, req: "AsyncRolloutRequest") -> float:
        tool_call_count = 0
        unique_categories = set()
        for msg in req.messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_call_count += 1
                    tool_name = getattr(tool_call.function, "name", "") or "unknown"
                    unique_categories.add("query" if any(k in tool_name.lower() for k in ["get", "query", "search", "list"]) else "operate")
        return tool_call_count + 0.1 * len(unique_categories)

    def _signature(self, req: "AsyncRolloutRequest") -> str:
        user_seed = ""
        for msg in req.messages:
            if msg.role == "user":
                user_seed = str(msg.content)
                break
        tools_from_schema = []
        if getattr(req, "tool_schemas", None):
            for tool_schema in req.tool_schemas:
                func = getattr(tool_schema, "function", None)
                name = getattr(func, "name", "") if func is not None else ""
                if name:
                    tools_from_schema.append(name)
        tools_from_kwargs = sorted((getattr(req, "tools_kwargs", {}) or {}).keys())
        tools = sorted(set(tools_from_schema) | set(tools_from_kwargs))
        return f"{user_seed}||{'|'.join(tools)}"


class TemporalCompatibilityOrchestrator:
    """最小侵入式时序兼容架构：运行期静默收集、批次后诊断、分歧点注入。"""

    def __init__(self) -> None:
        self.anchor_selector = SelfPlayAnchorSelector()
        self.diagnoser = ASTLogicalDiagnoser()

    def build_retry_plan(
        self,
        base_req_list: Sequence["AsyncRolloutRequest"],
        first_pass_outputs: Sequence["AsyncRolloutRequest"],
    ) -> Dict[tuple[int, int], str]:
        grouped: Dict[int, List["AsyncRolloutRequest"]] = {}
        for req in first_pass_outputs:
            grouped.setdefault(req.batch_data_id, []).append(req)

        self.anchor_selector.update_history(first_pass_outputs)

        valid_keys = {(req.batch_data_id, req.rollout_offset) for req in base_req_list}
        retry_hints: Dict[tuple[int, int], str] = {}
        for batch_data_id, group in grouped.items():
            for failed_req in group:
                key = (batch_data_id, failed_req.rollout_offset)
                if key not in valid_keys or self.anchor_selector.is_success(failed_req):
                    continue
                decision = self.anchor_selector.select_anchor(group, failed_req)
                if decision.anchor_request is None:
                    continue
                diagnosis = self.diagnoser.diagnose(failed_req, decision.anchor_request)
                if diagnosis is None:
                    continue
                retry_hints[key] = (
                    "[Masked Causal Attribution] "
                    f"AnchorPriority={decision.priority}; "
                    f"Reason={decision.reason}; "
                    f"FirstDivergenceTurn={diagnosis.first_divergence_turn}; "
                    f"Hint={diagnosis.masked_feedback}"
                )
        return retry_hints

    def inject_hints_into_base_requests(
        self,
        base_req_list: Sequence["AsyncRolloutRequest"],
        retry_hints: Dict[tuple[int, int], str],
    ) -> List["AsyncRolloutRequest"]:
        rewritten: List["AsyncRolloutRequest"] = []
        for req in base_req_list:
            key = (req.batch_data_id, req.rollout_offset)
            if key in retry_hints:
                req.messages.append(self._build_message(req, retry_hints[key]))
            rewritten.append(req)
        logger.info("Self-play retry hints injected for %d requests", len(retry_hints))
        return rewritten

    def _build_message(self, req: "AsyncRolloutRequest", content: str) -> Any:
        if req.messages:
            msg_cls = type(req.messages[0])
            try:
                return msg_cls(role="user", content=content)
            except Exception:
                return {"role": "user", "content": content}
        return {"role": "user", "content": content}
#######新增（结束）#######
