import json
from typing import Any, Dict, Iterable, List, Optional

from .data_models import CounterfactualSample, DivergencePoint, ToolCallNode, Trajectory


class ASTTrajectoryDiagnostics:
    """AST 风格轨迹诊断器。

    说明：这里不是解析真实编程语言 AST，而是把 tool-call 序列映射为“结构化节点序列”，
    再做结构层面的 first-divergence 定位，以降低格式噪声的干扰。
    """

    # 对逻辑无贡献、但会引入噪声的字段。
    IGNORED_FIELDS = {"timestamp", "request_id", "trace_id", "latency_ms"}

    def normalize_calls(self, decoded_calls: Iterable[Dict[str, Any]]) -> List[ToolCallNode]:
        """将原始调用标准化为可比较的节点序列。"""
        normalized: List[ToolCallNode] = []
        for call in decoded_calls or []:
            tool_name = call.get("name") or call.get("tool_name") or ""
            args = call.get("arguments") or call.get("args") or {}

            # 部分模型会把 arguments 作为 JSON 字符串输出，这里做鲁棒解码。
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}

            clean_args = {
                key: value
                for key, value in sorted(args.items(), key=lambda kv: kv[0])
                if key not in self.IGNORED_FIELDS
            }
            normalized.append(
                ToolCallNode(
                    tool_name=tool_name,
                    arguments=clean_args,
                    dependencies=sorted(call.get("dependencies", [])),
                )
            )

        # 并行等价调用可能顺序不同：排序后再比较，可显著减少误判。
        normalized.sort(key=lambda n: (n.tool_name, json.dumps(n.arguments, sort_keys=True, ensure_ascii=False)))
        return normalized

    def first_divergence(self, anchor: Trajectory, failed: Trajectory) -> Optional[DivergencePoint]:
        """定位第一逻辑分歧点。"""
        max_len = max(len(anchor.calls), len(failed.calls))
        for idx in range(max_len):
            anchor_call = anchor.calls[idx] if idx < len(anchor.calls) else None
            failed_call = failed.calls[idx] if idx < len(failed.calls) else None

            # 长度分歧：一方提前结束/缺失关键动作。
            if anchor_call is None or failed_call is None:
                return DivergencePoint(
                    index=idx,
                    divergence_type="length",
                    anchor_call=anchor_call,
                    failed_call=failed_call,
                    reason="One trajectory terminated earlier than the other.",
                )

            # 策略分歧：工具选择不同。
            if anchor_call.tool_name != failed_call.tool_name:
                return DivergencePoint(
                    index=idx,
                    divergence_type="strategy",
                    anchor_call=anchor_call,
                    failed_call=failed_call,
                    reason="Different tool selected at this decision point.",
                )

            # 参数分歧：工具相同但参数不一致。
            if anchor_call.arguments != failed_call.arguments:
                return DivergencePoint(
                    index=idx,
                    divergence_type="parameter",
                    anchor_call=anchor_call,
                    failed_call=failed_call,
                    reason="Same tool but semantically different arguments.",
                )
        return None

    def build_counterfactual(
        self,
        task_signature: str,
        turn_index: int,
        anchor_calls: Iterable[Dict[str, Any]],
        failed_calls: Iterable[Dict[str, Any]],
        anchor_reward: float,
        failed_reward: float,
    ) -> Optional[CounterfactualSample]:
        """从成功/失败轨迹构建反事实样本。"""
        anchor = Trajectory(task_signature, turn_index, self.normalize_calls(anchor_calls), anchor_reward)
        failed = Trajectory(task_signature, turn_index, self.normalize_calls(failed_calls), failed_reward)
        divergence = self.first_divergence(anchor, failed)
        if divergence is None:
            return None

        diagnosis = (
            f"At step {divergence.index}, failed policy shows {divergence.divergence_type} divergence. "
            f"Reason: {divergence.reason}"
        )
        return CounterfactualSample(
            task_signature=task_signature,
            turn_index=turn_index,
            anchor=anchor,
            failed=failed,
            divergence=divergence,
            diagnosis=diagnosis,
        )
