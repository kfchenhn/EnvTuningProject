import json
from typing import Any, Dict, Iterable, List, Optional

from .data_models import CounterfactualSample, DivergencePoint, ToolCallNode, Trajectory


class ASTTrajectoryDiagnostics:
    """Diagnose first logical divergence with AST-like normalized tool-call nodes."""

    IGNORED_FIELDS = {"timestamp", "request_id", "trace_id", "latency_ms"}

    def normalize_calls(self, decoded_calls: Iterable[Dict[str, Any]]) -> List[ToolCallNode]:
        normalized: List[ToolCallNode] = []
        for call in decoded_calls or []:
            tool_name = call.get("name") or call.get("tool_name") or ""
            args = call.get("arguments") or call.get("args") or {}
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

        # Canonicalize order for parallel-equivalent calls.
        normalized.sort(key=lambda n: (n.tool_name, json.dumps(n.arguments, sort_keys=True, ensure_ascii=False)))
        return normalized

    def first_divergence(self, anchor: Trajectory, failed: Trajectory) -> Optional[DivergencePoint]:
        max_len = max(len(anchor.calls), len(failed.calls))
        for idx in range(max_len):
            anchor_call = anchor.calls[idx] if idx < len(anchor.calls) else None
            failed_call = failed.calls[idx] if idx < len(failed.calls) else None

            if anchor_call is None or failed_call is None:
                return DivergencePoint(
                    index=idx,
                    divergence_type="length",
                    anchor_call=anchor_call,
                    failed_call=failed_call,
                    reason="One trajectory terminated earlier than the other.",
                )
            if anchor_call.tool_name != failed_call.tool_name:
                return DivergencePoint(
                    index=idx,
                    divergence_type="strategy",
                    anchor_call=anchor_call,
                    failed_call=failed_call,
                    reason="Different tool selected at this decision point.",
                )
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
