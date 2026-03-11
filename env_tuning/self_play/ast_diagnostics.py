from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DivergencePoint:
    index: int
    divergence_type: str
    success_action: Dict[str, Any]
    failed_action: Dict[str, Any]


class ASTTrajectoryDiagnostic:
    """AST-based first-divergence finder for tool-call trajectories."""

    NOISE_KEYS = {"timestamp", "request_id", "trace_id", "log_id"}

    def normalize_calls(self, trajectory_text: str) -> List[Dict[str, Any]]:
        calls = self._parse_json_or_python_list(trajectory_text)
        normalized: List[Dict[str, Any]] = []
        for item in calls:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool")
            if not isinstance(name, str):
                continue
            args = item.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            clean_args = {k: v for k, v in args.items() if k not in self.NOISE_KEYS}
            normalized.append({"name": name, "arguments": self._sort_structure(clean_args)})

        normalized.sort(key=lambda x: (x["name"], json.dumps(x["arguments"], ensure_ascii=False, sort_keys=True)))
        return normalized

    def first_divergence(self, success_text: str, failed_text: str) -> Optional[DivergencePoint]:
        success = self.normalize_calls(success_text)
        failed = self.normalize_calls(failed_text)
        max_len = min(len(success), len(failed))

        for i in range(max_len):
            s_call, f_call = success[i], failed[i]
            if s_call["name"] != f_call["name"]:
                return DivergencePoint(i, "strategy", s_call, f_call)
            if s_call["arguments"] != f_call["arguments"]:
                return DivergencePoint(i, "parameter", s_call, f_call)

        if len(success) != len(failed):
            idx = max_len
            s_call = success[idx] if idx < len(success) else {"name": "<missing>", "arguments": {}}
            f_call = failed[idx] if idx < len(failed) else {"name": "<missing>", "arguments": {}}
            return DivergencePoint(idx, "length", s_call, f_call)

        return None

    def build_counterfactual_report(self, success_text: str, failed_text: str) -> Dict[str, Any]:
        divergence = self.first_divergence(success_text, failed_text)
        if divergence is None:
            return {"has_divergence": False, "message": "No logical divergence found."}
        return {
            "has_divergence": True,
            "first_divergence_step": divergence.index,
            "divergence_type": divergence.divergence_type,
            "success_action": divergence.success_action,
            "failed_action": divergence.failed_action,
            "explanation_template": (
                "在该状态下，成功轨迹选择了 {success}，而失败轨迹选择了 {failed}。"
                "建议在此步优先遵循成功动作的工具与参数逻辑。"
            ).format(success=divergence.success_action, failed=divergence.failed_action),
        }

    def _parse_json_or_python_list(self, text: str) -> List[Any]:
        text = text.strip()
        if not text:
            return []
        try:
            result = json.loads(text)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

        try:
            node = ast.parse(text, mode="eval")
            value = ast.literal_eval(node)
            return value if isinstance(value, list) else [value]
        except Exception:
            return []

    def _sort_structure(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._sort_structure(value[k]) for k in sorted(value)}
        if isinstance(value, list):
            return [self._sort_structure(v) for v in value]
        return value
