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
    """基于 AST 思路的工具轨迹首个逻辑分歧定位器。

    设计目标：
    1) 允许输入 JSON / Python 字面量两类轨迹文本；
    2) 对噪声字段做标准化清洗，降低日志字段对比较的干扰；
    3) 尽量保留时序信息，同时仅对“并行组内”动作做顺序无关化；
    4) 输出首个分歧点，供后续 GRPO 偏好构造使用。
    """

    NOISE_KEYS = {"timestamp", "request_id", "trace_id", "log_id"}
    PARALLEL_GROUP_KEYS = ("parallel_group", "parallel_id", "batch_id", "group_id")

    def normalize_calls(self, trajectory_text: str) -> List[Dict[str, Any]]:
        """将原始调用轨迹归一化为可比较的“结构化动作序列”。"""
        calls = self._parse_json_or_python_list(trajectory_text)
        # 先按原始顺序保存，避免对串行控制流造成错误对齐。
        normalized_steps: List[Dict[str, Any]] = []
        for item in calls:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool")
            if not isinstance(name, str):
                continue
            args = item.get("arguments") or item.get("args") or item.get("parameters") or {}
            if not isinstance(args, dict):
                args = {}

            # 统一剔除与策略无关的噪声字段，避免误判参数分歧。
            clean_args = {k: v for k, v in args.items() if k not in self.NOISE_KEYS}
            normalized_steps.append(
                {
                    "name": name,
                    "arguments": self._sort_structure(clean_args),
                    "depends_on": self._normalize_depends_on(item.get("depends_on")),
                    "parallel_group": self._extract_parallel_group(item),
                }
            )

        # 只在并行组内部进行排序，使并行调用顺序变化不影响诊断。
        return self._canonicalize_parallel_groups(normalized_steps)

    def _canonicalize_parallel_groups(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not steps:
            return []

        canonical: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(steps):
            group = steps[idx].get("parallel_group")
            if group is None:
                canonical.append(steps[idx])
                idx += 1
                continue

            j = idx
            same_group: List[Dict[str, Any]] = []
            while j < len(steps) and steps[j].get("parallel_group") == group:
                same_group.append(steps[j])
                j += 1

            same_group.sort(key=self._call_sort_key)
            canonical.extend(same_group)
            idx = j
        return canonical

    def _call_sort_key(self, call: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(call.get("name", "")),
            json.dumps(call.get("arguments", {}), ensure_ascii=False, sort_keys=True),
            json.dumps(call.get("depends_on", []), ensure_ascii=False, sort_keys=True),
        )

    def first_divergence(self, success_text: str, failed_text: str) -> Optional[DivergencePoint]:
        """定位两条轨迹的首个策略/参数/依赖分歧点。"""
        success = self.normalize_calls(success_text)
        failed = self.normalize_calls(failed_text)
        max_len = min(len(success), len(failed))

        for i in range(max_len):
            s_call, f_call = success[i], failed[i]
            if s_call["name"] != f_call["name"]:
                return DivergencePoint(i, "strategy", s_call, f_call)
            if s_call["arguments"] != f_call["arguments"]:
                return DivergencePoint(i, "parameter", s_call, f_call)
            if s_call.get("depends_on", []) != f_call.get("depends_on", []):
                return DivergencePoint(i, "dependency", s_call, f_call)

        if len(success) != len(failed):
            idx = max_len
            s_call = success[idx] if idx < len(success) else {"name": "<missing>", "arguments": {}}
            f_call = failed[idx] if idx < len(failed) else {"name": "<missing>", "arguments": {}}
            return DivergencePoint(idx, "length", s_call, f_call)

        return None

    def build_counterfactual_report(self, success_text: str, failed_text: str) -> Dict[str, Any]:
        """构造可直接写入离线样本池的反事实诊断报告。"""
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
        """优先 JSON，失败后回退 Python literal 解析。"""
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

    def _normalize_depends_on(self, value: Any) -> List[str]:
        """将 depends_on 统一为排序后的字符串列表。"""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            normalized = [str(v) for v in value]
            return sorted(normalized)
        return [str(value)]

    def _extract_parallel_group(self, item: Dict[str, Any]) -> Optional[str]:
        """提取并行组标识，支持多个候选字段名。"""
        for key in self.PARALLEL_GROUP_KEYS:
            value = item.get(key)
            if value is not None:
                return str(value)
        return None

    def _sort_structure(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._sort_structure(value[k]) for k in sorted(value)}
        if isinstance(value, list):
            return [self._sort_structure(v) for v in value]
        return value
