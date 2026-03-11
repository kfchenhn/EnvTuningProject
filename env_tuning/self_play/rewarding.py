from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from bfcl_env.multi_turn_checker import response_checker, state_checker


@dataclass
class ValidationResult:
    """双维验证结果：状态正确性 + 响应正确性。"""

    state_ok: bool
    response_ok: bool

    @property
    def reward(self) -> float:
        return float(self.state_ok and self.response_ok)


class DualOutcomeValidator:
    """自博弈触发使用的双维结果验证器。

    说明：
    - state_ok：关注环境最终状态是否与 GT 一致；
    - response_ok：关注本轮返回是否满足查询/内容正确性；
    - reward：两者取乘积（都对才得 1）。
    """

    def validate(
        self,
        model_instances: Dict[str, Any],
        gt_instances: Dict[str, Any],
        model_results: List[Any],
        gt_results: List[Any],
        turn_index: int,
    ) -> ValidationResult:
        # 状态维度校验：适用于会改变底层对象状态的工具调用。
        try:
            state_ok = bool(state_checker(model_instances, gt_instances).get("valid", False))
        except Exception:
            state_ok = False

        # 响应维度校验：适用于检索/问答类不改变状态的调用。
        try:
            response_ok = bool(response_checker(model_results, gt_results, turn_index).get("valid", False))
        except Exception:
            response_ok = False

        return ValidationResult(state_ok=state_ok, response_ok=response_ok)
