# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Dict, Any, List
from .data_models import InstanceState
from .score_calculator import ScoreCalculator
# ######新增（开始）###### self-play temporal compatibility components
from .self_play_feedback import (
    SelfPlayAnchorSelector,
    ASTRetrospectiveDiagnoser,
    PersistentAnchorStore,
    ToolSchemaClassifier,
)
# #######新增（结束）######


class TurnManager:
    """管理多轮对话逻辑"""

    def __init__(
        self,
        score_calculator: ScoreCalculator,
        enable_temporal_compat: bool = True,
        anchor_store_backend: str = "auto",
        anchor_store_file_path: str = "/tmp/env_tuning_self_play_anchor_store.jsonl",
        anchor_store_redis_url: str = "",
        anchor_store_redis_prefix: str = "env_tuning:self_play_anchor",
    ):
        self.score_calculator = score_calculator
        # ######新增（开始）######
        # Keep minimal-intrusion defaults, but support persistent shared pool.
        self.enable_temporal_compat = enable_temporal_compat
        store = PersistentAnchorStore(
            backend=anchor_store_backend,
            redis_url=anchor_store_redis_url or None,
            redis_prefix=anchor_store_redis_prefix,
            file_path=anchor_store_file_path,
        )
        classifier = ToolSchemaClassifier()
        self.anchor_selector = SelfPlayAnchorSelector(store=store, classifier=classifier)
        self.ast_diagnoser = ASTRetrospectiveDiagnoser(self.anchor_selector)
        # #######新增（结束）######

    def advance_to_next_turn(self, state: InstanceState, entry_id: str) -> Tuple[bool, str, float, Dict[str, Any]]:
        """
        推进到下一轮

        Args:
            state: 实例状态
            entry_id: 条目ID

        Returns:
            Tuple[bool, str, float, Dict[str, Any]]: (是否终止, 响应内容, 评分, 额外数据)
        """
        state.flush_exec_results_to_all()

        prev_turn_idx = state.current_turn_index
        ground_truth_calls = self._get_ground_truth_calls(state, prev_turn_idx)

        base_score = self.score_calculator.calculate_turn_score(state, ground_truth_calls, entry_id)
        score = base_score
        extra: Dict[str, Any] = {}

        # ######新增（开始）######
        if self.enable_temporal_compat and ground_truth_calls:
            score, diagnostic_extra = self._retrospective_temporal_remap(
                state=state,
                entry_id=entry_id,
                turn_idx=prev_turn_idx,
                ground_truth_calls=ground_truth_calls,
                base_score=base_score,
            )
            extra.update(diagnostic_extra)
        # #######新增（结束）######

        state.current_turn_index += 1
        should_terminate, next_question = self._prepare_next_question(state)
        state.reset_single_turn_buffers()
        return should_terminate, next_question, score, extra

    # ######新增（开始）######
    def _retrospective_temporal_remap(
        self,
        state: InstanceState,
        entry_id: str,
        turn_idx: int,
        ground_truth_calls: List[str],
        base_score: float,
    ) -> Tuple[float, Dict[str, Any]]:
        attempts = state.single_turn_attempt_records
        current_calls = attempts[-1].decoded_calls if attempts else []

        # Refresh schema classifier from current task environment before diagnosis.
        self.anchor_selector.set_schema_source(state.involved_classes)

        task_signature = self.anchor_selector.build_task_signature(
            entry_id=entry_id,
            turn_index=turn_idx,
            question=state.question[turn_idx] if turn_idx < len(state.question) else "",
        )
        anchor = self.anchor_selector.select_anchor(current_calls, attempts, ground_truth_calls, task_signature)
        diagnostic = self.ast_diagnoser.diagnose(current_calls, anchor, ground_truth_calls)

        if diagnostic.is_success:
            self.anchor_selector.push_success_anchor(task_signature, current_calls)
            return 1.0, {
                "retrospective_feedback": {
                    "mode": "self_play_temporal_compat",
                    "status": "success",
                    "base_score": base_score,
                    "remapped_score": 1.0,
                    "anchor_type": anchor.anchor_type,
                    "topology_score": diagnostic.topology_score,
                    "attempt_count": len(attempts),
                }
            }

        penalty = -2.5
        remapped_score = min(base_score, penalty)
        reward_patch = {
            "t < t_diverge": "保留原有进度奖励（由上层回放承载）",
            "t = t_diverge": penalty,
            "t > t_diverge": 0.0,
        }
        return remapped_score, {
            "retrospective_feedback": {
                "mode": "self_play_temporal_compat",
                "status": "failure",
                "base_score": base_score,
                "remapped_score": remapped_score,
                "anchor_type": diagnostic.anchor_type,
                "topology_score": diagnostic.topology_score,
                "first_logical_divergence_point": diagnostic.divergence_step,
                "divergence_type": diagnostic.divergence_type,
                "masked_causal_report": diagnostic.masked_report,
                "reward_patch": reward_patch,
                "attempt_count": len(attempts),
            }
        }
    # #######新增（结束）######

    def _get_ground_truth_calls(self, state: InstanceState, turn_index: int) -> list:
        if turn_index < len(state.ground_truth):
            return state.ground_truth[turn_index]
        return []

    def _prepare_next_question(self, state: InstanceState) -> Tuple[bool, str]:
        if state.processed_question:
            return False, state.processed_question.pop(0)
        return True, ""

    def should_force_quit(self, state: InstanceState, max_step_limit: int) -> bool:
        return state.current_turn_attempt_counts > max_step_limit

    def is_sequence_complete(self, state: InstanceState) -> bool:
        return len(state.processed_question) == 0

    def get_current_turn_info(self, state: InstanceState) -> Dict[str, Any]:
        return {
            "current_turn": state.current_turn_index,
            "total_turns": state.total_turns,
            "attempt_count": state.current_turn_attempt_counts,
            "remaining_questions": len(state.processed_question),
        }

    def reset_turn_counters(self, state: InstanceState) -> None:
        state.current_turn_attempt_counts = 0
        state.reset_single_turn_buffers()
