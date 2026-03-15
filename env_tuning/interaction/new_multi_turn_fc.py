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

import json
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4

from verl.interactions.base import BaseInteraction
from .data_models import InstanceState, ResponseData, ResponseType, ExecutionResult
from .response_handler import ResponseHandler
from .execution_manager import ExecutionManager
from .score_calculator import ScoreCalculator
from .turn_manager import TurnManager
from bfcl_env.multi_turn_utils import execute_multi_turn_func_call


class MultiTurnFunctionCallInteraction(BaseInteraction):
    """多轮函数调用交互主类"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get("name", "multi_turn_function_call")
        self._instance_dict: Dict[str, InstanceState] = {}

        # ######新增（开始）######
        # Runtime switches for retrospective self-play feedback.
        self.max_step_limit = config.get("max_step_limit", 5)
        self.enable_blind_rollout_feedback = config.get("enable_blind_rollout_feedback", True)
        self.enable_temporal_compat = config.get("enable_temporal_compat", True)

        # Step-9 persistent shared historical anchor pool options.
        self.anchor_store_backend = config.get("anchor_store_backend", "auto")
        self.anchor_store_file_path = config.get("anchor_store_file_path", "/tmp/env_tuning_self_play_anchor_store.jsonl")
        self.anchor_store_redis_url = config.get("anchor_store_redis_url", "")
        self.anchor_store_redis_prefix = config.get("anchor_store_redis_prefix", "env_tuning:self_play_anchor")
        # #######新增（结束）######

        self.response_handler = ResponseHandler()
        self.execution_manager = ExecutionManager()
        self.score_calculator = ScoreCalculator()

        # ######新增（开始）######
        self.turn_manager = TurnManager(
            self.score_calculator,
            enable_temporal_compat=self.enable_temporal_compat,
            anchor_store_backend=self.anchor_store_backend,
            anchor_store_file_path=self.anchor_store_file_path,
            anchor_store_redis_url=self.anchor_store_redis_url,
            anchor_store_redis_prefix=self.anchor_store_redis_prefix,
        )
        # #######新增（结束）######

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """创建工具实例"""
        if instance_id is None:
            instance_id = str(uuid4())

        entry_id: str = kwargs["id"]
        initial_config: Dict[str, Any] = json.loads(kwargs["initial_config"])
        involved_classes: Dict[str, Any] = kwargs["involved_classes"]
        ground_truth: List[Any] = kwargs["ground_truth"]
        processed_question: List[str] = kwargs["processed_question"]
        question: List[str] = kwargs["question"]
        total_turns = len(question)

        _, model_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            instance_id,
            entry_id,
            long_context=("long_context" in entry_id or "composite" in entry_id),
            is_evaL_run=False,
        )

        execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            instance_id + "_ground_truth",
            entry_id,
            long_context=("long_context" in entry_id or "composite" in entry_id),
            is_evaL_run=True,
        )

        state = InstanceState(
            initial_config=initial_config,
            involved_classes=involved_classes,
            ground_truth=ground_truth,
            processed_question=processed_question,
            question=question,
            involved_instances=model_instances,
            total_turns=total_turns,
        )

        self._instance_dict[instance_id] = state
        return instance_id

    async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, Dict[str, Any]]:
        """简化的响应生成方法 - 只负责协调各个模块"""
        state = self._instance_dict[instance_id]
        entry_id = kwargs["id"]

        response_data = self.response_handler.parse_and_validate(messages)
        if response_data.has_error:
            # ######新增（开始）######
            # Parse-error branch now keeps real `instance_id` for cleanup.
            return await self._handle_response_error(response_data, state, instance_id, entry_id)
            # #######新增（结束）######

        special_case_result = self._handle_special_cases(response_data, state, entry_id)
        if special_case_result:
            return special_case_result

        execution_result = self._execute_function_calls(response_data, state, instance_id, entry_id)
        return self._determine_next_action(execution_result, state, entry_id)

    # ######新增（开始）######
    async def _handle_response_error(
        self,
        response_data: ResponseData,
        state: InstanceState,
        instance_id: str,
        entry_id: str,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """处理响应错误"""
        state.current_turn_attempt_counts += 1

        if self.turn_manager.should_force_quit(state, self.max_step_limit):
            should_term, content, score, extra = self.turn_manager.advance_to_next_turn(state, entry_id)
            if should_term:
                await self.finalize_interaction(instance_id=instance_id)

            ground_truth_calls = self.turn_manager._get_ground_truth_calls(state, state.current_turn_index - 1)
            if not ground_truth_calls:
                score = 0.0

            return should_term, content, score, extra

        return False, response_data.error_message or "Parse error", -3.0, {}

    # #######新增（结束）######

    def _handle_special_cases(
        self,
        response_data: ResponseData,
        state: InstanceState,
        entry_id: str,
    ) -> Optional[Tuple[bool, str, float, Dict[str, Any]]]:
        """处理特殊情况"""
        ground_truth_calls = self.turn_manager._get_ground_truth_calls(state, state.current_turn_index)

        if not ground_truth_calls:
            should_term, content, base_score, extra = self.turn_manager.advance_to_next_turn(state, entry_id)
            assert base_score == -1.0, "Ground truth call list is empty, returned score should be -1.0"
            if response_data.response_type == ResponseType.ANSWER:
                return should_term, content, 1.0, extra
            if response_data.response_type == ResponseType.TOOL_CALL:
                warning_hint = (
                    "(SYSTEM WARNING: You should not call any function in this turn because certain "
                    "function description(s) or parameter(s) is missing in this turn. Previous turn is "
                    "forced quit. Current function(s) will not be executed.) Next turn question:\n"
                )
                return should_term, warning_hint + content, 0.0, extra

        return None

    def _execute_function_calls(self, response_data: ResponseData, state: InstanceState, instance_id: str, entry_id: str) -> ExecutionResult:
        """执行函数调用"""
        return self.execution_manager.execute_function_calls(response_data.content, state, instance_id, entry_id)

    def _determine_next_action(self, execution_result: ExecutionResult, state: InstanceState, entry_id: str) -> Tuple[bool, str, float, Dict[str, Any]]:
        """决定下一步行动"""
        if not execution_result.should_continue:
            return self.turn_manager.advance_to_next_turn(state, entry_id)

        state.involved_instances = execution_result.new_instances
        state.add_exec_results(execution_result.execution_results)
        state.current_turn_attempt_counts += 1

        if self.turn_manager.should_force_quit(state, self.max_step_limit):
            return self.turn_manager.advance_to_next_turn(state, entry_id)

        # ######新增（开始）######
        user_hint, score = self.execution_manager.format_execution_response(
            execution_result.execution_results,
            execution_result.has_error,
            blind_mode=self.enable_blind_rollout_feedback,
        )
        # #######新增（结束）######
        return False, user_hint, score, {}

    async def calculate_score(self) -> float:
        """计算交互评分"""
        return 0.0

    async def finalize_interaction(self, instance_id: str = None, **kwargs) -> None:
        """清理交互资源"""
        if instance_id and instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
