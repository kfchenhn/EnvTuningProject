#######新增（开始）#######
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

from env_tuning.self_play_temporal_compat import ASTLogicalDiagnoser, SelfPlayAnchorSelector, TemporalCompatibilityOrchestrator


@dataclass
class FakeMessage:
    role: str
    content: str
    tool_calls: list | None = None


@dataclass
class FakeReq:
    batch_data_id: int
    rollout_offset: int
    messages: list[FakeMessage]
    reward_scores: dict = field(default_factory=dict)
    tools_kwargs: dict = field(default_factory=dict)


def _tool_call(name: str, arguments: dict, tool_id: str = "1"):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def _req_with_single_tool(batch_id: int, offset: int, tool_name: str, args: dict, rewards: dict, user: str = "u") -> FakeReq:
    return FakeReq(
        batch_data_id=batch_id,
        rollout_offset=offset,
        messages=[
            FakeMessage(role="user", content=user),
            FakeMessage(role="assistant", content="", tool_calls=[_tool_call(tool_name, args)]),
        ],
        reward_scores=rewards,
        tools_kwargs={"demo_tool": {}},
    )


def test_ast_diagnose_strategy_divergence():
    diagnoser = ASTLogicalDiagnoser()
    failed = _req_with_single_tool(0, 0, "update_order", {"order_id": 1}, {"user_turn_rewards": [0]})
    anchor = _req_with_single_tool(0, 1, "search_order", {"query": "abc"}, {"user_turn_rewards": [1]})

    result = diagnoser.diagnose(failed, anchor)
    assert result is not None
    assert result.divergence_type == "Strategy Divergence"
    assert "策略分歧" in result.masked_feedback


def test_ast_diagnose_parameter_divergence():
    diagnoser = ASTLogicalDiagnoser()
    failed = _req_with_single_tool(0, 0, "search_order", {"order_id": "1"}, {"user_turn_rewards": [0]})
    anchor = _req_with_single_tool(0, 1, "search_order", {"order_id": 1}, {"user_turn_rewards": [1]})

    result = diagnoser.diagnose(failed, anchor)
    assert result is not None
    assert result.divergence_type == "Parameter Divergence"
    assert "参数分歧" in result.masked_feedback


def test_anchor_selector_priority_flow():
    selector = SelfPlayAnchorSelector()
    failed = _req_with_single_tool(0, 0, "update_order", {"id": 1}, {"user_turn_rewards": [0]}, user="same")
    success_peer = _req_with_single_tool(0, 1, "search_order", {"id": 1}, {"user_turn_rewards": [1]}, user="same")

    # 优先级一：批次内成功同伴
    decision_1 = selector.select_anchor([failed, success_peer], failed)
    assert decision_1.priority == 1
    assert decision_1.reason == "in-batch peer"

    # 优先级二：历史成功
    selector.update_history([success_peer])
    failed_only = _req_with_single_tool(1, 0, "update_order", {"id": 2}, {"user_turn_rewards": [0]}, user="same")
    decision_2 = selector.select_anchor([failed_only], failed_only)
    assert decision_2.priority == 2
    assert decision_2.reason == "historical self"


def test_orchestrator_build_retry_and_inject_hint():
    orchestrator = TemporalCompatibilityOrchestrator()
    failed = _req_with_single_tool(2, 0, "update_order", {"id": 1}, {"user_turn_rewards": [0]}, user="u2")
    success = _req_with_single_tool(2, 1, "search_order", {"id": 1}, {"user_turn_rewards": [1]}, user="u2")

    hints = orchestrator.build_retry_plan([failed, success], [failed, success])
    assert (2, 0) in hints
    assert "Masked Causal Attribution" in hints[(2, 0)]

    rewritten = orchestrator.inject_hints_into_base_requests([failed, success], hints)
    failed_after = next(req for req in rewritten if req.rollout_offset == 0)
    assert failed_after.messages[-1].role == "user"
    assert "Masked Causal Attribution" in failed_after.messages[-1].content
#######新增（结束）#######
