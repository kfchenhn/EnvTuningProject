from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ActionNode:
    """A normalized action node used for structural comparison."""

    turn_index: int
    action_index: int
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class LogicDivergence:
    """First logic divergence between success and failure trajectories."""

    divergence_type: str
    turn_index: int
    action_index: int
    success_action: Optional[ActionNode]
    failed_action: Optional[ActionNode]
    reason: str


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value


def _normalize_arguments(raw_arguments: Any) -> Dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"value": raw_arguments}
    if not isinstance(raw_arguments, dict):
        return {"value": raw_arguments}
    normalized = {}
    for key in sorted(raw_arguments.keys()):
        value = raw_arguments[key]
        if isinstance(value, dict):
            value = _normalize_arguments(value)
        elif isinstance(value, list):
            value = [_normalize_scalar(v) for v in value]
        else:
            value = _normalize_scalar(value)
        normalized[str(key)] = value
    return normalized


def _is_noise_key(key: str) -> bool:
    noise_keywords = ("timestamp", "request_id", "trace_id", "nonce")
    key_lower = key.lower()
    return any(k in key_lower for k in noise_keywords)


def _extract_action(raw_action: Dict[str, Any], turn_index: int, action_index: int) -> ActionNode:
    tool_name = str(
        raw_action.get("name")
        or raw_action.get("tool_name")
        or raw_action.get("function", {}).get("name", "")
    )
    arguments = (
        raw_action.get("arguments")
        or raw_action.get("args")
        or raw_action.get("function", {}).get("arguments")
        or {}
    )
    normalized_args = _normalize_arguments(arguments)
    normalized_args = {
        k: v for k, v in normalized_args.items() if not _is_noise_key(k)
    }
    return ActionNode(
        turn_index=turn_index,
        action_index=action_index,
        tool_name=tool_name,
        arguments=normalized_args,
    )


def normalize_trajectory(raw_trajectory: Sequence[Iterable[Dict[str, Any]]]) -> List[List[ActionNode]]:
    """Normalize trajectory and canonically sort parallel calls."""

    normalized: List[List[ActionNode]] = []
    for turn_index, turn_actions in enumerate(raw_trajectory):
        nodes = [_extract_action(action, turn_index, idx) for idx, action in enumerate(turn_actions)]
        nodes.sort(key=lambda node: (node.tool_name, json.dumps(node.arguments, ensure_ascii=False, sort_keys=True)))
        for new_idx, node in enumerate(nodes):
            node.action_index = new_idx
        normalized.append(nodes)
    return normalized


def find_first_logic_divergence(
    success_trajectory: Sequence[Iterable[Dict[str, Any]]],
    failed_trajectory: Sequence[Iterable[Dict[str, Any]]],
) -> Optional[LogicDivergence]:
    """Find the first strategy/parameter divergence at normalized AST-like action level."""

    success = normalize_trajectory(success_trajectory)
    failed = normalize_trajectory(failed_trajectory)

    max_turns = max(len(success), len(failed))
    for turn_index in range(max_turns):
        success_turn = success[turn_index] if turn_index < len(success) else []
        failed_turn = failed[turn_index] if turn_index < len(failed) else []
        max_actions = max(len(success_turn), len(failed_turn))
        for action_index in range(max_actions):
            s_action = success_turn[action_index] if action_index < len(success_turn) else None
            f_action = failed_turn[action_index] if action_index < len(failed_turn) else None

            if s_action is None or f_action is None:
                return LogicDivergence(
                    divergence_type="strategy",
                    turn_index=turn_index,
                    action_index=action_index,
                    success_action=s_action,
                    failed_action=f_action,
                    reason="Action count mismatch at this branch.",
                )
            if s_action.tool_name != f_action.tool_name:
                return LogicDivergence(
                    divergence_type="strategy",
                    turn_index=turn_index,
                    action_index=action_index,
                    success_action=s_action,
                    failed_action=f_action,
                    reason="Tool choice diverged.",
                )
            if s_action.arguments != f_action.arguments:
                return LogicDivergence(
                    divergence_type="parameter",
                    turn_index=turn_index,
                    action_index=action_index,
                    success_action=s_action,
                    failed_action=f_action,
                    reason="Arguments diverged for the same tool.",
                )
    return None
