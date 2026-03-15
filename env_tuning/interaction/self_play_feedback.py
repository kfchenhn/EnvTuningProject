# ######新增（开始）######
"""Self-play retrospective feedback helpers.

This module implements three core capabilities required by step-9 evolution:
1) Persistent historical anchor pool (disk / Redis) for multi-worker sharing.
2) Schema-aware tool category auto-classification.
3) Semantic-slot-based argument comparison (stronger than type-only comparison).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AnchorSelection:
    """Selected anchor metadata."""

    anchor_type: str
    anchor_calls: List[str]
    source: str


@dataclass
class DiagnosticResult:
    """Retrospective diagnosis result for current trajectory."""

    is_success: bool
    divergence_step: int
    divergence_type: str
    masked_report: str
    topology_score: float
    anchor_type: str


class PersistentAnchorStore:
    """Persistent store for historical anchors.

    Priority:
      - Redis (if URL available and redis package installed)
      - Local JSONL file (shared path can be mounted across workers)
    """

    def __init__(
        self,
        backend: str = "auto",
        redis_url: Optional[str] = None,
        redis_prefix: str = "env_tuning:self_play_anchor",
        file_path: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.redis_url = redis_url or os.getenv("SELF_PLAY_ANCHOR_REDIS_URL")
        self.redis_prefix = redis_prefix
        self.file_path = file_path or os.getenv(
            "SELF_PLAY_ANCHOR_STORE_PATH",
            "/tmp/env_tuning_self_play_anchor_store.jsonl",
        )
        self._redis_client = None

        if self._use_redis():
            self._init_redis_client()

    def _use_redis(self) -> bool:
        if self.backend == "file":
            return False
        return bool(self.redis_url)

    def _init_redis_client(self) -> None:
        try:
            import redis  # type: ignore

            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self._redis_client.ping()
        except Exception:
            # Fail open to file mode to avoid breaking training startup.
            self._redis_client = None

    def _redis_key(self, task_signature: str) -> str:
        return f"{self.redis_prefix}:{task_signature}"

    def push(self, task_signature: str, calls: List[str], max_size: int = 20) -> None:
        if not calls:
            return

        # Redis backend
        if self._redis_client is not None:
            key = self._redis_key(task_signature)
            payload = json.dumps(calls, ensure_ascii=False)
            pipe = self._redis_client.pipeline()
            pipe.rpush(key, payload)
            pipe.ltrim(key, -max_size, -1)
            pipe.execute()
            return

        # File backend
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        record = {"task_signature": task_signature, "calls": calls}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get(self, task_signature: str, max_size: int = 20) -> List[List[str]]:
        # Redis backend
        if self._redis_client is not None:
            key = self._redis_key(task_signature)
            values = self._redis_client.lrange(key, -max_size, -1)
            output: List[List[str]] = []
            for val in values:
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        output.append([str(x) for x in parsed])
                except Exception:
                    continue
            return output

        # File backend (latest N entries for this signature)
        if not os.path.exists(self.file_path):
            return []

        output: List[List[str]] = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("task_signature") != task_signature:
                        continue
                    calls = obj.get("calls")
                    if isinstance(calls, list):
                        output.append([str(x) for x in calls])
                except Exception:
                    continue
        return output[-max_size:]


class ToolSchemaClassifier:
    """Schema-aware tool categorizer.

    It extracts tool metadata from `involved_classes` (dict/list/object) and uses
    both schema text and tool name to infer semantic category.
    """

    def __init__(self) -> None:
        self._tool_meta: Dict[str, str] = {}

    def load_schema(self, schema_source: Any) -> None:
        self._tool_meta = self._extract_tool_meta(schema_source)

    def classify(self, tool_name: str) -> str:
        text = f"{tool_name} {self._tool_meta.get(tool_name, '')}".lower()
        if any(k in text for k in ["get", "list", "search", "query", "fetch", "read"]):
            return "信息检索类"
        if any(k in text for k in ["create", "update", "delete", "book", "cancel", "write", "set"]):
            return "状态修改类"
        if any(k in text for k in ["message", "email", "chat", "post", "notify"]):
            return "通信交互类"
        if any(k in text for k in ["file", "path", "directory", "folder", "storage"]):
            return "文件系统类"
        if any(k in text for k in ["trade", "order", "portfolio", "stock", "crypto"]):
            return "交易金融类"
        if any(k in text for k in ["math", "calc", "compute", "equation"]):
            return "计算推理类"
        return "通用工具类"

    def _extract_tool_meta(self, schema_source: Any) -> Dict[str, str]:
        meta: Dict[str, str] = {}

        def _register(name: str, desc: str) -> None:
            if not name:
                return
            meta[name] = (meta.get(name, "") + " " + desc).strip()

        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                # Common schema patterns
                maybe_name = obj.get("name") or obj.get("tool_name") or obj.get("function")
                maybe_desc = obj.get("description") or obj.get("desc") or ""
                if isinstance(maybe_name, str):
                    _register(maybe_name, str(maybe_desc))
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)
            else:
                # class / callable / object fallback
                name = getattr(obj, "__name__", None)
                doc = getattr(obj, "__doc__", None)
                if isinstance(name, str):
                    _register(name, doc or "")

        _walk(schema_source)
        return meta


class SelfPlayAnchorSelector:
    """三级优先级锚点选择器（批内同伴 -> 历史自我 -> GT-Shadow）。"""

    def __init__(
        self,
        store: Optional[PersistentAnchorStore] = None,
        classifier: Optional[ToolSchemaClassifier] = None,
    ) -> None:
        self.store = store or PersistentAnchorStore()
        self.classifier = classifier or ToolSchemaClassifier()

    def set_schema_source(self, schema_source: Any) -> None:
        """Update tool schema metadata from current task env."""
        self.classifier.load_schema(schema_source)

    def build_task_signature(self, entry_id: str, turn_index: int, question: str) -> str:
        payload = f"{entry_id}|{turn_index}|{question}".encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def push_success_anchor(self, task_signature: str, calls: List[str], max_size: int = 20) -> None:
        self.store.push(task_signature, calls, max_size=max_size)

    def select_anchor(
        self,
        current_calls: List[str],
        attempts: List[Any],
        gt_calls: List[str],
        task_signature: str,
    ) -> AnchorSelection:
        # Priority-1: in-batch peer anchor (same turn attempts)
        peer_success = [
            attempt.decoded_calls
            for attempt in attempts
            if getattr(attempt, "decoded_calls", None) and self._exact_match(attempt.decoded_calls, gt_calls)
        ]
        if peer_success:
            return AnchorSelection("in_batch_peer", peer_success[0], "current_turn_attempts")

        # Priority-2: historical self anchor (persistent shared pool)
        historical = self.store.get(task_signature)
        if historical:
            current_nodes = self._to_ast_nodes(current_calls)
            best_hist = min(
                historical,
                key=lambda calls: self._distance(current_nodes, self._to_ast_nodes(calls)),
            )
            return AnchorSelection("historical_self", best_hist, "persistent_store")

        # Priority-3: GT-shadow partial anchor
        failed_calls = [attempt.decoded_calls for attempt in attempts if getattr(attempt, "decoded_calls", None)]
        if failed_calls:
            gt_nodes = self._to_ast_nodes(gt_calls)
            best_failed = min(
                failed_calls,
                key=lambda calls: self._distance(self._to_ast_nodes(calls), gt_nodes),
            )
            return AnchorSelection("gt_shadow_partial", best_failed, "best_failed_by_gt_topology")

        return AnchorSelection("gt_shadow_partial", gt_calls, "ground_truth_topology_only")

    def _exact_match(self, calls_a: List[str], calls_b: List[str]) -> bool:
        nodes_a = self._to_ast_nodes(calls_a)
        nodes_b = self._to_ast_nodes(calls_b)
        return self._signature_seq(nodes_a, include_name=True) == self._signature_seq(nodes_b, include_name=True)

    def _signature_seq(self, nodes: List[Dict[str, Any]], include_name: bool = True) -> List[str]:
        signatures = []
        for node in nodes:
            base = [
                node["category"],
                ",".join(node["semantic_slot_signature"]),
                f"dep:{int(node['has_dependency'])}",
            ]
            if include_name:
                base.insert(1, node["name"])
            signatures.append("|".join(base))
        return signatures

    def _distance(self, nodes_a: List[Dict[str, Any]], nodes_b: List[Dict[str, Any]]) -> int:
        # Sequence edit distance as practical approximation for AST topology distance.
        seq_a = self._signature_seq(nodes_a, include_name=True)
        seq_b = self._signature_seq(nodes_b, include_name=True)
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

    def _to_ast_nodes(self, calls: List[str]) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []
        for call in calls:
            node = self._parse_call(call)
            if node:
                nodes.append(node)
        return nodes

    def _parse_call(self, call_expr: str) -> Optional[Dict[str, Any]]:
        try:
            expr = ast.parse(call_expr, mode="eval").body
            if not isinstance(expr, ast.Call):
                return None
            name = self._resolve_name(expr.func)
            kw_pairs = sorted((kw.arg or "_", kw.value) for kw in expr.keywords)
            slot_sig = [self._semantic_slot_signature(arg, value) for arg, value in kw_pairs]
            has_dependency = any(self._contains_symbol(v) for _, v in kw_pairs)
            return {
                "name": name,
                "category": self.classifier.classify(name),
                "semantic_slot_signature": slot_sig,
                "has_dependency": has_dependency,
            }
        except Exception:
            return None

    def _resolve_name(self, func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            cur = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        return "unknown_tool"

    def _semantic_slot_signature(self, arg_name: str, value_node: ast.AST) -> str:
        """Semantic slot signature: `<arg_name>:<slot_type>:<value_semantics>`.

        This is stronger than type-only comparison and supports slot-level mismatch
        detection (e.g., using destination id where source id is required).
        """
        slot_type = self._slot_type_by_name(arg_name)
        value_sem = self._value_semantics(value_node)
        return f"{arg_name}:{slot_type}:{value_sem}"

    def _slot_type_by_name(self, arg_name: str) -> str:
        low = arg_name.lower()
        if any(k in low for k in ["id", "uuid", "token", "key", "secret"]):
            return "identity_slot"
        if any(k in low for k in ["date", "time", "timestamp", "deadline"]):
            return "time_slot"
        if any(k in low for k in ["from", "to", "source", "target", "location", "city", "country"]):
            return "location_slot"
        if any(k in low for k in ["amount", "price", "cost", "budget", "quantity", "count"]):
            return "numeric_slot"
        if any(k in low for k in ["status", "state", "mode", "type", "category"]):
            return "state_slot"
        if any(k in low for k in ["query", "keyword", "text", "prompt", "message", "content"]):
            return "text_slot"
        return "generic_slot"

    def _value_semantics(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int):
                return "int"
            if isinstance(node.value, float):
                return "float"
            if isinstance(node.value, str):
                return self._string_semantics(node.value)
            return type(node.value).__name__
        if isinstance(node, ast.Name):
            return "symbol_ref"
        if isinstance(node, ast.Attribute):
            return "attr_ref"
        if isinstance(node, ast.List):
            return "list"
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, ast.Tuple):
            return "tuple"
        return "expr"

    def _string_semantics(self, value: str) -> str:
        low = value.lower()
        if any(k in low for k in ["http://", "https://", "www."]):
            return "url_string"
        if any(k in low for k in ["@", "mail"]):
            return "contact_string"
        if any(k in low for k in ["-", "/", ":"]) and any(ch.isdigit() for ch in low):
            return "datetime_like_string"
        if any(ch.isdigit() for ch in low) and any(ch.isalpha() for ch in low):
            return "mixed_id_string"
        if low.isdigit():
            return "digit_string"
        return "text_string"

    def _contains_symbol(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, (ast.Name, ast.Attribute)):
                return True
        return False


class ASTRetrospectiveDiagnoser:
    """AST-based retrospective diagnoser with masked causal attribution."""

    def __init__(self, selector: SelfPlayAnchorSelector):
        self.selector = selector

    def diagnose(self, current_calls: List[str], anchor: AnchorSelection, gt_calls: List[str]) -> DiagnosticResult:
        current_nodes = self.selector._to_ast_nodes(current_calls)
        anchor_nodes = self.selector._to_ast_nodes(anchor.anchor_calls)
        gt_nodes = self.selector._to_ast_nodes(gt_calls)

        current_sig = self.selector._signature_seq(current_nodes, include_name=True)
        gt_sig = self.selector._signature_seq(gt_nodes, include_name=True)
        if current_sig == gt_sig:
            return DiagnosticResult(True, -1, "none", "", 1.0, anchor.anchor_type)

        compare_target = anchor_nodes if anchor_nodes else gt_nodes
        divergence_step, divergence_type, masked_report = self._first_divergence(current_nodes, compare_target)

        dist_to_gt = self.selector._distance(current_nodes, gt_nodes)
        max_len = max(len(current_nodes), len(gt_nodes), 1)
        topology_score = max(0.0, 1.0 - (dist_to_gt / max_len))

        return DiagnosticResult(
            is_success=False,
            divergence_step=divergence_step,
            divergence_type=divergence_type,
            masked_report=masked_report,
            topology_score=topology_score,
            anchor_type=anchor.anchor_type,
        )

    def _first_divergence(self, current: List[Dict[str, Any]], anchor: List[Dict[str, Any]]) -> Tuple[int, str, str]:
        min_len = min(len(current), len(anchor))
        for idx in range(min_len):
            cur = current[idx]
            anc = anchor[idx]
            if cur["name"] != anc["name"]:
                return idx, "strategy_divergence", self._masked_report(idx, cur["category"], anc["category"], "strategy_divergence")
            if cur["semantic_slot_signature"] != anc["semantic_slot_signature"]:
                return idx, "parameter_divergence", self._masked_report(idx, cur["category"], anc["category"], "parameter_divergence")
            if cur["has_dependency"] != anc["has_dependency"]:
                return idx, "parameter_divergence", self._masked_report(idx, cur["category"], anc["category"], "parameter_divergence")

        idx = min_len
        wrong_cat = current[idx]["category"] if idx < len(current) else "缺失动作"
        anchor_cat = anchor[idx]["category"] if idx < len(anchor) else "缺失动作"
        return idx, "strategy_divergence", self._masked_report(idx, wrong_cat, anchor_cat, "strategy_divergence")

    def _masked_report(self, step: int, wrong_cat: str, anchor_cat: str, divergence_type: str) -> str:
        if divergence_type == "parameter_divergence":
            return (
                f"在进行到第 {step + 1} 步时，你发生了【参数分歧】。"
                f"当前动作属于【{wrong_cat}】，关键语义槽位或依赖关系与目标拓扑不一致。"
                "建议先检查前置查询返回值与参数槽位绑定关系，再执行当前动作。"
            )
        return (
            f"在进行到第 {step + 1} 步时，你发生了【策略分歧】。"
            f"你执行了【{wrong_cat}】动作，而目标拓扑更接近先执行【{anchor_cat}】动作。"
            "建议先补齐前置状态再推进后续调用。"
        )


# #######新增（结束）######
