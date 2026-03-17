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

from typing import List, Dict, Any, Optional, Tuple
from .data_models import AttemptRecord
import difflib


#######新增（开始）#######
class SelfPlayAnchorSelector:
    """自博弈锚点智能选择器"""
    
    def __init__(self):
        # 历史成功锚点池：key为task_signature，value为成功轨迹列表
        self.historical_anchor_pool: Dict[str, List[AttemptRecord]] = {}
    
    def select_anchor(self, current_attempts: List[AttemptRecord], ground_truth: List[Any], 
                     task_signature: str) -> Optional[AttemptRecord]:
        """
        选择最合适的自博弈锚点
        
        Args:
            current_attempts: 当前轮次的尝试记录
            ground_truth: 地面真相
            task_signature: 任务签名（entry_id + turn + question）
            
        Returns:
            Optional[AttemptRecord]: 选中的锚点记录
        """
        # 优先级一：In-batch Peer Anchor
        anchor = self._select_in_batch_peer_anchor(current_attempts, ground_truth)
        if anchor:
            return anchor
        
        # 优先级二：Historical Self Anchor
        anchor = self._select_historical_self_anchor(task_signature, ground_truth)
        if anchor:
            return anchor
        
        # 优先级三：GT-Shadow Partial Anchor
        anchor = self._select_gt_shadow_partial_anchor(current_attempts, ground_truth)
        return anchor
    
    def _select_in_batch_peer_anchor(self, attempts: List[AttemptRecord], 
                                   ground_truth: List[Any]) -> Optional[AttemptRecord]:
        """优先级一：在当前批次中选择与GT拓扑一致的成功轨迹"""
        for attempt in attempts:
            if not attempt.has_error and self._is_topology_match(attempt.decoded_calls, ground_truth):
                return attempt
        return None
    
    def _select_historical_self_anchor(self, task_signature: str, 
                                     ground_truth: List[Any]) -> Optional[AttemptRecord]:
        """优先级二：从历史成功锚点池中选择拓扑距离最近的锚点"""
        if task_signature not in self.historical_anchor_pool:
            return None
        
        historical_anchors = self.historical_anchor_pool[task_signature]
        if not historical_anchors:
            return None
        
        # 找到拓扑距离最近的锚点
        min_distance = float('inf')
        best_anchor = None
        
        for anchor in historical_anchors:
            distance = self._calculate_topology_distance(anchor.decoded_calls, ground_truth)
            if distance < min_distance:
                min_distance = distance
                best_anchor = anchor
        
        return best_anchor
    
    def _select_gt_shadow_partial_anchor(self, attempts: List[AttemptRecord], 
                                       ground_truth: List[Any]) -> Optional[AttemptRecord]:
        """优先级三：从失败轨迹中选择与GT AST拓扑距离最近的局部优胜轨迹"""
        if not attempts:
            return None
        
        min_distance = float('inf')
        best_attempt = None
        
        for attempt in attempts:
            distance = self._calculate_topology_distance(attempt.decoded_calls, ground_truth)
            if distance < min_distance:
                min_distance = distance
                best_attempt = attempt
        
        return best_attempt
    
    def _is_topology_match(self, decoded_calls: List[Any], ground_truth: List[Any]) -> bool:
        """检查拓扑是否匹配（简化版本）"""
        # 简化的拓扑匹配：比较调用数量和基本结构
        if len(decoded_calls) != len(ground_truth):
            return False
        # 这里可以扩展更复杂的AST匹配逻辑
        return True
    
    def _calculate_topology_distance(self, calls1: List[Any], calls2: List[Any]) -> float:
        """计算拓扑距离（使用编辑距离近似）"""
        # 简化的距离计算：基于字符串表示的编辑距离
        str1 = str(calls1)
        str2 = str(calls2)
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    
    def add_successful_anchor(self, task_signature: str, attempt: AttemptRecord) -> None:
        """将成功轨迹添加到历史锚点池"""
        if task_signature not in self.historical_anchor_pool:
            self.historical_anchor_pool[task_signature] = []
        self.historical_anchor_pool[task_signature].append(attempt)


class ASTRetrospectiveDiagnoser:
    """AST后见逻辑诊断器"""
    
    def __init__(self):
        pass
    
    def diagnose_first_divergence(self, attempt_calls: List[Any], 
                                ground_truth: List[Any]) -> Dict[str, Any]:
        """
        诊断第一逻辑分歧点
        
        Args:
            attempt_calls: 尝试的调用序列
            ground_truth: 地面真相调用序列
            
        Returns:
            Dict[str, Any]: 诊断结果
        """
        # 解析调用序列为抽象节点
        attempt_nodes = self._parse_calls_to_nodes(attempt_calls)
        gt_nodes = self._parse_calls_to_nodes(ground_truth)
        
        # 找到第一分歧点
        divergence_point = self._find_first_divergence(attempt_nodes, gt_nodes)
        
        return {
            "divergence_type": divergence_point["type"],
            "divergence_position": divergence_point["position"],
            "masked_feedback": self._generate_masked_feedback(divergence_point)
        }
    
    def _parse_calls_to_nodes(self, calls: List[Any]) -> List[Dict[str, Any]]:
        """解析函数调用为抽象节点"""
        nodes = []
        for call in calls:
            if isinstance(call, dict) and "name" in call:
                node = {
                    "tool_name": call.get("name", ""),
                    "tool_category": self._categorize_tool(call.get("name", "")),
                    "param_types": self._extract_param_types(call)
                }
                nodes.append(node)
        return nodes
    
    def _categorize_tool(self, tool_name: str) -> str:
        """工具分类（示例实现）"""
        # 这里可以根据工具名进行更复杂的分类
        if "get" in tool_name.lower() or "fetch" in tool_name.lower():
            return "data_retrieval"
        elif "create" in tool_name.lower() or "add" in tool_name.lower():
            return "data_creation"
        elif "update" in tool_name.lower() or "modify" in tool_name.lower():
            return "data_modification"
        elif "delete" in tool_name.lower() or "remove" in tool_name.lower():
            return "data_deletion"
        else:
            return "other"
    
    def _extract_param_types(self, call: Dict[str, Any]) -> List[str]:
        """提取参数类型"""
        param_types = []
        if "arguments" in call:
            args = call["arguments"]
            if isinstance(args, dict):
                for key, value in args.items():
                    param_types.append(f"{key}:{type(value).__name__}")
        return param_types
    
    def _find_first_divergence(self, attempt_nodes: List[Dict[str, Any]], 
                             gt_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """找到第一逻辑分歧点"""
        min_len = min(len(attempt_nodes), len(gt_nodes))
        
        for i in range(min_len):
            attempt_node = attempt_nodes[i]
            gt_node = gt_nodes[i]
            
            # 检查工具类别分歧
            if attempt_node["tool_category"] != gt_node["tool_category"]:
                return {
                    "type": "strategy_divergence",
                    "position": i,
                    "attempt_category": attempt_node["tool_category"],
                    "gt_category": gt_node["tool_category"]
                }
            
            # 检查参数类型分歧
            if attempt_node["param_types"] != gt_node["param_types"]:
                return {
                    "type": "parameter_divergence", 
                    "position": i,
                    "attempt_params": attempt_node["param_types"],
                    "gt_params": gt_node["param_types"]
                }
        
        # 如果长度不同
        if len(attempt_nodes) != len(gt_nodes):
            return {
                "type": "sequence_length_divergence",
                "position": min_len,
                "attempt_length": len(attempt_nodes),
                "gt_length": len(gt_nodes)
            }
        
        return {"type": "no_divergence", "position": -1}
    
    def _generate_masked_feedback(self, divergence_point: Dict[str, Any]) -> str:
        """生成掩码式因果归因报告（仅输出动作类别，不输出真实工具名）"""
        div_type = divergence_point["type"]
        
        if div_type == "strategy_divergence":
            return f"Strategy divergence at position {divergence_point['position']}: " \
                   f"used {divergence_point['attempt_category']} instead of {divergence_point['gt_category']}"
        elif div_type == "parameter_divergence":
            return f"Parameter divergence at position {divergence_point['position']}: " \
                   f"parameter types mismatch"
        elif div_type == "sequence_length_divergence":
            return f"Sequence length divergence: attempt has {divergence_point['attempt_length']} calls, " \
                   f"expected {divergence_point['gt_length']}"
        else:
            return "No logical divergence detected"
#######新增（结束）#######