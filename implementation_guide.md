# 基于自博弈的工具调用改进方案实现文档

第一步：添加最小侵入开关配置参数
在 `env_tuning/config/multi_turn_fc_interaction_config.yaml` 中添加了三个新参数：
- `enable_blind_rollout_feedback: true`（运行期静默收集，关闭可操作增强反馈）
- `enable_temporal_compat: true`（启用后见重载）
- `max_step_limit: 5`（保留原本上限机制）

第二步：修改数据模型以支持轨迹记录
在 `env_tuning/interaction/data_models.py` 的 `InstanceState` 类中新增 `single_turn_attempt_records` 字段，用于保存同一轮内每次工具调用尝试的记录（decoded_calls、execution_results、has_error）。同时修改 `reset_single_turn_buffers` 方法，在重置时清空该记录。

第三步：实现盲盒反馈模式
在 `env_tuning/interaction/execution_manager.py` 的 `format_execution_response` 方法中增加 `blind_mode` 参数。当 `blind_mode=true` 时，只返回执行结果和继续/结束格式约束，不返回可操作增强提示。同时在 `execute_function_calls` 方法中记录每次尝试到 `single_turn_attempt_records`。

第四步：集成配置开关
在 `env_tuning/interaction/new_multi_turn_fc.py` 的构造函数中读取配置参数并传递给 `TurnManager`。在 `_determine_next_action` 中传入 `blind_mode` 开关。

第五步：创建自博弈反馈核心模块
新建 `env_tuning/interaction/self_play_feedback.py`，实现：
- `ToolCallNode` 类：抽象语法树节点，包含工具名、类别、参数类型
- `build_tool_call_ast` 函数：将解码调用转换为AST
- `calculate_ast_distance` 函数：计算AST距离（近似树编辑距离）
- `SelfPlayAnchorSelector` 类：三级优先级锚点选择器
- `ASTRetrospectiveDiagnoser` 类：后见逻辑诊断器

第六步：实现时序兼容架构
在 `env_tuning/interaction/turn_manager.py` 中：
- 修改构造函数接收开关参数
- 在 `advance_to_next_turn` 中添加后见重载调用
- 实现 `_retrospective_temporal_remap` 方法：选择锚点、AST诊断、时序映射奖励重载

第七步：测试和验证
- 验证所有模块导入正常
- 确保向后兼容（开关默认关闭）
- 确认启动脚本语法正确</content>
<parameter name="filePath">/mnt/whuscs/ckf/Tool-MT/EnvTuningProject/implementation_guide.md