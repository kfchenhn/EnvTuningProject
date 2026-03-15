# 基于 EnvTuningProject 的“自博弈锚点 + AST 后见诊断 + 最小侵入时序兼容”实现指南

> 目标：在尽量不改动原有训练主流程的前提下，把你提出的方案接入现有 `MultiTurnFunctionCallInteraction`。

## 第一步：先打开“最小侵入开关”，不改训练主循环

1. 在交互配置里增加三个参数：
   - `enable_blind_rollout_feedback: true`（运行期静默收集，关闭可操作增强反馈）
   - `enable_temporal_compat: true`（启用后见重载）
   - `max_step_limit`（保留原本上限机制）
2. 在 `MultiTurnFunctionCallInteraction` 构造函数读取这三个参数并传给 `TurnManager`。

**对应代码**：`env_tuning/config/multi_turn_fc_interaction_config.yaml`、`env_tuning/interaction/new_multi_turn_fc.py`。

---

## 第二步：把“轨迹回放池”先做成单轮轻量缓存

1. 在 `InstanceState` 中新增 `single_turn_attempt_records`，用于保存同一轮内每次工具调用尝试：
   - `decoded_calls`
   - `execution_results`
   - `has_error`
2. 在每轮 reset 时清空该缓存，保证与原多轮结构兼容。

**为什么这么做（最小侵入）**：
- 不改动上层采样与 GRPO 流程。
- 不引入新的存储系统，先使用进程内缓存验证方案有效性。

**对应代码**：`env_tuning/interaction/data_models.py`。

---

## 第三步：运行期切换到“盲盒反馈”

1. 在 `ExecutionManager.format_execution_response` 增加 `blind_mode` 参数。
2. 当 `blind_mode=true` 时，只返回执行结果 + 继续/结束格式约束，不返回可操作增强提示。
3. 在 `new_multi_turn_fc.py` 调用处传入 `blind_mode=self.enable_blind_rollout_feedback`。

**效果**：
- rollout 阶段仍可继续交互；
- 但不会泄露“下一步怎么做”的强提示，符合你方案中的“关闭增强反馈”。

**对应代码**：`env_tuning/interaction/execution_manager.py`、`env_tuning/interaction/new_multi_turn_fc.py`。

---

## 第四步：记录每一次尝试，给后见诊断准备输入

1. 在 `execute_function_calls` 执行成功返回前，把本次尝试写入 `single_turn_attempt_records`。
2. 这样你就有了“批次内（同轮内）多轨迹”的原始素材，可用于优先级一锚点选择。

**对应代码**：`env_tuning/interaction/execution_manager.py`。

---

## 第五步：新增“自博弈锚点智能选择器”模块

新建 `env_tuning/interaction/self_play_feedback.py`，实现：

1. **优先级一：In-batch Peer Anchor**
   - 在当前轮尝试中，若存在与 GT 拓扑一致的成功轨迹，直接作为锚点。
2. **优先级二：Historical Self Anchor**
   - 使用 `task_signature(entry_id + turn + question)` 作为键，维护历史成功锚点池。
   - 当前轮失败时从历史中选拓扑距离最近的锚点。
3. **优先级三：GT-Shadow Partial Anchor**
   - 若前两级不可用，则从失败轨迹中选与 GT AST 拓扑距离最近的“局部优胜”轨迹。
   - 极端情况下回退到“仅用 GT 拓扑做比较，不进模型上下文”。

**对应代码**：`env_tuning/interaction/self_play_feedback.py`。

---

## 第六步：新增“AST 后见逻辑诊断器”模块

同一文件中新增 `ASTRetrospectiveDiagnoser`：

1. 把函数调用字符串解析成抽象节点（工具名、工具类别、参数类型）。
2. 用结构签名序列做距离计算（编辑距离近似树编辑距离）。
3. 找到第一逻辑分歧点：
   - 工具不一致 -> `strategy_divergence`
   - 工具一致但参数类型不一致 -> `parameter_divergence`
4. 输出“掩码式因果归因报告”：
   - 仅输出动作类别，不输出真实工具名。

**对应代码**：`env_tuning/interaction/self_play_feedback.py`。

---

## 第七步：把后见诊断嵌入 TurnManager（关键最小改造点）

1. 在 `TurnManager.advance_to_next_turn` 保留原有流程：
   - flush
   - 读取 GT
   - 计算基础分
2. 在不改变函数签名的前提下，额外调用 `_retrospective_temporal_remap(...)`：
   - 选择锚点
   - 进行 AST 诊断
   - 返回重载后的 `score` 与 `extra["retrospective_feedback"]`
3. 成功轨迹写入历史池，失败轨迹输出：
   - `first_logical_divergence_point`
   - `divergence_type`
   - `masked_causal_report`
   - `temporal_mapping` 说明

**对应代码**：`env_tuning/interaction/turn_manager.py`。

---

## 第八步：实现“最小侵入式时序兼容架构”的奖励重载

当前实现采用“接口不变 + 分数重载”方式：

1. rollout 期间仍按原交互节奏逐步执行。
2. turn 结束时集中做后见评估。
3. 用 remap 后分数替代原分数（成功 `1.0`，失败重罚）。
4. 将“分歧点报告 + temporal mapping”放入 `extra`，供训练侧日志/分析系统消费。

> 这是最小侵入版本：不修改 RL loss，不修改上游 sampler，不修改数据协议。

---

## 第九步：渐进式迭代建议（从可跑到最优）

### 9.1 V1（本次已实现）
- 单进程历史池（内存）。
- 结构签名 + 编辑距离近似 TED。
- turn 级重载分数。

### 9.2 V2（推荐下一步）
- 将历史锚点池持久化到磁盘/Redis，支持多 worker 共享。
- 把“工具类别映射”替换为基于工具 schema 的自动分类器。
- 将参数比较从“类型级”升级到“语义槽位级”。

### 9.3 V3（对齐你论文中的完整形态）
- 在训练器 batch hook 处统一做“批次内多轨迹锚点选择”。
- 把当前编辑距离近似替换为真正树编辑距离（TED）并加入依赖边。
- 输出 token/step 级 reward patch，实现更细粒度的 `t_diverge` 前后奖励覆盖。

---

## 第十步：如何验证每一步是否生效

1. 打开交互日志，确认 rollout 反馈不再包含可操作提示。
2. 触发失败样本，检查 `extra["retrospective_feedback"]` 是否包含：
   - `anchor_type`
   - `first_logical_divergence_point`
   - `masked_causal_report`
3. 观察训练曲线：
   - 早期 reward 方差应更大（重载惩罚生效）
   - 中后期 failure 中的 topology_score 逐步提升
4. 对同一任务重复采样，确认锚点来源能从 `gt_shadow_partial` 逐步迁移到 `historical_self` / `in_batch_peer`。

---

## 第十一步：落地注意事项

1. 目前 `Historical Self Anchor` 为进程内缓存；分布式训练需共享存储。
2. 当前“AST”是函数调用抽象结构，不含完整依赖图；后续可补依赖边。
3. 如果你要严格实现“t_diverge 前保留原奖励、之后置零”的逐 step 覆盖，需要在上游 reward aggregator 增加 patch 接口；本次改造以最小侵入优先，先提供 turn 级重载和结构化诊断。

---

## 第十二步：建议的实施顺序（真正的迭代执行清单）

1. 先合并本次最小改造（可运行）。
2. 在验证集上收集失败案例，评估掩码报告可读性。
3. 增加 batch hook，升级为真正“批内多轨迹”锚点。
4. 升级 TED 与依赖拓扑建模。
5. 最后再做 step 级 reward patch，完成你方案中的“时光倒流映射”完整版。

---

以上步骤能确保你不是“一次性重写系统”，而是按你要求的“渐进式、迭代式、最小代码侵入”路径稳定落地。


## 补充优化（针对首版实现的完善）

1. **AST 规范化增强**：比较前对关键字参数按键排序，并把“参数类型 + 依赖存在性”纳入结构签名，减少仅因参数顺序导致的伪分歧。
2. **时序重载信号可消费化**：`retrospective_feedback` 中新增 `base_score`、`remapped_score` 与 `reward_patch` 字段，便于上层训练日志或后续 reward aggregator 直接接入。
3. **交互生命周期修复**：修复解析错误分支下 `finalize_interaction` 使用错误 ID 的问题，确保实例能正确回收。
4. **类型兼容性增强**：锚点选择器对 attempt 记录采用属性访问兼容，避免因数据类类型差异导致运行时耦合。


## 第九步补充实现（已落地）

本次代码已将第九步中的三个点直接实现：

1. **历史锚点池持久化（磁盘/Redis）**
   - 新增 `PersistentAnchorStore`：
     - `backend=auto|file|redis`
     - `SELF_PLAY_ANCHOR_REDIS_URL` 或配置中的 `anchor_store_redis_url` 可开启 Redis 共享。
     - 文件后端默认 `anchor_store_file_path=/tmp/env_tuning_self_play_anchor_store.jsonl`。
   - `TurnManager` 初始化时注入该 store，所有成功锚点写入持久化池并可跨 worker 读取。

2. **工具类别映射升级为 schema 自动分类**
   - 新增 `ToolSchemaClassifier`，运行时从 `state.involved_classes` 自动提取 tool 名称与描述。
   - 分类时综合“工具名 + schema 文本”，替代纯名称关键词硬编码。

3. **参数比较升级为语义槽位级**
   - AST 节点中不再仅保留 `type`，而是保留 `semantic_slot_signature`：
     - 形如 `<arg_name>:<slot_type>:<value_semantics>`。
   - `slot_type` 由参数名语义推断（identity/time/location/numeric/state/text/generic）。
   - `value_semantics` 由参数值 AST 推断（symbol_ref/attr_ref/url_string/datetime_like 等）。
   - 诊断阶段按语义槽位签名比较，分歧定位更贴近真实错误类型。
