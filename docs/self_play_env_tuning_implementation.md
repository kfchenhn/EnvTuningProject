# 基于 EnvTuningProject 实现“自博弈 + AST 诊断 + 双通道提示注入”方案（渐进式迭代指南）

本文档按“第一步、第二步……”给出可落地实施路径，目标是在现有 EnvTuning 四阶段课程框架上，逐步演进到你提出的改进方案。

## 第一步：对现有系统建立 POMDP 视角与模块边界映射

1. 明确状态 `s_t`：`InstanceState`（当前轮次、尝试次数、执行历史、环境实例快照）。
2. 明确观测 `o_t`：模型消息、上轮执行结果、系统报错提示。
3. 明确动作 `a_t`：
   - 工具调用（`<tool_call></tool_call>`）
   - 最终回答（`<answer></answer>`）
4. 明确奖励 `r_t`：
   - 阶段1：格式奖励（`format_reward.py`）
   - 阶段2/3：任务进度奖励（`bfcl_reward.py`）
5. 映射到代码：
   - 交互主循环：`env_tuning/interaction/new_multi_turn_fc.py`
   - 执行管理：`execution_manager.py`
   - 回合推进：`turn_manager.py`
   - 评分核验：`score_calculator.py`

> 交付检查：跑通 stage1/2/3 脚本，确认当前训练流水线可复现。

## 第二步：先补“可插拔自博弈基础设施”，不改主训练逻辑

新增 `env_tuning/self_play/` 子模块，先以“旁路”形式接入：

1. `data_models.py`
   - 定义 `Trajectory`、`ToolCallNode`、`DivergencePoint`、`CounterfactualSample`。
2. `replay_buffer.py`
   - 维护按任务签名分桶的历史成功轨迹池。
3. `anchor_selector.py`
   - 实现三级优先级锚点策略：
     - 批次内同伴锚点
     - 历史回放锚点
     - 课程诱导锚点
4. `ast_diagnostics.py`
   - 对工具调用序列进行标准化：字段去噪、参数排序、并行调用序列规范化。
   - 计算“第一逻辑分歧点”（策略分歧/参数分歧/长度分歧）。
5. `dual_channel.py`
   - 实现通道B干预频率退火调度器（阶段3指数衰减，阶段4关闭）。

> 交付检查：这些模块可被 import，且有最小单元用例可运行（见第六步）。

## 第三步：在交互态中挂载“自博弈轨迹缓存”

修改 `InstanceState`：

1. 增加 `self_play_counterfactuals`，用于存储反事实偏好样本。
2. 增加 `latest_failed_trajectory` / `latest_anchor_trajectory`，用于在线对比诊断。

实现原则：
- 不破坏原有字段语义。
- 所有新增字段使用默认值，保证旧数据兼容。

> 交付检查：旧训练配置无需改动也能启动，不因新增字段报错。

## 第四步：把“AST后见诊断”嵌入在线交互回路

修改 `new_multi_turn_fc.py`：

1. 初始化自博弈组件：
   - `SelfPlayReplayBuffer`
   - `AnchorSelector`
   - `ASTTrajectoryDiagnostics`
   - `DualChannelScheduler`
2. 在工具执行后记录当前轨迹（成功/失败）。
3. 对失败轨迹自动选锚点，触发 AST 分歧诊断，写入 `self_play_counterfactuals`。
4. 失败时触发通道B提示注入（当前版本为轻量模板，可后续替换为 LLM 生成+宪法裁判过滤）。

> 交付检查：交互时遇错不再只返回报错，能额外产生 channel-B 提示与对比样本缓存。

## 第五步：把“奖励核验”升级为双维度细粒度验证器

修改 `score_calculator.py`：

1. 新增 `evaluate_turn_dimensions`：
   - `state_score`：最终环境状态是否与 GT 一致。
   - `response_score`：执行返回结果是否满足查询需求。
   - `reward = state_score * response_score`。
2. `calculate_turn_score` 调用该双维度逻辑统一产出。

> 交付检查：每轮奖励仍兼容原先 0/1 语义，同时具备维度可解释性。

## 第六步：实现“最小可验证”本地测试闭环

建议新增（或本地临时）测试脚本验证三件事：

1. AST 诊断是否能忽略无意义字段并定位第一分歧。
2. 锚点队列是否遵循“批次 > 历史 > 课程”的选择优先级。
3. 双通道调度是否在 stage3 衰减、stage4 归零。

本次实现至少应执行：
- Python 语法编译检查。
- 关键模块 import 检查。

## 第七步：把双通道参数显式化到配置层

修改 `env_tuning/config/multi_turn_fc_interaction_config.yaml`：

1. `training_stage`
2. `self_play_replay_size`
3. `channel_b_decay_rate`
4. `channel_b_min_prob`

这样可配合四阶段课程做调度：
- 阶段2：高依赖提示（可设 stage=2, hint_prob=1）
- 阶段3：动态衰减
- 阶段4：完全关闭

## 第八步：将反事实样本接入 GRPO（下一迭代）

当前代码已完成“样本生产”，下一步建议：

1. 在 rollout 收集结束后，将 `self_play_counterfactuals` 序列化到 batch extra_info。
2. 在 GRPO loss 处引入 divergence-node 额外优势项：
   - 锚点动作正向优势
   - 第一分歧失败动作负向优势
3. 与现有 KL/clip 机制联合训练，维持稳定性。

> 超参建议（与你方案一致）：
- `clip_ratio_low=0.2`
- `clip_ratio_high=0.28`
- `kl_loss_coef=0.1`

## 第九步：按四阶段课程实施渐进式上线

1. **阶段一（格式重塑）**
   - 关闭自博弈诊断与提示增强。
2. **阶段二（真值冷启动）**
   - 启动通道B强提示，积累第一批成功锚点。
3. **阶段三（自博弈内化）**
   - 全开 AST 诊断 + 锚点选择 + 反事实样本生产。
   - 对通道B执行指数退火。
4. **阶段四（鲁棒实战）**
   - 关闭通道B，仅保留内化策略。
   - 加入环境噪声与冗余步数惩罚做 OOD 强化。

## 第十步：实验与验收标准

建议设置如下验收门槛：

1. 训练稳定性：无频繁梯度爆炸/坍塌。
2. 过程指标：
   - 平均每任务可提取 counterfactual 样本数
   - 第一分歧点定位成功率
   - 通道B使用率随阶段下降曲线
3. 结果指标：
   - BFCL V3 成功率
   - OOD 数据集（BFCL V4/ACEBench）提升
4. 代价指标：
   - 训练吞吐降低幅度
   - 额外显存占用

---

## 本次代码实现范围说明

本次已完成：
- 自博弈基础模块（锚点选择、AST诊断、回放池、双通道调度）。
- 交互层最小接入（在线记录轨迹 + 失败诊断 + 轻量提示注入）。
- 奖励层双维度验证接口。
- 配置层新增可调参数。

本次未完成（建议下个迭代）：
- 将反事实样本直接并入 `verl` 的 GRPO loss 计算图（需要深入改造 trainer/advantage 计算链路）。
- “增强生成器 + 宪法裁判”的独立 LLM 管道化服务。
