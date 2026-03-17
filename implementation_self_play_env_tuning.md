#######新增（开始）#######
# EnvTuningProject 自博弈改进方案渐进式实现文档

第一步：建立最小侵入的能力边界（新增独立模块，不改原有训练损失）。
- 新增 `env_tuning/self_play_temporal_compat.py`，将核心能力拆成三个可复用组件：
  - `SelfPlayAnchorSelector`：三级锚点选择（批次内同伴 -> 历史自我 -> GT-shadow 局部锚点）。
  - `ASTLogicalDiagnoser`：构建调用 AST、定位第一逻辑分歧点、输出掩码式归因提示。
  - `TemporalCompatibilityOrchestrator`：把“先收集后诊断再回注”的流程封装成一个代理入口。
- 这样做的目标是：主干 rollout 代码只需增加少量钩子，不需要修改损失函数与下游 batch 结构。

第二步：实现“运行期静默收集 + 批次后见诊断”。
- 保持第一轮 `asyncio.gather` 原样执行，先拿到完整轨迹。
- 以 `batch_data_id` 为分组单位，在同组不同 `rollout_offset` 中做对比：
  - 成功轨迹优先作为失败轨迹锚点。
  - 若无成功轨迹，回退到历史成功池。
  - 再无可用锚点时，使用 GT-shadow 策略选出“失败里最接近正确拓扑”的轨迹。
- 通过 AST 诊断获取失败轨迹的第一分歧点并生成掩码提示（不泄露具体工具名）。

第三步：实现“分歧提示注入 + 二次重生成”。
- 在 `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py` 的请求级 rollout 主流程里：
  - 第一轮 rollout 后调用 `build_retry_plan(...)` 生成失败样本的提示字典。
  - 重新通过 `_preprocess_prompt_to_async_rollout_requests(...)` 构造同批次请求。
  - 把提示写入对应失败轨迹（按 `(batch_data_id, rollout_offset)` 精确匹配）。
  - 再跑第二轮 `asyncio.gather(...)`，用重生成结果替换输出。
- 该机制在时间维度上兼容原框架，不影响后续 DataProto 打包与 PPO/GRPO 训练接口。

第四步：可解释性与可维护性增强。
- 在所有新增核心逻辑旁补充中文注释，说明输入输出和设计意图。
- 将新增能力通过 `env_tuning/__init__.py` 导出，便于未来在 reward、trainer 或离线分析脚本中复用。

第五步：渐进式迭代建议（建议按优先级逐步上线）。
- V1（当前版本）
  - 使用规则驱动的工具类别映射与参数模式比较，快速验证“二次重生成”收益。
- V2
  - 将 GT-shadow 的拓扑覆盖率从启发式改为真实 Tree Edit Distance（例如 zss/apted）。
  - 在 reward_scores 中增加 `self_play_feedback` 子字段，显式记录 `t_diverge` 与分歧类型。
- V3
  - 在 interaction/env 侧追加“掩码因果报告”消费逻辑，把提示作用到奖励重载：
    - `t < t_diverge` 保留原奖励。
    - `t = t_diverge` 强惩罚 + 注入报告。
    - `t > t_diverge` 置零。

第六步：如何验证 scripts 启动脚本保持可运行。
- 先做语法级验证：
  - `python -m py_compile env_tuning/self_play_temporal_compat.py verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
  - `bash -n scripts/run_multi_turn_fc_grpo_stage1.sh`
- 再做最小运行验证（建议在目标 conda 环境下）：
  - `conda run -p /mnt/whuscs/ckf/env/envtuning python -c "import env_tuning; from env_tuning import TemporalCompatibilityOrchestrator; print('ok')"`
- 最后做一次小 batch 训练冒烟（将步骤/数据量调小），观察日志中是否出现 retry hint 注入记录。

第七步：上线前检查清单。
- 检查 retry 触发率是否合理（避免全量样本都二次重生成导致吞吐下降）。
- 检查提示文本是否严格掩码（不包含真实工具名和答案字段）。
- 检查 OOM 风险（两次 rollout 的峰值内存与耗时开销）。
- 检查 reward 分布是否稳定（防止过度惩罚造成策略坍塌）。
#######新增（结束）#######
