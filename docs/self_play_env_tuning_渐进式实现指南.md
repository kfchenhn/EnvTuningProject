# EnvTuningProject 自博弈增强方案：渐进式迭代实现指南

第一步：完成“最小可运行骨架”改造（先不追求复杂功能）  
1. 在 `env_tuning/self_play/` 新建三个基础模块：
   - `anchor_selector.py`：实现三级锚点队列（同批成功 > 历史成功 > 课程诱导）。
   - `ast_diagnostics.py`：实现轨迹标准化、AST/结构对齐、第一逻辑分歧点定位。
   - `rewarding.py`：实现双维验证器（状态维度 + 响应维度）。
2. 在 `env_tuning/self_play/__init__.py` 暴露核心类，确保训练主流程可直接 import。  
3. 验收标准：
   - 模块能被 Python 正常导入；
   - 输入两条轨迹，能返回结构化分歧报告；
   - 输入当前 batch 轨迹，能返回 anchor 来源与对象。

第二步：将“双维验证器”接入现有评分链路  
1. 修改 `env_tuning/interaction/score_calculator.py`：
   - 使用 `DualOutcomeValidator` 统一状态校验与响应校验；
   - 回合得分由两维乘积（都正确才得 1）决定。
2. 设计日志字段：建议记录 `state_ok`、`response_ok`、`reward`，用于定位错误模式。  
3. 验收标准：
   - 与旧逻辑相比，不影响已有运行；
   - 评分更可解释，可区分“状态对但响应错”等情况。

第三步：落地“AST后见诊断 + 反事实报告”离线产物  
1. 在 rollout 后处理环节新增轨迹配对：
   - 以 `task_signature` 分组；
   - 失败轨迹优先匹配同 batch 成功锚点；否则回放池检索历史锚点。  
2. 调用 `ASTTrajectoryDiagnostic.build_counterfactual_report` 生成报告，写入离线样本。  
3. 反事实样本建议字段：
   - `task_signature`
   - `failed_trajectory`
   - `anchor_trajectory`
   - `first_divergence_step`
   - `divergence_type`
   - `success_action`
   - `failed_action`
   - `explanation_template`
4. 验收标准：
   - 每个失败样本尽量能匹配锚点；
   - 报告中能稳定定位“第一逻辑分歧点”。

第四步：实现“通道B在线挽救”与“通道A离线内化”闭环  
1. 通道B（在线）：
   - 当执行报错或连续低分时，注入合规提示（课程诱导锚点）；
   - 成功跑通后将轨迹回灌回放池，标记 `source=curriculum`。  
2. 通道A（离线）：
   - 使用分歧报告构造偏好对（正确动作 > 分歧点错误动作）；
   - 将偏好信号映射到 GRPO 优势估计（正样本正优势，分歧点负优势）。
3. 验收标准：
   - 随训练推进，通道B触发频率下降；
   - 通道A持续提升验证集多轮成功率。

第五步：完善四阶段课程与配置脚本  
1. 阶段一（格式重塑）：仅保留格式/工具名约束奖励。  
2. 阶段二（冷启动引导）：启用细粒度进度奖励与强提示。  
3. 阶段三（自博弈内化）：开启锚点队列 + AST诊断 + 双通道。  
4. 阶段四（鲁棒实战）：关闭提示增强，注入噪声与步数惩罚。  
5. 代码层执行：
   - 新增 `env_tuning/config/multi_turn_fc_grpo_stage4.yaml`；
   - 新增 `scripts/run_multi_turn_fc_grpo_stage4.sh`；
   - 与 stage1/2/3 串联形成完整训练流水线。

第六步：实验与消融的标准化执行顺序（建议固定流程）  
1. 主实验：Stage1→Stage2→Stage3→Stage4 全流程。  
2. 消融一：去掉 AST 诊断，仅字符串对齐。  
3. 消融二：去掉历史锚点，仅同批锚点。  
4. 消融三：关闭通道B，仅通道A。  
5. 消融四：降低 KL 系数（验证稳定性劣化）。  
6. 评估集：BFCL V3 + OOD（如 ACEBench 类场景）。

第七步：工程化与可观测性加固（上线前必须做）  
1. 监控指标：
   - `anchor_source_ratio(peer/history/curriculum)`
   - `first_divergence_step` 分布
   - `state_ok_rate` / `response_ok_rate`
   - 通道B触发率
   - 平均调用步数与冗余步数
2. 失败回放：固定采样失败轨迹与诊断报告，做周报复盘。  
3. 阈值告警：若通道B触发率回升或 history 锚点命中下降，自动告警。

第八步：你可以直接照着执行的“最短落地清单”  
1. 先跑 Stage1/2，确认格式合规和基础调用稳定；  
2. 开启 Stage3，自博弈模块全部打开，观察 anchor 命中与分歧报告质量；  
3. 若 Stage3 稳定，再跑 Stage4 做抗噪与 OOD 强化；  
4. 每阶段至少保留一次 checkpoint 做回退点；  
5. 训练后产出三份报告：主结果、消融、错误案例库。

> 建议：先确保“能稳定训练”再追求“最高分数”。该方案的核心收益来自可持续迭代闭环，而不是单次大改动。
