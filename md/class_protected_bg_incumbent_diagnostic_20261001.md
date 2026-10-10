# 逐类保护诊断（2026-10-01）

6 项新测试通过；57 次执行完成，包含 24 条完整缓存复现、13 次恢复对照和全部 20 次 BG 单动作恢复。另对原 320 个冻结状态重建随机流并复评分，全部原 S1/S2 及状态核验通过，复用原反事实效果，没有新增轨迹执行。

**关闭逐类硬约束开发分支，保留原 HOLD 与普通 BG。** 本次 20 个原动作对真实均值有益 6 次、有害 7 次、无变化 7 次；5 次恢复 BG 同时改善均值和 900 秒总尾部。7 个晚于 900 秒的动作对所测指标全部无影响。

原 320 个状态的同模型逐类筛选结果如下。数值是原动作对真实均值的条件效应；近零为绝对差≤1e−9 秒，原始数值与精确符号均保留。

| 逐类约束 | 状态数 | 原动作有益 | 原动作有害 | 近零 |
|---|---:|---:|---:|---:|
| 放行 | 20 | 11 | 4 | 5 |
| 拦截 | 300 | 138 | 81 | 81 |

硬约束拦掉 138/149 个有益动作，仍放过 4 个有害动作，不能支撑所需性能增量。

GPU/.8 的两个动作均预测每类 S2 改善。下表差值为“单独恢复 BG − 原候选”；两次恢复都减少 4 个 LLM 尾部作业，但总体效果不同。

| 原动作 → BG | 时刻 | 均值差（秒） | 900 总尾部差 | LLM 尾部差 |
|---|---:|---:|---:|---:|
| LLM3 → JAX1 | 131.137 | +0.164225 | −4 | −4 |
| LLM1 → CNN1 | 525.096 | −0.278702 | −3 | −4 |

下一步回到未加逐类硬约束的 BG incumbent 分支，只将虚拟续排从一次普通 BG 改为固定两次、随后 MW；延续因果到达计数、内层随机流与空闲调用计数，完整计入额外费用，沿用原 MW/BG 联合开发门槛。

限制：复用开发种子 41–43；两批状态与后续策略不同，单动作效果不能相加或作为独立样本。本轮没有建立新策略收益，也未单独证明偏差来自 MW 续排；第二次 BG 续排是待检验的性能假设。

[20 次恢复结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_bg_incumbent_counterfactual_20261001.json) · [恢复协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_bg_incumbent_counterfactual_20261001_protocol.json) · [320 状态筛查](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_original_root_screen_20261001.json) · [筛查协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_original_root_screen_20261001_protocol.json)。论文未改。
