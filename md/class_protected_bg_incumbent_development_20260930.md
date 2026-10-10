# BG 逐类保护：开发结果（2026-09-30）

已实现同 once-BG 模型下每类 S2≤BG incumbent，保留原 MW 总 S2 可行集、总 S1 目标及 8/128 预算；新增逐类费用记录。10 项测试、24 对场景的 48 条执行及全部回放检查通过。

| 指标 | 普通 BG | 逐类保护 | 变化 |
|---|---:|---:|---:|
| 1200 限制平均流时间 | 31.1614 | 31.0777 | −0.269% |
| 1200 共同完成任务 P90 | 58.2741 | 57.6830 | −1.014% |
| 300–900 等待面积 | 133281.66 | 131847.05 | −1.076% |
| 900 未完成数 | 427 | 418 | −9 |
| 1200 未完成数 | 0 | 0 | 持平 |

**HOLD，保留普通 BG。** 均值增量未达 1%；GPU/.8 的 900 尾部仍为 35→39，超过 5% 门槛，新增四项均为 LLM。Linux/.95 的 29→30、node007/.95 相对 MW 的 100→101 已回到门槛内。相对 MW 均值下降 8.068%，两套联合门槛仍未通过。

本次轨迹上，逐类检查删除 1711 个可行替代动作中的 1682 个（98.305%）；4591 个完成评分状态中 4563 个只剩 BG，仅发生 20 次严格改善。上一版候选均值为 30.4593，本次退步 2.030%；两版轨迹不同，320→20 仅作描述性比较。实际 CNN/LLM 类均值分别退化 0.671%/0.367%，JAX 改善 2.417%，逐类采样保护仍存在真实类别损失。

已冻结全部 20 次替换，下一步逐次恢复同状态 BG 动作、后续自由重规划，定位 GPU/.8 的尾部损失；再把原 320 个冻结状态的新逐类分解与已有单动作反事实配对，判断哪些有益或有害动作被拦截，优先修前瞻估值偏差。

限制：复用开发种子 41–43，无独立确认；一次 BG 后采用 MW 的续排近似仍未覆盖完整在线策略，计算费用不进入服务时钟。

[结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_bg_incumbent_development_20260930.json) · [冻结协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_bg_incumbent_development_20260930_protocol.json) · [逐类汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_bg_incumbent_development_20260930_class_summary.json) · [全部 20 个动作](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_protected_bg_incumbent_development_20260930_selection_summary.json)。论文未改。
