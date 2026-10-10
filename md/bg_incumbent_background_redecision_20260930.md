# BG 动作基准前瞻：开发结果（2026-09-30）

共享外层 8 条到达样本，先确定原 BG 动作及 S2 可行集，再按一次 BG 续排评分严格改善该动作。原阶段溢出回退 MW，前瞻阶段溢出保留 BG；两阶段费用分别计量，事件上限仍为 128。9 项新测试首轮通过，24 对场景、48 条新执行及全部执行检查通过。

| 指标 | 原 BG | 本次候选 | 变化 |
|---|---:|---:|---:|
| T=1200 限制平均流时间 | 31.1614 | 30.4593 | −2.253% |
| T=1200 共同完成任务 P90 | 58.2741 | 55.0882 | −5.467% |
| 300–900 等待面积 | 133281.66 | 126035.28 | −5.437% |
| 900 未完成数 | 427 | 419 | −8 |
| 1200 未完成数 | 0 | 0 | 持平 |

**HOLD，保留原 BG；本候选继续开发。** 低/高负载均值分别下降 1.837%/2.498%，总均值、分负载及合计尾部门槛通过。剩余失败都是 900 时刻的局部尾部：相对 BG，GPU/.8 为 35→39、Linux/.95 为 29→35；相对 MW，另有 node007/.95 为 100→107。

实际改善 BG 动作 320 次；前瞻溢出保留 BG 的情况实际发生 1 次。新增 BG 根评分计入 27374741 次选项探测，全部阶段合计 1058366926 次。24 个类别组中 7 组均值退化；Linux/.95 新增 6 个未完成均来自 LLM，而该类均值/P90仍改善，故尾部损失不能统一归因于类别总成本增加。

下一步冻结全部 320 次替换的动作名单，逐次换回同状态 BG 动作，后续仍由本候选自由重规划，配对检查流时间与尾部损失；以此定位前瞻估值偏差，再修改选择规则。保留全部结果，不按失败节点/类别挑动作，不相加彼此依赖的干预效果。

局限：开发种子仍为 41–43，无独立确认；一次 BG 后采用 MW 的续排近似尚未覆盖完整在线策略。两组采样 S2 约束不保证真实轨迹改善，也不保证旧评分 S1 不劣于 MW；计算费用未进入服务时钟。

产物：[结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_bg_incumbent_background_redecision_development_20260930.json)、[冻结协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_bg_incumbent_background_redecision_development_20260930_protocol.json)、[类别汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_bg_incumbent_background_redecision_development_20260930_class_summary.json)。论文未改。
