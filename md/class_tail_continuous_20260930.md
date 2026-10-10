逐类别S2约束已实现并冻结验证：每类预测流时间平方和均不得超过同状态MW，再选总S1改善动作；未知H、在线率、8样本、128事件预算及原1%/5%门槛保持。5项针对性测试通过，24条开发对照完整执行，结果 `CLASS_TAIL_CONTINUOUS_DEVELOPMENT_HOLD`，不采纳。

平均流时间33.805093→33.544361秒，仅降低0.7713%；各轨迹P90均值67.348690→67.534750秒，上升0.2763%。900秒未完成数有两组越线：gpu_a低负载35→40，gpu_b低负载46→52。全部7283个评价作业在1200秒完成；实际作业时间戳独立重建与摘要完全一致。

机制：4789次评分中4669次（97.49%）仅MW根满足逐类约束，实际改变动作108次，上轮汇总S2版本为1148次；同带候选均值由31.169992恶化至33.544361秒。严格逐类S2削弱了流时间收益，也未保护实际截面未完成数。证据为三个开发种子的缓存回放；未启动独立确认、未修改门槛或论文。

下一步：回到汇总S2版本，测试按τmax短期窗口预测的逐类未完成数约束，直接对应此次失败指标；保持总S1严格改善，并冻结原预算。τmax来自服务目录，评价时点900秒不进入策略；不再放宽本次逐类S2阈值。

代码：[逐类别选择器](/home/researcher/mine_code/scheduleurm/algorithm/class_tail_continuous_lane_rollout.py)。证据：[开发结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_tail_continuous_development_20260930.json)、[动作过滤与实际作业重建](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_class_tail_continuous_development_20260930_diagnostic.json)。
