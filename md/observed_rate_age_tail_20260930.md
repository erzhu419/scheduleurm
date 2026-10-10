已用累计已观察到达数除以 `min(now,H)` 替代真实到达率；累计数包含已服务作业，零曝光取零率。FIFO年龄、S2约束、8条规划样本、128事件预算和原性能门槛均保持，无调参。论文未改。

| 配对证据 | T平均流时间：MW→候选 | T各轨迹P90均值：MW→候选 | H未完成：MW→候选 |
|---|---|---|---|
| 开发：24条、3个种子 | 30.939790→28.641162秒（−7.4294%） | 62.545455→54.137866秒（−13.4424%） | 410→376 |
| 独立确认：96条、12个新种子 | 28.280042→26.334922秒（−6.8781%） | 55.024169→48.564648秒（−11.7394%） | 1628→1450 |

确认在北京时间08:49:21冻结，使用10001–10012，不复用已看过的1001–1012。结果为 `OBSERVED_RATE_AGE_TAIL_CONFIRMATION_PASS`；16个节点/负载/H,T组均通过，两个负载均值改善5.2110%/8.0007%，12个种子汇总均改善。25项针对性测试通过；192条主轨迹与96条旧控制参考完整执行。独立从实际作业时间戳重建指标，与摘要数值完全一致；每策略28332个作业均保留，T均全部完成。

限制：当前仍是公开H、固定率Poisson、目录报告2/4槽的确定性缓存回放。确认中8/96条轨迹均值、18/96条轨迹P90及7/24个类别均值回退，主要类别回退为Linux LLM与node007 CNN，均保留。有限样本S2不保证实际P90；候选目录探查57093052次、MW145064次（393.57倍），执行钟未计计算延迟。

下一步：移除策略对结束时点H的预知，在持续到达下验证性能增量；固定中段出生作业群，继续比较流时间、P90、未完成数与积压，保留现有规则和预算。

代码：[在线估计选择器](/home/researcher/mine_code/scheduleurm/algorithm/observed_rate_age_tail_lane_rollout.py)、[确认驱动](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_observed_rate_age_tail_confirmation.py)。证据：[确认结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_observed_rate_age_tail_confirmation_20260930.json)、[类别与估计误差](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_observed_rate_age_tail_confirmation_20260930_mechanism.json)、[实际作业独立重建](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_observed_rate_age_tail_confirmation_20260930_independent_job_rebuild.json)。
