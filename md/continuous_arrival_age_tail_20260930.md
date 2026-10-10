已移除策略的H/评价终点输入，累计率曝光使用now，规划窗口始终为τmax；真实到达持续至1200秒。固定评价300–900秒出生作业，未完成作业保留。5项针对性测试通过，24条开发对照完整执行，结果为 `CONTINUOUS_ARRIVAL_AGE_TAIL_DEVELOPMENT_HOLD`。

相对MW，1200秒时的出生群平均流时间33.805093→31.169992秒（−7.7950%），各轨迹共同群P90均值67.348690→59.025340秒（−12.3586%）；中段等待积压面积降低13.1634%。三种子汇总均改善，两个负载均改善；全部7283个评价作业在1200秒完成。实际作业时间戳独立重建与摘要完全一致。

HOLD原因：gpu_a、负载0.8在900秒的未完成数35→42（+20%，超过原5%门槛）。其中LLM为15→28；等待数17→16，已开工未完成数18→26。平均收益存在，但单一汇总S2约束未能保护此类截面尾部；未启动独立确认，未改门槛或论文。证据限于固定率、确定性缓存槽回放，执行钟不计计算延迟。

下一步：测试逐类别的预测S2约束，以同状态MW为每类基准，再选择总S1改善动作；检验LLM尾部代价能否减少，同时保留流时间增量。新规则先冻结开发协议，不把900秒评价时点输入策略。

代码：[持续到达选择器](/home/researcher/mine_code/scheduleurm/algorithm/continuous_arrival_age_tail_lane_rollout.py)。证据：[开发结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail_development_20260930.json)、[失败组及实际作业独立重建](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_continuous_arrival_age_tail_development_20260930_diagnostic.json)。
