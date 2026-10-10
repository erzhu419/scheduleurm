已实现汇总S2非劣＋逐类τmax预测未完成数非劣，再选总S1改善动作。保留未知H、在线率、8样本、128事件预算、300–900秒出生群和原性能门槛。5项针对性测试通过，24条开发对照完整执行；结果 `WINDOW_TAIL_CONTINUOUS_DEVELOPMENT_HOLD`，不采纳。

平均流时间33.805093→33.450361秒（−1.0493%），各轨迹P90均值67.348690→66.468360秒（−1.3071%）；但900秒评价群未完成数457→477，五个分组越线。gpu_a高负载的1200秒P90回退8.4962%，超过5%门槛。全部7283个评价作业最终完成，实际作业时间戳独立重建与摘要完全一致。

机制：4711次评分中4109次（87.22%）仅MW根可行，改变动作319次，上轮原汇总S2为1148次。预测风险比较均通过，但真实尾部更差；两次更强代理约束均显著削弱收益而未修复尾部。不调整阈值、不启动独立确认，论文未改；本轮证据仍限于三个开发种子的固定率缓存回放，计算延迟未进入服务钟。

下一步：停止追加硬代理阈值，回到原汇总S2策略，做同到达带的单动作替换闭环反事实。先定位哪些根选择真实推迟了LLM完成、哪些保留流时间收益，再据此设计新规则；未来到达仅用于离线诊断，不进入在线选择器。

代码：[窗口尾部选择器](/home/researcher/mine_code/scheduleurm/algorithm/window_tail_continuous_lane_rollout.py)。证据：[开发结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_window_tail_continuous_development_20260930.json)、[失败组与实际作业重建](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_window_tail_continuous_development_20260930_diagnostic.json)。
