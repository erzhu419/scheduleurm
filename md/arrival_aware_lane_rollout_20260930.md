本轮已实现已知到达率的异步规划选择器，并完成24组配对实验、48条新轨迹和24个固定状态的独立排序诊断。论文未改。

每次低队列决策枚举所有可行非空根动作，用8条独立Poisson规划样本共同评估；规划窗为最长完整服务周期并受公开H停止时刻限制，随后普通MaxWeight完整排空。每条样本最多128个未来到达，超限整次回退，不截断后评分。本轮零溢出。模型、字面到达带、规则、预算和1%/5%门槛在北京时间2026-09-30 07:54:52冻结，结果读取后未调整。

| 开发结果 | 普通MaxWeight | 新候选 |
|---|---:|---:|
| T平均流时间 | 30.939790秒 | 29.437911秒（改善4.8542%） |
| T共同作业P90的逐带平均 | 62.545455秒 | 59.613707秒（改善4.6874%） |
| H未完成作业 | 410 | 371 |
| T未完成作业 | 0 | 0 |

22/24条带的T均值改善，8个节点/负载组均值均改善；3个独立开发种子的汇总分别改善4.2184%、3.1757%、7.8208%。全部普通MaxWeight动作/作业时间复现、选择与预算检查通过，38项针对性测试通过。总目录探查含预处理和idle续跑：控制34720，候选13550569；流时间改善伴随390.28倍探查量，没有计算成本优势结论。

完整性能门槛仍为 **ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_HOLD**，两处失败保留：

- node007、负载.95：T组P90从86.569298增至98.418402秒（+13.6874%）。JAX在3个种子都变慢，153个JAX作业增加4569.36作业·秒，CNN/LLM合计节省4885.36作业·秒；118次将同状态MW的JAX根换成其他类，反向仅20次。问题是慢类等待被转移，整体均值不足以约束长尾。
- gpu_a、负载.95：H未完成84→90（+7.14%）。seed42的一批8个LLM作业从599.355秒推迟到602.352秒，跨过H；625秒该批已追回。保留实际截止点失败，不据此判定持续吞吐下降，也不改截止点。

32条独立规划样本的固定状态复评：新候选13个状态改变根，其中7个成本更低、6个更高，另11个保持MW，平均差−2.700928作业·秒；旧零未来清空根7个更低、17个更高，平均差+2.866416。有限窗排序有所修复，但仍存在采样误选，不能替代实际轨迹结论。

下一步：保留本轮候选，给选择器加入可见FIFO等待年龄，用相对MW的预测尾部成本约束均值优化，再冻结新方案作性能对照。优先解决node007高负载的慢类延期，不通过增加样本、调整窗口或放宽门槛追逐本轮结果。

限制：这是目录报告2/4槽的确定性缓存独立槽模型，使用已知到达率和H，有限规划窗不含窗外新增作业；3个开发种子复用，尚无独立确认、物理部署或异步稳定性结论。

代码：[选择器](/home/researcher/mine_code/scheduleurm/algorithm/arrival_aware_lane_rollout.py)、[实验](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_arrival_aware_lane_rollout.py)。证据：[性能结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_arrival_aware_lane_rollout_development_20260930.json)、[排序诊断](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/arrival_aware_root_ranking_20260930.json)、[类别与动作分解](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_arrival_aware_lane_rollout_development_20260930_mechanism.json)。
