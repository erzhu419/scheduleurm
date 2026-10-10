# 持续竞争终端估值（2026-10-03）

已实现同流持续到达的 MaxWeight 终端估值，复用原 48 个 H 状态；H 后新作业参与 FIFO/MW，但不计入评分。服务器 3 项测试通过；t128770 真实退出码 0，48/48 完成，无失败或 cap128 溢出。原 8 条路径、风险集合、H 前有限策略及论文均保持不变。

差值均为 parent−native，负值偏好 parent。前两列为八路径平均累计作业秒，最后一列为已有真实轨迹中同一 H 人口沿实际有限策略续排的完整累计作业秒；预测与实际使用不同到达轨迹。

| seed | 无新到达 MW 终端 | 持续竞争 MW 终端 | 真实 H 人口完整成本差 |
|---|---:|---:|---:|
| 41 | +9.973 | −12.360 | +31.108 |
| 42 | +11.783 | +4.852 | +1.084 |
| 43 | +7.855 | +9.593 | −28.347 |

持续竞争使 seed41 转向 parent，但其去掉任一路径后的范围跨零，真实 H 人口仍偏 native，原后续 cohort p90 还增加 1.220 秒。seed43 新评分的 8/8 路径均偏 native，仍未对应真实后续 mean −5.728 秒、p90 −45.233 秒的 parent 收益。本轮未建立可推广的性能增量，完整开发保持 HOLD。

缓存分解显示：seed43 的真实 H 窗口差 +0.159，加上实际有限策略的完成残值差 −28.506，反转为完整成本差 −28.347。同一 H 人口已能体现 parent 收益，支持下一项先检验终端控制策略：恢复原三层有限控制器的 H 状态，替换终端 MW，仍保留相同人口、路径与 cap128。此项已实现并完成 48/48 分支，但未改善 seed43 排序，详见[有限策略终端结果](native_terminal_flow_20261003.md)；原冻结协议保留在[设计记录](experiment_artifacts/aligned_hardware_native_terminal_flow_20261003_design.json)。

新增尾部与随机流恢复费用分别记录；本次只回传约 196 KB 汇总，无原始作业或 checkpoint 拉取。数据：[终端评估](experiment_artifacts/aligned_hardware_competing_terminal_flow_20261003_summary.json)、[真实人口分解](experiment_artifacts/aligned_hardware_cached_h_population_flow_bridge_20261003_summary.json)、[完成记录](experiment_artifacts/aligned_hardware_competing_terminal_flow_20261003_completion_receipt.json)。
