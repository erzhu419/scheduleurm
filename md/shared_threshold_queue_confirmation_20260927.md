# Shared-threshold 新队列确认（2026-09-27）

固定 4 个 physical catalogue 点模型，在 R=4/8/16 上生成 768 个未见过的 (R,Q)，跨节点配对得到 3072 次比较。四个随机种子为 2026092701–2026092704；各 R 内去重并排除开发网格。模型、队列、源代码及预算在运行前冻结。

主比较预算为 36R，独立 full 参考预算为 4096；R4 另用 Fraction 穷举核对。12/12 单元完成，参考全部闭合，证书检查通过；确认 runner 的 14 项针对性测试通过。

- LP 调用：63625 → 55404（−12.9%）。
- LP 矩阵项：4506690 → 3308835（−26.6%）；12 个 node/R 单元均满足预定成本门槛。
- shared 有 3070/3072 次完成 canonical 精确性证明；full 为 3066/3072。两者各有同一个实际动作偏差。
- 正式结论为 `DEPLOYMENT_ACTION_REVIEW_REQUIRED`：成本门槛通过，全状态动作及证明门槛未通过。

两个 shared 未闭合状态：

1. gpu_a，R8，Q=(6,6,37)，预算 288：两方法均选 service=(6,0,36)，参考为 (3,0,37)，相对评分损失 1/73（1.37%）。一次 CNN p3 → LLM p1 交换可得到参考动作，但尚未验证证书能否随之闭合。
2. gpu_b，R16，Q=(35,32,77)，预算 576：动作与评分已正确，仍有一个同分区域未完成 canonical 证明；该区域不是 singleton，不能认定预算后的免费收尾即可解决。

下一步回到开发：先测试分数变量优先分支，减少在仍有分数变量时对整数变量分支的预算消耗；必要时再研究有界动作交换和同分区域的 canonical 界。上述失败样本转为开发回归样本，后续确认须另取新队列。

限制：这是固定缓存点模型及指定队列采样分布下的搜索工作量证据，未验证实际墙钟时间或流时间收益。论文未修改。

证据：[冻结协议](experiment_artifacts/aligned_hardware_shared_threshold_queue_confirmation_20260927_protocol.json)、[正式结果](experiment_artifacts/aligned_hardware_shared_threshold_queue_confirmation_20260927.json)、[事后诊断](experiment_artifacts/aligned_hardware_shared_threshold_queue_confirmation_failure_diagnostic_20260927.json)。
