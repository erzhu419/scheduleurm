# 两步 current-policy 续排开发试验（2026-10-01）

**HOLD，完整24条开发结果已产出，执行检查全部通过。** 本地及后续未完成单核任务已取消，已有完成结果保留；最终选用原生7片、串行优化1片、[决策内并行](parallel_root_execution_20261001.md)16片，后者启动至调度器检测全部完成约30.27分钟。

实现仅将外层两次虚拟 BG 调用替换为冻结的旧 current policy；内层保持两次 BG 后转 MW，不递归调用新策略。保留原 BG incumbent、原 S2≤MW 候选范围、严格 S1 改善、8 样本/cap128、因果到达上下文、完整 lane 占用与逐层实际工作量记账。

24条固定开发tape（4 catalogue×2 load×3 seed）保留48条完整MW/候选轨迹，复用原BG和旧current的各24条缓存轨迹。沿用原MW/BG联合门槛：均值至少改善1%、各负载均值不退步、分组P90/未完成数不超过原值5%、总体未完成数不增加。只有3个共享开发种子，结果按原协议描述。

1200秒终点，候选平均流时间30.4858秒；相对MW的33.8051秒改善9.819%，相对普通BG的31.1614秒改善2.168%，相对上一版current的30.4957秒仅改善0.0325%。两负载均值均优于MW/BG，但分组门槛失败：GPU/.8在900秒未完成35→40；vsBG的GPU/.95为68→75，Linux/.95为29→31，且Linux/.95的分片P90均值在900/1200秒分别增加6.132%/6.429%。总体900秒未完成虽从BG的427降到417，1200秒双方均为0，联合结论仍HOLD。

相对上一版，24片11改善、11恶化、2相同；高负载均值恶化0.598%，seed43恶化1.966%。按每片类别任务占比加权后再平均24片，均值变化由CNN的+0.13290秒、JAX的−0.15286秒、LLM的+0.01005秒抵消为−0.00992秒；900秒类别未完成128/115/182→133/117/167。不能把新问题概括为单一LLM尾部。

新增 5 项测试通过，覆盖有限三层构造、两次调用的上下文/RNG/计数器、逐层收费、溢出回退、未来到达不可见与原风险范围/严格 tie 规则。

[固定22状态排序诊断](two_step_current_policy_root_ranking_20261001.md)已完成：正确数仍12/22（纠正3、误拒3），没有建立性能增量。该诊断未改写本次协议。

随后[固定22状态人口归因](two_step_current_policy_population_accounting_20261002.md)已完成：7个保留坏根中6个至少一类预测获益而完整cohort受损。下一步保持评分人口与原风险范围，单独对照第3次起使用冻结旧current的续排，区分续排近似与人口目标错位；这不是递归新策略的采纳。完整开发结论仍HOLD，论文未改。

证据：[原冻结协议](experiment_artifacts/aligned_hardware_two_step_current_policy_bg_incumbent_development_20261001_protocol.json)、[10KB结果摘要](experiment_artifacts/aligned_hardware_parallel_root_sharded_20261001_result_summary.json)、[完整汇总](experiment_artifacts/aligned_hardware_parallel_root_sharded_20261001.json)、[完成记录](experiment_artifacts/aligned_hardware_parallel_root_sharded_20261001_completion.json)。完整轨迹留服务器。
