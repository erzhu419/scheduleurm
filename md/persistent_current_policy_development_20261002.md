# 持续旧current完整闭环结果（2026-10-02）

24个候选任务t120493–t120516及汇总t120528全部done/exit0，北京时间13:57完成汇总与自动同步；登记时间一致、执行检查全部通过。裁决 **HOLD**：持续续排未带来相对有限两次current父算法的性能增量。

沿用4个真实目录×2负载×3共享开发seed的24条literal tapes、8样本/cap128、原普通BG的S2<=MW风险范围及回退规则，从time0持续应用候选。MW/BG/父算法的72条完整对照复用服务器已验证缓存；本轮新增24条候选轨迹。以下为1200秒的restricted mean与共同ID P90，以及900秒未完成数。

|策略|平均流时间（秒）|P90（秒）|900秒未完成数|
|---|---:|---:|---:|
|MW|33.8051|67.3487|457|
|BG|31.1614|58.2741|427|
|有限两次current父算法|30.4858|55.9850|417|
|持续current候选|30.7622|57.2803|417|

相父算法均值增加0.2764秒（+0.9066%）、P90增加1.2953秒（+2.3136%）；两负载均值分别退化0.1556%/1.3449%。900秒未完成数低负载+12、高负载−12，总数不变；1200秒全部清空。相MW/BG均值改善9.0014%/1.2813%，但原联合门槛唯一失败为gpu_a/load0.8/900秒未完成数35→39（+11.43%，超过5%上限）；主P90分组全部通过。父算法比较单列，不另加或重调门槛。

候选不升级。保留已通过精确等价验证的tick计算优化。[两个第三动作的持续续排对照](persistent_continuation_cross_20261002.md)现已完成：均值1/2、P90 2/2由改善翻为退化；两处原生都选第三动作5。缓存全24首次分歧检查也已完成，仅3/24新S2超BG，10/24返回BG；该局部结果不能解释全部HOLD。[全根父动作诊断](persistent_parent_incumbent_scores_20261002.md)现已完成，规则保留6/24父动作，但GPU2高负载仅改变1/3。[该组三处真实前缀的单动作服务回放](persistent_parent_action_replay_20261002.md)也已完成：三处1200秒均值/P90均改善，seed42保留900秒总未完成+1、LLM未完成+6的代价；保护规则仍错过seed42/43这两处改善。下一步固定三前缀补齐首动作与后续控制器交叉，先分辨动作排序与续排交互，再决定算法改动。

范围：三个共享seed不构成24个独立重复，确认集与论文未动。运行成本诊断保留在[费用小摘要](experiment_artifacts/aligned_hardware_persistent_current_policy_development_20261002_tail_cost_summary.json)与[单片profile](experiment_artifacts/aligned_hardware_persistent_current_policy_development_20261002_cpu_profile.json)：已完成22条时逻辑模拟工作约6.12倍、current缓存命中率0.42%；固定状态约1.82倍计算提速见[tick优化记录](tick_background_optimization_20261002.md)。完整新轨迹仍留服务器。

证据：[完成记录](experiment_artifacts/aligned_hardware_persistent_current_policy_development_20261002_runner_complete_receipt.json)、[性能小摘要](experiment_artifacts/aligned_hardware_persistent_current_policy_development_20261002_performance_brief.json)、[原汇总](experiment_artifacts/aligned_hardware_persistent_current_policy_development_20261002_summary.json)。
