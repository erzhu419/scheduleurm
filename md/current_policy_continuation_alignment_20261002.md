# 持续旧current续排对照（2026-10-02）

全部22状态及评分人口、8样本/cap128、原BG风险范围冻结，仅将第3次起的MW续排换为持久旧current（内部仍两次BG后MW）。24个后台均结束/exit0，3项测试及合并检查通过；新增服务轨迹0，原性能结论仍HOLD。

22/22对评分可用，正确数12→13：纠正5个坏动作，但误拒3个好动作并重新接受1个坏动作；原3个误拒好动作均未修复。类别方向错位总数仍28次，没有系统改善。

按缓存旧current标签（选A取ΔT_mean，选BG取0），22状态等权平均改善0.038176秒，但seed42恶化0.380339秒。实际选择为A8/BG12/第三动作2；20个有实际动作标签的状态平均改善0.013918秒。额外续排15,610次，已单列收费。

两个第三动作的标签现已补齐：[结果](current_policy_third_root_labels_20261002.md)。二者均改善条件均值和P90；全部22个实际选择的条件均值平均改善0.141617秒，但seed42仍恶化。下一步进入一次保留全部24条开发轨迹和原MW/BG联合门槛的完整闭环性能试验；当前仍不升级候选，不扫描深度/窗口。

局限：上述是状态条件效果，不是闭环性能增量；人口差异仍在，剩余错误不能单独归因为人口遗漏。本报告原排序统计只覆盖A/B对；第三动作实际效果见后续报告。未读取确认集或改论文。

证据：[小汇总](experiment_artifacts/aligned_hardware_current_policy_continuation_alignment_20261002_summary.json)、[逐状态摘要](experiment_artifacts/aligned_hardware_current_policy_continuation_alignment_20261002_records_summary.json)、[条件效果计算](experiment_artifacts/aligned_hardware_current_policy_continuation_alignment_20261002_conditional_effects.json)、[完成记录](experiment_artifacts/aligned_hardware_current_policy_continuation_alignment_20261002_completion.json)。完整数据留服务器，本地仅拉取约228KB的两份结果摘要。
