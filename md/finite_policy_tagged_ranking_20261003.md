# 固定群体的有限父策略自由续排评分

上一轮三个实际前缀交叉中，父首动作在有限续排下两处降低群体均值、一处持平；持续续排未提供统一性能增量。本轮固定GPU2/.95、seed41/42/43、frame11/39/29和两动作2–5/7–4/4–5，检验原生“两次旧当前策略调用后MW”的评分近似是否影响动作排序。

先精确恢复各有限策略的实际前缀，完整消费一次原生首提议，逐项复现缓存动作和费用。保留原8条planning paths、原风险集合、cap128。两种评分使用同一可见及prefix标记作业、固定anchor率和背景到达流；后续背景作业参与FIFO但不计分。自由有限策略从完整首提议后的三层状态开始，后续每次调度（含idle）调用完整原生父策略。48个状态×动作×样本单元独立并行，原评分聚合必须精确重现原生S1/S2。

测试覆盖原评分的精确成本/费用/子状态、完整状态下含idle的自由续排，以及保留固定溢出样本。测试exit0后登记；三个前缀准备exit0后放行48单元；全部exit0后合并，只同步小汇总。准备每项22 workers/CPU23，评分每项6 workers/CPU7，48项分别偏好node001/004/005/006各12项，允许001–006。

已于2026-10-02T16:27:26.529005+00:00登记。服务器t126953三项测试通过、真实exit0；首测t126947失败证据保留，已修复新kernel对原生`(None,)`空调度的识别，并纠正toy预约时长与completion offset的混用。前缀准备t126959/960/961、48项评分t126962–t127009及合并t127010均已确认真实exit0，原生精确评分复现通过，48项free均无overflow。仅116,818字节小汇总回传。

合并曾因scheduler延期诊断丢失已验证退出码而阻塞；48项原始退出记录和登记结果重新核实后补齐门槛，未重跑原分片。修复终态字段传递及retry清旧退出码，32项定向测试通过，常驻watch已加载修复。原失败分类、自动重复运行的FileExistsError和控制器停止记录保留于历史；本轮完成以[恢复凭据](experiment_artifacts/aligned_hardware_finite_policy_tagged_ranking_20261003_merge_recovery_receipt.json)为准。

S1为每条planning path的标记作业总流时间，以下为父动作减另一动作（秒），负值偏好父动作：

|seed|原近似S1差|自由有限续排S1差|
|---|---:|---:|
|41|−7.8954|+10.0612|
|42|−3.5083|+6.4416|
|43|−9.6393|+2.7514|

三个状态均翻转。续排近似确实影响同群体排序，但自由续排偏好的动作没有在上一轮真实出生群体中取得一致优势。后续已完成类别、visible/prefix及原8路径敏感性分解，并复用六条真实分支完成局部群体对齐：41出现局部收益与后期群体指标反向，43保留局部预测方向不一致及显著后期排队效应。[进一步分析](finite_policy_tagged_ranking_analysis_20261003.md)、[结果小汇总](experiment_artifacts/aligned_hardware_finite_policy_tagged_ranking_20261003_summary.json)。

局限：任一free样本溢出，则对应根的完整8样本评分不可用，保留部分成本和费用，不换样本或提高cap。真实[300,900]出生群体指标仅作独立参照。本轮检验同群体动作排序，不能证明全策略性能增量；原全24 HOLD和确认集保留，论文不改。

[实验入口](../algorithm/experiments/aligned_hardware_finite_policy_tagged_ranking.py)、[评分kernel](../algorithm/experiments/finite_policy_tagged_rollout.py)、[运行准备](experiment_artifacts/aligned_hardware_finite_policy_tagged_ranking_20261003_runner_preparation_receipt.json)。
