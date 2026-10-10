# Linux/.95 首动作×原生控制器对照

3条科学分支t130250/251/252和merge t130254全部done/exit0；t130248完整BG null复现缓存。原生锚点提案/费用、提案后状态保留、唯一动作覆盖、固定出生cohort和916秒干预不改变900状态均通过。只回传2,436B校验记录与92,084B摘要，完整轨迹留服务器。

下表均为BG首动作−finite首动作的条件效果；cohort为[300,900]，mean/P90在1200评价，单位秒。四臂在1200全部完成同一cohort，故P90可比。

| seed | finite续排 Δmean | BG续排 Δmean | finite续排 ΔP90 | BG续排 ΔP90 | Δ未完成900（finite/BG） |
|---|---:|---:|---:|---:|---:|
| 41 | +0.299 | 0.000 | -0.900 | 0.000 | -1 / 0 |
| 42 | -0.025 | -0.025 | 0.000 | 0.000 | 0 / 0 |
| 43 | -1.254 | -0.590 | -0.433 | -3.444 | -1 / -1 |

三个状态均无严格mean排序反转。seed41只在finite下发生cohort差异；BG下cohort等待/服务/逐类指标均不变。seed42两种控制器效果一致，仅JAX等待改善。seed43的收益大小和分配依赖原生控制器：finite下平均等待−1.357、服务中时间+0.104；BG下分别−0.459、−0.131。

总体mean/P90改善不能代表逐类改善：seed43 finite下CNN P90增加11.153；BG下CNN mean/P90增加3.487/5.478，LLM mean/P90增加0.560/0.177，JAX mean/P90则降低8.602/30.820。seed41 finite下三类mean都恶化，JAX P90另增8.579。没有采用统一BG首动作规则。

同样执行BG首动作后，seed43 finite续排仍比原BG的mean/P90高0.164/8.784秒；seed41为−0.152/+0.684秒，seed42均为零。追加的[六组缓存定位](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_cached_finite_BG_common_root_continuation_20261003_summary.json)已由t130256 done/exit0完成，零新增规划或模拟，只回传90,251B。固定BG首动作的seed42完整动作后缀和job时间一致；固定finite首动作的seed42仍有后续分歧，不能从cohort指标为零推出轨迹相同。

固定finite首动作后，原finite与BG的首次再分歧分别在frame55/173/161、132.905/1051.538/286.939秒：finite选CNN+LLM/LLM1/LLM1，BG均选JAX1。五处缓存分歧的finite自身BG incumbent及其原生S1/S2预测均与对应BG记录一致，外层评分完整完成。seed43两种共同首动作下的finite新预测都允许S1下降、S2上升；这些局部预测不能解释为那一次动作造成了整个后续性能差。

2026-10-03已决定采用BG作为[项目主结论](/home/researcher/mine_code/scheduleurm/md/main_result_BG_20261003.md)，本finite分支结束。原[后续单动作冻结设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_later_action_cross_20261003_design.json)保留为未执行方案，不再提交；全部已完成结果、原HOLD与冻结科学记录保留。

局限：每个控制器内的首动作对照是条件因果效果；跨控制器差值同时包含各自原生前缀记忆、完整锚点提案消耗与后续决策，不能归为纯续排算法效应。原full24仍HOLD，本轮未建立整策略增益。论文未改。

[摘要](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_native_controller_cross_20261003_summary.json)；[完成记录](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_native_controller_cross_20261003_completion_receipt.json)；[实现](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_finite_bg_native_controller_cross.py)。
