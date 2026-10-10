# Linux/.95 finite/BG单动作因果对照

全部3个科学对照t130201/202/203与merge t130205已done/exit0；原生锚点、完整提案后状态和单动作干预检查通过。t130199的89决策null完整复现原轨迹；seed42干预发生在916.316秒，900状态严格不变。只回传47,909B校验记录与67,798B摘要，完整轨迹/fixture留服务器。

下表为“BG首动作→finite续排 − 原finite”的条件变化；mean/P90单位秒，负值表示改善。cohort仍为[300,900]，mean/P90在1200评价，P90取同对共同完成ID。

| seed | Δmean | ΔP90 | Δ未完成900 | Δ未完成1200 |
|---|---:|---:|---:|---:|
| 41 | +0.299 | -0.900 | -1 | +0 |
| 42 | -0.025 | +0.000 | +0 | +0 |
| 43 | -1.254 | -0.433 | -1 | +0 |

seed41 mean恶化+0.299，来自平均等待+0.203和服务中时间+0.096；三类mean均恶化，JAX P90另增8.579。seed42仅JAX等待改善，整体mean−0.025，P90不变。seed43 mean改善−1.254，等待−1.357抵消服务中时间+0.104；三类mean都改善，但CNN P90增加11.153，LLM P90增加0.015。总体P90改善不能说明每类尾部改善。

原full24仍HOLD，本轮不建立整策略增益。首次失败t130193来自新入口将raw1200快照混比实际900；原数据/指标正确，失败协议与fixture保留，修复后完整null通过。科学费用与有效null费用分列，失败null的完整内部工作账本未落盘。

后续[冻结设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_native_controller_cross_20261003_design.json)已完成：同3状态补“finite首动作→BG自身原生状态续排”，新增3条科学分支并复用9条完成臂。三个状态均未出现严格mean排序反转，但seed43收益大小和逐类代价依赖控制器，详见[对照结果](/home/researcher/mine_code/scheduleurm/md/finite_BG_native_controller_cross_20261003.md)。跨控制器变化包含各自原生状态与锚点完整提案消耗，不能解释成单纯后续算法切换。论文未改。

[摘要](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_first_action_cross_20261003_summary.json)；[完成记录](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_first_action_cross_20261003_completion_receipt.json)；[实现](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_finite_bg_first_action_cross.py)。
