# 22 状态类别核算与下一步（2026-10-01）

复用旧 current 延续下的全部22个固定 A/BG 标签。各类 cohort 人数加权后还原总体均值差，22例全部通过，最大误差7.68×10⁻¹⁴秒；新增服务轨迹和评分提案均为0。

- 19/22 的类别均值差正负混合；8/22 出现同一类别 T均值改善但900秒未完成数增加。
- 7个仍保留的均值有害A全部存在类别收益抵消：JAX均值恶化5次、CNN4次、LLM1次；LLM均值改善6次，但900秒未完成数增加3次，其中2次与均值改善同时发生。

下一步先收齐冻结24条新策略轨迹，以 MW/BG 联合 mean flow、P90、未完成数门槛裁决。若仍 HOLD，再分解有限预测人群与完整 cohort 的 CNN/JAX 均值成本、LLM 尾部成本；本核算支持这个问题，尚未证明遗漏原因或新策略性能增量。论文未改。

更新：完整24条已HOLD，随后[固定22状态人口归因](two_step_current_policy_population_accounting_20261002.md)已完成。7个保留坏动作中6个至少一类预测获益而完整cohort受损；下一步保持人口不变，单独对照续排近似。

[结果与逐状态核算](experiment_artifacts/aligned_hardware_two_step_current_policy_class_accounting_20261001.json)。
