诊断完成：`AA_DIAGNOSIS_COMPLETE`。两个既有尾部状态、12 个固定窗口，476 次调用全部匹配冻结参考；实际计费 8806 次 LP、零整数搜索，49 项定向测试通过。

三个 A/A 标签调用同一实现。固定 a 相对 b/c 的六轮中位数比值，24 个状态窗口均未超过 1.05，最大 1.01202；预先登记的对称 max/min 则有 2/24 超过 1.05，最大 1.15628。两次均在首窗，慢标签为 b：Q=(88,19,118) 差 5.68%，Q=(184,78,235) 差 15.63%。固定方向失败未复现，同实现标签间的明显波动确实存在。

另行插桩的候选/短探测中位墙钟比为 0.99596、1.01388。两状态始终由初始 LP 闭合，分别 21、16 次 LP；全调用扣除主阶段的中位余项为 0.028–0.044 ms，分段样本内 GC 暂停占比最高 0.0794%。本轮没有定位出新增搜索或稳定的分流开销。

局限：这两个已观察状态在本地 CPU 上连续计时，节奏短于原 1058 态工作负载；窗口相关，插桩另有观察开销，GC 结果不能反推旧运行。同实现波动不足以解释旧确认的具体失败，固定方向零越界也不证明普遍具备 5% 分辨率。原 `CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD` 保留。

下一步把同实现 A/A 控制放入原工作负载的轮次间隔中，冻结一次计时校准，再决定新确认协议；不改原判据或覆盖旧结果。

证据：[汇总](experiment_artifacts/aligned_hardware_certificate_routed_aa_diagnosis_20260928.json)、[协议](experiment_artifacts/aligned_hardware_certificate_routed_aa_diagnosis_20260928_protocol.json)、[逐调用记录](experiment_artifacts/aligned_hardware_certificate_routed_aa_diagnosis_20260928_raw.json.gz)。
