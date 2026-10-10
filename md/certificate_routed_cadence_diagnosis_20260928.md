完整负载 A/A 校准完成：`CADENCE_DIAGNOSIS_COMPLETE`。复用原 16 个单元、1058 个计时状态及六轮调用顺序，19140 次调用全部匹配冻结结果；37 项定向测试通过。实际执行 candidate/probe/long 各 6404/6368/6368 次，计费 291978 次 LP、592128 个整数状态；没有新参考或 oracle 求解。

仅在两个 `node007/R64` 目标状态上，三个角色均调用同一 candidate。表内比值比较的是角色标签，不能解释为算法提速。

| 队列 Q | candidate/probe 标签 | candidate/long 标签 | 三标签最大/最小 |
|---|---:|---:|---:|
| (88,19,118) | 0.99688 | 0.99739 | 1.00313 |
| (184,78,235) | 1.00026 | 1.00059 | 1.00059 |

固定方向和对称比较均为 0/2 超过 5%。实际相邻轮次间隔分别为 1.32–4.79 秒、0.82–4.24 秒；两个状态均保留六次原始墙钟及 CPU 时间，逐调用复核一致。

局限：本轮恢复了计时工作负载及顺序，省略了旧筛选/参考计算，无法重现旧主机历史。两个已观察状态的单次校准既不能解释原慢例，也不能建立普遍的 5% 分辨率。原 `CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD` 保留，这条计时诊断到此结束。

下一步转向根节点预剪枝开发实验：在支付 LP 前利用已有可行性检查和保守上界，检验能否减少求解次数，并核对动作、证书和有限预算行为；先比较确定性计算量。

证据：[汇总](experiment_artifacts/aligned_hardware_certificate_routed_cadence_diagnosis_20260928.json)、[冻结协议](experiment_artifacts/aligned_hardware_certificate_routed_cadence_diagnosis_20260928_protocol.json)及同前缀 16 个压缩 cell。
