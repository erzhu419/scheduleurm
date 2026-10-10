# 早期根 canonical 路由开发结果（2026-09-28）

已新增 [early_root_canonical_frame_candidates.py](/home/researcher/mine_code/scheduleurm/algorithm/early_root_canonical_frame_candidates.py)：冻结的 zero-delta 引擎在实际根预算 M 后，评分已证明且 canonical 未闭合时立即补全；每次只建一个 LP 搜索、调用一个整数 probe。耗尽后保留费用并延续原 LP 搜索。

源码、2802 个旧开发状态、16 个单元、七个历史来源组及五臂计时顺序均在结果访问前冻结。预算保持144/256/8192；69项新增测试通过。质量实际执行2802次；计时实际执行84060次及160次预热，全部完成。没有新增参考、oracle或筛选调用。

质量为 `EARLY_ROOT_CANONICAL_DEVELOPMENT_ADVANCE`，计时为 **`EARLY_ROOT_CANONICAL_LOCAL_TIMING_HOLD`**。动作、证书、费用、重复路径、冻结输出及 CPU affinity 检查全部通过，质量失败和未闭合均为0。性能未达原门槛，暂不采用。

下表为候选/原 zero 比值；汇总使用每个状态六轮中位耗时之和，P99为状态中位耗时的99百分位比值。

|开发范围|汇总 wall|CPU|P99 wall|最大配对退化|
|---|---:|---:|---:|---:|
|全部2802|0.8404|0.8412|0.9180|2.6098|
|普通开发688|0.8779|0.8782|1.0017|2.5550|
|旧 fresh_confirmation 1024|0.8310|0.8326|1.5168|2.4898|
|旧 independent_fresh 1024|0.8762|0.8774|1.4375|2.6098|
|历史回归34|0.6139|0.6161|0.9640|2.1104|
|合并 trigger 16|1.0061|1.0037|1.0143|1.1821|

LP调用25,695→19,470（减少24.2%），整数状态22,783→897,240。2265次根 probe 中2213次完成、52次耗尽；耗尽仍在原 LP 预算内闭合。目标 gpu_a/R64、Q=(124,10,80)：LP143→7，整数状态仍为826，wall213.926→11.468ms。

反例 gpu_b/R64、Q=(122,235,280)：先耗尽8192，随后双方均13次LP，wall19.320→48.033ms。该单元29个提前耗尽队列全部没有LP节省；其两个旧1024来源组汇总/zero为1.3727和1.3247。当前补全从完整计数域重新搜索，没有使用根LP已关闭区域及对偶约束。

下一步另建“精确对偶约束的根 canonical 补全”候选：仅在证书自身界等于目标评分时，利用互补松弛固定对应边界；保留其余区域和统一8192预算，再冻结完整协议验证。这是待验证假设。

本轮是旧队列开发回归，使用四个缓存目录和虚拟同质资源；不构成新的独立确认。论文和旧算法、协议、结果未改，历次HOLD保留。

完整证据：[质量](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_early_root_canonical_development_20260928.json)、[质量协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_early_root_canonical_development_20260928_protocol.json)、[计时](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_early_root_canonical_timing_20260928.json)、[计时协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_early_root_canonical_timing_20260928_protocol.json)。
