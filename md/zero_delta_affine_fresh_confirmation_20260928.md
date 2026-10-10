# Zero-delta affine 独立确认，2026-09-28

结论：质量 `ADVANCE`，四臂计时 `HOLD`；独立新增组通过全部预注册门槛，三个旧控制状态未通过逐状态门槛。

事前冻结种子 2026092893–2026092896、原条件抽样律、排除集及所有预算/门槛；256 个不同 R/Q 抽样配对四个固定目录，产生 1024 个新状态，另保留全部 1778 个旧控制。新旧样本分别汇总，并保留旧 lineage。
质量阶段执行 1024 次独立 K4096 参考、2 次参考触发的精确 oracle、3072 次整选择器及 12288 次 native advance；参考完整，动作、评分、canonical、证书、路径和工作量核算通过。新增 62 项测试通过。
新状态整选择器的 child-bound evaluation 从 4893 降至 2081，跳过 2812 次（57.47%）；2072 次 bound improvement 和 2057 次 prune 保持一致。
计时采用种子 2026092897 冻结四臂顺序，六轮/状态：67248 次正式调用＋128 次 warmup，全部完成；所有时钟重新测量，执行、证书、重复路径与 CPU affinity 检查通过。

下表为候选/同轮对照的状态中位 wall time 总和比值；小于 1 表示候选较快。

| 样本组 | 状态数 | affine | branch | long |
|---|---:|---:|---:|---:|
| 独立新增 | 1024 | 0.998104 | 0.782855 | 0.536563 |
| 旧普通开发 | 688 | 0.994542 | 0.829562 | 0.689718 |
| 上轮 fresh，现为控制 | 1024 | 0.997834 | 0.777320 | 0.492493 |
| 历史退化控制 | 34 | 1.000219 | 0.969733 | 0.603741 |
| 两组 trigger 合并 | 16 | 0.959338 | 0.966100 | 0.749813 |

三个超过冻结 1.05 逐状态上限的控制状态：
- gpu_a/R64，Q=(88,8,23)：affine 1.162139，branch 1.095841；对应 CPU 比值 1.046353、0.996237。
- gpu_a/R64，Q=(118,119,269)：branch 1.063718；对应 CPU 比值 1.057601。
- cpu_a/R8，Q=(27,15,34)：affine 1.051389；对应 CPU 比值 1.021544。

新增普通组的真实尾部也保留：gpu_a/R64 Q=(158,196,145) 比 affine 慢 27.80%；Q=(124,10,80) 耗时为 long 的 5.275790 倍（CPU 5.268847 倍），候选 143 LP/111 split，long 28 LP/17 split。原普通组门槛只约束 pooled、cell 和 P99，因此这些尾部不改变其通过结论。
限制：本次相对 affine 的独立组平均收益仅约 0.19%，不证明逐状态加速。目录及虚拟同构 lanes 固定，配对副本和六轮重复具有依赖；不增加 physical scaling、flow-time 或显著性结论。旧 HOLD 和本次全部失败保留；论文与旧算法、协议、结果均未修改。
下一步：复用 branch/容量收紧扫描产生的真实变化坐标，让 guard 和 affine fallback 共用扫描，同时保留跨失败 LP 和零变化跳过的证书锚点。另一个后续路由实验是在根阶段评分已证明时提前 canonical completion：Q=(124,10,80) 根阶段 7 LP 已 score_exact，却继续到 143 LP；采用统一规则及原共享整数预算。两项分别另建候选、先冻结协议，保留本轮 HOLD。
原始证据：[质量汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_zero_delta_affine_fresh_confirmation_20260928.json)、[计时汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_zero_delta_affine_fresh_timing_20260928.json)，各自同名前缀下保存协议、16 个 cell 和运行日志。
