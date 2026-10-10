# 稀疏 dual delta 开发实验，2026-09-28

结论：质量与开发计时门槛均 `ADVANCE`，但本轮全样本状态中位耗时之和比冻结 zero 对照高 0.23%（CPU 高 0.21%），没有平均提速证据，保留候选与结果，不替换上一版。

实现将 branch 和容量收紧产生的变化合并为有序 pending delta，相对同一证书的最近实际求值锚点累计；guard 与 affine fallback 共用支持。零变化跳过、失败 LP 保留累计变化，新证书或实际求值重置。复制、合并、排序处理、支持读写和锚点维护分别收费。
事前冻结全部 2802 个旧状态、七个来源组、原 144/256 LP 和共享 8192 整数预算、六轮计时及所有原门槛；五臂为 sparse、zero、affine、branch、long，顺序种子 2026092898。全部状态现在均为开发控制。
77 项新增测试通过。质量阶段完成 2802 个 cold selector＋11208 个 native advance，共 14010 次接口调用；评分、动作、canonical、LP/剪枝/证明/完成路径及费用核算全部通过，未新增参考或 oracle 调用。
整选择器 guard visits 从 124416 降至 42059（减少 66.2%），affine visits 从 40026 降至 22456；另实际产生 41110 次 pending map 构建、87190 个复制条目、73742 个排序处理条目。child evaluations=6806、improvements=6686、prunes=6423 与上一版完全相同。
五臂计时完整完成 84060 次正式调用＋160 次 warmup；全部执行、参考、证书、重复路径、冻结输出及 affinity 检查通过，四个对照的所有预注册门槛均通过。

候选/同轮对照的状态中位 wall time 总和比值如下，小于 1 表示候选较快。

| 开发来源组 | 状态数 | zero | affine | branch | long |
|---|---:|---:|---:|---:|---:|
| 上轮独立组，现为控制 | 1024 | 1.004048 | 1.002420 | 0.783825 | 0.536229 |
| 更早 fresh，现为控制 | 1024 | 1.002933 | 1.002257 | 0.778476 | 0.491624 |
| 普通开发 | 688 | 1.001438 | 0.997628 | 0.829846 | 0.691231 |
| 历史退化控制 | 34 | 1.001643 | 0.993373 | 0.975410 | 0.609106 |
| 两组 trigger 合并 | 16 | 0.999225 | 0.992847 | 0.990977 | 0.761402 |

三个旧失败状态本轮均通过对应逐状态门槛，但同轮冻结 zero 也通过这三个状态；不能把历史 HOLD 转为本轮开发 ADVANCE 归因于稀疏化。
保留的实际反例：node007/R64 Q=(44,305,281) 候选/zero 为 1.244240；gpu_a/R64 Q=(124,10,80) 耗时 215.45 ms，long 为 39.56 ms，比例 5.446477（CPU 5.400127），仍是 143 LP/111 split 对 28 LP/17 split。
限制：普通组原门槛约束 pooled、cell 与 P99，未约束每个状态；零队列计时原为描述性，本轮 pooled/branch 为 1.085511。所有样本曾经使用，目录副本和六轮重复有依赖；不增加独立确认、physical scaling、flow-time 或显著性结论。历史 HOLD、全部尾部及原论文/算法/协议/结果均保留。
下一步在冻结 zero 引擎上另开统一的早期 canonical 路由：同一 search 推进至根预算 M，若 score_exact 且 canonical 未闭合则提前完成 score face；至多一次整数 probe，全过程共用 8192 预算，耗尽后只继续原 LP frontier，记录未闭合结果。根评分未证明时沿用原晚期流程；先冻结新协议再执行，不继续调本轮稀疏条件。
证据：[质量汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_sparse_zero_delta_affine_development_20260928.json)、[五臂计时汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_sparse_zero_delta_affine_timing_20260928.json)；同名前缀保存冻结协议、16 个 cell 和运行日志。
