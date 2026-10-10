# 原生自身紧对偶界裁域：质量 ADVANCE，开发计时 HOLD

在 fused 路径中分开缓存自身对偶上界与继承界。自身界等于当前最优评分时，用严格负的 lower 系数和严格正的 upper price 固定端点，并保留资源、时长和 canonical 覆盖。原始 M 阶段的选择及旧计数保持一致；后续仍使用同一 search、144/256 LP 上限和唯一 8192-state 整数探查。缓存、裁域、证据和快照复制均计入实际费用。

53 项定向测试通过（内核 16、driver 11、审计 9、计时 17）。冻结的 2802 状态质量实验执行 14010 次接口调用、5604 次 search 构造；全部参考、质量和费用审计通过，失败与未闭合均为 0。whole 与四阶段审计路径的额外快照复制费用分别计算。

相较真实 fused 对照，whole selector 的工作量为：

- LP：18932 → 17607；分支：7668 → 4106。
- 整数状态：35553 → 25274；根部整数探查：1031 → 232。
- 根部直接闭合：1234 → 2033，增加的 799 次闭合对应减少的 799 次探查；2641 次裁域均非空。

十臂计时完成 168120 次计时调用与 320 次 warmup，共 168440 次。执行、证书、参考、质量、路径重现与 CPU affinity 均通过；费用与质量错误均为 0。candidate/fused 的状态中位耗时之和比为 0.907893709（下降 9.21%），CPU 比为 0.909658335。普通子组的 pooled/cell/P99、历史 pooled/P99 和全部 trigger 门槛通过。

## 未解决的问题与证据范围

7 个历史逐状态 ≤1.05 门槛失败，对照为 fused、settled、native、face、early、zero、branch。历史 candidate/fused 最大比为 1.104461533，因此本轮保持 HOLD。

最大 fused/settled/face/early 回退同在 gpu2/R64、Q=[122,3,4]（state_id `previously_untimed_probe_regression:122:3:4`）。candidate/fused 均 LP5、分支1、整数补全304；本点新增3次检查、24次变量访问、96个边界值复制，裁域和事件均为0。墙钟中位为 11.127773/10.075293 ms，CPU 为 8.764139/8.485863 ms。face/early 无分支，补全仅101/102状态；继承的根部拆分扩大了补全工作，本轮检查未提供搜索收益。墙钟与 CPU 增幅不同，不能把全部墙钟差归因于新增操作。

旧慢点 gpu2/R32、Q=[61,0,35] 已发生11次裁域，分支89 → 84，但 LP142与整数604未减少。candidate/fused 墙钟为292.146692/284.093503 ms，比1.028347，在门槛内；它不是本次最大失败点。

2802 状态均已用于此前开发，既有来源标签不构成新的独立确认。本轮是固定缓存目录与虚拟并发度的本地 selector 计时；未验证新的 physical scaling、流时间或显著性。旧 HOLD 保留，论文与线上 dispatch 未改。

233 个状态各发生一次根部分裂，232 个随后停止，只有1个依靠分裂闭合；其余2032个根闭合不依赖分裂。下一步单独冻结验证取消根部阶段的免费分裂，保留裁域及廉价闭合，将完整已裁箱交给既有的一次整数探查；耗尽仍沿同一 search 的144/256路径继续。先处理Q=[122,3,4]所暴露的补全工作膨胀，再处理评分未闭合时的整数上界。

证据：[质量结果](experiment_artifacts/aligned_hardware_own_tight_dual_native_development_20260928.json)、[质量协议](experiment_artifacts/aligned_hardware_own_tight_dual_native_development_20260928_protocol.json)、[十臂计时](experiment_artifacts/aligned_hardware_own_tight_dual_native_timing_20260928.json)、[计时协议](experiment_artifacts/aligned_hardware_own_tight_dual_native_timing_20260928_protocol.json)。实现：[原生内核](../algorithm/own_tight_dual_native_kernel.py)、[selector](../algorithm/own_tight_dual_native_frame_candidates.py)。
