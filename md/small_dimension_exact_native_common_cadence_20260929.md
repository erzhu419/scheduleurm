# Small-dimension exact native 共同节拍对照：2026-09-29

结果为 **SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_TIMING_ADVANCE**：312 条轨迹完成，9 个预登记速度门槛全部通过；两算法的帧、动作、派发、所有作业及 H/T 快照均等于同一新节拍规范参考，超窗为 0。

69 项针对性测试通过。00:59:14 UTC 登记后执行，复用四目录/R8、负载 0.80/0.95、开发种子 41–43 的全部 24 条到达带和服务目录，保留 144/256/8192 预算及六轮 AB/BA 各三次的顺序。固定计算窗 τ=1 秒来自原 idle 时间单位；每次冷调用实际测 wall/CPU，策略时钟推进 max(τ,wall)，等待补齐单列。参考是固定 τ 的离线规范轨迹，真实求解费用另记，不充当部署截止基准。

两算法各完成 7392 次选择，完整参考 1232 次，共 16016 次实际策略调用。全部证书、参考、动作和 FIFO/截止记账通过；额外离线参考和 oracle 均 0，实际规范参考结果供审计共享。

| 实际计算费用 | candidate | direct |
|---|---:|---:|
| 平均 wall ms | 4.908397 | 6.222035 |
| 平均 CPU ms | 4.561188 | 5.743613 |
| 累计 wall s | 36.282868 | 45.993280 |
| 累计 CPU s | 33.716300 | 42.456786 |
| 逻辑 LP / 整数访问 | 35526 / 6204 | 35394 / 6204 |

按每带六轮平均后汇总，wall 比为 0.788873（降低 **21.11%**），CPU 比为 0.794132（降低 **20.59%**）；八个 node/load 组的 wall 比为 0.738876–0.820880。candidate 的逻辑 LP 为通用 20904 + 精确 14622（0/1/2 维 234/7410/6978）；逻辑 LP 增加 132，整数访问和分支数相同。实际费用包含全部 native、精确证据、root phase、dual-face 和探针工作。

下一步在未使用的到达种子上冻结并做独立共同节拍确认，再确定主场景的算法调用入口。保留原自由时钟 **UTILITY_HOLD**，不将本轮身份保持写成流时间提升。

证据：[汇总](experiment_artifacts/aligned_hardware_small_dimension_exact_native_common_cadence_development_20260929.json)、[协议](experiment_artifacts/aligned_hardware_small_dimension_exact_native_common_cadence_development_20260929_protocol.json)。四个 closed cell、日志和汇总共约 15.63 MiB，无服务器拉取。

## 已知限制

P99 实际 wall 为 22.029679/21.732295 ms，candidate 上升 **1.37%**，按原协议为描述性指标。新增 1 秒计算窗改变了批次和流时间：本轮两算法的 T 均值均为 42.955009 秒，旧自由时钟 direct 为 35.114586 秒；这是跨协议描述，不支持将 1 秒窗作为部署建议或声称端到端收益。负载仍参考原服务容量，非加入计算窗后的容量。证据限于已用开发种子、确定性缓存目录、虚拟 R8 和本地 CPU0；计时重复不增加独立观察。
