# Small-dimension exact native 新到达确认：2026-09-29

**SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_TIMING_ADVANCE**。832 条轨迹完成，动作、派发、作业和 H/T 快照全部一致；candidate/direct 超窗均 0，九个原速度门槛全部通过。11 项新入口测试通过；01:14:24 UTC 登记后执行。

新增独立确认入口，复用原 replay/summary 和算法。冻结新到达种子 13051–13058、64 条带、四缓存目录/R8、负载 0.80/0.95、H600/T1200、τ=1 秒、144/256/8192 预算，六轮 AB/BA 各三次。两算法各 19674 次选择，规范参考 3279 次，总计 42627 次；证书/完整参考/FIFO/时钟记账全部通过，额外离线参考和 oracle 均 0。

每带六轮平均后汇总，wall 比 **0.774113（下降22.59%）**，CPU 比 **0.771379（下降22.86%）**；八组 wall 比 0.728325–0.799032。实际累计 wall 为 93.831412/121.211550 秒，CPU 为 86.730460/112.435662 秒；P99 为 17.996949/18.911195 ms（下降 **4.83%**）。等待补齐单列，不算计算节省。候选逻辑 LP 89922/89616、整数访问 16032/15930，内部工作量仍有差异。

## 已知限制

离线规范参考有一次实际调用超过计算窗：1.027111 秒；保留超窗 0.027111 秒和固定规范时钟，按冻结协议不进入部署算法的超窗门槛。T 流时间均值两算法均为 43.133729 秒，H 未完成作业共 5044、T 为 0。证据限于八个新到达种子、同一 CPU0 和缓存模型；跨目录/负载共享种子，计时重复不增加独立样本。τ=1 秒仍是新增实验窗，负载参考原服务容量。开发期 P99 上升1.37%及旧自由时钟 **UTILITY_HOLD** 均保留。

下一步回到主场景调用入口，把真实计算时间纳入帧时长与容量定义，冻结相应时钟协议后再验证端到端效果。目前该算法仅由实验调用，独立确认支持计算收益，尚未建立流时间收益。

证据：[汇总](experiment_artifacts/aligned_hardware_small_dimension_exact_native_common_cadence_confirmation_20260929.json)、[协议](experiment_artifacts/aligned_hardware_small_dimension_exact_native_common_cadence_confirmation_20260929_protocol.json)。四 closed cells、协议、汇总和日志共35.57 MiB；本轮无服务器拉取。
