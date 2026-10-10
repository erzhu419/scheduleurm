# Direct-probe own-tight native：2026-09-29

本轮修改 scheduler 的 MaxWeight 帧候选选择：保留 own-tight 裁域，根阶段在构造分支子箱前停止，把完整当前箱域交给一次整数 probe；probe 耗尽后由同一 native 搜索继续。预算仍为 144/256 LP、8192 probe，参考预算和全部对照不变。

结果：质量通过，完整计时仍为 **DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD**。

51 项针对性测试通过。冻结后完成 2802 个状态、14010 次质量接口调用、5604 个搜索实例，失败与未闭合均为 0；16 个配置、11 个方案、6 次重复完成 184932 次计时与 352 次预热，共 185284 次选择器调用。执行、证书、参考、质量、重复路径、CPU 亲和性和冻结结果复现检查全部通过。

| 质量实验搜索工作 | 上一轮 own | 本轮 |
|---|---:|---:|
| LP | 17607 | 17607 |
| native 分支 | 4106 | 3873 |
| 整数检查 | 25274 | 26429 |
| 根 probe | 232 | 233 |
| 根阶段闭合 | 2033 | 2032 |

总体耗时采用逐状态六轮中位数之和。本轮/own 的 wall 比为 0.999701、CPU 比为 0.999117，增量收益接近零；本轮/fused 的 wall 比为 0.911521、CPU 比为 0.911559。所有总体、分配置及 P99 门槛通过。

11 个逐状态门槛失败：历史组相对 own、fused、settled、native、face、early、zero、affine、branch 的最大 wall 比分别为 1.073625、1.181952、1.170706、1.089590、1.069205、1.054891、1.201198、1.083492、1.059506；触发组相对 own、zero 分别为 1.064888、1.053583，门槛为 1.05。

工作交换已经确认：历史 GPU2/R64 Q[184,78,235] 的整数检查从 1 增至 9，Q[88,19,118] 从 1 增至 6。旧方案的一次分支及容量传播能生成固定箱，立即剪枝；本轮保留较大箱域后需要更多 count 转换，两例 LP 与动作保持一致。

目标 GPU2/R64 Q[122,3,4] 的本轮/own wall 中位数为 6.866580/7.765044 ms，CPU 为 6.678080/7.545384 ms；wall 降低 11.57%，分支 1→0、整数检查 304→101。fused、face、early 的 wall 分别为 7.704082、6.781682、6.738627 ms。

本轮相对 own 的历史最大退化是 Linux/R64 Q[184,78,235]：wall 2.794255/2.602636 ms，CPU 2.487812/2.442166 ms。触发最大退化是 Linux/R64 Q[192,192,317]：wall 2.386326/2.240918 ms，CPU 2.323667/2.183055 ms。两例均 phase=None，LP=2、branch=0、整数检查=0；raw-M 完整记录与 own 一致，新增停止分支逻辑未执行。

其他失败点为 GPU2/R64 Q[88,19,118]（native、face），GPU2/R32 Q[49,4,26]（early）、Q[61,0,35]（branch），Linux/R8 Q[27,15,34]（affine）。Q[61,0,35] 仍需要 142 次 LP、446 次 own 检查和 3256 次变量访问；本轮与 own 的完整记录相同，未降低该慢路径的工作量。

下一步优先分析并减少 raw-M 阶段的 LP 与 own-tight 裁域成本，再冻结下一版质量及全状态计时协议。当前主要失败点没有进入新增根阶段逻辑，单独继续改该处 probe 路由不能解决这些失败点。

证据：[质量汇总](experiment_artifacts/aligned_hardware_direct_probe_own_tight_native_development_20260929.json)、[计时汇总](experiment_artifacts/aligned_hardware_direct_probe_own_tight_native_timing_20260929.json)、[冻结计时协议](experiment_artifacts/aligned_hardware_direct_probe_own_tight_native_timing_20260929_protocol.json)。

## 已知限制

2802 个状态都已用于开发，沿用的 cohort 名称不构成新的独立确认；四份缓存目录、虚拟 lanes 与本地重复计时没有新增 physical scaling 或流时间证据。相同保存路径及计数不足以确定耗时差的原因或推翻 HOLD。历史 HOLD 全部保留，本轮尚不支持替换现用算法。
