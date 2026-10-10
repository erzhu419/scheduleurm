# Small-dimension exact native 独立确认：2026-09-29

结果为 **SMALL_DIMENSION_EXACT_NATIVE_INDEPENDENT_TIMING_ADVANCE**。新 1024 个状态相对 direct 的总体 wall 耗时降低 **25.65%**、CPU 耗时降低 **25.62%**、P99 降低 **8.17%**；原 155 个门槛、新组 33 个非退化门槛及至少 5% direct 收益门槛全部通过，共 189 项。

算法、十二个方案及原预算保持冻结。登记前核查未使用的四个抽样种子 2026092911–14，计时顺序种子为 2026092915；沿用原条件均匀抽样规则，排除当前全部旧队列及历史 discovery 未跑尾部。新 1024 个目录状态对应 256 个不同 R/Q 抽样，在四份缓存目录上配对；旧 2802 个状态及七组原 metadata 完整保留，单列回归控制。

57 项针对性测试通过。质量实验完成 1024 次独立 K4096 参考，全部闭合，无需 oracle；完成 12288 次 whole selector、24576 次 native 推进，共 36864 次方法接口、18432 次冷搜索构造。失败和未闭合均为 0。

| 新组 whole-selector 工作 | candidate | direct | own |
|---|---:|---:|---:|
| 逻辑 LP | 4237 | 4224 | 4224 |
| native 分支 | 331 | 331 | 408 |
| 整数访问 | 1537 | 1537 | 84 |

candidate 的 4237 次逻辑 LP 为 2179 次通用求解和 2058 次精确求解，后者的 0/1/2 维次数分别为 15/1040/1003。通用 LP 相对 direct 减少 48.41%，逻辑 LP 增加 13 次；精确证书改变了部分端点固定及后继工作。六条独立 native 检查链的成本与上述 whole-selector 成本分别保存，旧借用费用没有混入新组成效。

16 个配置、12 个方案、6 次重复全部结束：275472 次计时、384 次预热，共 275856 次选择器调用。执行、证书、参考、质量、重复路径、CPU 亲和性及冻结结果复现检查全部通过，计费和质量失败均为 0。总体耗时采用逐状态六轮中位数之和。

| 新组 candidate/对照 | wall 比 | CPU 比 | P99 比 |
|---|---:|---:|---:|
| direct | 0.743453 | 0.743763 | 0.918301 |
| own | 0.743291 | 0.743506 | 0.921320 |
| fused | 0.674235 | 0.674394 | 0.683424 |

新组 16 个配置相对 direct 的总体 wall 比为 0.657640–0.817876，均改善；最小值在 Linux/R64，最大值在 GPU2/R8。新组最大单状态 wall 比为 0.976916。旧历史组、触发组相对 direct 的最大单状态比为 0.999837、1.003807，原 1.05 门槛均通过；触发组相对 long 的总体比为 0.722700，原至少 5% 收益门槛通过。

下一步将通过独立确认的内核接入主场景 replay：固定动作规则、到达/服务轨迹与预算，显式计入候选选择耗时，验证动作、证书及端到端影响。当前新内核仍由独立实验入口调用。

证据：[质量汇总](experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_20260929.json)、[计时汇总](experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_timing_20260929.json)、[质量协议](experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_20260929_protocol.json)、[计时协议](experiment_artifacts/aligned_hardware_small_dimension_exact_native_confirmation_timing_20260929_protocol.json)。

## 已知限制

旧零队列组相对 direct 的总 wall 耗时增加 2.23%，最大单状态增加 15.93%；旧普通组相对 own 最大单状态增加 13.40%。零队列速度为描述性指标，普通组原协议没有逐状态速度门槛，这些回退完整保留。独立确认限于固定抽样规则、四份缓存目录、虚拟 lanes 和本地 CPU；目录配对和重复计时不是额外独立观察，没有新增 physical scaling、流时间或统计显著性结论。此前所有 HOLD 保留。
