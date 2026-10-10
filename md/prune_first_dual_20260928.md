本轮结论：开发验证 ADVANCE，完整配对计时 `PRUNE_FIRST_DUAL_LOCAL_TIMING_HOLD`。

新增搜索先用继承上界和规范动作检查剪枝，只对存活子节点重算父节点对偶界；界改善后再检查剪枝。新增 residual selector 接入该搜索，沿用原返回结构、预算和累计计数。

36 个新增定向测试通过。冻结原 754 状态、四级 native 快照和 144/256/8192 预算；开发实际执行 3016 次 native advance、754 次独立冷启动 selector，动作、搜索路径和证书全部保持一致，无对偶费用增长或参考不匹配。整条 selector 的 child evaluations 从 14485 降至 12378（−14.55%），canonical checks 从 20019 增至 22910（+14.44%）；LP 和整数证明费用不变。

计时完整执行 13572 次三组调用和 96 次预热，六次交替配对，执行、证书、参考、重复路径、CPU 亲和性及冻结结果复现全部通过。新 reference/oracle/screen 调用均为零，借用费用单列。

| 组 | 状态数 | candidate / branch | candidate / long |
| --- | ---: | ---: | ---: |
| 普通开发 | 688 | 0.8391 | 0.6983 |
| 历史回退 | 34 | 0.9777 | 0.6168 |
| 合并触发 | 16 | 1.0015 | 0.7758 |

四个旧失败状态的当前 candidate/branch 比率为 1.0233、1.0251、1.0336、1.0369，均回到原 1.05 门槛内。但新增两个超限状态：gpu_a/R64/Q(118,119,269) 为 1.052315，gpu_b/R32/Q(49,4,26) 为 1.074481。原历史逐态、触发逐态门槛失败，其余预设门槛通过，保留本轮及旧 HOLD。

限制：复用开发及条件触发队列、四个缓存目录和虚拟容量，仅为本地描述性计时。普通组通过的是总体、各容量和 p99 门槛，其单态最大比率仍为 1.1183；不构成普遍加速、独立队列确认、physical scaling 或流时间收益。GPU2/Q(49,4,26) 还比 branch 多 4 次分支、8 次容量传播和 17 次 canonical 检查。

下一步将每张父对偶证书的目标预存为上下界的精确仿射式，子节点按变化坐标更新，并计入准备费用；保留原质量及计时门槛，另冻结新实验。论文保持原样。

结果：[开发汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_prune_first_dual_development_20260928.json)、[计时汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_prune_first_dual_timing_20260928.json)。对应 protocol、16 个 cell gzip 和运行 log 均保存在同目录。
