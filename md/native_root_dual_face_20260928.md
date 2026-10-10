2026-09-28：native 根快照消除重复目录扫描；开发质量 ADVANCE，七臂计时 HOLD。

[新 native](/home/researcher/mine_code/scheduleurm/algorithm/native_root_zero_delta_affine_dual_reuse_integer_capacity_frame_candidates.py:224) 在自身分组后取得实际 M，返回原始根快照，再沿同一搜索继续 144/256。[新 selector](/home/researcher/mine_code/scheduleurm/algorithm/native_root_dual_face_frame_candidates.py:25) 取消外部根预算扫描，保留原对偶补全和唯一 8192 整数预算。旧源码及实验保持冻结。

60 项定向测试通过，执行前登记全部源码、2802 状态、预算和七臂六轮顺序。

- [开发结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_native_root_dual_face_development_20260928.json)：2802 次实际执行全部通过，动作、原始快照、LP/probe 路径、对偶证据及费用与父版本一致，唯外部扫描的 8 项费用为实际 0；失败及不闭合均为 0。
- 取消 24520 次选项访问、73560 次需求坐标读取等重复操作；native 准备与 2802 次根快照费用仍保留。LP 总量 18932、整数状态 38704，与父版本相同；2265 次根补全均完成，旧 52 个耗尽状态没有恢复。
- [计时结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_native_root_dual_face_timing_20260928.json)：117684 次计时加 224 次预热，实际执行 117908 次；质量、证书、路径、计数及 CPU 亲和性检查全部通过，失败为 0。

比例为各状态六轮中位数的汇总墙钟耗时，以及状态中位数的 P99：

| 本轮候选 / 对照 | 汇总耗时比例 | P99 比例 |
| --- | ---: | ---: |
| 上一版 face | 1.0017 | 1.0090 |
| zero | 0.7370 | 0.9084 |
| early | 0.8750 | 0.9995 |

本轮没有观察到相对父版本的增量提速。历史/trigger 单状态门槛仍失败：cpu_a/R8、Q=[27,15,34] 相对 zero 为 1.0867；gpu_b/R64、Q=[120,123,258] 相对 face 为 1.0927。后者双方均为 34 LP、0 整数状态、无补全，路径一致但实测仍退化。face、zero、affine 的历史和 trigger 门槛，以及 early 的 trigger 门槛未通过；branch/long 门槛通过。因此保留 HOLD，暂不采用。

旧耗尽反例 Q=[122,235,280] 本轮 candidate/face/early/zero 中位耗时为 5.628/5.507/46.234/17.967 ms，保留此前的尾部改善，但取消扫描没有进一步提速。最坏历史反例 candidate/zero 均用 2 LP，整数状态却为 3/0；另一个旧 trigger Q=[192,192,317] 的 zero 经一次 native 拆分和剪枝、无需新 LP 即可闭合，候选仍先支付根补全费用。

下一步在根评分已证明但 canonical 未闭合时，先推进同一 frontier 上无需新增 LP 的操作，遇到首个存活且需要 LP 的箱域即停，再决定是否补全。孩子均为 unsolved，这个阶段至多拆分 M 个已付 LP 的根；实际 heap、capacity、dual 和 guard 费用全部计入，保持原预算与门槛。

范围：本轮只有旧开发状态和四个缓存目录上的本地证据，没有新的独立确认、physical scaling 或流时间结论。旧 HOLD、用户已有改动和论文均保留；未从服务器拉取数据。
