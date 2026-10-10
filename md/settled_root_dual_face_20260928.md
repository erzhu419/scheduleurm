2026-09-28：无新增 LP 的根推进已完成；开发质量通过，八臂计时仍为 HOLD。

[新 selector](/home/researcher/mine_code/scheduleurm/algorithm/settled_root_dual_face_frame_candidates.py:26) 在评分已证明但 canonical 未闭合时，先沿同一 heap 推进，遇到需要 LP 的存活子箱域即停；剩余箱域才进入原补全。原始 M、144/256 续搜和唯一 8192 整数预算保持不变。

68 项定向测试通过。原算法源码、2802 个开发状态、预算与八臂六轮计划均在新算法结果访问前登记；修正审计及计时入口在实际计时前另行冻结。

- [开发汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_settled_root_dual_face_development_20260928_checked.json)：2802 次真实执行，2265 次推进中 1234 次免补全；失败、不闭合及耗尽为 0。LP 保持 18932，整数状态 38704→35553（−8.1%），分支 5403→7668；新增 3355 次 head 检查和 2265 次推进快照，累计费用只计一次。
- 修复 3 条快照剪枝计数误报：比较累计计数前扣除两端的待处理剪枝项。仅重新分析原记录，新增 selector/native 调用为 0；[原 INVALID 结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_settled_root_dual_face_development_20260928.json)、算法、动作、费用及原审计均保留。
- [计时汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_settled_root_dual_face_timing_20260928_checked.json)：16 单元全部完成，134496 次计时＋256 次预热＝134752 次调用；动作、证书、费用、重复路径和 CPU 亲和性全部通过，错误为 0。

六轮状态中位墙钟耗时的汇总比例：候选/native=1.0024、候选/face=1.0036、候选/zero=0.7385，没有父版本增量提速。native、face、early、zero、affine 的历史及 trigger 单状态门槛均失败；native 在原 independent_fresh 来源组的 P99 比例为 1.0729。branch/long 全部通过，整体仍为 `SETTLED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD`。

最坏历史/zero 状态 node007/R8、Q=[27,15,34]：双方均 2 LP、0 整数状态，动作及 native 前缀相同，墙钟却为 2.995/2.572 ms，比例 1.1641。cpu_a/R64、Q=[192,192,317] 也已免补全，相对 zero 仍为 1.2134。

旧耗尽反例 gpu_b/R64、Q=[122,235,280]：候选/native 均 4 LP、补全 5 状态完成，墙钟为 5.692/5.374 ms；耗尽没有恢复，也没有增量提速。

下一步冻结根判定与推进快照的融合实验，减少重复扫描和构造，保持动作、推进前缀、预算及完整状态与门槛，对比当前 settled 与真实 native。快照成本尚未隔离，仍需验证。

范围：原开发队列、四个缓存目录及虚拟同质 lanes 的本地验证；没有新增独立确认、physical scaling 或流时间结论。论文、旧实验和用户已有修改均保留，未从服务器拉取数据。
