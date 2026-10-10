本轮是 scheduler 的 MaxWeight 帧候选选择算法开发。线上入口仍为 [scheduler_algorithm/bridge.py](/home/researcher/mine_code/scheduleurm/skill/scheduler_algorithm/bridge.py:54)。

新增 [fused selector](/home/researcher/mine_code/scheduleurm/algorithm/fused_root_dual_face_frame_candidates.py)：在同一次初始化中完整保存真实 raw-M 快照、执行条件根结算；快照改为一次原始 heap 扫描，复用静态根记录。动作、LP/整数预算和后续搜索保持父版行为。

47 项定向测试通过。完整 2802 状态质量实验为 `FUSED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE`，错误和未闭合状态均为零；LP18932、整数状态35553、branch7668 与父版一致。原始根扫描9124→4562、结算扫描4680→2340，retained-list 扫描均归零；1234 状态免费闭合，1031 状态继续根 probe。

九臂计时固定 seed2026092903、六轮、CPU0/BLAS1/GC开启：151308 次计时＋288 次 warmup，全16配置完成。动作、证书、收费、重复路径、谱系及 affinity 检查全部通过。结论为 `FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD`。

比值为候选÷对照；总体采用各状态六轮中位耗时之和。

| 对照 | 总体 wall 比值 | 历史最坏状态比值 |
|---|---:|---:|
| settled | 0.997595 | 1.065229 |
| native | 1.001275 | 1.124530 |
| zero | 0.738546 | 1.078302 |
| long | 0.433535 | 0.989209 |

六项失败：相对 settled/native/face/early/zero 的历史逐状态1.05门槛，以及 face 的原 independent-source 组 P99=1.072259。全部 trigger 门槛通过，trigger/long=0.762501。融合相对 settled 的本轮差异仅0.2405%，尚不支持稳定加速。

已查具体历史失败为 gpu_b/R32、Q=[61,0,35]：候选/settled/native/zero 的 wall 中位数为225.675/211.856/220.513/209.287ms，CPU为222.671/209.673/215.086/208.236ms；四者均LP142、branch89、整数604，stage=initial、settle=null、face工作为零。此处没有新增根结算或对偶面证明，现有记录不能把退化归因于它们。只读取两个本地 cell 定位路径。

下一步改原生搜索箱域：当箱自身的精确对偶上界等于当前最优分数时，直接固定必须取边界的变量，在收紧后的原生箱内继续 canonical 搜索，减少重复域准备与搜索。先验证等价性，再冻结144/256/8192预算、完整队列、九臂门槛和全部新增费用，不按本轮队列或耗时设分支条件。

局限：本轮队列均已使用，四个缓存目录及虚拟 lanes 未改变；原 cohort 名称不代表新独立确认，现有记录不足以隔离历史回退的执行成本与测量因素。新选择器尚未接入线上，论文未改动，所有旧HOLD及原始INVALID结果保留。

完整证据：[质量汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_fused_root_dual_face_development_20260928.json)、[计时汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_fused_root_dual_face_timing_20260928.json)，对应 protocol、16个cell与运行日志均已保存。
