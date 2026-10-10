本轮结论：`AFFINE_DUAL_DEVELOPMENT_ADVANCE`；完整计时 `AFFINE_DUAL_LOCAL_TIMING_ADVANCE`。

新增搜索保存每张父对偶证书自身的精确目标值，在首次实际复用时准备共享整数仿射系数；子节点扫描上下界，只在变化坐标更新目标。较紧的继承上界独立保留，失败 LP 继续继承原证书。17 项实际费用包括证书目标保存、系数准备、扫描、系数读取、整数更新和有理转换。

58 个新增定向测试通过。原 754 状态、四级 native 快照、144/256/8192 预算保持；完整开发执行 3016 次 native advance 和 754 次独立冷启动 selector。动作、搜索路径、分支、LP、整数证明、规范动作及四项对偶结果计数保持一致，无不匹配或费用审计失败。整条 selector 实际准备 7218 张证书，执行 12378 次仿射边界转换，扫描 85495 个变量、更新 30462 个上下界坐标；所有准备费用单列。

完整三组计时执行 13572 次调用和 96 次预热，全部原执行、证书、参考、重复路径、CPU 亲和性、冻结结果复现及计时门槛通过。新 reference/oracle/screen 调用均为零；旧 HOLD 保留。

| 组 | 状态数 | candidate / branch | candidate / long |
| --- | ---: | ---: | ---: |
| 普通开发 | 688 | 0.8327 | 0.6933 |
| 历史回退 | 34 | 0.9841 | 0.6169 |
| 合并触发 | 16 | 0.9892 | 0.7732 |

六个关注状态当前全部 ≤1.05：GPU/R64 Q(88,8,23) 1.0360、Q(118,119,269) 1.0103；GPU2/R32 Q(49,4,26) 1.0231；GPU2/R64 Q(101,9,59) 1.0259、Q(85,34,58) 1.0211、Q(122,3,4) 1.0243。历史 34 状态及触发 16 状态的每个状态均通过原门槛。

限制：这是复用开发及条件触发队列、四个缓存目录和虚拟容量的本地描述性计时。普通组单态最大 candidate/branch 比率仍为 1.1108；16 个零队列状态总体为 1.0723，零队列质量通过，其时延依原协议仅作描述。不同单位的算术费用不能相加替代计时。本轮不构成独立普通队列确认、普遍加速、physical scaling 或流时间收益。

下一步另冻结新的普通队列种子，以同预算、完整质量和配对计时做独立确认。论文保持原样。

代码：[仿射搜索](/home/researcher/mine_code/scheduleurm/algorithm/affine_dual_reuse_integer_capacity_frame_candidates.py)、[residual selector](/home/researcher/mine_code/scheduleurm/algorithm/affine_dual_residual_capacity_frame_candidates.py)。结果：[开发汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_affine_dual_development_20260928.json)、[计时汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_affine_dual_timing_20260928.json)；对应 protocol、16 个 cell gzip 和日志均保存在同目录。
