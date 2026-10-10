2026-09-28：根阶段对偶面 canonical 补全，开发质量 ADVANCE，完整计时 HOLD。

新增 [候选实现](/home/researcher/mine_code/scheduleurm/algorithm/dual_face_root_canonical_frame_candidates.py)：只在根阶段评分已证明、canonical 尚未完成时，使用当前根箱域自身的紧对偶证书固定整数边界，再搜索 canonical 动作。未耗尽的补全替代后续搜索；耗尽时保留已付费用，沿同一个 native 搜索继续原预算。

执行前冻结源代码、2802 个状态、144/256 LP 预算、单次 8192 整数状态预算及原计时门槛。四组定向测试共 62 项通过。

- [开发结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_dual_face_root_canonical_development_20260928.json)：2802 次实际 whole-selector 执行，质量和费用门槛通过，失败及不闭合均为 0；2265 次根补全全部完成，上一版 52 个耗尽状态全部消除。
- LP 总量由 early 的 19470 降至 18932；整数状态由 897240 降至 38704。Q=[122,235,280]、gpu_b/R64 从 13 LP/8192 整数状态降至 4/5。
- [完整计时结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_dual_face_root_canonical_timing_20260928.json)：六臂、每状态六轮，100872 次计时加 192 次预热，实际执行 101064 次；质量、证书、计数及重复路径检查全部通过。
- 各状态六轮中位数的汇总耗时，相对 zero 为 0.7361、相对 early 为 0.8760。两个旧 1024 状态组相对 zero 的 P99 比例分别为 0.9175、0.7872，旧 P99 退化已消除。
- Q=[122,235,280] 的中位耗时为 5.230 ms，early 为 43.501 ms，zero 为 17.358 ms。

剩余问题：historical 的 cpu_a/R32、Q=[96,15,13] 相对 zero 为 1.0876；trigger 的 cpu_a/R64、Q=[192,192,317] 为 1.1536，仍超过原 1.05 单状态门槛。因此计时维持 HOLD，暂不采用。历史反例所有臂均为 3 LP/0 整数状态，新对偶补全未触发；继承的外部根预算扫描与 native 目录分组重复。

下一步让新 native 在一次分组后返回实际 M 的根快照，随后沿同一搜索继续 144/256，取消外部重复扫描，全部准备费用仍计入 whole-selector，并沿用同一比较协议。

本轮仅有开发状态证据，尚无新的独立确认；收益来自根箱域复用与对偶约束的联合候选。旧代码、协议、结果及论文均未改动；未从服务器拉取数据。
