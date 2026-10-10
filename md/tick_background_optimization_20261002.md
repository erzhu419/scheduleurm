# 背景到达整数时钟优化（2026-10-02）

新增tick kernel和串行/并行persistent执行类：背景出生时间仅实际更新时转换一次，释放与入队循环直接使用整数时钟，返回原精确Fraction评分。原冻结源码与运行任务保留。

服务器5项测试通过；四个真实固定状态的完整决策、精确评分、全部逻辑费用及三层随机/因果状态均一致。四个完整决策合计191.44→105.43秒（1.816倍），记录CPU1657.93→897.44秒（1.847倍）；重计算片三次AB/BA/AB配对合计提速1.734倍。本轮全部任务exit0，仅取回15KB结果。

保留新kernel用于后续候选评估，预算仍8/cap128。原持续续排完整闭环已24/24、执行检查通过，裁决HOLD：相有限父算法均值+0.9066%、P90+2.3136%。下一步沿用已有动作状态做后续控制器交叉诊断，见[完整闭环结果](persistent_current_policy_development_20261002.md)。

计时范围：完整决策wall含首次进程池启动，CPU为父进程与worker函数之和，不含子进程启动/import。四个完整决策采用固定先旧后新顺序；重片交替顺序。上述提速适用于这些固定状态，尚无整轨迹提速测量。

证据：[结果](experiment_artifacts/aligned_hardware_tick_background_rollout_20261002_summary.json)、[完成记录](experiment_artifacts/aligned_hardware_tick_background_rollout_20261002_runner_final_receipt.json)。实现：[kernel](../algorithm/tick_background_rollout.py)、[执行类](../algorithm/tick_persistent_current_policy_lane_rollout.py)。
