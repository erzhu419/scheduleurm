# 共同时间窗＋固定排空评分（2026-10-03）

已实现固定 `H=t0+2τmax` 的全部未完成作业积分＋无新到达 MaxWeight 排空评分。服务器 5 项测试通过，48 个原状态/动作/路径单元及合并全部真实退出码 0；8 条路径、cap128、原 BG 风险集合均保留。论文未改。

下表差值均为 parent−native；新评分为八路径平均累计作业秒，真实窗口为一个已有到达轨迹的累计作业秒。正值偏好 native。

| seed | 新评分差 | 真实同窗积分差 | 新评分去掉任一路径后的范围 |
|---|---:|---:|---:|
| 41 | +9.973 | +22.986 | [-0.183, +17.444] |
| 42 | +11.783 | +1.084 | [+5.922, +15.908] |
| 43 | +7.855 | +0.159 | [+6.667, +9.621] |

新评分仍全部偏好 native；seed43 的 CNN 收益 −3.913 被 matmul +2.691、LLM +9.077 抵消。实际同窗也全部偏好 native，而此前真实后续群体在 seed41/43 偏好 parent（seed43 平均流时间 −5.728 秒）。这轮仍未解决短期目标与后续排队收益的对应关系，保持完整开发 HOLD，不接入主选择器。

下一步只检验终端估值：保留当前窗口与评分作业，复用 H 状态，用持续采样到达参与 MW 竞争来完成窗口内作业；不重算昂贵的有限策略窗口续排。该项尚未实现，具体设计已[保存](experiment_artifacts/aligned_hardware_competing_terminal_flow_20261003_design.json)。

首次运行的真实三字段事件读取错误已修正，并补测公共入口；失败源码快照、退出记录及原批次结果保留。详细数据：[新评分分析](experiment_artifacts/aligned_hardware_common_horizon_flow_20261003_r1_analysis.json)、[实际窗口对照](experiment_artifacts/aligned_hardware_cached_common_horizon_bridge_20261003_summary.json)。

后续持续竞争终端评估已完成，见[结果与真实人口分解](competing_terminal_flow_20261003.md)。
