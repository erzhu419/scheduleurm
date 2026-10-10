新队列确认完成，结论为 `ROOT_PRUNING_FRESH_CONFIRMATION_HOLD`。4146 个注册状态全部执行：4096 个新队列状态、34 个历史回归、16 个零队列控制；24 项定向测试通过。执行 41460 次选择接口调用、4112 次新参考求解，另借用 34 个冻结参考，未触发参考 oracle。

新队列的 LP 调用从 51086 降至 33946（减少 33.55%），矩阵系数处理量从 2538723 降至 1629795（减少 35.80%），整数搜索均为零；16 个单元均通过新队列工作量门槛。全部最终动作匹配参考，证书和累计计费有效，四个预算点均无评分退化。

历史回归未通过单元级矩阵工作量门槛：

| 单元 | 旧矩阵系数 | 新矩阵系数 | 增量 |
|---|---:|---:|---:|
| gpu_b/R32 | 40848 | 41012 | +164 |
| gpu_b/R64 | 69939 | 70003 | +64 |

原因已定位到固定预算填满时的工作替换。例如 R32 的 Q=(61,0,35) 跳过两个空根 LP，省下 17 个系数，却在同一个 144 次上限内多求解两个各 96 系数的分支 LP，净增 175。完整闭合搜索反而少两次 LP、少 17 个系数。本轮保持冻结门槛和 HOLD。

下一步回到开发队列，测试保留原 `budget−root_count` 分支 LP 配额的规则，避免把跳过的根自动兑换为追加分支求解；真实 LP 和未闭合证书继续如实计费、记录。该规则尚未验证。

局限：新数据为固定采样律下的 1024 个 R/Q 样本、四目录配对产生的 4096 个状态，资源数仍为虚拟规模。本轮没有耗时、物理扩展或流时间改善结论，原 timing HOLD 保留；论文未修改。

证据：[汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_root_pruning_confirmation_20260928.json)、[冻结协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_root_pruning_confirmation_20260928_protocol.json)及同前缀 16 个压缩 cell。
