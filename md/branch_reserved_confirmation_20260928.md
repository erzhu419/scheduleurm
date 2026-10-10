独立新队列确认通过：`BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE`。10 项定向测试通过，16 个单元、4146 个状态全部完成；执行 41460 次选择接口调用和 16584 次原生搜索构造。规则与 144/256/8192 路由预算全程冻结。

新面板包含 4096 个状态，排除既有静态开发与上一轮确认队列；另保留原 34 个历史回归和 16 个零队列控制。新增 4112 次 full-K4096 参考求解，其中 2 次按冻结规则用独立 score-face oracle 补齐规范动作，共访问 5013 个整数状态；34 个历史参考复用。参考与 oracle 成本单独记账。

新队列的实际 routed 工作量：

| 指标 | 原算法 | 分支配额保留 | 变化 |
| --- | ---: | ---: | ---: |
| LP 调用 | 51250 | 34176 | −33.32% |
| LP 矩阵系数 | 2556555 | 1648388 | −35.52% |
| 分支展开 | 13145 | 13145 | 不变 |
| 容量坐标读取 | 1730592 | 1730592 | 不变 |

全部最终 routed 动作匹配有效独立参考；四个内核预算点均无评分损失，规范闭合数量与原算法相同。预算预留审核全部通过，逐状态 LP、矩阵与整数搜索量均未增加，16 个新队列单元各自的成本门槛通过；零队列收益未计入新队列收益。

34 个历史回归的独立门槛也通过：LP 3085→2988，矩阵 243251→238217，整数搜索 24768 不变。此前失败的 GPU2/R32 矩阵 40848→40532，GPU2/R64 矩阵 69939→69331。

局限：这是固定采样规律、四个缓存目录和虚拟 R8/16/32/64 下的工作量证据；1024 个不同 R/Q 样本跨目录配对，不能视为 4096 个独立观测。尚无耗时、真实物理扩容或流时间改善结论。原 root-pruning confirmation HOLD 与 certificate-routed timing HOLD 保留。

下一步冻结完整选择器的配对计时，比较新方法、原 routed 对照及原短/长预算基线，沿用既有成功门槛并单独检查 trigger/tail。论文继续暂缓修改。

证据：[汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_branch_reserved_confirmation_20260928.json)、[冻结协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_branch_reserved_confirmation_20260928_protocol.json)及同前缀 16 个压缩 cell。本轮仅使用本地缓存，无服务器拉取。
