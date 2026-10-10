分支 LP 配额保留开发实验通过：`BRANCH_RESERVED_DEVELOPMENT_ADVANCE`。738 个冻结状态全部完成，20 项定向测试通过；执行 7380 次选择接口调用、2952 次搜索构造，全部参考复用，无新参考求解或服务器拉取。

实现只改变续搜条件：实际 LP 次数加跳过根数必须小于预算，保留原 `budget−root_count` 分支 LP 配额。真实 LP 仍按实际执行计数，包含失败调用；同一前沿持续续跑。

722 个非零队列的 routed LP 调用从 16645 降至 14034（减少 15.69%），矩阵系数从 1073991 降至 942478（减少 12.25%）；容量检查和整数搜索量不变。全部最终动作匹配参考，四个预算点均无评分损失，闭合数量与旧算法一致；逐状态 LP、矩阵和整数搜索量均未增加。

34 个历史回归的独立成本门槛通过。之前失败的 GPU2/R32 矩阵量由旧算法 40848 降至 40532，GPU2/R64 由 69939 降至 69331；省下的根 LP 不再兑换为额外昂贵分支求解。

局限：本轮复用开发与历史回归队列、缓存目录及虚拟资源规模，没有新队列、耗时或流时间改善结论。原根预剪枝确认 HOLD 与 certificate-routed timing HOLD 保留，论文未修改。

下一步冻结新的独立队列面板，保持当前规则、144/256/8192 路由预算和四个内核预算点，重新确认质量与工作量；继续单独检查 34 个历史回归。

证据：[汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_branch_reserved_development_20260928.json)、[冻结协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_branch_reserved_development_20260928_protocol.json)及同前缀 16 个压缩 cell。
