本轮结果：`RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE`，43 项测试通过。

规范动作补全复用已付容量表，以“至多剩余槽位”的上界剪枝，等号继续原顺序搜索。前缀与原搜索共用 8192 总预算，复制、覆盖和查询费用单列。

746 态、1492 次冷启动全部完成，最终动作均完成证明并匹配独立参考，逐态成本门槛通过。

原失败队列 `gpu_a/R64/[120,123,258]` 两臂均从 8192 状态耗尽变为 970 状态完成：评分证明 654、前缀 65、规范动作搜索 251。旧反例 `[54,58,290]` 保持 91+1921；其余状态保持原选择及既有开销。

下一步冻结新种子与触发队列，独立验证未见状态，保留全部旧失败控制和原预算。

局限：本轮复用参考且含已知失败队列；尚无独立触发确认、时间收益或真实容量 scaling 结论。此前 HOLD 保留。

算法：`algorithm/residual_capacity_frontier_frame_candidates.py`；协议、汇总和原始单元结果：`md/experiment_artifacts/aligned_hardware_residual_capacity_frontier_development_20260928*`。
