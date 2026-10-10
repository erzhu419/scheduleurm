# 分数变量优先分支开发比较（2026-09-27）

新增独立 selector，仅将首个未固定变量改为优先选首个分数 LP 变量；无分数变量时沿用原顺序。预算、候选构造、精确证书及原版源码保持不变。

冻结后完成 12/12 个 node/R 单元：复用开发网格及上一轮确认队列，共 1143 个 (R,Q)、4572 次节点—状态比较。主预算 36R，full 参考预算 4096，R4 另用 Fraction 穷举。47 项针对性测试通过，所有参考闭合，证书检查通过。

- 新版 4572 次动作和评分全部匹配参考，4571 次完成 canonical 证明；原版为 4570 次证明、1 次动作偏差。
- gpu_a，R8，Q=(6,6,37)：动作已修正，288 次 LP 后仍有 20 个区域，评分证书 gap=0.20793818。
- gpu_b，R16，Q=(35,32,77)：514/576 次 LP 完成全部证明。
- 总 LP：76958 → 72064（−6.36%）；矩阵项：4461173 → 4113173（−7.80%）。
- gpu_a R4 出现退步：LP 7124 → 7139，矩阵项 429798 → 437048；全体共有 149 次状态的 LP 调用增加。

正式判定 `DEVELOPMENT_ACTION_REVIEW_REQUIRED`：精确性及逐单元成本门槛均未通过，暂不采用为默认规则。

一次原预算 R8 事后诊断显示：剩余区域含 19 个评分松弛和 1 个同分松弛；整数目标取整、gcd 格点取整及资源感知 canonical 下界均不能直接剪掉这 20 个区域。下一步先测试整数计数上界传播：根节点使用 `u_j <= floor(Q_c/a_cj)`，子区域扣除 mandatory lower counts 后使用剩余队列容量；例如 37 个 LLM 任务只能组成至多 4 个大小为 8 的批次。当前 LP 松弛仍允许超过 4 的分数批次数，尚未验证收紧后的全量表现。

限制：本轮全部为复用队列的开发结果，未做独立新队列确认，也不构成墙钟时间或流时间收益证据。论文未修改。

证据：[冻结协议](experiment_artifacts/aligned_hardware_fractional_shared_threshold_development_20260927_protocol.json)、[完整结果](experiment_artifacts/aligned_hardware_fractional_shared_threshold_development_20260927.json)、[独立事后诊断](experiment_artifacts/aligned_hardware_fractional_shared_threshold_failure_diagnostic_20260927.json)。
