短 LP 前瞻开发完成，判定 **SHORT_LOOKAHEAD_DEVELOPMENT_HOLD**。25 项定向测试通过；16 个 cell、738 态全部完成，两路独立执行共 1476 次。动作、证书、参考匹配及计费检查通过，无新增参考或 oracle 求解。

仅在初始评分已证实、规范动作未闭合且剩余区域全未求 LP 时逐次前瞻；最多支付初始区域数的 LP，遇到活的已解区域立即停止。规则、738 态名单及 144/256/8192 预算在运行前冻结。

| 722 个非零态的累计工作 | 原 branch-reserved | 短前瞻 |
| --- | ---: | ---: |
| LP 调用 | 14034 | 14036 |
| 矩阵系数 | 942478 | 942651 |
| 整数状态 | 24768 | 16576 |

三个状态启用前瞻。GPU/R64 的 Q=(153,147,290) 用 5 步闭合，省去 8192 个整数状态，最终 LP/矩阵工作不变；Q=(81,31,48) 保持原路径，仍访问 544 个整数状态。

两个反例各增加一次 LP：GPU/R64 Q=(39,3,313) 增加 77 个矩阵系数，整数探测仍为 179 态；GPU2/R64 Q=(145,1,67) 增加 96 个矩阵系数，仍为 5260 态。两者第一步均产生存活的已解区域。它们使两个 cell 的非零与独立历史子组成本门槛失败；34 个历史态合计同样增加 2 次 LP、173 个矩阵系数。整数节省不抵消这些增长，故不推进确认实验。

下一步检验父 LP 对偶证书的复用：按子区域的新上下界重算安全上界，在调用新 LP 前剪枝。现有内核只保留父区域的标量上界与 primal；本轮已否定仅凭“全未解”选择前瞻，不再按区域数调阈值。

局限：这是原开发队列上的结构实验，不支持新队列或墙钟收益结论。旧 timing HOLD 保留；论文未改，无服务器拉取。

证据：[汇总](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_short_lookahead_development_20260928.json)、[冻结协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_short_lookahead_development_20260928_protocol.json)及同前缀 16 个压缩 cell。
