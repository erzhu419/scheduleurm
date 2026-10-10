# 整数计数上界传播（2026-09-27）

新增 [整数容量 selector](../algorithm/integer_capacity_shared_threshold_frame_candidates.py)：预留 mandatory lower counts 后，按剩余槽位和队列容量整除收紧每个计数上界。分数变量优先规则、LP 预算及证书计算保持不变，传播开销独立计数；每个阈值根仍支付一次 LP。86 项针对性测试通过。

固定四个缓存 physical catalogue 点模型、R=4/8/16，主预算 36R、full 参考预算 4096；R4 另有 Fraction 穷举。开发与确认均在各自协议冻结后执行完 12/12 个单元，所有参考和证书检查通过。

| 阶段 | 精确动作及 canonical 证明 | 原共享版 → 新版 LP | 原共享版 → 新版矩阵项 |
| --- | --- | --- | --- |
| 复用开发队列 | 4572/4572 | 76958 → 53115（−30.98%） | 4461173 → 2597584（−41.77%） |
| 独立新队列确认 | 3072/3072 | 53057 → 35888（−32.36%） | 3116468 → 1757416（−43.61%） |

新确认使用种子 2026092705–08，每种子/R 各 64 个未见队列，排除全部开发队列及重复抽样；768 个 (R,Q) 跨四节点配对，形成 3072 次比较。两阶段均通过相对原共享版和分数优先版的逐 node/R 成本门槛。相对分数优先版，新确认 LP/矩阵项分别减少 29.18%/40.56%。

两个开发反例已闭合：gpu_a R8 Q=(6,6,37) 使用 90/288 次 LP；gpu_b R16 Q=(35,32,77) 使用 97/576 次 LP。新确认中原共享版有 5 次未闭合、其中 1 次动作偏差；分数优先版有 3 次未闭合，新版均完成证明。

正式结论为 `INTEGER_CAPACITY_DEVELOPMENT_ADVANCE` 与 `INTEGER_CAPACITY_FRESH_QUEUE_CONFIRMATION_ADVANCE`。下一步固定此版本，进行包含冷预处理和整数传播的配对求解耗时实验，再推进轨迹回放。论文未修改。

限制：这是固定点模型和采样分布下的队列确认，跨节点比较共享队列，R 扩展仍是虚拟同质槽位。部分状态的 LP 成本增加：新确认相对原共享版有 37 次，相对分数优先版有 44 次；通过的是逐 node/R 总量门槛。新增传播的需求坐标读取为开发 1747536 次、确认 1174392 次，因此 LP/矩阵项下降尚不等于实际耗时或流时间改善。

证据：[开发协议](experiment_artifacts/aligned_hardware_integer_capacity_development_20260927_protocol.json)、[开发结果](experiment_artifacts/aligned_hardware_integer_capacity_development_20260927.json)、[确认协议](experiment_artifacts/aligned_hardware_integer_capacity_queue_confirmation_20260927_protocol.json)、[确认结果](experiment_artifacts/aligned_hardware_integer_capacity_queue_confirmation_20260927.json)。
