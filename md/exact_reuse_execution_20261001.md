# 精确执行加速（2026-10-01）

新增完整内层决策复用、三层共享MW队列缓存、最深层BG精确整数时钟与S1/S2评分。保留8样本/cap128、物理服务时钟、根选择和原性能门槛；随机生成顺序不变。原冻结模块和仍在运行的24片任务未改。

四个固定真实开发状态的原生/优化完整proposal配对验证通过：返回决策及逻辑费用、精确Fraction评分、三层随机/计数状态全部一致。

|模型|原生CPU秒|优化CPU秒|加速|
|---|---:|---:|---:|
|cpu_a|159.92|43.44|3.68×|
|node007|235.15|50.93|4.62×|
|gpu_a|269.92|72.99|3.70×|
|gpu_b|358.69|104.10|3.45×|

合计1023.68→271.46 CPU秒，3.77×。完整Linux/.95/seed42轨迹也通过服务器内严格对照：两策略全部动作、任务与科学结果行相等；候选242帧、354任务。新整片wall/CPU为642.06/640.53秒（10.70分钟）；旧launch至完成行记录约52.26分钟，近似wall加速4.88×，包含加载、检查与落盘。四状态CPU配对为直接计时；整片旧wall来自历史元数据，范围不同，不作24片统一提速承诺。

21项定向测试通过。实际复用账目与冻结逻辑费用分列，在检查器扫描前取执行账目快照；汇总计数求和、缓存峰值取最大。只拉小JSON，完整轨迹留服务器。

完整评估入口：`python -m algorithm.experiments.aligned_hardware_exact_reuse_sharded --run-shard --shard-index N`，N=0..23；使用已登记的同名协议，最后`--merge`。后续迭代使用该入口。下一处热点是背景到达检查：本次四状态1667万次release仍至少重复构造3334万次Fraction，可进一步改为缓存整数时刻的比较。

[四状态结果](experiment_artifacts/aligned_hardware_exact_reuse_benchmark_20261001.json) · [完整轨迹对照](experiment_artifacts/aligned_hardware_exact_reuse_sharded_20261001_parity04.json) · [快速分片协议](experiment_artifacts/aligned_hardware_exact_reuse_sharded_20261001_protocol.json)。

后续切换（同日）：按用户指令取消旧版16片未完成任务，六节点旧Python worker均已退出，已有输出保留。24片固定评估选用7片旧完成结果（00、01、02、05、10、16、22）、已严格对照的优化版04，以及新启动的16片优化结果；任务t118665–t118680分布在node001–node006，前四节点各3片、后两节点各2片。原8样本/cap128、开发数据和服务性能门槛不变。

新增handoff入口固定上述来源，每条数据只计一次，仍调用原科学汇总；两项针对混合来源的测试通过。优化版实际执行费用只统计17片，旧7片原生费用另列；取消任务未落盘的工作量不可恢复，不据此推算24片总体加速。

本阶段来源已[登记](experiment_artifacts/aligned_hardware_exact_reuse_handoff_20261001_protocol.json)。随后按用户进一步并行要求，未完成的16片及等待任务t118711已由[决策内并行入口](parallel_root_execution_20261001.md)替换，已完成结果保留；新的最终汇总以该入口为准。
