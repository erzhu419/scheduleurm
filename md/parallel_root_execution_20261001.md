# 决策内并行（2026-10-01）

24条冻结轨迹继续按原协议评估，真实帧串行推进。每次决策将每个可行根动作的8样本拆成4组，每组2样本；目录8/9个动作对应最多32/36个计算片。每条轨迹使用持久22-worker进程池，scheduler声明CPU23、RAM6144；按六节点实际空闲物理核规划，16条最多368核。

新增策略3项与混合来源汇总1项测试通过。四个原先固定的真实状态又通过12项检查：完整决策、精确Fraction评分、原逻辑费用和三层随机/因果状态均一致。按原根动作与样本顺序截断首次overflow；实际投机计算另计，背景共享样本的原费用保留。

|真实模型|串行优化秒|并行秒|加速|
|---|---:|---:|---:|
|cpu_a|38.06|4.07|9.35×|
|node007|51.80|4.92|10.53×|
|gpu_a|74.47|6.82|10.92×|
|gpu_b|86.61|8.40|10.31×|

合计250.938→24.213秒，10.364×；测得CPU250.341→272.366秒，增加8.80%，平均占用11.25核。结果见[小JSON汇总](experiment_artifacts/aligned_hardware_parallel_root_sharded_20261001_anchor_summary.json)。

范围：以上为四个固定完整proposal的计时，wall包含池启动与传输，CPU包含父进程与微片函数（含worker planner构造），不含子进程启动/import。整轨迹提速及平均/P90流时间、未完成任务数仍以24条最终结果判断；原性能门槛不变。

正式切换：16个未完成的单核任务及旧汇总等待任务已取消，17个实际Python进程全部退出，已有输出保留。14:21:50 UTC重新登记时复用8片（原生7片、串行优化04片），其余16片由新进程池执行；完整协议留服务器，仅拉24条来源与登记时间的小摘要。

完整评估任务t118783–t118798及自动汇总t118799均已done/exit0。每个父任务CPU23/RAM6144，独立FullEval项目及Full22内存资源族，避免单核或短对照历史缩小内存预算。最终完整汇总约5MB，已单次取回本地；后续读取派生的[10KB摘要](experiment_artifacts/aligned_hardware_parallel_root_sharded_20261001_result_summary.json)，完整gzip轨迹留服务器。

启动后实际核验：16个父Python均进入新模块并推进candidate dispatch，每片22个spawn worker，总352个；node001–006分别66/66/22/66/66/66个，未见导入或CLI错误。每片进程RSS合计约1.40–1.45GiB。小状态见[worker记录](experiment_artifacts/aligned_hardware_parallel_root_sharded_20261001_full_worker_status.json)。

完整批次完成：16片用时最小12.18分钟、中位19.72分钟、最大30.07分钟；首片启动至全部检测完成30.27分钟（含调度器检测延迟，复用的8片不含在此时段）。执行检查全部通过，原联合性能门槛仍HOLD，详细算法结果见[开发结论](two_step_current_policy_bg_incumbent_development_20261001.md)。
