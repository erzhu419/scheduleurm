# Current-policy 开发试验并行迁移（2026-10-01）

本地任务 t116636 已按用户要求取消，相关进程全部退出。原协议和日志保留；日志记录3/24完成，但驱动未逐条落盘，故这3条不复用，24条均在服务器重新执行。原本地部分计算的工作量不可恢复，单列为中止执行。

新任务 **t116848–t116871** 共24片，启动时 node001–node006 各4片。每片 CPU1、RAM1024MB、GPU0，执行同一 tape 的原生 MW 与候选配对；两种策略共48条轨迹。六节点共享目录：`/home/hpcuser/scheduleurm_work/two_step_current_policy_sharded_20261001`。

算法、4个缓存 catalogue、种子、8样本/cap128、出生 cohort 与 MW/BG 性能门槛沿用原科学注册；分布式执行另行注册。每片先持久化轨迹，再写完成行；完成片重试直接复用，未完成片可重新执行。汇总只读24个小结果 JSON，按原 roster 排序；轨迹 gzip 留服务器端，未配置自动全量回拉。

新增5项测试通过：汇总与原驱动等价、字面顺序、原生费用一次计数、缺片/重复片拒收、持久化重试。服务器入口与两份缓存参考加载通过；代码包补齐了包初始化的实际依赖。

下一步：24片完成后在服务器执行以下汇总，再读取小结果比较 mean flow、P90、900/1200秒未完成数；当前仍未形成新性能结论。论文未改。

```bash
cd /home/hpcuser/scheduleurm_work/two_step_current_policy_sharded_20261001
/home/hpcuser/scheduleurm_work/conda_envs/scomp-py310/bin/python -m algorithm.experiments.aligned_hardware_two_step_current_policy_sharded --merge
```

[分布式协议](experiment_artifacts/aligned_hardware_two_step_current_policy_sharded_20261001_protocol.json) · [启动记录](experiment_artifacts/aligned_hardware_two_step_current_policy_sharded_20261001_launch.json)。
