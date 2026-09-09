# ETA 精确估计与迁移动作空间缺口记录

日期：2026-06-29

本文档只记录当前结论、边界和下一步设计方向。本轮没有修改 `scheduler.py`、`algorithm/` 或实验逻辑。

## 1. ETA 修正的当前事实

上一轮 fresh ETA 实验闭合使用的是任务自身日志中的 `tqdm` / progress signal，而不是单纯使用 historical duration。

已经闭合的部分：

- `skill/eta_tracker.py` 的 ETA 估计优先解析 `tqdm` 自带的 remaining 字段，例如 `[00:42<03:21, 12.34it/s]` 中的 `03:21`。
- 如果有 `tqdm` 或可解析 progress，`runtime_projection` 使用当前日志中的 elapsed + remaining，而不是只看历史均值。
- `skill/scheduler.py` 的 running-task refresh 会 tail 任务日志并写入 `task["eta_seconds"]`。
- fresh ETA v6 实验文档中，node007 fresh rows 明确用 workload-native progress output 和 `ScheduleurmStableRate`，并用 `force_replace=True` 覆盖旧 node007 replay rows。

仍需诚实保留的边界：

- 如果任务没有 `log_path`、日志无法 tail、或日志里没有可解析的 `tqdm` / progress marker，live scheduler 仍会 fallback 到 runtime history 或 duration EWMA。
- 因此，实验闭合可以说使用了 `tqdm` stable ETA；线上所有任务不能无条件说都已经摆脱 historical ETA。
- 对新实验或生产任务，必须把主循环加上稳定的 `tqdm`，并保证 progress unit 对应真实工作量。

## 2. 任务差异与机器差异的当前事实

用户关心的具体例子是：RE-SAC 不同环境耗时差异很大，HalfCheetah 可能最慢；如果 `jtl311linux` 的 CPU / 主频最快，而且没有后续任务，最小化总完成时间时应倾向把 HalfCheetah 派给 `jtl311linux`。

当前实现只部分支持这个逻辑。

已经支持的部分：

- `algorithm/theorem_dispatch/service_registry.py` 可以通过显式 `workload_key`、`service_workload_key`、`theorem_workload_key` 或 `command_fingerprint` 绑定 measured service row。
- 也会从 `description`、`cmd`、`project`、`signature` 中做文本推断。
- 对已经有 measured service certificate 的 workload/profile，theorem dispatcher 会用 lower service 进入 robust MaxWeight 打分。

当前缺口：

- RE-SAC 还没有明确拆成 `HalfCheetah`、`Ant`、`Hopper`、`Walker2d` 等 env-specific workload key。
- 当前文本规则显式识别了 `ant` / `mujoco` / `resac` / `bapr` 等宽类，但不足以保证仅凭 description 就把 HalfCheetah 绑定到独立 service row。
- `simulation/service_cache.py` 中的记录包含 `node_bucket`，但主要 lookup 仍是 `(workload_key, profile)`。这意味着同一 workload/profile 在不同机器上的服务率差异不会自然进入每个候选 placement 的 score。
- 因此，当前算法还不能严格保证“把最慢 env 派到最快 node”这种 node-aware assignment。

需要补的设计：

将 service cache 升级为 node-aware and env-aware service matrix：

```text
workload_env x node_bucket x resource_state x profile
  -> lower_service / eta_lcb / mean_rate / confidence / source
```

其中 `workload_env` 应从 `cmd` / `description` / `signature` / `command_fingerprint` 中解析，例如：

- `HalfCheetah-v2` / `HalfCheetah-v3`
- `Ant-v2` / `Ant-v3`
- `Hopper-v2` / `Hopper-v3`
- `Walker2d-v2` / `Walker2d-v3`

然后 theorem dispatcher 对候选 action `(task, node, gpu/cpu, co-location profile)` 绑定该候选 node 上的 measured lower service，而不是绑定一个全局 workload/profile 平均值。

## 3. 迁移动作空间的当前事实

用户提出的问题是：动作空间是否包含迁移，并且是否计算迁移成本。例如 `node003` 负载最高、`node005` 负载最低，或者 `jtl311linux` CPU 空闲但缺环境，是否会考虑把运行到一半的任务迁移过去，只要迁移资源足够。

当前结论：数学模型允许迁移，但当前 theorem-facing algorithm 实现还没有把“运行中任务迁移”作为完整候选动作闭合。

数学和论文层面已经有的部分：

- `paper/main.tex` 中 robust candidate MaxWeight 使用

```text
Q(t)^T lower_mu_t(a) - K_t(a) - G_t(a)
```

其中 `K_t(a)` 被解释为 switching / rollback / migration cost。
- `md/math.md` 中也把 placement、co-location、rollback/preemption attribute、statewise drain rule 等纳入 finite-feature action。
- 所以理论路线已经能容纳 migration / preemption / checkpoint / rollback。

代码层面已经有的相关能力：

- `skill/scheduler.py` 有 `rebalance` / `dispatch` 运维入口。
- `scheduler.py` 会维护 `eta_seconds`，并用它做 load-balanced routing 的基础。
- 对 checkpoint/resume 有大量保护逻辑：`ckpt_dir`、`resume_flag`、`resume_locations`、`resume_checkpoint_node`、`resume_from` 等。
- 对 queued task，可以重新 placement；对 staged / resumed task，会优先或强制使用已有 checkpoint locality，避免从 0 重跑。
- 对运行中任务，hard rule 明确写着 dispatch / clear-queue / rebalance 不能随意 touch running tasks。

当前没有闭合的部分：

- `algorithm/theorem_dispatch/global_dispatch.py` 和 `batch_policy.py` 当前候选 action 主要是 queued task 的 placement / batch placement。
- 候选 action row 是类似 `task=t;node=n;gpu=g` 的 launch/placement action，不是 `migrate running task r from node i to node j`。
- `penalty_units` 当前主要是 profile penalty / bounded penalty，尚未系统性接入 checkpoint sync time、environment staging time、lost work、restart delay、risk of failed resume 等 migration cost。
- `scheduler.py` 有 resume/relaunch 能力，但不是 theorem dispatcher 直接选择的“迁移动作”。

因此不能 claim：

```text
当前动作空间已经完整包含运行中任务迁移，并且已经按 measured migration cost 做 robust MaxWeight 决策。
```

更准确的表述是：

```text
当前数学 action space 支持 migration/switching cost；
当前系统有 checkpoint/resume/rebalance primitives；
但 theorem-facing implementation 目前闭合的是 queued placement/batch placement，
尚未闭合 running-task migration action family。
```

## 4. 服务器和港口的统一调度视角

用户提出“这个方法可以作为港口/服务器共同的调度问题”，这个判断是合理的，而且和当前数学路线一致。

统一抽象：

- job / vessel / task 都是需要被服务的实体。
- server / GPU / CPU / berth / crane / yard block 都是资源。
- placement / berth assignment / GPU assignment 是基础动作。
- co-location / congestion / interference / yard congestion 是 action-dependent service degradation。
- migration / re-berthing / checkpoint-restart / crane reassignment 是 switching action。
- `K_t(a)` 表示切换、迁移、同步、回滚、环境部署、或重新靠泊成本。
- `G_t(a)` 表示 OOM、拥塞、失败、干扰、不可恢复中断等风险惩罚。
- lower service certificate 对应保守服务率，例如 steps/sec、jobs/hour、container moves/hour、vessel processing rate。

这意味着论文可以把服务器调度作为实证主场，但理论对象可以表述为：

```text
state-dependent stochastic processing network with certified candidate actions,
lower-service estimates, bounded switching cost, and robust MaxWeight control.
```

港口不是额外故事，而是同一个数学对象的另一个实例。但如果要把港口作为正式 claim，需要对应的 service map、switching cost map、arrival/service model 和实验或案例数据。

## 5. 建议的下一步闭合路线

优先级从高到低：

1. `tqdm` / progress contract 硬化
   - 所有 theorem-facing benchmark 和生产候选任务必须提供可解析 progress。
   - 无 progress 的任务进入 low-confidence / probe / history-fallback bucket，不进入强 claim。

2. Env-aware service identity
   - 显式解析 MuJoCo / RE-SAC environment name。
   - 建立 `resac_halfcheetah`、`resac_ant`、`resac_hopper`、`resac_walker2d` 等独立 service family。

3. Node-aware service matrix
   - service lookup 从 `(workload_key, profile)` 扩展到 `(workload_env, node_bucket, resource_state, profile)`。
   - 保留旧 lookup 作为 fallback，但 theorem-ready action 必须优先使用候选 node 的 certificate。

4. Running-task migration action family
   - 定义 action：

```text
migrate(task_id, from_node, to_node, checkpoint_policy, sync_policy, resume_policy)
```

   - 定义 service vector：迁移后预计 lower service。
   - 定义 cost：

```text
K_t = checkpoint_flush_time
    + transfer_time
    + environment_stage_time
    + restart_delay
    + lost_work_penalty
    + resume_failure_risk_penalty
```

   - 只有 checkpoint/resume contract verified 的任务允许进入 migration candidate set。

5. Theorem dispatcher 接入迁移动作
   - 与 placement action 一起枚举：

```text
launch queued task
keep running task
migrate running task
defer / probe unknown task
```

   - 用同一个目标函数比较：

```text
Q(t)^T lower_mu_t(a) - K_t(a) - G_t(a)
```

6. 港口/服务器统一叙述
   - 论文主线仍以 Scheduleurm 实证为核心。
   - 理论定义中保留 switching/migration cost 和 state-dependent service network。
   - 港口作为 motivating/generalized instance，可以放在 introduction 或 discussion；除非补港口数据，否则不要作为实证 claim。

## 6. 当前可对外 claim 的边界

可以 claim：

- fresh ETA 实验闭合使用 task-native `tqdm` / progress stable ETA。
- 当前理论模型支持 candidate action、lower service、bounded penalty、switching/migration cost。
- 当前系统有 checkpoint/resume/rebalance primitives。
- 当前 theorem-facing implementation 已支持 measured-cache robust MaxWeight placement / batch placement。

不能 claim：

- 所有 live ETA 都已经完全脱离 history fallback。
- RESAC 不同环境已经全部 env-specific 校准。
- service cache 已经完整 node-aware。
- 当前 theorem action space 已经完整包含 running-task migration。
- 已经用 measured migration cost 做过 robust MaxWeight live migration 实验。
- 港口调度实证已经闭合。

