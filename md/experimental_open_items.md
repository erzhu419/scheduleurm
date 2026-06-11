# Scheduleurm 理论路线中仍需实验 / profiling 才能闭合的项目

本文档只放目前无法靠 Lean/数学证明单独闭合、必须依赖 scheduleurm 真实运行数据、profiling 或扰动实验校准的部分。其余非实验部分已经放到 proof 里的 Scheduleurm 证明链继续推进。

## 1. 服务率与干扰模型校准

需要从 scheduleurm 运行日志和专门 profiling 得到 regime-dependent service map：

[
\mu_i^z(a)=\mathbb E[S_i(a,z,\omega)].
]

最少要记录：

```text
job class / model type / batch size / workload phase
global configuration action a
GPU/CPU/NUMA/NIC placement
同 GPU co-location profile
VRAM/RAM 占用、GPU util、CPU util、I/O、网络指标
单位时间 goodput / progress / failed step / OOM / rollback
```

要产出的实验量：

```text
Amax_i, Smax_i                 bounded finite-support proof 的常数
μ_i^z(a), lower_i(a)           drift 证明中的 service lower bound
ε_est                          lower-service estimation error
B                              second-order drift constant
```

当前已闭合的本地 bucket service slices：

```text
q00 light_control_local: profiles 1-13 measured; profile 14 measured capacity boundary.
q10 cpu_heavy_local_bench: profiles 1-9 measured; profile 10 measured capacity boundary.
```

这些结果已经足够支持 reviewer-facing q00/q10 replay comparison：candidate、
legacy 和 SOTA-style policies 都使用同一个 measured service cache。它们仍然
不是完整 stability theorem certificate；还需要把这些 measured slices 接到
`epsilon_est`, `B`, `delta` 和负载点的 slack accounting 表。

## 2. Fabric metric 的 \(L\) 和 \(\rho\)

Lean 已经证明：如果 candidate set 是 full action space 在 finite-feature fabric metric 下的 \(\rho\)-cover，且服务率对该 metric 是 \(L\)-Lipschitz，则 support loss 和 coordinate capacity-set loss 都至多是 \(L\rho\)。

实验还需要校准：

```text
Φ(a)                           fabric/interference feature
d_Φ(a,a')                      weighted l1 fabric metric
L                              service Lipschitz envelope
ρ                              candidate generator 对 full-action samples 的 cover radius
```

这部分要能被 falsify，不能只写成漂亮假设。建议做 perturbation profiling：固定 job mix，只改变 placement、co-location 数、NUMA/NVLink/PCIe path、网络路径和并发度，测量 \(|\mu(a)-\mu(a')|/d_\Phi(a,a')\) 的上包络。

必须报告：

```text
Φ 的每个 feature 是从 scheduler 哪个 telemetry 字段来的
feature weight w_r 如何设定或拟合
L 的估计分位数 / worst-case envelope
哪些 co-location 或拓扑扰动导致 Lipschitz envelope 变大
ρ 是在全量 feasible action、采样 full action、还是历史 observed action 上估计的
candidate generator 是否存在 bounded-degree / local-neighborhood 结构
cover 是覆盖 all feasible actions、sampled feasible actions、还是 historical observed actions
cover 是 fixed family、statewise family、regimewise family，还是 uniform over all state/regime indices
```

如果某些干扰呈现跳变或非平滑，论文不能硬 claim 小 \(L\rho\)；应把这些区域标成需要 regime split、额外 feature、或 admission guard。

## 3. GPU co-location sweet spot / admission threshold

这部分必须依赖吞吐曲线，不能只靠抽象证明。

需要测：

[
g_z(n)=\text{同一 GPU 上 }n\text{ 个任务的总 goodput}
]

以及多类型任务 set function：

[
g_z(S).
]

要判断：

```text
g_z(n) 是否 unimodal
argmax_n g_z(n) 的 sweet spot 是否稳定
g_z(S) 是否近似 submodular / diminishing returns
不同任务类型组合是否存在明显反协同
admission 阈值是否随 regime z、VRAM、batch size、phase 改变
```

只有这些曲线成立后，sweet spot / admission threshold 才能作为 experiment-driven theorem 写进论文扩展部分。

2026-06-09 新观测：jtl110gpu / jtl110gpu2 上误把 BAPR/RL 任务塞到约
83% VRAM 后，短期看速度没有明显弱于 node007 上严格 1/3 VRAM 限制下的
任务。当前算法已经把 `post_vram_frac`、`post_vram_bucket` 和 co-location
profile 作为 finite features 记录，也允许通过 clean hard-rule mode 绕过
1/3 packing rule；但是 `hybrid_rl_resac_ant` 的 production/replay 选择仍主要
依赖已有 service cache。最新 live-robust replay 只把 q11 profile 1-3 当作
portfolio sanity 的实测证书，早期 BAPR dense curve 虽显示 2-6/GPU 近似平台，
但还没有把“83% VRAM 仍不慢”作为正式 production service row 接入。

因此这条现象应转成一个新的 profiling 任务：

```text
workload = production BAPR / RE-SAC long-run templates, not only short Ant smoke
nodes = jtl110gpu, jtl110gpu2, node007
profiles = keep adding tasks until physical OOM / progress degradation boundary
telemetry = post_vram_frac, task_count, GPU util, per-task progress, aggregate goodput
claim criterion = profile k at high VRAM is admitted only if lower-service(k)
                 remains within epsilon_est of the calibrated plateau
```

在这张表闭合前，论文和算法都不能把 1/3 VRAM 规则当真实 sweet spot，也不能把
83% VRAM 当无条件安全；正确表述是“VRAM fraction is a calibrated regime
feature, not a fixed capacity proxy.”

## 4. Active bucket 的具体采样模型

Lean 已经证明 active-bucket event 下 regret 依赖 \(|B_{active}|\)，并证明了 high-probability input event 可以提升为 high-probability regret bound。

仍需由 scheduleurm 的具体学习/采样机制给出：

```text
active bucket 定义：fabric neighborhood / co-location profile / semi-bandit factor
feedback model：full-information / semi-bandit / bandit / censored feedback
同一 action 执行后可观测哪些 job-class service components
每个 bucket 的观测次数 bucketCount(b)
bucket loss / confidence radius 的具体计算
adaptive sampling rule
bucket 是固定定义还是 online adaptive generated
change point 后旧样本如何 discount / reset
exploration 与 queue backlog 如何耦合，是否会牺牲 stability slack
noise/tail assumption 是否符合 sub-Gaussian、bounded、或 empirical Bernstein
```

如果实际系统只有被调度过的配置才有反馈，需要额外记录 selection probability 或 exploration policy，否则 high-probability concentration 只能作为 conditional theorem，不能作为完整 learning theorem。

lower-service domination 也必须按 confidence event 处理：

```text
需要证明：Pr[∀Q,i, lower_i(a(Q))≤E[S_i|Q]] ≥ 1-δ_conf
如果 lower 来自 BOCD / posterior / empirical LCB，需要记录其训练窗口、reset/discount 规则和 tail assumption
如果只对 observed actions 有 lower bound，则主 theorem 只能用于这些 actions 或需要 exploration/cover 证明
```

## 5. Hidden regime / BOCD 检测延迟

Lean 已经有 dwell-time / switching-window backlog budget：检测和切换窗口只要占每个 segment backlog mass 的 \(\theta\) 比例，就把 drift margin 从 \(m\) 降到 \(m-\chi\theta\)。

还需要实验或具体 detector model 给出：

```text
change point 发生频率和 dwell time 分布
BOCD / detector delay τ_detect 的经验分布或 tail bound
误报率、漏报率
检测窗口内 backlog mass / segment backlog mass 的 θ
switching / rollback / migration cost χ
```

没有这些量，average-regime stability 和 BOCD delay theorem 只能停在条件式 dwell/switching 证明，不能写成 scheduleurm 的实证闭合定理。

## 6. Penalty 与 slack 消耗

robust drift theorem 需要：

[
\delta > L\rho+\epsilon_{est}+\beta+\alpha_1.
]

这里 \(\beta\) 来自 queue-scaled penalty：

[
0\le K+R_t\le P_0+\beta\|Q(t)\|_1.
]

如果 scheduler 不是 exact argmax，而是 greedy / local search / time-limited ILP，需要校准 approximate oracle error：

[
\text{optimal candidate score}-\text{chosen score}
\le
\alpha_0+\alpha_1\|Q(t)\|_1.
]

需要从 scheduleurm 数据估计：

```text
P0                             固定切换/rollback/风险成本
β                              随 backlog 增长的 penalty rate
α0                             bounded optimization oracle error
α1                             queue-scaled optimization oracle error
δ                              目标负载相对 capacity 的 slack
η = δ-(Lρ+ε_est+β+α1)          最终 drift margin
```

当前状态（2026-06-08）：

```text
Module50 已补 scheduler candidate-family trace hook；
SCHEDULEURM_ORACLE_AUDIT_LOG 打开后，pick_placement 会记录真实候选全集、
chosen action、scheduler sort key、finite bucket/class/regime audit。

这可以验证 selected action 是否为 scheduler-score best。
但 theorem-grade α0/α1 仍需要每个 candidate 的 lower_service vector、
queue_vector 和 penalty_units，或者从同一决策状态的 measured service cache
严格重建这些量。

当前 production trace 状态是 NO_TRACE；历史 selected-only placement_algorithm_audit
不能冒充 full candidate-set oracle certificate。

Module55 已补 lower-service enrichment bridge：
  scheduler candidate-family trace
  + measured lower-service lookup
  + nonempty queue_vector
  -> robust_maxweight_lower_service trace
  -> Module52 alpha0/alpha1 audit。

它会拒绝：
  missing queue_vector；
  任意 candidate 缺 lower_service；
  production trace file 不存在。

当前 artifact 仍是 NO_TRACE，因为
  /home/erzhu419/.claude/scheduler/oracle_trace.jsonl
不存在。剩余工作是采集真实 production candidate trace，并为每个候选
candidate_bucket / class_key / regime_key 提供 measured lower-service row。
```

如果估计后 \(\eta\le0\)，理论不是错，而是说明 candidate cover、估计误差、penalty 或 solver error 已经吞掉全部 capacity slack，需要改 candidate generator、降低 penalty、改善 oracle 或加 admission control。

## 7. Production coverage 闭合顺序

Module51 已经把 raw queue history 和 reviewer-facing production population
拆开。Module73 进一步把无 scheduler id、无 log、无可复现 progress unit 的
external auto-adopted stdin/wait-for 进程排除在 controlled-arrival theorem
population 外。Module92--99 又把剩余非 CPU/SUMO/transit、CPU-fabric、
GPU/RL-fabric 和 c9_16 Transit profile-extension buckets 拆成 strict
service certificates。当前 30 天 `completed_active_production` theorem-facing
视角为：

```text
records = 3411
obligation mapped = 3411
measurement_required = 0
obligation mapped_fraction = 1.000000
strict view mapped = 3411 / 3411
representative mapped = 0
unmapped = 0
global_theorem_closed = true
```

曾经最大的 CPU/SUMO/transit blocker 闭合轨迹是：

```text
before Module56: cpu_sumo_transit_eval_or_control = 1245 / 2504 completed-active records
after Module56:  cpu_sumo_transit_eval_or_control = 1208 / 2442 completed-active records
after Module57:  cpu_sumo_transit_eval_or_control = 1211 / 2449 completed-active records
after Module58:  cpu_sumo_transit_eval_or_control = 1103 / 2453 completed-active records
after Module59:  cpu_sumo_transit_eval_or_control = 1049 / 2458 completed-active records
after Module60:  cpu_sumo_transit_eval_or_control = 1009 / 2465 completed-active records
after Module61:  cpu_sumo_transit_eval_or_control = 946 / 2469 completed-active records
after Module62:  cpu_sumo_transit_eval_or_control = 862 / 2471 completed-active records
after Module63:  cpu_sumo_transit_eval_or_control = 803 / 2487 completed-active records
after Module64:  cpu_sumo_transit_eval_or_control = 704 / 2553 completed-active records
after Module65:  cpu_sumo_transit_eval_or_control = 659 / 2589 completed-active records
after Module66:  cpu_sumo_transit_eval_or_control = 576 / 2679 completed-active records
after Module67:  cpu_sumo_transit_eval_or_control = 576 / 2679 completed-active records
after Module68:  cpu_sumo_transit_eval_or_control = 532 / 2755 completed-active records
after Module69:  cpu_sumo_transit_eval_or_control = 471 / 2781 completed-active records
after Module70:  cpu_sumo_transit_eval_or_control = 409 / 2797 completed-active records
after Module71:  cpu_sumo_transit_eval_or_control = 350 / 2805 completed-active records
after Module72:  cpu_sumo_transit_eval_or_control = 320 / 2840 completed-active records
after Module73:  cpu_sumo_transit_eval_or_control = 258 / 2766 completed-active records
after Module74:  cpu_sumo_transit_eval_or_control = 212 / 2766 completed-active records
after Module75:  cpu_sumo_transit_eval_or_control = 169 / 2766 completed-active records
after Module76:  cpu_sumo_transit_eval_or_control = 139 / 2766 completed-active records
after Module77:  cpu_sumo_transit_eval_or_control = 113 / 2766 completed-active records
after Module78:  cpu_sumo_transit_eval_or_control = 94 / 2910 completed-active records
after Module79:  cpu_sumo_transit_eval_or_control = 73 / 3058 completed-active records
after Module80:  cpu_sumo_transit_eval_or_control = 56 / 3101 completed-active records
after Module81:  cpu_sumo_transit_eval_or_control = 43 / 3095 completed-active records
after Module82:  cpu_sumo_transit_eval_or_control = 38 / 3136 completed-active records
after Module83:  cpu_sumo_transit_eval_or_control = 26 / 3199 completed-active records
after Module84:  cpu_sumo_transit_eval_or_control = 20 / 3214 completed-active records
after Module85:  cpu_sumo_transit_eval_or_control = 14 / 3211 completed-active records
after Module86:  cpu_sumo_transit_eval_or_control = 10 / 3255 completed-active records
after Module87:  cpu_sumo_transit_eval_or_control = 7 / 3280 completed-active records
after Module88:  cpu_sumo_transit_eval_or_control = 5 / 3327 completed-active records
after Module89:  cpu_sumo_transit_eval_or_control = 3 / 3333 completed-active records
after Module90:  cpu_sumo_transit_eval_or_control = 1 / 3331 completed-active records
after Module91:  cpu_sumo_transit_eval_or_control = 0 / 3347 completed-active records
```

所以 CPU/SUMO/transit production blocker 已经从 Module81 的 43 / 3095
闭合到 0 / 3347。随后 Module92--96 闭合了剩余 non-CPU/SUMO/transit
measurement-required obligations，Module97--99 又把 representative
strictness gaps 升级为 strict service certificates：

```text
Module92: Asumption Agent unittest / meta-QA / phase2 command certificates.
Module93: FreqDuet spacectx_screen_ep100_wu10 c3_8 shard certificate.
Module94: residual CFCMT, Asumption Agent, sensing, Nature, Scheduleurm control-plane, BAPR/RE-SAC eval, H2Oplus command certificates.
Module95: Transit real-demand c9_16 shards and RE-SAC conda-pack artifact command.
Module96: RE-SAC review5 JAX train production-fabric burst service certificate.
Module97: residual CPU-heavy production-fabric completed-command certificate.
Module98: project-level hybrid RL production-fabric completed-command certificates.
Module99: Transit real-demand c9_16 throughput_safe_wait_v6 finite-feature profile extension.
```

论文中应写成：

```text
completed-active production theorem population: full strict measured-bucket coverage;
mapped-slice capacity certificate: positive/usable;
raw 30-day history: not the theorem population; Module49 raw global flag remains false by design.
```

Module53 已经完成第 1 步。Module56 又完成了第一个 production
sub-bucket 的真实曲线测量。Module57 进一步闭合了一个很窄的 c9_16
direct-runner exact-config slice：

```text
closed exact slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_17_32
workload_key = freqduet_cpu_ablation_c17_32
feasible profiles = 1,2,4
capacity boundary = 8
completed-active mapped count = 147
residual freqduet_cpu_ablation|c_17_32 needing measurement after Module63 = 46

closed exact slice = runner_v3.py --config configs_freqduet/F_allfreq_alllayers_hiro.yaml within c_9_16
workload_key = freqduet_runner_v3_allfreq_alllayers_c9_16
feasible profiles = 1,2,4,8
completed-active mapped count = 1

closed command-shape slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_9_16
workload_key = freqduet_cpu_ablation_c9_16
feasible profiles = 1,2,4,8
completed-active mapped count = 110
unit rule = parsed jobs times episodes

closed completed-history slice = clean SimpleSAC run_multiseed_eval.sh within sumo_eval_cpu|c_le2
workload_key = sumo_eval_simple_sac_c_le2
feasible profiles = 1
completed-active mapped count = 54
unit = eval_json

closed completed-history slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_3_8
workload_key = freqduet_cpu_ablation_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 40
unit rule = parsed jobs times episodes

closed completed-history slice = runner_v3.py within freqduet_cpu_ablation|c_3_8
workload_key = freqduet_runner_v3_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 86
unit rule = parsed episodes

closed completed-history slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_33_64
workload_key = freqduet_cpu_ablation_c33_64_completed_history
feasible profiles = 1
completed-active mapped count = 63
unit rule = parsed jobs times episodes

closed completed-history slice = runner_v3.py within freqduet_cpu_ablation|c_le2
workload_key = freqduet_runner_v3_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 84
unit rule = parsed episodes

closed completed-history slice = Transit native_promotion_replan_validation within c_17_32
workload_key = transit_native_promotion_c17_32_seedrange_completed_history
feasible profiles = 1
completed-active mapped count = 91
unit rule = parsed seed-count times episodes

closed completed-history slice = BAMOR c_3_8 train_compare_baselines.py
workload_key = bamor_train_compare_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 58
unit rule = parsed training steps

closed completed-history slice = BAMOR c_3_8 train_bamor_mujoco.py
workload_key = bamor_mujoco_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 123
unit rule = parsed training steps

closed completed-history slice = BAMOR c_3_8 run_bamor_diagnostic_shard.py
workload_key = bamor_diagnostic_shard_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 25
unit rule = parsed training steps

closed completed-history slice = ZSW TSP/SUMO c_le2 runners
workload_key = zsw_tsp_sumo_eval_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 50
unit rule = parsed simulated SUMO seconds from --duration

closed completed-history slice = Transit native_promotion_replan_validation batch within c_33_64
workload_key = transit_native_promotion_c33_64_batch_completed_history
feasible profiles = 1
completed-active mapped count = 55
unit rule = parsed seed-count times episodes

closed completed-history slice = Transit native_promotion_replan_validation single-seed smoke/fix within c_33_64
workload_key = transit_native_promotion_c33_64_single_seed_completed_history
feasible profiles = 1
certificate record count = 2
unit rule = parsed seed-count times episodes

closed completed-history slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_65p
workload_key = freqduet_cpu_ablation_c65p_completed_history
feasible profiles = 1
completed-active mapped count = 7
unit rule = parsed jobs times episodes

closed completed-history slice = run_freqduet_promoted_ep100_hpc_batch.sh within c_65p
workload_key = freqduet_promoted_ep100_c65p_completed_history
feasible profiles = 1
completed-active mapped count = 6
unit rule = parsed job-count times 100 episodes

closed completed-history slice = Transit native_promotion_replan_validation within c_65p
workload_key = transit_native_promotion_c65p_completed_history
feasible profiles = 1
completed-history service records used = 48
current completed-active mapped count = 49
unit rule = statically parsed seed-count times episodes

closed completed-history slice = Transit/FreqHRL trading sweep within c_le2
workload_key = transit_trading_sweep_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 14
unit rule = parsed market-step grid units

closed completed-history slice = Transit/FreqHRL trading policy within c_le2
workload_key = transit_trading_policy_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 18
unit rule = parsed train/eval policy-market-step units

closed completed-history slice = Transit surrogate validation within c_le2
workload_key = transit_surrogate_validation_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 3
unit rule = parsed surrogate corridor-step units

closed completed-history slice = Transit native_promotion_replan_validation within c_le2
workload_key = transit_native_promotion_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 19
unit rule = parsed native variant-episode units

closed completed-history slice = Transit native wait-credit / real-demand control within c_le2
workload_key = transit_native_control_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 7
unit rule = parsed native control episode units

closed completed-history singleton = Transit/FreqHRL Windows IMPORT_OK check
workload_key = transit_freqhrl_import_smoke_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = import_check

closed completed-history slice = bounded-wait Transit native_promotion_replan_validation within c_9_16
workload_key = transit_native_promotion_c9_16_bounded_wait_completed_history
feasible profiles = 1
completed-active mapped count = 7
unit rule = statically parsed seed-count times episodes

closed completed-history slice = residual Transit native_promotion_replan_validation within c_9_16
workload_key = transit_native_promotion_c9_16_residual_completed_history
feasible profiles = 1
completed-active mapped count = 40
unit rule = statically parsed seed-count times episodes

closed completed-history slice = residual runner_v3.py within freqduet_cpu_ablation|c_9_16
workload_key = freqduet_runner_v3_c9_16_residual_completed_history
feasible profiles = 1
completed-active mapped count = 12
unit rule = parsed episodes

closed completed-history slice = CFCMT GTFS/LTA feed conversion within c_le2
workload_key = cfcmt_feed_conversion_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 5
unit rule = feed-conversion job

closed completed-history slice = CFCMT H2O environment validation within c_le2
workload_key = cfcmt_env_validation_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 5
unit rule = parsed --max-steps validation steps

closed completed-history slice = CFCMT SUMO/APC/AVL generation within c_le2
workload_key = cfcmt_sumo_generation_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 4
unit rule = parsed --duration-sec simulated seconds

closed completed-history slice = CFCMT SUMO/APC/AVL snapshot generation within c_le2
workload_key = cfcmt_snapshot_generation_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 5
unit rule = inferred snapshot windows

closed completed-history slice = CFCMT policy rollout validation within c_le2
workload_key = cfcmt_policy_rollout_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 10
unit rule = policy count times event budget

closed completed-history singleton = CFCMT traffic_signal_sumo_phase2.py
workload_key = cfcmt_traffic_signal_phase2_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = phase2_run

closed operational-semantics boundary = unobservable external auto-adopted stdin/wait-for
label = excluded_external_auto_adopted_unobservable
excluded completed-active count = 74
reason = no scheduler id, no scheduler log, no reproducible command/progress unit

closed completed-history slice = FreqDuet c17_32 direct runner_v3.py
workload_key = freqduet_runner_v3_c17_32_completed_history
feasible profiles = 1
completed-active mapped count = 16
unit rule = parsed episodes

closed completed-history slice = FreqDuet c17_32 paper longtrain shell shards
workload_key = freqduet_paper_longtrain_c17_32_completed_history
feasible profiles = 1
completed-active mapped count = 16
unit rule = one completed shard, not job-count times episodes because --skip-existing is present

closed completed-history slice = Transit native_promotion_replan_validation c17_32 residual
workload_key = transit_native_promotion_c17_32_residual_completed_history
feasible profiles = 1
completed-active mapped count = 14
unit rule = statically parsed seed-count times episodes

closed completed-history slice = BAMOR c_9_16 train_compare_baselines.py
workload_key = bamor_train_compare_c9_16_completed_history
feasible profiles = 1
completed-active mapped count = 6
unit rule = parsed training steps

closed completed-history slice = BAMOR c_9_16 train_bamor_mujoco.py
workload_key = bamor_mujoco_c9_16_completed_history
feasible profiles = 1
completed-active mapped count = 3
unit rule = parsed training steps

closed completed-history slice = BAMOR c_9_16 run_bamor_diagnostic_shard.py
workload_key = bamor_diagnostic_shard_c9_16_completed_history
feasible profiles = 1
completed-active mapped count = 34
unit rule = parsed training steps

closed completed-history slice = FreqDuet c_le2 run_freqduet_ablation.py
workload_key = freqduet_cpu_ablation_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 3
unit rule = parsed jobs times episodes

closed completed-history slice = FreqDuet c_le2 run_baseline_rule.py
workload_key = freqduet_baseline_rule_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 5
unit rule = parsed episodes

closed completed-history slice = FreqDuet c_le2 preflight/env checks
workload_key = freqduet_preflight_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 5
unit rule = preflight_check

closed completed-history slice = Transit/FreqHRL c_le2 analysis matrix/report jobs
workload_key = transit_freqhrl_analysis_matrix_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 11
unit rule = analysis_job

closed completed-history slice = Transit/FreqHRL c_le2 shard merge jobs
workload_key = transit_freqhrl_merge_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 6
unit rule = merge_job

closed completed-history slice = BAMOR c_le2 train_compare_baselines.py
workload_key = bamor_train_compare_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 7
unit rule = parsed training steps

closed completed-history slice = BAMOR c_le2 train_bamor_mujoco.py
workload_key = bamor_mujoco_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 16
unit rule = parsed training steps

closed completed-history slice = BAMOR c_le2 run_bamor_diagnostic_shard.py
workload_key = bamor_diagnostic_shard_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 3
unit rule = parsed training steps

closed completed-history slice = offline-sumo c_le2 eval_*.py production commands
workload_key = offline_sumo_eval_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 9
unit rule = one completed eval command

closed completed-history slice = H2Oplus/SimpleSAC c_le2 complex shell eval jobs
workload_key = h2oplus_shell_eval_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 4
unit rule = one completed shell eval job

closed completed-history singleton = ZSW c_le2 m1_metrics_parser.py
workload_key = zsw_metrics_parser_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = one completed metrics parser job

closed completed-history slice = RESCO config/main.py c_le2 control-eval runs
workload_key = resco_config_eval_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 5
unit rule = one completed RESCO config run

closed completed-history singleton = Nature emissions demand extraction
workload_key = nature_emissions_extract_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = one completed extraction job

closed completed-history singleton = Nature emissions direct SUMO binary run
workload_key = nature_emissions_sumo_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = one completed SUMO binary run

closed completed-history slice = Transit/FreqHRL c3_8 native-promotion persistent stress
workload_key = transit_native_promotion_c3_8_persistent_stress_completed_history
feasible profiles = 1
completed-active mapped count = 7
unit rule = parsed seed-count times episodes

closed completed-history slice = Transit/FreqHRL c3_8 native real-demand batch validation
workload_key = transit_native_real_demand_batch_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 7
unit rule = parsed native-control episode units

closed completed-history slice = Transit/FreqHRL c3_8 native real-demand alighting shards
workload_key = transit_native_real_demand_alighting_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 7
unit rule = parsed native-control episode units

closed completed-history slice = BAMOR c17_32 Mujoco training
workload_key = bamor_mujoco_c17_32_completed_history
feasible profiles = 1
completed-active mapped count = 3
unit rule = parsed training-step units

closed completed-history slice = BAMOR c17_32 diagnostic shard training
workload_key = bamor_diagnostic_shard_c17_32_completed_history
feasible profiles = 1
completed-active mapped count = 14
unit rule = parsed shard training-step units

closed completed-history slice = FreqDuet c33_64 direct runner_v3
workload_key = freqduet_runner_v3_c33_64_completed_history
feasible profiles = 1
completed-active mapped count = 15
unit rule = parsed episode units

closed completed-history slice = offline-sumo c33_64 eval commands
workload_key = offline_sumo_eval_c33_64_completed_history
feasible profiles = 1
completed-active mapped count = 5
unit rule = one completed eval command

closed population-boundary correction = external auto-adopt FreqDuet spin helpers
excluded command shape = freqduet_autoadopt_spin.py
completed-active removed count = 2
unit rule = none; no scheduler id/log/progress unit, so excluded from controlled arrivals

closed completed-history slice = Transit/FreqHRL c33_64 trading policy
workload_key = transit_trading_policy_c33_64_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = policy train/eval unit

closed completed-history slice = Transit/FreqHRL c33_64 pressure matrix
workload_key = transit_trading_pressure_matrix_c33_64_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = seed-step-asset-scenario-baseline

closed completed-history slice = Transit/FreqHRL c9_16 pressure matrix
workload_key = transit_trading_pressure_matrix_c9_16_completed_history
feasible profiles = 1
completed-active mapped count = 2
unit rule = seed-step-asset-scenario-baseline

closed completed-history slice = Transit/FreqHRL c_le2 pytest commands
workload_key = transit_freqhrl_tests_c_le2_completed_history
feasible profiles = 1
completed-active mapped count = 2
unit rule = one completed pytest command

closed completed-history slice = BAMOR c3_8 Mujoco policy-set union aggregation
workload_key = bamor_mujoco_policy_union_c3_8_completed_history
feasible profiles = 1
completed-active mapped count = 1
unit rule = one completed aggregation command
```

当前 CPU/SUMO/transit first probe order 已为空：

```text
```

后续应优先攻击剩余非 CPU/SUMO/transit production buckets，而不是再增加没有生产覆盖意义的 GPU-only benchmark。

Module54/56 已经完成第 2-5 步的第一个闭合实例：

```text
runner = algorithm/experiments/production_cpu_workload_curve.py
first sub_bucket = freqduet_cpu_ablation|c_17_32
profiles = 1,2,4,8
cpu/task = 24
ram/task = 65536
work_items/task = 24
profile 1 aggregate = 0.447090 episode/s
profile 2 aggregate = 0.539602 episode/s
profile 4 aggregate = 0.878374 episode/s
profile 8 = capacity boundary, 5 progressed and 3 blocked
theorem_status = measured_sub_bucket_loaded_into_service_cache
```

剩余实验工作不是再定义 benchmark，而是沿用该 runner 继续攻击剩余
sub-bucket，得到稳定 progress 窗口，写出 service_curve_verdict，并把结果加入
service cache 后重跑 production coverage / capacity / slack certificate。
