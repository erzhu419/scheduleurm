# Scheduleurm 理论路线中仍需实验 / profiling 才能闭合的项目

本文档只放目前无法靠 Lean/数学证明单独闭合、必须依赖 scheduleurm 真实运行数据、profiling 或扰动实验校准的部分。其余非实验部分已经放到 proof 里的 Scheduleurm 证明链继续推进。

当前投稿口径的总边界见：

```text
md/or_claim_scope_matrix_2026_06_11.md
md/or_submission_closure_status_2026_06_11.md
```

最新已闭合但仍需按边界表谨慎表述的项目：

```text
Module51 completed_active_production strict coverage: 3437 / 3437.
Module100 service-map theorem oracle bridge: SERVICE_MAP_THEOREM_ORACLE_PASS.
OR closure gate 2026-06-11: online / holdout lower-service / ablation /
  live-state dry-run trace / reviewer supplement all PASS.
Extended closure gate 2026-06-11: theorem-grade natural live-node trace /
  service-domain admission population / direct SOTA scaffold / exact measured
  finite-slice fabric-cover calibration all PASS.
Post-closure upgrade artifacts 2026-06-11:
	  production-wide live trace gate =
	    md/production_live_theorem_trace_gate_20260612.md
	    status PRODUCTION_WIDE_LIVE_TRACE_PASS for an earlier one-task queued
	    production snapshot, using a read-only theorem trace with oracle audit
	    PASS.  This is not a launched completion trace and not a future-queue claim.
  production shadow theorem trace =
    md/production_shadow_theorem_trace_20260612.md
    non-invasive active-production probe over current running production tasks;
    GPU theorem subset closes with alpha0=alpha1=0, but this is not launched
    production-wide dispatch.
  direct SOTA binary smoke round2 =
    md/direct_sota_baseline_binary_smoke_20260611_round2.md
    Pollux/AdaptDL and Decima entrypoints smoke-pass; Gavel, IADeep, and Salus
    remain blocked by dependency/full-stack requirements.
  Gavel direct native smoke =
    md/gavel_direct_native_smoke_20260612.md
    in an isolated copy, dependency imports, protobuf stub generation,
    entrypoint help, generated-jobs native simulation, native Gavel trace, and
    Scheduleurm q01 native trace-seed smoke pass.  This is native-stack and
    trace-compatibility evidence, not measured service-unit equivalence.
  Gavel native performance microbaseline =
    md/gavel_native_performance_microbaseline_20260612.md
    bounded Scheduleurm-exported q01/q11 windows complete in Gavel's native
    trace simulator: q01 4/4 jobs and q11 8/8 jobs.  This is a native simulator
    microbaseline, not measured service-unit equivalence and not full-stack
    external-system superiority.
  direct SOTA full-stack readiness =
    md/direct_sota_fullstack_readiness_20260612.md
    same-workload adapter seeds now exist for Gavel, Pollux/AdaptDL, IADeep,
    Salus, and Decima; Gavel native microbaseline exists, but direct full-stack
    same-workload ready count remains 0 because Gavel service-unit equivalence
    and other systems' Kubernetes,
    Docker, runtime, or scope requirements are not satisfied.
  SOTA claim matrix =
    md/sota_fullstack_claim_matrix_20260612.md
    all seed families are present; direct full-stack same-workload ready count
    remains 0.
  fabric-cover k-center curve =
    md/global_fabric_cover_calibration_cover_curve_20260611.md
    exact measured finite slices still use rho=0; smaller candidate families
    must spend the reported Lrho.
  fabric-cover metric contract =
    md/fabric_cover_contract_certificate_20260612.md
    feature map, numeric scales, projection population, exact rho, compressed
    Lrho, and exclusions are explicit; future/all-state ready is false.
  future-admitted fabric-cover gate =
    md/future_admitted_fabric_cover_gate_20260612.md
    future tasks admitted to exact positive measured profiles use identity
    projection with rho=0.  Arbitrary all-state fabric cover remains false.
  future-production admission contract =
    md/future_production_admission_contract_20260612.md
    active/future production tasks route to ADMIT_THEOREM_TRACE only with a
    service-domain certificate; otherwise they are PROBE_REQUIRED.
  active-bucket / hidden-regime event certificate =
    md/active_bucket_hidden_regime_certificate_20260611.md
    deterministic event-level accounting closes.
  adaptive sampler / detector probability certificate =
    md/adaptive_sampler_detector_certificate_20260612.md
    concrete deterministic round-robin sampler and bounded two-window detector
    model closes as a deployable extension certificate.
  adaptive live integration probe =
    md/adaptive_live_integration_probe_20260612.md
    optional adaptive_theorem_maxweight_v1 emits sampler/detector audit fields
    over live-state theorem trace; default scheduler remains unchanged.
  q00/q10 generalization gate =
    md/q00_q10_generalization_gate_20260612.md
    local q00/q10 buckets are closed and remote CPU evidence is inventoried;
    broad all-CPU/data-loader generalization remains false.
	  q00/q10 broad measured-envelope gate =
	    md/q00_q10_broad_envelope_gate_20260612.md
	    1 q00 and 102 q10 CPU-like service-cache workloads have positive
	    lower-service rows for admission-facing measured-envelope claims; all
	    possible CPU/data-loader programs remain out of scope.
	  SOTA full-stack superiority gate =
	    md/sota_fullstack_superiority_gate_20260612.md
	    executable strict gate records direct_fullstack_sota_superiority_ready =
	    false, full_stack_ready_count = 0, and hard blockers from missing
	    Docker/Go/Kubernetes tooling, missing service-unit equivalence, or scope
	    mismatch.  This closes the review risk by forbidding direct full-stack
	    superiority language under the current environment.
	  Gavel service-unit equivalence certificate =
	    md/gavel_service_unit_equivalence_certificate_20260612.md
	    trace/schema compatibility, throughput seed readiness, and bounded native
	    microbaseline readiness are true for q01/q11; exact arrival times are not
	    identical because native Gavel trace rows use a small ordering jitter, and
	    gavel_service_unit_equivalence_ready remains false.
	  declared finite-domain positive-cover gate =
	    md/declared_finite_domain_positive_cover_gate_20260612.md
	    declared service-cache universe has 193 buckets: 187 positive lower-service
	    rows, 6 capacity boundaries, and 0 uncovered buckets.  This closes the
	    declared finite positive domain, not arbitrary positive-service all-state
	    stability.
	  all-state conservative fabric-cover gate =
	    md/all_state_conservative_cover_gate_20260612.md
	    every scheduler-visible state is covered conservatively as either an exact
	    measured admitted positive-service profile or a zero-service probe/defer
	    action.  This is all-state safety, not positive-service all-state
	    stability.
	  production launch/completion gate =
	    md/production_launch_completion_gate_20260612.md
	    rolling safe-launch snapshots report active-production progress and
	    high-utilization/no-safe-launch conditions.  It returns WAIT_RESOURCE_OR_QUEUE,
	    closes large-scale active-production progress evidence, and does not claim
	    large-scale launched production completion.
	  controlled production completion gate =
	    md/controlled_production_completion_gate_20260612.md
	    bounded controlled completion is true for the existing launched trace
	    (6 submitted tasks, 7 theorem slots, 13 candidates, alpha0=alpha1=0);
	    controlled_32_task_completion_ready and large_scale_organic_launched_completion_ready
	    remain false until safe resources allow more launches.
	```

## 0. OR reviewer gate after GPT_revise_OR.md

`md/GPT_revise_OR.md` 的核心意见不是要求把理论路线降级，而是要求把
proof object、live scheduler hook、finite-slice replay、production bridge 和
external-baseline comparison 系统对齐。当前论文可以 claim 的是：

```text
robust candidate MaxWeight theorem with explicit slack accounting;
exact measured finite-slice instantiation on declared q00/q01/q10/q11 buckets;
online Poisson/bursty/load-sweep policy-semantics replay comparison on the same
  measured service cache;
finite measured lower-service capacity certificates for declared slices;
completed-active mapped production service-map oracle bridge;
synthetic dry-run live-node-state trace enrichment/audit pipeline closure;
theorem_maxweight_v1 certified live-node candidate trace with queue_vector,
  lower_service, penalty_units, and robust_maxweight_lower_service semantics;
service-domain admission closure for the completed-active production population;
exact measured finite-slice fabric-cover calibration with rho=0;
direct external-SOTA adapter scaffold plus policy-semantics fallback replay;
reviewer Lean supplement repackaged with build log and comment-aware
  no-sorry/admit/axiom grep.
```

当前不能 claim 的是：

```text
the deployed live scheduler already implements global robust MaxWeight for all dispatches;
raw 30-day history is a clean theorem population;
attempted-only / cancelled / external jobs are closed by the production theorem;
	direct binary or full-stack superiority over Gavel, Pollux, Sia, IADeep, or Salus;
	Lrho=0 or epsilon_est=0 outside exact enumerated measured slices;
	all future online dispatches are theorem-grade without rerunning trace audit;
	all-state safety via zero-service probe/defer is the same as all-state
	  positive-service stability;
	active-production progress observations are the same as a launched production
	  completion trace;
	bounded controlled completion is the same as 32-task or organic production
	  completion.
	```

2026-06-11 已完成的 OR reviewer gate：

```text
online arrival experiment:
  Poisson, bursty, and load-sweep arrivals;
  artifact = md/or_gate_online_arrivals.md
  scenario_count = 120
  candidate completed all jobs
  candidate not Pareto-dominated by SOTA-style policy semantics under 0.5%
  sampled-replay tolerance.

holdout calibration:
  artifact = md/or_gate_holdout_calibration.md
  sparse empirical-Bernstein diagnostic does NOT close the original mean-service
  slack, and this is intentionally not claimed.
  theorem-facing finite measured lower-service certificates do close with eta>0.

ablation:
  artifact = md/or_gate_ablation_suite.md
  full adaptive candidate beats legacy on geometric makespan and mean flow, and
  no legacy, sweetspot-hook, support-scorer, delay, statewise-guard,
  no-profile-penalty, or tie-break ablation policy Pareto-dominates it.

longer live trace:
  artifact = md/or_gate_live_trace.md
  trace_origin = synthetic_dryrun_live_node_probe
  trace_slot_count = 128
  min_required_slots = 96
  candidate_count_total = 767
  no queue mutation and no task launch.

reviewer artifact:
  artifact = md/or_gate_reviewer_supplement.md
  ScheduleurmUpload.lean sha256 =
    af79be4416e4c4add0fe41663fc0927af7058fe04412908b6688a8409227f01b
  lake env lean ScheduleurmUpload.lean = PASS
  comment-aware sorry/admit/axiom grep = clean.
```

要把 paper 从 current finite-slice OR case study 推到 stronger OR production
claim，仍必须补：

```text
larger production-wide live dispatch trace with actual queued production jobs
  and safe low-interference resources.  The bounded ScheduleurmBench live
  dispatch is already launched and completion bridged; an earlier 2026-06-12
  production queued trace closed one queued-production snapshot; the rolling
  launch/completion and controlled-completion gates correctly refuse launch under
  high-utilization/no-safe-launch conditions and report active-progress or bounded
  controlled evidence only.
direct external-system execution if claiming full-stack superiority over Gavel,
  Pollux, Sia, IADeep, or Salus.  Current evidence verifies local entrypoints
  for Pollux/AdaptDL and Decima, verifies isolated Gavel dependency/stub/help,
  generated-jobs, native trace, Scheduleurm trace-seed execution, and bounded
  q01/q11 native simulator microbaseline, emits same-workload adapter seeds for
  all inspected systems, and records Gavel service-unit-equivalence /
  Kubernetes / Docker / Go / runtime blockers.  The strict superiority gate
  makes this a hard-blocker certificate, not an unresolved ambiguity.
generalized perturbation-profiled fabric-cover evidence outside exact measured
  and future-admitted measured states, if the paper wants to claim arbitrary
  positive-service future/all-state cover.  The k-center curve quantifies finite
  measured-slice Lrho tradeoffs; the future-admitted gate closes identity
  projection for measured-profile admission only; the all-state conservative
  gate closes safety by routing unknown states to zero-service probe/defer.
live integration and A/B trace evidence before promoting active-bucket learning
  or hidden-regime results into deployed-scheduler empirical claims.  The
  deterministic active-bucket union bound, replay dwell/switching accounting,
  and deployable sampler/detector probability model are now executable and
  certified as extension inputs.
```

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
legacy 和 SOTA-inspired policy-semantics replay policies 都使用同一个 measured service cache。它们仍然
不是完整 stability theorem certificate；还需要把这些 measured slices 接到
`epsilon_est`, `B`, `delta` 和负载点的 slack accounting 表。

## 2. Fabric metric 的 \(L\) 和 \(\rho\)

Lean 已经证明：如果 candidate set 是 full action space 在 finite-feature fabric metric 下的 \(\rho\)-cover，且服务率对该 metric 是 \(L\)-Lipschitz，则 support loss 和 coordinate capacity-set loss 都至多是 \(L\rho\)。

当前状态要分清两层：

```text
exact measured finite slices:
  已经能直接用 measured service cache 做 finite-slice / service-map 证书；
  Module48 和 Module100 里的 Lrho=0 是 exact enumerated measured slice 口径。

general fabric-cover calibration:
  Lean theorem 已闭合；
  但 broad claim 还需要正式 perturbation profiling table 才能把 L 和 rho 写成实测证书。
```

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

投稿前的 generalized fabric-cover 表至少应有：

| Field | Required content |
|---|---|
| `feature_map` | \(\Phi_r(a)\) 的字段来源，如 node/GPU、co-location、VRAM bucket、CPU bucket、NUMA/PCIe class |
| `weights` | 每个 feature 的 \(w_r\)，以及是工程设定、拟合、还是 worst-case scaling |
| `cover_population` | fixed/statewise/regimewise；all feasible、sampled feasible、还是 historical observed |
| `projection` | 每个 full action 到 candidate action 的 \(\pi(a)\) 构造 |
| `rho` | \(\max_a d_\Phi(a,\pi(a))\)，并列出达到最大值的 action |
| `sensitivity_samples` | 用于估计 \(|\mu(a)-\mu(a')|/d_\Phi(a,a')\) 的扰动对 |
| `L` | worst-case envelope 或带 failure probability 的 confidence envelope |
| `Lrho` | 进入 slack accounting 的 \(L\rho\) |
| `exclusions` | 不满足 Lipschitz envelope 的区域如何 split、guard 或移出 generalized claim |

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
依赖已有 service cache。当前 robust q11 cache 把 profile 1-9 作为
first-boundary 之前的 feasible measured rows，profile 10 及以上被 fresh live
OOM / invalid-placement boundary 剔除；standalone q11 选择 profile 2，mixed
portfolio 选择 profile 3。早期 BAPR dense curve 虽显示 2-6/GPU 近似平台，
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

当前论文主线只能把 active-bucket 写成 extension / certificate theorem：

```text
已证明：
  active-bucket regret event;
  finite active-bucket union bound;
  high-probability input event -> high-probability stability certificate.

未实证闭合：
  当前 Scheduleurm sampler 的 selection probability、feedback model、
  censoring rule、adaptive bucket creation/reset 和 queue-coupled exploration。
```

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

投稿时安全写法：

```text
We prove a structured active-bucket learning certificate conditional on a
sampler/feedback confidence event.  The current experiments use measured
service-cache certificates rather than claiming a complete online learning
theorem for Scheduleurm's adaptive sampler.
```

## 5. Hidden regime / BOCD 检测延迟

Lean 已经有 dwell-time / switching-window backlog budget：检测和切换窗口只要占每个 segment backlog mass 的 \(\theta\) 比例，就把 drift margin 从 \(m\) 降到 \(m-\chi\theta\)。

当前论文主线只能把 hidden-regime 写成 extension：

```text
已证明：
  dwell/switching marked-window backlog budget;
  uniform-in-regime 和 average-regime 口径已经在 math.md 区分。

未实证闭合：
  BOCD/change-point detector 的 delay tail、false alarm/miss rate、
  per-segment dwell distribution、switching cost envelope。
```

还需要实验或具体 detector model 给出：

```text
change point 发生频率和 dwell time 分布
BOCD / detector delay τ_detect 的经验分布或 tail bound
误报率、漏报率
检测窗口内 backlog mass / segment backlog mass 的 θ
switching / rollback / migration cost χ
```

没有这些量，average-regime stability 和 BOCD delay theorem 只能停在条件式 dwell/switching 证明，不能写成 scheduleurm 的实证闭合定理。

投稿时安全写法：

```text
Hidden regimes are handled as an extension.  The main stability theorem is not
an average-regime BOCD theorem; average-regime claims require dwell-time,
detection-delay, and switching-loss evidence.
```

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

当前状态（2026-06-11）：

```text
Module50 已补 scheduler candidate-family trace hook；
SCHEDULEURM_ORACLE_AUDIT_LOG 打开后，pick_placement 会记录真实候选全集、
chosen action、scheduler sort key、finite bucket/class/regime audit。

这可以验证 selected action 是否为 scheduler-score best。Module50 当前在一条
emitted live trace 上是 SCHEDULER_SCORE_PASS，但它本身仍不是 theorem 证书，
因为 raw trace 使用 scheduler sort-key semantics。

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

Module101 的早期 smoke trace 曾把一条 emitted live scheduler trace 闭合：
  status = LIVE_SCHEDULER_THEOREM_ORACLE_PASS
  trace_slot_count = 2
  candidate_count_total = 2
  alpha0 = 0
  alpha1 = 0

2026-06-11 OR gate 已经把这条路径提升为更长的 live-state dry-run audit：
  artifact = md/or_gate_live_trace.md
  trace_slot_count = 128
  min_required_slots = 96
  candidate_count_total = 767
  trace_origin = synthetic_dryrun_live_node_probe
  no queue mutation and no task launch.

这解决了之前的 NO_TRACE blocker，但作用域是 per emitted trace。未来任何要写成
online oracle evidence 的 scheduler dispatch 仍必须重新走：
  Module50 raw trace
  -> Module55 lower-service enrichment
  -> Module52 theorem oracle audit。
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
records = 3437
strict mapped = 3437
measurement_required = 0
strict mapped_fraction = 1.000000
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
attempted production: not the theorem population.
future rolling queue rows: not automatically closed until rerun through the classifier/service-certificate pipeline.
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
