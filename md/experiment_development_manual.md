# Scheduleurm 实验架子开发手册

本文档是实验系统的开发规范，不是论文正文。它的目标是让任何人或 AI 在不重新猜测数学路线的情况下，把 scheduleurm 的真实运行数据、profiling 实验、校准常数和 Lean 证明一一对应起来。

核心原则：

```text
证明对象不降级；
实验只负责校准证明需要的量；
无法由当前数据校准的量必须显式标成 open empirical obligation；
任何报告给审稿人的数值都必须能追溯到原始 trace、代码 commit 和 theorem 名称。
```

当前最重要的 paper-facing Lean theorem 是：

```text
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle
```

它对应的实验目标是校准下面这个条件：

```text
δ > Lρ + ε_est + β + α1
```

其中 `α0` 不消耗线性 slack，但进入 drift 常数；`α1` 消耗 slack。实验架子必须围绕这个不等式组织，不能只做“看起来调度效果不错”的吞吐图。

---

## 1. 不可混淆的对象定义

### 1.1 时间槽

实验统一使用离散时间槽：

```text
slot k = 第 k 个 watcher/dispatch 观测窗口
slot_start_ts, slot_end_ts = Unix epoch seconds
Δ_k = slot_end_ts - slot_start_ts
```

默认 `Δ_k` 来自 watcher 的实际运行间隔，不强行假设是 60 秒。所有 rate 必须先归一到每秒，再按论文需要换算到固定 `Δ_ref = 60s`。

原始 timestamp 一律存 Unix epoch seconds；报告里可额外显示本地时间，但本地时间不能作为 join key。

### 1.2 Job class \(i\)

job class 是 queue vector 的坐标。第一版必须用确定性函数生成：

```text
class_key(task) =
  project + "/" +
  signature_prefix + "/" +
  workload_kind + "/" +
  resource_bucket
```

字段定义：

```text
project             task["project"]，缺失时取 cwd basename
signature_prefix    signature 的前 2 或 3 段；具体段数写入 manifest
workload_kind       train / eval / inference / data-prep / cpu-batch / unknown
resource_bucket     gpu-vram-small / gpu-vram-mid / gpu-vram-large / cpu-only
```

任何实验 run 必须在 manifest 中固定 `class_key_version`。同一份数据不能中途改变 class 规则后继续合并。

### 1.3 整数 queue state \(Q_i(k)\)

Lean 的 stochastic queue 证明是整数队列，所以实验中的 theorem-grade queue state 也必须是整数。

主定义：

```text
Q_i(k) = slot k 开始时 class i 的未完成 work units 总数
```

work unit 的优先级：

```text
Tier A: 训练脚本有真实 monotone progress current/total，比如 step、iter、epoch；unit = progress unit。
Tier B-count: tqdm 或日志能恢复真实 current/total；unit = recovered progress unit。
Tier B-time: 只有 tqdm elapsed<remaining 或 ETA；unit = ceil(remaining_seconds / Δ_ref)，只能作为 runtime-work proxy。
Tier C: 只有 runtime_history 或 ETA fallback；unit = ceil(eta_seconds / Δ_ref)，只能用于 exploratory report。
Tier D: 完全无进度信号；unit = 1 个 unfinished task，只能用于 queue-count baseline。
```

只有 Tier A 和 Tier B-count 可以直接进入 theorem-grade drift/capacity 校准。Tier B-time 只有在 manifest 中明确把 service process 定义为 time-work units、并额外给出 ETA/workload proxy 的保守误差界后，才能进入 theorem-grade 报告；否则只能用于 throughput/ETA 诊断。Tier C/D 可以用于工程诊断，但不能声称闭合 stochastic theorem。

任务状态纳入规则：

```text
queued      计入 Q
launching   计入 Q
running     计入 Q，剩余 work units 由真实 progress 或已认证的 time-work proxy 更新
done        不计入 Q
failed      如果会自动 requeue，则作为新 arrival；若 terminal failed，则不计入 service success
cancelled   不计入 Q；如果是实验主动停止，要记录 censoring
evicted     仍计入 Q，除非有可证实的 progress loss，需要作为 penalty/rollback cost
```

### 1.4 Arrival \(A_i(k)\)

```text
A_i(k) = slot (k,k+1] 内新进入 active queue 的 class i work units
```

来源：

```text
submit event
auto-requeue after crash/eviction
manual adopt event, if adopted process should enter scheduleurm queue model
```

重复提交被 scheduler dedup 拒绝的任务不计为 arrival。

### 1.5 Service \(S_i(k)\)

```text
S_i(k) = slot (k,k+1] 内 class i 的 completed/progressed work units
```

计算规则：

```text
running task progress 从 u_start 到 u_end:
  service = max(0, u_end - u_start)

task done 且有 total_units:
  service 至少补足到 total_units

task failed/OOM:
  completed service = 已确认 progress increase
  rollback loss 单独进入 penalty，不得当作负 service
```

如果 progress counter 重置，必须用 checkpoint/resume 证据判断是否是真实 rollback；否则该 slot 标成 `service_censored=true`，不得进入 theorem-grade service map。

### 1.6 Regime \(z\)

主 theorem 先使用 fixed/uniform regime，不把 hidden-regime 平均稳定混入主结论。

实验中 regime 是一个可观测标签：

```text
node_name
gpu_model / gpu_total_mb
driver_cuda_bucket
external_load_bucket
time_of_day_bucket, optional
workload_phase_bucket, optional
```

如果外部任务长期占 GPU，不能忽略；必须进入 `external_load_bucket`，或者该 slot 被排除出 clean profiling set。

regime 版本：

```text
fixed-regime report       只用同一个 z 的数据
uniform-regime report     每个 z 都单独验证 slack margin
average-regime report     只作为 extension，必须额外给 dwell/switching evidence
```

### 1.7 Action \(a\)

Lean 里的 action 是 global configuration action。scheduleurm 当前代码是逐任务 greedy dispatch，但实验必须把一个 dispatch cycle 的最终配置提升成 global action。

定义：

```text
a(k) = slot k dispatch 后的全局配置
```

它包含：

```text
每个 active running/launching task 的 node
每个 GPU task 的 gpu_idx
每个 CPU task 的 node
每个 task 的 class_key、resource estimate、current progress tier
每个 node/GPU 上的 co-location profile
本 cycle 的 migration/preemption/eviction/rollback 标记
```

`a_chosen(k)` 是 scheduler 实际选择的 action。`A_cand(k)` 是 candidate generator 给出的候选 action set。`A_full(k)` 是当前状态下满足硬资源约束、pinning 约束、ckpt 互斥约束、node availability 约束的完整 feasible action family。

如果 `A_full(k)` 太大无法枚举，必须区分：

```text
full_exact        小状态下精确枚举
full_sampled      大状态下采样 full feasible actions，只能给 sampled-cover evidence
historical_full   用历史 observed feasible actions，只能给 historical-cover evidence
```

报告不能把 sampled/historical cover 写成 all-feasible cover。

---

## 2. 和 Lean theorem 的一一对应表

| 实验量 | 含义 | Lean theorem/definition | 数据来源 |
|---|---|---|---|
| \(Q_i(k)\) | 整数 backlog/work units | stochastic queue model in `ConcreteStochasticModel.lean` | queue snapshot + progress parser |
| \(A_i(k)\) | arrival work units | `expectedLyapunovDrift_eq` | submit/requeue/adopt events |
| \(S_i(k)\) | service work units | `expectedLyapunovDrift_eq` | progress/done events |
| \(Amax_i\) | per-slot arrival bound | `expectedSecondOrder_le_of_coord_sample_bounds` | max observed/protocol cap |
| \(Smax_i\) | per-slot service bound | `expectedSecondOrder_le_of_coord_sample_bounds` | max observed/protocol cap |
| \(B\) | conditional second-order bound | `main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound` | empirical/protocol bound |
| \(\mu_i^z(a)\) | expected service map | `main_concrete_finite_support_stochastic_stability_from_coordinate_moments` | service samples grouped by action/regime |
| `lower_i(a)` | lower service certificate | robust drift theorem family | LCB/posterior lower bound |
| \(\epsilon_{est}\) | lower-service support error | `main_robust_candidate_maxweight_drift_approx_oracle` | validation residual |
| \(d_\Phi\) | finite-feature fabric metric | `FabricCandidateProjection.covers` | action feature extraction |
| \(L\) | service Lipschitz envelope | `fabric_service_lipschitz_of_feature_sensitivity` | perturbation profiling |
| \(\rho\) | candidate cover radius | `calibrated_fabric_cover_support_gap` | candidate/full action audit |
| \(P0,\beta\) | penalty growth constants | `robust_candidate_policy_lyapunov_drift_scaled_penalty_approx_oracle` | migration/rollback/preemption logs |
| \(\alpha0,\alpha1\) | approximate oracle loss | `QueueScaledApproxRobustScoreMaximizer` | oracle audit |
| \(\delta\) | offered-load capacity slack | `main_downward_capacity_support_slack` | capacity LP/support gap |
| \(\eta\) | final drift margin | `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle` | computed from all constants |

Reviewer-facing final margin:

```text
η = δ - (Lρ + ε_est + β + α1)
```

Theorem-grade stability statement can only be claimed when:

```text
η > 0
finite-set threshold N is computed from B/P0/α0/η
service lower domination is certified on the confidence event or by deterministic profiling
```

---

## 3. 实验文件布局

实现时新增目录：

```text
~/.claude/scheduler/experiments/
  runs/
    <run_id>/
      manifest.json
      raw/
        watcher.log.jsonl
        queue_snapshots.jsonl
        command_outputs.jsonl
      normalized/
        slots.jsonl
        task_samples.jsonl
        action_chosen.jsonl
        action_candidates.jsonl
        service_samples.jsonl
        arrival_samples.jsonl
      calibration/
        class_map.json
        regime_map.json
        fabric_features.json
        fabric_metric.json
        service_lower_bounds.json
        fabric_cover_audit.json
        oracle_audit.json
        penalty_fit.json
        capacity_slack.json
        drift_margin.json
      reports/
        theorem_checklist.md
        calibration_table.md
        throughput_curves.md
        reviewer_summary.md
```

Repo 内建议新增代码目录：

```text
algorithm/
  __init__.py
  action_model.py
  features.py
  scoring.py
  candidates.py
  placement.py
  README.md
  experiments/
    __init__.py
    schema.py
    trace_export.py
    slot_builder.py
    class_key.py
    regime_key.py
    progress_units.py
    action_model.py
    candidate_generator.py
    full_action_sampler.py
    fabric_metric.py
    service_model.py
    oracle_audit.py
    penalty_fit.py
    capacity_lp.py
    throughput_curves.py
    report.py
```

测试目录：

```text
skill/tests/test_experiment_schema.py
skill/tests/test_algorithm_math_surface.py
skill/tests/test_algorithm_policy.py
skill/tests/test_experiment_slot_builder.py
skill/tests/test_experiment_action_model.py
skill/tests/test_experiment_calibration.py
```

手册先固定接口；代码实现必须遵守这些文件名和字段，除非同步更新本手册。

---

## 4. Manifest schema

每个实验 run 必须生成 `manifest.json`。

```json
{
  "schema_version": "scheduleurm-exp-v1",
  "run_id": "2026-05-29T001500+0800_local_observe_v1",
  "created_at_ts": 1780004100.0,
  "scheduleurm_repo": "/home/erzhu419/mine_code/scheduleurm",
  "scheduleurm_commit": "<git rev-parse HEAD>",
  "scheduleurm_dirty": true,
  "proof_repo": "/home/erzhu419/mine_code/proof",
  "proof_commit": "<git rev-parse HEAD>",
  "proof_upload_sha256": "571e46608c878c5a3eba51cda296a56c37948d89d6f15d133c529b59dd612cbb",
  "theorem_target": "main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle",
  "slot_policy": {
    "source": "watcher_cycle",
    "delta_ref_s": 60,
    "min_slot_s": 10,
    "max_slot_s": 300
  },
  "class_key_version": "v1_project_sigprefix_kind_resource",
  "signature_prefix_depth": 3,
  "regime_key_version": "v1_node_gpu_external_load",
  "action_space_mode": "full_exact|full_sampled|historical_full",
  "candidate_generator_version": "v1_scheduleurm_greedy_neighborhood",
  "progress_tiers_allowed_for_theorem": ["A", "B-count"],
  "time_work_proxy_allowed_for_theorem": false,
  "notes": ""
}
```

`scheduleurm_dirty=true` 不禁止实验，但 reviewer report 必须说明数据来自 dirty working tree。正式提交结果最好使用 clean commit。

---

## 5. Normalized schema

### 5.1 `slots.jsonl`

每行一个 slot。

```json
{
  "slot_id": 17,
  "slot_start_ts": 1780005000.0,
  "slot_end_ts": 1780005062.4,
  "delta_s": 62.4,
  "watcher_event_ts": 1780005062.5,
  "source_event_type": "dispatch_cycle",
  "nodes_alive": ["local", "jtl110gpu", "jtl110gpu2"],
  "excluded_from_theorem": false,
  "exclude_reason": ""
}
```

### 5.2 `task_samples.jsonl`

每个 slot、每个 active task 一行。

```json
{
  "slot_id": 17,
  "task_id": "t0123",
  "status": "running",
  "class_key": "BAPR/r3/train/gpu-vram-mid",
  "project": "BAPR",
  "signature": "BAPR/r3/seed42",
  "node": "jtl110gpu",
  "gpu_idx": 0,
  "progress_tier": "A",
  "work_total_units": 100000,
  "work_done_units": 24000,
  "work_remaining_units": 76000,
  "eta_seconds": 45600,
  "eta_source": "progress_rate",
  "current_vram_mb": 3200,
  "peak_vram_mb": 3500,
  "current_ram_mb": 12000,
  "peak_ram_mb": 15000,
  "cpu_cores": 2,
  "external_or_adopted": false,
  "service_censored": false,
  "censor_reason": ""
}
```

### 5.3 `arrival_samples.jsonl`

```json
{
  "slot_id": 17,
  "class_key": "BAPR/r3/train/gpu-vram-mid",
  "arrival_units": 100000,
  "arrival_tasks": ["t0124"],
  "arrival_kind": "submit|requeue|adopt"
}
```

### 5.4 `service_samples.jsonl`

```json
{
  "slot_id": 17,
  "class_key": "BAPR/r3/train/gpu-vram-mid",
  "task_id": "t0123",
  "service_units": 1200,
  "service_units_per_s": 19.23,
  "service_units_per_delta_ref": 1154,
  "progress_tier": "A",
  "regime_key": "jtl110gpu/3080Ti12GB/extload-low",
  "action_id": "a_000017_chosen",
  "censored": false,
  "censor_reason": ""
}
```

### 5.5 `action_chosen.jsonl`

```json
{
  "slot_id": 17,
  "action_id": "a_000017_chosen",
  "action_kind": "chosen",
  "regime_key": "jtl110gpu/3080Ti12GB/extload-low",
  "global_config": [
    {
      "task_id": "t0123",
      "class_key": "BAPR/r3/train/gpu-vram-mid",
      "node": "jtl110gpu",
      "gpu_idx": 0,
      "cpu_cores": 2,
      "ram_mb": 16000,
      "est_vram_mb": 3500
    }
  ],
  "co_location": [
    {
      "node": "jtl110gpu",
      "gpu_idx": 0,
      "task_count": 3,
      "class_multiset": [["BAPR/r3/train/gpu-vram-mid", 2], ["eval/cpu", 1]],
      "used_vram_mb": 9200,
      "total_vram_mb": 12288,
      "gpu_util_pct": 87
    }
  ],
  "events": {
    "launched": ["t0124"],
    "evicted": [],
    "migrated": [],
    "preempted": []
  }
}
```

### 5.6 `action_candidates.jsonl`

每个 slot 可有多行。candidate 必须是完整 global config，不只是单个 placement。

```json
{
  "slot_id": 17,
  "candidate_id": "a_000017_cand_003",
  "candidate_source": "scheduleurm_greedy_neighbor|beam|exact_full|sampled_full",
  "is_in_A_cand": true,
  "is_in_A_full_sample": true,
  "global_config": [],
  "feature_vector": {
    "gpu_colocation_count_max": 4,
    "gpu_colocation_count_sum": 9,
    "vram_pressure_max": 0.78,
    "same_signature_pairs": 2,
    "node_switch_count": 1,
    "rollback_risk_count": 0
  },
  "resource_feasible": true,
  "feasibility_reason": "ok"
}
```

---

## 6. Feature metric \(d_\Phi\)

### 6.1 Required feature vector

第一版 feature 必须至少包含：

```text
per-GPU task_count
per-GPU class multiset hash
per-GPU used_vram_ratio
per-GPU estimated_vram_ratio_after_action
per-node free_ram_after_headroom ratio
per-node CPU reserved ratio
same_signature_colocation count
same_project_colocation count
node_switch_count relative to previous action
gpu_switch_count relative to previous action
eviction_or_preemption_count
external_load_bucket
```

每个 feature 必须有：

```text
name
type: integer / real / categorical / multiset
normalization
source field
missing value rule
```

### 6.2 Metric

```text
d_Φ(a,a') = Σ_r w_r |Φ_r(a)-Φ_r(a')|
```

categorical feature 用 `0/1` distance；multiset feature 用 normalized l1 count distance。

`fabric_metric.json` 必须写：

```json
{
  "metric_version": "v1",
  "features": [
    {
      "name": "gpu_colocation_count_max",
      "weight": 1.0,
      "normalization": "divide_by_max_tasks_per_gpu",
      "source": "action.co_location.task_count"
    }
  ],
  "distance": "weighted_l1"
}
```

---

## 7. Calibration algorithms

### 7.1 Service map \(\mu_i^z(a)\)

Group service samples by:

```text
class_key i
regime_key z
action feature bucket Φ(a)
```

Exact global actions are too sparse; service model should estimate on feature buckets, then map back to action features.

Required outputs:

```json
{
  "class_key": "BAPR/r3/train/gpu-vram-mid",
  "regime_key": "jtl110gpu/3080Ti12GB/extload-low",
  "feature_bucket": "hash",
  "n_samples": 48,
  "mean_service_per_delta_ref": 1120.5,
  "lcb_service_per_delta_ref": 980.0,
  "lcb_method": "empirical_bernstein|bootstrap|quantile",
  "confidence": 0.95,
  "usable_for_theorem": true
}
```

`lower_i(a)` is the LCB value. If sample count is too small, set:

```text
usable_for_theorem=false
reason="insufficient_samples"
```

### 7.2 \(\epsilon_{est}\)

Validation split must be time-based, not random per row, to avoid leakage across neighboring slots.

Definition:

```text
ε_est = sup over validation slots and q≥0, ||q||_1=1:
        qᵀ(μ_empirical(a) - lower(a))_+
```

Implementation simplification:

```text
ε_est = max_i max_validation_bucket max(0, μ_empirical_i(bucket) - lower_i(bucket))
```

This is conservative and matches support-function loss because q is nonnegative with l1 norm 1.

### 7.3 \(L\)

For pairs of actions in the same regime:

```text
ratio_i(a,a') = |μ_i(a)-μ_i(a')| / max(d_Φ(a,a'), d_min)
```

Set:

```text
L = max_i high_confidence_envelope(ratio_i)
```

`d_min` must be written in manifest. Pairs with `d_Φ=0` but materially different service expose missing features; they must fail calibration unless the regime is split or features are expanded.

### 7.4 \(\rho\)

For each sampled or exact full action `a_full`:

```text
ρ_sample(a_full) = min_{a_cand in A_cand} d_Φ(a_full, a_cand)
ρ = max_a_full ρ_sample(a_full)
```

Output must record cover domain:

```text
all_feasible
sampled_feasible
historical_observed
statewise_uniform
regimewise_uniform
```

Only `all_feasible` or explicitly justified `statewise_uniform` may be used for the strongest main theorem claim.

### 7.5 \(P0,\beta\)

Penalty model:

```text
K + R_t ≤ P0 + β ||Q(t)||_1
```

Penalty events:

```text
migration staging time
checkpoint sync time
eviction rollback estimated lost units
preemption lost work or delay
launch failure retry delay
resource pressure forced requeue
```

Fit a conservative upper envelope:

```text
penalty_units(k) ≤ P0 + β * Q_norm(k)
```

Use quantile fit only for engineering diagnosis. The theorem-grade value must be max-envelope or confidence-envelope with explicit failure probability.

### 7.6 \(\alpha0,\alpha1\): approximate oracle audit

For each slot:

```text
score(a;Q) = Qᵀ lower(a) - penalty(a)
```

Compute:

```text
best_candidate_score = max_{a in A_cand(k)} score(a;Q(k))
chosen_score = score(a_chosen(k);Q(k))
oracle_gap(k) = max(0, best_candidate_score - chosen_score)
```

Fit:

```text
oracle_gap(k) ≤ α0 + α1 ||Q(k)||_1
```

The report must include:

```text
number of audited slots
candidate generator used
whether best_candidate_score is exact over A_cand or approximate
α0
α1
max residual violation
```

If the best score is itself approximate, the result is not theorem-grade unless the approximation error is also bounded and added to `α0, α1`.

### 7.7 \(\delta\): capacity slack

For each regime or uniform regime set, solve:

```text
maximize δ
subject to λ_i + δ ≤ Σ_a x_a μ_i(a) for all i
           x ∈ Δ(A_full)
           optional downward closure handled by service domination
```

If using candidate capacity, report separately:

```text
δ_full
δ_cand
Lρ loss
δ_full - Lρ lower bound
```

Never report “operational iff” from this LP. The correct wording is:

```text
positive-slack sufficiency plus zero-slack conservation-law necessity form a capacity sandwich
```

### 7.8 \(B,Amax,Smax\)

For theorem-grade finite-support specialization:

```text
Amax_i = protocol or observed max arrival units per slot for class i
Smax_i = protocol or observed max service units per slot for class i
B = 1/2 * Σ_i (Amax_i^2 + Smax_i^2)
```

If using bounded conditional second moment instead:

```text
B = conservative upper confidence bound of E[1/2 Σ_i(A_i^2+S_i^2)|Q]
```

The report must say which version is used.

---

## 8. Throughput curve and current GPU occupancy problem

The user’s current constraint is that GPUs are often occupied, with 3-4 tasks per GPU. The experiment design must not wait for empty GPUs before collecting anything.

Use two modes:

```text
observational mode:
  use naturally occurring co-location from watcher logs;
  external/untracked load becomes regime feature or censor reason;
  useful for service map, ETA error, rough g_z(n).

controlled perturbation mode:
  run short, resumable profiling jobs under fixed co-location profiles;
  required for clean L,ρ,sweet-spot claims.
```

For single-GPU co-location:

```text
g_z(n) = total service units per Δ_ref for all theorem-grade tasks on the GPU
```

Report:

```text
n
sample_count
mean g_z(n)
LCB g_z(n)
failure/OOM rate
rollback rate
median ETA error in first 3 slots
median ETA error after warmup
```

The early ETA issue must be explicitly tracked:

```text
eta_warmup_slot_count = 3 by default
eta_abs_error_first_slots
eta_abs_error_after_warmup
eta_overestimate_factor_p95
```

Early ETA spikes may be excluded from migration/oracle scoring only if the exclusion rule is fixed in manifest before analysis.

---

## 9. Active-bucket learning layer

This layer is not required for the main stability theorem. It is an extension certificate.

Bucket definition must be deterministic:

```text
bucket_key =
  class_key + "/" +
  regime_key + "/" +
  fabric_neighborhood_bucket + "/" +
  co_location_profile_bucket
```

For each bucket:

```text
bucketCount(b)
mean loss/service
confidence radius
last_reset_ts
change_point_epoch
```

The theorem names this layer maps to:

```text
main_active_bucket_lcb_learning_regret
main_active_bucket_lcb_learning_regret_high_probability
main_active_bucket_local_failure_union_bound
main_high_probability_stability_from_certificate_event
```

Do not call this a full online learning theorem until the sampler, feedback model, selection probability and queue-coupled exploration safety are implemented.

---

## 10. Hidden-regime extension

Hidden-regime results stay out of the main theorem unless uniform-in-regime slack is verified.

For dwell/switching extension, collect:

```text
segment_id
regime_before
regime_after
change_ts
detect_ts
τ_detect
marked_window_slots
backlog_mass_in_marked_window
total_segment_backlog_mass
θ = marked / total
switching_penalty_units
```

Lean theorem:

```text
main_hidden_regime_dwell_switching_drift
```

Report wording:

```text
conditional dwell/switching certificate
```

not:

```text
full average-regime positive recurrence
```

---

## 11. Implementation phases

### Phase 0: Reproducibility identity

Add a command:

```bash
python3 -m algorithm.experiments.trace_export init-run --run-id <id>
```

It must write `manifest.json` with scheduleurm commit, proof commit, proof upload hash, dirty status and theorem target.

Acceptance:

```text
manifest exists
all paths absolute
proof_upload_sha256 matches ScheduleurmUpload.lean
manifest records algorithm.action_model/features/scoring/candidates/placement module versions
```

### Phase 1: Passive trace export

Input:

```text
~/.claude/scheduler/queue.json
~/.claude/scheduler/queue_archive.jsonl
~/.claude/scheduler/logs/watcher.log
~/.claude/scheduler/vram_history.json
~/.claude/scheduler/runtime_history.json
```

Output:

```text
raw/*.jsonl
normalized/slots.jsonl
normalized/task_samples.jsonl
```

No scheduler behavior changes in this phase.

Acceptance:

```text
can parse current existing logs without crashing
every active task has class_key
every slot has a regime_key for each alive node/GPU
all theorem-ineligible rows have explicit censor_reason
```

### Phase 2: Progress and service builder

Implement:

```text
progress_units.py
service_samples.jsonl
arrival_samples.jsonl
Q/A/S reconstruction check
```

Acceptance:

```text
for each class and slot:
  Q_i(k+1) = max(Q_i(k)-S_i(k),0)+A_i(k) plus explicit censor adjustment
no negative service
progress reset is censored or explained by rollback/resume
```

### Phase 3: Action model

Implement:

```text
action_chosen.jsonl
candidate_generator.py
action_candidates.jsonl
feature_vector extraction
algorithm.features.gpu_candidate_features
algorithm.candidates.active_bucket_representatives
algorithm.scoring.score_gpu_candidate
```

Acceptance:

```text
chosen action is reconstructible from queue snapshots
candidate action is complete global config
resource infeasible candidates are either excluded or labeled infeasible
every feature has source field and missing rule
active-bucket representative selection is auditable by candidate_bucket
```

### Phase 4: Fabric metric and cover audit

Implement:

```text
fabric_metric.py
full_action_sampler.py
fabric_cover_audit.json
```

Acceptance:

```text
ρ reported with cover domain
statewise/regimewise uniformity is explicitly checked or not claimed
d_Φ=0 with service gap triggers failed calibration unless features/regime split fixed
```

### Phase 5: Service lower bounds and \(ε_est\)

Implement:

```text
service_model.py
service_lower_bounds.json
ε_est validation
```

Acceptance:

```text
time-based validation split
LCB method recorded
usable_for_theorem=false for sparse buckets
ε_est computed from validation residuals
```

### Phase 6: Penalty and approximate oracle audit

Implement:

```text
penalty_fit.py
oracle_audit.py
penalty_fit.json
oracle_audit.json
```

Acceptance:

```text
P0,β envelope has no theorem-grade violations
α0,α1 envelope has no theorem-grade violations
chosen_score and best_candidate_score use the same lower_i(a) and penalty(a)
if best candidate search is approximate, its error is charged
```

### Phase 7: Capacity and drift margin

Implement:

```text
capacity_lp.py
capacity_slack.json
drift_margin.json
```

Acceptance:

```text
δ_full reported
η = δ-(Lρ+ε_est+β+α1) reported
B/P0/α0 finite-set threshold N reported
if η≤0, report says theorem condition not met rather than hiding it
```

### Phase 8: Throughput curves

Implement:

```text
throughput_curves.py
reports/throughput_curves.md
```

Acceptance:

```text
g_z(n) curves separated by regime and class mix
first-slot ETA error separated from post-warmup ETA error
observational and controlled data are not merged without label
```

### Phase 9: Reviewer report

Implement:

```text
report.py
reports/theorem_checklist.md
reports/calibration_table.md
reports/reviewer_summary.md
```

Acceptance:

```text
every theorem assumption is marked PASS / FAIL / EMPIRICAL-OPEN
every PASS links to a JSON artifact path
every FAIL says what extra experiment is needed
the report includes proof hash and theorem names
```

---

## 12. Required report table

`reports/calibration_table.md` must contain this exact table shape:

| Quantity | Value | Unit | Source artifact | Theorem role | Status |
|---|---:|---|---|---|---|
| \(Amax_i\) | | work units/slot | | second moment bound | PASS/FAIL |
| \(Smax_i\) | | work units/slot | | second moment bound | PASS/FAIL |
| \(B\) | | work units squared | | drift constant | PASS/FAIL |
| \(L\) | | service per metric | | fabric Lipschitz | PASS/FAIL |
| \(\rho\) | | metric distance | | candidate cover radius | PASS/FAIL |
| \(L\rho\) | | service units/slot | | candidate support loss | PASS/FAIL |
| \(\epsilon_{est}\) | | service units/slot | | lower-service estimation loss | PASS/FAIL |
| \(P0\) | | service units/slot | | fixed penalty constant | PASS/FAIL |
| \(\beta\) | | service units per backlog unit | | queue-scaled penalty | PASS/FAIL |
| \(\alpha0\) | | score units | | fixed oracle error | PASS/FAIL |
| \(\alpha1\) | | score per backlog unit | | queue-scaled oracle error | PASS/FAIL |
| \(\delta\) | | service units/slot | | full capacity slack | PASS/FAIL |
| \(\eta\) | | service units/slot | | final drift margin | PASS/FAIL |

The report must not replace missing values with zero. Missing values are `NA` and force status `EMPIRICAL-OPEN`.

---

## 13. Minimal command surface after implementation

The final experiment CLI should expose:

```bash
python3 -m algorithm.experiments.trace_export init-run --run-id <id>
python3 -m algorithm.experiments.trace_export export --run-id <id> --since <ts> --until <ts>
python3 -m algorithm.experiments.slot_builder build --run-id <id>
python3 -m algorithm.experiments.action_model build --run-id <id>
python3 -m algorithm.experiments.fabric_metric calibrate --run-id <id>
python3 -m algorithm.experiments.service_model calibrate --run-id <id>
python3 -m algorithm.experiments.penalty_fit fit --run-id <id>
python3 -m algorithm.experiments.oracle_audit audit --run-id <id>
python3 -m algorithm.experiments.capacity_lp solve --run-id <id>
python3 -m algorithm.experiments.throughput_curves build --run-id <id>
python3 -m algorithm.experiments.report build --run-id <id>
```

Each command must:

```text
read only its declared inputs
write deterministic JSON/Markdown outputs
be rerunnable without changing results unless raw input changed
exit nonzero on schema violation
```

---

## 14. Tests that must exist before trusting results

Minimum tests:

```text
test class_key is deterministic and versioned
test queue reconstruction Q(k+1)=max(Q-S,0)+A
test service never negative
test progress reset is censored
test action chosen reconstruction from queue/event snapshots
test candidate feature vector has no missing undocumented field
test d_Φ is symmetric and zero only for identical feature vectors
test cover audit computes max min-distance correctly on toy finite action sets
test oracle gap fit charges α0 and α1 correctly
test capacity LP recovers known two-action toy capacity
test report marks η≤0 as FAIL, not PASS
```

These tests should run through the existing regression entry point if possible:

```bash
python3 skill/test_regression.py
```

If experiment modules need third-party packages, keep core schema/JSON tests dependency-free.

---

## 15. What to show a reviewer

Reviewer-facing bundle:

```text
proof/ScheduleurmUpload.lean
md/lean_artifact_map.md
md/lean_verification_round4.md
md/math.md
md/experimental_open_items.md
experiments/runs/<run_id>/manifest.json
experiments/runs/<run_id>/reports/theorem_checklist.md
experiments/runs/<run_id>/reports/calibration_table.md
experiments/runs/<run_id>/reports/reviewer_summary.md
```

Do not show internal GPT review files unless asked. Do not claim full operational iff stability. Correct reviewer language:

```text
We prove a positive-slack robust MaxWeight sufficiency theorem and a zero-slack conservation-law necessity theorem, forming a capacity sandwich. The experiments calibrate the constants in the sufficient condition.
```

---

## 16. Stop conditions

The experiment pipeline must stop and refuse theorem-grade report if any of these holds:

```text
proof upload hash missing or mismatched
no theorem target in manifest
Q/A/S reconstruction fails without censor reason
service lower bound has uncovered class/regime/action bucket used by chosen policy
ρ computed only on historical actions but report requests all-feasible theorem
oracle audit uses approximate best score without charging its approximation error
η ≤ 0
no finite-set threshold N can be computed from B/P0/α0/η
```

Engineering report may still be generated, but it must be labeled:

```text
not theorem-grade; empirical diagnostic only
```
