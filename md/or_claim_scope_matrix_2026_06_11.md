# OR Claim Scope Matrix

Date: 2026-06-11, Asia/Shanghai.

This note is the submission-facing claim boundary for the current Scheduleurm
theory and experiment package.  It is meant to prevent reviewer-facing wording
from mixing theorem certificates, replay baselines, local bucket measurements,
and future extensions.

## Main Safe Claim

```text
The current completed-active Scheduleurm-controlled production population has
full service-domain admission coverage, positive mapped-slice capacity slack,
a service-map robust MaxWeight lower-service oracle bridge with
alpha0=alpha1=0, and a theorem-grade live-node candidate trace for certified
q01/q11 dispatch decisions.
```

Primary artifacts:

```text
md/experiment_artifacts/module51_production_coverage_drilldown.json
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module100_production_theorem_oracle_bridge.json
md/experiment_artifacts/or_submission_closure_2026_06_11.json
md/experiment_artifacts/or_submission_closure_2026_06_11_extended.json
md/experiment_artifacts/or_gate_live_trace.json
md/experiment_artifacts/natural_live_theorem_trace.json
md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003.json
md/experiment_artifacts/or_submission_closure_2026_06_11_live_dispatch.json
md/experiment_artifacts/admission_population_gate.json
md/experiment_artifacts/direct_sota_baseline_scaffold.json
md/experiment_artifacts/global_fabric_cover_calibration.json
md/experiment_artifacts/direct_sota_baseline_binary_smoke_20260611_round2.json
md/experiment_artifacts/global_fabric_cover_calibration_cover_curve_20260611.json
md/experiment_artifacts/active_bucket_hidden_regime_certificate_20260611.json
md/experiment_artifacts/production_live_theorem_trace_gate_20260611.json
md/experiment_artifacts/production_shadow_theorem_trace_20260612.json
md/experiment_artifacts/gavel_direct_native_smoke_20260612.json
md/experiment_artifacts/direct_sota_fullstack_readiness_20260612.json
md/experiment_artifacts/adaptive_sampler_detector_certificate_20260612.json
md/experiment_artifacts/production_queued_theorem_trace_20260612.json
md/experiment_artifacts/production_live_theorem_trace_gate_20260612.json
md/experiment_artifacts/fabric_cover_contract_certificate_20260612.json
md/experiment_artifacts/adaptive_live_integration_probe_20260612.json
md/experiment_artifacts/q00_q10_generalization_gate_20260612.json
md/experiment_artifacts/sota_fullstack_claim_matrix_20260612.json
md/experiment_artifacts/gavel_native_performance_microbaseline_20260612.json
md/experiment_artifacts/future_admitted_fabric_cover_gate_20260612.json
md/experiment_artifacts/future_production_admission_contract_20260612.json
md/experiment_artifacts/q00_q10_broad_envelope_gate_20260612.json
md/experiment_artifacts/sota_fullstack_superiority_gate_20260612.json
md/experiment_artifacts/all_state_conservative_cover_gate_20260612.json
md/experiment_artifacts/production_launch_completion_gate_20260612.json
md/experiment_artifacts/gavel_service_unit_equivalence_certificate_20260612.json
md/experiment_artifacts/declared_finite_domain_positive_cover_gate_20260612.json
md/experiment_artifacts/controlled_production_completion_gate_20260612.json
md/experiment_artifacts/or_submission_closure_2026_06_12_gap_closure.json
md/lean_verification_submission.md
```

## Claim Boundaries

| Topic | Current status | Safe wording | Do not claim |
|---|---|---|---|
| Production population | Admission gate has full service-domain coverage for the completed-active view; strict classifier coverage is reported separately. | `completed_active_production` is the theorem-facing production population when admitted by the measured service domain. | Raw history or attempted production is globally closed. |
| Raw scheduler history | Module49 raw-window global coverage is intentionally false: 6957 records, 4825 mapped, 2132 unmapped, mapped fraction about 0.694. | Raw history is operational telemetry and a source for bucket discovery. | Raw history is a clean arrival stream for the theorem. |
| Service-map oracle bridge | Module100 is `SERVICE_MAP_THEOREM_ORACLE_PASS`. | Measured service-map lower-service oracle bridge closes alpha0=alpha1=0 for the completed-active population. | Module100 is a live dispatch trace. |
| Live scheduler hook | `theorem_maxweight_v1` is an optional lower-service robust MaxWeight placement hook for service-certified GPU candidates; `sweetspot_v1` remains the scalar scoring hook. | Certified live-node candidate rows carry queue vector, lower service, bounded penalty, selected action, and robust score semantics. | Every uncertified future dispatch is automatically theorem-grade. |
| Live scheduler trace | OR gate is `LIVE_SCHEDULER_THEOREM_ORACLE_PASS` for a 128-slot synthetic dry-run; natural theorem trace has 12 slots and 60 certified candidates; launched bounded trace has 6 real tasks, 7 theorem slots, 13 candidate rows, and completion bridge PASS. An earlier 2026-06-12 queued-production trace had 1 admissible BAPR task, 1 theorem slot, oracle audit PASS, and no queue mutation or launch. The production launch/completion and controlled completion gates observe active-production progress and high local GPU utilization, so they refuse unsafe launch. The controlled gate recognizes the existing bounded launched completion trace, keeps `controlled_32_task_completion_ready=false`, and keeps `large_scale_organic_launched_completion_ready=false`. | The robust trace path works on current live-node candidate families; the launched ScheduleurmBench q01/q11 trace validates bounded real dispatch and completion; the shadow / launch-completion / controlled gates validate current active-production progress, canary readiness, and theorem-subset semantics without unsafe queue mutation. | The dry-run, queued trace, or shadow trace is a launched-job completion trace; the launched bounded trace is production-wide; bounded controlled completion proves 32-task or organic production completion. |
| Policy-semantics replay comparison | OR online gate covers 120 Poisson/bursty/load-sweep scenarios on the same measured service cache; direct SOTA scaffold records cloned repo entrypoints and blockers. Pollux/AdaptDL and Decima entrypoint smoke pass. Gavel dependency imports, protobuf stub generation, entrypoint help, native generated-jobs simulator smoke, native Gavel trace smoke, and Scheduleurm q01 native trace-seed smoke all pass in an isolated copy. The Gavel native microbaseline completes bounded q01 and q11 same-trace windows through Gavel's native simulator. The Gavel service-unit certificate now closes trace/schema compatibility and throughput seed readiness while keeping `gavel_service_unit_equivalence_ready=false`. Same-workload seed artifacts now exist for Gavel, Pollux/AdaptDL, IADeep, Salus, and Decima. The strict full-stack superiority gate records `direct_fullstack_sota_superiority_ready=false`, `full_stack_ready_count=0`, and hard blockers: missing Docker, Go, Kubernetes tooling, service-unit equivalence, or scope match. | Scheduleurm is not Pareto-dominated by SOTA-inspired policy-semantics replay baselines under this cache and 0.5% sampled-replay tolerance; direct-adapter smoke, Gavel native microbaseline, Gavel service-unit blocker, seed readiness, and hard full-stack blockers are reported separately from full-stack baseline readiness. | Directly beats Gavel/Pollux/Sia/IADeep binaries or full external stacks. |
| q00 local control | Profiles 1-13 measured; profile 14 boundary; the broad measured-envelope gate includes one admitted q00 service-cache workload. | q00 is closed for the declared local light-control bucket and for admitted measured q00 service-cache rows. | q00 represents every low-resource control or data-loader workload. |
| q10 local CPU | Profiles 1-9 measured; profile 10 boundary; the broad measured-envelope gate includes 102 admitted q10 CPU-like service-cache workloads with positive lower-service rows. | q10 is closed for the declared local CPU-heavy bucket and for admitted measured q10 service-cache rows. | q10 represents all possible remote CPU-node or data-loader-heavy deployments. |
| q11 robust hybrid RL | Profiles 1-9 remain feasible under the current lower-service cache; profile 10 is the first live robust boundary and excludes 10+. | q11 is closed for the declared `hybrid_rl_resac_ant` robust node bucket; current replay selects profile 2 standalone and profile 3 in the portfolio. | Historical profile-10 replay rows remain valid current robust actions, or q11 covers all RL/BAPR deployments. |
| General fabric cover \(L,\rho\) | Global fabric-cover calibration is closed for exact measured finite slices with \(\rho=0\). The 2026-06-12 contract artifact gives the feature map, numeric scales, projection population, exact \(\rho\), compressed-cover \(L\rho\), and exclusions. The future-admitted fabric-cover gate closes identity projection with \(\rho=0\) for future tasks admitted to exact measured positive profiles. The declared finite-domain gate covers 193 exact service-cache buckets: 187 positive lower-service rows, 6 capacity boundaries, and 0 uncovered buckets. The conservative all-state gate covers every scheduler-visible state by routing unmeasured states to a zero-service probe/defer action; it explicitly sets positive-service all-state readiness to false. | Exact measured-slice, future-admitted measured-state, and declared finite-domain claims may use the calibrated finite-feature table; smaller candidate-family claims must spend the reported \(L\rho\) from the cover curve or provide a stronger projection certificate. All-state safety may be claimed only as probe/defer safety for unknown states. | Small \(L\rho\), positive service, or stability for every arbitrary future feasible cluster state. |
| Exact measured finite slices | Module48/100 and OR holdout gate use measured finite service maps; the holdout diagnostic is separate from the theorem-facing finite lower-service model. | Exact finite-slice certificates can use the measured lower-service cache directly. | Sparse holdout residuals prove stochastic generalization outside measured rows. |
| Active-bucket learning | Lean gives conditional/high-probability event lifting. The active-bucket certificate closes the event-level finite union bound over 64 observed replay buckets. The 2026-06-12 certificate adds a concrete deterministic round-robin forced sampler with 1561 minimum forced samples per bucket and Hoeffding radius 0.0523. | Active-bucket learning is an extension theorem with a concrete deployable sampler probability model; live scheduler integration remains a separate A/B step. | Complete online learning theorem for the currently deployed scheduler sampler. |
| Hidden regimes | Lean gives dwell/switching backlog-budget theorem. The hidden-regime certificate records deterministic replay dwell/switching budgets over 120 online scenarios. The 2026-06-12 certificate adds a bounded two-window detector with union tail bound 0.00564 and detection-delay bound 1024 decisions under the declared minimum shift. | Hidden-regime stability is an extension with a concrete deployable detector probability model; live scheduler integration remains separate. | Average-regime stability is part of the main theorem or already deployed online detection. |

## Required Paper Wording

Use:

```text
SOTA-inspired policy-semantics replay baselines reproduce representative policy
semantics on the same measured Scheduleurm service cache.
```

Do not use:

```text
Scheduleurm directly outperforms Gavel, Pollux, Sia, IADeep, or Salus.
```

Use:

```text
The q00 and q10 results are declared local-bucket certificates.
```

Do not use:

```text
The q00 and q10 results generalize to all CPU/data-loader workloads.
```

Use:

```text
The live-state dry-run oracle bridge certifies candidate-family trace semantics
on current probed resources; future online oracle claims require rerunning the
trace/enrichment/audit pipeline on the claimed trace population.
```

Use:

```text
The launched live theorem-dispatch gate validates bounded q01/q11
ScheduleurmBench dispatch with real task launch, theorem slots, and completion
bridge; its candidate family is intentionally bounded to avoid interfering with
existing jobs.
```

Do not use:

```text
The scheduler is now permanently theorem-grade for every future dispatch, or the
dry-run trace is a natural dispatched-job completion trace, or the bounded
ScheduleurmBench trace is a production-wide online trace.
```

Use:

```text
The live scheduler currently exposes an optional placement-scoring hook; theorem
evidence comes from replay/oracle rows whose queue_vector, lower_service,
penalty_units, selected action, and score_semantics are explicitly audited.
```

Do not use:

```text
The deployed live scheduler is already the global robust MaxWeight algorithm
proved in the paper.
```

## OR Reviewer Upgrade Gate

The 2026-06-11 OR reviewer gate is closed for the current finite-slice case
study:

```text
online: md/or_gate_online_arrivals.md PASS
holdout/lower-service: md/or_gate_holdout_calibration.md PASS
ablation: md/or_gate_ablation_suite.md PASS
live-state dry-run trace: md/or_gate_live_trace.md PASS
natural theorem trace: md/natural_live_theorem_trace.md PASS
realization boundary: md/natural_live_theorem_realization_bridge.md PASS, with no completion claim
launched live theorem dispatch: md/or_gate_live_theorem_dispatch.md PASS, bounded q01/q11 ScheduleurmBench completion claim
admission population: md/admission_population_gate.md PASS
direct SOTA scaffold: md/direct_sota_baseline_scaffold.md PASS, full-stack not ready
direct SOTA binary smoke round2: md/direct_sota_baseline_binary_smoke_20260611_round2.md PASS for scaffold/fallback; Pollux and Decima entrypoints smoke-pass, full-stack not ready
Gavel direct native smoke: md/gavel_direct_native_smoke_20260612.md PASS for isolated dependency/stub/help/generated-jobs, native Gavel trace, and Scheduleurm q01 native trace-seed smoke; service-unit equivalence still blocks direct performance baseline claim
direct SOTA full-stack readiness: md/direct_sota_fullstack_readiness_20260612.md PASS for read-only readiness; same-workload adapter seeds exist, direct full-stack count remains 0
SOTA claim matrix: md/sota_fullstack_claim_matrix_20260612.md PASS for seed readiness across Gavel/Pollux/IADeep/Salus/Decima; direct full-stack count remains 0
global fabric cover: md/global_fabric_cover_calibration.md PASS for exact measured slices
global fabric cover curve: md/global_fabric_cover_calibration_cover_curve_20260611.md PASS for finite-feature k-center Lrho diagnostics
fabric-cover metric contract: md/fabric_cover_contract_certificate_20260612.md PASS for exact measured finite population and compressed-cover Lrho table; future/all-state ready=false
active-bucket / hidden-regime certificate: md/active_bucket_hidden_regime_certificate_20260611.md PASS for deterministic event-level accounting, main learning claim not ready
adaptive sampler / detector probability certificate: md/adaptive_sampler_detector_certificate_20260612.md PASS for deployable model, not live integration
adaptive live integration probe: md/adaptive_live_integration_probe_20260612.md PASS for optional adaptive_theorem_maxweight_v1 live-state trace fields; default scheduler unchanged
q00/q10 generalization gate: md/q00_q10_generalization_gate_20260612.md PASS for declared local buckets plus remote evidence inventory; broad all-CPU/data-loader ready=false
Gavel native performance microbaseline: md/gavel_native_performance_microbaseline_20260612.md PASS for bounded q01/q11 native Gavel simulator same-trace windows; service-unit equivalence ready=false and direct full-stack ready=false
future-admitted fabric cover: md/future_admitted_fabric_cover_gate_20260612.md PASS for exact measured positive-profile admission with identity projection rho=0; arbitrary all-state ready=false
future production admission contract: md/future_production_admission_contract_20260612.md PASS; active production rows route through ADMIT_THEOREM_TRACE or PROBE_REQUIRED, preventing unmeasured future jobs from silently entering theorem-facing claims
q00/q10 broad measured envelope: md/q00_q10_broad_envelope_gate_20260612.md PASS for admitted measured q00/q10 service-cache rows; all CPU/data-loader world ready=false
production-wide live theorem trace gate: md/production_live_theorem_trace_gate_20260612.md PASS for an earlier one-task queued-production theorem trace snapshot; no launch/completion claim
production shadow theorem trace: md/production_shadow_theorem_trace_20260612.md PASS for non-invasive active-production GPU theorem subset, not launched production dispatch
SOTA full-stack superiority gate: md/sota_fullstack_superiority_gate_20260612.md PASS as a hard-blocker certificate; direct_fullstack_sota_superiority_ready=false, full_stack_ready_count=0
all-state conservative fabric-cover gate: md/all_state_conservative_cover_gate_20260612.md PASS for all-state safety via measured-admitted or zero-service probe/defer partition; positive_service_all_state_cover_ready=false
production launch/completion gate: md/production_launch_completion_gate_20260612.md PASS as a safe launch gate; rolling snapshot reports active-production progress and high-utilization/no-safe-launch conditions, with large_scale_active_progress_ready=true and large_scale_launched_completion_ready=false
Gavel service-unit equivalence certificate: md/gavel_service_unit_equivalence_certificate_20260612.md PASS for trace/schema compatibility and bounded microbaseline readiness; gavel_service_unit_equivalence_ready=false
declared finite-domain positive cover: md/declared_finite_domain_positive_cover_gate_20260612.md PASS; 193 declared service-cache buckets, 187 positive, 6 boundary, 0 uncovered; positive_service_all_state_cover_ready=false
controlled production completion gate: md/controlled_production_completion_gate_20260612.md PASS for bounded controlled completion and canary contract; controlled_32_task_completion_ready=false and large_scale_organic_launched_completion_ready=false
reviewer supplement: md/or_gate_reviewer_supplement.md PASS
all-gate summary: md/experiment_artifacts/or_submission_closure_2026_06_12_gap_closure.md PASS
```

Before making a stronger online-production OR claim, the following still need
additional evidence:

```text
production-wide live dispatch trace beyond the bounded ScheduleurmBench q01/q11
  launched trace, when queued production jobs and safe low-interference resources
  exist at the same time;
direct external scheduler binary runs if making full-stack SOTA superiority claims;
live A/B integration evidence before promoting active-bucket learning or
hidden-regime detection into deployed-scheduler empirical claims.
```

## General Fabric Calibration Gate

A broad fabric-cover claim is allowed only after a table provides:

```text
feature map Phi(a)
feature weights w_r
metric d_Phi(a,a')
candidate projection pi(a)
cover population: fixed / statewise / regimewise / sampled / all feasible
rho = max_a d_Phi(a, pi(a))
service sensitivity samples
L = certified worst-case or confidence-envelope slope
Lrho = L * rho
failure probability, if the envelope is statistical
```

Until then, paper claims should distinguish exact measured finite-slice
certificates from generalized fabric-cover calibration.
