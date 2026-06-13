# Math-Code Alignment Check

Date: 2026-06-11, Asia/Shanghai.

This note records the current alignment between `md/math.md`, the Lean proof
spine, and the Scheduleurm replay / oracle code after the robust feasible-family
tightening.

## Core theorem route

| Math object | Current code / artifact | Alignment status |
|---|---|---|
| Statewise finite feasible family \(\mathcal A_s^{cand}\) | `simulation/service_cache.py`, `simulation/tasksets.py` | `ServiceRateCache.profiles()` excludes every profile at or above the first measured capacity boundary. |
| Conservative lower-service map \(\underline\mu\) | `simulation/service_cache.py` | Repeated valid measurements are quality-ranked by measurement completeness, then equal-quality rows keep the more conservative aggregate service; capacity-boundary rows are audit evidence, not feasible actions. |
| Guarded approximate oracle | `simulation/fast_forward.py`, `simulation/defaults.py` | Candidate policy uses support/makespan guard plus mean-flow tie-break, matching \(\alpha_0+\alpha_1\|Q\|_1\) approximate-oracle slack. |
| Statewise approximate-oracle theorem | `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | Use `main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle`. |
| Slack accounting \(\delta>L\rho+\epsilon_{est}+\beta+\alpha_1\) | `algorithm/experiments/slack_accounting.py` | Code reports the same inequality and keeps additive constants `B + P0 + alpha0` separate. |
| Robust lower-service oracle trace | `algorithm/experiments/oracle_trace_enrichment.py`, `algorithm/experiments/theorem_oracle_trace_bridge.py` | Trace bridge requires `robust_maxweight_lower_service`, queue vector, lower service for every candidate, and exactly one selected action. |
| Service-map production oracle bridge | `algorithm/experiments/production_theorem_oracle_trace.py` | Module100 is a service-map theorem oracle certificate, not a live-dispatch universal claim. |
| Live scheduler oracle closure | `algorithm/experiments/live_scheduler_oracle_closure.py` | Module101 is per emitted trace; it does not certify future dispatches automatically. |
| Active-production shadow trace | `algorithm/experiments/production_shadow_theorem_trace.py` | Reads current production queue/node state and evaluates the optional theorem hook without queue mutation or launch; closes only the theorem-subset rows it emits. |
| Queued-production theorem trace | `algorithm/experiments/production_queued_theorem_trace.py` | Reads current queued production tasks and emits a read-only theorem trace using live node probes; no launch or queue mutation. |
| Active-bucket sampler / detector model | `algorithm/adaptive_learning.py`, `algorithm/experiments/adaptive_sampler_detector_certificate.py` | Provides concrete deterministic round-robin forced exploration and bounded two-window detector probability certificate; not enabled in the live scheduler by default. |
| Optional adaptive live hook | `algorithm/adaptive_policy.py`, `algorithm/experiments/adaptive_live_integration_probe.py` | Registers opt-in `adaptive_theorem_maxweight_v1`; live-state probe verifies sampler/detector audit fields while the default scheduler remains unchanged. |
| Direct SOTA workload adapters | `algorithm/experiments/sota_adapters/`, `algorithm/experiments/gavel_direct_native_smoke.py`, `algorithm/experiments/gavel_native_performance_microbaseline.py`, `algorithm/experiments/direct_sota_fullstack_readiness.py` | Same-workload seed artifacts now exist.  Gavel isolated dependency/stub/help/generated-jobs/native-trace smoke and bounded q01/q11 native simulator microbaseline pass; Pollux optimizer-policy tests pass 9/9 under pinned compatibility paths; Decima's simulator entrypoint runs; the official Sia artifact is cloned but its simulator needs the official cvxpy CBC/GLPK + pymoo environment and its physical path needs AdaptDL/Kubernetes.  Measured service-unit equivalence and external full-stack superiority remain open because no same-workload external system runs end-to-end in comparable service units. |
| Gavel service-unit certificate | `algorithm/experiments/gavel_service_unit_equivalence_certificate.py` | Closes q01/q11 Gavel trace/schema compatibility, exact arrival preservation, throughput seed readiness, and bounded native microbaseline readiness while keeping `gavel_service_unit_equivalence_ready=false`; prevents native simulator rows from being misread as full-stack superiority. |
| Gavel service-unit calibration gate | `algorithm/experiments/gavel_service_unit_calibration_gate.py` | Separates scalar service-unit equivalence from a profile-aware same-workload native Gavel simulator calibration.  The paired q01/q11 holdout gate now passes the scoped profile-aware model with p95 relative error 0 while keeping scalar service-unit equivalence false because the single-scale p95 errors are about 0.5. |
| Gavel resident-delay/JCT holdout gate | `algorithm/experiments/gavel_resident_delay_jct_holdout_gate.py` | Closes a scoped measured-service finish-time/JCT holdout for the controlled co-location rows.  Immediate co-location beats defer on 3/3 rows even when defer uses the fastest observed resident-alone rate; direct external full-stack superiority remains false. |
| Direct full-stack SOTA superiority gate | `algorithm/experiments/sota_fullstack_superiority_gate.py` | Converts full-stack superiority into an executable reviewer gate.  Current output is a hard-blocker certificate: policy-semantics comparison is ready, but direct full-stack superiority is false because full-stack-ready count is 0. |
| SOTA candidate-action union gate | `simulation/sota_baselines.py`, `algorithm/experiments/sota_candidate_union_gate.py` | Promotes SOTA-inspired policy families from external replay baselines into the measured-cache finite candidate action family.  Metric-specific Scheduleurm+SOTA union selectors beat the SOTA makespan and mean-flow envelopes on every replay scenario within 0.5% tolerance; the guarded union is not Pareto-dominated, but it is not a universal two-metric dominator.  This is a candidate-set/action-family result, not direct external binary execution. |
| SOTA-facing algorithm upgrade gate | `simulation/sota_baselines.py`, `algorithm/theorem_dispatch/global_dispatch.py`, `algorithm/theorem_dispatch/state_service.py`, `algorithm/theorem_dispatch/eta_lcb.py`, `algorithm/experiments/sota_algorithm_upgrade_gate.py` | Closes five opt-in algorithm axes without changing the production default: adaptive scalarized Scheduleurm+SOTA single policy is not Pareto-dominated; state-dependent marginal cache exposes empty/high-VRAM/CPU-resident service rows; bounded global batch lookahead chooses short-ETA tied actions without oracle slack; reusable ETA/LCB avoids repeated probes for identical state keys; expanded SOTA families include finish-time fairness, resource-adaptive goodput, and packing/interference guards. |
| SOTA native execution ledger | `algorithm/experiments/sota_native_execution_attempts.py` | Records current-host native attempts: Gavel native simulator rows, Pollux/AdaptDL optimizer-policy pytest 9/9, Sia official artifact probe, Decima simulator entrypoint help, and Salus runtime blocker.  Native execution ready count is 3 and direct full-stack ready count is 0, keeping the full-stack superiority claim blocked. |
| Fabric-cover metric contract | `algorithm/experiments/fabric_cover_contract_certificate.py` | Reviewer-facing feature map, numeric scales, projection population, exact rho, compressed-cover Lrho, and exclusions are explicit for the measured finite population. |
| Future-admitted fabric cover | `algorithm/experiments/future_admitted_fabric_cover_gate.py` | Future admitted measured states use exact positive service-cache profiles and identity projection with rho=0; arbitrary all-state fabric cover remains false. |
| Declared finite-domain positive cover | `algorithm/experiments/declared_finite_domain_positive_cover_gate.py` | Defines the positive theorem population as the declared measured service-cache universe: 193 buckets with classification fraction 1.0, 187/193 positive lower-service rows, 6/193 capacity boundaries, and 0 uncovered buckets. |
| All-state conservative fabric cover | `algorithm/experiments/all_state_conservative_cover_gate.py` | Covers every scheduler-visible state by measured-admitted positive service or unknown zero-service probe/defer.  This aligns with safety/admission semantics, not positive-service all-state stability. |
| Future production admission contract | `algorithm/experiments/future_production_admission_contract.py` | Future production tasks route through strict theorem admission to theorem trace only with a service certificate; otherwise they are probe-required and excluded from theorem-facing claims. |
| Production launch/completion gate | `algorithm/experiments/production_launch_completion_gate.py` | Enforces the launch-side safety contract before any production launch.  Rolling snapshots close active-production progress/shadow-trace evidence only when a nonempty theorem shadow subset is present; otherwise the gate records pending evidence and still refuses launch when queue/resource conditions are unsafe. |
| Controlled launched completion gate | `algorithm/experiments/controlled_production_completion_gate.py` | Recognizes bounded launched completion evidence, defines the 32-task controlled threshold and organic production threshold, and refuses additional launch unless resources are safe. |
| Organic production canary recorder gate | `algorithm/experiments/organic_production_canary_recorder_gate.py` | Verifies strict organic production admission/trace recorder readiness and keeps the launched-completion claim false until the natural-launch thresholds are met. |
| Online/ablation interval dashboard | `algorithm/experiments/online_ablation_summary_ci.py` | Converts existing replay artifacts into reviewer-facing geomean, median, worst, 5%/95%, dominated-scenario, and loss-count summaries; no new live execution is claimed. |
| Diagonal-scaled LCB theorem | `/home/erzhu419/mine_code/proof/Scheduleurm/DiagonalScaling.lean`, `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | Proves coordinate-scaled support-loss and L1-weighted LCB support bounds, then exposes `main_diagonal_scaled_lcb_support_loss` and `main_diagonal_scaled_lcb_support_loss_l1` for the manuscript.  This is the math bridge from heterogeneous service units to an LCB lower-service capacity certificate. |
| Selected-profile holdout LCB gate | `algorithm/experiments/selected_profile_holdout_lcb_gate.py` | Tracks selected aggregate-window stochastic lower-service targets.  After CNN/LLM/RL/CPU/control probes, all 11 targets are sample-ready with 921 supplemental rates.  Absolute mean-service eta remains negative (-738.9938) and diagonal mean-service eta remains negative (-0.5587), but the theorem-facing LCB lower-service capacity certificate passes with \(\delta_{\mathrm{LCB}}=0.020993\). |
| Global theorem dispatcher prototype | `algorithm/theorem_dispatch/global_dispatch.py`, `algorithm/experiments/global_theorem_dispatcher_prototype_gate.py` | Provides a pure bounded global robust-MaxWeight action selector with exact oracle gap over enumerated theorem rows; it is not wired as the live scheduler default. |
| Optional algorithm upgrade gate | `algorithm/theorem_dispatch/batch_policy.py`, `algorithm/theorem_dispatch/state_service.py`, `algorithm/theorem_dispatch/eta_lcb.py`, `algorithm/experiments/or_algorithm_upgrade_gate.py`, `skill/scheduler.py` | Closes an opt-in algorithm-layer gate for five axes: global batch candidate construction, state-dependent marginal service lookup, online LCB/ETA updates, backlog-aware guarded replay, and q01 CNN/LLM/co-location non-regression.  The replay policy uses a lower-service no-upshift statewise drain: it cannot raise the base action and only lowers a profile after service and waiting-backlog guards pass.  The scheduler integration is an optional `global_theorem_maxweight_v1` soft-hint A/B hook: it builds the batch action under `algorithm/` and still leaves final placement to legacy checks.  The gate passes with 0 regressions, 4 improving tasksets, tolerance-aware SOTA Pareto safety, and no change to the legacy scheduler default. |
| Gate status dashboard | `algorithm/experiments/gate_status_dashboard.py` | Generates the reviewer claim ladder with scoped claim, adjacent strong claim, blocker, next threshold, artifact path, and raw evidence path. |
| Reviewer environment manifest gate | `algorithm/experiments/reviewer_environment_manifest_gate.py`, `environment.yml`, `requirements-reviewer.txt`, `scripts/reproduce_or_submission.sh` | Provides a current-host reproduction contract and environment files while keeping clean Docker/Nix container readiness false. |
| q00/q10 scope gate | `algorithm/experiments/q00_q10_generalization_gate.py` | Declared local q00/q10 buckets close; remote CPU evidence is inventoried; broad all-CPU/data-loader generalization remains false. |
| q00/q10 broad measured envelope | `algorithm/experiments/q00_q10_broad_envelope_gate.py` | Admitted measured q00/q10 service-cache rows have positive lower-service rows; all possible CPU/data-loader programs remain out of scope. |
| Corner-case lower-service gate | `algorithm/experiments/corner_case_lower_service_gate.py` | Admits the stable node007/node001 corner-case rows into a standalone service-cache snapshot and proves positive row-level lower-service capacity slack under a 0.8 offered-load fraction.  It does not alter the default replay cache or claim production-wide stability. |

## Current empirical feasible slices

| Bucket | Feasible measured profiles | First boundary | Current candidate action |
|---|---:|---:|---:|
| q00 `light_control_local` | 1-13 | 14 | 13 |
| q01 `gpu_heavy_jax_matmul` | 1-8 | none in current slice | 1 replay / 4-8 support certificates |
| q10 `cpu_heavy_local_bench` | 1-9 | 10 | 8 |
| q11 `hybrid_rl_resac_ant` | 1-9 | 10 | 2 |

The profile-10 q11 replay point is historical only.  Fresh live sanity makes it
the first robust capacity boundary for the current node bucket, so profile 10
and above are excluded from the current theorem-facing candidate family.

## Reviewer-facing boundaries

Safe:

```text
exact measured finite service-action slices
+ robust candidate MaxWeight slack accounting
+ bounded second-moment stochastic model
-> finite-set Foster recurrence certificate
```

Unsafe:

```text
raw scheduler history is the theorem population
attempted production is the theorem population
q11 profile 10 is still a robust feasible action
q00/q10/q11 buckets generalize to every CPU/RL/BAPR deployment
SOTA-style replay is direct binary execution of Gavel/Pollux/Sia/IADeep
SOTA candidate-action union is direct binary/full-stack superiority
future scheduler dispatches are automatically theorem-grade without admission/probe
generalized positive-service L,rho fabric calibration is complete without a profiling table
all-state zero-service probe/defer safety is positive-service all-state stability
shadow production trace is launched production dispatch
active-production progress observations are a launched production completion trace
adaptive sampler/detector certificate means online learning is deployed
optional algorithm-upgrade replay gate means production default global batch dispatch is deployed
SOTA-facing algorithm upgrade gate means direct full-stack SOTA superiority
```

## Remaining calibration gate

The generalized fabric-cover theorem is proved in Lean, but a broad empirical
claim still needs a named feature map \(\Phi\), weights, projection \(\pi\),
cover population, \(\rho\), service sensitivity envelope \(L\), and the resulting
\(L\rho\) table.  Until that table exists, the strong empirical claim should use
exact measured finite-slice / service-map certificates.

The 2026-06-12 gap-closure additions improve three extension interfaces without
changing the default scheduler path:

```text
production shadow trace:
  closes current active-production GPU theorem-subset semantics without launch;
  does not replace the WAIT production-wide live dispatch gate.

direct SOTA adapters:
  provide Scheduleurm workload seeds for the inspected external repositories;
  Gavel isolated native generated-jobs/native-trace/Scheduleurm trace-seed smoke and bounded q01/q11 native simulator microbaseline pass, but measured service-unit equivalence is still not a same-workload performance baseline;
  Pollux optimizer-policy pytest passes 9/9 under pinned local compatibility paths, the official Sia artifact is cloned but simulator/solver environment remains blocked, and Decima's simulator entrypoint runs;
  Gavel service-unit certificate closes trace/schema compatibility while keeping service-unit equivalence false;
  Gavel service-unit calibration gate passes a profile-aware same-workload native Gavel simulator model while keeping scalar service-unit equivalence false;
  SOTA candidate-action union closes the measured-cache policy-family envelope claim, but not direct full-stack superiority;
  SOTA-facing algorithm upgrade closes adaptive scalarized union / state-cache / lookahead / ETA-reuse / SOTA-family expansion as opt-in replay-certificate evidence;
  do not make Gavel/Pollux/IADeep/Salus/Decima direct full-stack baselines;
  strict full-stack superiority gate reports full_stack_ready_count=0 and records hard blockers.

future admission contracts:
  future-admitted measured-state fabric cover uses identity projection with rho=0;
  future production tasks without service certificates are probe-required before theorem use.

all-state conservative cover:
  declared finite-domain positive cover closes the exact measured service-cache universe;
  all scheduler-visible states are covered safely by measured-admitted or zero-service probe/defer;
  positive-service all-state stability remains explicitly false.

production launch/completion gate:
  is a rolling safe-launch/completion gate; the current snapshot is pending when progress
  observations or queued theorem-admitted production jobs are absent;
  controlled completion gate closes 32-task launched completion on a theorem-admitted controlled run, while organic production-wide completion remains false.
  organic canary recorder verifies strict admission/trace readiness but does not claim organic completion.

q00/q10 broad measured envelope:
  closes admitted measured service-cache rows, not all CPU/data-loader workloads.

adaptive sampler/detector:
  gives a concrete deployable probability model for extension theorems;
  optional live-state integration probe emits sampler/detector fields;
  default scheduler policy remains unchanged.

reviewer environment:
  one-command reproduction, Make targets, environment.yml, and requirements-reviewer.txt are tracked;
  no clean Docker/Nix container claim is made.
```
