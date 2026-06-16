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
| Direct SOTA workload adapters | `algorithm/experiments/sota_adapters/`, `algorithm/experiments/gavel_direct_native_smoke.py`, `algorithm/experiments/gavel_native_performance_microbaseline.py`, `algorithm/experiments/sota_fullstack_superiority_gate.py` | Same-workload seed artifacts now exist.  Gavel isolated dependency/stub/help/generated-jobs/native-trace smoke and bounded q01/q11 native simulator microbaseline pass; the separate Gavel physical scheduler/worker/RPC row closes a same-host probe; Pollux/AdaptDL, Sia, IADeep, and Salus have scoped same-host same-workload runtime-probe rows through their runtime paths; Decima's simulator entrypoint runs but remains a Spark-DAG simulator baseline.  Scalar Gavel service-unit equivalence remains open; direct full-stack SOTA superiority remains false. |
| Gavel service-unit certificate | `algorithm/experiments/gavel_service_unit_equivalence_certificate.py` | Closes q01/q11 Gavel trace/schema compatibility, exact arrival preservation, throughput seed readiness, and bounded native microbaseline readiness while keeping `gavel_service_unit_equivalence_ready=false`; prevents native simulator rows from being misread as full-stack superiority. |
| Gavel service-unit calibration gate | `algorithm/experiments/gavel_service_unit_calibration_gate.py` | Separates scalar service-unit equivalence from a profile-aware same-workload native Gavel simulator calibration.  The paired q01/q11 holdout gate now passes the scoped profile-aware model with p95 relative error 0 while keeping scalar service-unit equivalence false because the single-scale p95 errors are about 0.5. |
| Gavel resident-delay/JCT holdout gate | `algorithm/experiments/gavel_resident_delay_jct_holdout_gate.py` | Closes a scoped measured-service finish-time/JCT holdout for the controlled co-location rows.  Immediate co-location beats defer on 3/3 rows even when defer uses the fastest observed resident-alone rate; this remains a measured-service holdout, separate from the Gavel physical full-stack row. |
| Named external runtime-probe gate | `algorithm/experiments/sota_fullstack_superiority_gate.py` | Keeps the historical entrypoint name but emits the narrower claim `named_same_host_runtime_probe_ready=true` and `direct_fullstack_named_sota_superiority_ready=false` for Gavel, Pollux/AdaptDL, Sia, IADeep, and Salus on scoped same-host same-workload probes. |
| SOTA universe registry gate | `algorithm/experiments/sota_universe_registry_gate.py` | Separates the closed named-five runtime-probe claim from registered or arbitrary SOTA-universe language.  The current gate keeps `registered_sota_universe_superiority_ready=false` and `arbitrary_sota_superiority_ready=false` until comparable rows exist for every registered external system. |
| Registered SOTA runtime gate | `algorithm/experiments/registered_sota_runtime_gate.py` | Adds read-only runtime inventory for Tiresias, Shockwave, AlloX, Optimus, plus paper-only Themis/Gandiva rows.  It closes repo/paper discovery and records concrete dependency blockers while keeping same-workload full-stack readiness false. |
| Registered external-policy admitted-action universe gate | `algorithm/experiments/sota_admitted_universe_closure_gate.py`, `simulation/sota_baselines.py` | Maps all 11 non-adjacent registered external-policy systems into finite measured-cache policy-family actions and uses the closed measured-cache external-policy frontier to certify admitted-action policy-semantics dominance. It deliberately keeps `registered_sota_universe_superiority_ready=false` for external-binary/full-stack universe language. |
| Future workload protocol gate | `algorithm/experiments/future_workload_protocol_gate.py` | Aggregates strict production admission and future-admitted fabric cover, then checks ten synthetic future stress rows.  Measured q00/q01/q10/q11/CNN/LLM/CPU/control cases may admit or probe; unknown LLM/CPU/CUDA/Spark-DAG/NUMA cases route to `PROBE_REQUIRED`. |
| Multi-node original deployment gate | `algorithm/experiments/multinode_original_deployment_gate.py` | Keeps same-host runtime-probe evidence separate from original multi-node deployments.  Current read-only SSH inventory sees two reachable GPU nodes (`jtl110gpu`, `jtl110gpu2`) but does not promote a claim; multi-node superiority needs per-system original control-plane/worker runs. |
| Multi-node theorem shadow gate | `algorithm/experiments/multinode_theorem_shadow_gate.py`, `algorithm/theorem_dispatch/global_dispatch.py` | Builds a read-only cross-node theorem candidate family over visible nodes.  Current artifact sees `jtl110gpu` and `jtl110gpu2`, 18 certified candidate rows, 626 feasible configurations, and a selected robust-MaxWeight configuration spanning both GPU nodes with exact oracle gap zero.  It does not launch work or claim original multi-node full-stack superiority. |
| Decima Spark-DAG gate | `algorithm/experiments/decima_spark_dag_gate.py` | Audits the cloned Decima Spark-DAG simulator by file, import, baseline-entrypoint smoke, and a small dynamic-partition heuristic benchmark that completes.  It deliberately keeps Decima out of the GPU co-location full-stack gate until a Spark-DAG metric/performance bridge exists; learned-policy runtime remains blocked by TensorFlow-1. |
| Production-wide organic trace gate | `algorithm/experiments/production_wide_organic_trace_gate.py` | Wraps the queued live-trace gate, organic canary recorder, and launch/completion gate, separating recorder readiness from live trace closure and launched-completion evidence.  The strong production-wide organic claim remains false until natural launch/completion thresholds close. |
| Production organic readiness bridge gate | `algorithm/experiments/production_organic_readiness_bridge_gate.py`, `algorithm/experiments/production_queued_theorem_trace.py`, `algorithm/experiments/production_shadow_theorem_trace.py` | Closes the current production readiness bridge without launch: 3 queued BAPR-BUS rows are service-admitted and receive a read-only clean-bench theorem trace with 3 theorem slots, 6 candidates, and alpha0=alpha1=0; active-production shadow also closes.  This is queued/shadow theorem semantics, not launched organic completion. |
| Universal claim closure gate | `algorithm/experiments/universal_claim_closure_gate.py` | Aggregates the five broad-claim gates and reports 5/5 scoped boundaries ready with 0/5 universal strong claims ready.  This is the executable guardrail preventing named-system evidence from being stated as arbitrary SOTA/future/deployment/production coverage. |
| SOTA candidate-action union gate | `simulation/sota_baselines.py`, `algorithm/experiments/sota_candidate_union_gate.py` | Promotes SOTA-inspired policy families from external replay baselines into the measured-cache finite candidate action family.  Metric-specific Scheduleurm+SOTA union selectors beat the SOTA makespan and mean-flow envelopes on every replay scenario within 0.5% tolerance, and the fixed Pareto-slack online union policy is not Pareto-dominated and closes both SOTA-style envelopes on the same measured-cache replay.  This is a candidate-set/action-family result, not direct external binary execution. |
| SOTA strict bridge-action gate | `algorithm/experiments/sota_strict_dominance_frontier.py`, `algorithm/experiments/sota_bridge_action_gate.py`, `algorithm/experiments/remote_workload_selected_profile_probe.py` | Separates tolerance-level measured-cache dominance from strict ratio-\(\ge 1\) closure.  The frontier gate identifies the remaining q01 CNN static and hybrid portfolio rows.  The bridge-action gate searches finite phase-switch actions on the current measured cache, confirms those profiles still do not strictly close the frontier, audits target GPUs, and prepares real hybrid p2/p3 plus CNN tail-drain probes.  Current status is `SOTA_BRIDGE_WAIT_RESOURCE`: all target GPUs have scheduler blockers or high utilization, so no probe was launched. |
| SOTA-facing algorithm upgrade gate | `simulation/sota_baselines.py`, `algorithm/theorem_dispatch/global_dispatch.py`, `algorithm/theorem_dispatch/state_service.py`, `algorithm/theorem_dispatch/eta_lcb.py`, `algorithm/experiments/sota_algorithm_upgrade_gate.py` | Closes five opt-in algorithm axes without changing the production default: the fixed Pareto-slack Scheduleurm+SOTA policy dominates the SOTA-style measured-cache envelopes within tolerance; adaptive scalarization is retained as an ablation; state-dependent marginal cache exposes empty/high-VRAM/CPU-resident service rows; bounded global batch lookahead chooses short-ETA tied actions without oracle slack; reusable ETA/LCB avoids repeated probes for identical state keys; expanded SOTA families include finish-time fairness, resource-adaptive goodput, and packing/interference guards. |
| SOTA native execution ledger | `algorithm/experiments/sota_native_execution_attempts.py` | Historical current-host native attempt ledger remains useful for repo smoke provenance.  The current named runtime-probe evidence is reported by `sota_fullstack_superiority_gate.py`: named-system same-workload rows are closed for Gavel, Pollux/AdaptDL, Sia, IADeep, and Salus; Decima remains simulator-only. |
| Fabric-cover metric contract | `algorithm/experiments/fabric_cover_contract_certificate.py` | Reviewer-facing feature map, numeric scales, projection population, exact rho, compressed-cover Lrho, and exclusions are explicit for the measured finite population. |
| Future-admitted fabric cover | `algorithm/experiments/future_admitted_fabric_cover_gate.py` | Future admitted measured states use exact positive service-cache profiles and identity projection with rho=0; arbitrary all-state fabric cover remains false. |
| Declared finite-domain positive cover | `algorithm/experiments/declared_finite_domain_positive_cover_gate.py` | Defines the positive theorem population as the declared measured service-cache universe: 225 buckets with classification fraction 1.0, 215/225 positive lower-service rows, 10/225 capacity boundaries, and 0 uncovered buckets. |
| All-state conservative fabric cover | `algorithm/experiments/all_state_conservative_cover_gate.py` | Covers every scheduler-visible state by measured-admitted positive service or unknown zero-service probe/defer.  This aligns with safety/admission semantics, not positive-service all-state stability. |
| Future production admission contract | `algorithm/experiments/future_production_admission_contract.py` | Future production tasks route through strict theorem admission to theorem trace only with a service certificate; otherwise they are probe-required and excluded from theorem-facing claims. |
| Production launch/completion gate | `algorithm/experiments/production_launch_completion_gate.py` | Enforces the launch-side safety contract before any production launch.  Rolling snapshots close active-production progress/shadow-trace evidence only when a nonempty theorem shadow subset is present; otherwise the gate records pending evidence and still refuses launch when queue/resource conditions are unsafe. |
| Controlled launched completion gate | `algorithm/experiments/controlled_production_completion_gate.py` | Recognizes bounded launched completion evidence, defines the 32-task controlled threshold and organic production threshold, and refuses additional launch unless resources are safe. |
| Organic production canary recorder gate | `algorithm/experiments/organic_production_canary_recorder_gate.py` | Verifies strict organic production admission/trace recorder readiness and keeps the launched-completion claim false until the natural-launch thresholds are met. |
| Online/ablation interval dashboard | `algorithm/experiments/online_ablation_summary_ci.py` | Converts existing replay artifacts into reviewer-facing geomean, median, worst, 5%/95%, dominated-scenario, and loss-count summaries; no new live execution is claimed. |
| Diagonal-scaled LCB theorem | `/home/erzhu419/mine_code/proof/Scheduleurm/DiagonalScaling.lean`, `/home/erzhu419/mine_code/proof/Scheduleurm/MainTheorems.lean` | Proves coordinate-scaled support-loss and L1-weighted LCB support bounds, then exposes `main_diagonal_scaled_lcb_support_loss` and `main_diagonal_scaled_lcb_support_loss_l1` for the manuscript.  This is the math bridge from heterogeneous service units to an LCB lower-service capacity certificate. |
| Selected-profile holdout LCB gate | `algorithm/experiments/selected_profile_holdout_lcb_gate.py` | Tracks selected aggregate-window stochastic lower-service targets.  After CNN/LLM/RL/CPU/control probes, all 11 targets are sample-ready with 921 supplemental rates.  Absolute mean-service eta remains negative (-738.9820) and diagonal mean-service eta remains negative (-0.5587), but the theorem-facing LCB lower-service capacity certificate passes with \(\delta_{\mathrm{LCB}}=0.024712\). |
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
  SOTA candidate-action union closes the measured-cache policy-family envelope claim, while the separate named external runtime-probe gate closes scoped same-host same-workload runtime-probe rows;
  SOTA-facing algorithm upgrade closes adaptive scalarized union / state-cache / lookahead / ETA-reuse / SOTA-family expansion as opt-in replay-certificate evidence;
  do not make Decima or arbitrary future workloads direct GPU full-stack baselines;
  named external runtime-probe gate reports named_same_host_runtime_probe_ready=true and direct_fullstack_named_sota_superiority_ready=false for Gavel/Pollux/AdaptDL/Sia/IADeep/Salus measured probes.

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
