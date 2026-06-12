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
| Direct SOTA workload adapters | `algorithm/experiments/sota_adapters/`, `algorithm/experiments/gavel_direct_native_smoke.py`, `algorithm/experiments/gavel_native_performance_microbaseline.py`, `algorithm/experiments/direct_sota_fullstack_readiness.py` | Same-workload seed artifacts now exist.  Gavel isolated dependency/stub/help/generated-jobs/native-trace smoke and bounded q01/q11 native simulator microbaseline pass, but measured service-unit equivalence remains open; external full-stack superiority still requires native runner/stack validation. |
| Gavel service-unit certificate | `algorithm/experiments/gavel_service_unit_equivalence_certificate.py` | Closes q01/q11 Gavel trace/schema compatibility and throughput seed readiness while keeping `gavel_service_unit_equivalence_ready=false`; prevents native simulator rows from being misread as full-stack superiority. |
| Direct full-stack SOTA superiority gate | `algorithm/experiments/sota_fullstack_superiority_gate.py` | Converts full-stack superiority into an executable reviewer gate.  Current output is a hard-blocker certificate: policy-semantics comparison is ready, but direct full-stack superiority is false because full-stack-ready count is 0. |
| Fabric-cover metric contract | `algorithm/experiments/fabric_cover_contract_certificate.py` | Reviewer-facing feature map, numeric scales, projection population, exact rho, compressed-cover Lrho, and exclusions are explicit for the measured finite population. |
| Future-admitted fabric cover | `algorithm/experiments/future_admitted_fabric_cover_gate.py` | Future admitted measured states use exact positive service-cache profiles and identity projection with rho=0; arbitrary all-state fabric cover remains false. |
| Declared finite-domain positive cover | `algorithm/experiments/declared_finite_domain_positive_cover_gate.py` | Defines the positive theorem population as the declared measured service-cache universe: 193 buckets, 187 positive lower-service rows, 6 capacity boundaries, and 0 uncovered buckets. |
| All-state conservative fabric cover | `algorithm/experiments/all_state_conservative_cover_gate.py` | Covers every scheduler-visible state by measured-admitted positive service or unknown zero-service probe/defer.  This aligns with safety/admission semantics, not positive-service all-state stability. |
| Future production admission contract | `algorithm/experiments/future_production_admission_contract.py` | Future production tasks route to theorem trace only with a service certificate; otherwise they are probe-required and excluded from theorem-facing claims. |
| Production launch/completion gate | `algorithm/experiments/production_launch_completion_gate.py` | Enforces the launch-side safety contract before any production launch.  Rolling snapshots close active-production progress/shadow-trace evidence and refuse launch when queue/resource conditions are unsafe. |
| Controlled/canary production completion gate | `algorithm/experiments/controlled_production_completion_gate.py` | Recognizes bounded launched completion evidence, defines the 32-task controlled threshold and organic production threshold, and refuses additional launch unless resources are safe. |
| q00/q10 scope gate | `algorithm/experiments/q00_q10_generalization_gate.py` | Declared local q00/q10 buckets close; remote CPU evidence is inventoried; broad all-CPU/data-loader generalization remains false. |
| q00/q10 broad measured envelope | `algorithm/experiments/q00_q10_broad_envelope_gate.py` | Admitted measured q00/q10 service-cache rows have positive lower-service rows; all possible CPU/data-loader programs remain out of scope. |

## Current empirical feasible slices

| Bucket | Feasible measured profiles | First boundary | Current candidate action |
|---|---:|---:|---:|
| q00 `light_control_local` | 1-13 | 14 | 13 |
| q01 `gpu_heavy_jax_matmul` | 1-8 | none in current slice | 4 |
| q10 `cpu_heavy_local_bench` | 1-9 | 10 | 8 |
| q11 `hybrid_rl_resac_ant` | 1-9 | 10 | 2 standalone / 3 portfolio |

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
future scheduler dispatches are automatically theorem-grade without admission/probe
generalized positive-service L,rho fabric calibration is complete without a profiling table
all-state zero-service probe/defer safety is positive-service all-state stability
shadow production trace is launched production dispatch
active-production progress observations are a launched production completion trace
adaptive sampler/detector certificate means online learning is deployed
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
  Gavel service-unit certificate closes trace/schema compatibility while keeping service-unit equivalence false;
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
  closes large-scale active-production progress plus shadow theorem slots in the latest snapshot;
  refuses launched production work when queued jobs are absent or resources are unsafe;
  controlled completion gate recognizes bounded launched completion but keeps 32-task and organic completion claims false.

q00/q10 broad measured envelope:
  closes admitted measured service-cache rows, not all CPU/data-loader workloads.

adaptive sampler/detector:
  gives a concrete deployable probability model for extension theorems;
  optional live-state integration probe emits sampler/detector fields;
  default scheduler policy remains unchanged.
```
