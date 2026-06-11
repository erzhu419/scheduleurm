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
future scheduler dispatches are automatically theorem-grade
generalized L,rho fabric calibration is complete without a profiling table
```

## Remaining calibration gate

The generalized fabric-cover theorem is proved in Lean, but a broad empirical
claim still needs a named feature map \(\Phi\), weights, projection \(\pi\),
cover population, \(\rho\), service sensitivity envelope \(L\), and the resulting
\(L\rho\) table.  Until that table exists, the strong empirical claim should use
exact measured finite-slice / service-map certificates.
