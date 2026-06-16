# Stable-Rate Cache Refresh, 2026-06-13

This note records the artifacts refreshed after `build_default_cache()` began
prioritizing task-native `ScheduleurmStableRate` measurements for node007/CNN,
LLM, and RL rows.

## Refreshed Artifacts

| Artifact | Status | Main update |
|---|---:|---|
| `md/experiment_artifacts/sota_candidate_union_gate_20260613.json` | PASS | 16 measured-cache SOTA action-union scenarios; fixed Pareto-slack union closes both SOTA-style envelopes within tolerance. |
| `md/experiment_artifacts/sota_algorithm_upgrade_gate_20260613.json` | PASS | Includes node007 native CNN/LLM/ETA portfolio rows without regression. |
| `md/figures/module27_sota_pareto_data.json` | PASS | Pareto figure data regenerated from the stable-rate cache. |
| `md/experiment_artifacts/trace_benchmark_matrix_static_20260613.json` | PASS | Static five-taskset matrix regenerated. |
| `md/experiment_artifacts/trace_benchmark_matrix_poisson_20260613.json` | PASS | Poisson five-taskset matrix regenerated. |
| `md/experiment_artifacts/or_gate_online_arrivals.json` | PASS | 192 online scenarios now use `scheduleurm_sota_union_pareto_slack` as the theorem-facing candidate. |
| `md/experiment_artifacts/or_gate_holdout_calibration.json` | PASS | Holdout selected profiles aligned to the same theorem-facing candidate. |
| `md/experiment_artifacts/or_gate_ablation_suite.json` | PASS | Ablation suite regenerated. |
| `md/experiment_artifacts/online_ablation_summary_ci_20260613.json` | PASS | Distributional summary regenerated from the refreshed online/ablation gates. |
| `md/experiment_artifacts/selected_profile_holdout_lcb_gate_20260613.json` | PASS-LOWER-SERVICE | LCB lower-service capacity slack is `0.0247121365`; absolute and diagonal mean-service eta remain negative. |
| `md/experiment_artifacts/module48_portfolio_slack_certificate.json` | PASS | Finite-slice slack certificate now uses `scheduleurm_sota_union_pareto_slack`; `delta=eta=0.0787777101`. |

## Manuscript Numbers

The paper now uses these refreshed values:

- Online replay: geomean candidate/legacy makespan `1.670x`, mean flow
  `4.998x`, 192/192 completed, 0 Pareto-dominated scenarios.
- Online summary: 3 single-metric losses beyond 0.5% tolerance, all in one q01
  LLM bursty high-load makespan row; none is a Pareto-dominance violation.
- Ablation: full candidate vs legacy geomean makespan `1.807x`, mean flow
  `3.398x`, and no ablated policy Pareto-dominates the full candidate.
- Module48 finite slice: selected profiles
  `{cpu=8, cnn=3, jax=1, llm=10, rl=2}`, `5832` measured actions,
  `B=34499765.4854296`, `N=437938174`.
- Selected-profile stochastic LCB: `delta_LCB=0.0247121365`, coordinate LCB
  ratios `{cpu=0.3560, cnn=0.5843, jax=0.7881, llm=0.8769, rl=0.3137}`.

## Boundary

These are measured-cache replay and finite-slice theorem-condition artifacts.
They do not convert policy-semantics SOTA comparisons into direct external
full-stack SOTA superiority, and they do not certify arbitrary future
unmeasured states.
