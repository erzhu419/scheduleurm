# OR Gate: Module Ablation Suite

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| trace_count | 15 |
| full_candidate_vs_legacy_geomean_makespan | 1.79612 |
| full_candidate_vs_legacy_geomean_mean_flow | 3.46525 |
| ablation_dominator_count | 0 |

## Reviewer Axis Coverage

| Axis | Policy | Role |
|---|---|---|
| legacy rules | `legacy_fixed_caps` | fixed legacy caps and legacy-style profile choices |
| sweetspot hook | `ablation_sweetspot_scalar_hook` | scalar calibrated sweetspot/guarded-knee replay without queue-adaptive MaxWeight |
| support scorer | `ablation_support_only_makespan` | support/makespan objective without delay or queue-adaptive penalty |
| delay tie-break | `ablation_delay_only_mean_flow` | delay/mean-flow objective without support-preserving robust guard |
| statewise guard | `ablation_statewise_interference_guard` | statewise co-location/interference guard without the full adaptive scorer |
| profile penalty | `ablation_robust_no_profile_penalty` | same support-envelope MaxWeight scorer with bounded profile penalty removed |
| tie-break direction | `ablation_robust_high_profile_tiebreak` | same robust scorer and penalty with high-profile tie-break instead of low-profile tie-break |
| full robust lower-service scorer | `calibrated_adaptive_maxweight_penalty` | queue-adaptive robust lower-service MaxWeight with bounded profile penalty |

## Trace Rows

| taskset | trace | jobs | profiles | cand/legacy makespan | cand/legacy flow | dominators |
|---|---|---:|---|---:|---:|---:|
| q00_light_control | q00_light_control_static_seed23 | 512 | `{"light_control_local": 13}` | 11.9264 | 11.7628 | 0 |
| q00_light_control | q00_light_control_poisson_load0.85_seed23 | 512 | `{"light_control_local": 13}` | 9.48756 | 195.98 | 0 |
| q00_light_control | q00_light_control_bursty_load0.85_seed23 | 512 | `{"light_control_local": 13}` | 10.8697 | 50.3105 | 0 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_static_seed23 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.06843 | 1.11379 | 0 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed23 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.05144 | 1.51372 | 0 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed23 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.01958 | 1.33877 | 0 |
| q10_cpu_host_bound | q10_cpu_host_bound_static_seed23 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.50809 | 1.53392 | 0 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed23 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.32768 | 6.61624 | 0 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed23 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.39365 | 2.59144 | 0 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_static_seed23 | 160 | `{"hybrid_rl_resac_ant": 3}` | 1.15967 | 1.17386 | 0 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed23 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.03037 | 2.0321 | 0 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed23 | 160 | `{"hybrid_rl_resac_ant": 3}` | 1.10183 | 1.36171 | 0 |
| hybrid_research_portfolio | hybrid_research_portfolio_static_seed23 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 3}` | 1.15756 | 1.26914 | 0 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed23 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.00145 | 3.14783 | 0 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed23 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.08867 | 1.39623 | 0 |

## Scope

same measured service cache, with each module removed by policy semantics rather than by editing the legacy scheduler
