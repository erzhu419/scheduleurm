# OR Gate: Online Arrival Experiments

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| scenario_count | 120 |
| candidate_completes_all | true |
| candidate_not_pareto_dominated_all | true |
| candidate_vs_legacy_geomean_makespan | 1.64296 |
| candidate_vs_legacy_geomean_mean_flow | 4.63984 |
| worst_candidate_vs_legacy_makespan | 0.999688 |
| worst_candidate_vs_legacy_mean_flow | 1.03136 |
| max_candidate_backlog_jobs | 114 |

## Scenario Rows

| taskset | trace | jobs | candidate profiles | cand/legacy makespan | cand/legacy flow | SOTA nondominated | mean backlog |
|---|---|---:|---|---:|---:|---:|---:|
| q00_light_control | q00_light_control_poisson_load0.50_seed41 | 512 | `{"light_control_local": 13}` | 5.70935 | 204.217 | true | 5.93397 |
| q00_light_control | q00_light_control_poisson_load0.50_seed42 | 512 | `{"light_control_local": 13}` | 5.92662 | 206.134 | true | 6.19855 |
| q00_light_control | q00_light_control_poisson_load0.50_seed43 | 512 | `{"light_control_local": 13}` | 6.15281 | 204.521 | true | 6.48658 |
| q00_light_control | q00_light_control_poisson_load0.70_seed41 | 512 | `{"light_control_local": 13}` | 7.94401 | 204.53 | true | 8.73558 |
| q00_light_control | q00_light_control_poisson_load0.70_seed42 | 512 | `{"light_control_local": 13}` | 8.21335 | 205.203 | true | 9.12691 |
| q00_light_control | q00_light_control_poisson_load0.70_seed43 | 512 | `{"light_control_local": 13}` | 8.57361 | 200.207 | true | 9.74016 |
| q00_light_control | q00_light_control_poisson_load0.85_seed41 | 512 | `{"light_control_local": 13}` | 9.60683 | 197.692 | true | 11.2008 |
| q00_light_control | q00_light_control_poisson_load0.85_seed42 | 512 | `{"light_control_local": 13}` | 9.86839 | 195.145 | true | 11.8086 |
| q00_light_control | q00_light_control_poisson_load0.85_seed43 | 512 | `{"light_control_local": 13}` | 10.3587 | 177.576 | true | 13.5724 |
| q00_light_control | q00_light_control_poisson_load0.95_seed41 | 512 | `{"light_control_local": 13}` | 10.7018 | 178.698 | true | 13.968 |
| q00_light_control | q00_light_control_poisson_load0.95_seed42 | 512 | `{"light_control_local": 13}` | 10.8838 | 175.15 | true | 14.678 |
| q00_light_control | q00_light_control_poisson_load0.95_seed43 | 512 | `{"light_control_local": 13}` | 11.3613 | 125.782 | true | 21.2475 |
| q00_light_control | q00_light_control_bursty_load0.50_seed41 | 512 | `{"light_control_local": 13}` | 6.47668 | 172.05 | true | 8.23003 |
| q00_light_control | q00_light_control_bursty_load0.50_seed42 | 512 | `{"light_control_local": 13}` | 6.87604 | 133.033 | true | 11.3795 |
| q00_light_control | q00_light_control_bursty_load0.50_seed43 | 512 | `{"light_control_local": 13}` | 5.83985 | 182.326 | true | 6.94161 |
| q00_light_control | q00_light_control_bursty_load0.70_seed41 | 512 | `{"light_control_local": 13}` | 8.99045 | 152.393 | true | 13.5548 |
| q00_light_control | q00_light_control_bursty_load0.70_seed42 | 512 | `{"light_control_local": 13}` | 9.54717 | 99.3408 | true | 22.0889 |
| q00_light_control | q00_light_control_bursty_load0.70_seed43 | 512 | `{"light_control_local": 13}` | 8.1271 | 172.277 | true | 10.7407 |
| q00_light_control | q00_light_control_bursty_load0.85_seed41 | 512 | `{"light_control_local": 13}` | 10.5609 | 119.687 | true | 20.707 |
| q00_light_control | q00_light_control_bursty_load0.85_seed42 | 512 | `{"light_control_local": 13}` | 11.1859 | 75.3443 | true | 34.7567 |
| q00_light_control | q00_light_control_bursty_load0.85_seed43 | 512 | `{"light_control_local": 13}` | 9.82507 | 151.582 | true | 15.0706 |
| q00_light_control | q00_light_control_bursty_load0.95_seed41 | 512 | `{"light_control_local": 13}` | 11.3504 | 97.8704 | true | 27.4957 |
| q00_light_control | q00_light_control_bursty_load0.95_seed42 | 512 | `{"light_control_local": 13}` | 11.8114 | 58.1906 | true | 47.9447 |
| q00_light_control | q00_light_control_bursty_load0.95_seed43 | 512 | `{"light_control_local": 13}` | 10.9493 | 113.873 | true | 22.5851 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.50_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1 | 1.07911 | true | 1.11918 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.50_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00513 | 1.31858 | true | 1.54966 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.50_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1 | 1.09698 | true | 1.16431 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.70_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00614 | 1.25768 | true | 1.74024 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.70_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00667 | 1.45789 | true | 2.60667 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.70_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00171 | 1.1908 | true | 1.73186 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00792 | 1.52214 | true | 2.42344 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00768 | 1.55021 | true | 4.10831 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00451 | 1.62501 | true | 2.44905 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.95_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.028 | 1.68175 | true | 3.17457 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.95_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.06206 | 1.60044 | true | 5.76837 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.95_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.04084 | 1.82791 | true | 3.96261 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.50_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00476 | 1.38733 | true | 3.80292 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.50_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00644 | 1.33154 | true | 6.84042 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.50_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00551 | 1.36255 | true | 4.47123 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.70_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00651 | 1.3451 | true | 6.03851 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.70_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.05174 | 1.28902 | true | 10.2797 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.70_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.00728 | 1.38659 | true | 7.90734 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.02771 | 1.41003 | true | 7.91599 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.05332 | 1.25708 | true | 11.9698 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.0634 | 1.33315 | true | 11.0474 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.95_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.06405 | 1.37493 | true | 9.7353 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.95_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.05442 | 1.25751 | true | 13.0174 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.95_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.06338 | 1.2942 | true | 12.5125 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.50_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1 | 1.03136 | true | 3.85328 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.50_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1 | 1.07342 | true | 3.91889 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.50_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.00006 | 1.16977 | true | 4.06374 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.70_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.00022 | 1.59565 | true | 5.38875 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.70_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.01547 | 2.74434 | true | 5.58347 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.70_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.07826 | 2.28056 | true | 6.04128 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.2055 | 5.68334 | true | 6.77271 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.22967 | 6.12577 | true | 7.17692 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.27423 | 5.15098 | true | 8.10833 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.95_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.3493 | 6.91916 | true | 8.29577 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.95_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.37055 | 6.93737 | true | 9.09948 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.95_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.39687 | 5.63269 | true | 10.7057 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.50_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.01365 | 1.57761 | true | 5.69532 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.50_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.01284 | 2.64602 | true | 9.61839 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.50_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 0.999688 | 1.55855 | true | 6.06442 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.70_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.1725 | 3.41684 | true | 9.1599 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.70_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.35879 | 3.38092 | true | 19.837 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.70_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.24247 | 5.06851 | true | 9.91924 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.3883 | 4.63827 | true | 13.8126 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.47353 | 2.96797 | true | 31.2741 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.49073 | 4.80561 | true | 17.548 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.95_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.46414 | 4.23382 | true | 19.0599 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.95_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.48609 | 2.63306 | true | 39.5072 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.95_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.51183 | 3.45295 | true | 28.2463 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1 | 1.16044 | true | 0.933599 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.00047 | 1.26891 | true | 1.086 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.00096 | 1.35271 | true | 1.01247 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1 | 1.30549 | true | 1.59735 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.00256 | 1.47112 | true | 1.94944 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.01939 | 1.42263 | true | 2.17097 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.01117 | 1.7433 | true | 2.45171 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.00369 | 1.96683 | true | 2.97455 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.02497 | 1.60321 | true | 3.21284 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.07116 | 3.32557 | true | 3.44772 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.08665 | 3.47968 | true | 4.23522 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.02833 | 1.92838 | true | 4.15615 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.00445 | 1.4262 | true | 3.88471 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.03652 | 1.49864 | true | 10.854 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.00828 | 1.39368 | true | 4.31922 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.03955 | 1.44116 | true | 6.65907 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.09097 | 1.36297 | true | 19.7689 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.01112 | 1.72695 | true | 7.31408 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.04615 | 1.67758 | true | 8.90457 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.1304 | 1.44606 | true | 24.8393 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.13321 | 1.96401 | true | 14.2167 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.10807 | 1.98459 | true | 11.648 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.15374 | 1.47687 | true | 29.1045 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.15436 | 1.67131 | true | 21.276 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1 | 1.12489 | true | 1.63907 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1 | 1.21391 | true | 1.83322 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.00192 | 1.12764 | true | 1.50664 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1 | 2.03025 | true | 2.60816 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1 | 1.78443 | true | 3.0678 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.00353 | 2.26759 | true | 2.35215 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1 | 3.19728 | true | 3.74629 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1 | 3.07656 | true | 4.47347 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.00447 | 3.55723 | true | 3.25767 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.07119 | 3.99518 | true | 5.32448 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.09731 | 3.61283 | true | 6.59391 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.004 | 4.00219 | true | 4.45389 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.00529 | 1.89735 | true | 4.48966 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.05944 | 1.37533 | true | 11.632 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.006 | 1.48978 | true | 6.14035 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.01478 | 2.14831 | true | 8.03167 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.07273 | 1.38517 | true | 17.3955 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.03238 | 1.86532 | true | 10.9939 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.01577 | 2.21523 | true | 12.0753 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.12873 | 1.53605 | true | 21.2862 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.15563 | 2.04079 | true | 18.4536 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed41 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.0889 | 2.18477 | true | 16.5732 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed42 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.15253 | 1.63139 | true | 24.8892 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed43 | 400 | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 1, "hybrid_rl_resac_ant": 2}` | 1.15566 | 1.88704 | true | 23.7172 |

## Scope

rate-controlled replay on the same measured service cache; this is an online-arrival stress certificate, not a direct execution of external scheduler binaries
