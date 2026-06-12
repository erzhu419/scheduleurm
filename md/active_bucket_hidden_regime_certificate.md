# Active-Bucket / Hidden-Regime Certificate

## Summary

| Quantity | Value |
|---|---:|
| `scenario_count` | 120 |
| `active_bucket_count` | 64 |
| `event_level_union_bound_closed` | true |
| `deterministic_dwell_switching_event_closed` | true |
| `sampler_probability_model_certified` | false |
| `change_point_detector_tail_certified` | false |
| `main_claim_ready` | false |

## Scope

event-level active-bucket and hidden-regime certificate for measured Scheduleurm replay traces.  It supports extension text and theorem-input auditing.  This base gate alone is not a live adaptive-sampling probability theorem; concrete sampler/detector model evidence is reported by the separate adaptive_sampler_detector_certificate gate.

## Hidden-Regime Dwell/Switching

| Quantity | Value |
|---|---:|
| `regime_count` | 40 |
| `switching_count_if_concatenated` | 39 |
| `min_dwell_s` | 1674.105970557 |
| `total_dwell_s` | 2625611.886271578 |

## Active Buckets

| Bucket | Decisions | Allocated delta | Hoeffding radius |
|---|---:|---:|---:|
| `hybrid_research_portfolio|bursty_load0.50|load=0.50|cpu_heavy_local_bench|profile=8` | 1632 | 0.000781250 | 0.049034060 |
| `hybrid_research_portfolio|bursty_load0.50|load=0.50|gpu_heavy_jax_matmul|profile=1` | 166 | 0.000781250 | 0.153746021 |
| `hybrid_research_portfolio|bursty_load0.50|load=0.50|hybrid_rl_resac_ant|profile=2` | 736 | 0.000781250 | 0.073016165 |
| `hybrid_research_portfolio|bursty_load0.50|load=0.50|hybrid_rl_resac_ant|profile=3` | 26 | 0.000781250 | 0.388482317 |
| `hybrid_research_portfolio|bursty_load0.70|load=0.70|cpu_heavy_local_bench|profile=8` | 1672 | 0.000781250 | 0.048443977 |
| `hybrid_research_portfolio|bursty_load0.70|load=0.70|gpu_heavy_jax_matmul|profile=1` | 176 | 0.000781250 | 0.149314366 |
| `hybrid_research_portfolio|bursty_load0.70|load=0.70|hybrid_rl_resac_ant|profile=2` | 726 | 0.000781250 | 0.073517312 |
| `hybrid_research_portfolio|bursty_load0.70|load=0.70|hybrid_rl_resac_ant|profile=3` | 52 | 0.000781250 | 0.274698480 |
| `hybrid_research_portfolio|bursty_load0.85|load=0.85|cpu_heavy_local_bench|profile=8` | 1692 | 0.000781250 | 0.048156814 |
| `hybrid_research_portfolio|bursty_load0.85|load=0.85|gpu_heavy_jax_matmul|profile=1` | 182 | 0.000781250 | 0.146832514 |
| `hybrid_research_portfolio|bursty_load0.85|load=0.85|hybrid_rl_resac_ant|profile=2` | 708 | 0.000781250 | 0.074445988 |
| `hybrid_research_portfolio|bursty_load0.85|load=0.85|hybrid_rl_resac_ant|profile=3` | 82 | 0.000781250 | 0.218751481 |
| `hybrid_research_portfolio|bursty_load0.95|load=0.95|cpu_heavy_local_bench|profile=8` | 1696 | 0.000781250 | 0.048099992 |
| `hybrid_research_portfolio|bursty_load0.95|load=0.95|gpu_heavy_jax_matmul|profile=1` | 182 | 0.000781250 | 0.146832514 |
| `hybrid_research_portfolio|bursty_load0.95|load=0.95|hybrid_rl_resac_ant|profile=2` | 642 | 0.000781250 | 0.078179055 |
| `hybrid_research_portfolio|bursty_load0.95|load=0.95|hybrid_rl_resac_ant|profile=3` | 156 | 0.000781250 | 0.158597242 |
| `hybrid_research_portfolio|poisson_load0.50|load=0.50|cpu_heavy_local_bench|profile=8` | 3040 | 0.000781250 | 0.035927015 |
| `hybrid_research_portfolio|poisson_load0.50|load=0.50|gpu_heavy_jax_matmul|profile=1` | 184 | 0.000781250 | 0.146032331 |
| `hybrid_research_portfolio|poisson_load0.50|load=0.50|hybrid_rl_resac_ant|profile=2` | 1098 | 0.000781250 | 0.059780117 |
| `hybrid_research_portfolio|poisson_load0.70|load=0.70|cpu_heavy_local_bench|profile=8` | 3060 | 0.000781250 | 0.035809414 |
| `hybrid_research_portfolio|poisson_load0.70|load=0.70|gpu_heavy_jax_matmul|profile=1` | 206 | 0.000781250 | 0.138014374 |
| `hybrid_research_portfolio|poisson_load0.70|load=0.70|hybrid_rl_resac_ant|profile=2` | 1226 | 0.000781250 | 0.056573454 |
| `hybrid_research_portfolio|poisson_load0.85|load=0.85|cpu_heavy_local_bench|profile=8` | 3066 | 0.000781250 | 0.035774358 |
| `hybrid_research_portfolio|poisson_load0.85|load=0.85|gpu_heavy_jax_matmul|profile=1` | 222 | 0.000781250 | 0.132947888 |
| `hybrid_research_portfolio|poisson_load0.85|load=0.85|hybrid_rl_resac_ant|profile=2` | 1328 | 0.000781250 | 0.054357427 |
| `hybrid_research_portfolio|poisson_load0.95|load=0.95|cpu_heavy_local_bench|profile=8` | 3066 | 0.000781250 | 0.035774358 |
| `hybrid_research_portfolio|poisson_load0.95|load=0.95|gpu_heavy_jax_matmul|profile=1` | 230 | 0.000781250 | 0.130615288 |
| `hybrid_research_portfolio|poisson_load0.95|load=0.95|hybrid_rl_resac_ant|profile=2` | 1368 | 0.000781250 | 0.053556832 |
| `q00_light_control|bursty_load0.50|load=0.50|light_control_local|profile=13` | 3340 | 0.000781250 | 0.034275571 |
| `q00_light_control|bursty_load0.70|load=0.70|light_control_local|profile=13` | 3404 | 0.000781250 | 0.033951827 |
| `q00_light_control|bursty_load0.85|load=0.85|light_control_local|profile=13` | 3432 | 0.000781250 | 0.033813046 |
| `q00_light_control|bursty_load0.95|load=0.95|light_control_local|profile=13` | 3446 | 0.000781250 | 0.033744290 |
| `q00_light_control|poisson_load0.50|load=0.50|light_control_local|profile=13` | 6130 | 0.000781250 | 0.025300418 |
| `q00_light_control|poisson_load0.70|load=0.70|light_control_local|profile=13` | 6138 | 0.000781250 | 0.025283925 |
| `q00_light_control|poisson_load0.85|load=0.85|light_control_local|profile=13` | 6138 | 0.000781250 | 0.025283925 |
| `q00_light_control|poisson_load0.95|load=0.95|light_control_local|profile=13` | 6138 | 0.000781250 | 0.025283925 |
| `q01_gpu_bound_compute|bursty_load0.50|load=0.50|gpu_heavy_jax_matmul|profile=1` | 326 | 0.000781250 | 0.109710736 |
| `q01_gpu_bound_compute|bursty_load0.70|load=0.70|gpu_heavy_jax_matmul|profile=1` | 340 | 0.000781250 | 0.107428242 |
| `q01_gpu_bound_compute|bursty_load0.85|load=0.85|gpu_heavy_jax_matmul|profile=1` | 350 | 0.000781250 | 0.105882432 |
| `q01_gpu_bound_compute|bursty_load0.95|load=0.95|gpu_heavy_jax_matmul|profile=1` | 358 | 0.000781250 | 0.104692704 |
| `q01_gpu_bound_compute|poisson_load0.50|load=0.50|gpu_heavy_jax_matmul|profile=1` | 368 | 0.000781250 | 0.103260451 |
| `q01_gpu_bound_compute|poisson_load0.70|load=0.70|gpu_heavy_jax_matmul|profile=1` | 434 | 0.000781250 | 0.095085231 |
| `q01_gpu_bound_compute|poisson_load0.85|load=0.85|gpu_heavy_jax_matmul|profile=1` | 498 | 0.000781250 | 0.088765307 |
| `q01_gpu_bound_compute|poisson_load0.95|load=0.95|gpu_heavy_jax_matmul|profile=1` | 552 | 0.000781250 | 0.084311806 |
| `q10_cpu_host_bound|bursty_load0.50|load=0.50|cpu_heavy_local_bench|profile=8` | 1632 | 0.000781250 | 0.049034060 |
| `q10_cpu_host_bound|bursty_load0.70|load=0.70|cpu_heavy_local_bench|profile=8` | 1674 | 0.000781250 | 0.048415030 |
| `q10_cpu_host_bound|bursty_load0.85|load=0.85|cpu_heavy_local_bench|profile=8` | 1714 | 0.000781250 | 0.047846758 |
| `q10_cpu_host_bound|bursty_load0.95|load=0.95|cpu_heavy_local_bench|profile=8` | 1720 | 0.000781250 | 0.047763232 |
| `q10_cpu_host_bound|poisson_load0.50|load=0.50|cpu_heavy_local_bench|profile=8` | 3034 | 0.000781250 | 0.035962522 |
| `q10_cpu_host_bound|poisson_load0.70|load=0.70|cpu_heavy_local_bench|profile=8` | 3058 | 0.000781250 | 0.035821122 |
| `q10_cpu_host_bound|poisson_load0.85|load=0.85|cpu_heavy_local_bench|profile=8` | 3064 | 0.000781250 | 0.035786032 |
| `q10_cpu_host_bound|poisson_load0.95|load=0.95|cpu_heavy_local_bench|profile=8` | 3066 | 0.000781250 | 0.035774358 |
| `q11_cpu_gpu_coupled|bursty_load0.50|load=0.50|hybrid_rl_resac_ant|profile=2` | 994 | 0.000781250 | 0.062829664 |
| `q11_cpu_gpu_coupled|bursty_load0.50|load=0.50|hybrid_rl_resac_ant|profile=3` | 26 | 0.000781250 | 0.388482317 |
| `q11_cpu_gpu_coupled|bursty_load0.70|load=0.70|hybrid_rl_resac_ant|profile=2` | 908 | 0.000781250 | 0.065737776 |
| `q11_cpu_gpu_coupled|bursty_load0.70|load=0.70|hybrid_rl_resac_ant|profile=3` | 132 | 0.000781250 | 0.172413379 |
| `q11_cpu_gpu_coupled|bursty_load0.85|load=0.85|hybrid_rl_resac_ant|profile=2` | 884 | 0.000781250 | 0.066624168 |
| `q11_cpu_gpu_coupled|bursty_load0.85|load=0.85|hybrid_rl_resac_ant|profile=3` | 176 | 0.000781250 | 0.149314366 |
| `q11_cpu_gpu_coupled|bursty_load0.95|load=0.95|hybrid_rl_resac_ant|profile=2` | 800 | 0.000781250 | 0.070034646 |
| `q11_cpu_gpu_coupled|bursty_load0.95|load=0.95|hybrid_rl_resac_ant|profile=3` | 272 | 0.000781250 | 0.120108426 |
| `q11_cpu_gpu_coupled|poisson_load0.50|load=0.50|hybrid_rl_resac_ant|profile=2` | 1492 | 0.000781250 | 0.051283011 |
| `q11_cpu_gpu_coupled|poisson_load0.70|load=0.70|hybrid_rl_resac_ant|profile=2` | 1662 | 0.000781250 | 0.048589499 |
| `q11_cpu_gpu_coupled|poisson_load0.85|load=0.85|hybrid_rl_resac_ant|profile=2` | 1778 | 0.000781250 | 0.046977732 |
| `q11_cpu_gpu_coupled|poisson_load0.95|load=0.95|hybrid_rl_resac_ant|profile=2` | 1840 | 0.000781250 | 0.046179478 |
