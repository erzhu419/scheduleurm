# Selected-Profile Holdout LCB Gate

This artifact is reproducible.  A scoped stochastic lower-service claim is ready only when `gate_pass=true` and `scoped_claim_ready=true`.  Mean-service eta diagnostics remain separate: a negative absolute or diagonal eta blocks only that stronger mean-service claim, not the direct LCB lower-service capacity certificate.

| Quantity | Value |
|---|---:|
| `selected_profile_stochastic_lcb_ready` | false |
| `absolute_mean_load_eta_ready` | false |
| `diagonal_normalized_eta_ready` | false |
| `lcb_lower_service_capacity_ready` | false |
| `holdout_adjusted_eta` | -253.355 |
| `holdout_adjusted_eta_positive` | false |
| `supplemental_rate_count` | 921 |
| `sample_ready_count` | 9 |
| `sample_pending_count` | 2 |
| `target_count` | 11 |
| `selected_target_count` | 11 |

## Stochastic Reestimate

| Quantity | Value |
|---|---:|
| `epsilon_est_selected_actions` | 253.437 |
| `maximum_allowed_epsilon_est` | 0.0819347 |
| `epsilon_gap_to_positive_eta` | 253.355 |
| `eta` | -253.355 |
| `row_count` | 11 |
| `diagonal_normalized_eta` | -0.795603 |
| `diagonal_normalized_epsilon` | 0.995603 |
| `lcb_lower_service_delta` | 0.0341241 |

## Dominant Eta Blockers

| Workload | Profile | Context | Train n | Holdout n | Epsilon | LCB/task | Holdout/task |
|---|---:|---|---:|---:|---:|---:|---:|
| `light_control_local` | 13 | `selected` | 10 | 10 | 253.437 | 0 | 253.437 |
| `cpu_heavy_local_bench` | 8 | `selected` | 10 | 10 | 35.0899 | 2.2201 | 37.31 |
| `cpu_heavy_local_bench` | 8 | `mixed` | 10 | 10 | 31.2826 | 7.95817 | 39.2407 |
| `gpu_llm_distilgpt2` | 8 | `mixed` | 1 | 1 | 28.3899 | 1832.89 | 1861.28 |
| `gpu_cnn_torch_resnet50` | 3 | `selected` | 12 | 11 | 27.4856 | 16.9883 | 44.4739 |

## Dominant Normalized Eta Blockers

| Workload | Profile | Context | Epsilon(norm.) | LCB ratio | Holdout ratio | Normalizer |
|---|---:|---|---:|---:|---:|---:|
| `hybrid_rl_resac_ant` | 3 | `mixed` | 0.995603 | 0 | 0.995603 | 0.287929 |
| `hybrid_rl_resac_ant` | 3 | `selected` | 0.922588 | 0 | 0.922588 | 0.287929 |
| `cpu_heavy_local_bench` | 8 | `selected` | 0.640565 | 0.0405279 | 0.681092 | 54.7796 |
| `gpu_cnn_torch_resnet50` | 3 | `selected` | 0.612169 | 0.37837 | 0.990539 | 44.8986 |
| `cpu_heavy_local_bench` | 8 | `mixed` | 0.571062 | 0.145276 | 0.716339 | 54.7796 |

## Sample Targets

| Workload | Profile | Context | Current n | Target n | Gap | Ready |
|---|---:|---|---:|---:|---:|---:|
| `light_control_local` | 13 | `selected` | 20 | 20 | 0 | true |
| `gpu_heavy_jax_matmul` | 8 | `selected` | 20 | 20 | 0 | true |
| `cpu_heavy_local_bench` | 8 | `selected` | 20 | 20 | 0 | true |
| `hybrid_rl_resac_ant` | 3 | `selected` | 23 | 20 | 0 | true |
| `cpu_heavy_local_bench` | 8 | `mixed` | 20 | 20 | 0 | true |
| `gpu_heavy_jax_matmul` | 4 | `mixed` | 20 | 20 | 0 | true |
| `gpu_cnn_torch_resnet50` | 3 | `selected` | 23 | 20 | 0 | true |
| `gpu_llm_distilgpt2` | 8 | `selected` | 2 | 20 | 18 | false |
| `gpu_cnn_torch_resnet50` | 3 | `mixed` | 23 | 20 | 0 | true |
| `gpu_llm_distilgpt2` | 8 | `mixed` | 2 | 20 | 18 | false |
| `hybrid_rl_resac_ant` | 3 | `mixed` | 23 | 20 | 0 | true |

## Scope

Tracks selected-profile stochastic lower-service readiness.  The old absolute mean-load eta remains reported separately.  A positive gate requires either the diagonal-normalized eta certificate or the direct LCB lower-service capacity certificate, both backed by the Lean diagonal-scaling theorem.
