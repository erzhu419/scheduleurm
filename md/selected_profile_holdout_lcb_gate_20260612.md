# Selected-Profile Holdout LCB Gate

This artifact is reproducible.  A scoped stochastic lower-service claim is ready only when `gate_pass=true` and `scoped_claim_ready=true`.  Mean-service eta diagnostics remain separate: a negative absolute or diagonal eta blocks only that stronger mean-service claim, not the direct LCB lower-service capacity certificate.

| Quantity | Value |
|---|---:|
| `selected_profile_stochastic_lcb_ready` | true |
| `absolute_mean_load_eta_ready` | false |
| `diagonal_normalized_eta_ready` | false |
| `lcb_lower_service_capacity_ready` | true |
| `holdout_adjusted_eta` | -738.994 |
| `holdout_adjusted_eta_positive` | false |
| `supplemental_rate_count` | 921 |
| `sample_ready_count` | 11 |
| `sample_pending_count` | 0 |
| `target_count` | 11 |
| `selected_target_count` | 11 |

## Stochastic Reestimate

| Quantity | Value |
|---|---:|
| `epsilon_est_selected_actions` | 739.061 |
| `maximum_allowed_epsilon_est` | 0.0669218 |
| `epsilon_gap_to_positive_eta` | 738.994 |
| `eta` | -738.994 |
| `row_count` | 11 |
| `diagonal_normalized_eta` | -0.55866 |
| `diagonal_normalized_epsilon` | 0.75866 |
| `lcb_lower_service_delta` | 0.020993 |

## Dominant Eta Blockers

| Workload | Profile | Context | Train n | Holdout n | Epsilon | LCB/task | Holdout/task |
|---|---:|---|---:|---:|---:|---:|---:|
| `gpu_llm_distilgpt2` | 10 | `mixed` | 10 | 10 | 739.061 | 2560.66 | 3299.72 |
| `gpu_llm_distilgpt2` | 10 | `selected` | 10 | 10 | 651.764 | 2611.6 | 3263.37 |
| `light_control_local` | 13 | `selected` | 10 | 10 | 253.437 | 0 | 253.437 |
| `gpu_cnn_torch_resnet50` | 3 | `mixed` | 11 | 11 | 37.1481 | 12.7461 | 49.8942 |
| `cpu_heavy_local_bench` | 8 | `selected` | 10 | 10 | 35.0899 | 2.2201 | 37.31 |

## Dominant Normalized Eta Blockers

| Workload | Profile | Context | Epsilon(norm.) | LCB ratio | Holdout ratio | Normalizer |
|---|---:|---|---:|---:|---:|---:|
| `hybrid_rl_resac_ant` | 3 | `selected` | 0.75866 | 0 | 0.75866 | 0.334609 |
| `hybrid_rl_resac_ant` | 3 | `mixed` | 0.756131 | 0 | 0.756131 | 0.334609 |
| `gpu_cnn_torch_resnet50` | 3 | `mixed` | 0.727607 | 0.249654 | 0.977261 | 51.0552 |
| `gpu_cnn_torch_resnet50` | 3 | `selected` | 0.65088 | 0.258433 | 0.909313 | 51.0552 |
| `cpu_heavy_local_bench` | 8 | `selected` | 0.640565 | 0.0405279 | 0.681092 | 54.7796 |

## Sample Targets

| Workload | Profile | Context | Current n | Target n | Gap | Ready |
|---|---:|---|---:|---:|---:|---:|
| `light_control_local` | 13 | `selected` | 20 | 20 | 0 | true |
| `gpu_heavy_jax_matmul` | 8 | `selected` | 20 | 20 | 0 | true |
| `cpu_heavy_local_bench` | 8 | `selected` | 20 | 20 | 0 | true |
| `hybrid_rl_resac_ant` | 3 | `selected` | 21 | 20 | 0 | true |
| `cpu_heavy_local_bench` | 8 | `mixed` | 20 | 20 | 0 | true |
| `gpu_heavy_jax_matmul` | 4 | `mixed` | 20 | 20 | 0 | true |
| `gpu_cnn_torch_resnet50` | 3 | `selected` | 22 | 20 | 0 | true |
| `gpu_llm_distilgpt2` | 10 | `selected` | 20 | 20 | 0 | true |
| `gpu_cnn_torch_resnet50` | 3 | `mixed` | 22 | 20 | 0 | true |
| `gpu_llm_distilgpt2` | 10 | `mixed` | 20 | 20 | 0 | true |
| `hybrid_rl_resac_ant` | 3 | `mixed` | 21 | 20 | 0 | true |

## Scope

Tracks selected-profile stochastic lower-service readiness.  The old absolute mean-load eta remains reported separately.  A positive gate requires either the diagonal-normalized eta certificate or the direct LCB lower-service capacity certificate, both backed by the Lean diagonal-scaling theorem.
