# Measured Finite-Slice Slack Certificate

Scope: measured finite service-action slice; not a production-arrival or unmeasured-global-action certificate

```text
taskset = hybrid_research_portfolio
policy = scheduleurm_sota_union_pareto_slack
load_fraction = 0.8
selected_profiles = {'hybrid_rl_resac_ant': 2, 'gpu_heavy_jax_matmul': 1, 'gpu_cnn_torch_resnet50': 3, 'gpu_llm_distilgpt2': 10, 'cpu_heavy_local_bench': 8}
full_action_count = 5832
```

| Quantity | Value | Meaning |
|---|---:|---|
| `delta` | 0.078777710 | capacity slack |
| `L` | 5727.144 | fabric service Lipschitz envelope |
| `rho` | 0.000000000 | candidate cover radius |
| `Lrho` | 0.000000000 | candidate support loss |
| `epsilon_est` | 0.000000000 | lower-service estimation loss |
| `beta` | 0.000000000 | queue-scaled penalty slope |
| `alpha1` | 0.000000000 | queue-scaled oracle error |
| `eta` | 0.078777710 | remaining drift margin |
| `B` | 34499765.485 | second-order drift bound |
| `P0` | 0.000000000 | fixed penalty term |
| `alpha0` | 0.000000000 | fixed oracle-error term |
| `finite_set_threshold_N` | 437938174.000 | Foster finite-set threshold |

| Component | Usable | Source status |
|---|---:|---|
| `fabric` | true | candidate_action_family_equals_measured_full_slice |
| `service` | true | measured_service_map_used_as_lower_service_for_this_finite_slice |
| `penalty` | true | finite_max_envelope |
| `oracle` | true | exact_candidate_maxweight_oracle_by_construction |
| `capacity` | true | optimal |
| `moment` | true | finite_support_bound_from_measured_action_slice_and_declared_load |

## Load Certificate

| Class | lambda | selected service |
|---|---:|---:|
| `cpu_heavy_local_bench` | 43.823682800 | 54.779603500 |
| `gpu_cnn_torch_resnet50` | 81.688328376 | 102.110410470 |
| `gpu_heavy_jax_matmul` | 74.897009200 | 93.621261500 |
| `gpu_llm_distilgpt2` | 5187.720412000 | 6484.650515000 |
| `hybrid_rl_resac_ant` | 0.315110840 | 0.393888550 |

## Interpretation

This is a positive theorem-condition certificate for the measured finite service-action slice. It is not a production-arrival certificate and does not claim statistical generalization outside the measured Scheduleurm buckets.
