# Measured Finite-Slice Slack Certificate

Scope: measured finite service-action slice; not a production-arrival or unmeasured-global-action certificate

```text
taskset = hybrid_research_portfolio
cache_source = default_cache+live_overlay:md/experiment_artifacts/service_cache_v2_live_merged_20260701.json
policy = scheduleurm_sota_union_pareto_slack
load_fraction = 0.8
selected_profiles = {'hybrid_rl_resac_ant': 5, 'gpu_heavy_jax_matmul': 1, 'gpu_cnn_torch_resnet50': 4, 'gpu_llm_distilgpt2': 8, 'cpu_heavy_local_bench': 8}
full_action_count = 4725
```

| Quantity | Value | Meaning |
|---|---:|---|
| `delta` | 0.081934686 | capacity slack |
| `L` | 2984.372 | fabric service Lipschitz envelope |
| `rho` | 0.000000000 | candidate cover radius |
| `Lrho` | 0.000000000 | candidate support loss |
| `epsilon_est` | 0.000000000 | lower-service estimation loss |
| `beta` | 0.000000000 | queue-scaled penalty slope |
| `alpha1` | 0.000000000 | queue-scaled oracle error |
| `eta` | 0.081934686 | remaining drift margin |
| `B` | 11209514.968 | second-order drift bound |
| `P0` | 0.000000000 | fixed penalty term |
| `alpha0` | 0.000000000 | fixed oracle-error term |
| `finite_set_threshold_N` | 136810386.000 | Foster finite-set threshold |

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
| `gpu_cnn_torch_resnet50` | 83.293166011 | 104.116457513 |
| `gpu_heavy_jax_matmul` | 74.897009200 | 93.621261500 |
| `gpu_llm_distilgpt2` | 2955.332181600 | 3694.165227000 |
| `hybrid_rl_resac_ant` | 0.327738745 | 0.409673431 |

## Interpretation

This is a positive theorem-condition certificate for the measured finite service-action slice. It is not a production-arrival certificate and does not claim statistical generalization outside the measured Scheduleurm buckets.
