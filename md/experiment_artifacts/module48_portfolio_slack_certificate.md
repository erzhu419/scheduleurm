# Measured Finite-Slice Slack Certificate

Scope: measured finite service-action slice; not a production-arrival or unmeasured-global-action certificate

```text
taskset = hybrid_research_portfolio
policy = calibrated_global_guarded
load_fraction = 0.8
selected_profiles = {'hybrid_rl_resac_ant': 3, 'gpu_heavy_jax_matmul': 1, 'cpu_heavy_local_bench': 8}
full_action_count = 243
```

| Quantity | Value | Meaning |
|---|---:|---|
| `delta` | 0.066921842 | capacity slack |
| `L` | 47.897952 | fabric service Lipschitz envelope |
| `rho` | 0.000000000 | candidate cover radius |
| `Lrho` | 0.000000000 | candidate support loss |
| `epsilon_est` | 0.000000000 | lower-service estimation loss |
| `beta` | 0.000000000 | queue-scaled penalty slope |
| `alpha1` | 0.000000000 | queue-scaled oracle error |
| `eta` | 0.066921842 | remaining drift margin |
| `B` | 6361.355 | second-order drift bound |
| `P0` | 0.000000000 | fixed penalty term |
| `alpha0` | 0.000000000 | fixed oracle-error term |
| `finite_set_threshold_N` | 95072.000 | Foster finite-set threshold |

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
| `gpu_heavy_jax_matmul` | 55.175835200 | 68.969794000 |
| `hybrid_rl_resac_ant` | 0.267687369 | 0.334609211 |

## Interpretation

This is a positive theorem-condition certificate for the measured finite service-action slice. It is not a production-arrival certificate and does not claim statistical generalization outside the measured Scheduleurm buckets.
