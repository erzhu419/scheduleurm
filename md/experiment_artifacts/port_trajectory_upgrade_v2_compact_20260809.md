# Port trajectory-action certificate

- Status: `PORT_TRAJECTORY_UPGRADE_PASS`
- Full ledger hash: `6252b003d41d5495e12030935829c84b0e9d45b1be40c918a02c0d553b709b45`
- Compact certificate hash: `a24914a4370eeb96d86b69907cd4ebb1c48b1cb7168fa4b8eaac36046c5ae1a2`
- Selected global plan: `plan|spt_static+robust_maxweight+robust_maxweight`
- Finite trajectory family: `31` candidates.
- Generated-family oracle gap: `0.0`.
- Finite-family P0: `0.000645255` <= `0.01`.

## Public BACASP-S source-core comparison

| Policy | Quay makespan | Mean quay flow | Source objective | Pareto |
|---|---:|---:|---:|---:|
| `edd_static` | 287.868750 | 58.006667 | 4318300.000000 | false |
| `fcfs_static` | 287.868750 | 58.826875 | 4392118.750000 | false |
| `reconfiguration_greedy` | 178.118750 | 15.240729 | 705984.375000 | true |
| `spt_static` | 178.118750 | 15.240729 | 705984.375000 | true |
| `trajectory_robust_global` | 176.500000 | 16.451458 | 777562.500000 | true |

## Synthetic four-resource comparison

| Instance | Policy | Makespan | Mean flow | Weighted tardiness | Pareto |
|---|---|---:|---:|---:|---:|
| `balanced_heterogeneous_8` | `edd_static` | 371.020605 | 192.859237 | 806.859396 | false |
| `balanced_heterogeneous_8` | `fcfs_static` | 328.306959 | 190.835308 | 795.677340 | false |
| `balanced_heterogeneous_8` | `reconfiguration_greedy` | 234.129786 | 138.385812 | 402.968508 | false |
| `balanced_heterogeneous_8` | `spt_static` | 243.595126 | 142.309439 | 407.953312 | false |
| `balanced_heterogeneous_8` | `trajectory_robust_global` | 223.557958 | 135.938105 | 378.034261 | true |
| `berth_crane_pressure_9` | `edd_static` | 380.495731 | 195.914360 | 1061.951974 | false |
| `berth_crane_pressure_9` | `fcfs_static` | 321.900089 | 194.804673 | 1043.496882 | false |
| `berth_crane_pressure_9` | `reconfiguration_greedy` | 237.206180 | 150.081830 | 570.969705 | true |
| `berth_crane_pressure_9` | `spt_static` | 294.316059 | 149.214036 | 583.161948 | false |
| `berth_crane_pressure_9` | `trajectory_robust_global` | 245.150434 | 141.174900 | 541.655739 | true |
| `yard_gate_surge_9` | `edd_static` | 362.171706 | 223.918864 | 1264.806692 | false |
| `yard_gate_surge_9` | `fcfs_static` | 392.812703 | 233.764746 | 1363.780967 | false |
| `yard_gate_surge_9` | `reconfiguration_greedy` | 323.002531 | 190.034341 | 992.416786 | false |
| `yard_gate_surge_9` | `spt_static` | 327.482273 | 191.647115 | 1018.353755 | false |
| `yard_gate_surge_9` | `trajectory_robust_global` | 296.565893 | 179.997024 | 881.239078 | true |

The certificate supports exact selection over the registered finite
trajectory family and the stated deterministic port instances. It does
not establish unrestricted port optimality or stochastic stability.
