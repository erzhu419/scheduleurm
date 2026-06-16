# Corner-Case Live Probe Summary

JSON artifact: `md/experiment_artifacts/corner_case_live_20260613_summary.json`

| Quantity | Value |
|---|---:|
| scenario_count | 10 |
| all_required_present | true |
| all_required_pass | true |
| all_required_stable_eta | true |

| scenario | node | stable ETA | last rate | background rate | ratio to empty |
|---|---|---:|---:|---:|---:|
| `cpu_empty_node001_single` | `node001` | true | 20.4763 | 0 |  |
| `cpu_empty_node002_single` | `node002` | true | 21.8761 | 0 |  |
| `cpu_full_node001_add_single_v2` | `node001` | true | 20.4698 | 7.88848 | 0.9997 |
| `cpu_full_node003_add_single` | `node003` | true | 19.7824 | 6.06032 | 0.9043 |
| `cpu_half_node001_add_single` | `node001` | true | 20.7385 | 12.523 | 0.9480 |
| `cpu_half_node001_add_single_v2` | `node001` | true | 22.3118 | 12.4131 | 1.0896 |
| `node007_28p8gb_add_small` | `node007-direct` | true | 8325.15 | 2359.1 | 0.9754 |
| `node007_30gb_add_small` | `node007-direct` | false | 8075.04 | 2358.85 | 0.9461 |
| `node007_30gb_add_small_stable` | `node007-direct` | true | 9367.45 | 2342.6 | 1.0976 |
| `node007_empty_small` | `node007-direct` | true | 8534.69 | 0 |  |

## Policy Admission Matrix

| scenario | Scheduleurm theorem | Salus/Gandiva-style packing | Pollux/Sia-style goodput | Gavel-style finish time | legacy safety |
|---|---|---|---|---|---|
| `cpu_empty_node001_single` | admit: positive lower service | not applicable | baseline row | baseline row | capacity gate admits if CPU/RAM fit |
| `cpu_empty_node002_single` | admit: positive lower service | not applicable | baseline row | baseline row | capacity gate admits if CPU/RAM fit |
| `cpu_full_node001_add_single_v2` | admit: positive lower service | not applicable | admit: ratio 1.000 | needs resident-delay/JCT holdout | capacity gate admits if CPU/RAM fit |
| `cpu_full_node003_add_single` | admit: positive lower service | not applicable | admit: ratio 0.904 | needs resident-delay/JCT holdout | capacity gate admits if CPU/RAM fit |
| `cpu_half_node001_add_single` | admit: positive lower service | not applicable | admit: ratio 0.948 | needs resident-delay/JCT holdout | capacity gate admits if CPU/RAM fit |
| `cpu_half_node001_add_single_v2` | admit: positive lower service | not applicable | admit: ratio 1.090 | needs resident-delay/JCT holdout | capacity gate admits if CPU/RAM fit |
| `node007_28p8gb_add_small` | admit: positive lower service | admit if memory fit | admit: ratio 0.975 | needs resident-delay/JCT holdout | node007 override admits; generic one-third would block |
| `node007_30gb_add_small` | pending stable ETA | admit if memory fit | pending stable ETA | needs resident-delay/JCT holdout | node007 override admits; generic one-third would block |
| `node007_30gb_add_small_stable` | admit: positive lower service | admit if memory fit | admit: ratio 1.098 | needs resident-delay/JCT holdout | node007 override admits; generic one-third would block |
| `node007_empty_small` | admit: positive lower service | baseline row | baseline row | baseline row | baseline row |

Interpretation boundary: these are controlled synthetic marginal-efficiency probes.  They validate ETA logging, co-location feasibility, and rate ratios for this measured slice; they are not a global theorem population unless admitted into the service-cache/replay gate.
