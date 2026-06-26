# Corner-Case Live Probe Summary

JSON artifact: `md/experiment_artifacts/corner_case_live_fresh_eta_20260626_summary.json`

| Quantity | Value |
|---|---:|
| scenario_count | 5 |
| all_required_present | true |
| all_required_pass | true |
| all_required_stable_eta | true |

| scenario | node | stable ETA | last rate | background rate | ratio to empty |
|---|---|---:|---:|---:|---:|
| `cpu_empty_node001_single` | `node001` | true | 17.5044 | 0 |  |
| `cpu_full_node001_add_single_v2` | `node001` | true | 17.3834 | 20.3373 | 0.9931 |
| `cpu_half_node001_add_single_v2` | `node001` | true | 17.3543 | 33.5726 | 0.9914 |
| `node007_30gb_add_small_stable` | `node007-direct` | true | 443.983 | 1049.79 | 0.9027 |
| `node007_empty_small` | `node007-direct` | true | 491.827 | 0 |  |

## Policy Admission Matrix

| scenario | Scheduleurm theorem | Salus/Gandiva-style packing | Pollux/Sia-style goodput | Gavel-style finish time | legacy safety |
|---|---|---|---|---|---|
| `cpu_empty_node001_single` | admit: positive lower service | not applicable | baseline row | baseline row | capacity gate admits if CPU/RAM fit |
| `cpu_full_node001_add_single_v2` | admit: positive lower service | not applicable | admit: ratio 0.993 | needs resident-delay/JCT holdout | capacity gate admits if CPU/RAM fit |
| `cpu_half_node001_add_single_v2` | admit: positive lower service | not applicable | admit: ratio 0.991 | needs resident-delay/JCT holdout | capacity gate admits if CPU/RAM fit |
| `node007_30gb_add_small_stable` | admit: positive lower service | admit if memory fit | admit: ratio 0.903 | needs resident-delay/JCT holdout | node007 override admits; generic one-third would block |
| `node007_empty_small` | admit: positive lower service | baseline row | baseline row | baseline row | baseline row |

Interpretation boundary: these are controlled synthetic marginal-efficiency probes.  They validate ETA logging, co-location feasibility, and rate ratios for this measured slice; they are not a global theorem population unless admitted into the service-cache/replay gate.
