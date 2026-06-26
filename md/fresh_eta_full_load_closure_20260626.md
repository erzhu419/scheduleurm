# Fresh Full-Load ETA Closure, 2026-06-26

This document records the fresh task-native ETA recalibration used after the
node007 GPUs became idle.  The probes bypassed default `scheduler.py` placement
rules and used the experiment/simulation layer under `algorithm/`.  The only
use of `skill/scheduler.py` was remote command transport.

## Fresh service rows

All GPU rows used workload-native progress output and stable-rate termination.
Rows marked as placement-invalid are admitted only as capacity boundaries, not
as usable replay service points.

| workload | profile | running | stable | placement valid | per-GPU aggregate rate | mean task rate | min task rate | max task rate | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CNN | 5 | 20 | 20 | true | 54.660449 | 10.932090 | 5.746364 | 31.246791 |  |
| CNN | 6 | 24 | 24 | true | 52.363683 | 8.727281 | 5.241575 | 31.237621 |  |
| CNN | 7 | 28 | 28 | true | 56.468212 | 8.066887 | 4.758418 | 31.754258 |  |
| CNN | 8 | 32 | 31 | false | 65.013234 | 8.388804 | 3.670000 | 30.571686 | boundary: one task failed to reach stable placement |
| LLM | 5 | 20 | 20 | true | 1395.817993 | 279.163599 | 143.704522 | 366.194078 |  |
| LLM | 6 | 24 | 24 | true | 1528.326856 | 254.721143 | 106.799878 | 358.613177 |  |
| LLM | 7 | 28 | 28 | true | 1580.907363 | 225.843909 | 53.850059 | 385.768414 |  |
| LLM | 8 | 32 | 32 | true | 1903.441792 | 237.930224 | 61.111156 | 349.366526 |  |
| RL | 6 | 24 | 20 | false | 0.409371 | 0.081874 | 0.044843 | 0.285714 | boundary: LLVM pthread creation failures under 6/GPU |

Canonical corner-case rows also use task-native progress/ETA:

| scenario | stable ETA | marginal rate | resident rate | ratio to empty |
|---|---:|---:|---:|---:|
| CPU empty node001 single | true | 17.504426 | 0 |  |
| CPU 96-worker resident + single | true | 17.354271 | 33.572564 | 0.9914 |
| CPU 180-worker resident + single | true | 17.383411 | 20.337262 | 0.9931 |
| GPU empty node007 small CUDA | true | 491.827392 | 0 |  |
| GPU 30GB resident + small CUDA | true | 443.983110 | 1049.791100 | 0.9027 |

The corner-case lower-service gate passes with five admitted rows and positive
row-level lower-service slack at offered load fraction 0.8.  The Gavel-style
resident-delay/JCT holdout also passes on the three co-location rows.

## Algorithm update

The default replay cache now force-replaces old node007 rows with the fresh
task-native ETA rows.  CNN profile 8 and RL profile 6 are capacity boundaries,
so replay cannot silently use old optimistic high-profile rows.

The online candidate uses `online_pareto_slack`, a fixed finite-action tie-break
over the same measured candidate union.  It keeps the closed-batch CNN
tail-drain bridge for static queues, but uses the measured
resource-adaptive trajectory for single-CNN rolling Poisson/bursty queues where
the closed-batch bridge can overfit tail drain.  Heterogeneous portfolios keep
the measured Scheduleurm bridge action.

## Fresh gates

| artifact | result |
|---|---|
| `md/experiment_artifacts/corner_case_lower_service_gate_fresh_eta_20260626.json` | PASS |
| `md/experiment_artifacts/gavel_resident_delay_jct_holdout_fresh_eta_20260626.json` | PASS |
| `md/experiment_artifacts/sota_candidate_union_gate_fresh_eta_full_v6_20260626.json` | PASS |
| `md/experiment_artifacts/sota_strict_dominance_frontier_fresh_eta_full_v6_20260626.json` | PASS; strict frontier count 0 |
| `md/experiment_artifacts/or_gate_online_arrivals_fresh_eta_full_v6_20260626.json` | PASS; 192 online scenarios |
| `md/experiment_artifacts/or_gate_ablation_suite_fresh_eta_full_v6_20260626.json` | PASS; 24 ablation scenarios |
| `md/experiment_artifacts/online_ablation_summary_ci_fresh_eta_full_v6_20260626.json` | PASS |
| `md/experiment_artifacts/or_algorithm_upgrade_gate_fresh_eta_full_v6_20260626.json` | PASS; 0 regressions, 7 improving tasksets |
| `md/experiment_artifacts/selected_profile_holdout_lcb_gate_fresh_eta_full_v6_20260626.json` | PASS; selected LCB lower-service capacity |

## Main results after recalibration

| statement | value |
|---|---:|
| measured-cache strict SOTA frontier open rows | 0 |
| online scenarios | 192 |
| online Pareto-dominated candidate scenarios | 0 |
| ablation scenarios | 24 |
| ablation policies Pareto-dominating full candidate | 0 |
| online geomean legacy/candidate makespan | 1.618438 |
| online geomean legacy/candidate mean flow | 4.180099 |
| ablation geomean legacy/full-candidate makespan | 1.736207 |
| ablation geomean legacy/full-candidate mean flow | 3.005319 |
| selected-profile LCB lower-service delta | 0.034124 |

Boundary: this remains a measured-cache policy-semantics and controlled-probe
claim.  It is not a direct full-stack binary superiority claim over Gavel,
Pollux, Sia, IADeep, or Salus, and it does not certify arbitrary future
workloads without service admission.
