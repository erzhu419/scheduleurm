# Node007 Fresh Task-Native ETA Calibration, 2026-06-26

This calibration replaces the old node007 replay rows that depended on legacy
history-style ETA.  The probes explicitly bypassed the default `scheduler.py`
placement policy and measured selected co-location profiles through
`algorithm.experiments.remote_workload_selected_profile_probe`.  The only
scheduler-layer behavior used was remote command transport to `node007-direct`;
the measured profiles were fixed directly by the experiment command.

## Scope

Measured node: `node007-direct`.

Initial measured GPUs: `1,2,3`.  GPU 0 was left occupied by existing BAPR work
and was not used for the first probes.  After the GPUs became idle, the
high-profile extension used GPUs `0,1,2,3`.

Measured workloads:

| workload key | command template | profiles | unit | run id |
|---|---|---:|---|---|
| `gpu_cnn_torch_progress_stack` | `torch_cnn_resnet50.cmd.tpl` | 1-4 | `step` | `node007_eta_matrix_fresh_20260626_cnn_gpus1_3_v2` |
| `gpu_cnn_torch_progress_stack` | `torch_cnn_resnet50.cmd.tpl` | 5-8 | `step` | `node007_eta_matrix_fresh_20260626_cnn_gpus0_3_p5_8` |
| `gpu_llm_torch_decoder_stack` | `torch_llm_distilgpt2.cmd.tpl` | 1-4 | `step` | `node007_eta_matrix_fresh_20260626_llm_gpus1_3` |
| `gpu_llm_torch_decoder_stack` | `torch_llm_distilgpt2.cmd.tpl` | 5-8 | `step` | `node007_eta_matrix_fresh_20260626_llm_gpus0_3_p5_8` |
| `hybrid_rl_resac_ant_node007_tqdm` | `resac_ant_real_venv.cmd.tpl` | 1-5 | `iter` | `node007_eta_matrix_fresh_20260626_rl_gpus1_3` |
| `hybrid_rl_resac_ant_node007_tqdm` | `resac_ant_real_venv.cmd.tpl` | 6 boundary | `iter` | `node007_eta_matrix_fresh_20260626_rl_gpus0_3_p6_8` |

All rows used workload-native progress output and `ScheduleurmStableRate`
termination, not the TUI historical ETA.

## Stable Rates

Rates below are per task.  The replay cache stores aggregate service per GPU,
so the three-GPU aggregate from the probe is divided by three when constructing
`ProfileRecord.aggregate_rate`.

| workload | profile | samples | per-GPU aggregate stable rate | mean task rate | min task rate | max task rate |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 1 | 3 | 31.712409 | 31.712409 | 31.360761 | 31.975495 |
| CNN | 2 | 6 | 41.266314 | 20.633157 | 14.243769 | 31.455579 |
| CNN | 3 | 9 | 44.898619 | 14.966206 | 9.552968 | 30.760114 |
| CNN | 4 | 12 | 52.058229 | 13.014557 | 7.100000 | 30.772638 |
| LLM | 1 | 3 | 326.210224 | 326.210224 | 283.057200 | 391.340228 |
| LLM | 2 | 6 | 587.468337 | 293.734169 | 209.710195 | 393.799079 |
| LLM | 3 | 9 | 906.904105 | 302.301368 | 230.719146 | 327.629498 |
| LLM | 4 | 12 | 1059.975744 | 264.993936 | 103.477453 | 353.982622 |
| RL | 1 | 3 | 0.291317 | 0.291317 | 0.285714 | 0.294118 |
| RL | 2 | 6 | 0.331844 | 0.165922 | 0.135135 | 0.294118 |
| RL | 3 | 9 | 0.287929 | 0.095976 | 0.089286 | 0.107527 |
| RL | 4 | 12 | 0.304190 | 0.076048 | 0.066667 | 0.090090 |
| RL | 5 | 15 | 0.409673 | 0.081935 | 0.053191 | 0.294118 |
| CNN | 5 | 20 | 54.660449 | 10.932090 | 5.746364 | 31.246791 |
| CNN | 6 | 24 | 52.363683 | 8.727281 | 5.241575 | 31.237621 |
| CNN | 7 | 28 | 56.468212 | 8.066887 | 4.758418 | 31.754258 |
| CNN | 8 | 31/32 stable | boundary | 8.388804 | 3.670000 | 30.571686 |
| LLM | 5 | 20 | 1395.817993 | 279.163599 | 143.704522 | 366.194078 |
| LLM | 6 | 24 | 1528.326856 | 254.721143 | 106.799878 | 358.613177 |
| LLM | 7 | 28 | 1580.907363 | 225.843909 | 53.850059 | 385.768414 |
| LLM | 8 | 32 | 1903.441792 | 237.930224 | 61.111156 | 349.366526 |
| RL | 6 | 20/24 stable | boundary | 0.081874 | 0.044843 | 0.285714 |

CNN profile 8 and RL profile 6 are capacity boundaries in the default replay
cache.  They block older optimistic high-profile rows but are not used as exact
service actions.

## Replay Integration

`simulation.defaults.build_default_cache()` now calls
`_add_fresh_node007_eta_matrix(cache)` after the older node007 rows.  The fresh
rows use `force_replace=True`, so these task-native tqdm measurements override
older node007 rows for the same workload/profile during replay.

This does not change the production scheduler's legacy default policy.  It only
changes the experiment/replay service cache used by the algorithm and SOTA
comparison gates.

## Fresh SOTA Gates

Artifacts generated from this calibration:

| artifact | status |
|---|---|
| `md/experiment_artifacts/sota_candidate_union_gate_fresh_eta_20260626.json` | pass |
| `md/experiment_artifacts/sota_algorithm_upgrade_gate_fresh_eta_20260626.json` | pass |
| `md/experiment_artifacts/sota_strict_dominance_frontier_fresh_eta_20260626.json` | pass |
| `md/experiment_artifacts/sota_candidate_union_gate_fresh_eta_full_v6_20260626.json` | pass |
| `md/experiment_artifacts/sota_strict_dominance_frontier_fresh_eta_full_v6_20260626.json` | pass |
| `md/experiment_artifacts/or_gate_online_arrivals_fresh_eta_full_v6_20260626.json` | pass |
| `md/experiment_artifacts/or_gate_ablation_suite_fresh_eta_full_v6_20260626.json` | pass |
| `md/experiment_artifacts/online_ablation_summary_ci_fresh_eta_full_v6_20260626.json` | pass |
| `md/experiment_artifacts/selected_profile_holdout_lcb_gate_fresh_eta_full_v6_20260626.json` | pass |

Key conclusion under the measured-cache policy-semantics scope:

| gate statement | value |
|---|---|
| `fixed_online_policy_pareto_dominates_sota_style_all` | true |
| `union_makespan_beats_sota_envelope_all` | true |
| `union_mean_flow_beats_sota_envelope_all` | true |
| `strict_pareto_ready` | true |
| `strict_frontier_count` | 0 |

The full-load v6 refresh also closes 192 online replay scenarios with zero
Pareto-dominated candidate scenarios and keeps the measured-cache strict SOTA
frontier closed.  The v6 selector treats CNN closed-batch tail drain and
rolling-arrival resource-adaptive admission as distinct finite trajectory
actions in the same candidate union.  See
`md/fresh_eta_full_load_closure_20260626.md`.

Four-quadrant check against the registered SOTA-style policy envelope:

| group | SOTA rows checked | SOTA rows that Pareto-dominate our fixed policy | worst makespan envelope ratio | worst mean-flow envelope ratio |
|---|---:|---:|---:|---:|
| q00 | 14 | 0 | 1.000000 | 1.000000 |
| q01 | 56 | 0 | 1.000000 | 1.000000 |
| q10 | 14 | 0 | 1.000000 | 1.000000 |
| q11 | 14 | 0 | 1.000000 | 1.000000 |
| portfolio | 14 | 0 | 1.000000 | 1.000000 |

Boundary: this remains a measured-cache policy-semantics result.  It does not
claim direct full-stack binary superiority over Gavel, Pollux, Sia, IADeep, or
Salus.
