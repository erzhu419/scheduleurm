# Module65 ZSW TSP/SUMO c_le2 Completed-History Certificate

```text
run_id = module65_zsw_tsp_sumo_eval_c_le2_completed_history
scope = ZSW TSP/SUMO runner eval, no GPU, <=2 CPU cores, explicit --duration
workload_key = zsw_tsp_sumo_eval_c_le2_completed_history
record_count = 50
total_units_sim_second = 900000.000000
profile_domain = [1]
min_realized_sim_second_s = 5.896139229546
median_realized_sim_second_s = 11.969454927029
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only ZSW TSP/SUMO runner records requesting at most 2 CPU cores when `baseline_runner.py`, `m21_cycle_conserving_tsp_runner.py`, `oracle_tsp_runner.py`, or `m2_scored_oracle_tsp_runner.py` carries explicit `--duration`. It does not map CFCMT, offline-sumo, H2Oplus, direct SUMO binary, or commands without parseable duration.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 50 |
| Parsed simulated seconds | 900000 |
| Minimum realized simulated-second/s | 5.896139229546 |
| P10 realized simulated-second/s | 7.508263740999 |
| Median realized simulated-second/s | 11.969454927029 |
| Maximum duration s | 3052.845141 |

## Script Breakdown

| Script | Count |
|---|---:|
| `baseline_runner.py` | 18 |
| `m21_cycle_conserving_tsp_runner.py` | 29 |
| `m2_scored_oracle_tsp_runner.py` | 1 |
| `oracle_tsp_runner.py` | 2 |

## Artifacts

```text
md/experiment_artifacts/module65_zsw_tsp_sumo_eval_c_le2_completed_history.json
md/experiment_artifacts/module65_zsw_tsp_sumo_eval_c_le2_completed_history.md
md/experiment_artifacts/module65_zsw_tsp_sumo_eval_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | Script | CPU | Units | Duration s | Sim-second/s |
|---|---|---|---:|---:|---:|---:|
| `t4151` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 1 | 18000 | 3052.845141 | 5.896139229546 |
| `t4154` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 1 | 18000 | 2443.797951 | 7.365584373601 |
| `t4158` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 1 | 18000 | 2428.593672 | 7.411696822329 |
| `t4556` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 2 | 18000 | 2404.858352 | 7.484848321171 |
| `t4146` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 1 | 18000 | 2397.358513 | 7.508263740999 |
| `t4557` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 2 | 18000 | 2393.820990 | 7.519359248060 |
| `t4150` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 1 | 18000 | 2377.022153 | 7.572499892872 |
| `t4161` | `ZSW_platform` | `m21_cycle_conserving_tsp_runner.py` | 1 | 18000 | 2324.230942 | 7.744497189501 |

## Interpretation

Strict completed-history lower-service certificate for ZSW TSP/SUMO c_le2 eval runners. The parser accepts baseline_runner.py, m21_cycle_conserving_tsp_runner.py, oracle_tsp_runner.py, and m2_scored_oracle_tsp_runner.py only when --duration is explicit. Only profile 1 is loaded; CFCMT, offline-sumo, H2Oplus, direct SUMO binary, and unparseable commands are not claimed.
