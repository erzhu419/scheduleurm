# Module58 Completed Wall-Clock Audit

```text
workload_key = freqduet_cpu_ablation_c9_16
record_count = 110
all_have_positive_units_and_duration = True
parsed_units_sum = 138900
```

| Quantity | Min | P10 | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| `parsed_units_episode` | 440.000000 | 480.000000 | 1300.000000 | 2600.000000 | 3000.000000 |
| `duration_s` | 795.996133 | 852.183030 | 2615.258034 | 5613.975529 | 6592.062059 |
| `realized_episode_s` | 0.289775 | 0.382723 | 0.487672 | 0.623970 | 0.672824 |

## Slowest Records

| Task | CPU | RAM | Units | Duration s | Episode/s |
|---|---:|---:|---:|---:|---:|
| `t7461` | 11 | 8192 | 1100 | 3796.052 | 0.289775 |
| `t7613` | 11 | 8192 | 440 | 1493.576 | 0.294595 |
| `t7684` | 11 | 8192 | 440 | 1368.651 | 0.321484 |
| `t7756` | 13 | 16384 | 520 | 1570.978 | 0.331004 |
| `t7626` | 11 | 8192 | 440 | 1328.353 | 0.331237 |
| `t7784` | 13 | 16384 | 520 | 1522.743 | 0.341489 |
| `t7686` | 11 | 8192 | 440 | 1238.183 | 0.355359 |
| `t7628` | 11 | 8192 | 440 | 1229.452 | 0.357883 |
| `t6318` | 11 | 32768 | 1100 | 2976.018 | 0.369621 |
| `t7785` | 13 | 16384 | 520 | 1363.214 | 0.381451 |
| `t7786` | 13 | 16384 | 520 | 1358.687 | 0.382723 |
| `t6067` | 11 | 32768 | 2200 | 5659.433 | 0.388731 |

Completed production records provide a wall-clock sanity audit for the Module58 c9_16 command-shape certificate. The theorem service curve still comes from controlled profile probes; this audit verifies that mapped records have parseable positive work units and completed at nonzero realized service rates.
