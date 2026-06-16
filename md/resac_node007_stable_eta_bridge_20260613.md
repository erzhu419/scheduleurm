# RE-SAC Node007 Stable ETA Bridge, 2026-06-13

## Scope

This note records the controlled RE-SAC Ant-v2 ETA/service-rate probes on
`node007-direct` GPU0.  The measurements are not full training runs.  They stop
only after task-native progress logs produce stable rate windows.

The TUI ETA for these jobs showed minute-scale estimates during warmup.  That
ETA is not used for theorem-facing service evidence because JAX compile,
environment warmup, and evaluation intervals dominate the first few iterations.

## Stable ETA Rule

The probe wrapper parses RE-SAC log lines such as `s/iter`, emits
`ScheduleurmProgress`, skips the first warmup sample, and terminates only when
the tail window satisfies:

- `stable_windows = 3`
- `min_rate_samples = 5`
- `stable_cv <= 0.20`
- `stable_last_two_relative_delta <= 0.25`

The wrapper prints `ScheduleurmStableRate` before terminating the child process.
Only summaries with `measurement_valid = true` are admissible.

## Results

| Profile | Run | Stable tasks | Aggregate stable rate | Mean stable rate | Probe wall time |
|---|---|---:|---:|---:|---:|
| p2 | `resac_node007_direct_stable_eta_p2_coordinated_v2_20260613` | 2/2 | 0.292735043 iter/s | 0.146367522 iter/s | 244.53 s |
| p3 | `resac_node007_direct_stable_eta_p3_coordinated_v2_20260613` | 3/3 | 0.465894466 iter/s | 0.155298155 iter/s | 315.62 s |

Per-task stable rates:

| Profile | Task index | Stable rate | Seconds per iter | Tail window |
|---|---:|---:|---:|---|
| p2 | 0 | 0.153846154 | 6.50 | `[0.138888889, 0.138888889, 0.153846154]` |
| p2 | 1 | 0.138888889 | 7.20 | `[0.136986301, 0.138888889, 0.138888889]` |
| p3 | 0 | 0.285714286 | 3.50 | `[0.285714286, 0.285714286, 0.285714286]` |
| p3 | 1 | 0.0900900901 | 11.10 | `[0.0900900901, 0.0900900901, 0.0900900901]` |
| p3 | 2 | 0.0900900901 | 11.10 | `[0.0900900901, 0.0900900901, 0.0900900901]` |

## Interpretation

For this measured node/profile, p3 has higher aggregate stable throughput than
p2.  The measurement also shows why TUI ETA is misleading: early iterations
reported hundreds to thousands of seconds remaining, then stabilized after
warmup.  Scheduleurm-facing ETA should therefore use task-native stable progress
windows and cache identical `(workload, profile, node, GPU/load-state)` service
rows, not the TUI's short-run ETA.

The p3 run required coordinated profile launch, where one remote script starts
all co-located tasks and collects all logs.  One-process-per-SSH probing caused
a dropped SSH session in an earlier p3 attempt and is not reliable enough for
dense co-location evidence.

## Artifact Paths

- p2 summary:
  `/home/erzhu419/.claude/scheduler/experiments/runs/resac_node007_direct_stable_eta_p2_coordinated_v2_20260613/reports/profile_2_per_gpu_summary.json`
- p3 summary:
  `/home/erzhu419/.claude/scheduler/experiments/runs/resac_node007_direct_stable_eta_p3_coordinated_v2_20260613/reports/profile_3_per_gpu_summary.json`

