# Trace-Driven Fast-Forward Replay Validation

Date: 2026-06-04

This module validates an independent replay layer for comparing legacy fixed
co-location caps against calibrated/adaptive co-location choices without waiting
for every RE-SAC/BAPR task to run to completion.

The replay code lives under `simulation/` and does not modify `skill/scheduler.py`.

## Method

The replay uses a cache-first workflow:

1. Run a real workload profile only until stable task-native progress/ETA is
   observed.
2. Cache the measured service curve by
   `workload_key + command_fingerprint + node_bucket + profile`.
3. For later tasks with the same workload key and command fingerprint, skip the
   real scheduler probe and replay from the cached service curve.
4. Fast-forward a full batch with bootstrap task variation, so same-type tasks
   are not assumed to be perfectly identical.

Unknown profiles still require a real measurement.  Known profiles do not:
`cache_needs_probe(cache, "hybrid_rl_resac_ant", 10) == false`.

For GPU and hybrid workloads, profile lookup is exact.  A multi-task co-location
profile such as `5/GPU` or `10/GPU` is never inferred from `1/GPU` or from a
nearby profile.  If the cache lacks that exact profile, replay raises a
measurement obligation:

```text
missing exact service profile ... multi-task co-location rates must be measured in the real environment
```

This matters because tasks sharing one GPU can have a different ETA/rate curve
from isolated tasks even when the command parameters and algorithm are the same.

## Workload Classes

The replay supports three resource classes:

| Workload key | Type | Source |
|---|---|---|
| `hybrid_rl_resac_ant` | hybrid RL | real RE-SAC Ant service curve from module 12 |
| `gpu_heavy_jax_matmul` | GPU-heavy | real synthetic JAX matmul service curve from module 6 |
| `cpu_heavy_protocol` | CPU-heavy | protocol saturation curve placeholder |

The CPU-heavy curve is present so the replay system handles CPU workloads, but
it is not yet a theorem-grade empirical CPU trace.  The pass condition checks
the two empirical GPU/hybrid classes separately.

## Policies Compared

Current note: module26 upgrades the default calibrated policy from a pure
makespan selector to `calibrated_guarded_statewise`, and module27 adds a
multi-workload `calibrated_global_guarded` selector plus SOTA-style baselines.
The early replay method below remains valid, but reviewer-facing performance
numbers should use module27 unless explicitly discussing an ablation.

Legacy baseline:

```text
hybrid_rl_resac_ant: profile 5/GPU
gpu_heavy_jax_matmul: profile 3/GPU
cpu_heavy_protocol: profile 32 workers
```

Calibrated replay in the early module20 close-out:

```text
select profile minimizing cached makespan proxy for each workload class
```

The current default calibrated replay after module27 is:

```text
hybrid_rl_resac_ant: profile 1/GPU in the mixed portfolio, 10/GPU standalone
gpu_heavy_jax_matmul: profile 1/GPU
cpu_heavy_protocol: profile 16 workers
```

## Command

```bash
python3 -m simulation.cli \
  --trials 101 \
  --seed 42 \
  --min-makespan-improvement 1.05 \
  --min-class-improvement 1.03 \
  --cache-out /tmp/scheduleurm_replay_cache.json \
  --report-out /tmp/scheduleurm_replay_report.json
```

## Result

Validation status: pass.

| Metric | Legacy | Calibrated | Improvement |
|---|---:|---:|---:|
| Portfolio makespan | 76472.987 s | 59534.263 s | 1.285x |
| Weighted mean flow | 32431.821 s | 23778.165 s | 1.364x |

Per workload:

| Workload | Legacy profile | Calibrated profile | Makespan improvement | Mean-flow improvement |
|---|---:|---:|---:|---:|
| `hybrid_rl_resac_ant` | 5 | 1 | 1.194x | 1.242x |
| `gpu_heavy_jax_matmul` | 3 | 1 | 1.064x | 1.225x |
| `cpu_heavy_protocol` | 32 | 16 | 1.285x | 1.390x |

The pass condition requires:

```text
portfolio makespan improvement >= 1.05
hybrid_rl_resac_ant makespan improvement >= 1.03
gpu_heavy_jax_matmul makespan improvement >= 1.03
```

All passed.

## Validation

- `python3 -m py_compile simulation/... skill/tests/test_simulation_fast_forward.py`
- Targeted simulation test loader after module26: `checks=18 failed=0`
- CLI replay: `pass=true`

## Interpretation

This is the first closed loop that compares a legacy fixed cap with a calibrated
policy without running all tasks to completion.  It is still a replay layer, not
the live scheduler policy itself.  The next step is to use this replay output to
set live policy parameters or add a calibrated profile lookup, then run a small
shadow/live A/B with the same cached profiles.
