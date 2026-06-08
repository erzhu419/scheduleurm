# Module49 Production-Load Capacity Attempt

Date: 2026-06-08

This module attempts to move from a declared finite-slice load model to a
production-history load certificate.  It reads Scheduleurm task records from
`~/.claude/scheduler/queue_archive.jsonl` and `~/.claude/scheduler/queue.json`,
estimates arrival load over a 30-day window, maps tasks into measured service
buckets, and solves the capacity-slack LP on the four-quadrant measured action
slice.

## Artifacts

```text
algorithm/experiments/production_load_certificate.py
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module49_production_load_strict.md
md/experiment_artifacts/module49_production_load_representative.json
md/experiment_artifacts/module49_production_load_representative.md
```

## Strict Measured-Bucket Result

Strict mapping only counts tasks whose ScheduleurmBench signature directly
matches a measured q00/q01/q10/q11 bucket.

```text
record_count_window = 5159
mapped_task_count = 807
mapped_fraction = 0.156426
unmapped_task_count = 4352
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Estimated load:

| Workload | Lambda |
|---|---:|
| `light_control_local` | 0.794753086 |
| `gpu_heavy_jax_matmul` | 0.064814815 |
| `cpu_heavy_local_bench` | 0.021219136 |
| `hybrid_rl_resac_ant` | 0.014691358 |

Capacity LP:

```text
delta = 0.319917853
status = optimal
supporting profile combo:
  cpu_heavy_local_bench = 6
  gpu_heavy_jax_matmul = 5
  hybrid_rl_resac_ant = 3
  light_control_local = 3
```

## Representative Mapping Result

Representative mapping additionally assigns production RE-SAC/BAPR-like GPU RL
jobs to the measured `hybrid_rl_resac_ant` service bucket and CPU analysis/audit
jobs to the measured local CPU-heavy bucket.  This is useful for diagnosing
capacity but is not by itself theorem-grade bucket equivalence.

```text
record_count_window = 5159
mapped_task_count = 2590
representative_mapped_task_count = 1783
mapped_fraction = 0.502035
strict_mapped_fraction = 0.156426
unmapped_task_count = 2569
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Estimated load:

| Workload | Lambda |
|---|---:|
| `light_control_local` | 0.794753086 |
| `gpu_heavy_jax_matmul` | 0.064814815 |
| `cpu_heavy_local_bench` | 0.045524691 |
| `hybrid_rl_resac_ant` | 0.067777778 |

Capacity LP:

```text
delta = 0.266831433
status = optimal
supporting profile combo:
  cpu_heavy_local_bench = 6
  gpu_heavy_jax_matmul = 5
  hybrid_rl_resac_ant = 3
  light_control_local = 3
```

## Interpretation

This closes a narrower but important question: the observed mapped production
load is comfortably inside the measured four-quadrant service-action slice.

It does not close the full production theorem claim.  The remaining blockers
are empirical coverage blockers:

```text
strict unmapped: 4352 / 5159 tasks
representative unmapped: 2569 / 5159 tasks
representative-mapped but not theorem-grade: 1783 tasks
```

The next global-closure step is therefore not new drift algebra.  It is service
coverage: add measured buckets for the major unmapped CPU workloads and either
prove or measure bucket equivalence for RE-SAC/BAPR representative mapping.
