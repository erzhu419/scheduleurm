# Module49 Production-Load Capacity Attempt

Date: 2026-06-08

This module attempts to move from a declared finite-slice load model to a
production-history load certificate.  It reads Scheduleurm task records from
`~/.claude/scheduler/queue_archive.jsonl` and `~/.claude/scheduler/queue.json`,
estimates arrival load over a 30-day window, maps tasks into measured service
buckets, and solves the capacity-slack LP on the measured action slice.

## Artifacts

```text
algorithm/experiments/production_load_certificate.py
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module49_production_load_strict.md
md/experiment_artifacts/module49_production_load_representative.json
md/experiment_artifacts/module49_production_load_representative.md
```

## Strict Measured-Bucket Result

Strict mapping now includes the Module56 production sub-bucket
`freqduet_cpu_ablation_c17_32` and the Module57 exact-config direct-runner
sub-slice `freqduet_runner_v3_allfreq_alllayers_c9_16`.  It still only counts
records backed by a measured service curve or a ScheduleurmBench measured
bucket.

```text
record_count_window = 5176
mapped_task_count = 964
mapped_fraction = 0.186244
unmapped_task_count = 4212
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Estimated load:

| Workload | Count | Lambda |
|---|---:|---:|
| `light_control_local` | 206 | 0.794753086 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `cpu_heavy_local_bench` | 55 | 0.021219136 |
| `hybrid_rl_resac_ant` | 476 | 0.014691358 |
| `freqduet_cpu_ablation_c17_32` | 156 | 0.004333333 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | 1 | 0.000007716 |

Capacity LP:

```text
delta = 0.319917853
status = optimal
```

## Representative Mapping Result

Representative mapping additionally assigns production RE-SAC/BAPR-like GPU RL
jobs to the measured `hybrid_rl_resac_ant` service bucket and CPU analysis/audit
jobs to the measured local CPU-heavy bucket.  These representative assignments
are diagnostic unless backed by separate equivalence or service-measurement
certificates.

```text
record_count_window = 5176
mapped_task_count = 2552
representative_mapped_task_count = 1588
mapped_fraction = 0.493045
strict_mapped_fraction = 0.186244
unmapped_task_count = 2624
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Estimated load:

| Workload | Count | Lambda |
|---|---:|---:|
| `light_control_local` | 206 | 0.794753086 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `hybrid_rl_resac_ant` | 2001 | 0.061759259 |
| `cpu_heavy_local_bench` | 118 | 0.045524691 |
| `freqduet_cpu_ablation_c17_32` | 156 | 0.004333333 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | 1 | 0.000007716 |

Capacity LP:

```text
delta = 0.272849952
status = optimal
```

## Interpretation

This closes a narrower but important question: the observed mapped production
load is comfortably inside the currently measured service-action slice, the
Module56 FreqDuet c17_32 sub-bucket is theorem-grade in strict mapping, and the
Module57 exact direct-runner config is also theorem-grade.

It does not close the full production theorem claim.  The remaining blockers
are empirical coverage blockers:

```text
strict unmapped: 4212 / 5176 tasks
representative unmapped: 2624 / 5176 tasks
representative-mapped but not theorem-grade: 1588 tasks
```

The next global-closure step remains service coverage: add measured buckets for
the major remaining CPU/SUMO/transit workloads and either prove or measure
bucket equivalence for the representative GPU RL and CPU mappings.
