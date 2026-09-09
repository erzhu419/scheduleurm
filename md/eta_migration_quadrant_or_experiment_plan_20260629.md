# Scheduleurm ETA, Migration, Quadrant, and OR-Generalization Closure Plan

Date: 2026-06-29

## Objective

Upgrade the current paper artifact from a node007 fresh-ETA measured-cache replay to a hardware-aware, workload-environment-aware, migration-cost-aware scheduling experiment.  The target claim remains scoped and OR-facing:

- A finite measured-service action ledger supports robust MaxWeight-style dispatch under task-native progress-rate certificates.
- Hardware class, task environment, and co-location state are part of the service key.
- Controlled migration is a candidate action only for benchmark tasks with verified checkpoint/resume semantics.
- External systems remain SOTA-style policy families on the same measured service cache, not direct full-stack binary superiority.

## Fixed Hardware Groups

| Group | Nodes | Role | Measurement rule |
|---|---|---|---|
| `gpu_3080ti_12gb_dual` | `jtl110gpu`, `jtl110gpu2` | 2 x 3080Ti 12GB | Fully measure one node; run equivalence checks on the other. |
| `gpu_rtx2080_8gb_dual_cpu_fast` | `jtl311linux` | Fastest CPU, weaker GPU | Fully measure RL, CNN, LLM-feasible, and CPU workloads. |
| `gpu_node007_4x12gb` | `node007-direct` | 4 x homogeneous 12GB GPU | Rebuild the node007 matrix; do not reuse it for other groups. |
| `cpu_hpc_192c` | `node001`-`node006` | homogeneous 192-core CPU nodes | Fully measure one node; validate one additional node. |

## ETA Admission Rule

The theorem-facing service cache only admits rows with task-native progress evidence:

- accepted: `tqdm`, `ScheduleurmStableRate`, or a compatible progress line with units and elapsed time;
- rejected from theorem-facing claims: history-only ETA, no log, no progress line, or unstable warm-up samples;
- reusable: identical `(workload_env, node_bucket, resource_state, profile, command_fingerprint)` rows reuse the cached lower-service certificate.

Rows that fall back to historical ETA remain operational hints only and must be marked `low_confidence/history_fallback`.
Stable-rate admission intentionally skips warm-up samples.  Model initialization, JAX compilation, data loading, checkpoint flush, and logging setup are not service-rate samples.  The current live probes use task-native progress lines for ETA display, but theorem-facing `ScheduleurmStableRate` is computed from explicit task progress/rate lines after a warm-up skip, not from legacy history or transient first-iteration wall time.

## Service Cache v2

The service key is upgraded from `(workload_key, profile)` to:

```text
(workload_env, node_bucket, resource_state, profile)
```

The implementation remains backward compatible: old callers may still query `(workload_key, profile)`, but theorem dispatch must prefer the v2 key.

Required workload environments:

- RL/hybrid: `ant`, `halfcheetah`, `hopper`, `walker2d`, plus unknown RL as `unknown_rl_env`;
- GPU: CNN and LLM workload families;
- CPU: FreqDuet, CPU synthetic, SUMO-like CPU traces;
- OR-generalization: port, flexible job-shop scheduling problem (FJSP), resource-constrained project scheduling problem (RCPSP), and multi-mode RCPSP (MMRCPSP).

Required resource states:

- `empty`;
- `half_loaded`;
- `full_loaded`;
- `high_vram_resident`;
- `cpu_resident`;
- `mixed_colocation`;
- `unspecified` only for legacy or non-theorem rows.

## Action Space

The theorem-facing action family has three action types:

```text
launch(task,node,resource,profile)
keep(running_task,current_node)
migrate(task,from_node,to_node,checkpoint_policy,sync_policy,resume_policy)
```

Every row is scored by:

```text
Q^T lower_service - K_migration - G_risk
```

`migrate` is eligible only when all of the following hold:

- the task is controlled by the benchmark harness;
- checkpoint and resume commands are known;
- source and destination environments are staged or staging cost is explicitly measured;
- the task is not a normal user production job.

## Migration Cost Decomposition

For each controlled migration row, the cost certificate records:

- checkpoint flush time;
- sync or rsync time;
- environment staging time;
- resume warm-up time;
- lost-work time;
- failure/risk penalty.

Migration points are measured at 25%, 50%, and 75% progress.  With remaining
work `W_rem`, old service rate `mu_old`, and destination lower service rate
`mu_new_lcb`, a migration is beneficial only when the completion-time saving
exceeds the migration and risk cost:

```text
W_rem / mu_old - W_rem / mu_new_lcb > K_migration + G_risk
```

The earlier product-of-work-and-rate expression was dimensionally invalid and
must not be used in theorem-facing migration decisions.

This is a theorem-compatible bounded penalty, not an uncontrolled scheduler side effect.

## Workload Families To Probe

| Family | Examples | Migration status |
|---|---|---|
| Pure GPU | CNN benchmark, small LLM inference/training surrogate | checkpointable benchmark only |
| Hybrid RL | RE-SAC/BAPR-style Ant, HalfCheetah, Hopper, Walker2d | short controlled jobs |
| Pure CPU | FreqDuet-like CPU benchmark or real FreqDuet if safely resumable | checkpointable surrogate if real resume is unsafe |

## Four-Quadrant SOTA Figure

The main comparison figure should be a 2 x 2 grid:

- q00: low CPU, low GPU;
- q01: low CPU, high GPU;
- q10: high CPU, low GPU;
- q11: high CPU, high GPU.

Each panel plots every registered SOTA-style policy plus Scheduleurm:

- x-axis: normalized mean-flow cost, `policy / ours`;
- y-axis: normalized makespan cost, `policy / ours`;
- Scheduleurm is fixed at `(1,1)`;
- no legacy baseline;
- no dual axis;
- annotate the best SOTA family in each quadrant to show that the SOTA winner changes by quadrant.

## OR Generalization Experiments

The server scheduler is treated as one instance of a broader finite-action stochastic processing problem.

| Domain | State/action analogue | Migration analogue |
|---|---|---|
| Port terminal | berth/quay/yard/gate workload and machine heterogeneity | re-berthing, crane reassignment, yard rehandle |
| FJSP | operation-machine processing-rate matrix | rerouting a remaining operation to another eligible machine |
| RCPSP/MMRCPSP | activity-mode-resource choices | mode switch with setup or rework cost |

These experiments are replay/simulator gates, not empirical claims about physical ports.

## Measurement Gates

- `stable_rate_ready=true`;
- `eta_source=tqdm/progress`;
- `stable_skip_samples>=1` for fresh live probes unless the workload is already known to have no initialization transient;
- no history fallback;
- node-aware service lookup differentiates `jtl311linux`, `node007-direct`, and `jtl110gpu`;
- `jtl110gpu` and `jtl110gpu2` equivalence row passes tolerance;
- RL rows include Ant, HalfCheetah, Hopper, Walker2d for at least p1/p2/p4 or until measured capacity boundary.

## Algorithm Gates

- node-aware lookup selects `jtl311linux` for HalfCheetah when its lower-service row dominates;
- beneficial controlled migration is selected;
- high-cost migration is rejected;
- missing checkpoint/resume excludes the migration action;
- ordinary user running tasks are never migration candidates.

## Replay And Paper Gates

- rebuild service cache v2;
- rerun online arrivals, SOTA candidate union, quadrant Pareto, strict frontier, and slack accounting;
- redraw the four-quadrant SOTA figure;
- build the PDF;
- generate a corner-case coverage matrix and a claim-boundary report.

## Corner Cases

| Corner case | Required status |
|---|---|
| empty GPU | measured or explicitly pending |
| half-loaded GPU | measured or explicitly pending |
| full-loaded GPU | measured or explicitly pending |
| high VRAM resident | measured or explicitly pending |
| CPU resident | measured or explicitly pending |
| mixed CNN/RL/LLM | measured or explicitly pending |
| `jtl311linux` CPU-fast/GPU-weak | separate service rows |
| `node007-direct` multi-GPU packing | separate service rows |
| `node001`-`node006` CPU packing | separate service rows |
| unknown environment | probe/defer, no theorem claim |
| description-only RL environment | parser must extract env if possible |
| missing `tqdm` | history fallback, no theorem claim |
| missing checkpoint | migration excluded |
| local-only resume path | migration excluded unless staged |
| migration at 25/50/75% | cost rows or pending label |

## Execution Order

1. Land the v2 service schema and node/env/resource-state inference.
2. Add controlled migration action rows and tests.
3. Add dry-run/live manifests for cross-node ETA and migration probes.
4. Add OR generalization simulator gates.
5. Redraw the four-quadrant SOTA figure and update the manuscript.
6. Run focused tests and build the PDF.
