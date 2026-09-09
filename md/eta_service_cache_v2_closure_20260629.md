# ETA Service-Cache v2 Closure - 2026-06-29

## What Was Fixed

This round closed two cache-indexing bugs that could make the measured-cache
replay look artificially close or artificially weak.

1. Scoped live-v2 capacity boundaries.
   - `ServiceRateCache.add()` used to let a capacity boundary for one node block
     valid rows for another node with the same `(workload_key, profile)`.
   - This was wrong for cross-node ETA.  Example: `jtl311linux` CNN p4 is an
     unstable/capacity row, while `node007-direct` CNN p4 is a valid stable row.
   - Fix: only live-v2 service-cache boundaries are scoped by node; historical
     theorem-population boundaries remain global conservative caps.

2. State-specific rows no longer overwrite the legacy profile index.
   - `ServiceRateCache.add()` used to write every non-boundary row into the
     legacy `(workload_key, profile)` index.
   - This meant a `mixed_colocation` CNN p3 row could replace the ordinary CNN
     p3 empty-state service curve.
   - Fix: only `empty` / `unspecified` rows may update the legacy index.
     Resident-load and mixed co-location rows are available only through
     statewise lookup.

Regression tests added:

- `test_scoped_capacity_boundary_does_not_block_other_node_valid_profile`
- `test_state_specific_row_does_not_overwrite_legacy_profile_index`

Validation:

```text
python3 -m pytest skill/tests/test_eta_migration_v2.py skill/tests/test_simulation_tasksets.py skill/tests/test_simulation_fast_forward.py -q
19 passed
```

## Current ETA Coverage

Coverage artifact:

- `md/experiment_artifacts/eta_server_coverage_gate_20260629.json`
- `md/eta_server_coverage_gate_20260629.md`

Current status:

```text
Status: ETA_SERVER_COVERAGE_REQUIRED_READY
Pass: true
Measured rows: 25
Partial rows: 0
Missing rows: 0
Required missing rows: 0
```

Theorem-facing ETA rows use task-native `tqdm` / `ScheduleurmStableRate`.
History fallback rows are not admitted.

Measured hardware/load coverage now includes:

- `jtl110gpu:gpu_3080ti_12gb_dual`
  - CNN empty p1/p2, half-loaded p2, full-loaded p4, high-VRAM-resident p1,
    mixed co-location p2/p3.
  - LLM empty p1/p2.
  - RE-SAC Ant p1/p2, HalfCheetah p1/p2, Hopper p1, Walker2d p1.
- `jtl110gpu2:gpu_3080ti_12gb_dual`
  - CNN empty p1/p2.
  - LLM empty p1.
  - Used as homogeneous equivalence validation, not as a separate full matrix.
- `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast`
  - CNN empty p1/p2 and half-loaded p2.
  - LLM empty p1.
  - RE-SAC Ant p1, HalfCheetah p1/p2, Hopper p1, Walker2d p1.
- `node007-direct:gpu_node007_4x12gb`
  - CNN empty p1/p2/p4.
  - LLM empty p1.
  - RE-SAC Ant p1.
  - Note: the bucket name is historical; the live `nvidia-smi` query reports
    4 x RTX 2080 Ti with about 11 GB visible memory per GPU.
- `node001/node003/node005:cpu_hpc_192c`
  - CPU resident add-one rows for the CPU-heavy benchmark family.

Important lookup checks after rebuilding the merged cache:

```text
legacy gpu_cnn_torch_resnet50 p3: None
statewise CNN mixed_colocation p3 on jtl110gpu: 20.604914 step/s
node007 CNN empty p4: 28.5847268525 step/s
node007 LLM empty p1: 327.867544 step/s
node007 RE-SAC Ant empty p1: 0.2375653725 iter/s
jtl311 RE-SAC Hopper empty p1: 0.343413033 iter/s
jtl311 RE-SAC Walker2d empty p1: 0.328961607 iter/s
```

## Migration Cost Status

Migration cost artifact:

- `md/experiment_artifacts/live_checkpoint_migration_cost_gate_20260629.json`
- `md/live_checkpoint_migration_cost_gate_20260629.md`

Current status:

- Controlled pure-GPU CNN migration: measured.
- Controlled hybrid RL HalfCheetah migration: measured.
- Controlled CPU/FreqDuet surrogate migration: measured.
- Costs are decomposed into checkpoint/sync/resume/lost-work risk pieces.
- Ordinary user running tasks remain no-touch.

This closes controlled migration-cost evidence, not production-wide arbitrary
running-task migration.

## SOTA Replay After Corrected ETA Cache

After rebuilding the live-v2 merged cache, the SOTA policy-semantics replay was
rerun with:

```text
--live-cache-json md/experiment_artifacts/service_cache_v2_live_merged_20260629.json
```

Artifacts:

- `md/experiment_artifacts/sota_candidate_union_gate_live_v2_20260629.json`
- `md/experiment_artifacts/sota_strict_dominance_frontier_live_v2_20260629.json`
- `md/experiment_artifacts/sota_quadrant_pareto_gate_live_v2_20260629.json`

Current honest status:

```text
sota_candidate_union_gate: SOTA_CANDIDATE_UNION_NEEDS_REVIEW
sota_strict_dominance_frontier: MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_OPEN
sota_quadrant_pareto_gate: SOTA_QUADRANT_TOLERANCE_OPEN
```

The main strict frontier rows are:

1. `q01_gpu_bound_cnn_resnet50`, static arrivals.
   - Diagnosis: same measured CNN profile but different statewise drain
     trajectory; current bridge candidates improve flow but miss the strict
     makespan guard.

2. `q11_cpu_gpu_coupled`, static arrivals.
   - Diagnosis: best makespan and best mean-flow come from different policy
     families.

3. `hybrid_research_portfolio`, static arrivals.
   - Diagnosis: hybrid RL profile p3 minimizes makespan while p2 minimizes flow.

Therefore old Pareto/SOTA numbers generated before this cache fix should not be
used as final paper evidence.  The current clean claim is ETA coverage closure
plus measured-cache policy-semantics diagnostics; strict SOTA dominance still
needs additional trajectory/action bridge candidates or new service points.

## Next Work If We Continue Attacking Strict SOTA Dominance

1. Add a CNN tail-drain / co-location trajectory action that improves flow
   without violating the finish-time makespan envelope.
2. Add a q11 interpolation action between the current makespan-best and
   mean-flow-best policy families.
3. Add a hybrid RL p2/p3 bridge or critical-path-aware p3-to-p2 switch with a
   verified makespan guard.
4. Regenerate the four-quadrant SOTA figure only after these gates pass or after
   the paper claim is explicitly scoped to the current open-frontier evidence.
