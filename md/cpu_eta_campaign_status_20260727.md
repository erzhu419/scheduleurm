# CPU ETA Campaign Status (2026-07-27)

## Scope and completion-time contract

This campaign replaces history-only CPU ETA with task-native, phase-aware
natural-completion measurements. An admitted completion estimate includes

\[
T_{\mathrm{complete}}
=T_{\mathrm{init}}
+N_{\mathrm{outer}}T_{\mathrm{outer}}
+T_{\mathrm{checkpoint}}
+T_{\mathrm{final\ save}}.
\]

Every row keeps the two physical profile dimensions separate:

- `allocation_workers`: workers allocated to one task;
- `colocation_count`: independent tasks sharing a node.

No history fallback, incomplete progress row, transport failure, or
state-mismatched row enters the theorem-facing cache.

## Current closure

### Native independent-task co-location

The native FreqDuet/SUMO p1, p2, and p4 lower-service gates all pass:

- `native_cpu_colocation_state_targeted_p1_extended_lcb_gate_20260727formal_v6_state_targeted_p1_ext9.json`;
- `native_cpu_colocation_state_targeted_p2_lcb_gate_20260727formal_v6_state_targeted_p2.json`;
- `native_cpu_colocation_state_targeted_p4_merged_lcb_gate_20260727formal_v6_state_targeted_p4.json`.

These rows use the `colocation_count` axis and remain separate from the
single-task worker-allocation experiment. They are now written to cache under
their exact native workload keys, node buckets, and observed effective states:

- p1 on idle nodes: `empty`;
- p2/p4 on idle nodes: `controlled_colocation`;
- p1 under observed external load: `cpu_resident_external`;
- p2/p4 under observed external load: `controlled_colocation_external`.

The independent closure gate reports `PASS` for all
`2 workloads x 6 nodes x 3 profiles = 36` exact tuples. Every tuple has a
natural-completion model, a wave-level simultaneous split-conformal lower
service bound, independent holdout coverage, and no history fallback.

Primary artifacts:

- `native_cpu_colocation_lcb_cache_merge_20260727.json`;
- `native_cpu_colocation_eta_closure_gate_20260727.json`.

### Empty-state allocation-worker curve

The node001-node006 192-core class now has exact natural-completion rows for

`allocation_workers = 1,2,4,8,16,32,64,96,128,180,192`.

Each cell uses three fixed training replicates, nine calibration waves, and
one independent holdout. The simultaneous split-conformal gate reports:

- status: `PASS`;
- maximum point-ETA holdout error: `3.4944%` (limit `10%`);
- simultaneous ratio margin: `17.1893%`;
- all holdout lower-service inequalities: valid;
- missing or duplicate cells: zero.

The old p1 steady-rate-only row was rejected because it had no completion
model. A fresh 13-replicate p1 natural-completion split now supplies startup,
outer-loop, checkpoint, and final-save terms.

Primary artifact:

`native_cpu_allocation_worker_lcb_gate_20260727.json`.

### Controlled half/full resident states

The controlled experiment starts resident work independently of the target,
waits for readiness, performs a fresh five-window state survey, and then times
the target to natural completion.

- half load: 96 resident workers, target p1-p96;
- full load: 180 resident workers, target p1-p8;
- half-load capacity boundary: p128;
- full-load capacity boundary: p16.

The combined gate reports:

- status: `PASS`;
- 12 measured `(state, allocation_workers)` cells;
- three training, nine calibration, and one holdout replicate per cell;
- maximum point-ETA holdout error: `8.8696%` (limit `15%`);
- simultaneous ratio margin: `25.1696%`;
- all holdout lower-service inequalities: valid;
- missing, duplicate, or wrong-node repair cells: zero.

Primary artifact:

`native_cpu_controlled_resident_worker_lcb_gate_20260727.json`.

### Organic light/moderate external states

The observed node004 moderate and node006 light processes remain distinct
from controlled resident load. The gate covers eight safe worker profiles
(p2-p128) in each regime and reports:

- status: `PASS`;
- 16 measured `(regime, allocation_workers)` cells;
- maximum point-ETA holdout error: `12.7411%` (limit `15%`);
- simultaneous ratio margin: `8.1933%`;
- all holdout lower-service inequalities: valid;
- missing, duplicate, or wrong-node repair cells: zero.

These organic rows are attached only to their measured nodes. They are not
replicated across the homogeneous hardware class.

Primary artifact:

`native_cpu_external_state_worker_lcb_gate_20260727.json`.

## Transport correction

Six rows in the original controlled campaign failed with return code 255
because the long-lived SSH route disconnected. They were not workload
failures and were never admitted. Return code 139 remains classified as
shell-encoded `SIGSEGV` and also fails closed.

Natural-completion CPU probes now support `detached_poll` transport:

1. launch the controlled command under remote `setsid`;
2. write output and return code under a run-specific remote directory;
3. poll with short idempotent SSH calls;
4. fetch the complete log only after the atomic return-code file appears.

An SSH interruption can therefore delay observation but cannot terminate the
measurement task. All six 255 cells were rerun at the original state,
replicate, worker count, and node. The detached repairs passed.

The exploratory r12 repair on node003/node005 is retained only as a
cross-node equivalence diagnostic. It is excluded from the formal gate because
the original r12 assignments were node001/node002.

## Cache and factorial inventory

The allocation-worker intermediate cache is

`service_cache_v2_live_merged_cpu_all_state_lcb_20260727.json`.

It contains:

- 66 empty-state worker rows replicated under the declared homogeneous
  node001-node006 hardware contract;
- 72 controlled half/full rows replicated under that same contract;
- 16 organic light/moderate rows restricted to node004/node006;
- 12 state-scoped physical-capacity boundaries.

The final CPU cache is

`service_cache_v2_live_merged_cpu_native_all_state_lcb_20260727.json`.

It adds 36 exact native FreqDuet/SUMO co-location rows. These state-scoped
rows cannot update the legacy `(workload_key, profile)` fallback index.

The unified theorem evidence table contains 75 rows:

`cpu_eta_theorem_evidence_table_20260727.csv`.

The rebuilt factorial inventory is:

`full_factorial_eta_design_cpu_native_all_state_lcb_20260727.json`.

For the node001 192-core representative:

- empty: p1-p192 measured;
- half/96-resident: p1-p96 measured, p128/p180/p192 capacity-closed;
- full/180-resident: p1-p8 measured, p16 and above capacity-closed;
- no feasible controlled profile is pending.

For the native node001/node003 representatives:

- FreqDuet and SUMO p1 are natural-completion measured in `empty`;
- FreqDuet and SUMO p2/p4 are natural-completion measured in
  `controlled_colocation`;
- the old `freqduet_cpu_ablation_c17_32` and `sumo_eval_cpu` keys remain
  separate. Native evidence does not silently close their service-only rows.

The umbrella `cpu_eta_closure_gate_20260727.json` is `PASS`, including the
allocation-worker, controlled-loaded, organic-loaded, native 2x6x3, capacity,
and no-history checks.

## Remaining boundaries

This closes the node001-node006 CPU hardware class, not every CPU in the
cluster. Remaining CPU rows belong to different hardware/transport classes:

- `jtl311linux`: 8-core CPU-fast host, empty p2/p4/p8 and CPU-resident rows;
- `jtl110cpu/jtl110cpu2`: 128-core Windows/processor-group class, empty
  worker curve and CPU-resident rows;
- unknown environments remain `probe_defer_by_design`.

The exact 96/180 resident scenarios are restricted to the 192-core class.
They are not listed as pending on the 128-core host because that would be a
physically invalid design.

The legacy FreqDuet ablation and SUMO service-only keys are not replaced by
the newly measured native keys. Closing them requires their own exact
natural-completion commands or an explicit, separately validated equivalence
map.

The next global stage is to rerun algorithm/SOTA replay and theorem slack
accounting from the final native/all-state cache. It must not reuse any table
built from the earlier history/fallback cache.
