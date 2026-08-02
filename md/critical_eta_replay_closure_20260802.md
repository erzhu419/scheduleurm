# Critical ETA and Replay Closure, 2026-08-02

## Decision on the old pending matrix

The old `168 pending_probe` count is not a required execution checklist. The
latest exhaustive design contains 179 pending cells because it includes
unknown, dominated, duplicate-hardware, migration, and non-policy-reachable
states. Completing that Cartesian product would not strengthen the current OR
claim. It would mix exploratory coverage with the finite action family actually
used by Scheduleurm and the registered SOTA-style policies.

The strict closure population is the following 14 logical actions:

| Quadrant | Workload | Policy-reachable profiles |
|---|---|---|
| q00 | `light_control_local` | p1, p13 |
| q01 | `gpu_heavy_jax_matmul` | p1, p3 |
| q01 | `gpu_cnn_torch_resnet50` | p1, p3 |
| q01 | `gpu_llm_distilgpt2` | p1, p3, p10 |
| q10 | `cpu_heavy_local_bench` | p1, p8, p9 |
| q11 | `hybrid_rl_resac_ant` | p2, p5 |

For GPU actions, each feasible profile must be measured separately on the
3080Ti, 2080, and node007 2080Ti hardware classes; an infeasible profile must
produce a scoped capacity boundary. This is 27 GPU hardware/action cells before
the small homogeneous-node equivalence audit. They are not replaced by old
node007 rates.

## CPU closure completed in this run

The q00 and q10 measurements bypassed legacy scheduler admission limits and
used only controlled benchmark processes. Every task ran to natural exit with
task-native durable CSV/tqdm progress. Each task wrote and `fsync`ed four real
8 MiB checkpoint files plus its final artifact. Remote `stat`/`find` audits
showed 8 MiB file size and 16,384 allocated 512-byte blocks, so these were not
sparse-file timings.

The pre-registered split was:

- r01-r03: training;
- r04-r12: split-conformal calibration;
- r13: untouched lower-service/single-wave point holdout;
- r14-r18: prospective expected-completion validation for q10 only.

The exact run-window state and external-load bucket are now part of the
calibration signature. A production task entering during a measurement cannot
silently remain in the idle cell.

### Results

| Evidence | Result |
|---|---:|
| q00 simultaneous lower-service gate | PASS |
| q00 single-wave point ETA maximum error | 0.58% |
| q10 simultaneous lower-service component | PASS |
| q10 original single-wave median point gate | FAIL, 14.21% maximum |
| q10 prospective expected-completion gate | PASS |
| q10 prospective expected-completion maximum error | 3.12% |
| Exact q00/q10 cache rows inserted | 10 |
| Legacy two-key cache changed by these rows | no |
| Static/Poisson q00/q10 SOTA-policy replay | PASS |

The q10 failure is retained rather than hidden or repaired by resampling. It
showed that p1 completion time has material stochastic I/O/frequency variation.
The revised estimand is the expected group natural-completion makespan used by
replay, fitted from r01-r12 and tested on five future complete waves. It does
not claim that every individual task finishes within 10% of its pre-launch
estimate. Live ETA must still update from the task's own progress.

The strict q00/q10 replay selected Scheduleurm actions on calibrated lower
service and evaluated the frozen actions on a separate natural-completion point
view. Across static and Poisson arrivals, Scheduleurm was not Pareto dominated,
but all candidate-to-best-SOTA ratios were exactly 1.0. This is a correctness
and comparability closure, not evidence of superiority.

## Reuse policy for previous measurements

- Stable task-native service rows may remain in lower-service and theorem-slack
  calculations when their exact workload, hardware, load state, and profile
  match.
- Existing native FreqDuet/SUMO natural-completion records and homogeneous CPU
  equivalence evidence remain valid.
- A stable-rate-only row cannot be presented as JCT/ETA.
- A history-fallback row cannot enter theorem-facing replay.
- A node007 GPU row cannot stand in for jtl110gpu or jtl311linux.
- Unknown environments, dominated/unreachable profiles, and duplicate full
  measurements on homogeneous replicas do not need to be run.

## Resource-blocked remainder

All GPU classes were occupied during this closure. Therefore the nine logical
GPU actions, 27 hardware/action cells, homogeneous replica audit, full
four-quadrant replay, and final slack recomputation remain pending. No GPU probe
was launched, no old ETA was promoted, and no heartbeat was installed. The next
run should begin from these GPU cells, then regenerate every SOTA/Pareto replay
and theorem-slack table from the final exact cache.

## Artifacts

- `critical_cpu_colocation_actualio_r01_20260802.json` through r18
- `critical_light_colocation_actualio_r01_20260802.json` through r13
- `critical_cpu_colocation_lcb_gate_20260802.json`
- `critical_light_colocation_lcb_gate_20260802.json`
- `critical_cpu_expected_completion_prereg_20260802.json`
- `critical_cpu_expected_completion_gate_20260802.json`
- `critical_checkpoint_allocation_audit_20260802.json`
- `critical_cpu_phase_cache_merge_20260802.json`
- `service_cache_v2_critical_phase_20260802.json`
- `critical_q00_q10_phase_sota_replay_20260802.json`
