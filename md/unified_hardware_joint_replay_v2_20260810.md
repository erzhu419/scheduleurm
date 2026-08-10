# Unified Hardware Joint Replay v2

## Why v2 was required

The first unified replay grouped actions by hardware-local scenario but then
simulated each workload coordinate independently.  Each workload therefore saw
the full node lane count, so a mixed CNN/LLM/RL scenario did not enforce shared
GPU contention.  It also interpreted a directional loaded action with physical
profile 2 as two target jobs, although the measured trajectory contains one
resident and one target job.  Those two semantics could bias portfolio JCT,
makespan, backlog, loaded-action, and policy-comparison results.

No v1 performance number is eligible for the final paper after this finding.
The task-native measurement artifacts themselves are unaffected.

## Corrected replay contract

`algorithm/experiments/unified_hardware_or_replay.py` now emits schema version
2 and applies these rules:

1. All workload classes in one hardware-local scenario compete for the same
   physical CPU/GPU lanes in one event-driven simulation.
2. A dedicated profile-p action requires and dispatches p target jobs on one
   lane.  A directional loaded action requires exactly one resident and one
   target job, completes the target with its natural-completion model, and
   credits the resident only with its certified lower service during that
   measured semi-Markov frame.
3. A resident that does not finish during the loaded frame returns to the
   schedulable queue with reduced remaining work.  It cannot be selected by a
   second lane while the frame is active.
4. Migration actions remain one-job controlled actions.  The selector sees the
   risk-adjusted effective lower service; completion evaluation uses the
   physically measured migration delay and target natural-completion model.
5. MaxWeight scoring uses a fixed positive diagonal normalization.  Queue work
   and every service coordinate are divided by the same task-native canonical
   work unit for that workload, so CNN steps, LLM steps, RL iterations, and CPU
   units are never compared as raw magnitudes.
6. Queue backlog is integrated from arrival, dispatch, and resident-return
   events.  Unfinished system population is reported separately.  Per-lane
   frame audits must be nonoverlapping.
7. The legacy row uses historical fixed profile semantics on the same final
   measured cache and cannot consume loaded or migration actions.  It is not a
   fresh execution of the default legacy scheduler.

## Fail-closed checks

The final replay gate now requires:

- all four quadrants and all static, Poisson, bursty, and load-sweep protocols;
- one complete policy/arrival/migration matrix;
- task-native natural completion for every evaluated action;
- shared-lane simulation and nonoverlapping lane frames in every run;
- finite queue and unfinished-population metrics;
- loaded actions whose required queue coordinates exactly match their
  two-coordinate lower-service vectors;
- migration rows rebound to the exact final cache hash; and
- no loaded or migration action in the legacy policy.

The focused synthetic regression suite exercises mixed q01 workloads on shared
lanes, resident/target consumption, completion-blind lower-service selection,
migration admission, stale-cache rejection, and missing-input failure.  The
final numerical replay must still wait for all four hardware-local empty and
loaded measurement gates; this document does not substitute synthetic results
for those measurements.

## Claim boundary

Schema-v2 PASS supports a measured-cache, policy-semantics comparison over the
registered hardware and workload action universe.  It does not establish
direct full-stack external-system superiority, organic production stability,
or behavior on future unmeasured workload/hardware states.
