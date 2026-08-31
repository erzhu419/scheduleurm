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
   target job, completes the target with a phase-aware natural-completion model,
   and credits the resident only with its certified lower service until either
   the measured semi-Markov frame ends or the resident work is exhausted.
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
8. Actions for one workload must share a physical work unit.  Different probe
   horizons are evaluated over the shortest naturally completed horizon, with
   full initialization and terminal output charged once per fresh job.
9. A finite-population tail-drain guard admits a batch width only when the ready
   set and registered unfinished population remain integer-drainable.  When a
   future registered arrival is needed to form a drainable batch, replay waits
   for that arrival instead of launching a remainder-stranding action.

## Fail-closed checks

The final replay gate now requires:

- all four quadrants and all static, Poisson, bursty, and load-sweep protocols;
- one complete policy/arrival/migration matrix;
- phase-aware task-native natural completion for every evaluated action;
- shared-lane simulation and nonoverlapping lane frames in every run;
- finite queue and unfinished-population metrics;
- loaded actions whose required queue coordinates exactly match their
  two-coordinate lower-service vectors;
- migration rows rebound to the exact final cache hash; and
- no loaded or migration action in the legacy policy.

The focused regression suite exercises mixed q01 workloads on shared lanes,
resident/target consumption, completion-blind lower-service selection,
different probe horizons, physical-work-unit rejection, finite-population tail
drain, migration admission, stale-cache rejection, and missing-input failure.
The final four-node empty and loaded measurement gates passed under one
hash-bound service cache, and the resulting schema-v2 replay completed 16,920
policy runs.  Synthetic tests remain regression evidence, not a substitute for
those measurements.

## Claim boundary

Schema-v2 PASS supports a measured-cache, policy-semantics comparison over the
registered hardware and workload action universe.  It does not establish
direct full-stack external-system superiority, organic production stability,
or behavior on future unmeasured workload/hardware states.
