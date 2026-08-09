# Critical GPU Host-CPU Exposure Audit Policy

Date: 2026-08-09

## Scope

This policy amends only the post-measurement contamination audit. It does not
change the task commands, progress protocol, natural-completion model, GPU
monitor, workload totals, or the measurement-code hash
`1b42b9b7e37dae96f6dd3097e3796a2dd8c7b8b123608ecda126c0b8dcd31666`.

The former audit rejected any single user-process snapshot whose normalized
CPU use exceeded 5% of host capacity. That rule correctly exposed two
previously invisible transients, but it did not distinguish an isolated burst
from sustained host contention.

## Frozen Rule

The revised audit keeps the 5% per-snapshot trigger and retains every triggering
process in the certificate. It does not add `sha256sum`, `python3`, or any other
compute command to the allowlist.

For each consecutive group of triggering process snapshots, the process is
conservatively treated as present from the preceding process snapshot through
the following process snapshot. The burst contribution is

```text
peak normalized host CPU fraction * conservative burst duration
--------------------------------------------------------------- .
                    target duration
```

Contributions from disjoint bursts are summed. A row passes the host-process
contamination gate only when this conservative exposure is at most `0.001`
(0.1% of full-host CPU time over the target interval). GPU contamination,
aggregate host load, memory, missing samples, and uncontrolled GPU process
groups remain independent fail-closed gates.

## Pre-Amendment Diagnostics

- `jtl110gpu`, wave 2, `rl_after_cnn`: one `sha256sum` snapshot; conservative
  exposure `0.0009648394301405873`.
- `jtl110gpu2`, wave 3, `rl_after_cnn`: one `python3` snapshot; conservative
  exposure `0.0013578129053773607`.

These two pre-amendment waves are diagnostic only. The first would pass the new
rule and the second would fail it, but neither is admitted into the final cache.
Both waves must be rerun after this policy is committed. This prevents a
post-outcome gate change from converting an already observed row into formal
evidence.

## Claim Boundary

Passing this audit certifies bounded observed host-CPU contamination under the
hash-bound sampling protocol. It does not prove that the host was process-free,
identify an ended transient's full command line, or justify ignoring sustained
background work.
