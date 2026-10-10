# Shared threshold search: canonical identity with lower search work

The new selector shares one original-action incumbent and one best-first
frontier across requirement-group minimum-duration strata. Each stratum uses
its own original-index representatives for canonical branch bounds; returned
candidates are lifted at their actual minimum duration. Historical selectors,
the earlier negative cost result, and the paper remain unchanged.

The same four-node panel covers 500 native-lane states plus the named node007
R2 diagnostic. All 1,503 top-level full/independent/shared calls match the exact
full-catalogue oracle in score, duration, original action, counts, service, and
certificates; all budgets are respected. Thirty new tests pass, including
incomplete-search bounds, LP failure charging, representative changes, and
independent small-catalogue enumeration.

Main-panel search work, excluding the diagnostic:

| Counter | Full catalogue | Independent thresholds | Shared thresholds |
| --- | ---: | ---: | ---: |
| LP calls | 6,868 | 23,788 | 6,105 |
| Matrix coefficients | 399,484 | 1,489,503 | 301,729 |
| Candidate evaluations | 7,368 | 27,663 | 6,605 |

Shared search reduces LP calls by 11.1% and matrix coefficients by 24.5% versus
the full catalogue, and LP calls by 74.3% versus independent thresholds. Each
node's aggregate LP and matrix work decreases; none of the 500 states increases
LP calls relative to the full catalogue. Cold setup and lifting counters are
reported separately, without adding heterogeneous operations into a CPU-cost
estimate.

[Frozen result](experiment_artifacts/aligned_hardware_shared_threshold_catalogue_screen_development_20260927.json):
`SHARED_SEARCH_SCALING_ADVANCE`. Next compare fixed-budget search at R4/R8/R16
across these catalogues, preserving the queue scaling and accounting protocol
before execution. No larger-lane run was performed in this step.

Reproduce with `python3 -m algorithm.experiments.aligned_hardware_shared_threshold_catalogue_screen --output /tmp/shared-threshold.json --register`,
then the same command without `--register`.

## Limitations

This is a repeated cached point-model development panel, not independent
confirmation. Search-counter reductions establish neither wall-time nor
flow-time gains and do not transfer the gpu2 joint95 guarantee. No scheduler
policy was deployed.
