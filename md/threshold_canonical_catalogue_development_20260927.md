# Threshold canonical quotient: correctness restored, higher search cost

The new selector builds one requirement-group proxy catalogue per minimum
duration threshold. Representatives follow original action order at that
threshold, so both candidate ties and canonical branch bounds use the right
indices. The total LP budget is divided across independent threshold searches;
all calls and cold construction/lifting work are recorded. Historical selectors
and the paper are unchanged.

The registered four-node panel has 500 native-lane states and one known R2
diagnostic. All 1,002 top-level baseline/candidate calls match the independent
Fraction oracle in score, duration, action indices, counts, service, and exact
certificates. The previous 15 node007 action mismatches are now zero. Its R2
`Q=(1,0,4)` action is restored to `(0,8)`. An additional 31 targeted tests pass,
including other group orders, equal minima, idle floors, and failed LP calls.

Main-panel work, excluding the separate R2 diagnostic:

| Counter | Full catalogue | Threshold quotient | Ratio |
| --- | ---: | ---: | ---: |
| LP calls | 6,868 | 23,788 | 3.46 |
| Matrix coefficients | 399,484 | 1,489,503 | 3.73 |
| Candidate evaluations | 7,368 | 27,663 | 3.75 |

The [frozen result](experiment_artifacts/aligned_hardware_threshold_canonical_catalogue_screen_development_20260927.json)
passes the protocol's canonical-correctness gate (`MULTI_CATALOGUE_SCALING_ADVANCE`).
It does **not** establish an efficiency improvement: separate threshold searches
lack a shared incumbent and solve unnecessary local optima. Retain this negative
cost result. Next share the global incumbent and pruning across thresholds, then
repeat this same panel before running larger lane counts.

Reproduce with `python3 -m algorithm.experiments.aligned_hardware_threshold_canonical_catalogue_screen --output /tmp/threshold.json --register`,
then the same command without `--register`.

## Limitations

This is a cached point-model development panel containing a known diagnostic.
It adds no joint95, trajectory, flow-time, wall-time, or independent-confirmation
claim. No larger-lane scaling or scheduler policy change was performed.
