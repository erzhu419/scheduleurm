# Round 2 revision actions

This tracks the concrete changes made in response to `gpt_revise_round2.md`.

## 1. Lean artifact / `math.md` consistency

Concern: `math.md` cited new `main_*` theorem names, but a reviewer inspecting the upload file could not find them.

Action:

```text
Added md/lean_artifact_map.md with a searchable mapping from paper claims to theorem names.
Added a paper-facing Lean theorem wrapper:
  main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
Regenerated ScheduleurmUpload.lean after the Lean changes.
Updated math.md to point reviewers to md/lean_artifact_map.md.
```

## 2. Collapse the theory into one main theorem

Concern: A/B/C/D/E/F/G are useful internally, but the main paper needs one readable theorem.

Action:

```text
math.md Section 7 now states a single paper theorem:
  full-action support slack
  + fabric-cover candidate loss Lρ
  + lower-service estimation loss ε_est
  + queue-scaled penalty β
  + bounded finite-support stochastic model
  => positive recurrence.

Lean theorem:
  main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
```

## 3. Keep capacity geometry from becoming the claimed novelty

Concern: `conv{μ(a)}` is a standard SPN base, not the paper's main innovation.

Action:

```text
math.md Section 10 now lists the three main contributions as:
1. global configuration-action SPN model;
2. candidate fabric-cover approximation theorem;
3. robust candidate MaxWeight stability theorem.
```

## 4. Make `d_Φ`, `L`, and `ρ` falsifiable

Concern: Lipschitzness and cover radius cannot be free assumptions.

Action:

```text
md/experimental_open_items.md now requires:
  telemetry source for each feature Φ_r;
  feature weights w_r;
  empirical L envelope;
  perturbation experiments;
  ρ estimation domain;
  handling of discontinuous co-location effects.
```

Additional proof added after audit:

```text
FabricCandidateProjection.covers
fabric_service_lipschitz_of_feature_sensitivity
calibrated_fabric_cover_support_gap
main_candidate_restricted_capacity_approximation_from_calibration
main_candidate_restricted_capacity_coordinate_hausdorff_from_calibration
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric
```

These theorems do not invent the empirical constants, but they prove that
once profiling supplies a feature sensitivity envelope and the candidate
generator supplies a projection radius, the paper's `Lρ` loss follows
formally.

## 5. Keep hidden regimes and learning as extensions

Concern: Average-regime stability and structured learning are not ready to be the main OR theorem.

Action:

```text
math.md Section 10 explicitly places:
  uniform hidden-regime stability / dwell-switching budget,
  average-regime stability,
  structured active-bucket learning,
  sweet-spot threshold
as extension material.

md/experimental_open_items.md now spells out the feedback model needed for active-bucket learning.
```

Additional proof added after audit:

```text
active_finset_all_events_failure_probability
main_active_bucket_local_failure_union_bound
```

This proves the finite active-bucket union-bound step from local bucket
failure probabilities to the all-active-bucket event.
