# Round 3 revision actions

This tracks concrete changes made in response to `gpt_revise_round3.md`.

## 1. Downward-closed capacity region

Concern: Queueing capacity should be the downward closure of the stationary service convex hull, not just `conv{μ(a)}`.

Actions:

```text
Added InDownwardCapacityWithSlack.
Added InDownwardCapacityRegion.
Added inCapacityWithSlack_iff_downwardCapacityWithSlack.
Added downward_capacity_monotone.
Added downward_capacity_slack_implies_support_slack.
Added paper-facing wrapper main_downward_capacity_support_slack.
Updated math.md to define Λ as a downward-closed service region.
```

## 2. Positive recurrence wording

Concern: `PositiveRecurrentViaFiniteSet` should not be oversold as full Markov-chain positive recurrence without irreducibility or a closed-class condition.

Actions:

```text
math.md now calls the theorem conclusion a finite-set Foster recurrence certificate.
It states that standard irreducibility / single closed communicating class assumptions are needed to translate the certificate to ordinary countable-state positive recurrence.
```

## 3. Bounded-moment theorem statement

Concern: finite support is good for Lean, but the paper theorem should have a bounded conditional second-order moment version.

Actions:

```text
Added main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound.
Added main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound.
math.md now presents bounded conditional second moment as the general paper condition, with bounded finite-support samples as the constructive Lean specialization.
```

## 4. Lower-service confidence event

Concern: `lower_i(a(Q))≤E[S_i|Q]` must be tied to a confidence/certificate event if lower service comes from learning or posterior inference.

Actions:

```text
Added main_high_probability_stability_from_certificate_event.
math.md now separates deterministic certificate theorem from sampler-specific confidence-event proof.
experimental_open_items.md now explicitly requires lower-service domination probability and feedback/reset assumptions.
```

## 5. State/regime-dependent candidate cover

Concern: `A_full` and `A_cand` may depend on queue state, available jobs, or regime.

Actions:

```text
Added indexed_calibrated_fabric_cover_support_gap.
Added indexed_calibrated_fabric_cover_support_gap_uniform.
math.md now states fixed-family main theorem first and requires indexed/uniform cover certificates for dynamic feasibility.
experimental_open_items.md now asks whether cover is all-feasible, sampled-feasible, historical-observed, statewise, regimewise, or uniform.
```

## 6. Active-bucket learning wording

Concern: active-bucket results are still certificate theorems, not a full sampler theorem.

Actions:

```text
Kept active-bucket learning as extension.
Kept main_active_bucket_local_failure_union_bound as local-to-global probability aggregation.
math.md and experimental_open_items.md do not call this a full online-learning theorem for Scheduleurm.
```
