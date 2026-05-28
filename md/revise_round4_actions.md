# Round 4 revision actions

This tracks concrete changes made in response to `gpt_revise_round4.md`.

## 1. Artifact consistency and hashable verification

Concern: the reviewed consolidated upload file did not contain the theorem names listed in the artifact map/log.

Actions:

```text
Regenerated /home/erzhu419/mine_code/proof/ScheduleurmUpload.lean after round4 proof edits.
Verified that round4 theorem names are searchable in ScheduleurmUpload.lean.
Added lean_verification_round4.md with sha256, git commit, timestamp, build result, upload-file Lean check, and sorry/admit/axiom grep.
```

## 2. Approximate optimization oracle

Concern: real schedulers may use greedy search, local search, or time-limited ILP rather than exact robust candidate argmax.

Actions:

```text
Added ApproxRobustScoreMaximizer.
Added QueueScaledApproxRobustScoreMaximizer.
Added robust_candidate_policy_approx_full_support_scaled_penalty_approx_oracle.
Added robust_candidate_policy_lyapunov_drift_scaled_penalty_approx_oracle.
Added main_robust_candidate_maxweight_drift_approx_oracle.
Added main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle.
Added main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle.
```

The theorem now charges oracle error as:

```text
bounded part α0 -> additive drift constant
queue-scaled part α1 ||Q||1 -> slack consumption
```

The stability condition becomes:

```text
δ > Lρ + ε_est + β + α1
```

## 3. Paper wording

Actions:

```text
math.md now says the policy approximately maximizes candidate robust score.
math.md now states finite-set Foster recurrence certificate, not unconditional full positive recurrence.
experimental_open_items.md now includes α0 and α1 calibration obligations.
lean_artifact_map.md now includes approximate-oracle theorem names.
```
