# Lean Verification Log For Submission Packaging

Date: 2026-06-13, Asia/Shanghai.

Run directory:

```text
/home/erzhu419/mine_code/proof
```

Artifact identity:

```text
$ sha256sum ScheduleurmUpload.lean
25f0c744fb10fe8a9c3399f5072d5d16a81084082ab0264e92e9e7c8d08739c2  ScheduleurmUpload.lean

$ wc -l ScheduleurmUpload.lean
6227 ScheduleurmUpload.lean

$ git rev-parse HEAD
23b101432067cc005512f7667810ec03b8cffb77

$ date -Is
2026-06-13T01:26:21+08:00
```

Commands and results:

```text
$ lake build Scheduleurm
Build completed successfully (8054 jobs).

$ lake env lean ScheduleurmUpload.lean
<no output; exit code 0>

$ rg -n "\bsorry\b|\badmit\b|\baxiom\b" Scheduleurm ScheduleurmUpload.lean lakefile.toml || true
<no output>
```

Submission-facing theorem names confirmed searchable in `ScheduleurmUpload.lean`:

```text
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle: 1
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle: 1
main_diagonal_scaled_lcb_support_loss: 1
main_diagonal_scaled_lcb_support_loss_l1: 1
main_high_probability_stability_from_certificate_event: 1
main_operational_capacity_sandwich: 1
main_robust_candidate_maxweight_drift_approx_oracle: 1
QueueScaledApproxRobustScoreMaximizer: 9
main_candidate_restricted_capacity_coordinate_hausdorff_from_calibration: 1
main_active_bucket_lcb_learning_regret_high_probability: 1
```

Interpretation:

```text
The consolidated upload artifact is Lean-checkable, the split Scheduleurm
proof build succeeds, the diagonal-scaled LCB theorem names needed by the
selected-profile stochastic lower-service certificate are present, and the
theorem names needed by math.md / artifact map are searchable in the upload
file. No sorry/admit/axiom token was found.
```
