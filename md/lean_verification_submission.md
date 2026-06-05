# Lean Verification Log For Submission Packaging

Date: 2026-06-05, Asia/Shanghai.

Run directory:

```text
/home/erzhu419/mine_code/proof
```

Artifact identity:

```text
$ sha256sum ScheduleurmUpload.lean
08b661a71a10ceffcb8df719319e1a6f80801821e226e6c54036624a65fac777  ScheduleurmUpload.lean

$ git rev-parse HEAD
59bda6ab1b4f20fc1b6f67b151704cc300892ba7

$ date -Is
2026-06-05T12:27:02+08:00
```

Commands and results:

```text
$ lake build Scheduleurm
Build completed successfully (8053 jobs).

$ lake env lean ScheduleurmUpload.lean
<no output; exit code 0>

$ rg -n "\bsorry\b|\badmit\b|\baxiom\b" Scheduleurm ScheduleurmUpload.lean lakefile.toml || true
<no output>
```

Submission-facing theorem names confirmed searchable in `ScheduleurmUpload.lean`:

```text
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle: 1
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle: 1
main_high_probability_stability_from_certificate_event: 1
main_operational_capacity_sandwich: 1
main_robust_candidate_maxweight_drift_approx_oracle: 1
QueueScaledApproxRobustScoreMaximizer: 9
```

Interpretation:

```text
The consolidated upload artifact is Lean-checkable, the split Scheduleurm
proof build succeeds, and the theorem names needed by math.md / artifact map
are present in the upload file. No sorry/admit/axiom token was found.
```
