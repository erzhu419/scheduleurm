# Lean verification log after round 3

Run directory:

```text
/home/erzhu419/mine_code/proof
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

Key round3 theorem names confirmed searchable in `ScheduleurmUpload.lean`:

```text
main_downward_capacity_support_slack
main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound
main_high_probability_stability_from_certificate_event
indexed_calibrated_fabric_cover_support_gap
```
