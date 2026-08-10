# OR Generalization Regression Audit (2026-08-10)

## Scope

This audit re-runs the complete local regression surface whose test module name
starts with `test_port`, `test_fjsp`, `test_mmrcpsp`, or
`test_or_generalization`. It is independent of the in-progress GPU ETA
campaign and does not launch remote work.

## Command and result

```bash
files=$(rg --files skill/tests | rg '/test_(port|fjsp|mmrcpsp|or_generalization)')
python3 -m pytest -q -p no:cacheprovider $files
```

- Test files: 36
- Tests passed: 203
- Tests failed: 0
- Elapsed time: 229.53 seconds

## Evidence retained

- Port scheduling: the public BACASP-S source adapter, source-core feasibility,
  disjoint public holdout, four-resource statewise transition semantics,
  synthetic migration analogues, finite candidate construction, exact
  registered-family oracle checks, and the variable-duration deterministic
  frame bridge all pass their regression contracts.
- Flexible job-shop scheduling problem (FJSP): public-instance byte locks,
  feasible-machine and precedence checks, exact completion ledgers, frozen
  family and Hurink holdouts, CP-SAT comparison accounting, and renewal-frame
  mappings pass.
- Multi-mode resource-constrained project scheduling problem (MMRCPSP): public
  instance and manifest checks, renewable/nonrenewable resource feasibility,
  disjoint holdouts, deterministic trajectory families, static/Poisson/bursty/
  load-sweep renewal streams, and semantic reproducibility checks pass.
- The aggregate OR generalization certificate retains every registered
  negative result and keeps structural protocol validity separate from
  empirical superiority.

## Claim boundary

The result certifies reproducibility, feasibility, exact finite-family
selection, and the stated theorem-interface mappings for the registered
instances. It does not certify global FJSP/MMRCPSP/port optimality, universal
performance superiority, industrial port deployment, or physical port
positive recurrence. The server ETA replay remains separately gated on the
four hardware-local natural-completion and loaded-state certificates.
