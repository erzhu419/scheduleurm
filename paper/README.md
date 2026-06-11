# Scheduleurm Operations Research Draft

This directory contains a first manuscript draft using the INFORMS Operations
Research author submission template.

## Build

```bash
make
```

The directory vendors official-template files from
`/home/erzhu419/mine_code/Latex Templete/Operations Research/`:

- `informs4.cls`, used by `\documentclass[opre,dblanonrev]{informs4}` in the
  INFORMS OPRE author submission template, Ver. 1.1, February 21, 2025.
- `eqndefns-left.sty`, loaded by the official template for equation-width
  checking and equation environment definitions.
- `informs2014.bst`, the INFORMS author-year BibTeX style referenced by the
  official template.
- `informs_Logo.pdf`, used by the `informs4` title block.

## Claim Boundary

The draft is intentionally conservative:

- It claims a robust candidate MaxWeight theorem with explicit slack accounting.
- It reports exact measured finite-slice evidence, replay-to-live sanity checks,
  production completed-active coverage, and one live oracle trace.
- It does not claim direct binary execution of Gavel/Pollux/Sia/IADeep/Salus.
- It does not claim generalized fabric-cover constants \(L,\rho\) beyond the
  audited finite slices.
- It does not claim future scheduler dispatches are automatically theorem-grade.

Primary source notes in this repository:

- `md/math.md`
- `md/math_code_alignment_2026_06_11.md`
- `md/or_claim_scope_matrix_2026_06_11.md`
- `md/experiment_module29_task_list_benchmark.md`
- `md/experiment_module48_theorem_condition_calibration.md`
- `md/or_submission_closure_status_2026_06_11.md`
