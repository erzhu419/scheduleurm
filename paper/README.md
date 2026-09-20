# Scheduleurm Operations Research Draft

This directory contains the manuscript using the INFORMS Operations
Research author submission template.

## Reproduce current results and build

From the repository root:

```bash
make reproduce-or-submission
```

This recomputes the registered hardware replay, exports the current numerical
tables, regenerates the four-quadrant comparison figure, aggregates the frozen
cross-domain ledgers into the v7 macros, and compiles both `main.pdf` and
`electronic_companion.pdf`. The final PDF checks cover abstract length, page
limits, and main/EC separation. Each command stops on failure.

To recompute only the result artifacts:

```bash
make reproduce-current-results
```

Both reproduction targets force regeneration even when the submitted outputs
are present. The protocol is fixed in `paper/Makefile`: loads
`0.50,0.70,0.85,0.95`, seeds `41,42,43`, nominal arrival parameter `0.85`, and
16 jobs per workload. Generated files retain their existing artifact names;
the dates identify the input cohort.

For an incremental build from this directory:

```bash
make
```

The Makefile connects frozen inputs and Python sources to the generated JSON,
Markdown, TeX, and comparison figure, then to both PDFs. Missing or changed
result inputs trigger the corresponding generation rule before compilation.
GNU Make 4.3 or newer supports the grouped output rules; the local requirements
are Python, Matplotlib, `latexmk`, a LaTeX/BibTeX installation, and Poppler's
`pdfinfo` and `pdftotext`.

## Frozen inputs

All paths below are under `md/experiment_artifacts/`:

| Consumer | Inputs |
| --- | --- |
| Hardware replay | `service_cache_v2_cpu_gpu_loaded_final_20260809.json`, `critical_gpu_loaded_action_ledger_20260809.json`, `unified_migration_recalibration_certificate_20260809.json` |
| Hardware result exporter | Recomputed replay, `critical_gpu_all_hardware_statewise_slack_20260810.json`, `critical_gpu_loaded_action_ledger_merge_20260809.json` |
| Cross-domain v7 aggregation | `or_generalization_certificate_v6_20260810.json`, `fjsp_solver_action_union_holdout_v1.json`, `port_bacasp_s_external_holdout_r83_r84_20260830.json`, `port_action_union_r81_r82_20260830.json` |

The nine frozen files total about 3 MiB. The existing full replay output is
about 458 MiB and is replaced in place, with a temporary file during writing.
The process runs locally against these inputs.

## Separate verification commands

From the repository root, `make verify-lean` audits and builds the sibling
`../proof` package. `make reproduce-or-submission-legacy` runs the preserved
historical workflow, including environment-dependent probes, and writes its
historical reproduction manifest. `make artifact-manifest` is an alias for
that historical workflow. Record these outcomes separately from current
replay regeneration and the PDF build.

## Limitations

The current entrypoint recomputes replay and statistics from frozen measurements;
it does not launch new hardware measurements. Cross-domain v7 results are
reaggregated from frozen solver ledgers rather than independently rerunning the
original solvers. Lean and historical live evidence are outside this entrypoint.
Successful generation and compilation establish those operations' completion,
not a performance superiority or general stability result.

## Template

The directory vendors official-template files from
`/home/researcher/mine_code/Latex Templete/Operations Research/`:

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
