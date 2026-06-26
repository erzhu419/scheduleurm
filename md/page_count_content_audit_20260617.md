# Page Count and Content Audit, 2026-06-17

Baseline used: `paper/main_revise_supervisor.pdf` because it is the 38-page
annotated supervisor version present in the repository.

Current file: `paper/main.pdf`, rebuilt after moving Related Work forward,
pinning Table 1 before Related Work, and restoring the compact-but-complete
supplementary evidence summaries in Tables 16--17.

## Counts

| Version | Pages | Extracted text chars | Extracted word-like tokens | Table-title mentions |
|---|---:|---:|---:|---:|
| Supervisor annotated baseline | 38 | 140,389 | 12,175 | 27 |
| Current rebuilt manuscript | 34 | 120,754 | 10,718 | 27 |

The 4-page reduction is mostly from removing or compressing process-style
artifact material and from tighter float placement after moving Related Work.
It is not caused by removing a main theorem, experiment family, reference
family, or appendix proof section.

## Preserved Main Structure

The following scientific structure is still present:

- Claim scope and evidence matrix.
- Related Work, now moved forward after Scope and Contributions.
- Model: notation, configuration actions, capacity/support functions, candidate
  families, and fabric metrics.
- Robust Candidate MaxWeight theorem section: policy, assumptions,
  deterministic slack accounting, and proof decomposition.
- Scheduleurm instantiation: measured action slices and theorem evidence
  populations.
- Computational evidence: explicit task lists, online replay, policy-semantics
  baselines, theorem-condition/live-sanity checks, and production artifacts.
- Discussion and limitations.
- Conventional proof appendix EC.1--EC.6.
- Code and Data Availability.

## Preserved Tables

Both versions contain the manuscript-facing Table 1--Table 17 sequence.  Current
Table 16/17 are not the old long gate-ladder tables; they are expanded
supplementary evidence-gate summary tables.

Preserved or replaced table roles:

- Tables 1--13: main claim, notation, proof roadmap, measured slices,
  implementation crosswalk, replay, online stress, policy-semantics baselines,
  lower-service certificates, corner-case probes, JCT holdout, and production
  evidence remain present.
- Tables 14--15: still cover the Lean/formal crosswalk and empirical
  assumptions, but now use paper-facing proof groups and certificate categories
  rather than long Lean identifiers.
- Tables 16--17: restored as expanded evidence-gate summary tables, replacing
  the former two-page reviewer gate ladder.  They preserve the existence and
  scope of theorem/Scheduleurm evidence, finite-domain and LCB/slack gates,
  production bridge and live-oracle readiness, external-policy evidence,
  runtime probes, Gavel calibration, external action union, Decima-style bridge,
  learning/regime extensions, and future-state safety.

## Condensed or Removed Material

The following material was intentionally condensed or moved out of the main PDF
body:

- Long Lean theorem identifiers in Table 14.  The exact identifiers remain in
  the supplementary theorem crosswalk and build artifact.
- Full reviewer gate-ladder rows with `Scoped ready`, `Strong ready`, `Blocker`,
  and `Next threshold` columns.  These were replaced by Tables 16--17 and the
  machine-readable supplementary dashboard.
- Internal path, hash, status-flag, and run-log language.
- Process terms such as audit/gate/check where a normal paper phrase is enough.
- Raw dashboard detail that belongs in `md/` artifacts rather than the article.

## Potentially Sensitive Reduction

The only substantive compression that could be questioned is the old
Table 16/17 gate ladder.  It did contain many fine-grained gate rows.  The
current manuscript now keeps the same categories as expanded gate-family rows
and points to the supplementary artifact package for the machine-readable
dashboard.  This is the correct paper-facing version; the detailed dashboard
should not be printed as two dense appendix tables unless the supervisor
explicitly wants an engineering-run-log appendix.

## Current Judgment

No main mathematical theorem, proof appendix section, experiment table, SOTA
policy family, production evidence category, or reference family appears to be
lost.  The page-count reduction comes from cleanup and compression, plus tighter
float placement after Related Work was moved forward.  Table 1 is now pinned
before Related Work so that the claim-scope table does not float across the
literature section.

## Table Compression Follow-Up

After expanding Tables 16--17, all other manuscript tables were scanned for the
same risk: cleanup that makes the table too coarse to preserve proof,
experiment, or boundary information.

Tables updated:

- Table 1 now separates robust stability, measured domains, Scheduleurm theorem
  objects, controlled/production evidence, external comparison, and learning
  extensions.  This prevents production/live-oracle evidence from being hidden
  inside a single implementation row.
- Table 5 now includes the state index \(\chi(t)\) and admission/probe state.
  This restores the statewise feasible-family and unknown-state gate that are
  essential to the theorem-code mapping.
- Table 13 now includes the completed-active service-map bridge and live-oracle
  boundary, in addition to controlled launches, strict production history, and
  rolling queue snapshots.

Tables reviewed but not expanded:

- Tables 2--3 already carry notation and proof-spine information at the right
  level.
- Table 4 already defines measured slices, capacity boundaries, replay choices,
  and support/LCB usage.
- Tables 6--12 are numerical evidence tables with explicit scope notes; adding
  more gate text would reduce readability without adding necessary claim
  information.
- Tables 14--17 now cover formal proof groups, empirical certificates, and the
  expanded gate summaries.
