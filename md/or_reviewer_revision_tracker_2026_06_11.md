# OR reviewer revision tracker

Date: 2026-06-11, Asia/Shanghai.

Source review: `md/GPT_revise_OR.md`.

This tracker records the systematic fixes made after the OR-style review.  The
goal is not to weaken the paper into a small engineering note.  The goal is to
make the high-level Operations Research claim auditable: theorem objects,
implementation hooks, finite-slice experiments, production bridges, and external
policy comparisons must each have a precise scope and evidence chain.

## Revision stance

The paper now follows this framing:

```text
The central OR contribution is a robust candidate MaxWeight certificate with
explicit slack accounting for candidate-cover loss, lower-service error,
penalties, and approximate optimization.

Scheduleurm is the case-study scheduler used to instantiate the certificate on
measured finite slices, production mapped populations, and audited oracle traces.

The current live placement hook is a scheduler integration surface, not the
claim that every production dispatch already runs global robust MaxWeight.
```

## Concern-by-concern handling

| Reviewer concern | Fix applied | Files |
|---|---|---|
| The title and abstract sounded like a deployed global scheduler theorem. | Retitled as a certificate/case-study paper and rewrote the abstract to distinguish theorem-facing replay/oracle rows from the optional live placement hook. | `paper/main.tex` |
| The proof policy and live scheduler hook were not the same object. | Added a first-page claim matrix and a theorem-policy crosswalk.  The crosswalk maps `queue_vector`, `lower_service`, `penalty_units`, oracle error, candidate rows, and the live hook to their theorem roles. | `paper/main.tex`, `algorithm/README.md` |
| Finite-slice replay was being asked to carry too much empirical weight. | Kept the finite-slice result as a strong exact instantiation but explicitly marked online arrival experiments, holdout calibration, ablations, and longer live trace closure as required gates before any stronger production-online claim. | `paper/main.tex`, `md/experimental_open_items.md` |
| `SOTA-style baseline` was attackable. | Renamed the comparison to SOTA-inspired policy-semantics replay baselines and stated that the paper does not claim direct binary/full-stack superiority over Gavel, Pollux, Sia, IADeep, or Salus. | `paper/main.tex`, `md/or_claim_scope_matrix_2026_06_11.md`, `md/experimental_open_items.md` |
| Production service-map bridge was not a live theorem-grade proof. | Separated completed-active service-map bridge, raw telemetry, and live trace audit.  The OR gate now records a 128-slot / 767-candidate live-state dry-run audit, not statistical evidence for all online dispatches. | `paper/main.tex`, `md/or_claim_scope_matrix_2026_06_11.md` |
| Raw 30-day history was not globally closed. | Added raw history counts: 6957 records, 4825 mapped, 2132 unmapped, mapped fraction about 0.694, global theorem flag false. | `paper/main.tex`, `md/or_claim_scope_matrix_2026_06_11.md` |
| `Lrho=0`, `epsilon_est=0`, `alpha1=0` could be misread as broad calibration. | Clarified that zero modeled losses come from exact enumerated measured slices and exact service-map oracle construction; broad fabric-cover calibration still requires perturbation profiling. | `paper/main.tex`, `md/experimental_open_items.md` |
| Lean artifact needed to be reviewer-auditable, not just hash-based. | Added the supplement contract: `ScheduleurmUpload.lean`, `lakefile.toml`, `lean-toolchain`, one-command build script, theorem crosswalk, theorem-name grep, no `sorry/admit/axiom` log, and path-independent build log. | `paper/main.tex`, `md/experimental_open_items.md` |
| The paper needed a first-page claim matrix. | Added `Table~\ref{tab:claim-matrix}` with Claim / Scope / Evidence / Not claimed. | `paper/main.tex` |
| The algorithm README could mislead readers into treating `sweetspot_v1` as the proof policy. | Added explicit reviewer-facing boundary: `sweetspot_v1` is a one-task placement hook; theorem-grade rows require queue vector, lower service, penalty units, score semantics, selected action, and oracle audit. | `algorithm/README.md` |

## Empirical gates closed after the review

These items were not fabricated into the paper.  They were implemented as
separate OR closure gates and rerun on 2026-06-11:

```text
online arrival experiment:
  Poisson, bursty, load-sweep arrivals;
  artifact = md/or_gate_online_arrivals.md
  status = PASS
  scenario_count = 120

holdout lower-service calibration:
  train on one profile/window set;
  evaluate epsilon_est and LCB residuals on holdout profiles/windows.
  artifact = md/or_gate_holdout_calibration.md
  status = PASS
  note = sparse EB diagnostic is not claimed as stochastic generalization;
         finite measured lower-service certificates close with eta>0.

ablation:
  legacy rules only;
  scalar sweetspot hook;
  robust lower-service scorer;
  exact finite family vs candidate cover;
  no penalty vs bounded/queue-scaled penalty;
  support guard vs mean-flow tie-break.
  artifact = md/or_gate_ablation_suite.md
  status = PASS
  note = no ablation policy Pareto-dominates the full adaptive candidate.

longer live theorem trace:
  many emitted candidate slots;
  queue_vector, lower_service, penalty_units, selected action,
  score_semantics=robust_maxweight_lower_service, oracle-gap audit.
  artifact = md/or_gate_live_trace.md
  status = PASS
  trace_slot_count = 128
  candidate_count_total = 767
  scope = synthetic live-state dry-run, no queue mutation, no task launch.

full reviewer artifact package:
  path-independent Lean build and scripts regenerating paper tables/certificates.
  artifact = md/or_gate_reviewer_supplement.md
  status = PASS
  note = ScheduleurmUpload.lean, lakefile.toml, lean-toolchain, crosswalk,
         build log, and no-sorry/admit/axiom grep are included.
```

The executable roll-up is:

```text
python3 -m algorithm.experiments.or_submission_closure all
summary = md/experiment_artifacts/or_submission_closure_2026_06_11.md
json    = md/experiment_artifacts/or_submission_closure_2026_06_11.json
```

## Current reviewer-facing posture

Safe:

```text
This is a robust candidate MaxWeight certificate paper with a Scheduleurm case
study.  The measured finite-slice and completed-active production bridge claims
are auditable under the declared populations and artifacts.
```

Unsafe:

```text
Scheduleurm's deployed live scheduler has fully implemented and empirically
validated global robust MaxWeight over all future production dispatches.
```
