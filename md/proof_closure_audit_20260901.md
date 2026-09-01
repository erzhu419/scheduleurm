# Proof closure audit

Date: 2026-09-01

## Verdict

The paper-facing proof contract is closed across the manuscript, Electronic
Companion (EC), and Lean 4 artifact.

This means:

1. The main text defines the model, assumptions, loss budget, principal
   statements, and proof spine.
2. EC.1--EC.5 give the complete conventional support, approximation, robust
   drift, Foster-recurrence, and operational-capacity derivation. EC.6 gives the
   variable-duration derivation. EC.7 gives the conditional hidden-regime and
   active-bucket derivations. EC.8 gives exact searchable Lean identifiers.
3. Every formal theorem-like environment in the combined manuscript source has
   a label and an explicit human-readable proof. There are 12 such statements:
   10 in the main manuscript and 2 in the EC.
4. Every paper-facing formal statement in EC.8 resolves to a declaration in
   `ScheduleurmUpload.lean`. The broader theorem crosswalk maps the subordinate
   theorem families and every split-source module back to a manuscript or EC
   layer.
5. The development copy and submitted consolidated `ScheduleurmUpload.lean`
   are byte-identical, and the submitted file compiles successfully with the
   pinned Lean project.

## Directional coverage

### Main text to EC

The proof roadmap in Section 3 points to EC.1--EC.5, the frame result points to
EC.6, the learning and hidden-regime discussion points to EC.7, and the formal
artifact discussion points to EC.8. Thus each EC proof layer is introduced or
cited by the main text.

### EC to Lean

Tables EC.1 and EC.2 map the core, frame, action-union, operational-capacity,
hidden-regime, and active-bucket statements to exact Lean identifiers. EC.8
also identifies the intermediate checkpoints for EC.1--EC.4. The supplementary
`theorem_crosswalk.md` supplies the lower-level theorem and module map.

### Lean to EC and main text

The complete Lean source contains many implementation lemmas that are not
independent paper claims. They are not typeset line by line. Instead, every
split-source module is assigned to the human-readable main/EC layer that uses
it, while every paper-facing wrapper theorem is listed by exact identifier.
This is the appropriate reverse map: it exposes all formal source without
inflating the paper with 6,680 lines of proof-assistant implementation.

## Verification evidence

- `make all`: PASS.
- Submission-package check: PASS; 182-word abstract, 37 manuscript pages before
  references, 14-page EC, Lengthy Manuscript limit satisfied.
- Consolidated Lean compilation: PASS with
  `lake env lean ScheduleurmUpload.lean`.
- Placeholder scan: no `sorry`, `admit`, or source-level `axiom` declarations.
- EC.8 exact identifier resolution: all listed identifiers found in the
  consolidated Lean source.
- LaTeX references: no undefined or multiply-defined references.
- EC numbering is independent: propositions, equations, tables, and pages use
  the `EC.` prefix.

## Claim boundaries

The closure is mathematical and artifact-level, not a proof that every future
scheduler state satisfies the theorem assumptions.

- Service lower bounds, fabric-cover calibration, oracle audits, moment bounds,
  and load certificates are empirical inputs to the theorem.
- Lean checks the queue-indexed statewise specialization. The manuscript's
  fully augmented-filtration version retains its explicit Markov and local-return
  assumptions.
- EC.7 proves conditional composition results. It does not assert that the
  deployed sampler or detector already satisfies the required probability or
  dwell-time certificate.
- The operational result is a positive-slack sufficiency and zero-slack
  conservation-law necessity sandwich; it is not presented as a positive-slack
  if-and-only-if theorem.
