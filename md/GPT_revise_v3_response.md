# GPT Revise v3 Response Checklist

Date: 2026-06-16

This note records the release-candidate fixes made after `md/GPT_revise_v3.md`.
It is a scope-control document, not a new contribution claim.

## Completed Fixes

1. EC.3 penalty consistency:
   - `paper/main.tex` now states the penalty growth bound candidate-wise over all admitted candidates \(a\in\mathcal A^{\mathrm{cand}}_t\), not only for the selected action.
   - The appendix proof text now invokes the all-candidate penalty bound when moving from the candidate support maximum to the selected lower-service margin.

2. Lean supplement shell:
   - `md/experiment_artifacts/or_reviewer_supplement_20260612/` now contains the reviewer-facing upload shell: `ScheduleurmUpload.lean`, `lakefile.toml`, `lean-toolchain`, build logs, theorem crosswalk, manifest, README, SHA-256 manifest, and `no_sorry_audit.txt`.
   - `md/experiment_artifacts/or_reviewer_supplement_20260612.zip` was regenerated from that directory.

3. SOTA/external-system scope freeze:
   - The historical entrypoint `algorithm/experiments/sota_fullstack_superiority_gate.py` now emits a named external runtime-probe gate.
   - Current machine fields are:
     - `named_same_host_runtime_probe_ready=true`;
     - `named_same_host_runtime_probe_native_better=true`;
     - `direct_fullstack_sota_superiority_ready=false`;
     - `direct_fullstack_named_sota_superiority_ready=false`.
   - `sota_universe_registry_gate`, `sota_admitted_universe_closure_gate`, `registered_sota_adapter_closure_gate`, `multinode_original_deployment_gate`, and `gate_status_dashboard` were updated so scoped runtime-probe evidence is not promoted to direct full-stack SOTA superiority.

4. Production and controlled-completion consistency:
   - `paper/main.tex` now reports the latest strict organic history numbers: 4,058 strict launched rows, 4,047 completed rows, 35 workload domains, 13 nodes, and zero strict unadmitted launched rows.
   - The manuscript now includes a production/control evidence ledger separating 6-task controlled completion, 32-task controlled completion, strict organic scheduler history, and the rolling production wrapper.

5. Declared finite cover metadata:
   - `algorithm/experiments/declared_finite_domain_positive_cover_gate.py` now emits service-cache metadata fields: `service_cache_v2_metadata_ready`, measurement first/last seen timestamps, measurement window, timestamped/untimestamped bucket counts, and source-run identifiers.
   - `md/declared_finite_domain_positive_cover_gate_20260612.md` and its JSON artifact were regenerated with those fields.

6. Reproduction script:
   - `scripts/reproduce_or_submission.sh` now includes declared finite cover, named external runtime-probe gate, registered SOTA registry, registered adapter closure, SOTA admitted-universe closure, organic history, multi-node history, multi-node original-deployment audit, Decima same-domain benchmark, dashboard, Lean audit/build, tests, and PDF build steps.
   - Rolling/production gates that can wait on queue state remain precomputed artifacts rather than forced live launch steps.

7. Stale strong-claim cleanup:
   - Current md/code/tests/artifacts no longer contain the old direct-fullstack-true flags, the old named-direct-fullstack class, or the old non-future scoped-strong status outside the original reviewer-comment file.
   - The non-future aggregate is now scoped-only: `NON_FUTURE_SCOPED_CLAIMS_CLOSED_STRONG_EXTENSIONS_FALSE`, with `strong_claim_ready=false`.

## Current Safe Claim

The paper may claim a Lean-backed robust candidate MaxWeight theorem, a declared finite measured service-cache cover, scoped runtime-probe evidence for the named external systems, registered policy/action-family closure on the measured-cache universe, controlled launched-completion evidence, and strict organic scheduler-history evidence.

It should not claim direct full-stack SOTA superiority, arbitrary SOTA superiority, arbitrary future-workload positive service, external original multi-node deployment superiority, or automatic theorem-grade closure for future production arrivals.
