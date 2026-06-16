# GPT Revise v4 Response

This response records the final consistency pass for `md/GPT_revise_v4.md`.

## 1. SOTA Runtime-Probe Claim Scope

The 2026-06-13 SOTA artifact is already scoped as a named external runtime-probe
gate, not a direct full-stack SOTA superiority gate:

- `direct_fullstack_sota_superiority_ready=false`;
- `direct_fullstack_named_sota_superiority_ready=false`;
- `named_same_host_runtime_probe_ready=true`;
- row-level `full_stack_superiority_claim_allowed=false`.

The dashboard was updated so the reviewer-facing gate name is
`named_external_runtime_probe_gate`.  The historical entrypoint and filenames
remain unchanged for backward-compatible reproduction, but the scoped and strong
claims are now separated in both code and generated artifacts.

## 2. Dashboard Synchronization

`algorithm/experiments/gate_status_dashboard.py` now reads the 2026-06-13 SOTA
artifact under the named-runtime-probe row and uses
`direct_fullstack_sota_superiority_ready` as the adjacent strong-claim key.
The next-threshold wording now requires a common deployment, workload, and
metric contract before any direct-system language.

The production rows were also tightened:

- strict history-backed production completion is reported as history evidence;
- large-scale live-oracle production trace closure is a separate subgate and
remains false unless the live trace itself closes.

## 3. Reproduction Manifest Contract

`scripts/reproduce_or_submission.sh` now adds `regeneration_mode` metadata to
every reproduction step.  The manifest distinguishes:

- `safe_rerun`;
- `safe_readonly`;
- `precomputed_live_run`;
- `requires_external_stack`;
- `not_rerun`.

This prevents reviewer confusion between deterministic local checks,
read-only live-state gates, precomputed launch evidence, and external-stack
runtime probes.

## 4. Manuscript Changes

The appendix gate ladder is now explicitly described as an artifact index, while
Table 1 remains the main-text claim dashboard.  Production completion wording now
separates strict scheduler-history completion from live-oracle trace closure.

No new strong claims were added.  The final manuscript keeps the following
claims false unless a future artifact proves otherwise:

- direct full-stack SOTA superiority;
- arbitrary or unregistered SOTA superiority;
- arbitrary future-workload positive-service coverage;
- external SOTA original multi-node control-plane superiority;
- production-wide large-scale live-oracle trace closure.

## 5. Freeze Decision

This pass is a consistency and packaging pass, not a new experimental expansion.
The theoretical proof spine remains unchanged; the edits align manuscript,
dashboard, artifact metadata, and reproduction semantics.
