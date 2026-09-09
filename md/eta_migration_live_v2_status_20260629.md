# Scheduleurm ETA/Migration Live-v2 Status - 2026-06-29

## Current Theorem-Facing Cache

- Merged cache: `md/experiment_artifacts/service_cache_v2_live_merged_20260629.json`
- Merge gate: `md/experiment_artifacts/service_cache_v2_live_merge_gate_20260629.json`
- Status: `SERVICE_CACHE_V2_LIVE_MERGED`
- Admitted rows: `34`
- Workload envs: `ant`, `cnn`, `cpu`, `halfcheetah`, `hopper`, `llm`, `walker2d`
- Hardware classes: `gpu_3080ti_12gb_dual`, `gpu_rtx2080_8gb_dual_cpu_fast`, `gpu_node007_4x12gb`, `cpu_hpc_192c`
- Load states: `empty`, `half_loaded`, `full_loaded`, `high_vram_resident`, `mixed_colocation`, `cpu_resident`

All theorem-facing rows above use task-native `tqdm` / `ScheduleurmStableRate` progress. Rows with history fallback, missing progress, unstable early initialization, or invalid co-location placement are excluded from theorem-facing claims.  Multi-task-per-device rows now use no-early-exit measurement: the benchmark processes keep running after the first stable-rate certificate so the co-location load does not disappear while ETA is being measured.

## Important New Live Rows

- `jtl311_resac_halfcheetah_p1`: aggregate stable rate `0.28612013 iter/s` across two GPUs, per-resource lower service `0.143060065`.
- `jtl311_resac_ant_p1`: aggregate stable rate `0.501253133 iter/s`, per-resource lower service `0.2506265665`.
- `jtl311_resac_hopper_p1`: aggregate stable rate `0.686826066 iter/s`, per-resource lower service `0.343413033`.
- `jtl311_resac_walker2d_p1`: aggregate stable rate `0.657923214 iter/s`, per-resource lower service `0.328961607`.
- `jtl110gpu_resac_ant_p2`: phase-aware train/eval cycle-average aggregate stable rate `0.544995544 iter/s`, per-resource lower service `0.272497772`; the raw last-iteration rate is intentionally not used for theorem-facing ETA.
- `jtl110gpu_resac_halfcheetah_p2`: phase-aware train/eval cycle-average aggregate stable rate `0.288199935 iter/s`, per-resource lower service `0.144099967`.
- `jtl311linux_resac_halfcheetah_p2`: phase-aware train/eval cycle-average aggregate stable rate `0.202216644 iter/s`, per-resource lower service `0.101108322`.
- `jtl110gpu_resac_hopper_p1`: aggregate stable rate `1.111111112 iter/s`, per-resource lower service `0.555555556`.
- `jtl110gpu_resac_walker2d_p1`: aggregate stable rate `1.176470588 iter/s`, per-resource lower service `0.588235294`.
- `jtl110_cnn_full_loaded_add_cnn`: add-one stable rate `28.1020937 step/s`.
- `jtl110_llm_resident_add_cnn`: add-one stable rate `28.9391054 step/s`.
- `node003_cpu_full_resident_add_cpu`: add-one stable rate `12.530672 step/s`.
- `node005_cpu_half_resident_add_cpu`: add-one stable rate `16.392399 step/s`.
- `jtl110gpu` CNN no-early p1/p2: per-resource lower service `24.6681287` and `25.02203935 step/s`; p4 is an OOM/CUBLAS capacity boundary and is not admitted.
- `jtl311linux` CNN no-early p1/p2: per-resource lower service `13.25066665` and `11.82202882 step/s`; p4 is a partial/OOM capacity boundary and is not admitted.
- `node007-direct` CNN no-early p1/p2/p4: per-resource lower service `31.7271767`, `28.995703925`, and `28.5847268525 step/s`.
- `node007-direct` LLM p1: per-resource lower service `327.867544 step/s`.
- `node007-direct` RE-SAC Ant p1: per-resource lower service `0.2375653725 iter/s`.
- `jtl110gpu` LLM no-early p1/p2: per-resource lower service `287.54439` and `632.0858785 step/s`.
- `jtl110gpu2` equivalence now has admitted CNN p1, CNN p2 no-early, and LLM p1 rows; no system CUDA/cuDNN changes were made.

## Dispatcher Alignment

- Alignment gate: `md/experiment_artifacts/live_v2_dispatch_alignment_gate_20260629.json`
- Status: `LIVE_V2_DISPATCH_ALIGNMENT_PASS`
- The dispatcher now consumes `workload_env x node_bucket x resource_state x profile`.
- Description-only RE-SAC tasks are parsed into env-specific workload keys.
- HalfCheetah assignment is node-aware. With current measured rows, `jtl110gpu` has lower service `0.185185185`, `jtl311linux` has lower service `0.143060065`, so robust MaxWeight selects `jtl110gpu`.

This is the correct behavior: placement follows measured task-native progress, not the prior expectation that the CPU-fast node must always be faster.

## SOTA Replay After Live-v2 ETA

- Gate: `md/experiment_artifacts/sota_candidate_union_gate_live_v2_20260629.json`
- Status: `SOTA_CANDIDATE_UNION_NEEDS_REVIEW`
- Result after the service-cache index fixes: the live-v2 policy-semantics replay is clean, but the strict external-policy frontier is open.
- Main exposed tradeoffs: `q01_gpu_bound_cnn_resnet50` static, `q11_cpu_gpu_coupled` static, and `hybrid_research_portfolio` static.  These are trajectory/action-semantics gaps, not ETA-history gaps.

This means the no-early fresh ETA cache plus phase-aware RL cycle ETA supports the scoped measured-cache calibration claim, but still does not justify a stronger "strictly dominates every SOTA envelope" claim.  Strict domination would require additional measured trajectory/action semantics rather than looser ETA thresholds.  See `md/eta_service_cache_v2_closure_20260629.md` for the current closure record.

## Pending / Not Yet Claimed

- Controlled migration cost now has physical checkpoint/sync/resume rows for controlled benchmark payloads at 25%, 50%, and 75% progress:
  - pure GPU CNN checkpoint, `jtl110gpu -> jtl311linux`: \(K\approx 41.66\)--`41.83` seconds, beneficial under the measured rate gap.
  - hybrid RL HalfCheetah checkpoint, `jtl110gpu -> jtl311linux`: \(K\approx 20.78\)--`20.87` seconds, not beneficial under the measured rate gap.
  - pure CPU/FreqDuet surrogate checkpoint, `node003 -> node005`: \(K\approx 22.12\)--`47.68` seconds, beneficial under the measured rate gap.
  These are controlled benchmark migrations only; ordinary running user tasks remain no-touch.
- Direct full-stack external SOTA binary superiority is still not claimed.
- Arbitrary future workload coverage is still not claimed; unknown or missing-progress rows remain probe/defer only.
- Some full-load rows are intentionally boundary rows: `jtl110gpu` CNN p4 and `jtl311linux` CNN p4 are not admissible service curves under the current ResNet50 train/batch-size benchmark because they hit OOM or partial placement failure.  `node007-direct` CNN p4 is separately measured and admitted; the cache now scopes live-v2 boundaries by node so this row is not blocked.
