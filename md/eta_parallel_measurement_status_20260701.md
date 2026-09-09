# ETA Parallel Measurement Status 2026-07-01

## Purpose

This note records the current parallel ETA remeasurement state after moving from
single-node node007 fresh ETA to a hardware-class, workload-env, and load-state
aware service matrix.  Rows enter theorem-facing replay only when they come from
task-native `tqdm` / `ScheduleurmProgress` stable-rate measurements.  History ETA
fallback rows remain excluded.

## Resource Audit

The latest audit separates truly clean nodes from nodes that only look idle in a
GPU-oriented view.

| Node | Status | Action |
|---|---|---|
| `jtl110gpu2` | clean GPU node | Used as homogeneous 3080Ti representative/equivalence lane. |
| `node007-direct` | busy with controlled full-loaded RL probe | Keep waiting; do not launch another node007 lane. |
| `node003` | clean 192-core CPU node | Fresh CPU-resident row measured. |
| `node005` | clean 192-core CPU node | Fresh CPU-resident row measured. |
| `node006` | clean 192-core CPU node in the recheck | Available, but current 192-core design rows are already covered. |
| `node001` | external CPU load observed | Do not use for clean fresh ETA until audit is clean. |
| `node002` | external CPU load observed | Do not use for clean fresh ETA until audit is clean. |
| `node004` | external CPU load observed | Do not use for clean fresh ETA until audit is clean. |
| `jtl311linux` | local Python process observed during audit | Defer new jtl311 load-state probes until clean. |
| `jtl110gpu` | process-audit anomaly / not selected | Use `jtl110gpu2` homogeneous row or wait for clean audit. |
| `jtl110cpu`, `jtl110cpu2` | unreachable in latest audit | Do not launch. |

## New Valid Rows

Fresh CPU rows:

| Node bucket | Workload | State | Resident mix | Profile | Stable rate |
|---|---|---|---|---:|---:|
| `node003:cpu_hpc_192c` | `cpu_heavy_local_bench` | `cpu_resident` | `cpu_worker_resident` | 1 | 14.092438 |
| `node005:cpu_hpc_192c` | `cpu_heavy_local_bench` | `cpu_resident` | `cpu_worker_resident` | 1 | 16.861275 |

New jtl110gpu2 3080Ti homogeneous rows:

| Workload | Env | State | Resident mix | Profile | Stable rate |
|---|---|---|---|---:|---:|
| `gpu_cnn_torch_resnet50` | `cnn` | `cpu_resident` | `cpu_worker_resident` | 3 | 50.1194999 |
| `gpu_cnn_torch_resnet50` | `cnn` | `mixed_colocation` | `llm_plus_hybrid_rl` | 2 | 40.2283182 |
| `gpu_cnn_torch_resnet50` | `cnn` | `mixed_colocation` | `cpu_plus_gpu_target` | 1 | 48.300683 |
| `gpu_cnn_torch_resnet50` | `cnn` | `mixed_colocation` | `llm_plus_hybrid_rl` | 3 | 40.7274337 |
| `gpu_cnn_torch_resnet50` | `cnn` | `mixed_colocation` | `cpu_plus_gpu_target` | 2 | 46.021759 |
| `gpu_cnn_torch_resnet50` | `cnn` | `mixed_colocation` | `cpu_plus_gpu_target` | 3 | 54.3454301 |
| `gpu_llm_distilgpt2` | `llm` | `mixed_colocation` | `cnn_plus_hybrid_rl` | 1 | 427.333865 |
| `gpu_llm_distilgpt2` | `llm` | `mixed_colocation` | `llm_plus_hybrid_rl` | 1 | 451.387764 |
| `gpu_llm_distilgpt2` | `llm` | `mixed_colocation` | `cnn_plus_llm_plus_hybrid_rl` | 1 | 433.856751 |
| `gpu_heavy_jax_matmul` | `gpu_matmul` | `mixed_colocation` | `cpu_plus_gpu_target` | 1 | 426.320393 |

These rows are stored under `jtl110gpu2:gpu_3080ti_12gb_dual`, but
`ServiceRateCache.get_statewise(..., node_bucket="jtl110gpu:gpu_3080ti_12gb_dual")`
can retrieve them through the hardware-class fallback.  This matches the
user-confirmed homogeneous status of `jtl110gpu` and `jtl110gpu2`.

## Boundary Rows

The following were launched but did not produce stable task-native ETA and
therefore do not enter service cache records:

| Node | Workload | State | Resident mix | Reason |
|---|---|---|---|---|
| `jtl110gpu2` | `gpu_llm_distilgpt2` | `cpu_resident` | `cpu_worker_resident` | `stable_rate_not_ready` |
| `jtl110gpu2` | `gpu_llm_distilgpt2` | `mixed_colocation` | `cpu_plus_gpu_target` | `stable_rate_not_ready` |
| `jtl110gpu2` | hybrid RL rows from the earlier T2 attempt | several load states | several mixes | `add_probe_returncode_139` |
| `jtl110gpu2` | `hybrid_rl_resac_hopper` | `cpu_resident` | `cpu_worker_resident` | `add_probe_returncode_139` |
| `jtl110gpu2` | `hybrid_rl_resac_walker2d` | `cpu_resident` | `cpu_worker_resident` | `add_probe_returncode_139` |

These are risk/capacity-boundary evidence, not stable ETA rows.

## Current Gates

| Gate | Artifact | Status |
|---|---|---|
| ETA server coverage | `md/eta_server_coverage_gate_intermediate5_20260701.md` | `ETA_SERVER_COVERAGE_REQUIRED_READY` |
| Full-factorial design | `md/full_factorial_eta_design_intermediate5_20260701.md` | required domain designed; many optional stronger rows remain pending |
| Cache bug test | `skill/tests/test_eta_migration_v2.py` | state-specific rows do not overwrite legacy `(workload_key, profile)` lookup |

The remaining full-factorial pending rows are mostly optional stronger
load-state rows: hybrid RL under co-location, jtl311linux load states,
node007 load states, and inaccessible `jtl110cpu` resident rows.  They are not
the same as the current OR theorem-facing ETA coverage gate.

## Active Work

`node007-direct` is still running the controlled full-loaded RL probe.  Valid
rows already observed there include:

| Workload env | Valid profiles so far | Notes |
|---|---|---|
| `ant` | 1, 2, 3, 4 | profile 5/6 unstable or missing enough stable samples |
| `halfcheetah` | 1, 2, 3 | profile 4/5/6 unstable or missing enough stable samples |
| `hopper` | 1, 2, 3 | profile 4/5/6 unstable or partial only |
| `walker2d` | 1, 2, 3 | profile 4 is currently running |

Update: `walker2d` profile 4 also became valid with aggregate stable rate
2.5817966402.  Profile 5 produced 19/20 stable samples and remains a boundary
row, not an admitted row.  Profile 6 is currently running.

After the node007 lane finishes, rerun the final merge, full-factorial design,
ETA coverage, SOTA candidate union, quadrant Pareto, strict frontier, and slack
accounting artifacts from the final merged cache.

## Final Merge and Replay Results

Final merged cache:

| Artifact | Value |
|---|---:|
| `md/experiment_artifacts/service_cache_v2_live_merged_20260701.json` records | 539 |
| Scoped capacity/instability boundaries | 27 |

Final ETA coverage:

| Gate | Status |
|---|---|
| `md/eta_server_coverage_gate_20260701.md` | `ETA_SERVER_COVERAGE_REQUIRED_READY` |
| Required missing rows | 0 |

Final full-factorial design status:

| Status | Rows |
|---|---:|
| `measured` | 91 |
| `measured_to_capacity_boundary` | 26 |
| `partial_pending_probe` | 3 |
| `pending_probe` | 168 |
| `probe_defer_by_design` | 28 |

The remaining pending rows are mostly stronger optional T2 load-state rows:
hybrid RL co-location rows, jtl311linux load-state rows, node007 load-state
rows, and currently inaccessible `jtl110cpu` resident rows.

SOTA policy-semantics replay:

| Artifact | Key result |
|---|---|
| `md/sota_candidate_union_gate_live_v2_fullfactorial_20260701.md` | `SOTA_CANDIDATE_UNION_PASS`; fixed online policy Pareto-dominates the registered SOTA-style policy envelope on the measured cache. |
| `md/sota_quadrant_pareto_gate_live_v2_fullfactorial_20260701.md` | `SOTA_QUADRANT_STRICT_NONINFERIOR_PASS`; q01 strictly dominates, q00/q10/q11/hybrid portfolio are strict-noninferior. |
| `md/sota_strict_dominance_frontier_live_v2_fullfactorial_20260701.md` | `MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_CLOSED`; frontier count 0. |

Finite-slice slack:

| Artifact | Delta | Eta | Notes |
|---|---:|---:|---|
| `md/experiment_artifacts/module48_portfolio_slack_certificate_live_v2_fullfactorial_20260701.md` | 0.081934686 | 0.081934686 | Uses `default_cache+live_overlay:md/experiment_artifacts/service_cache_v2_live_merged_20260701.json`. |

This closes the current measured-cache policy-semantics line.  It still does not
claim direct full-stack binary superiority over unadapted external systems, nor
does it certify arbitrary future workloads outside the admission/service-cache
contract.
