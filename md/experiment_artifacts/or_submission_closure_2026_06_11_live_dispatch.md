# OR Submission Closure Gates

| Gate | Pass | Scope |
|---|---:|---|
| `online` | true | rate-controlled replay on the same measured service cache; this is an online-arrival stress certificate, not a direct execution of external scheduler binaries |
| `holdout` | true | finite measured-slice holdout calibration; profiles with only one completed-history sample are reported as insufficient rather than silently treated as theorem-grade stochastic estimates. The theorem certificate uses the finite measured lower-service model directly instead of claiming stochastic generalization from sparse samples. |
| `ablation` | true | same measured service cache, with each module removed by policy semantics rather than by editing the legacy scheduler |
| `live_trace` | true | candidate-family oracle audit over live node-state traces. When trace_origin is synthetic_dryrun_live_node_probe, the scheduler queue was not modified and no task was launched; it certifies placement candidate semantics on current probed resources, not actual dispatch completion. |
| `launched_live_theorem_dispatch` | true | stored, queue-mutating, launched bounded q01/q11 ScheduleurmBench validation with theorem_maxweight_v1; candidate family is bounded to avoid interfering with existing user jobs |
| `natural_live_theorem_trace` | true | live-node-probe dry run with measured q01/q11 workload records; queue is not mutated and no process is launched. Uncertified candidate profiles are blocked by the theorem policy during this probe. |
| `realization_bridge_boundary` | true | matches scheduler-emitted trace task ids to queue/archive records. Dry-run theorem traces intentionally have no realized completion claim until the same trace path is produced by actual dispatch. |
| `admission_population` | true | the reviewer-facing theorem population is completed-active production within the fixed window. Raw history and attempted-only records are reported but are not claimed as the theorem arrival stream. The live theorem dispatch mode should run with SCHEDULEURM_THEOREM_UNCERTIFIED_MODE=block when future tasks are to be admitted into the theorem population. |
| `direct_sota_scaffold` | true | direct external-system adapters are discovered and checked for local entrypoints. Full-stack superiority claims require the direct adapter status to become runnable on the same workload conversion. Until then the paper uses policy-semantics replay baselines on the same measured Scheduleurm service cache. |
| `global_fabric_cover` | true | calibrates the finite-feature metric and service Lipschitz envelope on measured Scheduleurm tasksets. The main theorem certificate uses the exact measured finite action slice with rho=0; representative cover radii are reported as extension/engineering diagnostics. |
| `reviewer_supplement` | true | repackages the Lean upload contract and rebuild log; theorem names still need to be kept textually synchronized with the manuscript at final submission time |
