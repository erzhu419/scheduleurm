# OR Submission Closure Gates

| Gate | Pass | Scope |
|---|---:|---|
| `online` | true | rate-controlled replay on the same measured service cache; this is an online-arrival stress certificate, not a direct execution of external scheduler binaries |
| `holdout` | true | finite measured-slice holdout calibration; profiles with only one completed-history sample are reported as insufficient rather than silently treated as theorem-grade stochastic estimates. The theorem certificate uses the finite measured lower-service model directly instead of claiming stochastic generalization from sparse samples. |
| `ablation` | true | same measured service cache, with each module removed by policy semantics rather than by editing the legacy scheduler |
| `live_trace` | true | candidate-family oracle audit over live node-state traces. When trace_origin is synthetic_dryrun_live_node_probe, the scheduler queue was not modified and no task was launched; it certifies placement candidate semantics on current probed resources, not actual dispatch completion. |
| `reviewer_supplement` | true | repackages the Lean upload contract and rebuild log; theorem names still need to be kept textually synchronized with the manuscript at final submission time |
