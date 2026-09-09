# Measured-Cache External-Policy Frontier

| Quantity | Value |
|---|---:|
| `strict_pareto_ready` | false |
| `within_tolerance_ready` | false |
| `frontier_count` | 7 |
| `pareto_tolerance` | 0.005 |

## Open Frontier Rows

| Taskset | Arrival | Makespan ratio | Flow ratio | Best makespan policy | Best flow policy | Diagnosis | Required action |
|---|---|---:|---:|---|---|---|---|
| `q01_gpu_bound_cnn_resnet50` | `static` | 0.993250685 | 1 | `sota_iadeep_salus_interference_guard` | `scheduleurm_sota_union_online_pareto_slack` | same measured CNN profile but different statewise drain trajectory; current phase-switch/SJF/LPT bridge candidates improve flow but fail the strict makespan guard | measure a new CNN tail/co-location service point or admit a bridge only after it improves flow without exceeding the finish-time makespan envelope |
| `q01_gpu_bound_llm_inference` | `static` | 0.997370048 | 0.999943106 | `sota_gavel_finish_time_fairness` | `sota_gavel_finish_time_fairness` | strict gap remains inside one policy family | increase replay samples and inspect whether the gap is sampling noise or requires a new measured profile |
| `q01_gpu_bound_llm_inference` | `poisson` | 0.999948604 | 0.650570333 | `sota_iadeep_salus_interference_guard` | `sota_iadeep_salus_interference_guard` | strict gap remains inside one policy family | increase replay samples and inspect whether the gap is sampling noise or requires a new measured profile |
| `q01_gpu_model_portfolio` | `static` | 1 | 0.999997054 | `scheduleurm_sota_union_online_pareto_slack` | `sota_gavel_finish_time_fairness` | portfolio inherits the ResNet-50 static tail-drain gap at a much smaller scale | close the CNN tail/co-location bridge first, then rerun the portfolio frontier on the same measured cache |
| `q01_gpu_model_portfolio` | `poisson` | 1 | 0.99455803 | `scheduleurm_sota_union_online_pareto_slack` | `sota_iadeep_salus_interference_guard` | portfolio inherits the ResNet-50 static tail-drain gap at a much smaller scale | close the CNN tail/co-location bridge first, then rerun the portfolio frontier on the same measured cache |
| `hybrid_research_portfolio` | `static` | 1 | 0.999999982 | `scheduleurm_sota_union_online_pareto_slack` | `scheduleurm_sota_union_guarded_mean_flow` | hybrid RL profile-3 minimizes makespan while profile-2 minimizes flow | measure or admit a hybrid RL co-location profile between current p2/p3, or add a critical-path-aware p3-to-p2 switch with a verified makespan guard |
| `hybrid_research_portfolio` | `poisson` | 1 | 0.999994689 | `scheduleurm_sota_union_online_pareto_slack` | `sota_gavel_finish_time_fairness` | hybrid RL profile-3 minimizes makespan while profile-2 minimizes flow | measure or admit a hybrid RL co-location profile between current p2/p3, or add a critical-path-aware p3-to-p2 switch with a verified makespan guard |

## Probe Order

1. Hybrid RL p2/p3 bridge: run RE-SAC/BAPR-like probes at 2-3 concurrent tasks under empty and high-VRAM-resident states; admit only if the LCB gives profile-3 makespan with profile-2 tail flow.
2. CNN tail/co-location bridge: current cache-only phase-switch, SJF, critical-SJF, and LPT-static bridge variants improve flow but miss the strict makespan envelope; run controlled ResNet-50 tail and CNN+LLM/CNN+CUDA service probes and admit only if the LCB improves flow without exceeding the finish-time makespan envelope.
3. q01 portfolio strict gap: rerun after the CNN bridge closes; do not add a separate portfolio-only claim unless the per-workload CNN service point is certified.

## Scope

This certificate only diagnoses strict measured-cache policy-semantics gaps against the registered external-policy envelope. It does not claim direct full-stack SOTA binary superiority.
