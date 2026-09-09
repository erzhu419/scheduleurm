# Measured-Cache External-Policy Frontier

| Quantity | Value |
|---|---:|
| `strict_pareto_ready` | false |
| `within_tolerance_ready` | false |
| `frontier_count` | 3 |
| `pareto_tolerance` | 0.005 |

## Open Frontier Rows

| Taskset | Arrival | Makespan ratio | Flow ratio | Best makespan policy | Best flow policy | Diagnosis | Required action |
|---|---|---:|---:|---|---|---|---|
| `q01_gpu_bound_cnn_resnet50` | `static` | 0.988708875 | 1.01082194 | `sota_gavel_finish_time_fairness` | `scheduleurm_sota_union_online_pareto_slack` | same measured CNN profile but different statewise drain trajectory; current phase-switch/SJF/LPT bridge candidates improve flow but fail the strict makespan guard | measure a new CNN tail/co-location service point or admit a bridge only after it improves flow without exceeding the finish-time makespan envelope |
| `q11_cpu_gpu_coupled` | `static` | 1 | 0.999983343 | `scheduleurm_sota_union_online_pareto_slack` | `sota_iadeep_salus_interference_guard` | best makespan and best mean-flow are supplied by different policy families | add a finite candidate action that interpolates the two policies and certify it with the same service cache |
| `hybrid_research_portfolio` | `static` | 1 | 0.999817413 | `scheduleurm_sota_union_online_pareto_slack` | `scheduleurm_sota_union_adaptive_scalarized` | hybrid RL profile-3 minimizes makespan while profile-2 minimizes flow | measure or admit a hybrid RL co-location profile between current p2/p3, or add a critical-path-aware p3-to-p2 switch with a verified makespan guard |

## Probe Order

1. Hybrid RL p2/p3 bridge: run RE-SAC/BAPR-like probes at 2-3 concurrent tasks under empty and high-VRAM-resident states; admit only if the LCB gives profile-3 makespan with profile-2 tail flow.
2. CNN tail/co-location bridge: current cache-only phase-switch, SJF, critical-SJF, and LPT-static bridge variants improve flow but miss the strict makespan envelope; run controlled ResNet-50 tail and CNN+LLM/CNN+CUDA service probes and admit only if the LCB improves flow without exceeding the finish-time makespan envelope.

## Scope

This certificate only diagnoses strict measured-cache policy-semantics gaps against the registered external-policy envelope. It does not claim direct full-stack SOTA binary superiority.
