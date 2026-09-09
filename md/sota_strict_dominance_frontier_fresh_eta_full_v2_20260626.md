# Measured-Cache External-Policy Frontier

| Quantity | Value |
|---|---:|
| `strict_pareto_ready` | false |
| `within_tolerance_ready` | true |
| `frontier_count` | 1 |
| `pareto_tolerance` | 0.005 |

## Open Frontier Rows

| Taskset | Arrival | Makespan ratio | Flow ratio | Best makespan policy | Best flow policy | Diagnosis | Required action |
|---|---|---:|---:|---|---|---|---|
| `q01_gpu_bound_cnn_resnet50` | `static` | 1 | 0.999475534 | `scheduleurm_sota_union_guarded_mean_flow` | `scheduleurm_sota_union_guarded_mean_flow` | same measured CNN profile but different statewise drain trajectory; current phase-switch/SJF/LPT bridge candidates improve flow but fail the strict makespan guard | measure a new CNN tail/co-location service point or admit a bridge only after it improves flow without exceeding the finish-time makespan envelope |

## Probe Order

1. CNN tail/co-location bridge: current cache-only phase-switch, SJF, critical-SJF, and LPT-static bridge variants improve flow but miss the strict makespan envelope; run controlled ResNet-50 tail and CNN+LLM/CNN+CUDA service probes and admit only if the LCB improves flow without exceeding the finish-time makespan envelope.

## Scope

This certificate only diagnoses strict measured-cache policy-semantics gaps against the registered external-policy envelope. It does not claim direct full-stack SOTA binary superiority.
