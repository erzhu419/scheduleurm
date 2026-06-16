# GPU CNN/LLM Workload Expansion (2026-06-12)

## Selected-Profile LCB Status

| Quantity | Value |
|---|---:|
| `sample_ready_count` | 7 |
| `sample_pending_count` | 0 |
| `supplemental_rate_count` | 79 |
| `gate_pass` | False |
| `eta` | -65.93159483563893 |
| `epsilon_est_selected_actions` | 65.99851667784168 |

## Additional GPU Workload Curves

| Benchmark | Profile/GPU | Valid | Rate Samples | Aggregate step/s | Mean step/s | Boundary note | Source |
|---|---:|---:|---:|---:|---:|---|---|
| `gpu_cnn_jax_convstack` | 1 | true | 2 | 182.052 | 91.0258 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_jax_convstack_jtl110gpu_profile1_20260612_002/reports/profile_1_per_gpu_summary.json` |
| `gpu_cnn_jax_convstack` | 2 | true | 4 | 241.69 | 60.4226 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_jax_convstack_jtl110gpu_profiles2_3_20260612_001/reports/profile_2_per_gpu_summary.json` |
| `gpu_cnn_jax_convstack` | 3 | true | 6 | 364.426 | 60.7377 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_jax_convstack_jtl110gpu_profiles2_3_20260612_001/reports/profile_3_per_gpu_summary.json` |
| `gpu_cnn_torch_resnet50` | 1 | true | 2 | 58.1404 | 29.0702 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_torch_resnet50_jtl110gpu_profile1_20260612_002/reports/profile_1_per_gpu_summary.json` |
| `gpu_cnn_torch_resnet50` | 2 | true | 4 | 85.0772 | 21.2693 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_torch_resnet50_jtl110gpu_profiles2_3_20260612_001/reports/profile_2_per_gpu_summary.json` |
| `gpu_cnn_torch_resnet50` | 3 | true | 6 | 102.11 | 17.0184 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_torch_resnet50_jtl110gpu_profiles2_3_20260612_001/reports/profile_3_per_gpu_summary.json` |
| `gpu_cnn_torch_resnet50` | 4 | false | 6 | 84.3922 | 14.0654 | actual per-GPU running {'0': 3, '1': 3} != expected {'0': 4, '1': 4} | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_torch_resnet50_jtl110gpu_profiles4_5_20260612_001/reports/profile_4_per_gpu_summary.json` |
| `gpu_cnn_torch_resnet50` | 5 | false | 0 | 0 | 0 | actual per-GPU running {} != expected {'0': 5, '1': 5} | `/home/erzhu419/.claude/scheduler/experiments/runs/cnn_torch_resnet50_jtl110gpu_profiles4_5_20260612_001/reports/profile_5_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 1 | true | 2 | 757.506 | 378.753 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profile1_20260612_003/reports/profile_1_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 2 | true | 4 | 1492.03 | 373.007 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles2_3_20260612_001/reports/profile_2_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 3 | true | 6 | 2255.32 | 375.887 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles2_3_20260612_001/reports/profile_3_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 4 | true | 8 | 2875.93 | 359.491 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles4_5_20260612_001/reports/profile_4_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 5 | true | 10 | 3398.81 | 339.881 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles4_5_20260612_001/reports/profile_5_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 6 | true | 12 | 4060.78 | 338.399 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles6_8_20260612_001/reports/profile_6_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 8 | true | 16 | 5097.99 | 318.625 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles6_8_20260612_001/reports/profile_8_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 10 | true | 20 | 6484.65 | 324.233 |  | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles10_12_20260612_001/reports/profile_10_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 11 | false | 21 | 6189.22 | 294.725 | actual per-GPU running {'0': 11, '1': 10} != expected {'0': 11, '1': 11} | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profile11_20260612_001/reports/profile_11_per_gpu_summary.json` |
| `gpu_llm_distilgpt2` | 12 | false | 12 | 3804.3 | 317.025 | actual per-GPU running {'0': 12} != expected {'0': 12, '1': 12} | `/home/erzhu419/.claude/scheduler/experiments/runs/llm_torch_distilgpt2_jtl110gpu_profiles10_12_20260612_001/reports/profile_12_per_gpu_summary.json` |

## Scope

Additional non-RL GPU workload probes. These do not alter the existing selected-profile theorem targets unless promoted through a later service-map admission step.
