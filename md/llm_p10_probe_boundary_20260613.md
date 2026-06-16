# LLM p10 Probe Boundary

## Summary

On 2026-06-13, `remote_llm_distilgpt2_jtl110gpu_p10_20260613` attempted a direct remote selected-profile probe on `jtl110gpu` with two RTX 3080 Ti GPUs and 10 concurrent `distilgpt2` forward workers per GPU.

Artifact:

`/home/erzhu419/.claude/scheduler/experiments/runs/remote_llm_distilgpt2_jtl110gpu_p10_20260613/reports/profile_10_per_gpu_summary.json`

Observed result:

| Quantity | Value |
|---|---:|
| `placement_valid` | false |
| `running_count` | 20 |
| `running_with_rate_count` | 1 |
| `expected_per_gpu_running` | `{"0": 10, "1": 10}` |
| `per_gpu_running` | `{"0": 1}` |
| `rate_count` | 1 |

Representative failed logs report:

```text
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`
```

## Interpretation

This is a capacity/runtime boundary, not a usable stochastic LCB sample.  The run must not be added to `selected_profile_holdout_supplemental_samples_20260612.json`, and it must not be used to claim positive service for `gpu_llm_distilgpt2` at profile 10 on this node.  Lower profiles may still be measured separately, but profile 10 remains outside this measurement certificate unless a later probe produces complete per-GPU rates without CUDA initialization failures.
