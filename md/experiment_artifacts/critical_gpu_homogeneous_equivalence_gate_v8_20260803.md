# Critical GPU homogeneous-equivalence gate

- JSON artifact: `md/experiment_artifacts/critical_gpu_homogeneous_equivalence_gate_v8_20260803.json`
- Status: `FAIL_EQUIVALENCE`
- Matched wave: `r01`
- Representative: `jtl110gpu`
- Equivalence node: `jtl110gpu2`
- Hardware-bucket pooling ready: `false`
- Equivalence observations enter main LCB: `false`
- Cell ratio interval: `[0.750000, 1.333333]`
- Robust median interval: `[0.909091, 1.100000]`

## Matched cells

| workload environment | profile | drain ratio | aggregate lower-service ratio | pass |
|---|---:|---:|---:|:---:|
| `Ant-v2` | 2 | 0.875037 | 1.142809 | yes |
| `Ant-v2` | 5 | 0.995691 | 1.004327 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 1 | 0.610916 | 1.636886 | no |
| `distilgpt2_forward_fp16_b2_s128` | 3 | 0.675702 | 1.479943 | no |
| `distilgpt2_forward_fp16_b2_s128` | 8 | 0.713418 | 1.401703 | no |
| `jax_matmul_8192` | 1 | 0.691147 | 1.446870 | no |
| `jax_matmul_8192` | 3 | 0.859697 | 1.163200 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 1 | 0.583872 | 1.712703 | no |
| `resnet50_synthetic_train_amp_b32_224` | 3 | 0.700158 | 1.428248 | no |

PASS certifies only that the declared jtl110gpu and jtl110gpu2 empty-state, matched-wave measurements support pooling the two registered 2xRTX-3080Ti nodes into one hardware bucket for the nine declared workload_env/profile cells. jtl110gpu2 remains an independent homogeneous-equivalence audit and is not an observation in the representative node's stochastic LCB fit. The empirical aggregate lower-service quantity is the minimum of task-native stable aggregate service and naturally completed aggregate service; it is an equivalence functional, not a new confidence bound. This gate does not certify loaded states, unmeasured actions, future software or hardware changes, or arbitrary future states.
