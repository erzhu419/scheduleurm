# Critical GPU homogeneous-equivalence gate

- JSON artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_homogeneous_equivalence_gate_r01_20260803.json`
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
| `Ant-v2` | 2 | 0.954580 | 1.047581 | yes |
| `Ant-v2` | 5 | 0.995612 | 1.004407 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 1 | 0.585181 | 1.708874 | no |
| `distilgpt2_forward_fp16_b2_s128` | 3 | 0.678297 | 1.474280 | no |
| `distilgpt2_forward_fp16_b2_s128` | 8 | 0.713323 | 1.401889 | no |
| `jax_matmul_8192` | 1 | 0.695797 | 1.437200 | no |
| `jax_matmul_8192` | 3 | 0.834602 | 1.198176 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 1 | 0.561596 | 1.780639 | no |
| `resnet50_synthetic_train_amp_b32_224` | 3 | 0.730564 | 1.368805 | no |

PASS certifies only that the declared jtl110gpu and jtl110gpu2 empty-state, matched-wave measurements support pooling the two registered 2xRTX-3080Ti nodes into one hardware bucket for the nine declared workload_env/profile cells. jtl110gpu2 remains an independent homogeneous-equivalence audit and is not an observation in the representative node's stochastic LCB fit. The empirical aggregate lower-service quantity is the minimum of task-native stable aggregate service and naturally completed aggregate service; it is an equivalence functional, not a new confidence bound. This gate does not certify loaded states, unmeasured actions, future software or hardware changes, or arbitrary future states.
