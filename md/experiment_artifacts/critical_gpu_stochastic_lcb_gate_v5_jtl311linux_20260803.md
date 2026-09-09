# Critical GPU stochastic completion/LCB gate

- JSON artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v5_jtl311linux_20260803.json`
- Status: `PASS`
- Node: `jtl311linux`
- Hardware-local bucket: `gpu_2080_8gb_dual`
- Resource state: `empty`
- Registered cells: `9`
- Training-admitted positive-service cells: `8`
- Training capacity exclusions: `1`
- Ready observations: `104` / `104`
- Missing waves: `[]`
- Smoke excluded: `True`
- Simultaneous conformal margin: `1.052391`

## Hardware-local certificate

| workload environment | profile | frozen drain (s) | upper drain (s) | holdout drain (s) | upper mean JCT (s) | lower service | covered |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `Ant-v2` | 2 | 548.536 | 1125.811 | 547.887 | 1123.211 | 0.142120 | yes |
| `Ant-v2` | 5 | 1445.173 | 2966.061 | 1442.723 | 2252.087 | 0.134859 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 1 | 6.540 | 13.423 | 6.611 | 13.327 | 17.879461 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 3 | 11.592 | 23.791 | 11.143 | 22.610 | 30.263865 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 5 | 19.341 | 39.695 | 16.737 | 36.015 | 30.230265 | yes |
| `jax_matmul_8192` | 1 | 17.454 | 35.822 | 17.617 | 35.553 | 2.233247 | yes |
| `jax_matmul_8192` | 3 | 93.822 | 192.560 | 104.581 | 176.958 | 1.246367 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 1 | 11.238 | 23.064 | 11.127 | 22.864 | 5.202929 | yes |

## Training-frozen capacity exclusions

| workload environment | profile | first capacity wave | capacity waves | lower service |
|---|---:|---:|---|---:|
| `resnet50_synthetic_train_amp_b32_224` | 2 | 3 | `[3]` | 0.000000 |

The certificate is hardware-local and statewise for the declared workload_env x node_bucket x empty resource_state x profile cells. Stationary actions require every child to expose a stable task-native rate and a natural completion model. The staged RE-SAC p5 action is instead certified as a finite admission-and-drain trajectory: every child supplies all 40 outer-loop observations and naturally completes, while a static within-trajectory rate remains diagnostic rather than an assumed service constant. Both modes include allocated checkpoint evidence, and their completion-time observations feed the same one-sided split-conformal lower-service construction. Training waves freeze the phase model; calibration-wave maximum scores provide one-sided joint coverage over both drain completion time and mean task JCT for the training-admitted actions. A cell with a controlled capacity failure in any training wave is conservatively removed before calibration and cannot be re-admitted by later successes; a capacity failure after support freezing invalidates the certificate. It neither uses smoke measurements nor pools hardware classes, and it does not extrapolate to loaded states, unmeasured profiles, or arbitrary future workload environments.
