# Critical GPU stochastic completion/LCB gate

- JSON artifact: `md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v8_jtl110gpu2_20260803.json`
- Status: `PASS`
- Node: `jtl110gpu2`
- Hardware-local bucket: `gpu_3080ti_12gb_dual`
- Resource state: `empty`
- Ready observations: `117` / `117`
- Missing waves: `[]`
- Smoke excluded: `True`
- Simultaneous conformal margin: `0.103207`

## Hardware-local certificate

| workload environment | profile | frozen drain (s) | upper drain (s) | holdout drain (s) | upper mean JCT (s) | lower service | covered |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `Ant-v2` | 2 | 474.814 | 523.818 | 477.544 | 520.209 | 0.305450 | yes |
| `Ant-v2` | 5 | 1128.375 | 1244.830 | 1123.090 | 969.553 | 0.321329 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 1 | 7.346 | 8.104 | 7.464 | 7.805 | 29.615037 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 3 | 9.545 | 10.531 | 9.515 | 10.343 | 68.372727 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 8 | 15.302 | 16.881 | 15.159 | 16.455 | 113.738805 | yes |
| `jax_matmul_8192` | 1 | 2.831 | 3.123 | 2.663 | 3.084 | 25.616737 | yes |
| `jax_matmul_8192` | 3 | 6.909 | 7.622 | 6.882 | 7.529 | 31.489805 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 1 | 8.320 | 9.179 | 8.189 | 9.175 | 13.073624 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 3 | 15.053 | 16.607 | 14.974 | 16.384 | 21.677473 | yes |

The certificate is hardware-local and statewise for the declared workload_env x node_bucket x empty resource_state x profile cells. Stationary actions require every child to expose a stable task-native rate and a natural completion model. The staged RE-SAC p5 action is instead certified as a finite admission-and-drain trajectory: every child supplies all 40 outer-loop observations and naturally completes, while a static within-trajectory rate remains diagnostic rather than an assumed service constant. Both modes include allocated checkpoint evidence, and their completion-time observations feed the same one-sided split-conformal lower-service construction. Training waves freeze the phase model; calibration-wave maximum scores provide one-sided joint coverage over both drain completion time and mean task JCT for the nine declared actions. It neither uses smoke measurements nor pools hardware classes, and it does not extrapolate to loaded states, unmeasured profiles, or arbitrary future workload environments.
