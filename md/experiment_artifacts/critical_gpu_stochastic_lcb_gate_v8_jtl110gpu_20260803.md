# Critical GPU stochastic completion/LCB gate

- JSON artifact: `md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v8_jtl110gpu_20260803.json`
- Status: `PASS`
- Node: `jtl110gpu`
- Hardware-local bucket: `gpu_3080ti_12gb_dual`
- Resource state: `empty`
- Ready observations: `117` / `117`
- Missing waves: `[]`
- Smoke excluded: `True`
- Simultaneous conformal margin: `0.246020`

## Hardware-local certificate

| workload environment | profile | frozen drain (s) | upper drain (s) | holdout drain (s) | upper mean JCT (s) | lower service | covered |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `Ant-v2` | 2 | 540.105 | 672.981 | 538.397 | 669.655 | 0.237748 | yes |
| `Ant-v2` | 5 | 1136.795 | 1416.469 | 1136.946 | 1123.554 | 0.282392 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 1 | 11.725 | 14.610 | 12.587 | 14.365 | 16.427562 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 3 | 14.006 | 17.452 | 13.543 | 17.333 | 41.255712 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 8 | 21.402 | 26.667 | 21.863 | 26.037 | 71.999109 | yes |
| `jax_matmul_8192` | 1 | 4.002 | 4.987 | 3.972 | 4.955 | 16.041841 | yes |
| `jax_matmul_8192` | 3 | 8.091 | 10.081 | 8.185 | 10.013 | 23.806547 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 1 | 14.489 | 18.053 | 15.052 | 18.057 | 6.647084 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 3 | 20.838 | 25.964 | 22.109 | 25.862 | 13.865231 | yes |

The certificate is hardware-local and statewise for the declared workload_env x node_bucket x empty resource_state x profile cells. Stationary actions require every child to expose a stable task-native rate and a natural completion model. The staged RE-SAC p5 action is instead certified as a finite admission-and-drain trajectory: every child supplies all 40 outer-loop observations and naturally completes, while a static within-trajectory rate remains diagnostic rather than an assumed service constant. Both modes include allocated checkpoint evidence, and their completion-time observations feed the same one-sided split-conformal lower-service construction. Training waves freeze the phase model; calibration-wave maximum scores provide one-sided joint coverage over both drain completion time and mean task JCT for the nine declared actions. It neither uses smoke measurements nor pools hardware classes, and it does not extrapolate to loaded states, unmeasured profiles, or arbitrary future workload environments.
