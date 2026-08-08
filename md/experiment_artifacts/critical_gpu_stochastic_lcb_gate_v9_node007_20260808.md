# Critical GPU stochastic completion/LCB gate

- JSON artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v9_node007_20260808.json`
- Status: `PASS`
- Node: `node007`
- Hardware-local bucket: `gpu_2080ti_11gb_quad`
- Resource state: `empty`
- Ready observations: `117` / `117`
- Missing waves: `None`
- Smoke excluded: `False`
- Simultaneous conformal margin: `0.060601`

## Hardware-local certificate

| workload environment | profile | frozen drain (s) | upper drain (s) | holdout drain (s) | upper mean JCT (s) | lower service | covered |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `Ant-v2` | 2 | 529.444 | 561.529 | 534.181 | 557.403 | 0.569872 | yes |
| `Ant-v2` | 5 | 1587.176 | 1683.361 | 1588.025 | 1195.360 | 0.475240 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 1 | 14.588 | 15.472 | 14.942 | 14.839 | 31.023836 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 10 | 62.978 | 66.794 | 63.078 | 64.017 | 71.862596 | yes |
| `distilgpt2_forward_fp16_b2_s128` | 3 | 25.610 | 27.162 | 25.879 | 25.072 | 53.016179 | yes |
| `jax_matmul_8192` | 1 | 14.311 | 15.178 | 14.220 | 15.117 | 10.541554 | yes |
| `jax_matmul_8192` | 3 | 47.365 | 50.235 | 48.042 | 45.230 | 9.555011 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 1 | 8.466 | 8.979 | 8.424 | 8.833 | 26.728526 | yes |
| `resnet50_synthetic_train_amp_b32_224` | 3 | 14.857 | 15.758 | 14.776 | 15.526 | 45.691879 | yes |

This hardware-local empty-state certificate retains the eight valid node007 v8 actions in every pre-registered wave and replaces only the DistilGPT2 p10 observation whose aggregate detached log was prefix-truncated by the transport output cap. The replacement uses the same train/calibration/holdout wave roles, natural completion, checkpoint allocation, and unchanged 12-observation child threshold, with exact chunked transfer verified by remote size and SHA-256. The correction was fixed by action identity and transport defect, not selected by measured performance. It does not add new candidate actions, loaded resource states, hardware classes, or arbitrary future workloads.
