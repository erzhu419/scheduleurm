# Node007 Task-Native ETA Matrix Final 2026-06-13

This artifact is the final node007-direct ETA/service matrix. It admits only `ScheduleurmStableRate` parsed from the workload code progress/tqdm logs. It does not use `tui-top` ETA.

| Quantity | Value |
|---|---:|
| `pass` | true |
| `node` | `node007-direct` |
| `gpu_count` | 4 |
| `CNN cache key` | `gpu_cnn_torch_progress_stack` |
| `LLM cache key` | `gpu_llm_torch_decoder_stack` |
| `RL cache key` | `hybrid_rl_resac_ant_node007_tqdm` |

## Stable-ETA Rows

| Workload | Case | Profile/task per GPU | GPUs | Tasks | Stable tasks | Unit | Aggregate stable rate | Per-resource stable rate | Mean per-task stable rate | Seconds/unit range | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `cnn_torch_gpu` | `single_gpu_single_task` | 1 | 1 | 1 | 1/1 | `step` | 31.67 | 31.67 | 31.67 | 0.0315756-0.0315756 | `profile_1_per_gpu_summary.json` |
| `cnn_torch_gpu` | `single_gpu_multi_task` | 2 | 1 | 2 | 2/2 | `step` | 29.43 | 29.43 | 14.715 | 0.0665779-0.0693963 | `profile_2_per_gpu_summary.json` |
| `cnn_torch_gpu` | `single_gpu_multi_task` | 3 | 1 | 3 | 3/3 | `step` | 55.59 | 55.59 | 18.53 | 0.0318269-0.0848896 | `profile_3_per_gpu_summary.json` |
| `cnn_torch_gpu` | `single_gpu_multi_task` | 4 | 1 | 4 | 4/4 | `step` | 29.8 | 29.8 | 7.45 | 0.132979-0.135501 | `profile_4_per_gpu_summary.json` |
| `cnn_torch_gpu` | `four_gpu_single_task_each` | 1 | 4 | 4 | 4/4 | `step` | 126.25 | 31.5625 | 31.5625 | 0.0314663-0.0318878 | `profile_1_per_gpu_summary.json` |
| `cnn_torch_gpu` | `four_gpu_two_tasks_each` | 2 | 4 | 8 | 8/8 | `step` | 197.886963 | 49.4717406 | 24.7358703 | 0.0314268-0.0642261 | `profile_2_per_gpu_summary.json` |
| `llm_torch_transformer` | `single_gpu_single_task` | 1 | 1 | 1 | 1/1 | `step` | 411.640608 | 411.640608 | 411.640608 | 0.0024293-0.0024293 | `profile_1_per_gpu_summary.json` |
| `llm_torch_transformer` | `single_gpu_multi_task` | 2 | 1 | 2 | 2/2 | `step` | 643.967962 | 643.967962 | 321.983981 | 0.00242417-0.00432049 | `profile_2_per_gpu_summary.json` |
| `llm_torch_transformer` | `single_gpu_multi_task` | 3 | 1 | 3 | 3/3 | `step` | 1054.49146 | 1054.49146 | 351.497154 | 0.00250287-0.00344541 | `profile_3_per_gpu_summary.json` |
| `llm_torch_transformer` | `single_gpu_multi_task` | 4 | 1 | 4 | 4/4 | `step` | 1217.40034 | 1217.40034 | 304.350084 | 0.00261837-0.00515197 | `profile_4_per_gpu_summary.json` |
| `llm_torch_transformer` | `four_gpu_single_task_each` | 1 | 4 | 4 | 4/4 | `step` | 1238.21605 | 309.554011 | 309.554011 | 0.00279362-0.00370061 | `profile_1_per_gpu_summary.json` |
| `llm_torch_transformer` | `four_gpu_two_tasks_each` | 2 | 4 | 8 | 8/8 | `step` | 2518.28872 | 629.57218 | 314.78609 | 0.00254911-0.003958 | `profile_2_per_gpu_summary.json` |
| `rl_resac_ant` | `single_gpu_single_task` | 1 | 1 | 1 | 1/1 | `iter` | 0.294117647 | 0.294117647 | 0.294117647 | 3.4-3.4 | `profile_1_per_gpu_summary.json` |
| `rl_resac_ant` | `single_gpu_multi_task` | 2 | 1 | 2 | 2/2 | `iter` | 0.290404041 | 0.290404041 | 0.14520202 | 6.6-7.2 | `profile_2_per_gpu_summary.json` |
| `rl_resac_ant` | `single_gpu_multi_task` | 3 | 1 | 3 | 3/3 | `iter` | 0.37823271 | 0.37823271 | 0.12607757 | 6.7-11.1 | `profile_3_per_gpu_summary.json` |
| `rl_resac_ant` | `single_gpu_multi_task` | 4 | 1 | 4 | 4/4 | `iter` | 0.353905854 | 0.353905854 | 0.0884764635 | 9.9-14.8 | `profile_4_per_gpu_summary.json` |
| `rl_resac_ant` | `single_gpu_multi_task` | 5 | 1 | 5 | 5/5 | `iter` | 0.490993046 | 0.490993046 | 0.0981986092 | 6.9-18.5 | `profile_5_per_gpu_summary.json` |
| `rl_resac_ant` | `four_gpu_single_task_each` | 1 | 4 | 4 | 4/4 | `iter` | 1.16806723 | 0.292016807 | 0.292016807 | 3.4-3.5 | `profile_1_per_gpu_summary.json` |
| `rl_resac_ant` | `four_gpu_two_tasks_each` | 2 | 4 | 8 | 8/8 | `iter` | 1.5755542 | 0.39388855 | 0.196944275 | 3.4-7.4 | `profile_2_per_gpu_summary.json` |

## Interpretation

- CNN: profile 3 and four-GPU profile 2 expose useful co-location service; profile 4 is lower and should not be blindly preferred.
- LLM: aggregate rate keeps improving through profile 4 on one GPU and remains strong under four-GPU profile 2.
- RE-SAC: single-GPU profile 5 gives the best measured single-GPU aggregate rate, while four-GPU profile 2 gives the strongest per-resource aggregate service among validated multi-GPU rows.
- Algorithm implication: service lookup must be workload/profile/load-state keyed and must reuse identical measured ETA states instead of re-probing every identical parameter/environment combination.

## Scope

Final node007-direct ETA/service matrix for CNN progress-stack, torch decoder LLM, and RE-SAC hybrid tasks. It validates single-GPU single-task, single-GPU multi-task, and multi-GPU multi-task service points. CNN/LLM keys are intentionally separate from ResNet50/DistilGPT2 because node007 used built-in fallback benchmarks.

## Source Runs

| Name | Run | Pass | Path |
|---|---|---:|---|
| `cnn_main` | `node007_eta_matrix_cnn_v2_20260613` | false | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/node007_eta_matrix_cnn_v2_20260613.json` |
| `cnn_p1_retry` | `node007_eta_matrix_cnn_p1_retry_20260613` | true | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/node007_eta_matrix_cnn_p1_retry_20260613.json` |
| `llm` | `node007_eta_matrix_llm_20260613` | true | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/node007_eta_matrix_llm_20260613.json` |
| `rl_main` | `node007_eta_matrix_rl_20260613` | false | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/node007_eta_matrix_rl_20260613.json` |
| `rl_p1_retry` | `node007_eta_matrix_rl_fourgpu_p1_retry_20260613` | true | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/node007_eta_matrix_rl_fourgpu_p1_retry_20260613.json` |
