# Measured-Cache External-Policy Bridge-Action Gate

| Quantity | Value |
|---|---:|
| `status` | `MEASURED_CACHE_EXTERNAL_POLICY_BRIDGE_FRONTIER_ALREADY_CLOSED` |
| `within_tolerance_ready` | true |
| `strict_pareto_ready` | true |
| `frontier_count` | 0 |
| `current_cache_phase_switch_strict_ready` | false |
| `resource_launch_ready` | false |
| `real_probe_launched` | false |

## Phase-Switch Search

The measured-cache external-policy frontier is already closed by the accepted trajectory-action union. The phase-switch rows below are diagnostic only; they do not include the deterministic shuffle-static tail-drain bridge that closes the q01 CNN frontier.

| Target | Best worst ratio | Strict ready | Best action | Diagnosis |
|---|---:|---:|---|---|
| `q01_gpu_bound_cnn_resnet50` | 0.999475534 | false | `bridge_gpu_cnn_torch_resnet50_p3_to_p1_T2` | current measured profiles do not contain a strict bridge action |
| `hybrid_research_portfolio` | 0.95486412 | false | `bridge_hybrid_rl_resac_ant_p3_to_p2_T96` | current measured profiles do not contain a strict bridge action |

## Resource Audit

| Node | GPU | Used / total MiB | Util % | Scheduler blockers | Safe |
|---|---:|---:|---:|---|---:|
| `jtl110gpu` | 0 | 1427/12288 | 0 | t11571 | false |
| `jtl110gpu` | 1 | 242/12288 | 0 | t11578 | false |
| `jtl110gpu2` | 0 | 1024/12288 | 100 | t11576 | false |
| `jtl110gpu2` | 1 | 10/12288 | 0 | t11577 | false |
| `node007-direct` | 0 | 277/11264 | 42 | t11572, t11579 | false |
| `node007-direct` | 1 | 277/11264 | 37 | t11573, t11580 | false |
| `node007-direct` | 2 | 277/11264 | 40 | t11574, t11581 | false |
| `node007-direct` | 3 | 747/11264 | 100 | t11575 | false |

## Launch Plan

| Probe | Launch safe | Command | Required action |
|---|---:|---|---|
| `hybrid_rl_p2_p3_bridge` | false | `PYTHONPATH=. python3 -m algorithm.experiments.remote_workload_selected_profile_probe build --run-id sota_bridge_action_20260616_085835_hybrid_rl_p2_p3_bridge --node '<waiting-for-safe-gpu>' --gpus '<gpu>' --profiles 2,3 --cwd /home/erzhu419/mine_code/RE-SAC --cmd-template-file /home/erzhu419/mine_code/scheduleurm/algorithm/experiments/templates/resac_ant_real_venv.cmd.tpl --output-root /tmp/scheduleurm_bridge_probes/sota_bridge_action_20260616_085835_hybrid_rl_p2_p3_bridge/hybrid_rl_p2_p3_bridge --max-iters 80 --timeout-s 420 --unit iter --terminate-on-stable --stable-windows 3 --min-rate-samples 5 --stable-cv 0.2 --stable-rel-delta 0.25 --stable-skip-samples 1 --coordinated-profile-launch` | measure a p2/p3 bridge under the current resident-state fabric; admit only if lower-confidence service gives p3-level makespan with p2-level tail flow |
| `cnn_tail_drain_bridge` | false | `PYTHONPATH=. python3 -m algorithm.experiments.remote_workload_selected_profile_probe build --run-id sota_bridge_action_20260616_085835_cnn_tail_drain_bridge --node '<waiting-for-safe-gpu>' --gpus '<gpu>' --profiles 1,2,3 --cwd /home/erzhu419/mine_code/scheduleurm --cmd-template-file /home/erzhu419/mine_code/scheduleurm/algorithm/experiments/templates/torch_cnn_resnet50.cmd.tpl --output-root /tmp/scheduleurm_bridge_probes/sota_bridge_action_20260616_085835_cnn_tail_drain_bridge/cnn_tail_drain_bridge --max-iters 120 --timeout-s 300 --unit step --terminate-on-stable --stable-windows 3 --min-rate-samples 5 --stable-cv 0.12 --stable-rel-delta 0.1 --stable-skip-samples 1 ` | measure CNN p1/p2/p3 tail-drain candidates and co-location states; admit only if the tail-drain flow gain does not exceed the current makespan envelope |

## Scope

This gate diagnoses and prepares bridge actions for strict measured-cache external-policy closure. It is not direct full-stack SOTA binary superiority and it does not mutate the legacy scheduler default.
