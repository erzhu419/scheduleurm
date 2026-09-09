# Live v2 Dispatch Alignment Gate

- Status: `LIVE_V2_DISPATCH_ALIGNMENT_PASS`
- Pass: `true`
- Cache: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/service_cache_v2_live_merged_20260629.json`

| Check | Pass |
|---|---:|
| `env_specific_workload_key` | true |
| `node_specific_empty_cnn_lookup` | true |
| `rl_env_node_specific_assignment` | true |
| `resource_state_marginal_lookup` | true |
| `global_maxweight_selects_higher_lower_service` | true |

This is an algorithm-alignment gate over the measured live v2 cache. It proves that dispatch lookup and scoring consume env/node/state keys; it is not a new live production launch trace.
