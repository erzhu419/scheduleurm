# Direct SOTA Baseline Scaffold

## Adapter Status

| Adapter | Systems | Repo | Files | Smoke | Direct runnable | Blocker |
|---|---|---|---:|---|---:|---|
| `gavel_simulation` | Gavel | `/home/erzhu419/mine_code/scheduleurm/reference/repos/gavel` | 2 | `FAIL` | false | requires Gavel trace/throughput-table conversion for Scheduleurm workloads |
| `pollux_adaptdl_scheduler` | Pollux, AdaptDL | `/home/erzhu419/mine_code/scheduleurm/reference/repos/adaptdl_pollux` | 3 | `FAIL` | false | requires Kubernetes/AdaptDL job CRD or a dedicated trace adapter |
| `iadeep_kubernetes_extender` | IADeep | `/home/erzhu419/mine_code/scheduleurm/reference/repos/iadeep` | 2 | `ERROR` | false | requires Kubernetes 1.18+, device plugin, etcd, Docker/NVIDIA runtime |
| `salus_gpu_sharing` | Salus | `/home/erzhu419/mine_code/scheduleurm/reference/repos/salus` | 3 | `FAIL` | false | requires Salus server and customized TensorFlow runtime |
| `decima_simulator` | Decima | `/home/erzhu419/mine_code/scheduleurm/reference/repos/decima_sim` | 3 | `PASS` | false | Spark-DAG simulator baseline, not a GPU co-location scheduler |

## Fallback Replay

| Taskset | Replayable | Candidate nondominated | Best makespan baseline | Best flow baseline |
|---|---:|---:|---|---|
| `q00_light_control` | true | true | `throughput_table_goodput` | `throughput_table_goodput` |
| `q01_gpu_bound_compute` | true | true | `throughput_table_goodput` | `delay_oracle` |
| `q10_cpu_host_bound` | true | true | `throughput_table_goodput` | `throughput_table_goodput` |
| `q11_cpu_gpu_coupled` | true | true | `interference_guard` | `throughput_table_goodput` |
| `hybrid_research_portfolio` | true | true | `throughput_table_goodput` | `delay_oracle` |

## Scope

direct external-system adapters are discovered and checked for local entrypoints. Full-stack superiority claims require the direct adapter status to become runnable on the same workload conversion. Until then the paper uses policy-semantics replay baselines on the same measured Scheduleurm service cache.
