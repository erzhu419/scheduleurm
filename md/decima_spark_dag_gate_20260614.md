# Decima Spark-DAG Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `repo_ready` | true |
| `spark_dag_simulator_smoke_ready` | true |
| `spark_dag_heuristic_benchmark_ready` | true |
| `spark_dag_metric_bridge_ready` | true |
| `learned_policy_runtime_ready` | false |
| `scheduleurm_gpu_colocation_comparable_ready` | false |
| `direct_fullstack_gpu_sota_claim_ready` | false |

## Required Files

| Path | Present |
|---|---:|
| `compute_baselines.py` | true |
| `test.py` | true |
| `spark_env/env.py` | true |
| `spark_env/job_generator.py` | true |
| `spark_env/job_dag.py` | true |
| `spark_env/task.py` | true |

## Smoke Results

| Probe | Return code | Note |
|---|---:|---|
| `import` | `0` | DECIMA_IMPORT_SMOKE_READY  |
| `baseline_help` | `0` |  |
| `heuristic_benchmark` | `0` | {"done": true, "finished_jobs": 3, "steps": 29, "total_reward": -1.4020200000000005, "wall_s": 0.008051}  |
| `learned_policy_help` | `1` | Traceback (most recent call last):   File "/home/erzhu419/mine_code/scheduleurm/reference/repos/decima_sim/test.py", line 2, in <module>     import tensorflow as tf ModuleNotFoundError: No module named 'tensorflow'  |

## Blockers

| Blocker |
|---|
| Decima learned-policy entrypoint is not runnable in the local environment; TensorFlow is the expected blocker if absent |
| Decima is a Spark-DAG simulator; a Scheduleurm-vs-Decima claim needs a Spark-DAG workload and metric bridge, not a GPU co-location probe |

## Scope

Decima evidence is Spark-DAG simulator evidence.  A Decima claim needs a Spark-DAG benchmark and metric bridge.  It should not be mixed into the GPU co-location full-stack superiority gate.
