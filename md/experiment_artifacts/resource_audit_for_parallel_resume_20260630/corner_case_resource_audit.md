# Corner-Case Resource Audit

JSON artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/resource_audit_for_parallel_resume_20260630/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `node007-direct` | true | `clean_idle` | true | true | 64 | 0.1 | 227.5 | 0:1/11264MB,0%; 1:1/11264MB,0%; 2:1/11264MB,0%; 3:1/11264MB,0% |  |
| `jtl110gpu` | true | `gpu_and_cpu_busy` | false | false | 24 | 18.0 | 487.1 | 0:4235/12288MB,0%; 1:4484/12288MB,0% |  |
| `jtl110gpu2` | true | `cpu_busy_gpu_idle` | false | false | 24 | 29.0 | 483.4 | 0:158/12288MB,0%; 1:10/12288MB,0% | 2694459:203.0% python |
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.0 | 29.0 | 0:58/8192MB,0%; 1:6/8192MB,0% | 1771442:99.9% python |
| `node003` | true | `clean_idle` | true | true | 192 | 0.1 | 179.0 |  |  |
| `node005` | true | `clean_idle` | true | true | 192 | 0.0 | 178.8 |  |  |
| `node006` | true | `clean_idle` | true | true | 192 | 0.0 | 177.1 |  |  |
