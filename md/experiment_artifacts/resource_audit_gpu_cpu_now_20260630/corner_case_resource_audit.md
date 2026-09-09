# Corner-Case Resource Audit

JSON artifact: `md/experiment_artifacts/resource_audit_gpu_cpu_now_20260630/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `jtl110gpu` | true | `gpu_and_cpu_busy` | false | false | 24 | 12.4 | 489.0 | 0:5429/12288MB,92%; 1:5677/12288MB,95% |  |
| `jtl110gpu2` | true | `clean_idle` | true | true | 24 | 0.2 | 496.7 | 0:152/12288MB,0%; 1:10/12288MB,0% |  |
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.0 | 28.9 | 0:58/8192MB,0%; 1:6/8192MB,0% | 1771442:99.9% python |
| `node007-direct` | true | `clean_idle` | true | true | 64 | 0.1 | 227.5 | 0:1/11264MB,0%; 1:1/11264MB,0%; 2:1/11264MB,0%; 3:1/11264MB,0% |  |
| `jtl110cpu` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
| `jtl110cpu2` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
| `node003` | true | `clean_idle` | true | true | 192 | 0.0 | 179.0 |  |  |
| `node005` | true | `clean_idle` | true | true | 192 | 0.1 | 178.7 |  |  |
| `node006` | true | `clean_idle` | true | true | 192 | 0.3 | 177.1 |  |  |
