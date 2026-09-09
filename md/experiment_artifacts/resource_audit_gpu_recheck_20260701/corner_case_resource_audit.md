# Corner-Case Resource Audit

JSON artifact: `md/experiment_artifacts/resource_audit_gpu_recheck_20260701/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `jtl110gpu` | true | `cpu_busy_gpu_idle` | false | false | 24 | 0.1 | 497.0 | 0:10/12288MB,0%; 1:259/12288MB,0% | ?:65532:2751746.0% 0.0 |
| `jtl110gpu2` | true | `clean_idle` | true | true | 24 | 0.3 | 496.7 | 0:152/12288MB,0%; 1:10/12288MB,0% |  |
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.1 | 21.5 | 0:58/8192MB,0%; 1:6/8192MB,0% | zhengli+:2000014:99.9% python |
| `node007-direct` | true | `gpu_and_cpu_busy` | false | false | 64 | 12.9 | 221.5 | 0:1/11264MB,0%; 1:1/11264MB,0%; 2:749/11264MB,100%; 3:1/11264MB,0% | zhengli+:124745:1033.0% python |
| `jtl110cpu` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
| `jtl110cpu2` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
| `node003` | true | `clean_idle` | true | true | 192 | 0.0 | 179.0 |  |  |
| `node005` | true | `clean_idle` | true | true | 192 | 0.2 | 178.8 |  |  |
| `node006` | true | `clean_idle` | true | true | 192 | 0.1 | 177.1 |  |  |
