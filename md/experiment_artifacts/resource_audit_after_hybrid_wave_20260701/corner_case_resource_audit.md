# Corner-Case Resource Audit

JSON artifact: `md/experiment_artifacts/resource_audit_after_hybrid_wave_20260701/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `jtl110gpu` | true | `cpu_busy_gpu_idle` | false | false | 24 | 0.4 | 497.0 | 0:10/12288MB,0%; 1:258/12288MB,0% | ?:65532:2751746.0% 0.0 |
| `jtl110gpu2` | true | `cpu_busy_gpu_idle` | false | false | 24 | 0.5 | 496.7 | 0:158/12288MB,0%; 1:10/12288MB,0% | erzhu419:3767346:100.0% snap |
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.1 | 18.3 | 0:58/8192MB,0%; 1:6/8192MB,0% | zhengli+:2000014:99.9% python |
| `node007-direct` | true | `gpu_busy` | false | true | 64 | 9.2 | 214.5 | 0:3523/11264MB,1%; 1:4227/11264MB,0%; 2:4227/11264MB,0%; 3:4227/11264MB,0% |  |
| `node003` | true | `clean_idle` | true | true | 192 | 0.0 | 179.0 |  |  |
| `node005` | true | `clean_idle` | true | true | 192 | 0.1 | 178.8 |  |  |
| `node006` | true | `clean_idle` | true | true | 192 | 0.1 | 177.1 |  |  |
