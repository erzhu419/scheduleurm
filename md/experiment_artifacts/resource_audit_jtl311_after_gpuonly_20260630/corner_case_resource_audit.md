# Corner-Case Resource Audit

JSON artifact: `md/experiment_artifacts/resource_audit_jtl311_after_gpuonly_20260630/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.1 | 26.0 | 0:58/8192MB,0%; 1:6/8192MB,0% | 1771442:99.9% python |
| `jtl110gpu2` | true | `clean_idle` | true | true | 24 | 0.0 | 496.7 | 0:153/12288MB,0%; 1:10/12288MB,0% |  |
| `node007-direct` | true | `gpu_and_cpu_busy` | false | false | 64 | 15.3 | 215.2 | 0:1259/11264MB,100%; 1:3767/11264MB,100%; 2:3767/11264MB,100%; 3:1/11264MB,0% | 132394:912.0% python |
