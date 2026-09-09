# Corner-Case Resource Audit

JSON artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/resource_audit_after_node007_20260630/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `jtl110gpu` | true | `gpu_and_cpu_busy` | false | false | 24 | 9.4 | 481.0 | 0:3149/12288MB,100%; 1:3397/12288MB,100% |  |
| `jtl110gpu2` | true | `cpu_busy_gpu_idle` | false | false | 24 | 4.5 | 489.4 | 0:152/12288MB,0%; 1:10/12288MB,0% | 2694459:174.0% python |
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.0 | 29.2 | 0:58/8192MB,0%; 1:6/8192MB,0% | 1771442:99.9% python |
| `node007-direct` | true | `clean_idle` | true | true | 64 | 0.1 | 227.5 | 0:1/11264MB,0%; 1:1/11264MB,0%; 2:1/11264MB,0%; 3:1/11264MB,0% |  |
| `node001` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.2 | 153.1 |  | 35671:115.0% fluent_mpi.20.1; 35672:99.6% fluent_mpi.20.1; 35673:99.6% fluent_mpi.20.1; 35675:99.6% fluent_mpi.20.1 |
| `node002` | true | `cpu_busy_gpu_idle` | false | false | 192 | 163.1 | 174.3 |  | 12701:4588.0% python; 251669:92.8% fluent_mpi.20.1; 251671:92.0% fluent_mpi.20.1; 251673:91.7% fluent_mpi.20.1 |
| `node003` | true | `clean_idle` | true | true | 192 | 0.0 | 179.0 |  |  |
| `node004` | true | `cpu_busy_gpu_idle` | false | false | 192 | 49.6 | 149.4 |  | 425050:113.0% fluent_mpi.20.1; 425052:99.6% fluent_mpi.20.1; 425058:99.6% fluent_mpi.20.1; 425060:99.6% fluent_mpi.20.1 |
| `node005` | true | `clean_idle` | true | true | 192 | 0.0 | 178.8 |  |  |
| `node006` | true | `clean_idle` | true | true | 192 | 0.2 | 177.1 |  |  |
