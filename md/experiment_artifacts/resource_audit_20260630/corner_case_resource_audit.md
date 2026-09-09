# Corner-Case Resource Audit

JSON artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/resource_audit_20260630/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `jtl110gpu` | true | `gpu_and_cpu_busy` | false | false | 24 | 13.0 | 473.7 | 0:6175/12288MB,100%; 1:6407/12288MB,100% |  |
| `jtl110gpu2` | true | `cpu_busy_gpu_idle` | false | false | 24 | 4.8 | 489.4 | 0:158/12288MB,0%; 1:10/12288MB,0% | 2694459:163.0% python; 3433883:100.0% ps |
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.0 | 29.1 | 0:58/8192MB,0%; 1:6/8192MB,0% | 1540386:99.9% python |
| `node007-direct` | true | `gpu_and_cpu_busy` | false | false | 64 | 15.3 | 210.2 | 0:1/11264MB,0%; 1:6315/11264MB,100%; 2:6315/11264MB,100%; 3:6315/11264MB,100% |  |
| `node001` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.5 | 148.6 |  | 35671:115.0% fluent_mpi.20.1; 35672:99.6% fluent_mpi.20.1; 35673:99.6% fluent_mpi.20.1; 35675:99.6% fluent_mpi.20.1 |
| `node002` | true | `cpu_busy_gpu_idle` | false | false | 192 | 172.5 | 173.8 |  | 12701:4588.0% python; 251669:92.8% fluent_mpi.20.1; 251671:92.0% fluent_mpi.20.1; 251673:91.7% fluent_mpi.20.1 |
| `node003` | true | `clean_idle` | true | true | 192 | 0.1 | 179.0 |  |  |
| `node004` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.1 | 149.5 |  | 425050:113.0% fluent_mpi.20.1; 425052:99.6% fluent_mpi.20.1; 425058:99.6% fluent_mpi.20.1; 425060:99.6% fluent_mpi.20.1 |
| `node005` | true | `clean_idle` | true | true | 192 | 0.1 | 178.8 |  |  |
| `node006` | true | `clean_idle` | true | true | 192 | 0.3 | 177.1 |  |  |
| `jtl110cpu` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
| `jtl110cpu2` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
