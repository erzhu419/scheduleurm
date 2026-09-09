# Corner-Case Resource Audit

JSON artifact: `md/experiment_artifacts/resource_audit_all_recheck_20260701/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `node001` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.5 | 152.8 |  | 35671:115.0% fluent_mpi.20.1; 35672:99.6% fluent_mpi.20.1; 35673:99.6% fluent_mpi.20.1; 35675:99.6% fluent_mpi.20.1 |
| `node002` | true | `cpu_busy_gpu_idle` | false | false | 192 | 152.6 | 174.3 |  | 12701:4589.0% python; 251669:92.9% fluent_mpi.20.1; 251671:92.0% fluent_mpi.20.1; 251673:91.7% fluent_mpi.20.1 |
| `node003` | true | `clean_idle` | true | true | 192 | 0.1 | 179.0 |  |  |
| `node004` | true | `cpu_busy_gpu_idle` | false | false | 192 | 49.0 | 144.8 |  | 425050:113.0% fluent_mpi.20.1; 425052:99.6% fluent_mpi.20.1; 425058:99.6% fluent_mpi.20.1; 425060:99.6% fluent_mpi.20.1 |
| `node005` | true | `clean_idle` | true | true | 192 | 0.0 | 178.8 |  |  |
| `node006` | true | `cpu_busy_gpu_idle` | false | false | 192 | 16.2 | 172.3 |  | 420200:93.0% fluent_mpi.20.1; 420198:92.6% fluent_mpi.20.1; 420199:92.5% fluent_mpi.20.1; 420202:92.4% fluent_mpi.20.1 |
| `jtl110gpu` | true | `gpu_and_cpu_busy` | false | false | 24 | 9.1 | 480.6 | 0:6221/12288MB,100%; 1:6469/12288MB,100% | 2908733:100.0% python |
| `jtl110gpu2` | true | `clean_idle` | true | true | 24 | 0.1 | 496.7 | 0:158/12288MB,0%; 1:10/12288MB,0% |  |
| `jtl311linux` | true | `cpu_busy_gpu_idle` | false | false | 16 | 1.0 | 23.4 | 0:58/8192MB,0%; 1:6/8192MB,0% | 1901005:99.9% python |
| `node007-direct` | true | `gpu_busy` | false | true | 64 | 8.0 | 219.0 | 0:1483/11264MB,100%; 1:1483/11264MB,100%; 2:1483/11264MB,100%; 3:1483/11264MB,100% |  |
| `jtl110cpu` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
| `jtl110cpu2` | false | `` | false | false |  | 0.0 | 0.0 |  |  |
