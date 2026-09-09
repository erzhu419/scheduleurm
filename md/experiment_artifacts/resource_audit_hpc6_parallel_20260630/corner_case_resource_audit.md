# Corner-Case Resource Audit

JSON artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/resource_audit_hpc6_parallel_20260630/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `node001` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.3 | 153.3 |  | 35671:115.0% fluent_mpi.20.1; 35672:99.6% fluent_mpi.20.1; 35673:99.6% fluent_mpi.20.1; 35675:99.6% fluent_mpi.20.1 |
| `node002` | true | `cpu_busy_gpu_idle` | false | false | 192 | 139.0 | 174.2 |  | 12701:4589.0% python; 251669:92.8% fluent_mpi.20.1; 251671:92.0% fluent_mpi.20.1; 251673:91.7% fluent_mpi.20.1 |
| `node003` | true | `clean_idle` | true | true | 192 | 0.1 | 179.0 |  |  |
| `node004` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.2 | 144.8 |  | 425050:113.0% fluent_mpi.20.1; 425052:99.6% fluent_mpi.20.1; 425058:99.6% fluent_mpi.20.1; 425060:99.6% fluent_mpi.20.1 |
| `node005` | true | `clean_idle` | true | true | 192 | 0.0 | 178.8 |  |  |
| `node006` | true | `clean_idle` | true | true | 192 | 0.1 | 177.1 |  |  |
