# Corner-Case Resource Audit

JSON artifact: `md/experiment_artifacts/resource_audit_hpc6_useraware_20260701/corner_case_resource_audit.json`

| node | reachable | class | hybrid safe | cpu safe | nproc | load 1m | mem available GB | GPUs | external busy processes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `node001` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.1 | 148.7 |  | aiyw05:35671:115.0% fluent_mpi.20.1; aiyw05:35672:99.6% fluent_mpi.20.1; aiyw05:35673:99.6% fluent_mpi.20.1; aiyw05:35675:99.6% fluent_mpi.20.1 |
| `node002` | true | `cpu_busy_gpu_idle` | false | false | 192 | 173.8 | 174.2 |  | aiyw10:12701:4589.0% python; aiyw02:251669:92.9% fluent_mpi.20.1; aiyw02:251671:92.0% fluent_mpi.20.1; aiyw02:251673:91.7% fluent_mpi.20.1 |
| `node003` | true | `clean_idle` | true | true | 192 | 0.1 | 178.9 |  |  |
| `node004` | true | `cpu_busy_gpu_idle` | false | false | 192 | 48.2 | 149.3 |  | aiyw06:425050:113.0% fluent_mpi.20.1; aiyw06:425052:99.6% fluent_mpi.20.1; aiyw06:425058:99.6% fluent_mpi.20.1; aiyw06:425060:99.6% fluent_mpi.20.1 |
| `node005` | true | `clean_idle` | true | true | 192 | 0.2 | 178.7 |  |  |
| `node006` | true | `cpu_busy_gpu_idle` | false | false | 192 | 16.2 | 172.3 |  | aiyw02:420200:93.1% fluent_mpi.20.1; aiyw02:420198:92.8% fluent_mpi.20.1; aiyw02:420202:92.7% fluent_mpi.20.1; aiyw02:420199:92.6% fluent_mpi.20.1 |
