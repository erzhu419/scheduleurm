# Gavel Direct Native Smoke

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `dependency_import_pass` | true |
| `protobuf_stub_generation_pass` | true |
| `entrypoint_help_pass` | true |
| `native_generated_jobs_pass` | true |
| `direct_native_trace_pass` | true |
| `scheduleurm_native_trace_seed_pass` | true |
| `same_workload_direct_baseline_ready` | false |

## Dependency Imports

| Module | Importable | Detail |
|---|---:|---|
| `grpc` | true | `OK` |
| `grpc_tools` | true | `OK` |
| `func_timeout` | true | `OK` |
| `cvxpy` | true | `OK` |
| `matrix_completion` | true | `OK` |
| `scipy` | true | `OK` |

## Generated-Jobs Native Smoke

| Field | Value |
|---|---|
| status | `PASS` |
| returncode | `0` |
| command | `/usr/bin/python3 scripts/drivers/simulate_scheduler_with_generated_jobs.py -c 1:0:0 --num_gpus_per_server 1:1:1 -s 0 -e 2 -p fifo --seed 0 -i 60 -l 0 -f 60 --throughputs_file simulation_throughputs.json -v` |

## Scheduleurm Native Trace Seed Smoke

| Field | Value |
|---|---|
| status | `PASS` |
| returncode | `0` |
| command | `/usr/bin/python3 scripts/drivers/simulate_scheduler_with_trace.py -t /tmp/scheduleurm_gavel_direct_native_smoke/gavel/scheduler/scheduleurm_q01_native_trace_seed.trace -p fifo --throughputs_file simulation_throughputs.json -c 2:0:0 --num_gpus_per_server 2:1:1 --seed 0 --time_per_iteration 60 -s 0 -e 2` |
| trace file | `/tmp/scheduleurm_gavel_direct_native_smoke/gavel/scheduler/scheduleurm_q01_native_trace_seed.trace` |

## Native Trace Smoke

| Field | Value |
|---|---|
| status | `PASS` |
| returncode | `0` |
| command | `/usr/bin/python3 scripts/drivers/simulate_scheduler_with_trace.py -t traces/physical_cluster/small_test.trace -p fifo --throughputs_file simulation_throughputs.json -c 2:0:0 --num_gpus_per_server 2:1:1 --seed 0 --time_per_iteration 60 -s 0 -e 2` |
| classifier | `PASS` |

## Blockers

- Scheduleurm-to-Gavel native trace seed is a compatibility mapping; measured service-unit and policy-semantics equivalence still require validation before direct SOTA comparison

## Scope

Isolated Gavel native-stack smoke audit.  A pass means the local clone can be prepared and its entrypoint can start in a temporary copy.  It does not mean Gavel has been run as a same-workload full-stack Scheduleurm baseline.
