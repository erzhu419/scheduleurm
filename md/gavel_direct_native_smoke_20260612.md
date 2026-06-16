# Gavel Direct Native Smoke

## Summary

| Quantity | Value |
|---|---:|
| `pass` | false |
| `dependency_import_pass` | false |
| `protobuf_stub_generation_pass` | false |
| `entrypoint_help_pass` | false |
| `native_generated_jobs_pass` | false |
| `direct_native_trace_pass` | false |
| `scheduleurm_native_trace_seed_pass` | false |
| `same_workload_direct_baseline_ready` | false |

## Dependency Imports

| Module | Importable | Detail |
|---|---:|---|
| `grpc` | true | `OK` |
| `grpc_tools` | false | `Traceback (most recent call last):   File "<string>", line 1, in <module> ModuleNotFoundError: No module named 'grpc_tools'` |
| `func_timeout` | false | `Traceback (most recent call last):   File "<string>", line 1, in <module> ModuleNotFoundError: No module named 'func_timeout'` |
| `cvxpy` | false | `Traceback (most recent call last):   File "<string>", line 1, in <module> ModuleNotFoundError: No module named 'cvxpy'` |
| `matrix_completion` | false | `Traceback (most recent call last):   File "<string>", line 1, in <module> ModuleNotFoundError: No module named 'matrix_completion'` |
| `scipy` | true | `OK` |

## Generated-Jobs Native Smoke

| Field | Value |
|---|---|
| status | `NOT_RUN` |
| returncode | `None` |
| command | `` |

## Scheduleurm Native Trace Seed Smoke

| Field | Value |
|---|---|
| status | `NOT_RUN` |
| returncode | `None` |
| command | `` |
| trace file | `` |

## Native Trace Smoke

| Field | Value |
|---|---|
| status | `NOT_RUN` |
| returncode | `None` |
| command | `` |
| classifier | `` |

## Blockers

- isolated Gavel dependency imports are incomplete
- Gavel protobuf stubs cannot be generated in the isolated copy
- Gavel native entrypoint help smoke does not pass
- Gavel generated-jobs native simulator smoke does not pass
- Gavel bounded native trace smoke is not a usable direct baseline
- Scheduleurm-to-Gavel native trace seed smoke does not pass
- Scheduleurm-to-Gavel native trace seed is a compatibility mapping; measured service-unit and policy-semantics equivalence still require validation before direct SOTA comparison

## Scope

Isolated Gavel native-stack smoke audit.  A pass means the local clone can be prepared and its entrypoint can start in a temporary copy.  It does not mean Gavel has been run as a same-workload full-stack Scheduleurm baseline.
