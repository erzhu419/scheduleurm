# OR Generalization Benchmarks

- Status: `OR_GENERALIZATION_SIMULATOR_PASS`
- Pass: `true`

| Domain | Rows | Selected actions | Migration analogue |
|---|---:|---|---:|
| `port_terminal` | 3 | `['port:ship_b:berth_slow_qc1', 'port:ship_a:reberth_fast']` | true |
| `fjsp` | 3 | `['fjsp:job1_op1:machine_a', 'fjsp:job2_op2:reroute_machine_c']` | true |
| `mmrcpsp` | 3 | `['mmrcpsp:activity_1:mode_low_resource', 'mmrcpsp:activity_2:mode_switch_fast']` | true |

## Scope

Simulator/replay action-model coverage for OR benchmark families; not physical-port empirical validation.
