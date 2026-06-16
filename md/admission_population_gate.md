# Service-Certified Admission Population

## Summary

| Population | Records | Strict admitted | Domain admitted | Closed |
|---|---:|---:|---:|---:|
| completed-active production | 4306 | 3857 | 4303 | false |
| attempted production | 5929 | 4279 | 5924 | false |
| active queue | 11 | 10 | 11 | false |

## Scope

the reviewer-facing theorem population is completed-active production within the fixed window. Raw history and attempted-only records are reported but are not claimed as the theorem arrival stream. The live theorem dispatch mode should run with SCHEDULEURM_THEOREM_UNCERTIFIED_MODE=block when future tasks are to be admitted into the theorem population.

## Completed-Active Service-Domain Blockers

| Task | Status | Project | Reason | Inferred workload |
|---|---|---|---|---|
| `t11229` | `done` | `sched` | `workload_key_not_inferred` | `` |
| `t11231` | `done` | `sched` | `workload_key_not_inferred` | `` |
| `t11329` | `done` | `freqduet` | `workload_key_not_inferred` | `` |

## Strict-Classifier Non-Matches

These rows are service-domain admitted but do not have a strict
production classifier rule. They should be described as admission
certified, not as strict classifier coverage.

| Task | Status | Project | Classifier reason | Inferred workload |
|---|---|---|---|---|
| `t10018` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10038` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10124` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10233` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10257` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10295` | `done` | `TransitDuet` | `unmapped_cpu` | `transit_freqhrl_import_smoke_c_le2_completed_history` |
| `t10298` | `done` | `TransitDuet` | `unmapped_cpu` | `transit_freqhrl_import_smoke_c_le2_completed_history` |
| `t10299` | `done` | `TransitDuet` | `unmapped_cpu` | `transit_freqhrl_import_smoke_c_le2_completed_history` |
| `t10321` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10332` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10346` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10347` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10381` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10390` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10436` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10470` | `done` | `CS-BAPR` | `unmapped_cpu` | `hybrid_rl_resac_ant` |
| `t10500` | `done` | `EnvBootstrap` | `unmapped_cpu` | `scheduleurm_hpc_relay_smoke_completed_history` |
| `t10547` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10564` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10588` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10592` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10599` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10600` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10604` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10605` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10606` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10607` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10618` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10620` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10633` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10635` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10639` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10648` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10652` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10654` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10659` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_unittest_completed_history` |
| `t10663` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10761` | `done` | `scheduleurm_bench_cwd` | `unmapped_gpu` | `gpu_heavy_jax_matmul` |
| `t10762` | `done` | `scheduleurm_bench_cwd` | `unmapped_gpu` | `gpu_heavy_jax_matmul` |
| `t10763` | `done` | `scheduleurm_bench_cwd` | `unmapped_gpu` | `gpu_heavy_jax_matmul` |
