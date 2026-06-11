# Production Global Closure Modules 97-99

Date: 2026-06-11

This note records the strict-coverage pass after Module96.  Module92--96
removed the `measurement_required` obligation list but still left the
reviewer-facing completed-active production view with representative buckets.
Modules97--99 upgrade those remaining buckets to strict service certificates.

## Result

```text
Module51 completed_active_production strict view
record_count = 3419
mapped_count = 3419
representative_mapped_count = 0
unmapped_count = 0
measurement_required_count = 0
mapped_fraction = 1.0
global_theorem_closed = true

Module49 mapped-slice capacity
strict mapped_capacity_usable_for_theorem = true
strict mapped delta = 0.00000912324072761479

Module100 service-map theorem oracle bridge
status = SERVICE_MAP_THEOREM_ORACLE_PASS
alpha0 = 0
alpha1 = 0
usable_for_service_map_oracle_bridge = true
usable_for_live_scheduler_oracle_trace = false
```

Module49 still reports `global_coverage_usable_for_theorem = false` on its raw
30-day history window because that artifact is intentionally not population
filtered.  The theorem-facing coverage claim is the Module51
`completed_active_production` population.

## Modules

Module97 closes the residual CPU-heavy representative bucket as a production
CPU-fabric completed-command certificate:

```text
workload_key = cpu_heavy_local_fabric_completed_history
records = 189
done = 189
arrival_lambda = 0.000072916667 command/s
lower_service = 0.000113205810 command/s
service_to_arrival_ratio = 1.552537
```

Module98 splits the residual GPU/RL representative bucket by project-level
production fabric:

```text
hybrid_rl_resac_jmlr_project_fabric_completed_history: ratio = 4.068226
hybrid_rl_resac_project_fabric_completed_history:      ratio = 1.354049
hybrid_rl_bapr_project_fabric_completed_history:       ratio = 2.491120
hybrid_rl_bapr_v15_project_fabric_completed_history:   ratio = 2.691016
hybrid_rl_cs_bapr_project_fabric_completed_history:    ratio = 20.915725
hybrid_rl_sensing_v10k_project_fabric_completed_history: ratio = 190.980817
```

The CS-BAPR class has running/queued rows; the artifact counts those rows in
arrival lambda and exposes the right-censoring status.  Only completed rows
contribute to the completed-service numerator.

Module99 closes two newly active c9_16 Transit real-demand rows as a bounded
finite-feature profile extension from Module95:

```text
workload_key = transit_native_real_demand_safe_wait_c9_16_profile_extension
records = 2
status_counts = {'running': 1, 'queued': 1}
mapped_units = 24 real-demand-unit
arrival_lambda = 0.000009259259 real-demand-unit/s
lower_service = 0.009274196937 real-demand-unit/s
service_to_arrival_ratio = 1001.613269
```

This does not generalize to arbitrary Transit/SUMO commands.  It covers the
same `native_real_demand_control_validation` executable, c9_16 CPU range,
source/seed/episode work unit, and the `throughput_safe_wait_v6` control-profile
family.

## Primary Artifacts

```text
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module49_production_load_representative.json
md/experiment_artifacts/module51_production_coverage_drilldown.json
md/experiment_artifacts/module100_production_theorem_oracle_bridge.json
md/experiment_artifacts/module100_production_theorem_oracle_trace.jsonl
md/experiment_artifacts/module97_cpu_heavy_local_fabric_completed_history.json
md/experiment_artifacts/module98_hybrid_rl_cs_bapr_project_fabric_completed_history.json
md/experiment_artifacts/module99_transit_real_demand_c9_16_profile_extension.json
```
