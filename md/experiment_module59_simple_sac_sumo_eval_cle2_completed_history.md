# Module59 SimpleSAC SUMO Eval c_le2 Completed-History Certificate

Date: 2026-06-08

Module59 attacks the `sumo_eval_cpu|c_le2` blocker after Module58.  The largest
clean sub-family is SimpleSAC `run_multiseed_eval.sh`: each task evaluates one
method, one SUMO seed, and one OD scale, then writes one JSON result.

## Scope

```text
sub_bucket = sumo_eval_cpu|c_le2
workload_key = sumo_eval_simple_sac_c_le2
node_bucket = local:cpu
resource_kind = cpu_sumo_transit
vram_mb/task = 0
cpu/task <= 2
progress_unit = eval_json
profile_domain = [1]
strict command shape = clean run_multiseed_eval.sh <method> <sumo_seed> <od_scale>
```

This is not a 1/2/4/8 live co-location curve.  The clean SimpleSAC eval tasks
do not expose stable progress before the final JSON is written, and the
completed production history only observed reliable profile-1 lower service.
Higher co-location profiles remain unclaimed.

## Artifacts

```text
md/experiment_artifacts/module59_simple_sac_sumo_eval_cle2_completed_history.json
md/experiment_artifacts/module59_simple_sac_sumo_eval_cle2_completed_history.md
md/experiment_artifacts/module59_simple_sac_sumo_eval_cle2_completed_history_reports/
```

## Completed-History Service Point

The service cache loads only profile 1, using the minimum realized completed
rate as a conservative lower-service point.

| Profile | Source | Aggregate eval/s | Mean eval/s |
|---:|---|---:|---:|
| 1 | completed-history min rate | 0.001525164 | 0.001525164 |

Completed-task audit:

```text
completed-active mapped count = 54
raw-history mapped count = 71
max completed duration = 655.667244 s
median completed duration = 498.933274 s
min realized eval/s = 0.001525164
```

## Production Closure Effect

After Module59:

```text
completed_active_production records = 2458
strict mapped = 290
representative mapped = 1080
measurement_required = 1378
remaining cpu_sumo_transit_eval_or_control = 1049
```

The mapped capacity LP remains positive but tight:

```text
strict mapped delta = 0.001497772
representative mapped delta = 0.001497772
```

This tight slack is a real theorem-condition warning.  Module59 improves
coverage, but its conservative profile-1 lower-service certificate consumes
most of the measured mapped capacity slack.  Later global closure should either
measure higher SimpleSAC co-location profiles with real progress instrumentation
or keep this slice as a conservative low-throughput service class.
