"""Complete-frame point catalogues from every measured three-class GPU node.

The node roster and lane counts come from the cache, before any reduction or
performance comparison.  Each node uses its own dedicated p1 native job sizes
and measured phase rates.  Loaded options retain the original complete-frame
composition: fresh resident initialization, target overlap, dedicated resident
drain, and resident finalization.  These deterministic cached phase models do
not add a physical-execution confidence guarantee.
"""
from __future__ import annotations

import json
from pathlib import Path

from algorithm.experiments import unified_hardware_or_replay as replay
from algorithm.experiments.aligned_hardware_frame_model import (
    DEFAULT_CACHE, DEFAULT_LOADED_LEDGER, WORKLOADS, FrameModel, FrameOption,
)


def load_catalogues(
    cache_path: Path = DEFAULT_CACHE,
    *,
    loaded_ledger_path: Path = DEFAULT_LOADED_LEDGER,
) -> tuple[FrameModel, ...]:
    """Build all measured q01 GPU catalogues without borrowing across nodes.

    ``source_action_ids`` identify the dedicated cache records and loaded ledger
    entries used by each option.  Their stored ``source`` fields provide the
    underlying measurement-artifact lineage.  Excluded RL residents have no
    complete-job coordinate or dedicated p1 drain in this three-class model.
    """
    payload = json.loads(Path(cache_path).read_text())
    _, rows, exact = replay._validate_cache(payload)
    ledger = json.loads(Path(loaded_ledger_path).read_text())
    loaded = replay._validate_loaded_ledger(ledger, exact=exact)
    ledger_rows = {row["action_id"]: row for row in ledger["actions"]}
    buckets = sorted({
        row.node_bucket for row in rows
        if row.resource_kind == "gpu" and row.workload_key in WORKLOADS
    })
    models: list[FrameModel] = []
    for bucket in buckets:
        dedicated = tuple(
            row for row in rows
            if row.node_bucket == bucket
            and row.resource_kind == "gpu"
            and row.workload_key in WORKLOADS
            and row.completion_model_ready
            and not row.capacity_boundary
            and row.resource_state == "empty"
            and not row.resident_mix
        )
        singletons = {}
        for workload in WORKLOADS:
            candidates = [row for row in dedicated
                          if row.workload_key == workload and row.colocation_count == 1]
            if len(candidates) != 1:
                raise ValueError(f"{bucket}/{workload}: requires one dedicated p1 completion model")
            singletons[workload] = candidates[0]
        evaluations = {
            replay._record_action_id(row): replay._evaluation_from_record(
                replay._record_action_id(row), row, kind="aligned_dedicated"
            ) for row in dedicated
        }
        scales = tuple(evaluations[replay._record_action_id(singletons[key])].replay_job_units
                       for key in WORKLOADS)
        lane_counts = {int(row.gpu_count_observed) for row in dedicated}
        if len(lane_counts) != 1 or min(lane_counts) <= 0:
            raise ValueError(f"{bucket}: dedicated GPU lane metadata disagree")
        options: list[FrameOption] = []
        for row in dedicated:
            action_id = replay._record_action_id(row)
            evaluation = evaluations[action_id]
            singleton = singletons[row.workload_key]
            coordinate = WORKLOADS.index(row.workload_key)
            if row.allocation_workers != 1 or row.colocation_count != row.profile:
                raise ValueError(f"{action_id}: GPU colocation width differs from profile")
            if (row.workload_env, replay._work_unit(row)) != (
                singleton.workload_env, replay._work_unit(singleton)
            ):
                raise ValueError(f"{action_id}: dedicated environment or native unit mismatch")
            if evaluation.replay_job_units != scales[coordinate]:
                raise ValueError(f"{action_id}: dedicated native job sizes disagree")
            width = int(row.colocation_count)
            loop = width * scales[coordinate] / evaluation.phase_service_rate
            duration = evaluation.fixed_overhead_s + loop
            options.append(FrameOption(
                action_id=action_id,
                requirements=tuple(width if i == coordinate else 0 for i in range(3)),
                duration=duration,
                completion_offsets=tuple((duration,) * width if i == coordinate else ()
                                         for i in range(3)),
                source_action_ids=(action_id,),
                phases=(("initialization_s", float(row.startup_overhead_s)),
                        ("loop_s", loop),
                        ("finalization_s", float(row.finalization_overhead_s))),
            ))
        excluded: list[dict[str, str]] = []
        for action in loaded:
            if action.lower.node_bucket != bucket or action.lower.workload_key not in WORKLOADS:
                continue
            required = action.lower.job_requirements()
            if not set(required).issubset(WORKLOADS):
                excluded.append({
                    "action_id": action.lower.action_id,
                    "reason": "resident class has no arrivals in the three-class q01 protocol",
                })
                continue
            source = ledger_rows[action.lower.action_id]
            for role in ("target", "resident"):
                key = source[f"{role}_workload_key"]
                singleton = singletons[key]
                record = replay._exact_record(
                    exact, workload_key=key,
                    workload_env=source[f"{role}_workload_env"],
                    node_bucket=bucket, resource_state="mixed_colocation",
                    resident_mix=source["resident_mix"], profile=source["profile"],
                    allocation_workers=1, colocation_count=source["profile"],
                )
                if (record.workload_env, replay._work_unit(record)) != (
                    singleton.workload_env, replay._work_unit(singleton)
                ):
                    raise ValueError(f"{action.lower.action_id}: {role} environment or native unit mismatch")
            target = WORKLOADS.index(action.lower.workload_key)
            if action.evaluation.replay_job_units != scales[target]:
                raise ValueError(f"{action.lower.action_id}: target native job size mismatch")
            resident_key = next(key for key in required if key != action.lower.workload_key)
            resident = WORKLOADS.index(resident_key)
            resident_record = singletons[resident_key]
            resident_id = replay._record_action_id(resident_record)
            resident_evaluation = evaluations[resident_id]
            initialization = float(resident_record.startup_overhead_s)
            target_frame = (action.evaluation.fixed_overhead_s
                            + scales[target] / action.evaluation.phase_service_rate)
            resident_rate = action.lower.service_vector()[resident_key]
            residual = max(0.0, scales[resident] - resident_rate * target_frame)
            drain = residual / resident_evaluation.phase_service_rate
            finalization = float(resident_record.finalization_overhead_s)
            target_finish = initialization + target_frame
            duration = target_finish + drain + finalization
            options.append(FrameOption(
                action_id=action.lower.action_id,
                requirements=tuple(required.get(key, 0) for key in WORKLOADS),
                duration=duration,
                completion_offsets=tuple((target_finish,) if i == target else
                                         (duration,) if i == resident else () for i in range(3)),
                loaded=True,
                source_action_ids=(action.lower.action_id, resident_id),
                phases=(("resident_initialization_s", initialization),
                        ("target_overlap_s", target_frame),
                        ("resident_overlap_rate_native_per_s", resident_rate),
                        ("resident_remaining_native_units", residual),
                        ("resident_dedicated_drain_s", drain),
                        ("resident_finalization_s", finalization)),
            ))
        models.append(FrameModel(
            node=bucket.rsplit(":", 1)[1],
            scenario_id=f"q01:{replay._safe_id(bucket)}",
            lanes=next(iter(lane_counts)), workloads=WORKLOADS, scales=scales,
            options=tuple(sorted(options, key=lambda option: option.action_id)),
            excluded_actions=tuple(excluded),
        ))
    return tuple(models)
