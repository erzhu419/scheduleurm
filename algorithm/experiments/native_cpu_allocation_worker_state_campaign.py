"""State-certified natural-completion CPU allocation-worker campaign.

This campaign fills the allocation-worker axis that is intentionally distinct
from independent-task colocation profiles.  It bypasses the legacy scheduler,
uses all available node001-node006 lanes, and accepts a measurement only when:

* a five-window dispatch survey matches the node's preregistered load regime;
* the benchmark exits naturally;
* the completion model includes startup, outer-loop, checkpoint, and final save;
* task-native progress/tqdm observations reach the declared total.

External work is never synthesized or relabelled.  Non-idle lanes only receive
worker counts that fit below the measured external-load p95 plus a safety
reserve, so a calibration probe does not consume another user's CPU allocation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .native_cpu_colocation_state_targeted_campaign import (
    _deploy_nodes,
    _survey_nodes,
)
from .phase_aware_cpu_completion_matrix import (
    CpuCompletionCell,
    _run_cell,
)
from .remote_workload_selected_profile_probe import (
    RUN_ROOT,
    _deploy_progress_wrapper,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
DEFAULT_PROFILES = (2, 4, 8, 16, 32, 64, 96, 128, 180, 192)
IDLE_REGIME = "cpu_external_idle"
LOGICAL_CPUS = 192


def build_native_cpu_allocation_worker_state_campaign(
    *,
    allow_launch: bool = False,
    tag: str = "native_cpu_allocation_worker_state_20260727",
    nodes: Sequence[str] = DEFAULT_NODES,
    profiles: Sequence[int] = DEFAULT_PROFILES,
    idle_replicates: int = 3,
    loaded_replicates: int = 1,
    idle_replicate_offset: int = 0,
    loaded_replicate_offset: int = 0,
    safety_reserve_cores: int = 8,
    max_parallel: int = 6,
) -> dict[str, Any]:
    node_values = tuple(dict.fromkeys(str(node) for node in nodes))
    profile_values = tuple(sorted({max(1, int(value)) for value in profiles}))
    if not node_values:
        raise ValueError("at least one controlled node is required")
    if not profile_values:
        raise ValueError("at least one worker profile is required")
    if int(idle_replicates) <= 0 or int(loaded_replicates) <= 0:
        raise ValueError("replicate counts must be positive")
    if int(idle_replicate_offset) < 0 or int(loaded_replicate_offset) < 0:
        raise ValueError("replicate offsets must be nonnegative")

    started = time.time()
    deploy_rows = _deploy_nodes(
        node_values,
        tag=f"{tag}_worker_campaign",
        max_parallel=max_parallel,
    )
    deploy_ready = {
        str(row["node"])
        for row in deploy_rows
        if bool(row.get("ready"))
    }
    progress_deploy_rows = _deploy_progress_nodes(
        tuple(node for node in node_values if node in deploy_ready),
        tag=tag,
        max_parallel=max_parallel,
    )
    progress_deploy_ready = {
        str(row["node"])
        for row in progress_deploy_rows
        if bool(row.get("ready"))
    }
    survey_rows = _survey_nodes(
        tuple(
            node
            for node in node_values
            if node in deploy_ready and node in progress_deploy_ready
        ),
        tag=f"{tag}_worker_campaign",
        max_parallel=max_parallel,
    )
    survey_by_node = {
        str(row["node"]): row
        for row in survey_rows
        if bool(row.get("ready"))
    }
    target_regimes = {
        node: str(row["dispatch_external_cpu_regime"])
        for node, row in survey_by_node.items()
    }
    safe_profiles_by_node = {
        node: _safe_profiles_for_survey(
            profile_values,
            row,
            logical_cpus=LOGICAL_CPUS,
            safety_reserve_cores=safety_reserve_cores,
        )
        for node, row in survey_by_node.items()
    }
    plan_rows = _build_plan(
        nodes=tuple(node for node in node_values if node in survey_by_node),
        profiles=profile_values,
        target_regimes=target_regimes,
        safe_profiles_by_node=safe_profiles_by_node,
        idle_replicates=idle_replicates,
        loaded_replicates=loaded_replicates,
        idle_replicate_offset=idle_replicate_offset,
        loaded_replicate_offset=loaded_replicate_offset,
    )

    if not allow_launch:
        rows = [
            {**row, "launched": False, "status": "MANIFEST"}
            for row in plan_rows
        ]
    else:
        rows = _run_node_lanes(
            plan_rows,
            tag=tag,
            max_parallel=max_parallel,
        )

    accepted = [row for row in rows if bool(row.get("measurement_valid"))]
    deferred = [
        row
        for row in rows
        if str(row.get("status") or "").startswith("DEFERRED")
    ]
    failed = [
        row
        for row in rows
        if row.get("launched")
        and not row.get("measurement_valid")
        and row not in deferred
    ]
    cell_counts = _cell_counts(rows)
    required_counts = {
        f"{regime}|w{profile}": (
            int(idle_replicates)
            if regime == IDLE_REGIME
            else int(loaded_replicates)
        )
        for regime in sorted(set(target_regimes.values()))
        for profile in profile_values
        if any(
            target_regimes.get(node) == regime
            and profile in safe_profiles_by_node.get(node, ())
            for node in node_values
        )
    }
    coverage_ready = bool(required_counts) and all(
        int(cell_counts.get(key, 0)) >= count
        for key, count in required_counts.items()
    )
    return {
        "gate": "native_cpu_allocation_worker_state_campaign",
        "schema_version": 1,
        "allow_launch": bool(allow_launch),
        "tag": tag,
        "nodes": list(node_values),
        "profiles": list(profile_values),
        "idle_replicates": int(idle_replicates),
        "loaded_replicates": int(loaded_replicates),
        "idle_replicate_offset": int(idle_replicate_offset),
        "loaded_replicate_offset": int(loaded_replicate_offset),
        "safety_reserve_cores": int(safety_reserve_cores),
        "measurement_contract": {
            "eta_source": "task_native_tqdm_plus_natural_completion_model",
            "completion_components": [
                "startup",
                "outer_loop",
                "periodic_checkpoint",
                "final_save",
            ],
            "legacy_scheduler_limits_bypassed": True,
            "dispatch_state_protocol": "five_window_exact_regime_preflight",
            "non_idle_capacity_rule": (
                "workers <= logical_cpus - dispatch_external_cpu_core_equiv_p95 "
                "- safety_reserve_cores"
            ),
        },
        "deploy_rows": deploy_rows,
        "progress_deploy_rows": progress_deploy_rows,
        "dispatch_survey_rows": survey_rows,
        "target_regimes": target_regimes,
        "safe_profiles_by_node": {
            node: list(values)
            for node, values in safe_profiles_by_node.items()
        },
        "planned_count": len(plan_rows),
        "launched_count": sum(bool(row.get("launched")) for row in rows),
        "accepted_count": len(accepted),
        "deferred_count": len(deferred),
        "failed_count": len(failed),
        "accepted_cell_counts": cell_counts,
        "required_cell_counts": required_counts,
        "coverage_ready": coverage_ready,
        "status": (
            "PASS"
            if allow_launch and coverage_ready and not failed
            else ("MANIFEST" if not allow_launch else "PARTIAL")
        ),
        "elapsed_wall_s": time.time() - started,
        "rows": rows,
        "selection_independence": (
            "The plan is fixed from node identity, worker profile, replicate index, "
            "and pre-dispatch load state. No completion time, ETA, or service-rate "
            "observation is read when assigning later cells."
        ),
    }


def _safe_profiles_for_survey(
    profiles: Sequence[int],
    survey: Mapping[str, Any],
    *,
    logical_cpus: int,
    safety_reserve_cores: int,
) -> tuple[int, ...]:
    regime = str(survey.get("dispatch_external_cpu_regime") or "")
    if regime == IDLE_REGIME:
        return tuple(int(value) for value in profiles if int(value) <= logical_cpus)
    external_p95 = max(
        0.0,
        float(survey.get("dispatch_external_cpu_fraction_p95") or 0.0)
        * float(logical_cpus),
    )
    budget = max(
        1,
        int(math.floor(float(logical_cpus) - external_p95 - safety_reserve_cores)),
    )
    return tuple(int(value) for value in profiles if int(value) <= budget)


def _build_plan(
    *,
    nodes: Sequence[str],
    profiles: Sequence[int],
    target_regimes: Mapping[str, str],
    safe_profiles_by_node: Mapping[str, Sequence[int]],
    idle_replicates: int,
    loaded_replicates: int,
    idle_replicate_offset: int = 0,
    loaded_replicate_offset: int = 0,
) -> list[dict[str, Any]]:
    nodes_by_regime: dict[str, list[str]] = {}
    for node in nodes:
        regime = str(target_regimes[node])
        nodes_by_regime.setdefault(regime, []).append(node)

    plan: list[dict[str, Any]] = []
    for regime, regime_nodes in sorted(nodes_by_regime.items()):
        replicate_count = (
            int(idle_replicates)
            if regime == IDLE_REGIME
            else int(loaded_replicates)
        )
        replicate_offset = (
            int(idle_replicate_offset)
            if regime == IDLE_REGIME
            else int(loaded_replicate_offset)
        )
        cursor = 0
        for local_replicate in range(replicate_count):
            replicate = replicate_offset + local_replicate
            rotated = (
                tuple(profiles[local_replicate % len(profiles) :])
                + tuple(profiles[: local_replicate % len(profiles)])
            )
            for profile in rotated:
                candidates = [
                    node
                    for node in regime_nodes
                    if int(profile) in {
                        int(value)
                        for value in safe_profiles_by_node.get(node, ())
                    }
                ]
                if not candidates:
                    continue
                node = candidates[cursor % len(candidates)]
                cursor += 1
                plan.append(
                    {
                        "cell_id": (
                            f"{regime}_w{int(profile)}_r{int(replicate)}"
                        ),
                        "node": node,
                        "workers": int(profile),
                        "replicate": int(replicate),
                        "target_regime": regime,
                        "resource_state": _resource_state_for_regime(regime),
                    }
                )
    return plan


def _run_node_lanes(
    plan_rows: Sequence[Mapping[str, Any]],
    *,
    tag: str,
    max_parallel: int,
) -> list[dict[str, Any]]:
    lanes: dict[str, list[dict[str, Any]]] = {}
    for row in plan_rows:
        lanes.setdefault(str(row["node"]), []).append(dict(row))

    def run_lane(node: str, lane: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in lane:
            preflight = _survey_nodes(
                (node,),
                tag=f"{tag}_{row['cell_id']}_preflight",
                max_parallel=1,
            )[0]
            observed_regime = str(
                preflight.get("dispatch_external_cpu_regime") or ""
            )
            if (
                not preflight.get("ready")
                or observed_regime != str(row["target_regime"])
            ):
                output.append(
                    {
                        **dict(row),
                        "launched": False,
                        "measurement_valid": False,
                        "status": "DEFERRED_REGIME_MISMATCH",
                        "dispatch_preflight": preflight,
                    }
                )
                continue

            cell = CpuCompletionCell(
                cell_id=str(row["cell_id"]),
                workload_key="cpu_heavy_local_bench",
                workload_label="cpu",
                node=node,
                workers=int(row["workers"]),
                steps=160,
                work_items=90_000,
                checkpoint_interval=40,
                checkpoint_bytes=8 * 1024 * 1024,
                resource_state=str(row["resource_state"]),
                observed_external_load=float(
                    preflight.get("dispatch_external_cpu_fraction") or 0.0
                )
                * LOGICAL_CPUS,
            )
            measured = _run_cell(
                cell,
                repeat_index=int(row["replicate"]),
                run_prefix=tag,
                deploy_bundle=False,
            )
            model = measured.get("completion_model") or {}
            natural_ready = bool(
                measured.get("measurement_valid")
                and measured.get("completion_model_ready")
                and model.get("natural_exit")
                and not model.get("stopped_on_stable")
                and float(model.get("total_wall_s") or 0.0) > 0.0
                and float(model.get("save_observed_s") or 0.0) >= 0.0
            )
            output.append(
                {
                    **measured,
                    "replicate": int(row["replicate"]),
                    "target_regime": str(row["target_regime"]),
                    "dispatch_preflight": preflight,
                    "measurement_valid": natural_ready,
                    "status": "READY" if natural_ready else "NOT_READY",
                }
            )
        return output

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_parallel), len(lanes) or 1))
    ) as pool:
        futures = {
            pool.submit(run_lane, node, lane): node
            for node, lane in sorted(lanes.items())
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(
        key=lambda row: (
            str(row.get("target_regime") or ""),
            int(row.get("replicate") or 0),
            int(row.get("workers") or 0),
            str(row.get("node") or ""),
        )
    )
    return rows


def _deploy_progress_nodes(
    nodes: Sequence[str],
    *,
    tag: str,
    max_parallel: int,
) -> list[dict[str, Any]]:
    def deploy(node: str) -> dict[str, Any]:
        run_id = f"{tag}_{node}_progress_deploy_once"
        raw_dir = RUN_ROOT / run_id / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_root = (
            f"/home/zhengliang01/scheduleurm_work/eta_completion/{run_id}"
        )
        try:
            _deploy_progress_wrapper(
                node,
                raw_dir,
                cwd="/tmp",
                output_root=output_root,
            )
            return {"node": node, "ready": True, "error": None}
        except Exception as exc:
            return {
                "node": node,
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    rows = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_parallel), len(nodes) or 1))
    ) as pool:
        futures = {pool.submit(deploy, node): node for node in nodes}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: str(row.get("node") or ""))
    return rows


def _cell_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not row.get("measurement_valid"):
            continue
        key = f"{row.get('target_regime')}|w{int(row.get('workers') or 0)}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _resource_state_for_regime(regime: str) -> str:
    return {
        "cpu_external_idle": "empty",
        "cpu_external_light": "cpu_external_light",
        "cpu_external_moderate": "cpu_external_moderate",
        "cpu_external_heavy": "cpu_external_heavy",
        "cpu_external_saturated": "cpu_external_saturated",
    }.get(str(regime), "unknown_external_state")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in str(raw or "").split(",")
            if item.strip()
        )
    )


def _parse_profiles(raw: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                max(1, int(item.strip()))
                for item in str(raw or "").split(",")
                if item.strip()
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument(
        "--tag",
        default="native_cpu_allocation_worker_state_20260727",
    )
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument(
        "--profiles",
        default=",".join(str(value) for value in DEFAULT_PROFILES),
    )
    parser.add_argument("--idle-replicates", type=int, default=3)
    parser.add_argument("--loaded-replicates", type=int, default=1)
    parser.add_argument("--idle-replicate-offset", type=int, default=0)
    parser.add_argument("--loaded-replicate-offset", type=int, default=0)
    parser.add_argument("--safety-reserve-cores", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ARTIFACT_ROOT
            / "native_cpu_allocation_worker_state_campaign_20260727.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_native_cpu_allocation_worker_state_campaign(
        allow_launch=bool(args.allow_launch),
        tag=str(args.tag),
        nodes=_parse_csv(args.nodes),
        profiles=_parse_profiles(args.profiles),
        idle_replicates=int(args.idle_replicates),
        loaded_replicates=int(args.loaded_replicates),
        idle_replicate_offset=int(args.idle_replicate_offset),
        loaded_replicate_offset=int(args.loaded_replicate_offset),
        safety_reserve_cores=int(args.safety_reserve_cores),
        max_parallel=int(args.max_parallel),
    )
    result["artifact_sha256"] = _canonical_sha256(result)
    _atomic_write_json(args.output, result)
    print(args.output)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "planned_count",
                    "launched_count",
                    "accepted_count",
                    "deferred_count",
                    "failed_count",
                    "coverage_ready",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if not args.allow_launch or result.get("coverage_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
