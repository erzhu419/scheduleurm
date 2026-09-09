"""State-targeted split-conformal campaign for exact native CPU cells.

The campaign freezes one hardware-normalized dispatch regime per node from a
pre-workload survey.  It then runs each workload/node/profile cell only when a
fresh five-window preflight matches that target.  State-mismatched attempts do
not launch a controlled workload.  Training, calibration, and holdout rows are
selected without reading completion time, ETA error, or service rate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)
from .native_cpu_colocation_completion_matrix import (
    ARTIFACT_ROOT,
    DEFAULT_KINDS,
    DEFAULT_NODES,
    DEFAULT_SEED_BASE,
    DISPATCH_CPU_REGIMES,
    DISPATCH_STATE_PREFIX,
    REMOTE_RUNNER,
    RUN_ROOT,
    _calibration_cell_id_for_cell,
    _deploy_remote_bundle,
    _parse_prefixed_json,
    build_native_cpu_colocation_completion_matrix,
    colocation_cells,
)
from .native_cpu_colocation_stochastic_calibration_campaign import (
    _acquire_node_locks,
    _release_node_locks,
)
from .phase_aware_colocation_stochastic_lcb_gate import (
    _row_ready,
    build_colocation_stochastic_lcb_gate,
)
from .remote_workload_selected_profile_probe import _run_remote_capture


SELECTION_PROTOCOL = "preregistered_dispatch_regime_preflight_v1"
TRAINING_REPEATS = 3
CALIBRATION_REPEATS = 9
HOLDOUT_REPEATS = 1
DEFAULT_MAX_ATTEMPTS_PER_CELL = 60
SEED_ROLE_STRIDE = 100_000
SEED_REPLICATE_STRIDE = 1_000


def run_state_targeted_colocation_campaign(
    *,
    tag: str,
    profile: int = 1,
    allow_launch: bool = False,
    nodes: Iterable[str] = DEFAULT_NODES,
    kinds: Iterable[str] = DEFAULT_KINDS,
    max_parallel: int = 6,
    max_attempts_per_cell: int = DEFAULT_MAX_ATTEMPTS_PER_CELL,
    seed_base: int = DEFAULT_SEED_BASE + 400_000,
    init_cache_state: str = "warm",
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    """Run a six-lane exact-cell campaign with dispatch-only acceptance."""

    started = time.time()
    safe_tag = _safe_id(tag)
    selected_nodes = _ordered_subset(nodes, DEFAULT_NODES, "node")
    selected_kinds = _ordered_subset(kinds, DEFAULT_KINDS, "kind")
    selected_profile = max(1, int(profile))
    parallelism = max(1, min(int(max_parallel), len(selected_nodes)))
    attempt_limit = max(1, int(max_attempts_per_cell))
    canonical_cells = colocation_cells(
        run_id=f"state_targeted_contract_{safe_tag}",
        nodes=selected_nodes,
        profiles=(selected_profile,),
        kinds=selected_kinds,
        tasks_per_node=selected_profile,
        seed_base=seed_base,
    )
    expected_cell_ids = tuple(
        sorted(_calibration_cell_id_for_cell(cell) for cell in canonical_cells)
    )
    manifest = {
        "campaign": "native_cpu_colocation_state_targeted_campaign",
        "schema_version": 1,
        "tag": safe_tag,
        "allow_launch": bool(allow_launch),
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "profile": selected_profile,
        "profile_axis": "colocation_count",
        "nodes": list(selected_nodes),
        "kinds": list(selected_kinds),
        "expected_cell_ids": list(expected_cell_ids),
        "expected_cell_count": len(expected_cell_ids),
        "training_repeat_count": TRAINING_REPEATS,
        "calibration_repeat_count": CALIBRATION_REPEATS,
        "holdout_repeat_count": HOLDOUT_REPEATS,
        "max_parallel": parallelism,
        "parallelism_scope": "nodes",
        "same_node_workloads_serialized": True,
        "max_attempts_per_cell": attempt_limit,
        "seed_base": int(seed_base),
        "init_cache_state": str(init_cache_state),
        "selection_uses_fields": [
            "measurement_valid",
            "dispatch_state_ready",
            "dispatch_sample_count",
            "dispatch_external_cpu_regime",
            "measurement_protocol",
            "eta_source_not_history",
        ],
        "performance_outcome_fields_used_for_selection": [],
        "legacy_scheduler_limits_bypassed": True,
        "controlled_benchmarks_only": list(selected_kinds),
        "user_task_control_operations": [],
    }
    if not allow_launch:
        return {
            **manifest,
            "status": "MANIFEST",
            "pass": False,
            "elapsed_wall_s": time.time() - started,
        }

    lock_handles: list[Any] = []
    lock_rows: list[dict[str, Any]] = []
    try:
        lock_handles, lock_rows = _acquire_node_locks(
            selected_nodes,
            tag=safe_tag,
        )
        deployment_rows = _deploy_nodes(
            selected_nodes,
            tag=safe_tag,
            max_parallel=parallelism,
        )
        if not all(row["ready"] for row in deployment_rows):
            return {
                **manifest,
                "node_lock_rows": lock_rows,
                "deployment_rows": deployment_rows,
                "status": "DEPLOYMENT_GAP",
                "pass": False,
                "elapsed_wall_s": time.time() - started,
            }

        survey_rows = _survey_nodes(
            selected_nodes,
            tag=safe_tag,
            max_parallel=parallelism,
        )
        if not all(row["ready"] for row in survey_rows):
            return {
                **manifest,
                "node_lock_rows": lock_rows,
                "deployment_rows": deployment_rows,
                "dispatch_survey_rows": survey_rows,
                "status": "DISPATCH_SURVEY_GAP",
                "pass": False,
                "elapsed_wall_s": time.time() - started,
            }
        target_regimes = {
            str(row["node"]): str(row["dispatch_external_cpu_regime"])
            for row in survey_rows
        }

        lane_results = _run_node_lanes(
            nodes=selected_nodes,
            kinds=selected_kinds,
            profile=selected_profile,
            tag=safe_tag,
            target_regimes=target_regimes,
            max_parallel=parallelism,
            max_attempts_per_cell=attempt_limit,
            seed_base=int(seed_base),
            init_cache_state=init_cache_state,
        )
        lane_ready = all(row.get("ready") for row in lane_results)
        if not lane_ready:
            return {
                **manifest,
                "node_lock_rows": lock_rows,
                "deployment_rows": deployment_rows,
                "dispatch_survey_rows": survey_rows,
                "target_dispatch_regimes": target_regimes,
                "lane_results": lane_results,
                "status": "TARGETED_MEASUREMENT_GAP",
                "pass": False,
                "elapsed_wall_s": time.time() - started,
            }

        paths_by_role = _write_composite_artifacts(
            tag=safe_tag,
            profile=selected_profile,
            lane_results=lane_results,
            expected_cell_ids=expected_cell_ids,
            target_regimes=target_regimes,
        )
        gate = build_colocation_stochastic_lcb_gate(
            training_paths=paths_by_role["training"],
            calibration_paths=paths_by_role["calibration"],
            holdout_path=paths_by_role["holdout"][0],
            miscoverage_alpha=float(miscoverage_alpha),
            expected_cell_ids=expected_cell_ids,
        )
        gate_path = ARTIFACT_ROOT / (
            f"native_cpu_colocation_state_targeted_p{selected_profile}_"
            f"lcb_gate_{safe_tag}.json"
        )
        _atomic_write_json(gate_path, gate)
        evidence_paths = [
            *paths_by_role["training"],
            *paths_by_role["calibration"],
            *paths_by_role["holdout"],
        ]
        gate_pass = bool(gate.get("pass"))
        return {
            **manifest,
            "node_lock_rows": lock_rows,
            "deployment_rows": deployment_rows,
            "dispatch_survey_rows": survey_rows,
            "target_dispatch_regimes": target_regimes,
            "lane_results": lane_results,
            "training_paths": [
                str(path) for path in paths_by_role["training"]
            ],
            "calibration_paths": [
                str(path) for path in paths_by_role["calibration"]
            ],
            "holdout_path": str(paths_by_role["holdout"][0]),
            "gate_path": str(gate_path),
            "lcb_gate_path": str(gate_path),
            "gate_pass": gate_pass,
            "gate": gate,
            "evidence_sha256": _file_sha256_map(evidence_paths),
            "status": "PASS" if gate_pass else "GATE_FAIL",
            "pass": gate_pass,
            "elapsed_wall_s": time.time() - started,
            "claim_boundary": (
                "The target external-load regime for each node is frozen by a "
                "five-window dispatch-only survey before any performance row is "
                "launched. Every later attempt repeats that preflight; a mismatch "
                "launches no controlled task. Rows are accepted only for protocol "
                "readiness and target-state equality, never by completion time, "
                "ETA error, or service rate. Training, calibration, and holdout "
                "are ordered separately within every exact workload/node/profile "
                "cell, so the resulting guarantee is cellwise rather than a "
                "simultaneous all-node guarantee."
            ),
        }
    except BlockingIOError as exc:
        return {
            **manifest,
            "node_lock_rows": lock_rows,
            "status": "NODE_LOCK_GAP",
            "error": str(exc),
            "pass": False,
            "elapsed_wall_s": time.time() - started,
        }
    finally:
        _release_node_locks(lock_handles)


def _deploy_nodes(
    nodes: Sequence[str],
    *,
    tag: str,
    max_parallel: int,
) -> list[dict[str, Any]]:
    def deploy(node: str) -> dict[str, Any]:
        raw = RUN_ROOT / _safe_id(f"{tag}_{node}_deploy_once") / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        try:
            _deploy_remote_bundle(node, raw)
            return {"node": node, "ready": True, "error": None}
        except Exception as exc:
            return {
                "node": node,
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return _parallel_by_node(nodes, max_parallel=max_parallel, fn=deploy)


def _survey_nodes(
    nodes: Sequence[str],
    *,
    tag: str,
    max_parallel: int,
) -> list[dict[str, Any]]:
    def survey(node: str) -> dict[str, Any]:
        raw = RUN_ROOT / _safe_id(f"{tag}_{node}_dispatch_survey") / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        rc, out, err = _run_remote_capture(
            node,
            f"python3 -u {REMOTE_RUNNER} --remote-dispatch-state",
            raw / "dispatch_survey",
            timeout_s=180,
        )
        payload = _parse_prefixed_json(out, DISPATCH_STATE_PREFIX)
        regime = str(
            (payload or {}).get("dispatch_external_cpu_regime") or ""
        ).strip()
        ready = bool(
            int(rc) == 0
            and payload
            and payload.get("measurement_protocol") == MEASUREMENT_PROTOCOL
            and payload.get("dispatch_state_ready")
            and int(payload.get("dispatch_sample_count") or 0)
            >= DISPATCH_CPU_SAMPLE_COUNT
            and regime in DISPATCH_CPU_REGIMES
        )
        return {
            "node": node,
            "ready": ready,
            "returncode": int(rc),
            "dispatch_external_cpu_regime": regime or None,
            "dispatch_sample_count": (
                payload.get("dispatch_sample_count") if payload else None
            ),
            "dispatch_external_cpu_fraction": (
                payload.get("dispatch_external_cpu_fraction")
                if payload
                else None
            ),
            "dispatch_external_cpu_fraction_p95": (
                payload.get("dispatch_external_cpu_fraction_p95")
                if payload
                else None
            ),
            "measurement_protocol": (
                payload.get("measurement_protocol") if payload else None
            ),
            "stderr_tail": (err or "")[-1000:],
        }

    return _parallel_by_node(nodes, max_parallel=max_parallel, fn=survey)


def _run_node_lanes(
    *,
    nodes: Sequence[str],
    kinds: Sequence[str],
    profile: int,
    tag: str,
    target_regimes: Mapping[str, str],
    max_parallel: int,
    max_attempts_per_cell: int,
    seed_base: int,
    init_cache_state: str,
    role_specs: Sequence[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    def run_lane(node: str) -> dict[str, Any]:
        selected: dict[str, dict[int, dict[str, Any]]] = {
            "training": {},
            "calibration": {},
            "holdout": {},
        }
        attempts = []
        selected_role_specs = tuple(
            role_specs
            or (
                ("training", TRAINING_REPEATS, 0),
                ("calibration", CALIBRATION_REPEATS, 1),
                ("holdout", HOLDOUT_REPEATS, 2),
            )
        )
        for role, repeat_count, role_index in selected_role_specs:
            for replicate in range(1, repeat_count + 1):
                selected[role][replicate] = {}
                for kind in kinds:
                    accepted, cell_attempts = _run_cell_until_accepted(
                        node=node,
                        kind=kind,
                        profile=profile,
                        role=role,
                        role_index=role_index,
                        replicate=replicate,
                        tag=tag,
                        target_regime=target_regimes[node],
                        max_attempts=max_attempts_per_cell,
                        seed_base=seed_base,
                        init_cache_state=init_cache_state,
                    )
                    attempts.extend(cell_attempts)
                    if accepted is None:
                        return {
                            "node": node,
                            "ready": False,
                            "target_dispatch_regime": target_regimes[node],
                            "selected": selected,
                            "attempts": attempts,
                            "gap": {
                                "role": role,
                                "replicate": replicate,
                                "kind": kind,
                                "reason": "attempt_limit_exhausted",
                            },
                        }
                    selected[role][replicate][
                        str(accepted["calibration_cell_id"])
                    ] = accepted
                print(
                    json.dumps(
                        {
                            "phase": role,
                            "replicate": replicate,
                            "node": node,
                            "target_dispatch_regime": target_regimes[node],
                            "accepted_kind_count": len(kinds),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return {
            "node": node,
            "ready": True,
            "target_dispatch_regime": target_regimes[node],
            "selected": selected,
            "attempts": attempts,
            "accepted_count": sum(
                len(cells)
                for replicates in selected.values()
                for cells in replicates.values()
            ),
        }

    return _parallel_by_node(nodes, max_parallel=max_parallel, fn=run_lane)


def _run_cell_until_accepted(
    *,
    node: str,
    kind: str,
    profile: int,
    role: str,
    role_index: int,
    replicate: int,
    tag: str,
    target_regime: str,
    max_attempts: int,
    seed_base: int,
    init_cache_state: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts = []
    role_seed_base = (
        int(seed_base)
        + int(role_index) * SEED_ROLE_STRIDE
        + int(replicate) * SEED_REPLICATE_STRIDE
    )
    for attempt in range(1, max_attempts + 1):
        run_id = _safe_id(
            f"native_cpu_colocation_targeted_p{profile}_{role}_"
            f"r{replicate:02d}_{node}_{kind}_a{attempt:02d}_{tag}"
        )
        artifact_path = ARTIFACT_ROOT / f"{run_id}.json"
        canonical_cell = colocation_cells(
            run_id=run_id,
            nodes=(node,),
            profiles=(profile,),
            kinds=(kind,),
            tasks_per_node=profile,
            seed_base=role_seed_base,
        )[0]
        cell_id = _calibration_cell_id_for_cell(canonical_cell)
        if artifact_path.exists():
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            action = "REUSED"
        else:
            payload = build_native_cpu_colocation_completion_matrix(
                allow_launch=True,
                run_id=run_id,
                nodes=(node,),
                profiles=(profile,),
                kinds=(kind,),
                max_parallel=1,
                tasks_per_node=profile,
                seed_base=role_seed_base,
                init_cache_state=init_cache_state,
                required_dispatch_regimes={cell_id: target_regime},
                deploy_bundle=False,
            )
            _atomic_write_json(artifact_path, payload)
            action = "LAUNCHED_OR_PREFLIGHTED"
        rows = [
            dict(row)
            for row in payload.get("rows") or []
            if str(row.get("calibration_cell_id") or "") == cell_id
        ]
        row = rows[0] if len(rows) == 1 else None
        accepted = bool(
            row
            and payload.get("measurement_protocol") == MEASUREMENT_PROTOCOL
            and row.get("dispatch_external_cpu_regime") == target_regime
            and row.get("required_dispatch_regime") == target_regime
            and _row_ready(row, min_interval_sample_count=5)
        )
        attempt_row = {
            "node": node,
            "kind": kind,
            "calibration_cell_id": cell_id,
            "role": role,
            "replicate": replicate,
            "attempt": attempt,
            "action": action,
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256_file(artifact_path),
            "status": payload.get("status"),
            "row_status": row.get("status") if row else None,
            "controlled_tasks_launched": bool(
                row and row.get("launched")
            ),
            "observed_dispatch_regime": (
                row.get("dispatch_external_cpu_regime") if row else None
            ),
            "target_dispatch_regime": target_regime,
            "measurement_ready": bool(
                row and _row_ready(row, min_interval_sample_count=5)
            ),
            "accepted": accepted,
            "performance_outcome_fields_used_for_selection": [],
        }
        attempts.append(attempt_row)
        if accepted and row is not None:
            selected = {
                **row,
                "selected_source_artifact": str(artifact_path),
                "selected_source_sha256": attempt_row["artifact_sha256"],
                "selection_protocol": SELECTION_PROTOCOL,
                "selection_role": role,
                "selection_replicate": replicate,
                "selection_attempt": attempt,
                "performance_outcome_fields_used_for_selection": [],
            }
            return selected, attempts
    return None, attempts


def _write_composite_artifacts(
    *,
    tag: str,
    profile: int,
    lane_results: Sequence[Mapping[str, Any]],
    expected_cell_ids: Sequence[str],
    target_regimes: Mapping[str, str],
    role_repeat_counts: Mapping[str, int] | None = None,
) -> dict[str, list[Path]]:
    paths = {"training": [], "calibration": [], "holdout": []}
    role_repeats = dict(
        role_repeat_counts
        or {
            "training": TRAINING_REPEATS,
            "calibration": CALIBRATION_REPEATS,
            "holdout": HOLDOUT_REPEATS,
        }
    )
    expected = set(expected_cell_ids)
    for role, repeat_count in role_repeats.items():
        for replicate in range(1, repeat_count + 1):
            selected_rows = {}
            for lane in lane_results:
                cells = (
                    lane.get("selected", {})
                    .get(role, {})
                    .get(replicate, {})
                )
                for cell_id, row in cells.items():
                    if cell_id in selected_rows:
                        raise ValueError(
                            f"duplicate selected composite cell {cell_id!r}"
                        )
                    selected_rows[cell_id] = dict(row)
            if set(selected_rows) != expected:
                missing = sorted(expected - set(selected_rows))
                extra = sorted(set(selected_rows) - expected)
                raise ValueError(
                    f"composite {role} r{replicate:02d} cell mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            path = ARTIFACT_ROOT / (
                f"native_cpu_colocation_state_targeted_p{profile}_"
                f"{role}_r{replicate:02d}_{tag}.json"
            )
            rows = [selected_rows[cell_id] for cell_id in sorted(expected)]
            payload = {
                "gate": "native_cpu_colocation_completion_matrix",
                "schema_version": 1,
                "measurement_protocol": MEASUREMENT_PROTOCOL,
                "selection_protocol": SELECTION_PROTOCOL,
                "role": role,
                "replicate": replicate,
                "profile": profile,
                "profile_axis": "colocation_count",
                "status": "COMPLETE",
                "all_completion_models_ready": True,
                "selected_count": len(rows),
                "admitted_count": len(rows),
                "target_dispatch_regimes": dict(
                    sorted(target_regimes.items())
                ),
                "expected_calibration_cell_ids": sorted(expected),
                "performance_outcome_fields_used_for_selection": [],
                "rows": rows,
            }
            _atomic_write_json(path, payload)
            paths[role].append(path)
    return paths


def _parallel_by_node(
    nodes: Sequence[str],
    *,
    max_parallel: int,
    fn,
) -> list[dict[str, Any]]:
    rows = []
    workers = max(1, min(int(max_parallel), len(nodes)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, node): node for node in nodes}
        for future in as_completed(futures):
            node = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "node": node,
                        "ready": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return sorted(rows, key=lambda row: str(row.get("node") or ""))


def _ordered_subset(
    values: Iterable[str],
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    requested = {str(value).strip() for value in values if str(value).strip()}
    unknown = requested - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label}s: {sorted(unknown)}")
    selected = tuple(value for value in allowed if value in requested)
    if not selected:
        raise ValueError(f"at least one {label} is required")
    return selected


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    ).strip("_") or "run"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_sha256_map(paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path): _sha256_file(path)
        for path in paths
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in str(value).split(",")
        if part.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", type=int, default=1)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument(
        "--max-attempts-per-cell",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_CELL,
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE + 400_000,
    )
    parser.add_argument(
        "--init-cache-state",
        choices=("cold", "warm", "unspecified"),
        default="warm",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_state_targeted_colocation_campaign(
        tag=args.tag,
        profile=args.profile,
        allow_launch=args.allow_launch,
        nodes=_parse_csv(args.nodes),
        kinds=_parse_csv(args.kinds),
        max_parallel=args.max_parallel,
        max_attempts_per_cell=args.max_attempts_per_cell,
        seed_base=args.seed_base,
        init_cache_state=args.init_cache_state,
        miscoverage_alpha=args.alpha,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not args.allow_launch or result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
