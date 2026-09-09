"""Resumable split-conformal campaign for native CPU co-location profiles."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable, Mapping

from .native_cpu_colocation_completion_matrix import (
    ARTIFACT_ROOT,
    DEFAULT_KINDS,
    DEFAULT_NODES,
    RUN_ROOT,
    WORKLOAD_METADATA,
    build_native_cpu_colocation_completion_matrix,
    build_native_cpu_colocation_measurement_scope,
    colocation_cells,
)
from .phase_aware_colocation_stochastic_lcb_gate import (
    build_colocation_stochastic_lcb_gate,
)
from .file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
    MEASUREMENT_PROTOCOL as P1_MEASUREMENT_PROTOCOL,
)


TRAINING_REPEATS = 3
CALIBRATION_REPEATS = 9
HOLDOUT_REPEATS = 1
SEED_WAVE_STRIDE = 1_000
P1_GATE_NAME = "phase_aware_stochastic_lcb_gate"
P1_GATE_NAMES = {
    P1_GATE_NAME,
    "phase_aware_colocation_stochastic_lcb_gate",
}
LOCK_ROOT = RUN_ROOT / ".native_cpu_campaign_locks"


def build_native_cpu_colocation_stochastic_campaign(
    *,
    allow_launch: bool = False,
    tag: str = "20260727",
    profiles: Iterable[int] = (2,),
    nodes: Iterable[str] = DEFAULT_NODES,
    kinds: Iterable[str] = DEFAULT_KINDS,
    max_parallel: int = 6,
    seed_base: int = 827_000,
    init_cache_state: str = "warm",
    miscoverage_alpha: float = 0.10,
    p1_campaign_path: Path | None = None,
    p1_gate_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the p1 certificate and serialize access to every target node."""

    selected_profiles = tuple(int(value) for value in profiles)
    selected_nodes = tuple(str(value) for value in nodes)
    selected_kinds = tuple(str(value) for value in kinds)
    prerequisite = _validate_p1_prerequisite(
        campaign_path=p1_campaign_path,
        gate_path=p1_gate_path,
        required_nodes=selected_nodes,
        required_kinds=selected_kinds,
        required=bool(
            allow_launch and any(profile > 1 for profile in selected_profiles)
        ),
    )
    if allow_launch and not prerequisite["ready"]:
        return _blocked_campaign_result(
            tag=tag,
            profiles=selected_profiles,
            nodes=selected_nodes,
            kinds=selected_kinds,
            max_parallel=max_parallel,
            prerequisite=prerequisite,
            status="P1_PREREQUISITE_GAP",
        )

    lock_handles: list[Any] = []
    lock_rows: list[dict[str, Any]] = []
    try:
        if allow_launch:
            lock_handles, lock_rows = _acquire_node_locks(
                selected_nodes,
                tag=_safe_id(tag),
            )
        result = _run_native_cpu_colocation_stochastic_campaign_unlocked(
            allow_launch=allow_launch,
            tag=tag,
            profiles=selected_profiles,
            nodes=selected_nodes,
            kinds=selected_kinds,
            max_parallel=max_parallel,
            seed_base=seed_base,
            init_cache_state=init_cache_state,
            miscoverage_alpha=miscoverage_alpha,
        )
    except BlockingIOError as exc:
        return _blocked_campaign_result(
            tag=tag,
            profiles=selected_profiles,
            nodes=selected_nodes,
            kinds=selected_kinds,
            max_parallel=max_parallel,
            prerequisite=prerequisite,
            status="NODE_LOCK_GAP",
            error=str(exc),
        )
    finally:
        _release_node_locks(lock_handles)

    result["p1_prerequisite"] = prerequisite
    result["node_lock_rows"] = lock_rows
    result["node_lock_ready"] = bool(not allow_launch or lock_rows)
    return result


def _run_native_cpu_colocation_stochastic_campaign_unlocked(
    *,
    allow_launch: bool = False,
    tag: str = "20260727",
    profiles: Iterable[int] = (2,),
    nodes: Iterable[str] = DEFAULT_NODES,
    kinds: Iterable[str] = DEFAULT_KINDS,
    max_parallel: int = 6,
    seed_base: int = 827_000,
    init_cache_state: str = "warm",
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    """Run 3 training, 9 calibration, and 1 untouched holdout wave."""

    safe_tag = _safe_id(tag)
    selected_profiles = tuple(sorted({max(1, int(value)) for value in profiles}))
    selected_nodes = tuple(str(value) for value in nodes)
    selected_kinds = tuple(str(value) for value in kinds)
    wave_specs = [
        *[("training", index) for index in range(1, TRAINING_REPEATS + 1)],
        *[
            ("calibration", index)
            for index in range(1, CALIBRATION_REPEATS + 1)
        ],
        ("holdout", 1),
    ]
    wave_rows = []
    paths_by_role: dict[str, list[Path]] = {
        "training": [],
        "calibration": [],
        "holdout": [],
    }
    stopped_on_gap = False
    expected_calibration_cell_ids: tuple[str, ...] = ()

    for wave_number, (role, replicate) in enumerate(wave_specs, start=1):
        wave_run_id = (
            f"native_cpu_colocation_stochastic_{safe_tag}_"
            f"{role}_r{replicate:02d}"
        )
        wave_seed_base = int(seed_base) + wave_number * SEED_WAVE_STRIDE
        expected_cells = colocation_cells(
            run_id=wave_run_id,
            nodes=selected_nodes,
            profiles=selected_profiles,
            kinds=selected_kinds,
            tasks_per_node=max(selected_profiles),
            seed_base=wave_seed_base,
        )
        expected_scope = build_native_cpu_colocation_measurement_scope(
            run_id=wave_run_id,
            cells=expected_cells,
            tasks_per_node=max(selected_profiles),
            seed_base=wave_seed_base,
            init_cache_state=init_cache_state,
        )
        expected_scope_sha256 = _canonical_json_sha256(expected_scope)
        wave_calibration_cell_ids = tuple(
            expected_scope["calibration_cell_ids"]
        )
        if not expected_calibration_cell_ids:
            expected_calibration_cell_ids = wave_calibration_cell_ids
        elif expected_calibration_cell_ids != wave_calibration_cell_ids:
            raise RuntimeError("calibration cell set changed across campaign waves")
        path = ARTIFACT_ROOT / (
            f"native_cpu_colocation_stochastic_{_profile_tag(selected_profiles)}_"
            f"{role}_r{replicate:02d}_{safe_tag}.json"
        )
        paths_by_role[role].append(path)
        if path.exists():
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result = {
                    "status": "INVALID_ARTIFACT",
                    "all_completion_models_ready": False,
                    "artifact_error": f"{type(exc).__name__}: {exc}",
                }
            scope_errors = _artifact_scope_errors(
                result,
                expected_scope=expected_scope,
                expected_scope_sha256=expected_scope_sha256,
            )
            if scope_errors:
                action = "REJECTED_STALE_OR_INCOMPLETE_ARTIFACT"
                result = {
                    **result,
                    "all_completion_models_ready": False,
                    "artifact_scope_errors": scope_errors,
                }
            else:
                action = "REUSED"
        elif allow_launch:
            result = build_native_cpu_colocation_completion_matrix(
                allow_launch=True,
                run_id=wave_run_id,
                nodes=selected_nodes,
                profiles=selected_profiles,
                kinds=selected_kinds,
                max_parallel=max_parallel,
                tasks_per_node=max(selected_profiles),
                seed_base=wave_seed_base,
                init_cache_state=init_cache_state,
            )
            _atomic_write_json(path, result)
            action = "LAUNCHED"
        else:
            result = build_native_cpu_colocation_completion_matrix(
                allow_launch=False,
                run_id=wave_run_id,
                nodes=selected_nodes,
                profiles=selected_profiles,
                kinds=selected_kinds,
                max_parallel=max_parallel,
                tasks_per_node=max(selected_profiles),
                seed_base=wave_seed_base,
                init_cache_state=init_cache_state,
            )
            _atomic_write_json(path, result)
            action = "MANIFEST"

        ready = bool(result.get("all_completion_models_ready"))
        wave_rows.append(
            {
                "wave": wave_number,
                "role": role,
                "replicate": replicate,
                "action": action,
                "output": str(path),
                "ready": ready,
                "status": result.get("status"),
                "admitted_count": result.get("admitted_count"),
                "selected_count": result.get("selected_count"),
                "measurement_scope_sha256": result.get(
                    "measurement_scope_sha256"
                ),
                "artifact_scope_errors": result.get(
                    "artifact_scope_errors", []
                ),
            }
        )
        print(json.dumps(wave_rows[-1], sort_keys=True), flush=True)
        if allow_launch and not ready:
            stopped_on_gap = True
            break

    complete_wave_count = len(wave_rows)
    expected_wave_count = len(wave_specs)
    gate_path = ARTIFACT_ROOT / (
        f"native_cpu_colocation_stochastic_{_profile_tag(selected_profiles)}_"
        f"lcb_gate_{safe_tag}.json"
    )
    gate = None
    if (
        allow_launch
        and not stopped_on_gap
        and complete_wave_count == expected_wave_count
    ):
        gate = build_colocation_stochastic_lcb_gate(
            training_paths=paths_by_role["training"],
            calibration_paths=paths_by_role["calibration"],
            holdout_path=paths_by_role["holdout"][0],
            miscoverage_alpha=miscoverage_alpha,
            expected_cell_ids=expected_calibration_cell_ids,
        )
        _atomic_write_json(gate_path, gate)

    return {
        "gate": "native_cpu_colocation_stochastic_calibration_campaign",
        "schema_version": 1,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "dispatch_cpu_sample_count": DISPATCH_CPU_SAMPLE_COUNT,
        "allow_launch": bool(allow_launch),
        "tag": safe_tag,
        "profiles": list(selected_profiles),
        "profile_axis": "colocation_count",
        "nodes": list(selected_nodes),
        "kinds": list(selected_kinds),
        "training_repeat_count": TRAINING_REPEATS,
        "calibration_repeat_count": CALIBRATION_REPEATS,
        "holdout_repeat_count": HOLDOUT_REPEATS,
        "expected_wave_count": expected_wave_count,
        "completed_wave_count": complete_wave_count,
        "max_parallel": int(max_parallel),
        "parallelism_scope": "six homogeneous nodes per wave",
        "paired_seed_across_nodes": True,
        "distinct_seed_across_waves": True,
        "init_cache_state": init_cache_state,
        "expected_calibration_cell_ids": list(
            expected_calibration_cell_ids
        ),
        "expected_calibration_cell_count": len(
            expected_calibration_cell_ids
        ),
        "stopped_on_measurement_gap": stopped_on_gap,
        "wave_rows": wave_rows,
        "lcb_gate_path": str(gate_path),
        "gate_path": str(gate_path),
        "lcb_gate": gate,
        "gate_pass": bool(gate and gate.get("pass")),
        "pass": bool(gate and gate.get("pass")),
        "status": (
            "PASS"
            if gate and gate.get("pass")
            else (
                "FAIL"
                if gate is not None
                else ("STOPPED_ON_GAP" if stopped_on_gap else "MANIFEST")
            )
        ),
        "claim_boundary": (
            "The campaign validates only exact native CPU co-location cells. "
            "Profiles are advanced independently; a p2 certificate does not "
            "automatically admit p4 or a different external-load bucket."
        ),
    }


def _artifact_scope_errors(
    payload: Mapping[str, Any],
    *,
    expected_scope: Mapping[str, Any],
    expected_scope_sha256: str,
) -> list[str]:
    errors = []
    if payload.get("gate") != "native_cpu_colocation_completion_matrix":
        errors.append("gate_identity_mismatch")
    if payload.get("measurement_scope_sha256") != expected_scope_sha256:
        errors.append("measurement_scope_sha256_mismatch")
    if payload.get("measurement_scope") != expected_scope:
        errors.append("measurement_scope_payload_mismatch")
    if not payload.get("all_completion_models_ready"):
        errors.append("completion_models_not_all_ready")
    expected_cells = set(expected_scope.get("calibration_cell_ids") or [])
    observed_cells = {
        str(row.get("calibration_cell_id") or "")
        for row in (payload.get("rows") or [])
        if str(row.get("calibration_cell_id") or "")
    }
    if observed_cells != expected_cells:
        errors.append("calibration_cell_set_mismatch")
    if int(payload.get("selected_count") or 0) != int(
        expected_scope.get("expected_cell_count") or 0
    ):
        errors.append("selected_count_mismatch")
    return errors


def _validate_p1_prerequisite(
    *,
    campaign_path: Path | None,
    gate_path: Path | None,
    required_nodes: Iterable[str],
    required: bool,
    required_kinds: Iterable[str] = DEFAULT_KINDS,
) -> dict[str, Any]:
    if not required:
        return {
            "required": False,
            "ready": True,
            "status": "NOT_REQUIRED_FOR_MANIFEST",
        }
    errors: list[str] = []
    campaign_identity = _json_file_identity(campaign_path, "campaign", errors)
    gate_identity = _json_file_identity(gate_path, "gate", errors)
    campaign = campaign_identity.get("payload") or {}
    gate = gate_identity.get("payload") or {}

    if campaign:
        if campaign.get("status") != "PASS" or not bool(
            campaign.get("gate_pass", campaign.get("pass"))
        ):
            errors.append("p1_campaign_not_passed")
        if campaign.get("measurement_protocol") != P1_MEASUREMENT_PROTOCOL:
            errors.append("p1_campaign_protocol_mismatch")
        declared_gate = str(
            campaign.get("gate_path")
            or campaign.get("lcb_gate_path")
            or ""
        ).strip()
        if not declared_gate:
            errors.append("p1_campaign_gate_path_missing")
        elif gate_path is not None and Path(declared_gate).resolve() != Path(
            gate_path
        ).resolve():
            errors.append("p1_campaign_gate_path_mismatch")

    if gate:
        if gate.get("gate") not in P1_GATE_NAMES:
            errors.append("p1_gate_identity_mismatch")
        if not gate.get("pass") or gate.get("status") != "PASS":
            errors.append("p1_gate_not_passed")
        if not gate.get("all_rows_ready"):
            errors.append("p1_gate_rows_not_all_ready")
        if not gate.get("all_lower_service_valid_on_holdout"):
            errors.append("p1_lower_service_not_all_valid")
        rows = list(gate.get("rows") or [])
        selected_nodes = set(str(value) for value in required_nodes)
        selected_kinds = set(str(value) for value in required_kinds)
        unknown_kinds = sorted(selected_kinds - set(WORKLOAD_METADATA))
        if unknown_kinds:
            errors.append(
                "p1_unknown_required_kinds:" + ",".join(unknown_kinds)
            )
        required_cells = {
            (
                node,
                str(WORKLOAD_METADATA[kind]["workload_key"]),
                1,
            )
            for node in selected_nodes
            for kind in selected_kinds
            if kind in WORKLOAD_METADATA
        }
        observed_cells = {
            (
                str(row.get("node") or ""),
                str(row.get("workload_key") or ""),
                int(row.get("colocation_count") or 1),
            )
            for row in rows
        }
        missing_cells = sorted(required_cells - observed_cells)
        if missing_cells:
            errors.append(
                "p1_gate_missing_exact_cells:"
                + ",".join(
                    f"{node}|{workload}|p{profile}"
                    for node, workload, profile in missing_cells
                )
            )
        for row in rows:
            row_cell = (
                str(row.get("node") or ""),
                str(row.get("workload_key") or ""),
                int(row.get("colocation_count") or 1),
            )
            if row_cell not in required_cells:
                continue
            if (
                not row.get("ready")
                or not row.get("dispatch_state_ready")
                or row.get("measurement_protocol") != MEASUREMENT_PROTOCOL
                or int(row.get("dispatch_sample_count") or 0)
                < DISPATCH_CPU_SAMPLE_COUNT
                or not row.get("lower_service_valid_on_holdout")
                or float(row.get("lower_service_units_per_s") or 0.0) <= 0.0
            ):
                errors.append(
                    "p1_exact_cell_certificate_invalid:"
                    + "|".join(
                        (
                            row_cell[0] or "unknown-node",
                            row_cell[1] or "unknown-workload",
                            f"p{row_cell[2]}",
                        )
                    )
                )
        evidence_paths = [
            *[Path(value) for value in (gate.get("training_paths") or [])],
            *[Path(value) for value in (gate.get("calibration_paths") or [])],
        ]
        holdout = str(gate.get("holdout_path") or "").strip()
        if holdout:
            evidence_paths.append(Path(holdout))
        else:
            errors.append("p1_holdout_path_missing")
        evidence_sha256 = {}
        for path in evidence_paths:
            if not path.is_file():
                errors.append(f"p1_evidence_missing:{path}")
                continue
            evidence_sha256[str(path.resolve())] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    else:
        evidence_sha256 = {}

    return {
        "required": True,
        "ready": not errors,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "campaign_path": campaign_identity.get("path", ""),
        "campaign_sha256": campaign_identity.get("sha256", ""),
        "gate_path": gate_identity.get("path", ""),
        "gate_sha256": gate_identity.get("sha256", ""),
        "evidence_sha256": evidence_sha256,
    }


def _json_file_identity(
    path: Path | None,
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    if path is None:
        errors.append(f"p1_{label}_path_missing")
        return {}
    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        errors.append(f"p1_{label}_file_missing:{selected}")
        return {"path": str(selected)}
    try:
        raw = selected.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"p1_{label}_invalid:{type(exc).__name__}")
        return {"path": str(selected)}
    return {
        "path": str(selected),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": payload,
    }


def _acquire_node_locks(
    nodes: Iterable[str],
    *,
    tag: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    try:
        for node in sorted(set(str(value) for value in nodes)):
            path = LOCK_ROOT / f"{_safe_id(node)}.lock"
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                owner = handle.read().strip()
                handle.close()
                raise BlockingIOError(
                    f"CPU campaign node lock is held for {node}: {owner}"
                )
            owner = {
                "pid": os.getpid(),
                "tag": tag,
                "node": node,
                "acquired_unix_s": time.time(),
            }
            handle.seek(0)
            handle.truncate()
            json.dump(owner, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            handles.append(handle)
            rows.append({"node": node, "path": str(path), **owner})
    except BaseException:
        _release_node_locks(handles)
        raise
    return handles, rows


def _release_node_locks(handles: Iterable[Any]) -> None:
    for handle in reversed(list(handles)):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _blocked_campaign_result(
    *,
    tag: str,
    profiles: Iterable[int],
    nodes: Iterable[str],
    kinds: Iterable[str],
    max_parallel: int,
    prerequisite: Mapping[str, Any],
    status: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "gate": "native_cpu_colocation_stochastic_calibration_campaign",
        "schema_version": 1,
        "allow_launch": True,
        "tag": _safe_id(tag),
        "profiles": sorted({int(value) for value in profiles}),
        "nodes": list(nodes),
        "kinds": list(kinds),
        "max_parallel": int(max_parallel),
        "completed_wave_count": 0,
        "expected_wave_count": (
            TRAINING_REPEATS + CALIBRATION_REPEATS + HOLDOUT_REPEATS
        ),
        "p1_prerequisite": dict(prerequisite),
        "pass": False,
        "status": status,
        "error": error,
    }


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _profile_tag(profiles: Iterable[int]) -> str:
    return "_".join(f"p{int(value)}" for value in profiles)


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "run"


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(
        int(part.strip())
        for part in str(value).split(",")
        if part.strip()
    )


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in str(value).split(",")
        if part.strip()
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--tag", default="20260727")
    parser.add_argument("--profiles", default="2")
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--seed-base", type=int, default=827_000)
    parser.add_argument(
        "--init-cache-state",
        choices=("cold", "warm", "unspecified"),
        default="warm",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--p1-campaign", type=Path)
    parser.add_argument("--p1-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_native_cpu_colocation_stochastic_campaign(
        allow_launch=args.allow_launch,
        tag=args.tag,
        profiles=_parse_ints(args.profiles),
        nodes=_parse_strings(args.nodes),
        kinds=_parse_strings(args.kinds),
        max_parallel=args.max_parallel,
        seed_base=args.seed_base,
        init_cache_state=args.init_cache_state,
        miscoverage_alpha=args.alpha,
        p1_campaign_path=args.p1_campaign,
        p1_gate_path=args.p1_gate,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] or not args.allow_launch else 2


if __name__ == "__main__":
    raise SystemExit(main())
