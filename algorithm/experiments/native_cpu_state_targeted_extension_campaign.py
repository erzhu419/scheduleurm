"""State-targeted extension for native CPU completion and service LCBs.

Target strata are frozen from training data. Calibration and holdout attempts
are admitted using only the dispatch-time state signature, never an ETA error
or realized completion outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .native_cpu_colocation_stochastic_calibration_campaign import (
    _acquire_node_locks,
    _release_node_locks,
)
from .native_cpu_workload_completion_matrix import (
    ARTIFACT_ROOT,
    RUN_ROOT,
    _deploy_wrapper,
    build_native_cpu_workload_completion_matrix,
    smoke_cells,
)
from .phase_aware_stochastic_lcb_gate import (
    _state_signature,
    build_stochastic_completion_lcb_gate,
)
from .file_progress_completion_wrapper import MEASUREMENT_PROTOCOL


SELECTION_PROTOCOL = "dispatch_state_only_acceptance_sampling_v1"
DEFAULT_TARGET_CALIBRATION_COUNT = 12
DEFAULT_MAX_CALIBRATION_ATTEMPTS = 30
DEFAULT_MAX_HOLDOUT_ATTEMPTS = 20
SOURCE_FILES = (
    "progress_units.py",
    "file_progress_completion_wrapper.py",
    "native_cpu_workload_completion_matrix.py",
    "phase_aware_stochastic_lcb_gate.py",
    "native_cpu_state_targeted_extension_campaign.py",
)


def run_state_targeted_extension(
    *,
    base_tag: str,
    extension_tag: str,
    allow_launch: bool = False,
    target_calibration_count: int = DEFAULT_TARGET_CALIBRATION_COUNT,
    max_calibration_attempts: int = DEFAULT_MAX_CALIBRATION_ATTEMPTS,
    max_holdout_attempts: int = DEFAULT_MAX_HOLDOUT_ATTEMPTS,
    max_parallel: int = 6,
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    started = time.time()
    safe_tag = _safe_id(extension_tag)
    training_paths = sorted(
        ARTIFACT_ROOT.glob(
            f"native_cpu_stochastic_training_r*_{base_tag}.json"
        )
    )
    base_calibration_paths = sorted(
        ARTIFACT_ROOT.glob(
            f"native_cpu_stochastic_calibration_r*_{base_tag}.json"
        )
    )
    if len(training_paths) != 3 or len(base_calibration_paths) != 9:
        raise ValueError(
            "state-targeted extension requires exactly three base training "
            "and nine base calibration artifacts"
        )
    training = [_load_complete(path) for path in training_paths]
    base_calibration = [
        _load_complete(path) for path in base_calibration_paths
    ]
    target_rows = _freeze_training_targets(training)
    target_signatures = {
        node: _state_signature(row) for node, row in target_rows.items()
    }
    minimum_calibration_count = _minimum_conformal_calibration_count(
        miscoverage_alpha
    )
    requested_target_count = max(
        int(target_calibration_count),
        minimum_calibration_count,
    )
    base_matching_counts = {
        node: _matching_row_count(
            base_calibration,
            node=node,
            signature=target_signatures[node],
        )
        for node in sorted(target_signatures)
    }

    if not allow_launch:
        return {
            "campaign": "native_cpu_state_targeted_extension",
            "schema_version": 1,
            "status": "MANIFEST",
            "pass": False,
            "allow_launch": False,
            "base_tag": base_tag,
            "extension_tag": safe_tag,
            "measurement_protocol": MEASUREMENT_PROTOCOL,
            "selection_protocol": SELECTION_PROTOCOL,
            "minimum_calibration_count": minimum_calibration_count,
            "target_calibration_count": requested_target_count,
            "base_matching_calibration_counts": base_matching_counts,
            "target_signatures": {
                node: list(signature)
                for node, signature in target_signatures.items()
            },
        }

    nodes = tuple(sorted(target_signatures))
    lock_handles, lock_rows = _acquire_node_locks(nodes, tag=safe_tag)
    try:
        deployment_rows = _deploy_wrappers(
            nodes,
            tag=safe_tag,
            max_parallel=max_parallel,
        )
        if not all(row["ready"] for row in deployment_rows):
            return _failure_report(
                base_tag=base_tag,
                extension_tag=safe_tag,
                status="DEPLOYMENT_GAP",
                started=started,
                target_signatures=target_signatures,
                base_matching_counts=base_matching_counts,
                target_calibration_count=requested_target_count,
                minimum_calibration_count=minimum_calibration_count,
                deployment_rows=deployment_rows,
                lock_rows=lock_rows,
            )

        calibration_results = _run_parallel_calibration_extensions(
            nodes=nodes,
            extension_tag=safe_tag,
            target_signatures=target_signatures,
            base_matching_counts=base_matching_counts,
            target_calibration_count=requested_target_count,
            max_attempts=max_calibration_attempts,
            max_parallel=max_parallel,
        )
        calibration_ready = all(
            result["target_count_after_extension"]
            >= requested_target_count
            for result in calibration_results.values()
        )
        if not calibration_ready:
            return _failure_report(
                base_tag=base_tag,
                extension_tag=safe_tag,
                status="CALIBRATION_STATE_GAP",
                started=started,
                target_signatures=target_signatures,
                base_matching_counts=base_matching_counts,
                target_calibration_count=requested_target_count,
                minimum_calibration_count=minimum_calibration_count,
                deployment_rows=deployment_rows,
                lock_rows=lock_rows,
                calibration_results=calibration_results,
            )

        holdout_results = _run_parallel_holdout_extensions(
            nodes=nodes,
            extension_tag=safe_tag,
            target_signatures=target_signatures,
            max_attempts=max_holdout_attempts,
            max_parallel=max_parallel,
        )
        holdout_ready = all(
            result.get("selected_row") is not None
            for result in holdout_results.values()
        )
        if not holdout_ready:
            return _failure_report(
                base_tag=base_tag,
                extension_tag=safe_tag,
                status="HOLDOUT_STATE_GAP",
                started=started,
                target_signatures=target_signatures,
                base_matching_counts=base_matching_counts,
                target_calibration_count=requested_target_count,
                minimum_calibration_count=minimum_calibration_count,
                deployment_rows=deployment_rows,
                lock_rows=lock_rows,
                calibration_results=calibration_results,
                holdout_results=holdout_results,
            )

        composite_holdout_path = (
            ARTIFACT_ROOT
            / f"native_cpu_state_targeted_holdout_{safe_tag}.json"
        )
        composite_holdout = _build_composite_holdout(
            holdout_results=holdout_results,
            target_signatures=target_signatures,
            extension_tag=safe_tag,
        )
        _atomic_write_json(composite_holdout_path, composite_holdout)
        extension_calibration_paths = [
            Path(attempt["artifact_path"])
            for result in calibration_results.values()
            for attempt in result["attempts"]
            if attempt["measurement_ready"]
        ]
        gate = build_stochastic_completion_lcb_gate(
            training_paths=training_paths,
            calibration_paths=[
                *base_calibration_paths,
                *extension_calibration_paths,
            ],
            holdout_path=composite_holdout_path,
            miscoverage_alpha=miscoverage_alpha,
        )
        gate_path = (
            ARTIFACT_ROOT
            / f"native_cpu_stochastic_lcb_gate_{safe_tag}.json"
        )
        _atomic_write_json(gate_path, gate)
        evidence_paths = [
            *training_paths,
            *base_calibration_paths,
            *extension_calibration_paths,
            composite_holdout_path,
        ]
        gate_pass = bool(gate.get("pass"))
        return {
            "campaign": "native_cpu_state_targeted_extension",
            "schema_version": 1,
            "base_tag": base_tag,
            "extension_tag": safe_tag,
            "allow_launch": True,
            "measurement_protocol": MEASUREMENT_PROTOCOL,
            "selection_protocol": SELECTION_PROTOCOL,
            "miscoverage_alpha": float(miscoverage_alpha),
            "minimum_calibration_count": minimum_calibration_count,
            "target_calibration_count": requested_target_count,
            "base_matching_calibration_counts": base_matching_counts,
            "target_signatures": {
                node: list(signature)
                for node, signature in target_signatures.items()
            },
            "deployment_rows": deployment_rows,
            "node_lock_rows": lock_rows,
            "calibration_results": calibration_results,
            "holdout_results": _public_holdout_results(holdout_results),
            "composite_holdout_path": str(composite_holdout_path),
            "gate_path": str(gate_path),
            "gate_pass": gate_pass,
            "pass": gate_pass,
            "status": "PASS" if gate_pass else "GATE_FAIL",
            "source_sha256": _source_sha256(),
            "evidence_sha256": _file_sha256_map(evidence_paths),
            "elapsed_wall_s": time.time() - started,
            "claim_boundary": (
                "Targets are frozen from the three training waves. Additional "
                "calibration and holdout attempts are accepted using only the "
                "pre-child dispatch-state signature and measurement readiness. "
                "No realized completion time, ETA error, or lower-service result "
                "is read by the selection rule. The composite holdout supports "
                "cellwise, not simultaneous cross-node, coverage."
            ),
        }
    finally:
        _release_node_locks(lock_handles)


def _freeze_training_targets(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_node: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        for raw in payload.get("rows") or []:
            row = dict(raw)
            if not _measurement_ready(row):
                raise ValueError(
                    f"training row is not measurement-ready: {row.get('node')}"
                )
            by_node.setdefault(str(row.get("node") or ""), []).append(row)
    expected_nodes = {cell.node for cell in smoke_cells()}
    if set(by_node) != expected_nodes:
        raise ValueError("base training artifacts do not cover all six nodes")
    targets = {}
    for node, rows in sorted(by_node.items()):
        if len(rows) != len(payloads):
            raise ValueError(f"training count mismatch for {node}")
        counts = Counter(_state_signature(row) for row in rows)
        signature, count = counts.most_common(1)[0]
        if count < len(payloads):
            raise ValueError(
                f"training target is not pre-stable for {node}: {counts}"
            )
        targets[node] = next(
            row for row in rows if _state_signature(row) == signature
        )
    return targets


def _run_parallel_calibration_extensions(
    *,
    nodes: Sequence[str],
    extension_tag: str,
    target_signatures: Mapping[str, tuple[Any, ...]],
    base_matching_counts: Mapping[str, int],
    target_calibration_count: int,
    max_attempts: int,
    max_parallel: int,
) -> dict[str, dict[str, Any]]:
    results = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_parallel), len(nodes)))
    ) as pool:
        futures = {
            pool.submit(
                _fill_calibration_node,
                node=node,
                extension_tag=extension_tag,
                target_signature=target_signatures[node],
                starting_count=int(base_matching_counts[node]),
                target_count=int(target_calibration_count),
                max_attempts=max(1, int(max_attempts)),
            ): node
            for node in nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            results[node] = future.result()
            print(
                json.dumps(
                    {
                        "phase": "calibration",
                        "node": node,
                        "target_count": results[node][
                            "target_count_after_extension"
                        ],
                        "attempt_count": len(results[node]["attempts"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {node: results[node] for node in sorted(results)}


def _fill_calibration_node(
    *,
    node: str,
    extension_tag: str,
    target_signature: tuple[Any, ...],
    starting_count: int,
    target_count: int,
    max_attempts: int,
) -> dict[str, Any]:
    count = int(starting_count)
    attempts = []
    for attempt in range(1, max_attempts + 1):
        if count >= target_count:
            break
        artifact_path = ARTIFACT_ROOT / (
            f"native_cpu_state_targeted_calibration_{node}_"
            f"a{attempt:02d}_{extension_tag}.json"
        )
        run_id = (
            f"native_cpu_state_targeted_calibration_{node}_"
            f"a{attempt:02d}_{extension_tag}"
        )
        payload, action = _read_or_launch_one_node(
            artifact_path=artifact_path,
            run_id=run_id,
            node=node,
            seed_offset=50_000 + attempt * 1_000,
        )
        row = _only_row(payload, node=node)
        ready = _measurement_ready(row)
        matches = bool(ready and _state_signature(row) == target_signature)
        if matches:
            count += 1
        attempts.append(
            {
                "attempt": attempt,
                "action": action,
                "artifact_path": str(artifact_path),
                "artifact_sha256": _sha256_file(artifact_path),
                "measurement_ready": ready,
                "dispatch_state_match": matches,
                "observed_signature": (
                    list(_state_signature(row)) if ready else None
                ),
                "target_count_after_attempt": count,
            }
        )
    return {
        "node": node,
        "starting_target_count": int(starting_count),
        "required_target_count": int(target_count),
        "target_count_after_extension": count,
        "attempts": attempts,
    }


def _run_parallel_holdout_extensions(
    *,
    nodes: Sequence[str],
    extension_tag: str,
    target_signatures: Mapping[str, tuple[Any, ...]],
    max_attempts: int,
    max_parallel: int,
) -> dict[str, dict[str, Any]]:
    results = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_parallel), len(nodes)))
    ) as pool:
        futures = {
            pool.submit(
                _select_holdout_node,
                node=node,
                extension_tag=extension_tag,
                target_signature=target_signatures[node],
                max_attempts=max(1, int(max_attempts)),
            ): node
            for node in nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            results[node] = future.result()
            print(
                json.dumps(
                    {
                        "phase": "holdout",
                        "node": node,
                        "selected": (
                            results[node].get("selected_row") is not None
                        ),
                        "attempt_count": len(results[node]["attempts"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {node: results[node] for node in sorted(results)}


def _select_holdout_node(
    *,
    node: str,
    extension_tag: str,
    target_signature: tuple[Any, ...],
    max_attempts: int,
) -> dict[str, Any]:
    attempts = []
    selected_row = None
    selected_path = ""
    for attempt in range(1, max_attempts + 1):
        artifact_path = ARTIFACT_ROOT / (
            f"native_cpu_state_targeted_holdout_{node}_"
            f"a{attempt:02d}_{extension_tag}.json"
        )
        run_id = (
            f"native_cpu_state_targeted_holdout_{node}_"
            f"a{attempt:02d}_{extension_tag}"
        )
        payload, action = _read_or_launch_one_node(
            artifact_path=artifact_path,
            run_id=run_id,
            node=node,
            seed_offset=90_000 + attempt * 1_000,
        )
        row = _only_row(payload, node=node)
        ready = _measurement_ready(row)
        matches = bool(ready and _state_signature(row) == target_signature)
        attempts.append(
            {
                "attempt": attempt,
                "action": action,
                "artifact_path": str(artifact_path),
                "artifact_sha256": _sha256_file(artifact_path),
                "measurement_ready": ready,
                "dispatch_state_match": matches,
                "observed_signature": (
                    list(_state_signature(row)) if ready else None
                ),
            }
        )
        if matches:
            selected_row = row
            selected_path = str(artifact_path)
            break
    return {
        "node": node,
        "selected_row": selected_row,
        "selected_artifact_path": selected_path,
        "selection_basis": (
            "measurement_ready_and_exact_dispatch_state_signature"
        ),
        "performance_outcome_fields_used_for_selection": [],
        "attempts": attempts,
    }


def _read_or_launch_one_node(
    *,
    artifact_path: Path,
    run_id: str,
    node: str,
    seed_offset: int,
) -> tuple[dict[str, Any], str]:
    if artifact_path.is_file():
        payload = _load_complete(artifact_path)
        if payload.get("run_id") != run_id:
            raise ValueError(f"stale run id in {artifact_path}")
        _only_row(payload, node=node)
        return payload, "REUSED"
    payload = build_native_cpu_workload_completion_matrix(
        allow_launch=True,
        run_id=run_id,
        nodes={node},
        max_parallel=1,
        seed_offset=int(seed_offset),
        init_cache_state="warm",
        deploy_wrapper=False,
    )
    _atomic_write_json(artifact_path, payload)
    return payload, "LAUNCHED"


def _build_composite_holdout(
    *,
    holdout_results: Mapping[str, Mapping[str, Any]],
    target_signatures: Mapping[str, tuple[Any, ...]],
    extension_tag: str,
) -> dict[str, Any]:
    rows = []
    selection = []
    for node in sorted(holdout_results):
        result = holdout_results[node]
        row = result.get("selected_row")
        if not isinstance(row, Mapping):
            raise ValueError(f"no selected holdout row for {node}")
        rows.append(dict(row))
        selection.append(
            {
                "node": node,
                "artifact_path": result["selected_artifact_path"],
                "artifact_sha256": _sha256_file(
                    Path(result["selected_artifact_path"])
                ),
                "target_signature": list(target_signatures[node]),
                "selection_basis": result["selection_basis"],
                "performance_outcome_fields_used_for_selection": [],
            }
        )
    return {
        "gate": "native_cpu_state_targeted_composite_holdout",
        "schema_version": 1,
        "extension_tag": extension_tag,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "selection_basis_fields": [
            "measurement_valid",
            "returncode",
            "dispatch_state_ready",
            "node",
            "workload_key",
            "workload_env",
            "observed_effective_resource_state",
            "measurement_protocol",
            "dispatch_sample_count",
            "dispatch_external_cpu_regime",
            "allocation_workers",
            "colocation_count",
            "total_units",
            "init_cache_state",
        ],
        "performance_outcome_fields_used_for_selection": [],
        "selected_count": len(rows),
        "all_completion_models_ready": all(
            _measurement_ready(row) for row in rows
        ),
        "selection": selection,
        "rows": rows,
        "claim_boundary": (
            "Each selected row is the first measurement-ready attempt in the "
            "pre-registered dispatch-state stratum for that node. Completion "
            "time and ETA error are not selection inputs."
        ),
    }


def _minimum_conformal_calibration_count(alpha: float) -> int:
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError("miscoverage alpha must lie strictly between zero and one")
    count = 1
    while math.ceil((count + 1) * (1.0 - value)) > count:
        count += 1
    return count


def _matching_row_count(
    payloads: Sequence[Mapping[str, Any]],
    *,
    node: str,
    signature: tuple[Any, ...],
) -> int:
    return sum(
        1
        for payload in payloads
        for row in (payload.get("rows") or [])
        if str(row.get("node") or "") == node
        and _measurement_ready(row)
        and _state_signature(row) == signature
    )


def _measurement_ready(row: Mapping[str, Any]) -> bool:
    model = row.get("completion_model") or {}
    try:
        returncode_ready = int(row.get("returncode")) == 0
        child_ready = int(model.get("child_returncode")) == 0
    except (TypeError, ValueError):
        return False
    return bool(
        row.get("measurement_valid")
        and returncode_ready
        and child_ready
        and model.get("natural_exit")
        and model.get("completion_model_ready")
        and int(model.get("interval_sample_count") or 0) >= 5
        and row.get("dispatch_state_ready")
        and row.get("measurement_protocol") == MEASUREMENT_PROTOCOL
        and int(row.get("dispatch_sample_count") or 0) >= 5
        and str(row.get("dispatch_external_cpu_regime") or "")
        and str(row.get("eta_source") or "") != "history"
    )


def _only_row(payload: Mapping[str, Any], *, node: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in (payload.get("rows") or [])
        if str(row.get("node") or "") == node
    ]
    if len(rows) != 1 or int(payload.get("selected_count") or 0) != 1:
        raise ValueError(f"one-node artifact has the wrong scope for {node}")
    return rows[0]


def _load_complete(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("all_completion_models_ready"):
        raise ValueError(f"measurement artifact is incomplete: {path}")
    return payload


def _deploy_wrappers(
    nodes: Sequence[str],
    *,
    tag: str,
    max_parallel: int,
) -> list[dict[str, Any]]:
    root = RUN_ROOT / f"native_cpu_state_targeted_{tag}" / "deploy"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_parallel), len(nodes)))
    ) as pool:
        futures = {
            pool.submit(_deploy_wrapper, node, root / node): node
            for node in nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            try:
                future.result()
                rows.append({"node": node, "ready": True})
            except Exception as exc:
                rows.append(
                    {
                        "node": node,
                        "ready": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return sorted(rows, key=lambda row: str(row["node"]))


def _failure_report(
    *,
    base_tag: str,
    extension_tag: str,
    status: str,
    started: float,
    target_signatures: Mapping[str, tuple[Any, ...]],
    base_matching_counts: Mapping[str, int],
    target_calibration_count: int,
    minimum_calibration_count: int,
    deployment_rows: Sequence[Mapping[str, Any]],
    lock_rows: Sequence[Mapping[str, Any]],
    calibration_results: Mapping[str, Any] | None = None,
    holdout_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "campaign": "native_cpu_state_targeted_extension",
        "schema_version": 1,
        "base_tag": base_tag,
        "extension_tag": extension_tag,
        "allow_launch": True,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "minimum_calibration_count": minimum_calibration_count,
        "target_calibration_count": target_calibration_count,
        "base_matching_calibration_counts": dict(base_matching_counts),
        "target_signatures": {
            node: list(signature)
            for node, signature in target_signatures.items()
        },
        "deployment_rows": list(deployment_rows),
        "node_lock_rows": list(lock_rows),
        "calibration_results": dict(calibration_results or {}),
        "holdout_results": _public_holdout_results(holdout_results or {}),
        "gate_path": "",
        "gate_pass": False,
        "pass": False,
        "status": status,
        "source_sha256": _source_sha256(),
        "elapsed_wall_s": time.time() - started,
    }


def _public_holdout_results(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        node: {
            key: value
            for key, value in result.items()
            if key != "selected_row"
        }
        for node, result in results.items()
    }


def _source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def _file_sha256_map(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(Path(path).resolve()): _sha256_file(Path(path))
        for path in paths
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_id(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "run"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
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
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--extension-tag", required=True)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument(
        "--target-calibration-count",
        type=int,
        default=DEFAULT_TARGET_CALIBRATION_COUNT,
    )
    parser.add_argument(
        "--max-calibration-attempts",
        type=int,
        default=DEFAULT_MAX_CALIBRATION_ATTEMPTS,
    )
    parser.add_argument(
        "--max-holdout-attempts",
        type=int,
        default=DEFAULT_MAX_HOLDOUT_ATTEMPTS,
    )
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_state_targeted_extension(
        base_tag=args.base_tag,
        extension_tag=args.extension_tag,
        allow_launch=args.allow_launch,
        target_calibration_count=args.target_calibration_count,
        max_calibration_attempts=args.max_calibration_attempts,
        max_holdout_attempts=args.max_holdout_attempts,
        max_parallel=args.max_parallel,
        miscoverage_alpha=args.alpha,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not args.allow_launch:
        return 0
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
