"""Merge the currently available, separately calibrated GPU execution classes.

This gate does not weaken the all-hardware merge that still waits for
``jtl311linux``.  It creates a scoped cache for the three GPU nodes whose
natural-completion certificates are available, and it refuses to pool the two
nominally identical RTX 3080 Ti nodes after their equivalence gate failed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ServiceRateCache

from .critical_gpu_completion_campaign import (
    ARTIFACT_ROOT,
    NODE_SPECS,
    PROTOCOL,
    TRAINING_WAVES,
    campaign_cells,
)
from .critical_gpu_lcb_cache_merge import (
    DEFAULT_BASE,
    _legacy_index_snapshot,
    _profile_record,
    _row_key,
    _validate_certificate_row,
    _validated_gate_rows,
    write_merge_outputs,
)
from .critical_gpu_p10_transport_completion_campaign import (
    CAMPAIGN_DATE as CORRECTION_DATE,
    PROTOCOL as CORRECTION_PROTOCOL,
)


AVAILABLE_NODES = ("jtl110gpu", "jtl110gpu2", "node007")
PENDING_NODES = ("jtl311linux",)
EFFECTIVE_NODE_BUCKETS = {
    "jtl110gpu": "gpu_3080ti_12gb_dual:jtl110gpu",
    "jtl110gpu2": "gpu_3080ti_12gb_dual:jtl110gpu2",
    "node007": "gpu_2080ti_11gb_quad:node007",
}
DEFAULT_GATE_PATHS = {
    "jtl110gpu": ARTIFACT_ROOT
    / "critical_gpu_stochastic_lcb_gate_v8_jtl110gpu_20260803.json",
    "jtl110gpu2": ARTIFACT_ROOT
    / "critical_gpu_stochastic_lcb_gate_v8_jtl110gpu2_20260803.json",
    "node007": ARTIFACT_ROOT
    / f"critical_gpu_stochastic_lcb_gate_v9_node007_{CORRECTION_DATE}.json",
}
DEFAULT_EQUIVALENCE_GATE = (
    ARTIFACT_ROOT / "critical_gpu_homogeneous_equivalence_gate_v8_20260803.json"
)
DEFAULT_REPORT = (
    ARTIFACT_ROOT / "critical_gpu_available_lcb_cache_merge_gate_20260808.json"
)
DEFAULT_MARKDOWN = DEFAULT_REPORT.with_suffix(".md")
DEFAULT_CACHE_OUTPUT = (
    ARTIFACT_ROOT / "service_cache_v2_available_gpu_phase_20260808.json"
)


def build_critical_gpu_available_lcb_cache_merge(
    *,
    base_cache_path: Path = DEFAULT_BASE,
    gate_paths: Mapping[str, Path] = DEFAULT_GATE_PATHS,
    equivalence_gate_path: Path = DEFAULT_EQUIVALENCE_GATE,
) -> dict[str, Any]:
    validation_errors: list[dict[str, Any]] = []
    wait_reasons: list[dict[str, Any]] = []
    supplied = {str(node): Path(path) for node, path in gate_paths.items()}
    if set(supplied) != set(AVAILABLE_NODES):
        validation_errors.append(
            {
                "code": "AVAILABLE_GATE_SET_MISMATCH",
                "detail": (
                    f"expected {AVAILABLE_NODES!r}, observed {tuple(sorted(supplied))!r}"
                ),
            }
        )

    equivalence = _load_json(Path(equivalence_gate_path), wait_reasons, validation_errors)
    if equivalence is not None:
        _validate_nonpooling_equivalence(
            equivalence,
            path=Path(equivalence_gate_path),
            errors=validation_errors,
        )

    gate_rows: dict[str, tuple[list[Mapping[str, Any]], Path]] = {}
    source_audits: list[dict[str, Any]] = []
    for node in AVAILABLE_NODES:
        path = supplied.get(node)
        payload = _load_json(path, wait_reasons, validation_errors)
        if payload is None or path is None:
            continue
        source_status = str(payload.get("status") or "")
        if source_status.startswith("WAIT"):
            wait_reasons.append(
                {
                    "code": "GPU_GATE_NOT_READY",
                    "node": node,
                    "detail": f"{path}: status={source_status}",
                }
            )
            source_audits.append(
                {
                    "node": node,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": source_status,
                    "effective_node_bucket": EFFECTIVE_NODE_BUCKETS[node],
                    "row_count": 0,
                }
            )
            continue
        try:
            rows = (
                _validated_composite_node007_rows(payload, path=path)
                if node == "node007"
                else _validated_gate_rows(payload, node=node, path=path)
            )
        except (KeyError, TypeError, ValueError) as exc:
            validation_errors.append(
                {
                    "code": "GPU_GATE_VALIDATION_FAILED",
                    "node": node,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        gate_rows[node] = (rows, path)
        source_audits.append(
            {
                "node": node,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": payload.get("status"),
                "source_protocol": (
                    payload.get("correction_measurement_protocol")
                    if node == "node007"
                    else payload.get("measurement_protocol")
                ),
                "effective_node_bucket": EFFECTIVE_NODE_BUCKETS[node],
                "row_count": len(rows),
            }
        )

    common = {
        "gate": "critical_gpu_available_lcb_cache_merge",
        "schema_version": 1,
        "available_nodes": list(AVAILABLE_NODES),
        "pending_nodes": list(PENDING_NODES),
        "global_all_hardware_cache_ready": False,
        "node_to_operational_execution_class": dict(EFFECTIVE_NODE_BUCKETS),
        "equivalence_gate_path": str(equivalence_gate_path),
        "base_cache_path": str(base_cache_path),
        "source_gate_audits": source_audits,
        "wait_reasons": wait_reasons,
        "validation_errors": validation_errors,
        "expected_inserted_count": 27,
        "inserted_count": 0,
        "inserted_rows": [],
        "available_scope_cache_ready": False,
        "legacy_index_unchanged": False,
        "claim_boundary": (
            "This cache covers only the nine declared empty-state actions on "
            "jtl110gpu, jtl110gpu2, and node007. The two RTX 3080 Ti hosts are "
            "kept as separate operational execution classes because the "
            "pre-registered equivalence gate failed; no cross-host service rows "
            "are pooled. node007 uses the p10 transport-corrected composite "
            "certificate. jtl311linux remains pending and is neither imputed nor "
            "represented by another host. The cache does not cover loaded states, "
            "unmeasured profiles, or arbitrary future workloads."
        ),
    }
    if validation_errors:
        return {**common, "status": "FAIL_VALIDATION", "pass": False}
    if wait_reasons or len(gate_rows) != len(AVAILABLE_NODES):
        return {**common, "status": "WAIT_AVAILABLE_GATES", "pass": False}

    base_path = Path(base_cache_path)
    if not base_path.is_file():
        common["validation_errors"].append(
            {"code": "MISSING_BASE_CACHE", "detail": str(base_path)}
        )
        return {**common, "status": "FAIL_BASE_CACHE", "pass": False}
    try:
        base_bytes = base_path.read_bytes()
        cache = ServiceRateCache.from_snapshot(json.loads(base_bytes.decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        common["validation_errors"].append(
            {"code": "INVALID_BASE_CACHE", "detail": f"{type(exc).__name__}: {exc}"}
        )
        return {**common, "status": "FAIL_BASE_CACHE", "pass": False}

    legacy_before = _legacy_index_snapshot(cache)
    inserted = []
    exact_keys: set[tuple[Any, ...]] = set()
    try:
        for node in AVAILABLE_NODES:
            rows, path = gate_rows[node]
            for row in rows:
                source_record = _profile_record(row, source=path)
                record = replace(
                    source_record,
                    node_bucket=EFFECTIVE_NODE_BUCKETS[node],
                    source=(
                        f"{path};physical_node={node};"
                        f"source_node_bucket={source_record.node_bucket}"
                    ),
                )
                key = (
                    record.workload_key,
                    record.workload_env,
                    record.node_bucket,
                    record.resource_state,
                    record.profile,
                )
                if key in exact_keys:
                    raise ValueError(f"duplicate exact GPU cache key {key!r}")
                exact_keys.add(key)
                cache.add(record, force_replace=True)
                lookup = cache.lookup_statewise_exact(
                    record.workload_key,
                    workload_env=record.workload_env,
                    node_bucket=record.node_bucket,
                    resource_state=record.resource_state,
                    allocation_workers=record.allocation_workers,
                    colocation_count=record.colocation_count,
                    resident_mix=record.resident_mix,
                )
                if not lookup.is_exact or lookup.record != record:
                    raise ValueError(f"exact statewise lookup failed for {key!r}")
                inserted.append({**record.snapshot(), "physical_node": node})
    except (KeyError, TypeError, ValueError) as exc:
        common["validation_errors"].append(
            {"code": "CACHE_INSERTION_FAILED", "detail": f"{type(exc).__name__}: {exc}"}
        )
        return {**common, "status": "FAIL_CACHE_INSERTION", "pass": False}

    if len(inserted) != 27:
        common["validation_errors"].append(
            {
                "code": "INSERTED_ROW_COUNT_MISMATCH",
                "detail": f"expected 27, observed {len(inserted)}",
            }
        )
        return {**common, "status": "FAIL_CACHE_INSERTION", "pass": False}
    legacy_after = _legacy_index_snapshot(cache)
    if legacy_after != legacy_before:
        common["validation_errors"].append(
            {"code": "LEGACY_INDEX_CHANGED", "detail": "statewise insertion changed legacy lookup"}
        )
        return {**common, "status": "FAIL_LEGACY_INDEX", "pass": False}

    return {
        **common,
        "status": "PASS",
        "pass": True,
        "base_cache_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "inserted_count": len(inserted),
        "inserted_rows": inserted,
        "available_scope_cache_ready": True,
        "legacy_index_unchanged": True,
        "service_cache_snapshot": cache.snapshot(),
    }


def _validated_composite_node007_rows(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> list[Mapping[str, Any]]:
    required = {
        "gate": "critical_gpu_p10_transport_correction_gate",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "certificate_ready": True,
        "node": "node007",
        "node_bucket": NODE_SPECS["node007"].node_bucket,
        "hardware_class": NODE_SPECS["node007"].hardware_class,
        "resource_state": "empty",
        "source_measurement_protocol": PROTOCOL,
        "correction_measurement_protocol": CORRECTION_PROTOCOL,
        "correction_scope": "pre_registered_measurement_transport_integrity",
        "performance_conditioned_selection": False,
        "all_measurements_ready": True,
        "expected_cell_count": 9,
        "expected_wave_count": 13,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"{path}: {field} expected {expected!r}, got {payload.get(field)!r}"
            )
    if payload.get("validation_errors") or payload.get("wait_reasons"):
        raise ValueError(f"{path}: passing composite gate contains errors or waits")
    certificate = payload.get("certificate") or {}
    if not all(
        certificate.get(field) is True
        for field in (
            "constructed",
            "finite_sample_rank_ready",
            "all_holdout_bounds_valid",
        )
    ):
        raise ValueError(f"{path}: composite certificate is not ready")
    rows = certificate.get("rows") or []
    expected = {
        _row_key(cell): cell
        for cell in campaign_cells(node="node007", wave=TRAINING_WAVES[0])
    }
    observed = {_row_key(row): row for row in rows}
    if len(rows) != 9 or set(observed) != set(expected):
        raise ValueError(f"{path}: composite certificate cell set differs")
    for key, row in observed.items():
        _validate_certificate_row(
            row,
            expected=expected[key],
            node="node007",
            path=path,
        )
    return [observed[key] for key in sorted(observed)]


def _validate_nonpooling_equivalence(
    payload: Mapping[str, Any],
    *,
    path: Path,
    errors: list[dict[str, Any]],
) -> None:
    comparison = payload.get("comparison") or {}
    required = {
        "gate": "critical_gpu_homogeneous_equivalence_gate",
        "status": "FAIL_EQUIVALENCE",
        "pass": False,
        "measurements_ready": True,
        "hardware_bucket_pooling_ready": False,
        "same_nine_workload_env_profile_cells": True,
        "representative_node": "jtl110gpu",
        "equivalence_node": "jtl110gpu2",
    }
    mismatches = [
        f"{field}={payload.get(field)!r} expected {expected!r}"
        for field, expected in required.items()
        if payload.get(field) != expected
    ]
    if comparison.get("constructed") is not True:
        mismatches.append("comparison.constructed is not true")
    if int(comparison.get("matched_cell_count") or 0) != 9:
        mismatches.append("comparison.matched_cell_count is not 9")
    if payload.get("validation_errors") or payload.get("wait_reasons"):
        mismatches.append("equivalence gate contains validation errors or waits")
    if mismatches:
        errors.append(
            {
                "code": "NONPOOLING_EQUIVALENCE_CONTRACT_INVALID",
                "detail": f"{path}: " + "; ".join(mismatches),
            }
        )


def _load_json(
    path: Path | None,
    waits: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        waits.append({"code": "MISSING_SOURCE", "detail": str(path)})
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append({"code": "UNREADABLE_SOURCE", "detail": f"{path}: {exc}"})
        return None
    if not isinstance(payload, dict):
        errors.append({"code": "INVALID_SOURCE_ROOT", "detail": str(path)})
        return None
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--jtl110gpu-gate", type=Path, default=DEFAULT_GATE_PATHS["jtl110gpu"])
    parser.add_argument("--jtl110gpu2-gate", type=Path, default=DEFAULT_GATE_PATHS["jtl110gpu2"])
    parser.add_argument("--node007-gate", type=Path, default=DEFAULT_GATE_PATHS["node007"])
    parser.add_argument("--equivalence-gate", type=Path, default=DEFAULT_EQUIVALENCE_GATE)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--cache-output", type=Path, default=DEFAULT_CACHE_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_available_lcb_cache_merge(
        base_cache_path=args.base_cache,
        gate_paths={
            "jtl110gpu": args.jtl110gpu_gate,
            "jtl110gpu2": args.jtl110gpu2_gate,
            "node007": args.node007_gate,
        },
        equivalence_gate_path=args.equivalence_gate,
    )
    disk = write_merge_outputs(
        report,
        report_path=args.report_output,
        markdown_path=args.markdown_output,
        cache_output_path=args.cache_output,
    )
    print(json.dumps({"status": disk["status"], "pass": disk["pass"], "cache_output_written": disk["cache_output_written"]}, indent=2, sort_keys=True))
    if disk["pass"]:
        return 0
    return 3 if str(disk["status"]).startswith("WAIT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
