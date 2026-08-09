"""Export the cache-bound migration recalibration gate for final replay.

The upstream gate recomputes controlled migration decisions from exact rows in
the final service cache and reuses old artifacts only for measured checkpoint,
sync, staging, resume, lost-work, and risk costs.  This module validates that
gate and emits the narrower schema consumed by ``unified_hardware_or_replay``.
It never launches or migrates a task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_FINAL_CACHE = ARTIFACT_ROOT / "service_cache_v2_cpu_gpu_loaded_final_20260809.json"
DEFAULT_RECALIBRATION = ARTIFACT_ROOT / "migration_rate_recalibration_gate_20260809.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "unified_migration_recalibration_certificate_20260809.json"
DEFAULT_MARKDOWN = ARTIFACT_ROOT / "unified_migration_recalibration_certificate_20260809.md"
EXPECTED_FAMILIES = ("pure_cpu", "pure_gpu", "hybrid_rl")
EXPECTED_POINTS = (0.25, 0.50, 0.75)


def build_unified_migration_recalibration_certificate(
    *,
    final_cache_path: Path = DEFAULT_FINAL_CACHE,
    recalibration_path: Path = DEFAULT_RECALIBRATION,
) -> dict[str, Any]:
    cache_path = Path(final_cache_path)
    gate_path = Path(recalibration_path)
    common = {
        "gate": "unified_migration_recalibration_certificate",
        "schema_version": 1,
        "final_cache_path": str(cache_path),
        "recalibration_gate_path": str(gate_path),
        "rates_recomputed_from_final_cache": False,
        "no_touch_safety_ready": False,
        "rows": [],
    }
    missing = [str(path) for path in (cache_path, gate_path) if not path.is_file()]
    if missing:
        return {
            **common,
            "status": "WAIT_INPUTS",
            "pass": False,
            "blockers": [{"code": "MISSING_INPUT", "path": path} for path in missing],
        }

    try:
        cache_bytes = cache_path.read_bytes()
        cache_payload = json.loads(cache_bytes.decode("utf-8"))
        if not isinstance(cache_payload, dict):
            raise ValueError("final cache root must be an object")
        gate_bytes = gate_path.read_bytes()
        gate = json.loads(gate_bytes.decode("utf-8"))
        if not isinstance(gate, dict):
            raise ValueError("recalibration gate root must be an object")
        cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
        _validate_gate_header(gate, cache_sha256=cache_sha256)
        rows = _export_rows(gate.get("rows"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            **common,
            "status": "FAIL_RECALIBRATION_CONTRACT",
            "pass": False,
            "blockers": [
                {
                    "code": "RECALIBRATION_CONTRACT_INVALID",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    return {
        **common,
        "status": "MIGRATION_RECALIBRATION_PASS",
        "pass": True,
        "service_cache_sha256": cache_sha256,
        "recalibration_gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "rates_recomputed_from_final_cache": True,
        "no_touch_safety_ready": True,
        "row_count": len(rows),
        "beneficial_row_count": sum(row["beneficial_by_threshold"] is True for row in rows),
        "nonbeneficial_row_count": sum(row["beneficial_by_threshold"] is False for row in rows),
        "rows": rows,
        "blockers": [],
        "claim_boundary": (
            "This certificate covers only controlled benchmark migrations with "
            "verified checkpoint/resume measurements at 25, 50, and 75 percent "
            "progress. Rates are exact hardware/load-state per-task lower service "
            "from the bound final cache. Ordinary running user tasks are excluded."
        ),
    }


def _validate_gate_header(gate: Mapping[str, Any], *, cache_sha256: str) -> None:
    if gate.get("gate") != "migration_rate_recalibration_gate":
        raise ValueError("upstream gate identity mismatch")
    if gate.get("schema_version") != 1:
        raise ValueError("upstream schema_version mismatch")
    if gate.get("status") != "PASS" or gate.get("pass") is not True:
        raise ValueError("upstream recalibration gate is not passing")
    if gate.get("certificate_ready") is not True:
        raise ValueError("upstream certificate_ready is not true")
    if str(gate.get("final_unified_cache_sha256") or "") != cache_sha256:
        raise ValueError("upstream recalibration is not bound to the final cache hash")
    if gate.get("ordinary_user_tasks_excluded") is not True:
        raise ValueError("ordinary user-task exclusion is missing")
    if gate.get("legacy_or_state_fallback_allowed") is not False:
        raise ValueError("legacy/state fallback must be disabled")
    if gate.get("old_cost_artifact_rates_used") is not False:
        raise ValueError("historical migration rates were reused")
    if gate.get("old_cost_artifact_beneficial_flags_inherited") is not False:
        raise ValueError("historical beneficial flags were inherited")
    if int(gate.get("blocked_row_count") or 0) != 0:
        raise ValueError("upstream gate has blocked rows")
    rows = gate.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("upstream gate rows are missing")
    if int(gate.get("required_row_count") or -1) != len(rows):
        raise ValueError("upstream required_row_count mismatch")
    if int(gate.get("ready_row_count") or -1) != len(rows):
        raise ValueError("upstream ready_row_count mismatch")


def _export_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("upstream rows must be a list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    coverage = {family: set() for family in EXPECTED_FAMILIES}
    for index, source in enumerate(value):
        if not isinstance(source, dict):
            raise ValueError(f"row {index} must be an object")
        if source.get("recalculation_ready") is not True:
            raise ValueError(f"row {index} is not recalculation-ready")
        family = _text(source.get("family"), f"row {index}.family")
        if family not in coverage:
            raise ValueError(f"row {index} has unsupported family {family!r}")
        point = _finite(source.get("progress_fraction"), f"row {index}.progress_fraction")
        if not any(math.isclose(point, expected, abs_tol=1e-12) for expected in EXPECTED_POINTS):
            raise ValueError(f"row {index} has unsupported progress point {point}")
        coverage[family].add(point)
        migration_id = _text(source.get("migration_id"), f"row {index}.migration_id")
        action_id = f"migrate:{migration_id}:p{int(round(point * 100))}"
        if action_id in seen:
            raise ValueError(f"duplicate action id {action_id!r}")
        seen.add(action_id)
        current_rate = _positive(
            source.get("current_lower_service_units_per_s"),
            f"{action_id}.current_lower_service",
        )
        target_rate = _positive(
            source.get("target_lower_service_units_per_s"),
            f"{action_id}.target_lower_service",
        )
        remaining = _positive(source.get("remaining_work_units"), f"{action_id}.remaining_work")
        cost = _validated_cost(source.get("migration_cost"), label=action_id)
        keep_s = remaining / current_rate
        migrate_s = cost["total_penalty_units"] + remaining / target_rate
        beneficial = keep_s > migrate_s
        if source.get("beneficial_by_recalculated_threshold") is not beneficial:
            raise ValueError(f"{action_id}: upstream beneficial decision is inconsistent")
        if source.get("migration_action_theorem_ready") is not beneficial:
            raise ValueError(f"{action_id}: upstream theorem-ready decision is inconsistent")
        _close(source.get("keep_remaining_completion_s"), keep_s, f"{action_id}.keep time")
        _close(source.get("migrate_remaining_completion_s"), migrate_s, f"{action_id}.migrate time")
        source_key = _service_key(source.get("source_state_request"), label=f"{action_id}.source")
        target_key = _service_key(source.get("target_state_request"), label=f"{action_id}.destination")
        unit = _text(
            (source.get("source_rate_record") or {}).get("unit"),
            f"{action_id}.remaining_work_unit",
        )
        rows.append(
            {
                "action_id": action_id,
                "family": family,
                "workload_key": _text(source.get("workload_key"), f"{action_id}.workload_key"),
                "progress_fraction": point,
                "measurement_valid": True,
                "controlled_benchmark": True,
                "checkpoint_verified": True,
                "resume_verified": True,
                "rates_recomputed_from_cache": True,
                "ordinary_running_tasks_touched": False,
                "source_service_key": source_key,
                "destination_service_key": target_key,
                "current_lower_service": current_rate,
                "target_lower_service": target_rate,
                "effective_migration_lower_service": remaining / migrate_s,
                "remaining_work": remaining,
                "remaining_work_unit": unit,
                "migration_cost": cost,
                "beneficial_by_threshold": beneficial,
                "theorem_ready": beneficial,
                "source_recalibration_audit": {
                    "migration_net_saving_s": source.get("migration_net_saving_s"),
                    "cost_source": source.get("cost_source"),
                },
            }
        )
    for family, points in coverage.items():
        if not all(any(math.isclose(point, expected, abs_tol=1e-12) for point in points) for expected in EXPECTED_POINTS):
            raise ValueError(f"migration family {family!r} lacks 25/50/75 percent coverage")
    return sorted(rows, key=lambda row: row["action_id"])


def _service_key(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} state request must be an object")
    return {
        "workload_key": _text(value.get("workload_key"), f"{label}.workload_key"),
        "workload_env": _text(value.get("workload_env"), f"{label}.workload_env"),
        "node_bucket": _text(value.get("node_bucket"), f"{label}.node_bucket"),
        "resource_state": _text(value.get("resource_state"), f"{label}.resource_state"),
        "profile": _positive_int(value.get("profile"), f"{label}.profile"),
        "allocation_workers": _positive_int(
            value.get("allocation_workers"), f"{label}.allocation_workers"
        ),
        "colocation_count": _positive_int(
            value.get("colocation_count"), f"{label}.colocation_count"
        ),
        "resident_mix": str(value.get("resident_mix") or ""),
    }


def _validated_cost(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}.migration_cost must be an object")
    fields = (
        "checkpoint_flush_s",
        "sync_s",
        "environment_staging_s",
        "resume_warmup_s",
        "lost_work_s",
    )
    out = {field: _nonnegative(value.get(field), f"{label}.{field}") for field in fields}
    out["risk_penalty_units"] = _nonnegative(
        value.get("risk_penalty_units"), f"{label}.risk_penalty_units"
    )
    out["total_time_s"] = _nonnegative(value.get("total_time_s"), f"{label}.total_time_s")
    out["total_penalty_units"] = _nonnegative(
        value.get("total_penalty_units"), f"{label}.total_penalty_units"
    )
    _close(out["total_time_s"], sum(out[field] for field in fields), f"{label}.total_time_s")
    _close(
        out["total_penalty_units"],
        out["total_time_s"] + out["risk_penalty_units"],
        f"{label}.total_penalty_units",
    )
    return out


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Unified Migration Recalibration Certificate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Final cache SHA-256: `{report.get('service_cache_sha256', '')}`",
        f"- Rows: `{len(report.get('rows') or [])}`",
        "",
    ]
    for blocker in report.get("blockers") or []:
        lines.append(f"- `{blocker.get('code')}`: {blocker.get('detail') or blocker.get('path') or ''}")
    if report.get("blockers"):
        lines.append("")
    if report.get("rows"):
        lines.extend(
            [
                "| Action | Family | Progress | Old lower rate | New lower rate | Effective migration rate | Beneficial |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["rows"]:
            lines.append(
                f"| `{row['action_id']}` | `{row['family']}` | {row['progress_fraction']:.2f} | "
                f"{row['current_lower_service']:.6g} | {row['target_lower_service']:.6g} | "
                f"{row['effective_migration_lower_service']:.6g} | "
                f"{str(bool(row['beneficial_by_threshold'])).lower()} |"
            )
        lines.append("")
    lines.extend([str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], *, output: Path, markdown_output: Path) -> None:
    _atomic_write(
        Path(output), json.dumps(dict(report), indent=2, sort_keys=True) + "\n"
    )
    _atomic_write(Path(markdown_output), markdown_report(report))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _nonnegative(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _close(actual: Any, expected: float, label: str) -> None:
    number = _finite(actual, label)
    if not math.isclose(number, float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} mismatch: {number!r} != {expected!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-cache", type=Path, default=DEFAULT_FINAL_CACHE)
    parser.add_argument("--recalibration", type=Path, default=DEFAULT_RECALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_unified_migration_recalibration_certificate(
        final_cache_path=args.final_cache,
        recalibration_path=args.recalibration,
    )
    write_outputs(report, output=args.output, markdown_output=args.markdown_output)
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, sort_keys=True))
    return 0 if report["pass"] else (3 if str(report["status"]).startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
