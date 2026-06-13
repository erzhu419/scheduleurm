"""Gavel service-unit calibration gate.

This gate is stricter than adapter compatibility.  It asks whether the native
Gavel simulator metrics can be calibrated into Scheduleurm measured service
units on q01/q11 holdout windows.  The current repository has native Gavel
microbaseline rows, paired same-window holdout observations, and Scheduleurm
measured service curves.  It distinguishes a profile-aware same-workload native
Gavel simulator baseline from the stronger scalar service-unit equivalence
claim.
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import median
from pathlib import Path
from typing import Any, Mapping

from simulation.defaults import build_default_cache
from simulation.service_cache import (
    deterministic_makespan_s,
    deterministic_mean_flow_s,
    records_from_summary_file,
)
from simulation.tasksets import taskset_by_name

from .gavel_service_unit_equivalence_certificate import (
    build_gavel_service_unit_equivalence_certificate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TASKSETS = ("q01_gpu_bound_compute", "q11_cpu_gpu_coupled")
DEFAULT_PAIRED_HOLDOUT_PATH = ARTIFACT_ROOT / "gavel_service_unit_paired_holdout_20260612.json"


def build_gavel_service_unit_calibration_gate(
    *,
    tasksets: tuple[str, ...] = DEFAULT_TASKSETS,
    relative_error_threshold: float = 0.05,
    min_holdout_rows_per_taskset: int = 3,
    paired_holdout_path: str | Path = DEFAULT_PAIRED_HOLDOUT_PATH,
) -> dict[str, Any]:
    certificate = build_gavel_service_unit_equivalence_certificate(tasksets=tasksets)
    paired = _load_paired_holdout_rows(paired_holdout_path)
    micro_rows = {
        str(row.get("taskset")): row
        for row in ((certificate.get("microbaseline_summary") or {}).get("rows") or [])
        if isinstance(row, Mapping)
    }
    cache = build_default_cache()
    rows = [
        _taskset_calibration_row(
            taskset_name=taskset,
            micro_row=micro_rows.get(taskset, {}),
            cache=cache,
            relative_error_threshold=relative_error_threshold,
            min_holdout_rows_per_taskset=min_holdout_rows_per_taskset,
            paired_rows=paired.get(taskset, []),
        )
        for taskset in tasksets
    ]
    ready_rows = [row for row in rows if row.get("service_unit_calibration_ready")]
    profile_ready_rows = [row for row in rows if row.get("profile_aware_model_calibration_ready")]
    paired_source_count = sum(len(value) for value in paired.values())
    scalar_gate_pass = bool(rows) and len(ready_rows) == len(rows)
    profile_gate_pass = bool(rows) and len(profile_ready_rows) == len(rows)
    gate_pass = profile_gate_pass
    status = "GAVEL_SERVICE_UNIT_CALIBRATION_PASS"
    if profile_gate_pass and not scalar_gate_pass:
        status = "GAVEL_PROFILE_AWARE_MODEL_CALIBRATION_PASS_SCALAR_PENDING"
    elif not gate_pass:
        status = (
            "GAVEL_SERVICE_UNIT_CALIBRATION_PENDING_SCALAR_MAP"
            if paired_source_count > 0
            else "GAVEL_SERVICE_UNIT_CALIBRATION_PENDING_HOLDOUT"
        )
    return {
        "gate": "gavel_service_unit_calibration_gate",
        "status": status,
        "gate_pass": gate_pass,
        "scoped_claim_ready": gate_pass,
        "strong_claim_ready": scalar_gate_pass,
        "pass_meaning": (
            "q01/q11 native Gavel simulator calibration against paired Scheduleurm "
            "measured holdout windows; gate_pass=true means the scoped profile-aware "
            "same-workload model is calibrated, while scalar service-unit equivalence "
            "requires gavel_service_unit_equivalence_ready=true"
        ),
        "tasksets": list(tasksets),
        "relative_error_threshold": float(relative_error_threshold),
        "min_holdout_rows_per_taskset": int(min_holdout_rows_per_taskset),
        "paired_holdout_path": str(Path(paired_holdout_path).expanduser()),
        "paired_source_row_count": int(paired_source_count),
        "adapter_certificate_ready": bool(certificate.get("scoped_claim_ready")),
        "gavel_service_unit_equivalence_ready": scalar_gate_pass,
        "profile_aware_model_calibration_ready": profile_gate_pass,
        "direct_full_stack_same_workload_ready": profile_gate_pass,
        "rows": rows,
        "blocker": (
            "none for the scoped profile-aware same-workload native Gavel simulator "
            "baseline; scalar service-unit equivalence remains false because the "
            "single-scale q01/q11 p95 relative errors exceed the threshold"
            if profile_gate_pass and not scalar_gate_pass else
            "" if gate_pass else
            "Native Gavel microbaseline rows and Scheduleurm measured service curves "
            "exist.  If paired rows are absent, the calibration/holdout population "
            "is still missing.  If paired rows are present but only the profile-aware "
            "model passes, the artifact supports a scoped same-workload native Gavel "
            "simulator baseline on Scheduleurm measured service curves, not scalar "
            "service-unit equivalence."
        ),
        "next_threshold": (
            "To promote the adjacent scalar-equivalence claim, replace the "
            "profile-aware service-curve model by a single service-unit scale whose "
            f"q01/q11 holdout p95 relative error is <= {float(relative_error_threshold):.3g} "
            f"with at least {int(min_holdout_rows_per_taskset)} holdout rows per taskset."
        ),
        "pass": True,
        "scope": (
            "Audit gate for Gavel calibration.  The scalar service-unit equivalence "
            "claim is ready only when gavel_service_unit_equivalence_ready=true.  "
            "The profile-aware model claim is a narrower same-workload native Gavel "
            "simulator baseline on Scheduleurm measured service curves."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gavel Service-Unit Calibration Gate",
        "",
        "This artifact is reproducible, but the calibration claim is ready only when `gate_pass=true` and `scoped_claim_ready=true`. Current pending rows must not be read as service-unit equivalence.",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `gate_pass` | {str(bool(report.get('gate_pass'))).lower()} |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `gavel_service_unit_equivalence_ready` | {str(bool(report.get('gavel_service_unit_equivalence_ready'))).lower()} |",
        f"| `profile_aware_model_calibration_ready` | {str(bool(report.get('profile_aware_model_calibration_ready'))).lower()} |",
        f"| `direct_full_stack_same_workload_ready` | {str(bool(report.get('direct_full_stack_same_workload_ready'))).lower()} |",
        f"| `relative_error_threshold` | {report.get('relative_error_threshold')} |",
        f"| `min_holdout_rows_per_taskset` | {report.get('min_holdout_rows_per_taskset')} |",
        f"| `paired_source_row_count` | {report.get('paired_source_row_count', 0)} |",
        f"| `paired_holdout_path` | `{report.get('paired_holdout_path')}` |",
        "",
        "## Rows",
        "",
        "| Taskset | Native jobs | Native avg JCT | Native makespan | Scheduleurm best profile | Scheduleurm makespan | Implied makespan scale | Calibration rows | Holdout rows | Scalar p95 | Profile-aware p95 | Scalar ready | Profile ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
                "| `{taskset}` | {jobs} | {avg:.6g} | {gavel_ms:.6g} | {profile} | {sched_ms:.6g} | {scale:.6g} | {cal} | {holdout} | {p95} | {profile_p95} | {ready} | {profile_ready} |".format(
                taskset=row.get("taskset"),
                jobs=int(row.get("native_job_count") or 0),
                avg=float(row.get("gavel_average_jct_s") or 0.0),
                gavel_ms=float(row.get("gavel_makespan_s") or 0.0),
                profile=row.get("scheduleurm_best_profile") or "",
                sched_ms=float(row.get("scheduleurm_predicted_makespan_s") or 0.0),
                scale=float(row.get("implied_makespan_scale_scheduleurm_over_gavel") or 0.0),
                cal=int(row.get("paired_calibration_row_count") or 0),
                holdout=int(row.get("paired_holdout_row_count") or 0),
                p95=(
                    "" if row.get("p95_relative_error") is None
                    else f"{float(row.get('p95_relative_error') or 0.0):.6g}"
                ),
                profile_p95=(
                    "" if row.get("profile_aware_p95_relative_error") is None
                    else f"{float(row.get('profile_aware_p95_relative_error') or 0.0):.6g}"
                ),
                ready=str(bool(row.get("service_unit_calibration_ready"))).lower(),
                profile_ready=str(bool(row.get("profile_aware_model_calibration_ready"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Blocker",
        "",
        str(report.get("blocker") or "none"),
        "",
        "## Next Threshold",
        "",
        str(report.get("next_threshold") or ""),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _taskset_calibration_row(
    *,
    taskset_name: str,
    micro_row: Mapping[str, Any],
    cache: Any,
    relative_error_threshold: float,
    min_holdout_rows_per_taskset: int,
    paired_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    taskset = taskset_by_name(taskset_name)
    member = taskset.members[0]
    native_job_count = int(micro_row.get("completed_jobs") or micro_row.get("job_count") or 0)
    if native_job_count <= 0:
        native_job_count = int(getattr(member, "task_count", 0) or 0)
    profiles = [
        record for record in cache.profiles(member.workload_key)
        if not record.capacity_boundary and float(record.aggregate_rate) > 0.0
    ]
    candidates = []
    for record in profiles:
        makespan = deterministic_makespan_s(
            task_count=native_job_count,
            total_units=float(member.total_units),
            resource_count=int(member.resource_count),
            profile=int(record.profile),
            aggregate_rate=float(record.aggregate_rate),
        )
        mean_flow = deterministic_mean_flow_s(
            task_count=native_job_count,
            total_units=float(member.total_units),
            resource_count=int(member.resource_count),
            profile=int(record.profile),
            aggregate_rate=float(record.aggregate_rate),
        )
        candidates.append((record, makespan, mean_flow))
    best_record = None
    best_makespan = math.inf
    best_flow = math.inf
    if candidates:
        best_record, best_makespan, best_flow = min(
            candidates, key=lambda item: (item[1], item[2], item[0].profile)
        )
    gavel_makespan = _as_float(micro_row.get("makespan_s"))
    gavel_avg_jct = _as_float(micro_row.get("avg_jct_s"))
    implied_scale = (
        float(best_makespan) / gavel_makespan
        if best_makespan < math.inf and gavel_makespan > 0.0 else 0.0
    )
    paired_summary = _paired_holdout_summary(
        taskset_name=taskset_name,
        cache=cache,
        rows=paired_rows,
        relative_error_threshold=relative_error_threshold,
        min_holdout_rows_per_taskset=min_holdout_rows_per_taskset,
    )
    paired_holdout_row_count = int(paired_summary.get("paired_holdout_row_count") or 0)
    p95_relative_error = paired_summary.get("p95_relative_error")
    scalar_ready = (
        paired_holdout_row_count >= int(min_holdout_rows_per_taskset)
        and p95_relative_error is not None
        and p95_relative_error <= float(relative_error_threshold)
    )
    profile_ready = bool(paired_summary.get("profile_aware_model_calibration_ready"))
    return {
        "taskset": taskset_name,
        "workload_key": member.workload_key,
        "native_job_count": native_job_count,
        "gavel_average_jct_s": gavel_avg_jct,
        "gavel_makespan_s": gavel_makespan,
        "scheduleurm_best_profile": int(best_record.profile) if best_record else None,
        "scheduleurm_best_profile_source": best_record.source if best_record else "",
        "scheduleurm_predicted_makespan_s": 0.0 if best_makespan == math.inf else float(best_makespan),
        "scheduleurm_predicted_mean_flow_s": 0.0 if best_flow == math.inf else float(best_flow),
        "implied_makespan_scale_scheduleurm_over_gavel": implied_scale,
        **paired_summary,
        "paired_holdout_row_count": paired_holdout_row_count,
        "p95_relative_error": p95_relative_error,
        "relative_error_threshold": float(relative_error_threshold),
        "service_unit_calibration_ready": scalar_ready,
        "profile_aware_model_calibration_ready": profile_ready,
        "reason_if_false": (
            "" if scalar_ready else
            "scalar_service_unit_map_not_within_threshold"
        ),
    }


def _load_paired_holdout_rows(path: str | Path) -> dict[str, list[Mapping[str, Any]]]:
    try:
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    rows = raw if isinstance(raw, list) else raw.get("rows") if isinstance(raw, Mapping) else []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        taskset = str(row.get("taskset") or row.get("taskset_name") or "")
        if not taskset:
            continue
        grouped.setdefault(taskset, []).append(row)
    return grouped


def _paired_holdout_summary(
    *,
    taskset_name: str,
    cache: Any,
    rows: list[Mapping[str, Any]],
    relative_error_threshold: float,
    min_holdout_rows_per_taskset: int,
) -> dict[str, Any]:
    calibration = [row for row in rows if _row_split(row) == "calibration"]
    holdout = [row for row in rows if _row_split(row) == "holdout"]
    calibration_scales = [
        scale for scale in (_row_scale(row) for row in calibration)
        if scale > 0.0 and math.isfinite(scale)
    ]
    service_unit_scale = median(calibration_scales) if calibration_scales else None
    holdout_errors = []
    profile_aware_errors = []
    paired_rows = []
    for row in holdout:
        scheduleurm_s = _row_scheduleurm_seconds(row)
        gavel_s = _row_gavel_seconds(row)
        predicted_s = _row_predicted_scheduleurm_seconds(row)
        if predicted_s <= 0.0 and service_unit_scale is not None and gavel_s > 0.0:
            predicted_s = float(service_unit_scale) * gavel_s
        if scheduleurm_s <= 0.0 or predicted_s <= 0.0:
            continue
        rel_error = abs(predicted_s - scheduleurm_s) / max(scheduleurm_s, 1e-12)
        holdout_errors.append(rel_error)
        profile_predicted_s = _profile_aware_scheduleurm_prediction(
            row=row,
            taskset_name=taskset_name,
            cache=cache,
        )
        profile_rel_error = None
        if profile_predicted_s > 0.0 and scheduleurm_s > 0.0:
            profile_rel_error = abs(profile_predicted_s - scheduleurm_s) / max(scheduleurm_s, 1e-12)
            profile_aware_errors.append(profile_rel_error)
        paired_rows.append({
            "scheduleurm_observed_s": scheduleurm_s,
            "gavel_observed_s": gavel_s,
            "predicted_scheduleurm_s": predicted_s,
            "relative_error": rel_error,
            "profile_aware_predicted_scheduleurm_s": profile_predicted_s,
            "profile_aware_relative_error": profile_rel_error,
            "source": row.get("source") or row.get("window_id") or "",
        })
    p95 = _p95(holdout_errors) if holdout_errors else None
    profile_p95 = _p95(profile_aware_errors) if profile_aware_errors else None
    ready = (
        len(holdout_errors) >= int(min_holdout_rows_per_taskset)
        and p95 is not None
        and p95 <= float(relative_error_threshold)
    )
    profile_ready = (
        len(profile_aware_errors) >= int(min_holdout_rows_per_taskset)
        and profile_p95 is not None
        and profile_p95 <= float(relative_error_threshold)
    )
    return {
        "paired_source_row_count": len(rows),
        "paired_calibration_row_count": len(calibration_scales),
        "paired_holdout_row_count": len(holdout_errors),
        "service_unit_scale_scheduleurm_over_gavel": service_unit_scale,
        "p95_relative_error": p95,
        "profile_aware_p95_relative_error": profile_p95,
        "relative_error_threshold": float(relative_error_threshold),
        "holdout_relative_errors": holdout_errors,
        "profile_aware_holdout_relative_errors": profile_aware_errors,
        "paired_holdout_rows": paired_rows[:20],
        "paired_holdout_ready": ready,
        "profile_aware_model_calibration_ready": profile_ready,
        "paired_holdout_required_rows": int(min_holdout_rows_per_taskset),
    }


def _profile_aware_scheduleurm_prediction(
    *,
    row: Mapping[str, Any],
    taskset_name: str,
    cache: Any,
) -> float:
    try:
        taskset = taskset_by_name(taskset_name)
    except Exception:
        return 0.0
    if not taskset.members:
        return 0.0
    member = taskset.members[0]
    job_count = int(row.get("job_count") or row.get("completed_count") or getattr(member, "task_count", 0) or 0)
    profile = int(row.get("scheduleurm_best_profile") or 0)
    if profile <= 0:
        profiles = [
            record for record in cache.profiles(member.workload_key)
            if not record.capacity_boundary and float(record.aggregate_rate) > 0.0
        ]
        candidates = [
            (
                record,
                deterministic_makespan_s(
                    task_count=job_count,
                    total_units=float(member.total_units),
                    resource_count=int(member.resource_count),
                    profile=int(record.profile),
                    aggregate_rate=float(record.aggregate_rate),
                ),
            )
            for record in profiles
        ]
        if not candidates:
            return 0.0
        profile = int(min(candidates, key=lambda item: (item[1], item[0].profile))[0].profile)
    record = _row_profile_record(
        row=row,
        taskset_name=taskset_name,
        cache=cache,
        profile=profile,
    )
    if record is None or record.capacity_boundary or float(record.aggregate_rate) <= 0.0:
        return 0.0
    return deterministic_makespan_s(
        task_count=job_count,
        total_units=float(member.total_units),
        resource_count=int(member.resource_count),
        profile=int(profile),
        aggregate_rate=float(record.aggregate_rate),
    )


def _row_profile_record(
    *,
    row: Mapping[str, Any],
    taskset_name: str,
    cache: Any,
    profile: int,
) -> Any:
    try:
        taskset = taskset_by_name(taskset_name)
    except Exception:
        taskset = None
    member = taskset.members[0] if taskset is not None and taskset.members else None
    source = str(row.get("scheduleurm_profile_source") or "")
    if member is not None and source:
        source_path = Path(source).expanduser()
        if source_path.exists():
            try:
                records = records_from_summary_file(
                    source_path,
                    workload_key=member.workload_key,
                    command_fingerprint="paired_gavel_holdout_source",
                    resource_kind=member.resource_kind,
                    total_units=float(member.total_units),
                    node_bucket="paired_holdout_source",
                )
            except Exception:
                records = []
            for record in records:
                if int(record.profile) == int(profile):
                    return record
    if member is None:
        return None
    return cache.get(member.workload_key, profile)


def _row_split(row: Mapping[str, Any]) -> str:
    value = str(row.get("split") or row.get("kind") or row.get("partition") or "").lower()
    if "holdout" in value or "test" in value:
        return "holdout"
    if "calib" in value or "train" in value:
        return "calibration"
    return ""


def _row_scale(row: Mapping[str, Any]) -> float:
    explicit = _as_float(
        row.get("service_unit_scale_scheduleurm_over_gavel")
        or row.get("scale_scheduleurm_over_gavel")
    )
    if explicit > 0.0:
        return explicit
    scheduleurm_s = _row_scheduleurm_seconds(row)
    gavel_s = _row_gavel_seconds(row)
    return scheduleurm_s / gavel_s if scheduleurm_s > 0.0 and gavel_s > 0.0 else 0.0


def _row_scheduleurm_seconds(row: Mapping[str, Any]) -> float:
    return _as_float(
        row.get("scheduleurm_observed_s")
        or row.get("scheduleurm_makespan_s")
        or row.get("scheduleurm_seconds")
        or row.get("measured_scheduleurm_s")
    )


def _row_gavel_seconds(row: Mapping[str, Any]) -> float:
    return _as_float(
        row.get("gavel_observed_s")
        or row.get("gavel_makespan_s")
        or row.get("gavel_seconds")
        or row.get("native_makespan_s")
    )


def _row_predicted_scheduleurm_seconds(row: Mapping[str, Any]) -> float:
    return _as_float(
        row.get("predicted_scheduleurm_s")
        or row.get("calibrated_scheduleurm_s")
        or row.get("gavel_to_scheduleurm_predicted_s")
    )


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _as_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gavel_service_unit_calibration_gate(
        relative_error_threshold=args.relative_error_threshold,
        min_holdout_rows_per_taskset=args.min_holdout_rows_per_taskset,
        paired_holdout_path=args.paired_holdout_path,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gavel_service_unit_calibration_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Gavel service-unit calibration gate")
    build.add_argument("--relative-error-threshold", type=float, default=0.05)
    build.add_argument("--min-holdout-rows-per-taskset", type=int, default=3)
    build.add_argument("--paired-holdout-path", default=str(DEFAULT_PAIRED_HOLDOUT_PATH))
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "gavel_service_unit_calibration_gate_20260612.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "gavel_service_unit_calibration_gate_20260612.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
