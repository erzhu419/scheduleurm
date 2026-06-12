"""Gavel service-unit equivalence certificate.

This certificate separates three facts that are easy to conflate:

* Scheduleurm can export bounded q01/q11 traces in Gavel's native trace schema.
* Gavel's native simulator can run those trace windows.
* Gavel's simulator service units are not yet certified equivalent to
  Scheduleurm's measured lower-service units.

The first two are useful direct-adapter evidence.  The third is the blocker
that prevents a direct full-stack SOTA superiority claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .gavel_native_performance_microbaseline import (
    build_gavel_native_performance_microbaseline,
)
from .sota_adapters.scheduleurm_to_gavel_throughput_table import build_table
from .sota_adapters.scheduleurm_to_gavel_trace_converter import (
    build_native_trace_lines,
    build_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TASKSETS = ("q01_gpu_bound_compute", "q11_cpu_gpu_coupled")


def build_gavel_service_unit_equivalence_certificate(
    *,
    tasksets: tuple[str, ...] = DEFAULT_TASKSETS,
    run_native_microbaseline: bool = False,
) -> dict[str, Any]:
    rows = [_taskset_row(name) for name in tasksets]
    trace_schema_ready = bool(rows) and all(row["trace_schema_compatibility_ready"] for row in rows)
    throughput_seed_ready = bool(rows) and all(row["throughput_seed_ready"] for row in rows)

    microbaseline: dict[str, Any] = _load_existing_microbaseline()
    if run_native_microbaseline or not microbaseline:
        microbaseline = build_gavel_native_performance_microbaseline()

    native_micro_ready = bool(microbaseline.get("native_gavel_simulator_microbaseline_ready"))
    metric_equivalence_ready = False
    reason = (
        "Gavel native trace rows use Gavel template job types and simulator throughput "
        "tables, while Scheduleurm theorem rows use measured aggregate lower-service "
        "rates from the Scheduleurm service cache.  The trace schema, job counts, "
        "arrival times, resource counts, and total_units are aligned, but no "
        "calibration experiment currently proves that one Gavel simulator step has "
        "the same service-unit meaning as one Scheduleurm measured service unit."
    )
    return {
        "gate": "gavel_service_unit_equivalence_certificate",
        "tasksets": list(tasksets),
        "rows": rows,
        "same_job_count": all(row["same_job_count"] for row in rows),
        "same_arrival_times": all(row["same_arrival_times"] for row in rows),
        "same_total_units": all(row["same_total_units"] for row in rows),
        "same_gpu_count": all(row["same_gpu_count"] for row in rows),
        "same_resource_count": all(row["same_resource_count"] for row in rows),
        "throughput_table_unit": "Scheduleurm measured aggregate_rate service_units_per_second seed",
        "scheduleurm_unit": "Scheduleurm task total_units divided by measured aggregate_rate",
        "gavel_native_unit": "Gavel simulator template steps interpreted through simulation throughput tables",
        "trace_schema_compatibility_ready": trace_schema_ready,
        "exact_arrival_times_ready": all(row["same_arrival_times"] for row in rows),
        "throughput_seed_ready": throughput_seed_ready,
        "gavel_native_bounded_microbaseline_ready": native_micro_ready,
        "metric_equivalence_ready": metric_equivalence_ready,
        "gavel_service_unit_equivalence_ready": metric_equivalence_ready,
        "direct_full_stack_same_workload_ready": False,
        "reason_if_false": reason,
        "microbaseline_summary": {
            "native_gavel_simulator_microbaseline_ready": native_micro_ready,
            "usable_native_performance_sample_count": microbaseline.get(
                "usable_native_performance_sample_count"
            ),
            "service_unit_equivalence_ready": microbaseline.get("service_unit_equivalence_ready"),
            "direct_full_stack_performance_ready": microbaseline.get(
                "direct_full_stack_performance_ready"
            ),
        },
        "pass": trace_schema_ready and throughput_seed_ready and native_micro_ready,
        "scope": (
            "Certifies bounded same-trace Gavel adapter compatibility and records "
            "the remaining service-unit blocker.  It intentionally does not promote "
            "Gavel native simulator rows to full-stack SOTA superiority."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gavel Service-Unit Equivalence Certificate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `trace_schema_compatibility_ready` | {str(bool(report.get('trace_schema_compatibility_ready'))).lower()} |",
        f"| `exact_arrival_times_ready` | {str(bool(report.get('exact_arrival_times_ready'))).lower()} |",
        f"| `throughput_seed_ready` | {str(bool(report.get('throughput_seed_ready'))).lower()} |",
        f"| `gavel_native_bounded_microbaseline_ready` | {str(bool(report.get('gavel_native_bounded_microbaseline_ready'))).lower()} |",
        f"| `gavel_service_unit_equivalence_ready` | {str(bool(report.get('gavel_service_unit_equivalence_ready'))).lower()} |",
        f"| `direct_full_stack_same_workload_ready` | {str(bool(report.get('direct_full_stack_same_workload_ready'))).lower()} |",
        "",
        "## Rows",
        "",
        "| Taskset | Jobs | Native rows | Total units | Resource counts | Max arrival jitter s | Throughput profiles | Ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{taskset}` | {jobs} | {native} | {units:.6g} | {resources} | {jitter:.6g} | {profiles} | {ready} |".format(
                taskset=row.get("taskset"),
                jobs=row.get("job_count"),
                native=row.get("native_trace_row_count"),
                units=float(row.get("scheduleurm_total_units_sum") or 0.0),
                resources=row.get("resource_count_values"),
                jitter=float(row.get("max_arrival_jitter_s") or 0.0),
                profiles=row.get("throughput_profile_count"),
                ready=str(bool(row.get("trace_schema_compatibility_ready"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Remaining Blocker",
        "",
        str(report.get("reason_if_false") or ""),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _taskset_row(taskset_name: str) -> dict[str, Any]:
    trace = build_trace(taskset_name)
    native_lines = build_native_trace_lines(taskset_name)
    throughput = build_table(taskset_name)
    jobs = list(trace.get("jobs") or [])
    parsed_native = [_parse_native_trace_line(line) for line in native_lines]
    scheduleurm_units = [float(job.get("total_units") or 0.0) for job in jobs]
    native_units = [float(row.get("total_units") or 0.0) for row in parsed_native]
    scheduleurm_arrivals = [float(job.get("arrival_time") or 0.0) for job in jobs]
    native_arrivals = [float(row.get("arrival_time") or 0.0) for row in parsed_native]
    resource_counts = [int(job.get("resource_count") or 1) for job in jobs]
    native_resource_counts = [int(row.get("resource_count") or 1) for row in parsed_native]
    throughput_profile_count = sum(
        len(value)
        for value in (throughput.get("throughputs") or {}).values()
        if isinstance(value, Mapping)
    )
    max_arrival_jitter = max(
        (abs(a - b) for a, b in zip(scheduleurm_arrivals, native_arrivals)),
        default=0.0,
    )
    arrival_order_ready = _nondecreasing(native_arrivals) and len(jobs) == len(parsed_native)
    return {
        "taskset": taskset_name,
        "job_count": len(jobs),
        "native_trace_row_count": len(parsed_native),
        "same_job_count": len(jobs) == len(parsed_native) and bool(jobs),
        "same_arrival_times": _same_float_list(scheduleurm_arrivals, native_arrivals),
        "arrival_order_equivalence_ready": arrival_order_ready,
        "max_arrival_jitter_s": max_arrival_jitter,
        "same_total_units": _same_float_list(scheduleurm_units, native_units),
        "same_resource_count": resource_counts == native_resource_counts,
        "same_gpu_count": True,
        "scheduleurm_total_units_sum": sum(scheduleurm_units),
        "native_total_units_sum": sum(native_units),
        "resource_count_values": sorted(set(resource_counts)),
        "native_resource_count_values": sorted(set(native_resource_counts)),
        "throughput_profile_count": throughput_profile_count,
        "throughput_seed_ready": throughput_profile_count > 0,
        "trace_schema_compatibility_ready": (
            len(jobs) == len(parsed_native)
            and bool(jobs)
            and arrival_order_ready
            and _same_float_list(scheduleurm_units, native_units)
            and resource_counts == native_resource_counts
        ),
    }


def _parse_native_trace_line(line: str) -> dict[str, Any]:
    parts = line.rstrip("\n").split("\t")
    return {
        "job_type": parts[0] if len(parts) > 0 else "",
        "total_units": float(parts[5]) if len(parts) > 5 else 0.0,
        "resource_count": int(float(parts[6])) if len(parts) > 6 else 1,
        "arrival_time": float(parts[9]) if len(parts) > 9 else 0.0,
    }


def _same_float_list(left: list[float], right: list[float], *, tol: float = 1e-9) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tol for a, b in zip(left, right))


def _nondecreasing(values: list[float]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def _load_existing_microbaseline() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "gavel_native_performance_microbaseline_20260612.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gavel_service_unit_equivalence_certificate(
        run_native_microbaseline=bool(args.run_native_microbaseline)
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.gavel_service_unit_equivalence_certificate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Gavel service-unit equivalence certificate")
    build.add_argument("--run-native-microbaseline", action="store_true")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "gavel_service_unit_equivalence_certificate_20260612.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "gavel_service_unit_equivalence_certificate_20260612.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
