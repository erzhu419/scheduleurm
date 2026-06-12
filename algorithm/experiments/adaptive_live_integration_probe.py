"""Probe optional adaptive theorem policy integration on live state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.oracle_trace import load_trace_slots

from .production_queued_theorem_trace import (
    generate_production_queued_theorem_trace,
    markdown_report as production_trace_markdown_report,
)
from .natural_live_theorem_trace import (
    generate_natural_live_theorem_trace,
    markdown_report as natural_trace_markdown_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_adaptive_live_integration_probe(
    *,
    trace_path: str | Path = ARTIFACT_ROOT / "adaptive_live_integration_probe.jsonl",
    output_path: str | Path | None = None,
    max_tasks: int = 64,
    max_tasks_per_gpu: int = 8,
) -> dict[str, Any]:
    trace_report = generate_production_queued_theorem_trace(
        trace_path=trace_path,
        output_path=None,
        max_tasks=max_tasks,
        max_tasks_per_gpu=max_tasks_per_gpu,
        algorithm="adaptive_theorem_maxweight_v1",
        append=False,
    )
    trace_origin = "queued_production"
    if int(trace_report.get("trace_slot_count") or 0) == 0:
        trace_report = generate_natural_live_theorem_trace(
            trace_path=trace_path,
            output_path=None,
            task_count=max(1, min(8, int(max_tasks))),
            workloads=("hybrid_rl_resac_ant", "gpu_heavy_jax_matmul"),
            max_tasks_per_gpu=max_tasks_per_gpu,
            algorithm="adaptive_theorem_maxweight_v1",
            append=False,
        )
        trace_origin = "synthetic_measured_workload_fallback"
    slots = load_trace_slots(trace_path)
    selected = [_selected_candidate(slot) for slot in slots]
    sampler_rows = [
        ((row.get("algorithm_audit") or {}).get("adaptive_sampler") or {})
        for row in selected if row
    ]
    detector_rows = [
        ((row.get("algorithm_audit") or {}).get("regime_detector") or {})
        for row in selected if row
    ]
    forced_count = sum(1 for row in sampler_rows if row.get("forced_exploration"))
    bucket_counts = {}
    for row in sampler_rows:
        bucket = str(row.get("candidate_bucket") or "")
        if bucket:
            bucket_counts[bucket] = int(row.get("bucket_count_after") or bucket_counts.get(bucket, 0))
    report = {
        "gate": "adaptive_live_integration_probe",
        "trace_path": str(trace_path),
        "trace_origin": trace_origin,
        "trace_report": trace_report,
        "slot_count": len(slots),
        "selected_count": len([row for row in selected if row]),
        "adaptive_sampler_rows": len([row for row in sampler_rows if row]),
        "regime_detector_rows": len([row for row in detector_rows if row]),
        "forced_exploration_count": forced_count,
        "bucket_counts": bucket_counts,
        "base_oracle_usable": bool((trace_report.get("oracle_bridge") or {}).get("usable_for_theorem")),
        "adaptive_fields_present": (
            bool(slots)
            and len(sampler_rows) == len(slots)
            and len(detector_rows) == len(slots)
        ),
        "live_scheduler_default_changed": False,
        "pass": (
            bool((trace_report.get("oracle_bridge") or {}).get("usable_for_theorem"))
            and bool(slots)
            and len(sampler_rows) == len(slots)
            and len(detector_rows) == len(slots)
        ),
        "scope": (
            "optional adaptive_theorem_maxweight_v1 live-state probe. It uses queued production tasks when present "
            "and otherwise falls back to measured synthetic q01/q11 live-node probes. The queue is not mutated, "
            "no process is launched, and the default scheduler policy is unchanged."
        ),
    }
    if output_path:
        _write_json(output_path, report)
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Adaptive Live Integration Probe",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `slot_count` | {report.get('slot_count', 0)} |",
        f"| `trace_origin` | `{report.get('trace_origin', '')}` |",
        f"| `selected_count` | {report.get('selected_count', 0)} |",
        f"| `adaptive_sampler_rows` | {report.get('adaptive_sampler_rows', 0)} |",
        f"| `regime_detector_rows` | {report.get('regime_detector_rows', 0)} |",
        f"| `forced_exploration_count` | {report.get('forced_exploration_count', 0)} |",
        f"| `base_oracle_usable` | {str(bool(report.get('base_oracle_usable'))).lower()} |",
        f"| `live_scheduler_default_changed` | {str(bool(report.get('live_scheduler_default_changed'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Underlying Production Trace",
        "",
        (
            production_trace_markdown_report(report.get("trace_report") or {})
            if report.get("trace_origin") == "queued_production"
            else natural_trace_markdown_report(report.get("trace_report") or {})
        ),
    ]
    return "\n".join(lines)


def _selected_candidate(slot: Mapping[str, Any]) -> Mapping[str, Any]:
    for row in slot.get("candidates") or []:
        if row.get("selected"):
            return row
    return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_probe(args: argparse.Namespace) -> int:
    report = build_adaptive_live_integration_probe(
        trace_path=args.trace_path,
        output_path=args.output,
        max_tasks=args.max_tasks,
        max_tasks_per_gpu=args.max_tasks_per_gpu,
    )
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.adaptive_live_integration_probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe", help="Probe optional adaptive theorem policy on live state")
    probe.add_argument("--trace-path", default=str(ARTIFACT_ROOT / "adaptive_live_integration_probe_20260612.jsonl"))
    probe.add_argument("--output", default=str(ARTIFACT_ROOT / "adaptive_live_integration_probe_20260612.json"))
    probe.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "adaptive_live_integration_probe_20260612.md"))
    probe.add_argument("--max-tasks", type=int, default=64)
    probe.add_argument("--max-tasks-per-gpu", type=int, default=8)
    probe.set_defaults(func=_cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
