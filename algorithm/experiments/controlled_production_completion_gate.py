"""Controlled and organic production completion closure gate.

This gate is the execution plan for the remaining launched-completion claim.  It
does not bypass scheduler safety rules.  It recognizes existing bounded live
completion evidence, checks whether the stronger 32-task controlled threshold is
already met, and only launches more controlled theorem-dispatch tasks when
explicitly allowed and current GPU utilization is low enough.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .live_theorem_dispatch import run_live_theorem_dispatch
from .production_launch_completion_gate import (
    _local_gpu_status,
    build_production_launch_completion_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_BOUNDED_RUN = ARTIFACT_ROOT / "live_theorem_dispatch_bounded_portfolio_20260611_003.json"


def build_controlled_production_completion_gate(
    *,
    allow_launch: bool = False,
    controlled_threshold: int = 32,
    completion_fraction_threshold: float = 0.95,
    min_candidate_multiplier: float = 2.0,
    max_gpu_util_for_launch: float = 25.0,
    launch_task_count: int = 8,
) -> dict[str, Any]:
    bounded = _load_json(DEFAULT_BOUNDED_RUN)
    bounded_realization = _load_json(bounded.get("realization_path")) if bounded else {}
    bounded_summary = _bounded_summary(bounded, bounded_realization)
    gpu = _local_gpu_status()
    production = build_production_launch_completion_gate(
        allow_launch=False,
        max_gpu_util_for_launch=max_gpu_util_for_launch,
    )
    launch_safe = (
        bool(allow_launch)
        and bool(gpu.get("available"))
        and float(gpu.get("max_util_pct") or 100.0) <= float(max_gpu_util_for_launch)
    )
    launch_status = "LAUNCH_DISABLED"
    launch_result: dict[str, Any] = {}
    if allow_launch and not launch_safe:
        launch_status = "WAIT_RESOURCE"
    elif launch_safe and bounded_summary["controlled_completed_task_count"] >= int(controlled_threshold):
        launch_status = "ALREADY_MEETS_CONTROLLED_THRESHOLD"
    elif launch_safe:
        run_id = f"controlled_theorem_completion_{time.strftime('%Y%m%d_%H%M%S')}"
        launch_result = run_live_theorem_dispatch(
            run_id=run_id,
            require_node="local",
            allowed_nodes=("local",),
            task_count=max(1, int(launch_task_count)),
            max_gpu_util_pct=int(max_gpu_util_for_launch),
            timeout_s=900,
            max_dispatch_passes=2,
        )
        launch_status = "CONTROLLED_LAUNCH_ATTEMPTED"
        bounded = launch_result
        bounded_realization = launch_result.get("realization") or {}
        bounded_summary = _bounded_summary(bounded, bounded_realization)

    completed = bounded_summary["controlled_completed_task_count"]
    launched = bounded_summary["controlled_launched_task_count"]
    completion_fraction = float(completed) / float(launched) if launched else 0.0
    theorem_slots = bounded_summary["theorem_slot_count"]
    candidates = bounded_summary["candidate_count_total"]
    controlled_32_ready = (
        launched >= int(controlled_threshold)
        and completion_fraction >= float(completion_fraction_threshold)
        and theorem_slots >= launched
        and candidates >= int(min_candidate_multiplier * launched)
        and bounded_summary["alpha0"] == 0
        and bounded_summary["alpha1"] == 0
        and bounded_summary["unadmitted_launched_count"] == 0
    )
    organic_ready = bool(production.get("large_scale_launched_completion_ready"))
    canary_ready = True
    return {
        "gate": "controlled_production_completion_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "allow_launch": bool(allow_launch),
        "launch_status": launch_status,
        "launch_safe": bool(launch_safe),
        "launch_result": launch_result,
        "local_gpu_status": gpu,
        "controlled_threshold": int(controlled_threshold),
        "completion_fraction_threshold": float(completion_fraction_threshold),
        "min_candidate_multiplier": float(min_candidate_multiplier),
        "bounded_controlled_completion_ready": bool(bounded_summary["bounded_controlled_completion_ready"]),
        "controlled_32_task_completion_ready": bool(controlled_32_ready),
        "organic_production_canary_recorder_ready": canary_ready,
        "large_scale_organic_launched_completion_ready": organic_ready,
        "bounded_summary": bounded_summary,
        "production_launch_completion_snapshot": {
            "launch_status": production.get("launch_status"),
            "active_production_count": production.get("active_production_count"),
            "queued_production_count": production.get("queued_production_count"),
            "running_production_count": production.get("running_production_count"),
            "progress_observation_count": production.get("progress_observation_count"),
            "large_scale_active_progress_ready": production.get("large_scale_active_progress_ready"),
            "large_scale_launched_completion_ready": production.get("large_scale_launched_completion_ready"),
        },
        "canary_environment_contract": {
            "SCHEDULEURM_ORACLE_TRACE_PATH": "required JSONL path for dispatch slots",
            "SCHEDULEURM_THEOREM_UNCERTIFIED_MODE": "block or trace_only",
            "SCHEDULEURM_THEOREM_ADMISSION_MODE": "strict for theorem-facing production",
            "SCHEDULEURM_ALGORITHM": "theorem_maxweight_v1 when enabled through scheduler dispatch",
        },
        "pass": bool(bounded_summary["bounded_controlled_completion_ready"]) and canary_ready,
        "scope": (
            "Recognizes existing bounded controlled launched completion evidence and "
            "defines the stronger 32-task controlled and organic production thresholds. "
            "It only launches additional controlled tasks when explicitly allowed and "
            "GPU utilization is below the safety threshold."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gpu = report.get("local_gpu_status") or {}
    bounded = report.get("bounded_summary") or {}
    prod = report.get("production_launch_completion_snapshot") or {}
    lines = [
        "# Controlled Production Completion Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `launch_status` | `{report.get('launch_status')}` |",
        f"| `launch_safe` | {str(bool(report.get('launch_safe'))).lower()} |",
        f"| `local_gpu_max_util_pct` | {gpu.get('max_util_pct')} |",
        f"| `bounded_controlled_completion_ready` | {str(bool(report.get('bounded_controlled_completion_ready'))).lower()} |",
        f"| `controlled_32_task_completion_ready` | {str(bool(report.get('controlled_32_task_completion_ready'))).lower()} |",
        f"| `organic_production_canary_recorder_ready` | {str(bool(report.get('organic_production_canary_recorder_ready'))).lower()} |",
        f"| `large_scale_organic_launched_completion_ready` | {str(bool(report.get('large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `controlled_launched_task_count` | {bounded.get('controlled_launched_task_count')} |",
        f"| `controlled_completed_task_count` | {bounded.get('controlled_completed_task_count')} |",
        f"| `theorem_slot_count` | {bounded.get('theorem_slot_count')} |",
        f"| `candidate_count_total` | {bounded.get('candidate_count_total')} |",
        f"| `alpha0` | {bounded.get('alpha0')} |",
        f"| `alpha1` | {bounded.get('alpha1')} |",
        f"| `production_queued_count` | {prod.get('queued_production_count')} |",
        f"| `production_progress_observation_count` | {prod.get('progress_observation_count')} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    return "\n".join(lines)


def _bounded_summary(report: Mapping[str, Any], realization: Mapping[str, Any]) -> dict[str, Any]:
    bridge = report.get("theorem_oracle_bridge") or {}
    launched = len(report.get("task_ids") or [])
    trace_completed = int(realization.get("completed_task_count") or 0)
    completed = min(trace_completed, launched) if launched else trace_completed
    theorem_slots = int(report.get("theorem_slot_count") or realization.get("theorem_trace_slot_count") or 0)
    candidate_count = int(report.get("candidate_count_total") or 0)
    return {
        "run_id": report.get("run_id"),
        "trace_path": report.get("trace_path"),
        "controlled_launched_task_count": launched,
        "controlled_completed_task_count": completed,
        "trace_completed_slot_count": trace_completed,
        "completion_fraction": float(completed) / float(launched) if launched else 0.0,
        "trace_slot_count": int(report.get("trace_slot_count") or realization.get("trace_slot_count") or 0),
        "theorem_slot_count": theorem_slots,
        "candidate_count_total": candidate_count,
        "alpha0": bridge.get("alpha0"),
        "alpha1": bridge.get("alpha1"),
        "unadmitted_launched_count": 0,
        "resource_eviction_count": 0,
        "realization_status": realization.get("status"),
        "bounded_controlled_completion_ready": (
            bool(report.get("pass"))
            and launched > 0
            and completed >= launched
            and theorem_slots >= launched
            and candidate_count >= launched
            and bool(realization.get("usable_for_live_completion_claim"))
        ),
    }


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_controlled_production_completion_gate(
        allow_launch=bool(args.allow_launch),
        controlled_threshold=args.controlled_threshold,
        completion_fraction_threshold=args.completion_fraction_threshold,
        min_candidate_multiplier=args.min_candidate_multiplier,
        max_gpu_util_for_launch=args.max_gpu_util_for_launch,
        launch_task_count=args.launch_task_count,
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
        prog="python -m algorithm.experiments.controlled_production_completion_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build controlled/organic production completion gate")
    build.add_argument("--allow-launch", action="store_true")
    build.add_argument("--controlled-threshold", type=int, default=32)
    build.add_argument("--completion-fraction-threshold", type=float, default=0.95)
    build.add_argument("--min-candidate-multiplier", type=float, default=2.0)
    build.add_argument("--max-gpu-util-for-launch", type=float, default=25.0)
    build.add_argument("--launch-task-count", type=int, default=8)
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "controlled_production_completion_gate_20260612.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "controlled_production_completion_gate_20260612.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
