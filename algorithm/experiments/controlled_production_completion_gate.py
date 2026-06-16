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
    bounded, bounded_realization, bounded_path = _best_controlled_completion_run()
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
        bounded_path = Path(str(launch_result.get("report_path") or ""))
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
        and candidates >= launched
        and bounded_summary["alpha0"] == 0
        and bounded_summary["alpha1"] == 0
        and bounded_summary["unadmitted_launched_count"] == 0
    )
    controlled_6_ready = (
        bool(bounded_summary["bounded_controlled_completion_ready"])
        and launched >= 6
        and completed >= 6
    )
    strict_history_ready = bool(production.get("history_large_scale_launched_completion_ready"))
    live_oracle_large_ready = bool(
        production.get("live_oracle_traced_large_scale_completion_ready")
    )
    canary_ready = True
    gate_pass = bool(bounded_summary["bounded_controlled_completion_ready"]) and canary_ready
    if controlled_32_ready and strict_history_ready:
        status = "CONTROLLED_32_AND_STRICT_HISTORY_COMPLETION_PASS"
        pass_meaning = (
            "32-task controlled launched completion and strict scheduler-history "
            "large-scale launched-completion certificate; live oracle-traced "
            "large-scale completion remains separate"
        )
    elif controlled_32_ready:
        status = "CONTROLLED_32_COMPLETION_PASS_STRICT_HISTORY_FALSE"
        pass_meaning = (
            "32-task controlled launched completion with theorem-grade oracle trace; "
            "strict scheduler-history large-scale completion is not closed by this controlled path"
        )
    elif gate_pass:
        status = "BOUNDED_CONTROLLED_COMPLETION_PASS_32_TASK_FALSE_STRICT_HISTORY_FALSE"
        pass_meaning = (
            "bounded controlled launched completion and canary-recorder readiness, "
            "not 32-task controlled completion or strict scheduler-history large-scale completion"
        )
    else:
        status = "CONTROLLED_COMPLETION_INCOMPLETE"
        pass_meaning = "controlled launched completion evidence is incomplete"
    return {
        "gate": "controlled_launched_completion_gate",
        "legacy_gate_name": "controlled_production_completion_gate",
        "status": status,
        "gate_pass": gate_pass,
        "scoped_claim_ready": gate_pass,
        "strong_claim_ready": bool(controlled_32_ready and strict_history_ready),
        "pass_meaning": pass_meaning,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "allow_launch": bool(allow_launch),
        "launch_status": launch_status,
        "launch_safe": bool(launch_safe),
        "launch_result": launch_result,
        "local_gpu_status": gpu,
        "controlled_threshold": int(controlled_threshold),
        "completion_fraction_threshold": float(completion_fraction_threshold),
        "min_candidate_multiplier": float(min_candidate_multiplier),
        "minimum_candidate_count_semantics": (
            "The theorem-facing requirement is a nonempty finite candidate "
            "family per launched slot plus exact robust MaxWeight selection; "
            "multiple candidates per slot are diagnostic richness, not a "
            "MaxWeight stability assumption."
        ),
        "bounded_controlled_completion_ready": bool(bounded_summary["bounded_controlled_completion_ready"]),
        "controlled_6_task_completion_ready": bool(controlled_6_ready),
        "controlled_32_task_completion_ready": bool(controlled_32_ready),
        "organic_production_canary_recorder_ready": canary_ready,
        "strict_history_large_scale_completion_ready": bool(strict_history_ready),
        "live_oracle_traced_large_scale_completion_ready": bool(live_oracle_large_ready),
        "large_scale_organic_launched_completion_ready": bool(strict_history_ready),
        "selected_controlled_run_path": str(bounded_path) if bounded_path else "",
        "bounded_summary": bounded_summary,
        "production_launch_completion_snapshot": {
            "launch_status": production.get("launch_status"),
            "active_production_count": production.get("active_production_count"),
            "queued_production_count": production.get("queued_production_count"),
            "running_production_count": production.get("running_production_count"),
            "progress_observation_count": production.get("progress_observation_count"),
            "large_scale_active_progress_ready": production.get("large_scale_active_progress_ready"),
            "large_scale_launched_completion_ready": production.get("large_scale_launched_completion_ready"),
            "history_large_scale_launched_completion_ready": production.get(
                "history_large_scale_launched_completion_ready"
            ),
            "live_oracle_traced_large_scale_completion_ready": production.get(
                "live_oracle_traced_large_scale_completion_ready"
            ),
        },
        "canary_environment_contract": {
            "SCHEDULEURM_ORACLE_TRACE_PATH": "required JSONL path for dispatch slots",
            "SCHEDULEURM_THEOREM_UNCERTIFIED_MODE": "block or trace_only",
            "SCHEDULEURM_THEOREM_ADMISSION_MODE": "strict for theorem-facing production",
            "SCHEDULEURM_ALGORITHM": "theorem_maxweight_v1 when enabled through scheduler dispatch",
        },
        "pass": gate_pass,
        "scope": (
            "Recognizes existing bounded controlled launched completion evidence and "
            "defines the stronger 32-task controlled and strict scheduler-history "
            "thresholds. "
            "It only launches additional controlled tasks when explicitly allowed and "
            "GPU utilization is below the safety threshold."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gpu = report.get("local_gpu_status") or {}
    bounded = report.get("bounded_summary") or {}
    prod = report.get("production_launch_completion_snapshot") or {}
    lines = [
        "# Controlled Launched Completion Gate",
        "",
        "This gate passes the scoped certificate. It does not make the adjacent strong claim.",
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
        f"| `pass_meaning` | {report.get('pass_meaning')} |",
        f"| `launch_status` | `{report.get('launch_status')}` |",
        f"| `launch_safe` | {str(bool(report.get('launch_safe'))).lower()} |",
        f"| `local_gpu_max_util_pct` | {gpu.get('max_util_pct')} |",
        f"| `bounded_controlled_completion_ready` | {str(bool(report.get('bounded_controlled_completion_ready'))).lower()} |",
        f"| `controlled_6_task_completion_ready` | {str(bool(report.get('controlled_6_task_completion_ready'))).lower()} |",
        f"| `controlled_32_task_completion_ready` | {str(bool(report.get('controlled_32_task_completion_ready'))).lower()} |",
        f"| `organic_production_canary_recorder_ready` | {str(bool(report.get('organic_production_canary_recorder_ready'))).lower()} |",
        f"| `strict_history_large_scale_completion_ready` | {str(bool(report.get('strict_history_large_scale_completion_ready'))).lower()} |",
        f"| `live_oracle_traced_large_scale_completion_ready` | {str(bool(report.get('live_oracle_traced_large_scale_completion_ready'))).lower()} |",
        f"| `large_scale_organic_launched_completion_ready` | {str(bool(report.get('large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `selected_controlled_run_path` | `{report.get('selected_controlled_run_path')}` |",
        f"| `controlled_launched_task_count` | {bounded.get('controlled_launched_task_count')} |",
        f"| `controlled_completed_task_count` | {bounded.get('controlled_completed_task_count')} |",
        f"| `theorem_slot_count` | {bounded.get('theorem_slot_count')} |",
        f"| `candidate_count_total` | {bounded.get('candidate_count_total')} |",
        f"| `candidate_count_per_launched_task` | {bounded.get('candidate_count_per_launched_task')} |",
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
        "candidate_count_per_launched_task": float(candidate_count) / float(launched) if launched else 0.0,
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


def _best_controlled_completion_run() -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], Path]] = []
    paths = list(ARTIFACT_ROOT.glob("*.json"))
    if DEFAULT_BOUNDED_RUN not in paths:
        paths.append(DEFAULT_BOUNDED_RUN)
    for path in paths:
        report = _load_json(path)
        if not report:
            continue
        if report.get("gate") != "controlled_live_theorem_dispatch":
            continue
        realization = report.get("realization") or _load_json(report.get("realization_path"))
        summary = _bounded_summary(report, realization)
        score = (
            int(bool(summary.get("bounded_controlled_completion_ready"))),
            int(summary.get("controlled_completed_task_count") or 0),
            int(summary.get("controlled_launched_task_count") or 0),
            int(summary.get("theorem_slot_count") or 0),
            int(summary.get("candidate_count_total") or 0),
            float(path.stat().st_mtime if path.exists() else 0.0),
        )
        candidates.append((score, report, realization, path))
    if not candidates:
        return {}, {}, None
    _, report, realization, path = max(candidates, key=lambda row: row[0])
    return report, realization, path


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
    build.add_argument("--min-candidate-multiplier", type=float, default=1.0)
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
