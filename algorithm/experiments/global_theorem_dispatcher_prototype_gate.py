"""Gate for the bounded global theorem-dispatch prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.theorem_dispatch.global_dispatch import select_global_action


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_global_theorem_dispatcher_prototype_gate() -> dict[str, Any]:
    queue = {"gpu_heavy_jax_matmul": 5.0, "hybrid_rl_resac_ant": 3.0}
    candidates = [
        {
            "task_id": "q01-0",
            "action_id": "node=a|gpu=0",
            "resource_id": "a:0",
            "lower_service": {"gpu_heavy_jax_matmul": 2.0},
            "penalty_units": 0.0,
            "score_semantics": "robust_maxweight_lower_service",
            "theorem_ready": True,
        },
        {
            "task_id": "q11-0",
            "action_id": "node=a|gpu=1",
            "resource_id": "a:1",
            "lower_service": {"hybrid_rl_resac_ant": 1.5},
            "penalty_units": 0.0,
            "score_semantics": "robust_maxweight_lower_service",
            "theorem_ready": True,
        },
        {
            "task_id": "fallback-0",
            "action_id": "node=cpu|legacy",
            "resource_id": "cpu",
            "lower_service": {},
            "penalty_units": 0.0,
            "score_semantics": "scheduler_sort_key_minimization",
            "theorem_ready": False,
        },
    ]
    result = select_global_action(candidates, queue, max_batch_size=2)
    snapshot = result.snapshot()
    ready = (
        snapshot["global_action_dispatch_ready"]
        and snapshot["candidate_configuration_count"] >= 2
        and snapshot["oracle_gap_alpha0"] == 0.0
        and snapshot["oracle_gap_alpha1"] == 0.0
        and snapshot["fallback_legacy_rows_excluded"]
    )
    return {
        "gate": "global_theorem_dispatcher_prototype_gate",
        "status": "GLOBAL_THEOREM_DISPATCHER_PROTOTYPE_PASS_NOT_LIVE_DEFAULT",
        "gate_pass": ready,
        "scoped_claim_ready": ready,
        "strong_claim_ready": False,
        "pass_meaning": (
            "pure bounded global action selector prototype with exact oracle gap "
            "over the enumerated family, not deployed live scheduler default"
        ),
        "queue_vector": queue,
        "candidate_rows": candidates,
        "result": snapshot,
        "global_action_dispatch_ready": ready,
        "live_scheduler_default_global_dispatcher_ready": False,
        "pass": ready,
        "scope": (
            "Prototype-only theorem dispatcher.  It enumerates bounded global "
            "configurations from theorem-ready rows and excludes legacy fallback rows."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    result = report.get("result") or {}
    lines = [
        "# Global Theorem Dispatcher Prototype Gate",
        "",
        "This gate passes the scoped prototype certificate. It does not claim the live scheduler uses global batch dispatch by default.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `global_action_dispatch_ready` | {str(bool(report.get('global_action_dispatch_ready'))).lower()} |",
        f"| `live_scheduler_default_global_dispatcher_ready` | {str(bool(report.get('live_scheduler_default_global_dispatcher_ready'))).lower()} |",
        f"| `candidate_configuration_count` | {result.get('candidate_configuration_count')} |",
        f"| `oracle_gap_alpha0` | {result.get('oracle_gap_alpha0')} |",
        f"| `oracle_gap_alpha1` | {result.get('oracle_gap_alpha1')} |",
        f"| `score_semantics` | `{result.get('score_semantics')}` |",
        f"| `fallback_legacy_rows_excluded` | {str(bool(result.get('fallback_legacy_rows_excluded'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    return "\n".join(lines)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_global_theorem_dispatcher_prototype_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.global_theorem_dispatcher_prototype_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build global theorem dispatcher prototype gate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "global_theorem_dispatcher_prototype_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "global_theorem_dispatcher_prototype_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
