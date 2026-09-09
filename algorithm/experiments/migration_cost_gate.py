"""Controlled migration cost gate.

The gate validates the action semantics used by theorem dispatch.  It can be
extended with live checkpoint/resume probes, but the default report is a
deterministic certificate over measured-style rows and never migrates user jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.migration import (
    MigrationCost,
    build_keep_action_row,
    build_migration_action_row,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_migration_cost_gate() -> dict[str, Any]:
    tasks = [
        _task("cnn_bench", "pure_gpu", checkpoint=True),
        _task("resac_halfcheetah", "hybrid_rl", checkpoint=True),
        _task("freqduet_cpu_surrogate", "pure_cpu", checkpoint=True),
        _task("user_running_job", "production", checkpoint=True, controlled=False),
        _task("missing_checkpoint", "pure_gpu", checkpoint=False),
    ]
    rows: list[dict[str, Any]] = []
    points = (0.25, 0.50, 0.75)
    for task in tasks:
        for point in points:
            remaining = 100.0 * (1.0 - point)
            cheap = MigrationCost(
                checkpoint_flush_s=2.0,
                sync_s=3.0,
                environment_staging_s=1.0,
                resume_warmup_s=2.0,
                lost_work_s=1.0,
                risk_penalty_units=1.0,
            )
            expensive = MigrationCost(
                checkpoint_flush_s=40.0,
                sync_s=35.0,
                environment_staging_s=20.0,
                resume_warmup_s=20.0,
                lost_work_s=10.0,
                risk_penalty_units=20.0,
            )
            rows.append(build_migration_action_row(
                task=task,
                workload_key=str(task["workload_key"]),
                from_node="node003",
                to_node="jtl311linux",
                current_rate=1.0,
                target_lower_service=1.8,
                remaining_work=remaining,
                progress_fraction=point,
                cost=cheap,
            ))
            rows.append(build_migration_action_row(
                task={**task, "id": f"{task['id']}_expensive"},
                workload_key=str(task["workload_key"]),
                from_node="node003",
                to_node="jtl311linux",
                current_rate=1.0,
                target_lower_service=1.8,
                remaining_work=remaining,
                progress_fraction=point,
                cost=expensive,
            ))
    keep = build_keep_action_row(
        task={"id": "keep_ref"},
        workload_key="hybrid_rl_resac_ant",
        current_node="node003",
        current_lower_service=1.0,
    )
    selected = select_global_action(
        rows + [keep],
        {
            "gpu_cnn_torch_resnet50": 50.0,
            "hybrid_rl_resac_halfcheetah": 50.0,
            "freqduet_cpu_surrogate": 50.0,
            "hybrid_rl_resac_ant": 1.0,
        },
        max_batch_size=3,
    )
    allowed_rows = [row for row in rows if row.get("theorem_ready")]
    blocked_rows = [row for row in rows if not row.get("theorem_ready")]
    return {
        "gate": "migration_cost_gate",
        "pass": bool(allowed_rows) and bool(blocked_rows),
        "status": "MIGRATION_ACTION_MODEL_PASS" if allowed_rows and blocked_rows else "MIGRATION_ACTION_MODEL_OPEN",
        "migration_points": list(points),
        "row_count": len(rows),
        "allowed_count": len(allowed_rows),
        "blocked_count": len(blocked_rows),
        "selected_action": selected.snapshot(),
        "rows": rows,
        "no_touch_safety_ready": all(
            row.get("migration_block_reason") in {
                "not_controlled_benchmark",
                "ordinary_running_task_not_touchable",
                "checkpoint_missing",
                "resume_missing",
                "migration_cost_not_recovered",
            }
            for row in blocked_rows
        ),
        "scope": "Controlled benchmark migration rows only; ordinary production running tasks are excluded.",
    }


def _task(task_id: str, family: str, *, checkpoint: bool, controlled: bool = True) -> dict[str, Any]:
    key = {
        "pure_gpu": "gpu_cnn_torch_resnet50",
        "hybrid_rl": "hybrid_rl_resac_halfcheetah",
        "pure_cpu": "freqduet_cpu_surrogate",
        "production": "hybrid_rl_resac_ant",
    }.get(family, family)
    return {
        "id": task_id,
        "family": family,
        "workload_key": key,
        "controlled_benchmark": controlled,
        "production_running": not controlled,
        "checkpoint_verified": checkpoint,
        "resume_verified": checkpoint,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Migration Cost Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Allowed rows: `{report.get('allowed_count')}`",
        f"- Blocked rows: `{report.get('blocked_count')}`",
        f"- No-touch safety: `{str(bool(report.get('no_touch_safety_ready'))).lower()}`",
        "",
        "| Action | Progress | Ready | Reason | Penalty |",
        "|---|---:|---:|---|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            f"| `{row.get('action_id')}` | {float(row.get('progress_fraction') or 0.0):.2f} | "
            f"{str(bool(row.get('theorem_ready'))).lower()} | `{row.get('migration_block_reason') or ''}` | "
            f"{float(row.get('penalty_units') or 0.0):.3f} |"
        )
    lines.extend(["", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "migration_cost_gate_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "migration_cost_gate_20260629.md")
    args = parser.parse_args()
    report = build_migration_cost_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
