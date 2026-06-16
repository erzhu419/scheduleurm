"""Node007 ETA matrix for final task-native progress validation.

This runner intentionally bypasses the production scheduler and uses
``remote_workload_selected_profile_probe`` measurement windows.  It validates
ETA/service rows from task-native progress logs, not from tui-top.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .remote_workload_selected_profile_probe import build_remote_workload_selected_profile_probe


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    cwd: str
    template: str
    unit: str
    max_iters: int
    timeout_s: int
    stable_windows: int
    min_rate_samples: int
    stable_cv: float
    stable_rel_delta: float
    stable_skip_samples: int
    dense_profiles: tuple[int, ...]


WORKLOADS: tuple[Workload, ...] = (
    Workload(
        key="cnn_torch_gpu",
        label="Torch CNN fallback/resnet workload",
        cwd="/home/erzhu419/mine_code/scheduleurm",
        template="algorithm/experiments/templates/torch_cnn_resnet50.cmd.tpl",
        unit="step",
        max_iters=40,
        timeout_s=420,
        stable_windows=3,
        min_rate_samples=3,
        stable_cv=0.25,
        stable_rel_delta=0.35,
        stable_skip_samples=0,
        dense_profiles=(2, 3, 4),
    ),
    Workload(
        key="llm_torch_transformer",
        label="Torch Transformer/LLM workload",
        cwd="/home/erzhu419/mine_code/scheduleurm",
        template="algorithm/experiments/templates/torch_llm_distilgpt2.cmd.tpl",
        unit="step",
        max_iters=25,
        timeout_s=480,
        stable_windows=3,
        min_rate_samples=3,
        stable_cv=0.30,
        stable_rel_delta=0.40,
        stable_skip_samples=0,
        dense_profiles=(2, 3, 4),
    ),
    Workload(
        key="rl_resac_ant",
        label="RE-SAC Ant-v2 hybrid RL workload",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        template="algorithm/experiments/templates/resac_ant_real_venv.cmd.tpl",
        unit="iter",
        max_iters=30,
        timeout_s=900,
        stable_windows=3,
        min_rate_samples=5,
        stable_cv=0.20,
        stable_rel_delta=0.25,
        stable_skip_samples=1,
        dense_profiles=(2, 3, 4, 5),
    ),
)


CASES: tuple[dict[str, Any], ...] = (
    {"case": "single_gpu_single_task", "gpus": [0], "profiles": (1,)},
    {"case": "single_gpu_multi_task", "gpus": [0], "profiles": "dense"},
    {"case": "four_gpu_single_task_each", "gpus": [0, 1, 2, 3], "profiles": (1,)},
    {"case": "four_gpu_two_tasks_each", "gpus": [0, 1, 2, 3], "profiles": (2,)},
)


def build_node007_eta_matrix(
    *,
    run_id: str | None = None,
    workloads: set[str] | None = None,
    cases: set[str] | None = None,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> dict[str, Any]:
    run_id = run_id or time.strftime("node007_eta_matrix_%Y%m%d_%H%M%S")
    selected_workloads = [w for w in WORKLOADS if workloads is None or w.key in workloads]
    selected_cases = [c for c in CASES if cases is None or str(c["case"]) in cases]
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for workload in selected_workloads:
        for case in selected_cases:
            profiles = (
                workload.dense_profiles
                if case["profiles"] == "dense"
                else tuple(int(x) for x in case["profiles"])
            )
            child_run_id = f"{run_id}_{workload.key}_{case['case']}"
            try:
                result = build_remote_workload_selected_profile_probe(
                    run_id=child_run_id,
                    node="node007-direct",
                    gpus=[int(x) for x in case["gpus"]],
                    profiles=[int(x) for x in profiles],
                    cwd=workload.cwd,
                    cmd_template_file=str(REPO_ROOT / workload.template),
                    output_root=f"/tmp/scheduleurm_eta_matrix/{child_run_id}",
                    max_iters=workload.max_iters,
                    timeout_s=workload.timeout_s,
                    unit=workload.unit,
                    terminate_on_stable=True,
                    stable_windows=workload.stable_windows,
                    min_rate_samples=workload.min_rate_samples,
                    stable_cv=workload.stable_cv,
                    stable_rel_delta=workload.stable_rel_delta,
                    stable_skip_samples=workload.stable_skip_samples,
                    coordinated_profile_launch=True,
                )
                error = ""
            except Exception as exc:
                result = {
                    "run_id": child_run_id,
                    "pass": False,
                    "summary_paths": [],
                }
                error = repr(exc)
            results.append({
                "workload": workload.key,
                "case": str(case["case"]),
                "result": result,
                "error": error,
            })
            rows.extend(_rows_from_result(workload, case, profiles, result, error))
    report = {
        "gate": "node007_eta_matrix",
        "run_id": run_id,
        "node": "node007-direct",
        "eta_source": "task-native tqdm/progress logs parsed by progress_wrapper; tui-top ETA is not used",
        "workloads": [w.key for w in selected_workloads],
        "cases": [str(c["case"]) for c in selected_cases],
        "rows": rows,
        "results": results,
        "pass": bool(rows) and all(bool(row.get("measurement_valid")) for row in rows),
    }
    out = Path(output or ARTIFACT_ROOT / "node007_eta_matrix_20260613.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = Path(markdown_output or REPO_ROOT / "md" / "node007_eta_matrix_20260613.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown_report(report), encoding="utf-8")
    return report


def _rows_from_result(
    workload: Workload,
    case: Mapping[str, Any],
    profiles: tuple[int, ...],
    result: Mapping[str, Any],
    error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_paths = list(result.get("summary_paths") or [])
    for profile, summary_path in zip(profiles, summary_paths):
        path = Path(str(summary_path))
        if not path.exists():
            rows.append({
                "workload": workload.key,
                "case": str(case["case"]),
                "profile": int(profile),
                "gpus": list(case["gpus"]),
                "measurement_valid": False,
                "summary_path": str(path),
                "error": error or "summary_missing",
            })
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        stable_rates = [float(x) for x in summary.get("stable_rates_unit_s") or []]
        sec_per_unit = [1.0 / x for x in stable_rates if x > 0]
        rows.append({
            "workload": workload.key,
            "label": workload.label,
            "case": str(case["case"]),
            "profile": int(profile),
            "gpus": list(case["gpus"]),
            "gpu_count": len(case["gpus"]),
            "expected_task_count": len(case["gpus"]) * int(profile),
            "running_count": int(summary.get("running_count") or 0),
            "stable_rate_ready_count": int(summary.get("stable_rate_ready_count") or 0),
            "measurement_valid": bool(summary.get("measurement_valid")),
            "aggregate_stable_rate": float(summary.get("aggregate_stable_rate_unit_s") or 0.0),
            "mean_stable_rate": float(summary.get("mean_stable_rate_unit_s") or 0.0),
            "min_seconds_per_unit": min(sec_per_unit) if sec_per_unit else 0.0,
            "max_seconds_per_unit": max(sec_per_unit) if sec_per_unit else 0.0,
            "unit": workload.unit,
            "elapsed_wall_s": float(summary.get("elapsed_wall_s") or 0.0),
            "summary_path": str(path),
            "error": error,
        })
    if not summary_paths:
        rows.append({
            "workload": workload.key,
            "case": str(case["case"]),
            "profile": ",".join(str(x) for x in profiles),
            "gpus": list(case["gpus"]),
            "measurement_valid": False,
            "summary_path": "",
            "error": error or "probe_failed_before_summary",
        })
    return rows


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Node007 ETA Matrix",
        "",
        f"Run: `{report.get('run_id')}`",
        "",
        "ETA source: task-native tqdm/progress logs parsed by `progress_wrapper`; `tui-top` ETA is not used.",
        "",
        "| Workload | Case | GPUs | Profile | Stable tasks | Aggregate rate | Mean rate | Seconds/unit range | Valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{workload}` | `{case}` | {gpu_count} | {profile} | {stable}/{expected} | "
            "{agg:.6g} | {mean:.6g} | {lo:.4g}-{hi:.4g} {unit} | {valid} |".format(
                workload=row.get("workload"),
                case=row.get("case"),
                gpu_count=row.get("gpu_count", len(row.get("gpus") or [])),
                profile=row.get("profile"),
                stable=int(row.get("stable_rate_ready_count") or 0),
                expected=int(row.get("expected_task_count") or 0),
                agg=float(row.get("aggregate_stable_rate") or 0.0),
                mean=float(row.get("mean_stable_rate") or 0.0),
                lo=float(row.get("min_seconds_per_unit") or 0.0),
                hi=float(row.get("max_seconds_per_unit") or 0.0),
                unit=f"s/{row.get('unit')}",
                valid=str(bool(row.get("measurement_valid"))).lower(),
            )
        )
    lines.extend(["", f"Overall pass: `{str(bool(report.get('pass'))).lower()}`", ""])
    return "\n".join(lines)


def _cmd_build(args: argparse.Namespace) -> int:
    workload_filter = (
        {item.strip() for item in str(args.workloads).split(",") if item.strip()}
        if args.workloads
        else None
    )
    case_filter = (
        {item.strip() for item in str(args.cases).split(",") if item.strip()}
        if args.cases
        else None
    )
    report = build_node007_eta_matrix(
        run_id=args.run_id,
        workloads=workload_filter,
        cases=case_filter,
        output=args.output,
        markdown_output=args.markdown_output,
    )
    print(json.dumps({
        "run_id": report.get("run_id"),
        "pass": report.get("pass"),
        "rows": len(report.get("rows") or []),
        "output": args.output,
        "markdown_output": args.markdown_output,
    }, indent=2, sort_keys=True))
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.node007_eta_matrix")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run-id", default=None)
    build.add_argument("--workloads", default="")
    build.add_argument("--cases", default="")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "node007_eta_matrix_20260613.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "node007_eta_matrix_20260613.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
