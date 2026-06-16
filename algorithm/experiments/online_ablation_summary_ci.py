"""Reviewer-facing interval summaries for online replay and ablations.

This script does not run new scheduling experiments.  It consumes the existing
online-arrival and ablation gate artifacts and emits the distributional columns
reviewers asked for: median, worst case, 5/95 percentiles, dominated-scenario
counts, and candidate loss counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_online_ablation_summary_ci(
    *,
    online_path: str | Path = ARTIFACT_ROOT / "or_gate_online_arrivals.json",
    ablation_path: str | Path = ARTIFACT_ROOT / "or_gate_ablation_suite.json",
    pareto_tolerance: float | None = None,
) -> dict[str, Any]:
    online = _load_json(online_path)
    ablation = _load_json(ablation_path)
    tolerance = (
        float(pareto_tolerance)
        if pareto_tolerance is not None
        else float(online.get("pareto_tolerance") or 0.005)
    )
    online_rows = list(online.get("scenarios") or [])
    ablation_rows = list(ablation.get("rows") or [])
    online_summary = {
        "candidate_vs_legacy_makespan": _summary(
            row.get("candidate_vs_legacy_makespan") for row in online_rows
        ),
        "candidate_vs_legacy_mean_flow": _summary(
            row.get("candidate_vs_legacy_mean_flow") for row in online_rows
        ),
        "candidate_mean_backlog_jobs": _summary(
            row.get("candidate_mean_backlog_jobs") for row in online_rows
        ),
        "candidate_max_backlog_jobs": _summary(
            row.get("candidate_max_backlog_jobs") for row in online_rows
        ),
    }
    online_dominated = [
        _scenario_id(row) for row in online_rows
        if not bool(row.get("candidate_not_pareto_dominated_by_sota"))
    ]
    loss_rows = _online_loss_rows(online_rows, tolerance)
    ablation_dominated = [
        _scenario_id(row) for row in ablation_rows
        if row.get("full_candidate_pareto_dominated_by_ablation")
    ]
    return {
        "gate": "online_ablation_summary_ci",
        "status": "ONLINE_ABLATION_DISTRIBUTIONAL_SUMMARY_PASS",
        "gate_pass": bool(online_rows) and bool(ablation_rows),
        "scoped_claim_ready": bool(online_rows) and bool(ablation_rows),
        "strong_claim_ready": False,
        "pass_meaning": (
            "distributional summary of existing replay and ablation artifacts, "
            "not new live execution or stochastic generalization"
        ),
        "online_source": str(online_path),
        "ablation_source": str(ablation_path),
        "pareto_tolerance": tolerance,
        "online_scenario_count": len(online_rows),
        "ablation_scenario_count": len(ablation_rows),
        "online_summary": online_summary,
        "online_candidate_dominated_scenario_count": len(online_dominated),
        "online_candidate_dominated_scenarios": online_dominated[:200],
        "candidate_loss_gt_tolerance_count": len(loss_rows),
        "candidate_loss_gt_tolerance_rows": loss_rows[:200],
        "ablation_candidate_dominated_scenario_count": len(ablation_dominated),
        "ablation_candidate_dominated_scenarios": ablation_dominated[:200],
        "pass": bool(online_rows) and bool(ablation_rows),
        "scope": (
            "Summarizes policy-semantics replay on the same measured service cache. "
            "Rows are scenario-level replay summaries, not direct external binary runs."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Online Replay and Ablation Summary Intervals",
        "",
        "This gate passes the scoped certificate. It does not make the adjacent strong claim.",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `online_scenario_count` | {report.get('online_scenario_count', 0)} |",
        f"| `ablation_scenario_count` | {report.get('ablation_scenario_count', 0)} |",
        f"| `online_candidate_dominated_scenario_count` | {report.get('online_candidate_dominated_scenario_count', 0)} |",
        f"| `candidate_loss_gt_tolerance_count` | {report.get('candidate_loss_gt_tolerance_count', 0)} |",
        f"| `ablation_candidate_dominated_scenario_count` | {report.get('ablation_candidate_dominated_scenario_count', 0)} |",
        "",
        "## Online Distribution",
        "",
        "| Metric | Geomean | Median | Worst | 5% | 95% | Min | Max | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, summary in (report.get("online_summary") or {}).items():
        lines.append(
            "| `{metric}` | {geomean:.6g} | {median:.6g} | {worst:.6g} | {p05:.6g} | {p95:.6g} | {minv:.6g} | {maxv:.6g} | {n} |".format(
                metric=metric,
                geomean=float(summary.get("geomean") or 0.0),
                median=float(summary.get("median") or 0.0),
                worst=float(summary.get("worst") or 0.0),
                p05=float(summary.get("p05") or 0.0),
                p95=float(summary.get("p95") or 0.0),
                minv=float(summary.get("min") or 0.0),
                maxv=float(summary.get("max") or 0.0),
                n=int(summary.get("n") or 0),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def write_loss_csv(path: str | Path, report: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(report.get("candidate_loss_gt_tolerance_rows") or [])
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "trace",
                "taskset",
                "arrival_mode",
                "seed",
                "load_factor",
                "policy",
                "metric",
                "relative_loss",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def _online_loss_rows(rows: Iterable[Mapping[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        candidate = _policy_row(row, str(row.get("candidate_policy") or ""))
        if not candidate:
            continue
        cand_makespan = float(candidate.get("makespan_s") or 0.0)
        cand_flow = float(candidate.get("mean_flow_s") or 0.0)
        for policy in row.get("results") or []:
            if policy is candidate:
                continue
            name = str(policy.get("policy") or "")
            makespan = float(policy.get("makespan_s") or 0.0)
            flow = float(policy.get("mean_flow_s") or 0.0)
            if makespan > 0 and cand_makespan > makespan * (1.0 + tolerance):
                out.append(_loss_row(row, name, "makespan", cand_makespan / makespan - 1.0))
            if flow > 0 and cand_flow > flow * (1.0 + tolerance):
                out.append(_loss_row(row, name, "mean_flow", cand_flow / flow - 1.0))
    return out


def _loss_row(row: Mapping[str, Any], policy: str, metric: str, loss: float) -> dict[str, Any]:
    return {
        "trace": row.get("trace"),
        "taskset": row.get("taskset"),
        "arrival_mode": row.get("arrival_mode"),
        "seed": row.get("seed"),
        "load_factor": row.get("load_factor"),
        "policy": policy,
        "metric": metric,
        "relative_loss": float(loss),
    }


def _policy_row(row: Mapping[str, Any], policy_name: str) -> Mapping[str, Any]:
    for payload in row.get("results") or []:
        if str(payload.get("policy") or "") == policy_name:
            return payload
    return {}


def _scenario_id(row: Mapping[str, Any]) -> str:
    return str(row.get("trace") or f"{row.get('taskset')}:{row.get('arrival_mode')}:{row.get('seed')}")


def _summary(values: Iterable[Any]) -> dict[str, Any]:
    xs = sorted(float(x) for x in values if _finite_positive_or_zero(x))
    if not xs:
        return {"n": 0}
    positives = [x for x in xs if x > 0.0]
    geomean = math.exp(sum(math.log(x) for x in positives) / len(positives)) if positives else 0.0
    return {
        "n": len(xs),
        "geomean": geomean,
        "median": _quantile(xs, 0.5),
        "worst": min(xs),
        "p05": _quantile(xs, 0.05),
        "p95": _quantile(xs, 0.95),
        "min": min(xs),
        "max": max(xs),
    }


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _finite_positive_or_zero(value: Any) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and x >= 0.0


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_online_ablation_summary_ci(
        online_path=args.online_path,
        ablation_path=args.ablation_path,
        pareto_tolerance=args.pareto_tolerance,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    if args.loss_csv_output:
        write_loss_csv(args.loss_csv_output, report)
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.online_ablation_summary_ci"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Summarize online replay and ablation intervals")
    build.add_argument("--online-path", default=str(ARTIFACT_ROOT / "or_gate_online_arrivals.json"))
    build.add_argument("--ablation-path", default=str(ARTIFACT_ROOT / "or_gate_ablation_suite.json"))
    build.add_argument("--pareto-tolerance", type=float, default=None)
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "online_ablation_summary_ci_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "online_ablation_summary_ci_20260612.md"))
    build.add_argument("--loss-csv-output", default=str(ARTIFACT_ROOT / "ablation_pareto_by_scenario_20260612.csv"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
