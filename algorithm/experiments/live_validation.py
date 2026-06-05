"""Replay-to-live validation for small end-to-end sanity runs.

This module compares an explicit task-list replay report against observed live
job completion times.  It is intentionally small: service-curve calibration
does not require long production jobs to finish, but a paper artifact should
still include a few live sanity runs showing that replayed JCT/makespan is not
detached from real execution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def predicted_metrics(
    replay_report: Mapping[str, Any],
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    results = list(replay_report.get("results") or [])
    if policy is None:
        row = next((r for r in results if str(r.get("policy", "")).startswith("calibrated_")), None)
    else:
        row = next((r for r in results if r.get("policy") == policy), None)
    if row is None:
        raise ValueError(f"policy {policy or 'calibrated_*'} not found in replay report")
    return {
        "policy": row["policy"],
        "makespan_s": _as_float(row.get("makespan_s")),
        "mean_flow_s": _as_float(row.get("mean_flow_s")),
        "p90_flow_s": _as_float(row.get("p90_flow_s")),
        "completed_jobs": int(row.get("completed_jobs") or 0),
        "profiles": row.get("profiles") or {},
    }


def observed_metrics(live_report: Mapping[str, Any]) -> dict[str, Any]:
    jobs = list(live_report.get("jobs") or live_report.get("observations") or [])
    flows = []
    completions = []
    censored = 0
    for job in jobs:
        if bool(job.get("censored")):
            censored += 1
            continue
        if job.get("flow_s") is not None:
            flow = _as_float(job.get("flow_s"))
        else:
            arrival = _as_float(job.get("arrival_s", job.get("start_s")))
            completion = _as_float(job.get("completion_s", job.get("end_s")))
            flow = max(0.0, completion - arrival)
        completion_time = _as_float(job.get("completion_s", job.get("end_s", flow)))
        flows.append(flow)
        completions.append(completion_time)
    return {
        "run_id": live_report.get("run_id", ""),
        "makespan_s": max(completions) if completions else 0.0,
        "mean_flow_s": mean(flows) if flows else 0.0,
        "p90_flow_s": _quantile(flows, 0.90),
        "completed_jobs": len(flows),
        "censored_jobs": censored,
    }


def compare_replay_to_live(
    replay_report: Mapping[str, Any],
    live_report: Mapping[str, Any],
    *,
    policy: str | None = None,
    max_relative_error: float = 0.20,
) -> dict[str, Any]:
    pred = predicted_metrics(replay_report, policy=policy)
    obs = observed_metrics(live_report)
    rows = []
    for metric in ("makespan_s", "mean_flow_s", "p90_flow_s"):
        p = _as_float(pred.get(metric))
        o = _as_float(obs.get(metric))
        rows.append(
            {
                "metric": metric,
                "predicted": p,
                "observed": o,
                "absolute_error": abs(o - p),
                "relative_error": abs(o - p) / p if p > 0 else None,
            }
        )
    completed_match = pred["completed_jobs"] == obs["completed_jobs"]
    errors_ok = all(
        row["relative_error"] is not None and row["relative_error"] <= max_relative_error
        for row in rows
    )
    return {
        "policy": pred["policy"],
        "profiles": pred["profiles"],
        "predicted": pred,
        "observed": obs,
        "metric_errors": rows,
        "max_relative_error": max_relative_error,
        "completed_jobs_match": completed_match,
        "usable_for_live_sanity": completed_match
        and obs["censored_jobs"] == 0
        and errors_ok,
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _cmd_compare(args: argparse.Namespace) -> int:
    out = compare_replay_to_live(
        _json(args.replay_report),
        _json(args.live_report),
        policy=args.policy or None,
        max_relative_error=args.max_relative_error,
    )
    _write_json(args.output, out)
    print(args.output)
    return 0 if out.get("usable_for_live_sanity") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.live_validation")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("compare", help="Compare replay report against live job completions")
    s.add_argument("--replay-report", required=True)
    s.add_argument("--live-report", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--policy", default="")
    s.add_argument("--max-relative-error", type=float, default=0.20)
    s.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
