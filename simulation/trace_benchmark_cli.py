"""CLI for explicit task-list benchmark replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .defaults import build_default_cache
from .tasksets import benchmark_tasksets, taskset_by_name
from .trace_benchmark import (
    build_task_trace,
    load_trace,
    replay_trace_matrix,
    replay_trace_suite,
    save_trace,
)


DEFAULT_MATRIX_TASKSETS = (
    "q00_light_control",
    "q01_gpu_bound_compute",
    "q10_cpu_host_bound",
    "q11_cpu_gpu_coupled",
    "hybrid_research_portfolio",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", default="hybrid_research_portfolio")
    parser.add_argument("--trace-in", default="")
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--arrival-mode", choices=("static", "poisson"), default="static")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-seed", type=int, default=7)
    parser.add_argument("--list-tasksets", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--matrix-tasksets", default=",".join(DEFAULT_MATRIX_TASKSETS))
    args = parser.parse_args()

    if args.list_tasksets:
        print(json.dumps({"tasksets": sorted(benchmark_tasksets())}, indent=2, sort_keys=True))
        return 0

    cache = build_default_cache()
    if args.matrix:
        names = [name.strip() for name in args.matrix_tasksets.split(",") if name.strip()]
        missing_by_taskset = {}
        for name in names:
            missing = taskset_by_name(name).missing_measurements(cache)
            if missing:
                missing_by_taskset[name] = missing
        if missing_by_taskset:
            report = {
                "pass": False,
                "error": "one or more matrix tasksets have missing measurement obligations",
                "missing_by_taskset": missing_by_taskset,
            }
            _write_report(args.report_out, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2
        report = replay_trace_matrix(
            cache,
            names,
            arrival_mode=args.arrival_mode,
            trace_seed=args.seed,
            replay_seed=args.replay_seed,
        )
        report["pass"] = _matrix_passes(report)
        _write_report(args.report_out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["pass"] else 1

    taskset = taskset_by_name(args.taskset)
    missing = taskset.missing_measurements(cache)
    if missing:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "taskset has missing measurement obligations",
                    "taskset": taskset.snapshot(cache),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    trace = load_trace(Path(args.trace_in)) if args.trace_in else build_task_trace(
        taskset,
        arrival_mode=args.arrival_mode,
        seed=args.seed,
    )
    report = replay_trace_suite(cache, trace, seed=args.replay_seed)
    report["taskset"] = taskset.snapshot(cache)
    report["pass"] = _passes(report)

    if args.trace_out:
        save_trace(trace, Path(args.trace_out))
    _write_report(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


def _write_report(path_str: str, report: dict) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _passes(report: dict) -> bool:
    candidate = next(
        (row for row in report["results"] if str(row["policy"]).startswith("calibrated_")),
        None,
    )
    legacy_rel = report.get("relative_to_legacy", {})
    if candidate is None or candidate["policy"] not in legacy_rel:
        return False
    rel = legacy_rel[candidate["policy"]]
    sota_ok = bool(
        report.get("sota_tasklist_comparison", {}).get(
            "candidate_not_pareto_dominated",
            False,
        )
    )
    return (
        rel["makespan_improvement"] >= 1.0
        and rel["mean_flow_improvement"] >= 1.0
        and sota_ok
    )


def _matrix_passes(report: dict) -> bool:
    aggregate = report.get("aggregate_by_policy", [])
    legacy = next((row for row in aggregate if row["policy"] == "legacy_fixed_caps"), None)
    if legacy is None:
        return False
    legacy_ok = (
        legacy["candidate_vs_policy_sum_makespan"] >= 1.0
        and legacy["candidate_vs_policy_job_weighted_mean_flow"] >= 1.0
    )
    return legacy_ok and bool(report.get("aggregate_candidate_not_pareto_dominated"))


if __name__ == "__main__":
    raise SystemExit(main())
