"""CLI for SOTA-style replay baseline comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .defaults import build_default_cache
from .sota_baselines import compare_against_sota_suite
from .tasksets import benchmark_tasksets, taskset_by_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", default="hybrid_research_portfolio")
    parser.add_argument("--trials", type=int, default=101)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report-out", default="")
    parser.add_argument("--allow-partial-taskset", action="store_true")
    parser.add_argument("--list-tasksets", action="store_true")
    args = parser.parse_args()

    if args.list_tasksets:
        print(json.dumps({"tasksets": sorted(benchmark_tasksets())}, indent=2, sort_keys=True))
        return 0

    cache = build_default_cache()
    taskset = taskset_by_name(args.taskset)
    missing = taskset.missing_measurements(cache)
    if missing and not args.allow_partial_taskset:
        report = {
            "pass": False,
            "error": "taskset has missing measurement obligations; run real probes first or pass --allow-partial-taskset for exploratory replay",
            "taskset": taskset.snapshot(cache),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    report = compare_against_sota_suite(
        cache,
        taskset.workload_specs(),
        trials=args.trials,
        seed=args.seed,
    )
    report["taskset"] = taskset.snapshot(cache)
    report["pass"] = report["candidate_not_pareto_dominated"]
    if args.report_out:
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
