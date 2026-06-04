"""CLI for trace-driven fast-forward replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .defaults import build_default_cache, calibrated_policy, default_workload_specs, legacy_policy
from .fast_forward import compare_policies
from .tasksets import benchmark_tasksets, taskset_by_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-out", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--trials", type=int, default=101)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-makespan-improvement", type=float, default=1.05)
    parser.add_argument("--min-class-improvement", type=float, default=1.03)
    parser.add_argument("--taskset", default="hybrid_research_portfolio")
    parser.add_argument("--list-tasksets", action="store_true")
    parser.add_argument("--allow-partial-taskset", action="store_true")
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
    specs = taskset.workload_specs()
    if args.taskset == "hybrid_research_portfolio":
        specs = default_workload_specs()
    comparison = compare_policies(
        cache,
        specs,
        baseline=legacy_policy(),
        candidate=calibrated_policy(),
        trials=args.trials,
        seed=args.seed,
    )
    report = comparison.snapshot()
    empirical_classes = ("hybrid_rl_resac_ant", "gpu_heavy_jax_matmul")
    per_workload = comparison.per_workload_improvements
    empirical_pass = all(
        per_workload.get(key, {}).get("makespan_improvement", 0.0) >= float(args.min_class_improvement)
        for key in empirical_classes
    )
    report["pass"] = (
        comparison.makespan_improvement >= float(args.min_makespan_improvement)
        and empirical_pass
    )
    report["min_makespan_improvement"] = float(args.min_makespan_improvement)
    report["min_class_improvement"] = float(args.min_class_improvement)
    report["empirical_classes_checked"] = list(empirical_classes)
    report["cache_workloads"] = cache.available_workloads()
    report["taskset"] = taskset.snapshot(cache)

    if args.cache_out:
        cache.save(Path(args.cache_out))
    if args.report_out:
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
