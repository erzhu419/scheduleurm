"""CLI for trace-driven fast-forward replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .defaults import build_default_cache, calibrated_policy, default_workload_specs, legacy_policy
from .fast_forward import compare_policies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-out", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--trials", type=int, default=101)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-makespan-improvement", type=float, default=1.05)
    parser.add_argument("--min-class-improvement", type=float, default=1.03)
    args = parser.parse_args()

    cache = build_default_cache()
    comparison = compare_policies(
        cache,
        default_workload_specs(),
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
