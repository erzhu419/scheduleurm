"""Four-quadrant SOTA Pareto gate.

This report separates three claims that are easy to conflate:

* tolerance Pareto dominance on the measured-cache replay gate;
* strict non-inferiority against the SOTA-style envelope;
* strict Pareto dominance, which additionally needs a strict improvement in at
  least one metric in every scoped quadrant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from simulation.service_cache import ServiceRateCache

from .sota_candidate_union_gate import PARETO_TOLERANCE, _cache_for_args, build_sota_candidate_union_gate
from .sota_strict_dominance_frontier import build_sota_strict_dominance_frontier


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


QUADRANT_TASKSETS = {
    "q00_low_cpu_low_gpu": ("q00_light_control",),
    "q01_low_cpu_high_gpu": (
        "q01_gpu_bound_compute",
        "q01_gpu_bound_cnn_resnet50",
        "q01_gpu_bound_llm_inference",
        "q01_gpu_model_portfolio",
    ),
    "q10_high_cpu_low_gpu": ("q10_cpu_host_bound",),
    "q11_high_cpu_high_gpu": ("q11_cpu_gpu_coupled",),
    "hybrid_portfolio": ("hybrid_research_portfolio",),
}


def build_sota_quadrant_pareto_gate(
    *,
    strict_eps: float = 1e-12,
    cache: ServiceRateCache | None = None,
    cache_source: str = "simulation.defaults.build_default_cache",
) -> dict[str, Any]:
    gate = build_sota_candidate_union_gate(cache=cache, cache_source=cache_source)
    frontier = build_sota_strict_dominance_frontier(
        strict_eps=strict_eps,
        cache=cache,
        cache_source=cache_source,
    )
    rows = [row for row in gate.get("scenarios") or [] if row.get("replayable", True)]
    frontier_keys = {
        (str(row.get("taskset") or ""), str(row.get("arrival_mode") or ""))
        for row in frontier.get("frontier") or []
    }
    quadrants = []
    for quadrant, tasksets in QUADRANT_TASKSETS.items():
        scoped = [row for row in rows if row.get("taskset") in set(tasksets)]
        quadrants.append(_quadrant_row(quadrant, scoped, frontier_keys, strict_eps))
    tolerance_ready = all(bool(row["tolerance_pareto_ready"]) for row in quadrants)
    strict_noninferiority_ready = all(bool(row["strict_noninferiority_ready"]) for row in quadrants)
    strict_dominance_ready = all(bool(row["strict_pareto_dominance_ready"]) for row in quadrants)
    if strict_dominance_ready:
        status = "SOTA_QUADRANT_STRICT_DOMINANCE_PASS"
    elif strict_noninferiority_ready:
        status = "SOTA_QUADRANT_STRICT_NONINFERIOR_PASS"
    elif tolerance_ready:
        status = "SOTA_QUADRANT_TOLERANCE_PASS"
    else:
        status = "SOTA_QUADRANT_TOLERANCE_OPEN"
    return {
        "gate": "sota_quadrant_pareto_gate",
        "status": status,
        "pass": bool(tolerance_ready),
        "tolerance_pareto_ready": bool(tolerance_ready),
        "strict_noninferiority_ready": bool(strict_noninferiority_ready),
        "strict_pareto_dominance_ready": bool(strict_dominance_ready),
        "pareto_tolerance": float(PARETO_TOLERANCE),
        "strict_eps": float(strict_eps),
        "quadrants": quadrants,
        "strict_frontier_count": int(frontier.get("frontier_count") or 0),
        "cache_source": str(cache_source),
        "cache_record_count": gate.get("cache_record_count"),
        "scope": (
            "Four-quadrant measured-cache policy-semantics gate.  Strict "
            "dominance is reported separately from the 0.5% replay-tolerance "
            "Pareto claim."
        ),
    }


def _quadrant_row(
    quadrant: str,
    rows: list[Mapping[str, Any]],
    frontier_keys: set[tuple[str, str]],
    strict_eps: float,
) -> dict[str, Any]:
    ratios = [
        (
            str(row.get("taskset") or ""),
            str(row.get("arrival_mode") or ""),
            _candidate_vs_sota_best_makespan(row),
            _candidate_vs_sota_best_mean_flow(row),
        )
        for row in rows
    ]
    worst_makespan = min((item[2] for item in ratios), default=0.0)
    worst_flow = min((item[3] for item in ratios), default=0.0)
    best_makespan = max((item[2] for item in ratios), default=0.0)
    best_flow = max((item[3] for item in ratios), default=0.0)
    open_rows = [
        {
            "taskset": taskset,
            "arrival_mode": arrival,
            "makespan_ratio": makespan,
            "mean_flow_ratio": flow,
        }
        for taskset, arrival, makespan, flow in ratios
        if (taskset, arrival) in frontier_keys
    ]
    tolerance_ready = (
        bool(ratios)
        and worst_makespan >= 1.0 - PARETO_TOLERANCE
        and worst_flow >= 1.0 - PARETO_TOLERANCE
    )
    strict_noninferiority = (
        bool(ratios)
        and worst_makespan >= 1.0 - strict_eps
        and worst_flow >= 1.0 - strict_eps
    )
    strict_improvement = best_makespan > 1.0 + strict_eps or best_flow > 1.0 + strict_eps
    return {
        "quadrant": quadrant,
        "scenario_count": len(ratios),
        "worst_makespan_ratio": worst_makespan,
        "worst_mean_flow_ratio": worst_flow,
        "best_makespan_ratio": best_makespan,
        "best_mean_flow_ratio": best_flow,
        "tolerance_pareto_ready": bool(tolerance_ready),
        "strict_noninferiority_ready": bool(strict_noninferiority),
        "strict_improvement_present": bool(strict_improvement),
        "strict_pareto_dominance_ready": bool(strict_noninferiority and strict_improvement),
        "open_strict_rows": open_rows,
    }


def _candidate_vs_sota_best_makespan(row: Mapping[str, Any]) -> float:
    candidate = float(row.get("candidate_makespan_s") or 0.0)
    best = float(row.get("best_sota_makespan_s") or 0.0)
    return best / candidate if candidate > 0.0 else 0.0


def _candidate_vs_sota_best_mean_flow(row: Mapping[str, Any]) -> float:
    candidate = float(row.get("candidate_mean_flow_s") or 0.0)
    best = float(row.get("best_sota_mean_flow_s") or 0.0)
    return best / candidate if candidate > 0.0 else 0.0


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# SOTA Four-Quadrant Pareto Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `tolerance_pareto_ready` | {str(bool(report.get('tolerance_pareto_ready'))).lower()} |",
        f"| `strict_noninferiority_ready` | {str(bool(report.get('strict_noninferiority_ready'))).lower()} |",
        f"| `strict_pareto_dominance_ready` | {str(bool(report.get('strict_pareto_dominance_ready'))).lower()} |",
        f"| `pareto_tolerance` | {float(report.get('pareto_tolerance') or 0.0):.6g} |",
        f"| `strict_frontier_count` | {int(report.get('strict_frontier_count') or 0)} |",
        "",
        "## Quadrants",
        "",
        "| Quadrant | Scenarios | Worst makespan ratio | Worst flow ratio | Strict noninferior | Strict improvement | Strict dominate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("quadrants") or []:
        lines.append(
            "| `{quadrant}` | {count} | {ms:.9g} | {flow:.9g} | {noninf} | {improve} | {dom} |".format(
                quadrant=row.get("quadrant"),
                count=int(row.get("scenario_count") or 0),
                ms=float(row.get("worst_makespan_ratio") or 0.0),
                flow=float(row.get("worst_mean_flow_ratio") or 0.0),
                noninf=str(bool(row.get("strict_noninferiority_ready"))).lower(),
                improve=str(bool(row.get("strict_improvement_present"))).lower(),
                dom=str(bool(row.get("strict_pareto_dominance_ready"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",), nargs="?", default="build")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "sota_quadrant_pareto_gate_20260613.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "md" / "sota_quadrant_pareto_gate_20260613.md",
    )
    parser.add_argument(
        "--live-cache-json",
        default="",
        help="Optional service-cache v2 snapshot to overlay on top of the default cache.",
    )
    args = parser.parse_args()
    cache, cache_source = _cache_for_args(args)
    report = build_sota_quadrant_pareto_gate(cache=cache, cache_source=cache_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
