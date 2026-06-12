"""Active-bucket learning and hidden-regime certificate audit.

The Lean artifact proves deterministic/high-probability spines once the active
bucket event and hidden-regime dwell/switching event are supplied.  This module
binds those abstract inputs to the measured Scheduleurm replay layer:

* active buckets are finite tuples of taskset, arrival mode/load, workload, and
  selected measured profile;
* the union-bound accounting is explicit over only observed active buckets;
* regime dwell/switching budgets are deterministic replay facts.

It does not claim that the live scheduler has enabled adaptive sampling or
change-point detection.  Concrete deployable probability models are reported by
``adaptive_sampler_detector_certificate.py``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulation.defaults import build_default_cache

from .or_submission_closure import (
    DEFAULT_LOADS,
    DEFAULT_SEEDS,
    DEFAULT_TASKSETS,
    build_rate_controlled_trace,
    replay_trace_suite_with_backlog,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_learning_regime_certificate(
    *,
    taskset_names: Sequence[str] = DEFAULT_TASKSETS,
    loads: Sequence[float] = DEFAULT_LOADS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    modes: Sequence[str] = ("poisson", "bursty"),
    total_failure_budget: float = 0.05,
) -> dict[str, Any]:
    cache = build_default_cache()
    scenarios: list[dict[str, Any]] = []
    bucket_counter: Counter[str] = Counter()
    bucket_payloads: dict[str, dict[str, Any]] = {}
    for taskset_name in taskset_names:
        for mode in modes:
            for load in loads:
                for seed in seeds:
                    trace = build_rate_controlled_trace(
                        cache,
                        taskset_name,
                        arrival_mode=mode,
                        load_factor=float(load),
                        seed=int(seed),
                    )
                    suite = replay_trace_suite_with_backlog(cache, trace, seed=int(seed) + 1000)
                    scenario = _scenario_certificate(trace.snapshot(), suite, load_factor=float(load))
                    scenarios.append(scenario)
                    for bucket in scenario["active_buckets"]:
                        key = str(bucket["bucket_key"])
                        bucket_counter[key] += int(bucket.get("observed_decisions") or 0)
                        bucket_payloads.setdefault(key, dict(bucket))
    bucket_rows = _active_bucket_rows(bucket_counter, bucket_payloads, total_failure_budget)
    regime = _hidden_regime_certificate(scenarios)
    deterministic_event_closed = bool(bucket_rows) and regime["deterministic_dwell_switching_event_closed"]
    sampler_probability_certified = False
    detector_tail_certified = False
    return {
        "gate": "active_bucket_hidden_regime_certificate",
        "tasksets": list(taskset_names),
        "loads": [float(x) for x in loads],
        "seeds": [int(x) for x in seeds],
        "modes": list(modes),
        "total_failure_budget": float(total_failure_budget),
        "scenario_count": len(scenarios),
        "active_bucket_count": len(bucket_rows),
        "active_bucket_rows": bucket_rows,
        "hidden_regime": regime,
        "deterministic_event_closed": deterministic_event_closed,
        "event_level_union_bound_closed": (
            bool(bucket_rows)
            and sum(float(row["allocated_failure_budget"]) for row in bucket_rows)
            <= float(total_failure_budget) + 1e-12
        ),
        "sampler_probability_model_certified": sampler_probability_certified,
        "feedback_probability_model_certified": sampler_probability_certified,
        "change_point_detector_tail_certified": detector_tail_certified,
        "main_claim_ready": deterministic_event_closed
        and sampler_probability_certified
        and detector_tail_certified,
        "pass": deterministic_event_closed,
        "scope": (
            "event-level active-bucket and hidden-regime certificate for measured "
            "Scheduleurm replay traces.  It supports extension text and theorem-input "
            "auditing.  This base gate alone is not a live adaptive-sampling "
            "probability theorem; concrete sampler/detector model evidence is "
            "reported by the separate adaptive_sampler_detector_certificate gate."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    regime = report.get("hidden_regime") or {}
    lines = [
        "# Active-Bucket / Hidden-Regime Certificate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `scenario_count` | {report.get('scenario_count', 0)} |",
        f"| `active_bucket_count` | {report.get('active_bucket_count', 0)} |",
        f"| `event_level_union_bound_closed` | {str(bool(report.get('event_level_union_bound_closed'))).lower()} |",
        f"| `deterministic_dwell_switching_event_closed` | {str(bool(regime.get('deterministic_dwell_switching_event_closed'))).lower()} |",
        f"| `sampler_probability_model_certified` | {str(bool(report.get('sampler_probability_model_certified'))).lower()} |",
        f"| `change_point_detector_tail_certified` | {str(bool(report.get('change_point_detector_tail_certified'))).lower()} |",
        f"| `main_claim_ready` | {str(bool(report.get('main_claim_ready'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Hidden-Regime Dwell/Switching",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `regime_count` | {regime.get('regime_count', 0)} |",
        f"| `switching_count_if_concatenated` | {regime.get('switching_count_if_concatenated', 0)} |",
        f"| `min_dwell_s` | {_fmt(regime.get('min_dwell_s'))} |",
        f"| `total_dwell_s` | {_fmt(regime.get('total_dwell_s'))} |",
        "",
        "## Active Buckets",
        "",
        "| Bucket | Decisions | Allocated delta | Hoeffding radius |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("active_bucket_rows") or []:
        lines.append(
            "| `{bucket}` | {n} | {delta:.9f} | {radius:.9f} |".format(
                bucket=row.get("bucket_key"),
                n=row.get("observed_decisions", 0),
                delta=float(row.get("allocated_failure_budget") or 0.0),
                radius=float(row.get("bounded_unit_hoeffding_radius") or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _scenario_certificate(
    trace_snapshot: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    load_factor: float,
) -> dict[str, Any]:
    candidate = _candidate_result(suite.get("results") or [])
    buckets = _active_buckets_from_candidate(
        trace_snapshot=trace_snapshot,
        candidate=candidate,
        load_factor=load_factor,
    )
    return {
        "trace": trace_snapshot.get("name"),
        "taskset": trace_snapshot.get("taskset_name"),
        "arrival_mode": trace_snapshot.get("arrival_mode"),
        "seed": trace_snapshot.get("seed"),
        "load_factor": float(load_factor),
        "job_count": trace_snapshot.get("job_count", 0),
        "candidate_policy": candidate.get("policy"),
        "candidate_makespan_s": candidate.get("makespan_s"),
        "candidate_mean_flow_s": candidate.get("mean_flow_s"),
        "active_buckets": buckets,
    }


def _active_buckets_from_candidate(
    *,
    trace_snapshot: Mapping[str, Any],
    candidate: Mapping[str, Any],
    load_factor: float,
) -> list[dict[str, Any]]:
    policy_config = candidate.get("policy_config") or {}
    adaptive_counts = policy_config.get("adaptive_profile_counts") or {}
    if adaptive_counts:
        rows = []
        for workload, counts in sorted(adaptive_counts.items()):
            for profile, count in sorted((counts or {}).items(), key=lambda row: int(row[0])):
                rows.append(_bucket_payload(
                    trace_snapshot,
                    load_factor=load_factor,
                    workload_key=str(workload),
                    profile=int(profile),
                    observed_decisions=int(count),
                ))
        return rows
    profiles = candidate.get("profiles") or {}
    workload_job_counts = Counter(
        str(job.get("workload_key") or "")
        for job in trace_snapshot.get("jobs") or []
    )
    return [
        _bucket_payload(
            trace_snapshot,
            load_factor=load_factor,
            workload_key=str(workload),
            profile=int(profile),
            observed_decisions=int(workload_job_counts.get(str(workload), 0)),
        )
        for workload, profile in sorted(profiles.items())
    ]


def _bucket_payload(
    trace_snapshot: Mapping[str, Any],
    *,
    load_factor: float,
    workload_key: str,
    profile: int,
    observed_decisions: int,
) -> dict[str, Any]:
    arrival_mode = str(trace_snapshot.get("arrival_mode") or "")
    taskset = str(trace_snapshot.get("taskset_name") or "")
    key = f"{taskset}|{arrival_mode}|load={load_factor:.2f}|{workload_key}|profile={profile}"
    return {
        "bucket_key": key,
        "taskset": taskset,
        "arrival_mode": arrival_mode,
        "load_factor": float(load_factor),
        "workload_key": workload_key,
        "profile": int(profile),
        "observed_decisions": max(0, int(observed_decisions)),
    }


def _active_bucket_rows(
    counter: Counter[str],
    payloads: Mapping[str, Mapping[str, Any]],
    total_failure_budget: float,
) -> list[dict[str, Any]]:
    count = max(1, len(counter))
    per_bucket_delta = max(1e-12, float(total_failure_budget) / count)
    rows = []
    for key, observed in sorted(counter.items()):
        n = max(1, int(observed))
        radius = math.sqrt(math.log(2.0 / per_bucket_delta) / (2.0 * n))
        row = dict(payloads.get(key) or {})
        row.update({
            "observed_decisions": int(observed),
            "allocated_failure_budget": per_bucket_delta,
            "bounded_unit_hoeffding_radius": radius,
        })
        rows.append(row)
    return rows


def _hidden_regime_certificate(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        scenarios,
        key=lambda row: (
            str(row.get("taskset") or ""),
            str(row.get("arrival_mode") or ""),
            float(row.get("load_factor") or 0.0),
            int(row.get("seed") or 0),
        ),
    )
    dwell_rows = []
    prev_regime = None
    switching = 0
    for row in ordered:
        regime = _regime_key(row)
        if prev_regime is not None and regime != prev_regime:
            switching += 1
        prev_regime = regime
        dwell = max(0.0, float(row.get("candidate_makespan_s") or 0.0))
        dwell_rows.append({
            "trace": row.get("trace"),
            "regime_key": regime,
            "dwell_s": dwell,
        })
    dwells = [float(row["dwell_s"]) for row in dwell_rows if float(row["dwell_s"]) > 0.0]
    return {
        "regime_count": len({row["regime_key"] for row in dwell_rows}),
        "scenario_count": len(dwell_rows),
        "switching_count_if_concatenated": switching,
        "min_dwell_s": min(dwells) if dwells else 0.0,
        "total_dwell_s": sum(dwells),
        "deterministic_dwell_switching_event_closed": bool(dwell_rows) and bool(dwells),
        "dwell_rows": dwell_rows[:200],
        "claim_scope": (
            "deterministic replay dwell/switching budget; not a stochastic "
            "hidden-Markov or BOCD detection-delay tail bound"
        ),
    }


def _regime_key(row: Mapping[str, Any]) -> str:
    mode = str(row.get("arrival_mode") or "")
    taskset = str(row.get("taskset") or "")
    return f"{taskset}|{mode}"


def _candidate_result(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    for row in results:
        if str(row.get("policy") or "").startswith("calibrated_adaptive"):
            return dict(row)
    for row in results:
        if str(row.get("policy") or "").startswith("calibrated_"):
            return dict(row)
    return {}


def _fmt(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{out:.9f}" if math.isfinite(out) else "NA"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_learning_regime_certificate(
        taskset_names=[x.strip() for x in args.taskset if x.strip()],
        loads=tuple(float(x) for x in args.load),
        seeds=tuple(int(x) for x in args.seed),
        modes=tuple(x.strip() for x in args.mode if x.strip()),
        total_failure_budget=args.total_failure_budget,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.learning_regime_certificate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build active-bucket and hidden-regime certificate")
    build.add_argument("--taskset", action="append", default=list(DEFAULT_TASKSETS))
    build.add_argument("--load", action="append", type=float, default=list(DEFAULT_LOADS))
    build.add_argument("--seed", action="append", type=int, default=list(DEFAULT_SEEDS))
    build.add_argument("--mode", action="append", default=["poisson", "bursty"])
    build.add_argument("--total-failure-budget", type=float, default=0.05)
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "active_bucket_hidden_regime_certificate.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "active_bucket_hidden_regime_certificate.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
