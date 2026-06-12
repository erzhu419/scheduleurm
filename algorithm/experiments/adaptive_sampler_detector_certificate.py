"""Concrete sampler and change-detector probability certificate.

The base learning/regime certificate proves the deterministic event-level
inputs.  This module adds a concrete probability model that can feed the Lean
high-probability lifting theorems without changing the live scheduler:

* active buckets use deterministic round-robin forced exploration, so coverage
  is probability-one conditional on the active bucket set;
* regime changes use a bounded two-window mean-shift detector with explicit
  Hoeffding false-alarm and missed-detection bounds.

The report is a model certificate.  It is not a claim that the current
production scheduler has already deployed the sampler or detector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from algorithm.adaptive_learning import (
    TwoWindowMeanShiftDetector,
    detector_false_alarm_bound,
    detector_miss_bound,
    sampler_hoeffding_radius,
)

from .learning_regime_certificate import build_learning_regime_certificate
from .or_submission_closure import DEFAULT_LOADS, DEFAULT_SEEDS, DEFAULT_TASKSETS


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_adaptive_sampler_detector_certificate(
    *,
    taskset_names: Sequence[str] = DEFAULT_TASKSETS,
    loads: Sequence[float] = DEFAULT_LOADS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    modes: Sequence[str] = ("poisson", "bursty"),
    total_failure_budget: float = 0.05,
    min_samples_per_bucket: int = 128,
    detector_window: int = 512,
    detector_threshold: float = 0.20,
    detector_min_shift: float = 0.40,
) -> dict[str, Any]:
    base = build_learning_regime_certificate(
        taskset_names=taskset_names,
        loads=loads,
        seeds=seeds,
        modes=modes,
        total_failure_budget=total_failure_budget,
    )
    sampler = _sampler_certificate(
        base,
        min_samples_per_bucket=min_samples_per_bucket,
        failure_budget=0.5 * float(total_failure_budget),
    )
    detector = _detector_certificate(
        base,
        window_size=detector_window,
        threshold=detector_threshold,
        min_shift=detector_min_shift,
        failure_budget=0.5 * float(total_failure_budget),
    )
    model_ready = (
        bool(base.get("deterministic_event_closed"))
        and bool(base.get("event_level_union_bound_closed"))
        and bool(sampler.get("sampler_probability_model_certified"))
        and bool(detector.get("change_point_detector_tail_certified"))
    )
    sampler_ready = bool(sampler.get("sampler_probability_model_certified"))
    detector_ready = bool(detector.get("change_point_detector_tail_certified"))
    return {
        "gate": "adaptive_sampler_detector_probability_certificate",
        "total_failure_budget": float(total_failure_budget),
        "base_gate": {
            "scenario_count": base.get("scenario_count"),
            "active_bucket_count": base.get("active_bucket_count"),
            "deterministic_event_closed": base.get("deterministic_event_closed"),
            "event_level_union_bound_closed": base.get("event_level_union_bound_closed"),
            "hidden_regime": base.get("hidden_regime"),
        },
        "sampler": sampler,
        "detector": detector,
        "sampler_probability_model_certified": sampler_ready,
        "change_point_detector_tail_certified": detector_ready,
        "concrete_probability_model_ready": model_ready,
        "live_scheduler_integrated": False,
        "pass": model_ready,
        "scope": (
            "concrete probability certificate for a deployable sampler/detector "
            "model. It closes the stochastic inputs conditionally for this model, "
            "but does not claim the current production scheduler has already "
            "enabled the sampler or change detector."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    base = report.get("base_gate") or {}
    sampler = report.get("sampler") or {}
    detector = report.get("detector") or {}
    lines = [
        "# Adaptive Sampler / Detector Probability Certificate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `concrete_probability_model_ready` | {str(bool(report.get('concrete_probability_model_ready'))).lower()} |",
        f"| `live_scheduler_integrated` | {str(bool(report.get('live_scheduler_integrated'))).lower()} |",
        f"| `active_bucket_count` | {base.get('active_bucket_count', 0)} |",
        f"| `scenario_count` | {base.get('scenario_count', 0)} |",
        f"| `sampler_certified` | {str(bool(sampler.get('sampler_probability_model_certified'))).lower()} |",
        f"| `detector_certified` | {str(bool(detector.get('change_point_detector_tail_certified'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Sampler",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `model` | `{sampler.get('model')}` |",
        f"| `total_decisions` | {sampler.get('total_decisions', 0)} |",
        f"| `active_bucket_count` | {sampler.get('active_bucket_count', 0)} |",
        f"| `min_forced_samples_per_bucket` | {sampler.get('min_forced_samples_per_bucket', 0)} |",
        f"| `required_min_samples_per_bucket` | {sampler.get('required_min_samples_per_bucket', 0)} |",
        f"| `coverage_probability_lower_bound` | {_fmt(sampler.get('coverage_probability_lower_bound'))} |",
        f"| `bounded_unit_radius_at_forced_min` | {_fmt(sampler.get('bounded_unit_radius_at_forced_min'))} |",
        "",
        "## Detector",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `model` | `{detector.get('model')}` |",
        f"| `window_size` | {detector.get('window_size', 0)} |",
        f"| `threshold` | {_fmt(detector.get('threshold'))} |",
        f"| `min_shift` | {_fmt(detector.get('min_shift'))} |",
        f"| `switching_count` | {detector.get('switching_count', 0)} |",
        f"| `per_change_miss_bound` | {_fmt(detector.get('per_change_miss_bound'))} |",
        f"| `per_window_false_alarm_bound` | {_fmt(detector.get('per_window_false_alarm_bound'))} |",
        f"| `union_tail_bound` | {_fmt(detector.get('union_tail_bound'))} |",
        f"| `detection_delay_bound_decisions` | {detector.get('detection_delay_bound_decisions', 0)} |",
        "",
    ]
    return "\n".join(lines)


def _sampler_certificate(
    base: Mapping[str, Any],
    *,
    min_samples_per_bucket: int,
    failure_budget: float,
) -> dict[str, Any]:
    rows = list(base.get("active_bucket_rows") or [])
    bucket_count = max(1, len(rows))
    total_decisions = sum(max(0, int(row.get("observed_decisions") or 0)) for row in rows)
    forced_min = total_decisions // bucket_count
    per_bucket_delta = max(1e-12, float(failure_budget) / bucket_count)
    radius = sampler_hoeffding_radius(
        sample_count=forced_min,
        active_bucket_count=bucket_count,
        failure_budget=failure_budget,
    )
    return {
        "model": "deterministic_round_robin_forced_active_bucket_sampler",
        "active_bucket_count": bucket_count,
        "total_decisions": total_decisions,
        "min_forced_samples_per_bucket": forced_min,
        "required_min_samples_per_bucket": max(1, int(min_samples_per_bucket)),
        "coverage_probability_lower_bound": 1.0 if forced_min >= min_samples_per_bucket else 0.0,
        "allocated_failure_budget": float(failure_budget),
        "per_bucket_failure_budget": per_bucket_delta,
        "bounded_unit_radius_at_forced_min": radius,
        "sampler_probability_model_certified": (
            bool(rows)
            and forced_min >= max(1, int(min_samples_per_bucket))
            and radius < 1.0
        ),
        "claim_scope": (
            "probability-one coverage under a deterministic forced exploration "
            "schedule over the finite active bucket set"
        ),
    }


def _detector_certificate(
    base: Mapping[str, Any],
    *,
    window_size: int,
    threshold: float,
    min_shift: float,
    failure_budget: float,
) -> dict[str, Any]:
    hidden = base.get("hidden_regime") or {}
    switches = max(0, int(hidden.get("switching_count_if_concatenated") or 0))
    m = max(1, int(window_size))
    tau = max(0.0, float(threshold))
    gap = max(0.0, float(min_shift))
    detector = TwoWindowMeanShiftDetector(
        window_size=m,
        threshold=tau,
        min_shift=gap,
    )
    out = detector.tail_certificate(
        switching_count=switches,
        failure_budget=failure_budget,
    )
    out.update({
        "claim_scope": (
            "conditional detector tail bound for bounded normalized feedback "
            "with a declared minimum regime mean shift"
        ),
        "per_change_miss_bound": detector_miss_bound(
            window_size=m,
            threshold=tau,
            min_shift=gap,
        ),
        "per_window_false_alarm_bound": detector_false_alarm_bound(
            window_size=m,
            threshold=tau,
        ),
    })
    return out


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return "NA"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_adaptive_sampler_detector_certificate(
        taskset_names=[x.strip() for x in args.taskset if x.strip()],
        loads=tuple(float(x) for x in args.load),
        seeds=tuple(int(x) for x in args.seed),
        modes=tuple(x.strip() for x in args.mode if x.strip()),
        total_failure_budget=args.total_failure_budget,
        min_samples_per_bucket=args.min_samples_per_bucket,
        detector_window=args.detector_window,
        detector_threshold=args.detector_threshold,
        detector_min_shift=args.detector_min_shift,
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
        prog="python -m algorithm.experiments.adaptive_sampler_detector_certificate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build concrete sampler/detector probability certificate")
    build.add_argument("--taskset", action="append", default=list(DEFAULT_TASKSETS))
    build.add_argument("--load", action="append", type=float, default=list(DEFAULT_LOADS))
    build.add_argument("--seed", action="append", type=int, default=list(DEFAULT_SEEDS))
    build.add_argument("--mode", action="append", default=["poisson", "bursty"])
    build.add_argument("--total-failure-budget", type=float, default=0.05)
    build.add_argument("--min-samples-per-bucket", type=int, default=128)
    build.add_argument("--detector-window", type=int, default=512)
    build.add_argument("--detector-threshold", type=float, default=0.20)
    build.add_argument("--detector-min-shift", type=float, default=0.40)
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "adaptive_sampler_detector_certificate.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "adaptive_sampler_detector_certificate.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
