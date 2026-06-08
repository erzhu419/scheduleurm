"""Compose live-slice service rates into a task-list live sanity report.

The real probes stop after stable progress windows, while the paper-facing
validation needs all-job JCT and makespan.  This module bridges those two
surfaces by replaying the exact task list with service rates read from live
summary files.  It deliberately reuses the trace replay event model instead of
inventing a separate completion proxy.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache, records_from_summary_file
from simulation.tasksets import TaskSetMember, taskset_by_name
from simulation.trace_benchmark import (
    _assign_to_resources,
    _replay_resource_jobs,
    build_task_trace,
)

from .live_validation import compare_replay_to_live, predicted_metrics


def build_composed_live_report(
    replay_report: Mapping[str, Any],
    *,
    taskset_name: str,
    source_paths: Mapping[str, Iterable[str | Path]],
    policy: str | None = None,
    arrival_mode: str = "static",
    trace_seed: int = 42,
    replay_seed: int = 7,
    run_id: str = "composed_live_proxy",
) -> dict[str, Any]:
    """Return a live-style job report using measured live-slice summaries."""

    pred = predicted_metrics(replay_report, policy=policy)
    taskset = taskset_by_name(taskset_name)
    members = {member.workload_key: member for member in taskset.members}
    cache, source_records = _cache_from_sources(members, source_paths)
    trace = build_task_trace(taskset, arrival_mode=arrival_mode, seed=trace_seed)
    completions: dict[str, float] = {}
    grouped: dict[str, list[Any]] = {}
    rng = random.Random(replay_seed)

    profiles = {str(key): int(value) for key, value in pred["profiles"].items()}
    for job in trace.jobs:
        grouped.setdefault(job.workload_key, []).append(job)
    for key, jobs in grouped.items():
        if key not in profiles:
            raise KeyError(f"policy {pred['policy']!r} has no profile for workload {key!r}")
        target_profile = int(profiles[key])
        member = members[key]
        _require_profile(cache, key, target_profile)
        for resource_jobs in _assign_to_resources(jobs, member.resource_count):
            completions.update(
                _replay_resource_jobs(
                    cache,
                    workload_key=key,
                    jobs=resource_jobs,
                    target_profile=target_profile,
                    rng=rng,
                )
            )

    job_rows = []
    for job in trace.jobs:
        completion = completions[job.job_id]
        job_rows.append(
            {
                "job_id": job.job_id,
                "workload_key": job.workload_key,
                "arrival_s": job.arrival_s,
                "completion_s": completion,
                "flow_s": completion - job.arrival_s,
                "profile": profiles[job.workload_key],
            }
        )
    return {
        "run_id": run_id,
        "taskset": taskset_name,
        "trace": trace.snapshot(),
        "policy": pred["policy"],
        "profiles": profiles,
        "source_records": [record.snapshot() for record in source_records],
        "jobs": job_rows,
    }


def _cache_from_sources(
    members: Mapping[str, TaskSetMember],
    source_paths: Mapping[str, Iterable[str | Path]],
) -> tuple[ServiceRateCache, list[ProfileRecord]]:
    cache = ServiceRateCache()
    records = []
    for workload_key, paths in source_paths.items():
        if workload_key not in members:
            raise KeyError(f"source workload {workload_key!r} is not in the selected taskset")
        member = members[workload_key]
        for path in paths:
            for record in records_from_summary_file(
                Path(path).expanduser(),
                workload_key=workload_key,
                command_fingerprint="live_proxy_source",
                resource_kind=member.resource_kind,
                total_units=member.total_units,
                node_bucket="live_proxy",
            ):
                cache.add(record)
                records.append(record)
    return cache, records


def _require_profile(cache: ServiceRateCache, workload_key: str, target_profile: int) -> None:
    for profile in range(1, max(1, int(target_profile)) + 1):
        record = cache.get(workload_key, profile)
        if record is None or record.capacity_boundary or record.aggregate_rate <= 0:
            raise KeyError(
                f"live source cache lacks usable {workload_key!r} profile {profile}; "
                "provide every profile that can appear as an active-count state"
            )


def _parse_sources(values: Iterable[str]) -> dict[str, list[Path]]:
    sources: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must be workload_key=/path/to/summary.json, got {value!r}")
        workload_key, path = value.split("=", 1)
        workload_key = workload_key.strip()
        if not workload_key or not path.strip():
            raise ValueError(f"invalid --source value {value!r}")
        sources.setdefault(workload_key, []).append(Path(path.strip()).expanduser())
    return sources


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    live = build_composed_live_report(
        _json(args.replay_report),
        taskset_name=args.taskset,
        source_paths=_parse_sources(args.source),
        policy=args.policy or None,
        arrival_mode=args.arrival_mode,
        trace_seed=args.seed,
        replay_seed=args.replay_seed,
        run_id=args.run_id,
    )
    _write_json(args.output_live_report, live)
    if args.output_comparison:
        comparison = compare_replay_to_live(
            _json(args.replay_report),
            live,
            policy=args.policy or None,
            max_relative_error=args.max_relative_error,
        )
        _write_json(args.output_comparison, comparison)
        print(args.output_comparison)
        return 0 if comparison.get("usable_for_live_sanity") else 2
    print(args.output_live_report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.portfolio_live_proxy")
    p.add_argument("--replay-report", required=True)
    p.add_argument("--output-live-report", required=True)
    p.add_argument("--output-comparison", default="")
    p.add_argument("--taskset", default="hybrid_research_portfolio")
    p.add_argument("--arrival-mode", choices=("static", "poisson"), default="static")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay-seed", type=int, default=7)
    p.add_argument("--policy", default="")
    p.add_argument("--run-id", default="composed_live_proxy")
    p.add_argument("--max-relative-error", type=float, default=0.20)
    p.add_argument(
        "--source",
        action="append",
        default=[],
        help="Repeat as workload_key=/path/to/profile_N_summary.json.",
    )
    p.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
