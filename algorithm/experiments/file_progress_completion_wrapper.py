"""Run a command while turning durable file progress into completion ETA.

Some native workloads write trustworthy outer-loop progress to a CSV but print
too little stdout for an online ETA.  This wrapper observes the durable file,
emits canonical progress lines, and accounts for startup plus post-progress
checkpoint/final-save time in a natural-completion model.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from statistics import fmean, median, pstdev
import subprocess
import sys
import time
from typing import Iterable

from algorithm.experiments.progress_units import (
    CompletionTimingTracker,
    ProgressObservation,
    completion_model_line,
)


RESOURCE_STATE_PREFIX = "ScheduleurmResourceState "
DISPATCH_CPU_SAMPLE_WINDOW_S = 1.0
DISPATCH_CPU_SAMPLE_COUNT = 5
MEASUREMENT_PROTOCOL = "dispatch_multisample_and_run_window_v3"


def _cpu_counters() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values) - idle, sum(values)


def _process_cpu_seconds(value: os.times_result) -> float:
    return float(value.user + value.system + value.children_user + value.children_system)


def _sample_dispatch_cpu_state(
    *,
    window_s: float = DISPATCH_CPU_SAMPLE_WINDOW_S,
    sample_count: int = DISPATCH_CPU_SAMPLE_COUNT,
) -> dict[str, float | int | bool | str | list[float] | None]:
    """Measure the CPU state available before the controlled child exists."""

    sample_window_s = max(0.2, float(window_s))
    count = max(3, int(sample_count))
    sample_started = time.perf_counter()
    busy_fractions = []
    logical_cpus = max(1, int(os.cpu_count() or 1))
    for _ in range(count):
        busy_start, total_start = _cpu_counters()
        time.sleep(sample_window_s)
        busy_end, total_end = _cpu_counters()
        delta_total = max(1, int(total_end - total_start))
        busy_fractions.append(
            max(
                0.0,
                min(
                    1.0,
                    float(busy_end - busy_start) / float(delta_total),
                ),
            )
        )
    elapsed = max(1e-9, time.perf_counter() - sample_started)
    mean_fraction = float(fmean(busy_fractions))
    median_fraction = float(median(busy_fractions))
    p95_fraction = _nearest_rank(busy_fractions, 0.95)
    minimum_fraction = min(busy_fractions)
    maximum_fraction = max(busy_fractions)
    std_fraction = float(pstdev(busy_fractions))
    cv = (
        std_fraction / mean_fraction
        if mean_fraction > 1e-12
        else (0.0 if std_fraction <= 1e-12 else None)
    )
    regime = _dispatch_cpu_regime(
        mean_fraction=mean_fraction,
        p95_fraction=p95_fraction,
    )
    return {
        "dispatch_state_ready": True,
        "dispatch_state_observed_before_child": True,
        "dispatch_sample_window_s": elapsed,
        "dispatch_sample_subwindow_s": sample_window_s,
        "dispatch_sample_count": count,
        "dispatch_logical_cpu_count": logical_cpus,
        "dispatch_system_busy_fraction": mean_fraction,
        "dispatch_system_busy_fraction_median": median_fraction,
        "dispatch_system_busy_fraction_p95": p95_fraction,
        "dispatch_system_busy_fraction_min": minimum_fraction,
        "dispatch_system_busy_fraction_max": maximum_fraction,
        "dispatch_system_busy_fraction_std": std_fraction,
        "dispatch_system_busy_fraction_cv": cv,
        "dispatch_system_busy_fraction_samples": busy_fractions,
        "dispatch_system_busy_core_equiv": (
            mean_fraction * float(logical_cpus)
        ),
        "dispatch_system_busy_core_equiv_median": (
            median_fraction * float(logical_cpus)
        ),
        "dispatch_system_busy_core_equiv_p95": (
            p95_fraction * float(logical_cpus)
        ),
        "dispatch_system_busy_core_equiv_min": (
            minimum_fraction * float(logical_cpus)
        ),
        "dispatch_system_busy_core_equiv_max": (
            maximum_fraction * float(logical_cpus)
        ),
        "dispatch_external_cpu_regime": regime,
        # No controlled child exists during this window, so all observed work
        # is external to the action being evaluated.
        "dispatch_external_cpu_fraction": mean_fraction,
        "dispatch_external_cpu_fraction_p95": p95_fraction,
        "dispatch_external_cpu_core_equiv": (
            mean_fraction * float(logical_cpus)
        ),
        "dispatch_external_cpu_core_equiv_p95": (
            p95_fraction * float(logical_cpus)
        ),
    }


def _nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = max(1, int(math.ceil(float(quantile) * len(ordered))))
    return ordered[min(len(ordered) - 1, rank - 1)]


def _dispatch_cpu_regime(
    *,
    mean_fraction: float,
    p95_fraction: float,
) -> str:
    """Hardware-normalized pre-dispatch load state used by service cells."""

    mean_value = max(0.0, min(1.0, float(mean_fraction)))
    p95_value = max(0.0, min(1.0, float(p95_fraction)))
    if p95_value <= 0.02:
        return "cpu_external_idle"
    if mean_value <= 0.10 and p95_value <= 0.25:
        return "cpu_external_light"
    if mean_value <= 0.35 and p95_value <= 0.50:
        return "cpu_external_moderate"
    if mean_value <= 0.70 and p95_value <= 0.85:
        return "cpu_external_heavy"
    return "cpu_external_saturated"


def _progress_count(pattern: str, *, has_header: bool) -> tuple[int, str]:
    paths = [Path(path) for path in sorted(glob.glob(pattern)) if Path(path).is_file()]
    if not paths:
        return 0, ""
    # Native runners in this repository write one authoritative CSV per job.
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        count = sum(1 for line in handle if line.strip())
    if has_header and count:
        count -= 1
    return max(0, count), str(path)


def _artifact_summary(patterns: Iterable[str]) -> tuple[int, int]:
    seen: set[str] = set()
    total_bytes = 0
    for pattern in patterns:
        for raw_path in glob.glob(pattern, recursive=True):
            path = Path(raw_path)
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            total_bytes += int(path.stat().st_size)
    return len(seen), total_bytes


def run_file_progress_command(
    *,
    command: list[str],
    progress_file_glob: str,
    total: int,
    unit: str,
    poll_s: float,
    csv_has_header: bool,
    artifact_globs: Iterable[str] = (),
    sample_dispatch_state: bool = True,
    dispatch_sample_window_s: float = DISPATCH_CPU_SAMPLE_WINDOW_S,
    dispatch_sample_count: int = DISPATCH_CPU_SAMPLE_COUNT,
) -> int:
    dispatch_state = (
        _sample_dispatch_cpu_state(
            window_s=dispatch_sample_window_s,
            sample_count=dispatch_sample_count,
        )
        if sample_dispatch_state
        else {
            "dispatch_state_ready": False,
            "dispatch_state_observed_before_child": False,
            "dispatch_sample_window_s": 0.0,
            "dispatch_sample_count": 0,
            "dispatch_logical_cpu_count": int(os.cpu_count() or 1),
            "dispatch_system_busy_fraction": None,
            "dispatch_system_busy_core_equiv": None,
            "dispatch_external_cpu_regime": None,
            "dispatch_external_cpu_fraction": None,
            "dispatch_external_cpu_core_equiv": None,
        }
    )
    started = time.perf_counter()
    busy_start, total_start = _cpu_counters()
    process_start = os.times()
    tracker = CompletionTimingTracker(total_units=total, unit=unit)

    print("ScheduleurmPhase name=initialization event=start", flush=True)
    child = subprocess.Popen(command)
    print("ScheduleurmPhase name=initialization event=end", flush=True)
    print("ScheduleurmPhase name=outer_loop event=start", flush=True)

    last_count = -1
    progress_path = ""
    returncode: int | None = None
    while returncode is None:
        count, path = _progress_count(progress_file_glob, has_header=csv_has_header)
        if count != last_count and count > 0:
            progress_path = path or progress_path
            elapsed = max(1e-9, time.perf_counter() - started)
            rate = count / elapsed
            eta = max(0, total - count) / max(rate, 1e-9)
            print(
                f"Native progress {count}/{total} rate={rate:.9g} {unit}/s "
                f"ETA {eta:.1f}s path={progress_path}",
                flush=True,
            )
            tracker.observe_progress(
                ProgressObservation(
                    current=count,
                    total=total,
                    rate_per_s=rate,
                    seconds_per_unit=1.0 / rate,
                    unit=unit,
                    source="durable_csv",
                    line="",
                ),
                elapsed_s=elapsed,
            )
            last_count = count
        try:
            returncode = int(child.wait(timeout=max(0.2, float(poll_s))))
        except subprocess.TimeoutExpired:
            pass
    final_count, final_path = _progress_count(
        progress_file_glob, has_header=csv_has_header
    )
    if final_count > max(0, last_count):
        progress_path = final_path or progress_path
        elapsed = max(1e-9, time.perf_counter() - started)
        rate = final_count / elapsed
        print(
            f"Native progress {final_count}/{total} rate={rate:.9g} {unit}/s "
            f"ETA 0.0s path={progress_path}",
            flush=True,
        )
        tracker.observe_progress(
            ProgressObservation(
                current=final_count,
                total=total,
                rate_per_s=rate,
                seconds_per_unit=1.0 / rate,
                unit=unit,
                source="durable_csv",
                line="",
            ),
            elapsed_s=elapsed,
        )

    print("ScheduleurmPhase name=outer_loop event=end", flush=True)
    elapsed = max(1e-9, time.perf_counter() - started)
    model = tracker.finalize(
        elapsed_s=elapsed,
        child_returncode=int(returncode),
        stopped_on_stable=False,
    )
    artifact_count, artifact_bytes = _artifact_summary(artifact_globs)
    model.update(
        {
            "progress_source": "durable_csv",
            "progress_file": progress_path,
            "artifact_count": artifact_count,
            "artifact_bytes": artifact_bytes,
        }
    )
    print(completion_model_line(model), flush=True)

    process_end = os.times()
    busy_end, total_end = _cpu_counters()
    logical_cpus = max(1, int(os.cpu_count() or 1))
    delta_total = max(1, int(total_end - total_start))
    busy_fraction = max(
        0.0, min(1.0, float(busy_end - busy_start) / float(delta_total))
    )
    system_busy_core_equiv = busy_fraction * float(logical_cpus)
    own_cpu_s = max(
        0.0,
        _process_cpu_seconds(process_end) - _process_cpu_seconds(process_start),
    )
    own_cpu_core_equiv = own_cpu_s / elapsed
    run_window_external_cpu_core_equiv = max(
        0.0, system_busy_core_equiv - own_cpu_core_equiv
    )
    print(
        RESOURCE_STATE_PREFIX
        + json.dumps(
            {
                "schema_version": 3,
                "measurement_protocol": MEASUREMENT_PROTOCOL,
                "logical_cpus": logical_cpus,
                **dispatch_state,
                "measurement_wall_s": elapsed,
                "run_window_system_busy_fraction": busy_fraction,
                "run_window_system_busy_core_equiv": system_busy_core_equiv,
                "own_cpu_s": own_cpu_s,
                "own_cpu_core_equiv": own_cpu_core_equiv,
                "run_window_external_cpu_core_equiv": (
                    run_window_external_cpu_core_equiv
                ),
                # Compatibility aliases remain run-window quantities. They
                # must never be used as dispatch-time theorem covariates.
                "system_busy_fraction": busy_fraction,
                "system_busy_core_equiv": system_busy_core_equiv,
                "external_cpu_core_equiv": run_window_external_cpu_core_equiv,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return int(returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-file-glob", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--unit", default="episode")
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--csv-has-header", action="store_true")
    parser.add_argument("--skip-dispatch-state-sample", action="store_true")
    parser.add_argument(
        "--dispatch-sample-window-s",
        type=float,
        default=DISPATCH_CPU_SAMPLE_WINDOW_S,
    )
    parser.add_argument(
        "--dispatch-sample-count",
        type=int,
        default=DISPATCH_CPU_SAMPLE_COUNT,
    )
    parser.add_argument("--artifact-glob", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a child command is required after --")
    return run_file_progress_command(
        command=command,
        progress_file_glob=str(args.progress_file_glob),
        total=max(1, int(args.total)),
        unit=str(args.unit),
        poll_s=max(0.2, float(args.poll_s)),
        csv_has_header=bool(args.csv_has_header),
        artifact_globs=[str(value) for value in args.artifact_glob],
        sample_dispatch_state=not bool(args.skip_dispatch_state_sample),
        dispatch_sample_window_s=max(
            0.2,
            float(args.dispatch_sample_window_s),
        ),
        dispatch_sample_count=max(3, int(args.dispatch_sample_count)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
