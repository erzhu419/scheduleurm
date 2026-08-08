"""Line-oriented progress wrapper for real workload service-curve runs.

The scheduler watches task logs line by line.  A terminal tqdm bar is useful for
humans but often collapses into carriage-return fragments in remote logs.  This
wrapper preserves the child process output and adds one normalized progress line
whenever a task-native counter/rate can be parsed.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.experiments.progress_units import (
    CompletionTimingTracker,
    ProgressObservation,
    canonical_unit,
    completion_model_line,
    parse_progress_line,
)


def _format_hms(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _command_for_parser(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def build_progress_line(
    obs: ProgressObservation,
    *,
    total_override: int | None = None,
    unit_override: str | None = None,
) -> str | None:
    """Render a scheduler-stable progress line from a parsed child log line."""

    current = obs.current
    total = int(total_override) if total_override and int(total_override) > 0 else obs.total
    rate = obs.rate_per_s
    unit = canonical_unit(unit_override or obs.unit)
    if current is None and rate is None:
        return None
    if current is not None and total is not None and int(current) > int(total):
        return None

    parts = ["ScheduleurmProgress"]
    label = unit.capitalize() if unit else "Unit"
    if current is not None and total is not None:
        parts += [label, f"{int(current)}/{int(total)}"]
    elif current is not None:
        parts += [label, str(int(current))]
    else:
        parts += [label, "unknown"]

    if rate is not None and float(rate) > 0:
        rate_f = float(rate)
        parts.append(f"rate={rate_f:.9g} {unit}/s")
        if current is not None and total is not None:
            remaining = max(0, int(total) - int(current))
            eta_s = remaining / rate_f
            parts.append(f"ETA {eta_s:.1f}s")
        if obs.seconds_per_unit is not None:
            parts.append(f"seconds_per_{unit}={float(obs.seconds_per_unit):.6g}")
    parts.append(f"source={obs.source}")
    return " ".join(parts)


def build_tqdm_line(
    obs: ProgressObservation,
    *,
    total_override: int | None = None,
    unit_override: str | None = None,
    elapsed_s: float = 0.0,
) -> str | None:
    """Render a persistent tqdm-style ETA line for scheduler log tails.

    Real terminal tqdm bars often update with carriage returns.  Remote log tails
    can lose those updates, so the wrapper emits a normal newline with the same
    elapsed/remaining/rate contract that eta_tracker understands.
    """

    current = obs.current
    total = int(total_override) if total_override and int(total_override) > 0 else obs.total
    rate = obs.rate_per_s
    unit = canonical_unit(unit_override or obs.unit)
    if current is None or total is None or int(total) <= 0:
        return None
    current_i = int(current)
    total_i = int(total)
    if current_i < 0 or current_i > total_i:
        return None
    if rate is None or float(rate) <= 0:
        return None

    rate_f = float(rate)
    remaining_s = max(0.0, float(total_i - current_i) / max(rate_f, 1e-12))
    elapsed = _format_hms(elapsed_s)
    remaining = _format_hms(remaining_s)
    pct = 100.0 * float(current_i) / float(total_i)
    filled = int(round(10.0 * float(current_i) / float(total_i)))
    bar = "#" * max(0, min(10, filled)) + "-" * max(0, 10 - min(10, filled))
    if obs.seconds_per_unit is not None and float(obs.seconds_per_unit) >= 1.0:
        rate_text = f"{float(obs.seconds_per_unit):.3g}s/{unit}"
    else:
        rate_text = f"{rate_f:.3g}{unit}/s"
    return (
        f"ScheduleurmTqdm {unit}: {pct:5.1f}%|{bar}| "
        f"{current_i}/{total_i} [{elapsed}<{remaining}, {rate_text}]"
    )


def _normalize_child_line(
    line: str,
    *,
    command: Sequence[str],
    total_override: int | None,
    unit_override: str | None,
) -> tuple[str | None, ProgressObservation | None]:
    obs = parse_progress_line(line, cmd=_command_for_parser(command))
    if obs is None:
        return None, None
    return build_progress_line(obs, total_override=total_override, unit_override=unit_override), obs


def _progress_source_rank(obs: ProgressObservation) -> int:
    """Rank explicit task counters above duplicate terminal progress bars."""

    source = str(obs.source or "").lower()
    if source == "per_second":
        return 1
    if source == "current_total":
        return 2
    return 3


def _matches_declared_total(
    obs: ProgressObservation,
    *,
    total_override: int | None,
) -> bool:
    if total_override is None or int(total_override) <= 0:
        return True
    if obs.total is not None and int(obs.total) != int(total_override):
        return False
    return obs.current is None or int(obs.current) <= int(total_override)


def _stable_rate_decision(
    rates: Sequence[float],
    *,
    windows: int,
    min_samples: int,
    max_cv: float,
    max_last_two_rel_delta: float,
    skip_samples: int,
) -> dict[str, object]:
    usable = [float(x) for x in rates[int(max(0, skip_samples)):] if float(x) > 0.0]
    needed = max(int(min_samples), int(windows), 1)
    if len(usable) < needed:
        return {
            "ready": False,
            "reason": f"need_at_least_{needed}_positive_rate_samples",
            "usable_samples": len(usable),
            "tail": usable[-int(max(1, windows)):],
        }
    tail = usable[-int(max(1, windows)):]
    avg = sum(tail) / len(tail)
    var = sum((x - avg) ** 2 for x in tail) / len(tail)
    cv = math.sqrt(var) / max(abs(avg), 1e-12)
    rel = (
        abs(tail[-1] - tail[-2]) / max(abs(tail[-2]), 1e-12)
        if len(tail) >= 2
        else 0.0
    )
    return {
        "ready": cv <= float(max_cv) and rel <= float(max_last_two_rel_delta),
        "tail": tail,
        "mean_rate": avg,
        "last_rate": tail[-1],
        "cv": cv,
        "last_two_relative_delta": rel,
        "usable_samples": len(usable),
        "rule": (
            f"{len(tail)} windows after {int(max(0, skip_samples))} skipped samples, "
            f"cv<={float(max_cv):.6g}, last-two relative delta<={float(max_last_two_rel_delta):.6g}"
        ),
    }


def _cycle_stable_rate_decision(
    rates: Sequence[float],
    *,
    cycle_units: int,
    windows: int,
    min_samples: int,
    max_cv: float,
    max_last_two_rel_delta: float,
    skip_samples: int,
) -> dict[str, object]:
    """Stable-rate decision for periodic workloads.

    Hybrid RL jobs often alternate cheap training iterations with periodic
    evaluation/checkpoint iterations.  Treating each line as an independent
    service sample makes a genuinely steady periodic job look unstable.  This
    helper first aggregates complete cycles by harmonic service time, then
    applies the ordinary tail-window stability test to cycle-average rates.
    """

    cycle_n = int(max(0, cycle_units))
    if cycle_n <= 1:
        return _stable_rate_decision(
            rates,
            windows=windows,
            min_samples=min_samples,
            max_cv=max_cv,
            max_last_two_rel_delta=max_last_two_rel_delta,
            skip_samples=skip_samples,
        )
    usable = [float(x) for x in rates[int(max(0, skip_samples)):] if float(x) > 0.0]
    needed_raw = max(int(min_samples), int(windows) * cycle_n, cycle_n)
    if len(usable) < needed_raw:
        return {
            "ready": False,
            "reason": f"need_at_least_{needed_raw}_positive_raw_rate_samples_for_{cycle_n}_unit_cycles",
            "usable_samples": 0,
            "raw_usable_samples": len(usable),
            "tail": [],
            "rule": (
                f"cycle-average over {cycle_n} native units after "
                f"{int(max(0, skip_samples))} skipped samples"
            ),
        }

    cycle_rates: list[float] = []
    for start in range(0, len(usable) - cycle_n + 1, cycle_n):
        chunk = usable[start : start + cycle_n]
        elapsed = sum(1.0 / max(float(rate), 1e-12) for rate in chunk)
        if elapsed > 0.0:
            cycle_rates.append(float(len(chunk)) / elapsed)

    decision = _stable_rate_decision(
        cycle_rates,
        windows=windows,
        min_samples=max(1, int(windows)),
        max_cv=max_cv,
        max_last_two_rel_delta=max_last_two_rel_delta,
        skip_samples=0,
    )
    decision["raw_usable_samples"] = len(usable)
    decision["cycle_units"] = cycle_n
    decision["cycle_rate_samples"] = len(cycle_rates)
    decision["rule"] = (
        f"{int(max(1, windows))} cycle-average windows over {cycle_n} native units "
        f"after {int(max(0, skip_samples))} skipped raw samples, "
        f"cv<={float(max_cv):.6g}, "
        f"last-two relative delta<={float(max_last_two_rel_delta):.6g}"
    )
    return decision


def _stable_rate_line(decision: dict[str, object], *, unit: str) -> str:
    tail = ",".join(f"{float(x):.9g}" for x in decision.get("tail", []) or [])
    return (
        "ScheduleurmStableRate "
        f"unit={canonical_unit(unit)} "
        f"mean_rate={float(decision.get('mean_rate') or 0.0):.9g} "
        f"last_rate={float(decision.get('last_rate') or 0.0):.9g} "
        f"cv={float(decision.get('cv') or 0.0):.9g} "
        f"last_two_relative_delta={float(decision.get('last_two_relative_delta') or 0.0):.9g} "
        f"samples={int(decision.get('usable_samples') or 0)} "
        f"tail={tail} "
        f"rule={str(decision.get('rule') or '').replace(' ', '_')}"
    )


def _child_return_code_for_shell(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return int(returncode)


def _usable_for_stable_rate(obs: ProgressObservation) -> bool:
    """Use task-native rate lines for stable ETA, not derived display bars."""

    return obs.source in {"rate_equals", "seconds_per_unit"}


_PHASE_LINE_RE = re.compile(r"\bname=([^\s]+)\s+event=(start|end)\b")
_NON_SERVICE_PHASES = {
    "admission_delay",
    "initialization",
    "warmup",
    "checkpoint",
    "final_save",
}


def _update_active_phases(line: str, active_phases: set[str]) -> None:
    """Track explicit benchmark phases without changing legacy unphased jobs."""

    if "ScheduleurmPhase" not in line:
        return
    match = _PHASE_LINE_RE.search(line)
    if not match:
        return
    name, event = match.groups()
    if event == "start":
        active_phases.add(name)
    else:
        active_phases.discard(name)


def _observation_allowed_in_phases(
    obs: ProgressObservation,
    active_phases: set[str],
) -> bool:
    """Exclude library tqdm bars emitted during setup/save from task progress."""

    if obs.source != "per_second":
        return True
    if not active_phases:
        return True
    return not bool(active_phases & _NON_SERVICE_PHASES) and "outer_loop" in active_phases


def _terminate_child(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_wrapped_command(
    command: Sequence[str],
    *,
    total: int | None,
    unit: str,
    terminate_on_stable: bool = False,
    stable_windows: int = 3,
    min_rate_samples: int = 3,
    stable_cv: float = 0.08,
    stable_rel_delta: float = 0.05,
    stable_skip_samples: int = 0,
    stable_cycle_units: int = 0,
    admission_delay_s: float = 0.0,
) -> int:
    if not command:
        raise ValueError("wrapped command is empty")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start_monotonic = time.monotonic()
    completion_tracker = CompletionTimingTracker(total_units=total, unit=unit)
    delay_s = max(0.0, float(admission_delay_s))
    if delay_s > 0.0:
        start_line = (
            "ScheduleurmPhase name=admission_delay event=start "
            f"delay_s={delay_s:.6g}"
        )
        print(start_line, flush=True)
        completion_tracker.observe_phase_line(start_line, elapsed_s=0.0)
        time.sleep(delay_s)
        elapsed_after_delay = time.monotonic() - start_monotonic
        end_line = (
            "ScheduleurmPhase name=admission_delay event=end "
            f"delay_s={delay_s:.6g}"
        )
        print(end_line, flush=True)
        completion_tracker.observe_phase_line(
            end_line,
            elapsed_s=elapsed_after_delay,
        )
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )
    assert proc.stdout is not None
    rates: list[float] = []
    stable_reported = False
    stopped_on_stable = False
    active_phases: set[str] = set()
    best_progress_source_rank = 0
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        print(line, flush=True)
        elapsed_s = time.monotonic() - start_monotonic
        completion_tracker.observe_phase_line(line, elapsed_s=elapsed_s)
        _update_active_phases(line, active_phases)
        progress_line, obs = _normalize_child_line(
            line,
            command=command,
            total_override=total,
            unit_override=unit,
        )
        if obs is not None and not _observation_allowed_in_phases(obs, active_phases):
            progress_line, obs = None, None
        if obs is not None and not _matches_declared_total(
            obs,
            total_override=total,
        ):
            progress_line, obs = None, None
        if obs is not None:
            source_rank = _progress_source_rank(obs)
            if source_rank < best_progress_source_rank:
                progress_line, obs = None, None
            else:
                best_progress_source_rank = max(
                    best_progress_source_rank,
                    source_rank,
                )
        if progress_line:
            print(progress_line, flush=True)
        if obs is not None:
            completion_tracker.observe_progress(obs, elapsed_s=elapsed_s)
            tqdm_line = build_tqdm_line(
                obs,
                total_override=total,
                unit_override=unit,
                elapsed_s=elapsed_s,
            )
            if tqdm_line:
                print(tqdm_line, flush=True)
        if obs is None or not _usable_for_stable_rate(obs):
            continue
        if obs.rate_per_s is None or float(obs.rate_per_s) <= 0.0:
            continue
        rates.append(float(obs.rate_per_s))
        decision = (
            _cycle_stable_rate_decision(
                rates,
                cycle_units=stable_cycle_units,
                windows=stable_windows,
                min_samples=min_rate_samples,
                max_cv=stable_cv,
                max_last_two_rel_delta=stable_rel_delta,
                skip_samples=stable_skip_samples,
            )
            if int(stable_cycle_units or 0) > 1
            else _stable_rate_decision(
                rates,
                windows=stable_windows,
                min_samples=min_rate_samples,
                max_cv=stable_cv,
                max_last_two_rel_delta=stable_rel_delta,
                skip_samples=stable_skip_samples,
            )
        )
        if bool(decision.get("ready")) and not stable_reported:
            print(_stable_rate_line(decision, unit=unit), flush=True)
            stable_reported = True
        if bool(decision.get("ready")) and terminate_on_stable:
            _terminate_child(proc)
            stopped_on_stable = True
            break
    rc = _child_return_code_for_shell(proc.wait())
    model = completion_tracker.finalize(
        elapsed_s=time.monotonic() - start_monotonic,
        child_returncode=rc,
        stopped_on_stable=stopped_on_stable,
    )
    print(completion_model_line(model), flush=True)
    return 0 if stopped_on_stable else rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default="iter")
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--terminate-on-stable", action="store_true")
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--min-rate-samples", type=int, default=3)
    parser.add_argument("--stable-cv", type=float, default=0.08)
    parser.add_argument("--stable-rel-delta", type=float, default=0.05)
    parser.add_argument("--stable-skip-samples", type=int, default=0)
    parser.add_argument(
        "--stable-cycle-units",
        type=int,
        default=0,
        help=(
            "Aggregate complete periodic workload cycles before testing "
            "stable rate. Use 0 for ordinary raw-rate stability."
        ),
    )
    parser.add_argument(
        "--admission-delay-s",
        type=float,
        default=float(os.environ.get("SCHEDULEURM_ADMISSION_DELAY_S", "0") or 0.0),
        help="Count a controlled staged-admission delay in startup and JCT.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    total = int(args.total) if int(args.total or 0) > 0 else None
    return run_wrapped_command(
        command,
        total=total,
        unit=args.unit,
        terminate_on_stable=bool(args.terminate_on_stable),
        stable_windows=args.stable_windows,
        min_rate_samples=args.min_rate_samples,
        stable_cv=args.stable_cv,
        stable_rel_delta=args.stable_rel_delta,
        stable_skip_samples=args.stable_skip_samples,
        stable_cycle_units=args.stable_cycle_units,
        admission_delay_s=args.admission_delay_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
