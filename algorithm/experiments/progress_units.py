"""Progress-unit parsing for service-curve experiments.

The live service-curve runner measures throughput in task-native progress
units.  Synthetic benchmarks print ``step/s``; many RL jobs print lines such as
``Iter 1989 | ... | 10.6s/iter``.  This module keeps that parsing out of the
scheduler policy so experiments can compare workload classes without changing
placement behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from statistics import median
from typing import Any, Mapping


@dataclass(frozen=True)
class ProgressObservation:
    """Latest task-native progress signal found in a log line or task row."""

    current: int | None
    total: int | None
    rate_per_s: float | None
    seconds_per_unit: float | None
    unit: str
    source: str
    line: str


_PHASE_MARKER_RE = re.compile(
    r"ScheduleurmPhase\s+name=(?P<name>[A-Za-z0-9_.-]+)\s+"
    r"event=(?P<event>start|end)(?:\s+index=(?P<index>[A-Za-z0-9_.-]+))?",
    re.IGNORECASE,
)
_COMPLETION_MODEL_PREFIX = "ScheduleurmCompletionModel "


class CompletionTimingTracker:
    """Wall-clock completion model assembled around task-native progress.

    Stable service and completion time are deliberately separate contracts.
    Progress observations estimate the service portion.  A naturally completed
    run additionally exposes startup, periodic phase, and finalization costs.
    A probe stopped after reaching a stable window never becomes a completion
    model, even though its service-rate row may still be valid.
    """

    def __init__(self, *, total_units: int | None = None, unit: str = "unit"):
        self.total_units = int(total_units) if total_units and int(total_units) > 0 else None
        self.unit = canonical_unit(unit)
        self._progress: list[tuple[int, float]] = []
        self._phase_starts: dict[tuple[str, str], float] = {}
        self._phase_durations: dict[str, list[float]] = {}

    def observe_progress(self, obs: ProgressObservation, *, elapsed_s: float) -> None:
        if obs.current is None:
            return
        current = int(obs.current)
        elapsed = max(0.0, float(elapsed_s))
        if current < 0:
            return
        if obs.total is not None and int(obs.total) > 0:
            self.total_units = int(obs.total)
        self.unit = canonical_unit(obs.unit or self.unit)
        if self._progress and current < self._progress[-1][0]:
            # A restarted child must not splice two progress clocks together.
            self._progress = []
        if self._progress and current == self._progress[-1][0]:
            self._progress[-1] = (current, elapsed)
        else:
            self._progress.append((current, elapsed))

    def observe_phase_line(self, line: str, *, elapsed_s: float) -> bool:
        match = _PHASE_MARKER_RE.search(str(line or ""))
        if not match:
            return False
        name = str(match.group("name") or "unknown").lower()
        index = str(match.group("index") or "")
        key = (name, index)
        elapsed = max(0.0, float(elapsed_s))
        if str(match.group("event") or "").lower() == "start":
            self._phase_starts[key] = elapsed
            return True
        started = self._phase_starts.pop(key, None)
        if started is not None and elapsed >= started:
            self._phase_durations.setdefault(name, []).append(elapsed - started)
        return True

    def finalize(
        self,
        *,
        elapsed_s: float,
        child_returncode: int,
        stopped_on_stable: bool,
    ) -> dict[str, Any]:
        elapsed = max(0.0, float(elapsed_s))
        points = list(self._progress)
        first_current = points[0][0] if points else None
        last_current = points[-1][0] if points else None
        first_progress_s = points[0][1] if points else 0.0
        last_progress_s = points[-1][1] if points else 0.0

        interval_unit_seconds: list[float] = []
        for (left_unit, left_s), (right_unit, right_s) in zip(points, points[1:]):
            delta_units = int(right_unit) - int(left_unit)
            delta_s = float(right_s) - float(left_s)
            if delta_units > 0 and delta_s > 0.0:
                interval_unit_seconds.append(delta_s / float(delta_units))

        loop_units = (
            max(0, int(last_current) - int(first_current))
            if first_current is not None and last_current is not None
            else 0
        )
        loop_wall_s = max(0.0, last_progress_s - first_progress_s) if len(points) >= 2 else 0.0
        amortized_unit_s = loop_wall_s / float(loop_units) if loop_units > 0 else 0.0
        base_unit_s = float(median(interval_unit_seconds)) if interval_unit_seconds else 0.0
        periodic_extra_s = max(0.0, loop_wall_s - float(loop_units) * base_unit_s)

        total = int(self.total_units or 0)
        counter_offset = 0
        if total > 0 and last_current is not None and int(last_current) == total - 1:
            counter_offset = 1
        completed_units = (
            max(0, int(last_current) + counter_offset)
            if last_current is not None
            else 0
        )
        first_completed_units = (
            max(0, int(first_current) + counter_offset)
            if first_current is not None
            else 0
        )
        reached_total = total > 0 and completed_units >= total
        natural_exit = not bool(stopped_on_stable) and int(child_returncode) == 0
        completion_ready = bool(
            natural_exit
            and reached_total
            and len(points) >= 2
            and amortized_unit_s > 0.0
        )
        finalization_s = max(0.0, elapsed - last_progress_s) if natural_exit and points else 0.0
        startup_to_first_progress_s = first_progress_s if points else 0.0
        startup_overhead_s = max(
            0.0,
            startup_to_first_progress_s
            - float(first_completed_units) * amortized_unit_s,
        )
        predicted_total_s = (
            startup_overhead_s
            + float(total) * amortized_unit_s
            + finalization_s
            if total > 0 and amortized_unit_s > 0.0
            else 0.0
        )
        model_abs_error_s = (
            abs(predicted_total_s - elapsed) if predicted_total_s > 0.0 else 0.0
        )
        phase_durations = {
            name: [float(value) for value in values]
            for name, values in sorted(self._phase_durations.items())
        }
        checkpoint_s = sum(phase_durations.get("checkpoint", []))
        save_s = sum(phase_durations.get("save", [])) + sum(
            phase_durations.get("final_save", [])
        )
        return {
            "schema_version": 1,
            "completion_model_ready": completion_ready,
            "natural_exit": natural_exit,
            "stopped_on_stable": bool(stopped_on_stable),
            "child_returncode": int(child_returncode),
            "unit": self.unit,
            "total_units": total,
            "first_progress_unit": first_current,
            "last_progress_unit": last_current,
            "counter_offset": counter_offset,
            "progress_observation_count": len(points),
            "interval_sample_count": len(interval_unit_seconds),
            "startup_to_first_progress_s": startup_to_first_progress_s,
            "startup_overhead_s": startup_overhead_s,
            "loop_observed_units": loop_units,
            "loop_wall_s": loop_wall_s,
            "base_unit_s": base_unit_s,
            "amortized_unit_s": amortized_unit_s,
            "completion_unit_s": amortized_unit_s,
            "periodic_extra_s": periodic_extra_s,
            "finalization_after_last_progress_s": finalization_s,
            "terminal_overhead_s": finalization_s,
            "checkpoint_observed_s": checkpoint_s,
            "save_observed_s": save_s,
            "total_wall_s": elapsed,
            "predicted_total_s": predicted_total_s,
            "model_abs_error_s": model_abs_error_s,
            "model_relative_error": (
                model_abs_error_s / elapsed if elapsed > 0.0 else 0.0
            ),
            "phase_durations_s": phase_durations,
            "readiness_reason": _completion_readiness_reason(
                natural_exit=natural_exit,
                reached_total=reached_total,
                point_count=len(points),
                amortized_unit_s=amortized_unit_s,
                stopped_on_stable=bool(stopped_on_stable),
                child_returncode=int(child_returncode),
            ),
        }


def _completion_readiness_reason(
    *,
    natural_exit: bool,
    reached_total: bool,
    point_count: int,
    amortized_unit_s: float,
    stopped_on_stable: bool,
    child_returncode: int,
) -> str:
    if stopped_on_stable:
        return "stable_service_probe_not_natural_completion"
    if int(child_returncode) != 0:
        return f"child_returncode_{int(child_returncode)}"
    if not natural_exit:
        return "not_natural_exit"
    if not reached_total:
        return "progress_did_not_reach_total"
    if int(point_count) < 2:
        return "insufficient_progress_observations"
    if float(amortized_unit_s) <= 0.0:
        return "nonpositive_amortized_unit_time"
    return "ready"


def completion_model_line(model: Mapping[str, Any]) -> str:
    return _COMPLETION_MODEL_PREFIX + json.dumps(
        dict(model), sort_keys=True, separators=(",", ":")
    )


def parse_completion_model(text: str | None) -> dict[str, Any] | None:
    latest = None
    for line in str(text or "").splitlines():
        if not line.startswith(_COMPLETION_MODEL_PREFIX):
            continue
        try:
            value = json.loads(line[len(_COMPLETION_MODEL_PREFIX):])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            latest = value
    return latest


def completion_eta_seconds(
    model: Mapping[str, Any],
    *,
    current_unit: int = 0,
    include_startup: bool = False,
) -> float:
    """Estimate remaining completion time from a natural-run model.

    The amortized outer-loop unit time already contains observed periodic
    checkpoint/evaluation stalls.  Startup is included only for pre-launch
    estimates; finalization is always still owed until task completion.
    """

    if not bool(model.get("completion_model_ready")):
        raise ValueError("completion model is not ready")
    total = max(0, int(model.get("total_units") or 0))
    offset = max(0, int(model.get("counter_offset") or 0))
    completed = max(0, int(current_unit) + offset)
    remaining_units = max(0, total - completed)
    unit_value = (
        model.get("completion_unit_s")
        if "completion_unit_s" in model
        else model.get("amortized_unit_s")
    )
    unit_s = max(0.0, float(unit_value or 0.0))
    remaining = remaining_units * unit_s
    terminal_value = (
        model.get("terminal_overhead_s")
        if "terminal_overhead_s" in model
        else model.get("finalization_after_last_progress_s")
    )
    remaining += max(0.0, float(terminal_value or 0.0))
    if include_startup:
        startup_value = (
            model.get("startup_overhead_s")
            if "startup_overhead_s" in model
            else model.get("startup_to_first_progress_s")
        )
        remaining += max(0.0, float(startup_value or 0.0))
    return remaining


_UNIT_ALIASES = {
    "it": "iter",
    "iter": "iter",
    "iters": "iter",
    "iteration": "iter",
    "iterations": "iter",
    "step": "step",
    "steps": "step",
    "episode": "episode",
    "episodes": "episode",
    "epoch": "epoch",
    "epochs": "epoch",
    "update": "update",
    "updates": "update",
    "sample": "sample",
    "samples": "sample",
}

_LABELED_CURRENT_RE = re.compile(
    r"(?:^|[^\w])"
    r"(?<!/)"
    r"(Iter|Iteration|Epoch|Step|Episode|Update|Sample)"
    r"[:\s#]+(\d+)"
    r"(?:\s*(?:/|of)\s*(\d+))?",
    re.IGNORECASE,
)
_BARE_FRACTION_RE = re.compile(r"(?:^|[\s\[(])(\d+)\s*/\s*(\d+)(?:[\s\])]|$)")
_RATE_EQUALS_RE = re.compile(
    r"\brate\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*/\s*s\b",
    re.IGNORECASE,
)
_PER_SECOND_RE = re.compile(
    r"\b([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*/\s*s\b",
    re.IGNORECASE,
)
_SECONDS_PER_RE = re.compile(
    r"\b([0-9]+(?:\.[0-9]+)?)\s*s\s*/\s*([A-Za-z]+)\b",
    re.IGNORECASE,
)

_CMD_TOTAL_PATTERNS = [
    re.compile(r"--max[_-]?iters[=\s]+(\d+)"),
    re.compile(r"--num[_-]?iters[=\s]+(\d+)"),
    re.compile(r"--iterations[=\s]+(\d+)"),
    re.compile(r"--max[_-]?steps[=\s]+(\d+)"),
    re.compile(r"--total[_-]?steps[=\s]+(\d+)"),
    re.compile(r"--num[_-]?steps[=\s]+(\d+)"),
    re.compile(r"--n[_-]?steps[=\s]+(\d+)"),
    re.compile(r"--epochs[=\s]+(\d+)"),
    re.compile(r"--num[_-]?epochs[=\s]+(\d+)"),
    re.compile(r"--n[_-]?epochs[=\s]+(\d+)"),
]


def canonical_unit(unit: str | None) -> str:
    key = (unit or "unit").strip().lower()
    return _UNIT_ALIASES.get(key, key or "unit")


def _extract_total_from_cmd(cmd: str | None) -> int | None:
    if not cmd:
        return None
    for pat in _CMD_TOTAL_PATTERNS:
        m = pat.search(cmd)
        if not m:
            continue
        try:
            value = int(m.group(1))
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def _current_from_line(line: str) -> tuple[int | None, int | None, str | None]:
    labeled = None
    for m in _LABELED_CURRENT_RE.finditer(line or ""):
        try:
            current = int(m.group(2))
            total = int(m.group(3)) if m.group(3) is not None else None
        except ValueError:
            continue
        if current < 0 or (total is not None and (total <= 0 or current > total)):
            continue
        labeled = (current, total, canonical_unit(m.group(1)))

    bare = None
    for m in _BARE_FRACTION_RE.finditer(line or ""):
        try:
            current = int(m.group(1))
            total = int(m.group(2))
        except ValueError:
            continue
        if total > 0 and 0 <= current <= total:
            bare = (current, total, None)
    if labeled is not None:
        if labeled[1] is None and bare is not None and bare[0] == labeled[0]:
            return (bare[0], bare[1], labeled[2])
        return labeled
    return bare or (None, None, None)


def _rate_from_line(line: str) -> tuple[float | None, float | None, str | None, str | None]:
    for pat, source in ((_RATE_EQUALS_RE, "rate_equals"), (_PER_SECOND_RE, "per_second")):
        latest = None
        for m in pat.finditer(line or ""):
            try:
                rate = float(m.group(1))
            except ValueError:
                continue
            if rate > 0:
                latest = (rate, 1.0 / rate, canonical_unit(m.group(2)), source)
        if latest is not None:
            return latest

    latest = None
    for m in _SECONDS_PER_RE.finditer(line or ""):
        try:
            seconds = float(m.group(1))
        except ValueError:
            continue
        if seconds > 0:
            latest = (1.0 / seconds, seconds, canonical_unit(m.group(2)), "seconds_per_unit")
    if latest is not None:
        return latest
    return (None, None, None, None)


def parse_progress_line(line: str, *, cmd: str | None = None) -> ProgressObservation | None:
    """Parse one log line into a task-native progress observation.

    Returns ``None`` when the line has neither a progress counter nor a rate.
    """

    if not line:
        return None
    current, total, current_unit = _current_from_line(line)
    rate, seconds, rate_unit, rate_source = _rate_from_line(line)
    if current is None and rate is None:
        return None
    if total is None:
        total = _extract_total_from_cmd(cmd)
    if current is not None and total is not None and current > total:
        return None
    unit = canonical_unit(rate_unit or current_unit)
    source = rate_source or ("current_total" if total is not None else "current_only")
    return ProgressObservation(
        current=current,
        total=total,
        rate_per_s=rate,
        seconds_per_unit=seconds,
        unit=unit,
        source=source,
        line=line,
    )


def parse_progress_observation(
    text: str | None,
    *,
    cmd: str | None = None,
    runtime_current_unit: int | None = None,
    runtime_total_units: int | None = None,
) -> ProgressObservation | None:
    """Return the latest progress observation in a log tail.

    Scheduler task rows often already carry ``runtime_current_unit`` and
    ``runtime_total_units``.  Those fields are used as fallbacks when the latest
    rate-bearing line does not repeat the counter.
    """

    latest = None
    for line in (text or "").splitlines():
        obs = parse_progress_line(line, cmd=cmd)
        if obs is not None:
            latest = obs
    if latest is None:
        return None

    current = latest.current
    total = latest.total
    if current is None and runtime_current_unit is not None:
        current = int(runtime_current_unit)
    if total is None:
        if runtime_total_units is not None:
            total = int(runtime_total_units)
        else:
            total = _extract_total_from_cmd(cmd)
    return ProgressObservation(
        current=current,
        total=total,
        rate_per_s=latest.rate_per_s,
        seconds_per_unit=latest.seconds_per_unit,
        unit=latest.unit,
        source=latest.source,
        line=latest.line,
    )


def task_progress_observation(task: Mapping[str, Any]) -> ProgressObservation | None:
    """Parse progress from a scheduler task row."""

    return parse_progress_observation(
        task.get("last_progress_line"),
        cmd=task.get("cmd"),
        runtime_current_unit=task.get("runtime_current_unit"),
        runtime_total_units=task.get("runtime_total_units"),
    )


def task_rate_per_s(task: Mapping[str, Any]) -> float:
    obs = task_progress_observation(task)
    if obs is None or obs.rate_per_s is None:
        return 0.0
    return float(obs.rate_per_s)
