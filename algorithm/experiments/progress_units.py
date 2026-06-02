"""Progress-unit parsing for service-curve experiments.

The live service-curve runner measures throughput in task-native progress
units.  Synthetic benchmarks print ``step/s``; many RL jobs print lines such as
``Iter 1989 | ... | 10.6s/iter``.  This module keeps that parsing out of the
scheduler policy so experiments can compare workload classes without changing
placement behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
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
    if labeled is not None:
        return labeled

    bare = None
    for m in _BARE_FRACTION_RE.finditer(line or ""):
        try:
            current = int(m.group(1))
            total = int(m.group(2))
        except ValueError:
            continue
        if total > 0 and 0 <= current <= total:
            bare = (current, total, None)
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
