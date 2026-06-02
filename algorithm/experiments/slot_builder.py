"""Slot-level queue, arrival, and service reconstruction helpers.

The concrete stochastic theorem uses

    Q(k+1) = max(Q(k) - S(k), 0) + A(k)

on nonnegative class-indexed work units.  This module keeps that recurrence
explicit and conservative: negative progress is censored unless an upstream
normalizer has already certified a rollback/resume interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def nonnegative_vector(vec: Mapping[str, Any]) -> Dict[str, float]:
    return {str(k): max(0.0, _as_float(v)) for k, v in vec.items()}


def queue_step(
    queue: Mapping[str, Any],
    arrivals: Mapping[str, Any],
    service: Mapping[str, Any],
) -> Dict[str, float]:
    q = nonnegative_vector(queue)
    a = nonnegative_vector(arrivals)
    s = nonnegative_vector(service)
    classes = sorted(set(q) | set(a) | set(s))
    return {
        cls: max(q.get(cls, 0.0) - s.get(cls, 0.0), 0.0) + a.get(cls, 0.0)
        for cls in classes
    }


def progress_service_units(
    previous_units: Any,
    current_units: Any,
    *,
    allow_reset: bool = False,
) -> Dict[str, Any]:
    prev = max(0.0, _as_float(previous_units))
    curr = max(0.0, _as_float(current_units))
    if curr >= prev:
        return {
            "service_units": curr - prev,
            "service_censored": False,
            "censor_reason": "",
        }
    if allow_reset:
        return {
            "service_units": curr,
            "service_censored": False,
            "censor_reason": "progress_reset_allowed",
        }
    return {
        "service_units": 0.0,
        "service_censored": True,
        "censor_reason": "progress_counter_reset",
    }


def build_slot_record(
    *,
    slot_id: str,
    queue_start: Mapping[str, Any],
    arrivals: Mapping[str, Any],
    service: Mapping[str, Any],
    delta_s: float,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    q0 = nonnegative_vector(queue_start)
    a = nonnegative_vector(arrivals)
    s = nonnegative_vector(service)
    q1 = queue_step(q0, a, s)
    return {
        "slot_id": slot_id,
        "delta_s": max(0.0, _as_float(delta_s)),
        "queue_start": q0,
        "arrival_units": a,
        "service_units": s,
        "queue_end": q1,
        "metadata": dict(metadata or {}),
    }


def aggregate_units(
    rows: Iterable[Mapping[str, Any]],
    *,
    class_key_field: str = "class_key",
    units_field: str = "units",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        cls = str(row.get(class_key_field) or "")
        if not cls:
            continue
        out[cls] = out.get(cls, 0.0) + max(0.0, _as_float(row.get(units_field)))
    return out


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_step(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
    out = build_slot_record(
        slot_id=str(payload.get("slot_id") or "slot"),
        queue_start=payload.get("queue_start") or {},
        arrivals=payload.get("arrival_units") or payload.get("arrivals") or {},
        service=payload.get("service_units") or payload.get("service") or {},
        delta_s=_as_float(payload.get("delta_s")),
        metadata=payload.get("metadata") or {},
    )
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.slot_builder")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("step", help="Build one queue-recurrence slot record")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=_cmd_step)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
