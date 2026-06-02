"""Queue-scaled penalty envelope fitting.

The Lean theorem consumes a deterministic bound of the form

    penalty_units(k) <= P0 + beta * ||Q(k)||_1.

This module fits exactly that finite-sample envelope.  It is intentionally
conservative: the returned constants cover every supplied theorem-grade row,
and any violation is reported explicitly rather than hidden in a regression
residual.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _jsonl_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fit_nonnegative_linear_envelope(
    points: Iterable[Tuple[float, float]],
    *,
    intercept_name: str,
    slope_name: str,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Fit `y <= intercept + slope * x` with nonnegative constants.

    The intercept covers all zero-backlog rows.  The slope is then the maximum
    remaining ratio.  This is not a least-squares fit; it is the finite
    deterministic certificate required by the drift theorem.
    """
    clean: list[tuple[float, float]] = []
    for x, y in points:
        q = max(0.0, _as_float(x))
        val = max(0.0, _as_float(y))
        clean.append((q, val))

    intercept = 0.0
    for q, val in clean:
        if q <= eps:
            intercept = max(intercept, val)

    slope = 0.0
    for q, val in clean:
        if q > eps:
            slope = max(slope, max(0.0, val - intercept) / q)

    max_violation = 0.0
    worst: Dict[str, Any] | None = None
    for idx, (q, val) in enumerate(clean):
        bound = intercept + slope * q
        violation = max(0.0, val - bound)
        if violation > max_violation:
            max_violation = violation
            worst = {"index": idx, "q_norm": q, "value": val, "bound": bound}

    return {
        intercept_name: intercept,
        slope_name: slope,
        "point_count": len(clean),
        "max_residual_violation": max_violation,
        "worst_violation": worst,
        "usable_for_theorem": bool(clean) and max_violation <= 1e-9,
        "fit_method": "finite_max_envelope",
    }


def fit_penalty_envelope(
    rows: Iterable[Mapping[str, Any]],
    *,
    q_key: str = "q_norm",
    penalty_key: str = "penalty_units",
) -> Dict[str, Any]:
    points = [
        (_as_float(row.get(q_key)), _as_float(row.get(penalty_key)))
        for row in rows
        if row.get("usable_for_theorem", True)
    ]
    out = fit_nonnegative_linear_envelope(
        points, intercept_name="P0", slope_name="beta")
    out["q_key"] = q_key
    out["penalty_key"] = penalty_key
    return out


def _cmd_fit(args: argparse.Namespace) -> int:
    rows = _jsonl_rows(Path(args.input).expanduser())
    out = fit_penalty_envelope(rows, q_key=args.q_key, penalty_key=args.penalty_key)
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0 if out.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.penalty_fit")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit", help="Fit P0,beta from penalty event rows")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--q-key", default="q_norm")
    s.add_argument("--penalty-key", default="penalty_units")
    s.set_defaults(func=_cmd_fit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
