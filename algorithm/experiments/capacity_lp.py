"""Capacity slack LP and drift-margin accounting."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def service_vector(action: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("service_vector", "mean_service", "mu", "lower_service"):
        value = action.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def solve_capacity_slack(
    actions: Sequence[Mapping[str, Any]],
    lam: Mapping[str, Any],
) -> Dict[str, Any]:
    """Solve max delta s.t. lambda_i + delta <= sum_a x_a mu_i(a)."""
    if not actions:
        return {
            "status": "no_actions",
            "delta": None,
            "usable_for_theorem": False,
            "mix": {},
        }
    classes = sorted(set(str(k) for k in lam.keys()) | {
        str(k) for action in actions for k in service_vector(action).keys()
    })
    n = len(actions)
    try:
        from scipy.optimize import linprog
    except Exception as e:
        return {
            "status": f"scipy_unavailable:{type(e).__name__}",
            "delta": None,
            "usable_for_theorem": False,
            "mix": {},
        }

    c = [0.0] * n + [-1.0]
    a_ub = []
    b_ub = []
    for cls in classes:
        row = [-_as_float(service_vector(action).get(cls)) for action in actions]
        row.append(1.0)
        a_ub.append(row)
        b_ub.append(-_as_float(lam.get(cls)))
    a_eq = [[1.0] * n + [0.0]]
    b_eq = [1.0]
    # The theorem consumes a nonnegative capacity slack.  If the arrival vector
    # is outside the convex service hull, the LP must fail rather than certify
    # a negative `delta`.
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return {
            "status": res.message,
            "delta": None,
            "usable_for_theorem": False,
            "mix": {},
            "classes": classes,
        }
    mix = {
        str(actions[i].get("action_id") or actions[i].get("id") or f"action_{i}"): float(res.x[i])
        for i in range(n)
        if float(res.x[i]) > 1e-10
    }
    delta = float(res.x[-1])
    return {
        "status": "optimal",
        "delta": delta,
        "usable_for_theorem": delta > 0.0,
        "mix": mix,
        "classes": classes,
    }


def drift_margin_certificate(
    *,
    delta: float,
    L: float,
    rho: float,
    epsilon_est: float,
    beta: float,
    alpha1: float,
    B: float,
    P0: float,
    alpha0: float,
    alpha: float = 1.0,
    N: int | None = None,
) -> Dict[str, Any]:
    eta = _as_float(delta) - (
        _as_float(L) * _as_float(rho)
        + _as_float(epsilon_est)
        + _as_float(beta)
        + _as_float(alpha1)
    )
    additive = _as_float(B) + _as_float(P0) + _as_float(alpha0)
    threshold = None
    if eta > 0:
        raw_threshold = (additive + max(0.0, _as_float(alpha))) / eta
        threshold = int(math.ceil(raw_threshold - 1e-12))
        threshold = max(0, threshold)
    passed = eta > 0 and (N is None or int(N) >= int(threshold or 0))
    return {
        "delta": _as_float(delta),
        "L": _as_float(L),
        "rho": _as_float(rho),
        "Lrho": _as_float(L) * _as_float(rho),
        "epsilon_est": _as_float(epsilon_est),
        "beta": _as_float(beta),
        "alpha1": _as_float(alpha1),
        "eta": eta,
        "B": _as_float(B),
        "P0": _as_float(P0),
        "alpha0": _as_float(alpha0),
        "alpha": max(0.0, _as_float(alpha)),
        "drift_additive_constant": additive,
        "finite_set_threshold_N": threshold,
        "N_claimed": N,
        "usable_for_theorem": passed,
        "condition": "eta > 0 and B+P0+alpha0+alpha <= eta*N",
    }


def _cmd_solve(args: argparse.Namespace) -> int:
    payload = _json(Path(args.input).expanduser()) or {}
    actions = payload.get("actions") or []
    lam = payload.get("lambda") or payload.get("lam") or {}
    out = solve_capacity_slack(actions, lam)
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0 if out.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.capacity_lp")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="Solve finite-action capacity slack LP")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=_cmd_solve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
