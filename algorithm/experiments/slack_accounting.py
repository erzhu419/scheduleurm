"""Consolidated drift-margin slack accounting.

The paper-facing stability condition is

    delta > L * rho + epsilon_est + beta + alpha1.

Separate calibration modules estimate each component.  This module combines
their JSON reports into a single conservative certificate that can be shown in a
paper table or used as an experiment pass/fail gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .capacity_lp import drift_margin_certificate


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_slack_certificate(
    *,
    fabric: Mapping[str, Any] | None = None,
    service: Mapping[str, Any] | None = None,
    penalty: Mapping[str, Any] | None = None,
    oracle: Mapping[str, Any] | None = None,
    capacity: Mapping[str, Any] | None = None,
    moment: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    alpha: float = 1.0,
    N: int | None = None,
) -> dict[str, Any]:
    fabric = dict(fabric or {})
    service = dict(service or {})
    penalty = dict(penalty or {})
    oracle = dict(oracle or {})
    capacity = dict(capacity or {})
    moment = dict(moment or {})
    overrides = dict(overrides or {})

    L = _pick("L", overrides, fabric)
    rho = _pick("rho", overrides, fabric)
    epsilon_est = _pick("epsilon_est", overrides, service)
    beta = _pick("beta", overrides, penalty)
    alpha1 = _pick("alpha1", overrides, oracle)
    delta = _pick("delta", overrides, capacity)
    B = _pick("B", overrides, moment)
    P0 = _pick("P0", overrides, penalty)
    alpha0 = _pick("alpha0", overrides, oracle)

    certificate = drift_margin_certificate(
        delta=delta,
        L=L,
        rho=rho,
        epsilon_est=epsilon_est,
        beta=beta,
        alpha1=alpha1,
        B=B,
        P0=P0,
        alpha0=alpha0,
        alpha=alpha,
        N=N,
    )
    component_status = _component_status(
        fabric=fabric,
        service=service,
        penalty=penalty,
        oracle=oracle,
        capacity=capacity,
        moment=moment,
        overrides=overrides,
    )
    slack_consumed = certificate["Lrho"] + epsilon_est + beta + alpha1
    certificate.update(
        {
            "slack_consumed": slack_consumed,
            "slack_ratio": slack_consumed / delta if delta > 0 else None,
            "component_status": component_status,
            "all_components_theorem_usable": all(
                row["usable_for_theorem"] for row in component_status
            ),
            "paper_condition": "delta > Lrho + epsilon_est + beta + alpha1",
        }
    )
    certificate["usable_for_theorem"] = bool(
        certificate["usable_for_theorem"]
        and certificate["all_components_theorem_usable"]
    )
    return certificate


def markdown_slack_table(cert: Mapping[str, Any]) -> str:
    rows = [
        ("delta", cert.get("delta"), "capacity slack"),
        ("L", cert.get("L"), "fabric service Lipschitz envelope"),
        ("rho", cert.get("rho"), "candidate cover radius"),
        ("Lrho", cert.get("Lrho"), "candidate support loss"),
        ("epsilon_est", cert.get("epsilon_est"), "lower-service estimation loss"),
        ("beta", cert.get("beta"), "queue-scaled penalty slope"),
        ("alpha1", cert.get("alpha1"), "queue-scaled oracle error"),
        ("eta", cert.get("eta"), "remaining drift margin"),
        ("B", cert.get("B"), "second-order drift bound"),
        ("P0", cert.get("P0"), "fixed penalty term"),
        ("alpha0", cert.get("alpha0"), "fixed oracle-error term"),
        ("finite_set_threshold_N", cert.get("finite_set_threshold_N"), "Foster finite-set threshold"),
    ]
    out = [
        "| Quantity | Value | Meaning |",
        "|---|---:|---|",
    ]
    for name, value, meaning in rows:
        out.append(f"| `{name}` | {_fmt(value)} | {meaning} |")
    out.extend(
        [
            "",
            "| Component | Usable | Source status |",
            "|---|---:|---|",
        ]
    )
    for row in cert.get("component_status", []):
        out.append(
            "| `{name}` | {usable} | {status} |".format(
                name=row["name"],
                usable=str(bool(row["usable_for_theorem"])).lower(),
                status=row["status"],
            )
        )
    return "\n".join(out) + "\n"


def _pick(name: str, overrides: Mapping[str, Any], *sources: Mapping[str, Any]) -> float:
    if name in overrides:
        return _as_float(overrides.get(name))
    for source in sources:
        if name in source:
            return _as_float(source.get(name))
    return 0.0


def _component_status(**components: Mapping[str, Any]) -> list[dict[str, Any]]:
    overrides = components.get("overrides") or {}
    override_keys = {
        "fabric": ("L", "rho"),
        "service": ("epsilon_est",),
        "penalty": ("P0", "beta"),
        "oracle": ("alpha0", "alpha1"),
        "capacity": ("delta",),
        "moment": ("B",),
    }
    rows = []
    for name, payload in components.items():
        if name == "overrides":
            continue
        if payload:
            usable = bool(payload.get("usable_for_theorem", True))
            status = str(payload.get("status") or payload.get("fit_method") or "provided")
        elif all(key in overrides for key in override_keys.get(name, ())):
            usable = True
            status = "provided_by_override"
        else:
            usable = False
            status = "missing"
        rows.append({"name": name, "usable_for_theorem": usable, "status": status})
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    val = _as_float(value)
    if abs(val) >= 1000:
        return f"{val:.3f}"
    if abs(val) >= 1:
        return f"{val:.6f}"
    return f"{val:.9f}"


def _cmd_build(args: argparse.Namespace) -> int:
    overrides = {
        key: value
        for key, value in {
            "delta": args.delta,
            "L": args.L,
            "rho": args.rho,
            "epsilon_est": args.epsilon_est,
            "beta": args.beta,
            "alpha1": args.alpha1,
            "B": args.B,
            "P0": args.P0,
            "alpha0": args.alpha0,
        }.items()
        if value is not None
    }
    cert = build_slack_certificate(
        fabric=_json(args.fabric),
        service=_json(args.service),
        penalty=_json(args.penalty),
        oracle=_json(args.oracle),
        capacity=_json(args.capacity),
        moment=_json(args.moment),
        overrides=overrides,
        alpha=args.alpha,
        N=args.N,
    )
    _write_json(args.output, cert)
    if args.markdown_output:
        Path(args.markdown_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).expanduser().write_text(
            markdown_slack_table(cert),
            encoding="utf-8",
        )
    print(args.output)
    return 0 if cert.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.slack_accounting")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build", help="Build consolidated drift-margin certificate")
    s.add_argument("--output", required=True)
    s.add_argument("--markdown-output", default="")
    s.add_argument("--fabric", default="")
    s.add_argument("--service", default="")
    s.add_argument("--penalty", default="")
    s.add_argument("--oracle", default="")
    s.add_argument("--capacity", default="")
    s.add_argument("--moment", default="")
    s.add_argument("--delta", type=float, default=None)
    s.add_argument("--L", type=float, default=None)
    s.add_argument("--rho", type=float, default=None)
    s.add_argument("--epsilon-est", type=float, default=None)
    s.add_argument("--beta", type=float, default=None)
    s.add_argument("--alpha1", type=float, default=None)
    s.add_argument("--B", type=float, default=None)
    s.add_argument("--P0", type=float, default=None)
    s.add_argument("--alpha0", type=float, default=None)
    s.add_argument("--alpha", type=float, default=1.0)
    s.add_argument("--N", type=int, default=None)
    s.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
