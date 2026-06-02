"""Calibration report helpers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping


def pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _finite_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _display(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, Mapping):
        if not value:
            return "NA"
        return ",".join(f"{k}:{v}" for k, v in sorted(value.items()))
    val = _finite_float(value)
    if val is None:
        return str(value)
    return f"{val:.10g}"


def _status(value: Any, *, positive: bool = False, nonnegative: bool = True) -> str:
    if isinstance(value, Mapping):
        if not value:
            return "EMPIRICAL-OPEN"
        vals = [_finite_float(v) for v in value.values()]
        if any(v is None for v in vals):
            return "EMPIRICAL-OPEN"
        if positive:
            return pass_fail(all(float(v) > 0.0 for v in vals if v is not None))
        if nonnegative:
            return pass_fail(all(float(v) >= 0.0 for v in vals if v is not None))
        return "PASS"
    val = _finite_float(value)
    if val is None:
        return "EMPIRICAL-OPEN"
    if positive:
        return pass_fail(val > 0.0)
    if nonnegative:
        return pass_fail(val >= 0.0)
    return "PASS"


def calibration_table(constants: Mapping[str, Any]) -> str:
    """Build the manual's theorem-constant table shape."""
    lrho = constants.get("Lrho")
    if lrho is None and constants.get("L") is not None and constants.get("rho") is not None:
        L = _finite_float(constants.get("L"))
        rho = _finite_float(constants.get("rho"))
        if L is not None and rho is not None:
            lrho = L * rho
    rows = [
        ("Amax_i", constants.get("Amax_i"), "work units/slot", "slot_builder", "second moment bound", _status(constants.get("Amax_i"))),
        ("Smax_i", constants.get("Smax_i"), "work units/slot", "slot_builder", "second moment bound", _status(constants.get("Smax_i"))),
        ("B", constants.get("B"), "work units squared", "slot_builder", "drift constant", _status(constants.get("B"))),
        ("L", constants.get("L"), "service per metric", "fabric_metric", "fabric Lipschitz", _status(constants.get("L"))),
        ("rho", constants.get("rho"), "metric distance", "fabric_metric", "candidate cover radius", _status(constants.get("rho"))),
        ("Lrho", lrho, "service units/slot", "fabric_metric", "candidate support loss", _status(lrho)),
        ("epsilon_est", constants.get("epsilon_est"), "service units/slot", "service_model", "lower-service estimation loss", _status(constants.get("epsilon_est"))),
        ("P0", constants.get("P0"), "service units/slot", "penalty_fit", "fixed penalty constant", _status(constants.get("P0"))),
        ("beta", constants.get("beta"), "service units per backlog unit", "penalty_fit", "queue-scaled penalty", _status(constants.get("beta"))),
        ("alpha0", constants.get("alpha0"), "score units", "oracle_audit", "fixed oracle error", _status(constants.get("alpha0"))),
        ("alpha1", constants.get("alpha1"), "score per backlog unit", "oracle_audit", "queue-scaled oracle error", _status(constants.get("alpha1"))),
        ("delta", constants.get("delta"), "service units/slot", "capacity_lp", "full capacity slack", _status(constants.get("delta"), positive=True)),
        ("eta", constants.get("eta"), "service units/slot", "capacity_lp", "final drift margin", _status(constants.get("eta"), positive=True)),
    ]
    lines = [
        "| Quantity | Value | Unit | Source artifact | Theorem role | Status |",
        "|---|---:|---|---|---|---|",
    ]
    for name, value, unit, source, role, status in rows:
        lines.append(f"| {name} | {_display(value)} | {unit} | {source} | {role} | {status} |")
    return "\n".join(lines) + "\n"


def summarize_certificate(constants: Mapping[str, Any]) -> Dict[str, Any]:
    eta = _finite_float(constants.get("eta"))
    table = calibration_table(constants)
    empirical_open = "EMPIRICAL-OPEN" in table
    theorem_usable = bool(constants.get("usable_for_theorem")) and eta is not None and eta > 0 and not empirical_open
    if eta is not None and eta <= 0.0:
        status = "FAIL"
    elif theorem_usable:
        status = "PASS"
    elif empirical_open:
        status = "EMPIRICAL-OPEN"
    else:
        status = "FAIL"
    return {
        "usable_for_theorem": theorem_usable,
        "eta": eta,
        "finite_set_threshold_N": constants.get("finite_set_threshold_N"),
        "status": status,
        "calibration_table_md": table,
    }


def _json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("report input must be a JSON object")
    return dict(data)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    constants = _json(Path(args.input).expanduser())
    summary = summarize_certificate(constants)
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary["calibration_table_md"], encoding="utf-8")
    if args.summary_output:
        _write_json(Path(args.summary_output).expanduser(), summary)
    print(args.output)
    return 0 if summary.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.report")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build", help="Build theorem calibration table and summary")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--summary-output", default="")
    s.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
