"""Service lower-bound calibration for theorem-facing experiments."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
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


def _json_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def service_bucket_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("class_key") or ""),
        str(row.get("regime_key") or ""),
        str(row.get("feature_bucket") or row.get("candidate_bucket") or ""),
    )


def _service_units(row: Mapping[str, Any]) -> float:
    for key in (
        "service_units_per_delta_ref",
        "mean_service_per_delta_ref",
        "service_units",
        "service",
    ):
        if key in row:
            return max(0.0, _as_float(row.get(key)))
    return 0.0


def empirical_bernstein_lcb(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    value_range: float | None = None,
) -> tuple[float, float, float]:
    vals = [max(0.0, _as_float(v)) for v in values]
    if not vals:
        return 0.0, 0.0, 0.0
    n = len(vals)
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, mean
    variance = statistics.pvariance(vals)
    delta = max(1e-12, min(0.5, 1.0 - float(confidence)))
    radius = math.sqrt(2.0 * variance * math.log(2.0 / delta) / n)
    if value_range is not None and value_range > 0:
        radius += 3.0 * value_range * math.log(2.0 / delta) / n
    return mean, radius, max(0.0, mean - radius)


def calibrate_service_lower_bounds(
    samples: Iterable[Mapping[str, Any]],
    *,
    min_samples: int = 5,
    confidence: float = 0.95,
    max_service_units: float | None = None,
) -> list[Dict[str, Any]]:
    grouped: dict[Tuple[str, str, str], list[float]] = defaultdict(list)
    for row in samples:
        grouped[service_bucket_key(row)].append(_service_units(row))

    out: list[Dict[str, Any]] = []
    for (class_key, regime_key, feature_bucket), values in sorted(grouped.items()):
        mean, radius, lcb = empirical_bernstein_lcb(
            values, confidence=confidence, value_range=max_service_units)
        usable = len(values) >= max(1, int(min_samples))
        out.append({
            "class_key": class_key,
            "regime_key": regime_key,
            "feature_bucket": feature_bucket,
            "n_samples": len(values),
            "mean_service_per_delta_ref": mean,
            "lcb_service_per_delta_ref": lcb,
            "lcb_radius": radius,
            "lcb_method": "empirical_bernstein",
            "confidence": confidence,
            "usable_for_theorem": usable,
            "reason": "" if usable else "insufficient_samples",
        })
    return out


def lower_bound_lookup(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for row in rows:
        if not row.get("usable_for_theorem", True):
            continue
        out[service_bucket_key(row)] = max(0.0, _as_float(row.get("lcb_service_per_delta_ref")))
    return out


def estimate_epsilon_est(
    validation_rows: Iterable[Mapping[str, Any]],
    lower_bounds: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    lower_rows = list(lower_bounds)
    lower = lower_bound_lookup(lower_rows)
    all_lower_keys = {service_bucket_key(row) for row in lower_rows}
    grouped: dict[Tuple[str, str, str], list[float]] = defaultdict(list)
    for row in validation_rows:
        grouped[service_bucket_key(row)].append(_service_units(row))

    residuals: list[Dict[str, Any]] = []
    epsilon = 0.0
    missing: list[Tuple[str, str, str]] = []
    unusable: list[Tuple[str, str, str]] = []
    for key, values in sorted(grouped.items()):
        empirical = sum(values) / max(1, len(values))
        if key not in lower:
            if key in all_lower_keys:
                unusable.append(key)
            else:
                missing.append(key)
            residual = empirical
            lcb = 0.0
        else:
            lcb = lower[key]
            residual = max(0.0, empirical - lcb)
        epsilon = max(epsilon, residual)
        residuals.append({
            "class_key": key[0],
            "regime_key": key[1],
            "feature_bucket": key[2],
            "validation_samples": len(values),
            "empirical_service_per_delta_ref": empirical,
            "lcb_service_per_delta_ref": lcb,
            "positive_residual": residual,
        })
    return {
        "epsilon_est": epsilon,
        "validation_bucket_count": len(grouped),
        "missing_lower_bound_count": len(missing),
        "missing_lower_bounds": [
            {"class_key": k[0], "regime_key": k[1], "feature_bucket": k[2]}
            for k in missing[:100]
        ],
        "unusable_lower_bound_count": len(unusable),
        "unusable_lower_bounds": [
            {"class_key": k[0], "regime_key": k[1], "feature_bucket": k[2]}
            for k in unusable[:100]
        ],
        "usable_for_theorem": bool(grouped) and not missing and not unusable,
        "residuals": residuals,
    }


def _cmd_calibrate(args: argparse.Namespace) -> int:
    rows = _json_rows(Path(args.input).expanduser())
    out = calibrate_service_lower_bounds(
        rows,
        min_samples=args.min_samples,
        confidence=args.confidence,
        max_service_units=args.max_service_units,
    )
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0 if out else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.service_model")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("calibrate", help="Build service LCB rows")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--min-samples", type=int, default=5)
    s.add_argument("--confidence", type=float, default=0.95)
    s.add_argument("--max-service-units", type=float, default=None)
    s.set_defaults(func=_cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
