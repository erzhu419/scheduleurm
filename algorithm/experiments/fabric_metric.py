"""Fabric metric calibration and candidate-cover audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from algorithm.features import finite_feature_metric


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


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_id(action: Mapping[str, Any], fallback: int = 0) -> str:
    return str(action.get("action_id") or action.get("id") or f"action_{fallback}")


def action_features(action: Mapping[str, Any]) -> Mapping[str, Any]:
    value = action.get("features")
    if isinstance(value, Mapping):
        return value
    value = action.get("fabric_features")
    if isinstance(value, Mapping):
        return value
    return action


def service_vector(action: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("service_vector", "mean_service", "mu", "lower_service"):
        value = action.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _regime_key(features: Mapping[str, Any]) -> Any:
    finite = features.get("finite")
    if isinstance(finite, Mapping) and finite.get("regime_key") is not None:
        return finite.get("regime_key")
    return features.get("regime_key")


def audit_cover(
    full_actions: Sequence[Mapping[str, Any]],
    candidate_actions: Sequence[Mapping[str, Any]],
    *,
    rho: float | None = None,
    cover_domain: str = "sampled_feasible",
) -> Dict[str, Any]:
    witnesses: list[Dict[str, Any]] = []
    max_min = 0.0
    for idx, full in enumerate(full_actions):
        if not candidate_actions:
            nearest = None
            min_dist = math.inf
        else:
            pairs = [
                (finite_feature_metric(action_features(full), action_features(cand)), cand_idx, cand)
                for cand_idx, cand in enumerate(candidate_actions)
            ]
            min_dist, cand_idx, nearest_action = min(pairs, key=lambda item: item[0])
            nearest = action_id(nearest_action, cand_idx)
        max_min = max(max_min, min_dist)
        witnesses.append({
            "full_action_id": action_id(full, idx),
            "nearest_candidate_id": nearest,
            "min_distance": min_dist,
            "covered_by_rho": None if rho is None else min_dist <= rho + 1e-12,
        })
    usable = bool(full_actions) and bool(candidate_actions)
    if rho is not None:
        usable = usable and max_min <= rho + 1e-12
    return {
        "cover_domain": cover_domain,
        "full_count": len(full_actions),
        "candidate_count": len(candidate_actions),
        "rho_sample": max_min if math.isfinite(max_min) else None,
        "rho_claimed": rho,
        "usable_for_theorem": usable,
        "witnesses": witnesses,
    }


def audit_statewise_cover(
    slots: Iterable[Mapping[str, Any]],
    *,
    rho: float | None = None,
    cover_domain: str = "statewise_uniform",
) -> Dict[str, Any]:
    slot_reports = []
    max_rho = 0.0
    usable = True
    for slot in slots:
        report = audit_cover(
            list(slot.get("full_actions") or []),
            list(slot.get("candidate_actions") or []),
            rho=rho,
            cover_domain=cover_domain,
        )
        report["slot_id"] = slot.get("slot_id")
        if report["rho_sample"] is not None:
            max_rho = max(max_rho, _as_float(report["rho_sample"]))
        usable = usable and bool(report.get("usable_for_theorem"))
        slot_reports.append(report)
    return {
        "cover_domain": cover_domain,
        "slot_count": len(slot_reports),
        "rho_sample": max_rho,
        "rho_claimed": rho,
        "usable_for_theorem": bool(slot_reports) and usable,
        "slots": slot_reports,
    }


def calibrate_lipschitz(
    actions: Sequence[Mapping[str, Any]],
    *,
    d_min: float = 1e-9,
    same_regime_only: bool = True,
) -> Dict[str, Any]:
    ratios: list[Dict[str, Any]] = []
    zero_metric_violations: list[Dict[str, Any]] = []
    for i, left in enumerate(actions):
        for j in range(i + 1, len(actions)):
            right = actions[j]
            lf = action_features(left)
            rf = action_features(right)
            if same_regime_only and _regime_key(lf) != _regime_key(rf):
                continue
            dist = finite_feature_metric(lf, rf)
            denom = max(dist, d_min)
            lsvc = service_vector(left)
            rsvc = service_vector(right)
            for cls in sorted(set(lsvc) | set(rsvc)):
                gap = abs(_as_float(lsvc.get(cls)) - _as_float(rsvc.get(cls)))
                ratio = gap / denom
                ratios.append({
                    "left_action_id": action_id(left, i),
                    "right_action_id": action_id(right, j),
                    "class_key": cls,
                    "distance": dist,
                    "service_gap": gap,
                    "ratio": ratio,
                })
                if dist <= d_min and gap > 1e-9:
                    zero_metric_violations.append(ratios[-1])
    L = max((row["ratio"] for row in ratios), default=0.0)
    return {
        "L": L,
        "pair_count": len(ratios),
        "d_min": d_min,
        "same_regime_only": same_regime_only,
        "zero_metric_violation_count": len(zero_metric_violations),
        "zero_metric_violations": zero_metric_violations[:100],
        "usable_for_theorem": bool(ratios) and not zero_metric_violations,
    }


def calibrate_cover_and_lipschitz(
    full_actions: Sequence[Mapping[str, Any]],
    candidate_actions: Sequence[Mapping[str, Any]],
    *,
    rho: float | None = None,
    cover_domain: str = "sampled_feasible",
    d_min: float = 1e-9,
) -> Dict[str, Any]:
    cover = audit_cover(full_actions, candidate_actions, rho=rho, cover_domain=cover_domain)
    lip = calibrate_lipschitz(full_actions, d_min=d_min)
    L = _as_float(lip.get("L"))
    rho_sample = _as_float(cover.get("rho_sample"))
    rho_cert = _as_float(rho if rho is not None else rho_sample)
    return {
        "L": L,
        "rho": rho_cert,
        "Lrho": L * rho_cert,
        "cover": cover,
        "lipschitz": lip,
        "usable_for_theorem": bool(cover.get("usable_for_theorem")) and bool(lip.get("usable_for_theorem")),
    }


def _cmd_cover(args: argparse.Namespace) -> int:
    full = _json_rows(Path(args.full).expanduser())
    cand = _json_rows(Path(args.candidates).expanduser())
    out = audit_cover(full, cand, rho=args.rho, cover_domain=args.cover_domain)
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0 if out.get("usable_for_theorem") else 2


def _cmd_calibrate(args: argparse.Namespace) -> int:
    full = _json_rows(Path(args.full).expanduser())
    cand_path = Path(args.candidates).expanduser() if args.candidates else None
    cand = _json_rows(cand_path) if cand_path else list(full)
    out = calibrate_cover_and_lipschitz(
        full,
        cand,
        rho=args.rho,
        cover_domain=args.cover_domain,
        d_min=args.d_min,
    )
    _write_json(Path(args.output).expanduser(), out)
    print(args.output)
    return 0 if out.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.fabric_metric")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cover", help="Audit candidate cover radius rho")
    s.add_argument("--full", required=True)
    s.add_argument("--candidates", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--rho", type=float, default=None)
    s.add_argument("--cover-domain", default="sampled_feasible")
    s.set_defaults(func=_cmd_cover)
    s = sub.add_parser("calibrate", help="Calibrate rho cover and L Lipschitz constants")
    s.add_argument("--full", required=True)
    s.add_argument("--candidates", default="")
    s.add_argument("--output", required=True)
    s.add_argument("--rho", type=float, default=None)
    s.add_argument("--cover-domain", default="sampled_feasible")
    s.add_argument("--d-min", type=float, default=1e-9)
    s.set_defaults(func=_cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
