"""Global finite-fabric perturbation calibration for measured tasksets."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

from algorithm.features import finite_feature_metric
from simulation.defaults import build_default_cache, calibrated_candidate_policy
from simulation.tasksets import TaskSetMember, taskset_by_name

from .empirical_slack_certificate import (
    _action_features,
    _action_id,
    _class_service,
    _profile_domain,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TASKSETS = (
    "q00_light_control",
    "q01_gpu_bound_compute",
    "q10_cpu_host_bound",
    "q11_cpu_gpu_coupled",
    "hybrid_research_portfolio",
)
DEFAULT_COVER_POINTS = (1, 2, 4, 8, 16, 32, 64, 128)


def build_global_fabric_cover_calibration(
    *,
    tasksets: tuple[str, ...] = DEFAULT_TASKSETS,
    cover_points: tuple[int, ...] = DEFAULT_COVER_POINTS,
) -> dict[str, Any]:
    cache = build_default_cache()
    rows = []
    for taskset_name in tasksets:
        taskset = taskset_by_name(taskset_name)
        domains = {
            member.workload_key: _profile_domain(cache, member)
            for member in taskset.members
        }
        policy = calibrated_candidate_policy(cache, taskset.workload_specs())
        selected_profiles = {
            spec.workload_key: policy.select_profile(cache, spec).profile
            for spec in taskset.workload_specs()
        }
        rows.append(_taskset_row(
            cache=cache,
            taskset_name=taskset_name,
            members=tuple(taskset.members),
            domains=domains,
            selected_profiles=selected_profiles,
            cover_points=cover_points,
        ))
    return {
        "gate": "global_fabric_cover_perturbation_calibration",
        "tasksets": list(tasksets),
        "rows": rows,
        "pass": all(bool(row.get("exact_measured_slice_usable_for_theorem")) for row in rows),
        "scope": (
            "calibrates the finite-feature metric and service Lipschitz envelope "
            "on measured Scheduleurm tasksets. The main theorem certificate uses "
            "the exact measured finite action slice with rho=0. Greedy k-center "
            "cover curves over the same finite feature metric report how much "
            "candidate-cover slack Lrho would be consumed when the candidate "
            "family is smaller than the measured full action slice."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Global Fabric-Cover Perturbation Calibration",
        "",
        "## Summary",
        "",
        "| Taskset | Actions | L | exact rho | exact Lrho | representative rho | representative Lrho | Main usable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{taskset}` | {actions} | {L:.9f} | {rho:.9f} | {Lrho:.9f} | {rrho:.9f} | {rLrho:.9f} | {usable} |".format(
                taskset=row.get("taskset"),
                actions=row.get("full_action_count", 0),
                L=float(row.get("L") or 0.0),
                rho=float(row.get("exact_cover", {}).get("rho") or 0.0),
                Lrho=float(row.get("exact_cover", {}).get("Lrho") or 0.0),
                rrho=float(row.get("representative_cover", {}).get("rho") or 0.0),
                rLrho=float(row.get("representative_cover", {}).get("Lrho") or 0.0),
                usable=str(bool(row.get("exact_measured_slice_usable_for_theorem"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Perturbation Witnesses",
        "",
    ])
    for row in report.get("rows") or []:
        worst = row.get("worst_perturbation") or {}
        lines.extend([
            f"### `{row.get('taskset')}`",
            "",
            f"- worst class: `{worst.get('workload_key', '')}`",
            f"- profiles: `{worst.get('left_profile', '')}` to `{worst.get('right_profile', '')}`",
            f"- distance: `{_fmt(worst.get('distance'))}`",
            f"- service gap: `{_fmt(worst.get('service_gap'))}`",
            f"- ratio: `{_fmt(worst.get('ratio'))}`",
            "",
        ])
    lines.extend([
        "## Greedy Candidate-Cover Curve",
        "",
        "Each row is computed on the same finite-feature metric used by the",
        "fabric-cover theorem.  The exact measured finite-slice certificate is",
        "the final row with rho=0; earlier rows quantify the support slack that",
        "a smaller candidate generator would spend.",
        "",
        "| Taskset | Candidate count | rho | Lrho | Worst uncovered action |",
        "|---|---:|---:|---:|---|",
    ])
    for row in report.get("rows") or []:
        for point in row.get("greedy_cover_curve") or []:
            lines.append(
                "| `{taskset}` | {k} | {rho:.9f} | {Lrho:.9f} | `{worst}` |".format(
                    taskset=row.get("taskset"),
                    k=point.get("candidate_count", 0),
                    rho=float(point.get("rho") or 0.0),
                    Lrho=float(point.get("Lrho") or 0.0),
                    worst=point.get("worst_action_id") or "",
                )
            )
    lines.append("")
    return "\n".join(lines)


def _taskset_row(
    *,
    cache: Any,
    taskset_name: str,
    members: tuple[TaskSetMember, ...],
    domains: Mapping[str, list[int]],
    selected_profiles: Mapping[str, int],
    cover_points: tuple[int, ...],
) -> dict[str, Any]:
    L_report = _perturbation_lipschitz(cache, members, domains)
    actions = _materialized_actions(members=members, domains=domains)
    full_count = len(actions)
    representative = _representative_cover_radius(
        cache=cache,
        members=members,
        domains=domains,
        selected_profiles=selected_profiles,
    )
    L = float(L_report.get("L") or 0.0)
    greedy_curve = _greedy_cover_curve(
        actions=actions,
        selected_action_id=_action_id(selected_profiles),
        L=L,
        cover_points=cover_points,
    )
    return {
        "taskset": taskset_name,
        "profile_domains": {key: list(value) for key, value in domains.items()},
        "selected_profiles": {key: int(value) for key, value in selected_profiles.items()},
        "full_action_count": full_count,
        "L": L,
        "exact_cover": {
            "rho": 0.0,
            "Lrho": 0.0,
            "candidate_count": full_count,
            "cover_domain": "exact_measured_finite_slice",
        },
        "representative_cover": {
            "rho": representative["rho"],
            "Lrho": L * float(representative["rho"]),
            "candidate_count": 1,
            "cover_domain": "selected_profile_representative",
            "worst_action_id": representative["worst_action_id"],
        },
        "greedy_cover_curve": greedy_curve,
        "greedy_cover_first_zero_rho_candidate_count": _first_zero_rho(greedy_curve),
        "perturbation_pair_count": L_report["pair_count"],
        "zero_metric_violation_count": L_report["zero_metric_violation_count"],
        "worst_perturbation": L_report["worst"],
        "exact_measured_slice_usable_for_theorem": (
            full_count > 0
            and L_report["pair_count"] > 0
            and L_report["zero_metric_violation_count"] == 0
        ),
    }


def _perturbation_lipschitz(
    cache: Any,
    members: tuple[TaskSetMember, ...],
    domains: Mapping[str, list[int]],
) -> dict[str, Any]:
    base = {member.workload_key: int(domains[member.workload_key][0]) for member in members}
    rows = []
    zero_violations = []
    for member in members:
        key = member.workload_key
        for left_profile, right_profile in itertools.combinations(domains[key], 2):
            left = dict(base)
            right = dict(base)
            left[key] = int(left_profile)
            right[key] = int(right_profile)
            lf = _features(left, members)
            rf = _features(right, members)
            dist = finite_feature_metric(lf, rf)
            gap = abs(
                _class_service(cache, member, int(left_profile))
                - _class_service(cache, member, int(right_profile))
            )
            ratio = gap / max(dist, 1e-9)
            row = {
                "workload_key": key,
                "left_profile": int(left_profile),
                "right_profile": int(right_profile),
                "distance": dist,
                "service_gap": gap,
                "ratio": ratio,
            }
            rows.append(row)
            if dist <= 1e-9 and gap > 1e-9:
                zero_violations.append(row)
    worst = max(rows, key=lambda row: float(row["ratio"])) if rows else {}
    return {
        "L": float(worst.get("ratio") or 0.0),
        "pair_count": len(rows),
        "zero_metric_violation_count": len(zero_violations),
        "worst": worst,
        "rows": rows,
    }


def _representative_cover_radius(
    *,
    cache: Any,
    members: tuple[TaskSetMember, ...],
    domains: Mapping[str, list[int]],
    selected_profiles: Mapping[str, int],
) -> dict[str, Any]:
    selected_features = _features(selected_profiles, members)
    max_dist = 0.0
    worst_action = ""
    keys = [member.workload_key for member in members]
    for combo in itertools.product(*(domains[key] for key in keys)):
        profiles = dict(zip(keys, (int(x) for x in combo)))
        dist = finite_feature_metric(_features(profiles, members), selected_features)
        if dist > max_dist:
            max_dist = dist
            worst_action = _action_id(profiles)
    return {"rho": max_dist, "worst_action_id": worst_action}


def _materialized_actions(
    *,
    members: tuple[TaskSetMember, ...],
    domains: Mapping[str, list[int]],
) -> list[dict[str, Any]]:
    keys = [member.workload_key for member in members]
    actions = []
    for combo in itertools.product(*(domains[key] for key in keys)):
        profiles = dict(zip(keys, (int(x) for x in combo)))
        action_id = _action_id(profiles)
        actions.append({
            "action_id": action_id,
            "profiles": profiles,
            "features": dict(_features(profiles, members)),
        })
    return actions


def _greedy_cover_curve(
    *,
    actions: list[dict[str, Any]],
    selected_action_id: str,
    L: float,
    cover_points: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not actions:
        return []
    full_count = len(actions)
    targets = sorted({
        min(full_count, max(1, int(k)))
        for k in tuple(cover_points) + (full_count,)
    })
    target_set = set(targets)
    by_id = {str(action["action_id"]): action for action in actions}
    first = by_id.get(selected_action_id, actions[0])
    centers = [first]
    selected_ids = {str(first["action_id"])}
    rows: list[dict[str, Any]] = []
    max_target = max(targets)
    while len(centers) <= max_target:
        if len(centers) in target_set:
            rows.append(_cover_row(actions=actions, centers=centers, L=L))
        if len(centers) >= max_target:
            break
        next_action = _farthest_uncovered_action(actions=actions, centers=centers, selected_ids=selected_ids)
        if next_action is None:
            break
        centers.append(next_action)
        selected_ids.add(str(next_action["action_id"]))
    return rows


def _cover_row(
    *,
    actions: list[dict[str, Any]],
    centers: list[dict[str, Any]],
    L: float,
) -> dict[str, Any]:
    rho = 0.0
    worst_action_id = ""
    for action in actions:
        dist = min(
            finite_feature_metric(action["features"], center["features"])
            for center in centers
        )
        if dist > rho:
            rho = float(dist)
            worst_action_id = str(action["action_id"])
    return {
        "candidate_count": len(centers),
        "rho": rho,
        "Lrho": float(L) * rho,
        "worst_action_id": worst_action_id,
        "center_action_ids": [str(center["action_id"]) for center in centers],
    }


def _farthest_uncovered_action(
    *,
    actions: list[dict[str, Any]],
    centers: list[dict[str, Any]],
    selected_ids: set[str],
) -> dict[str, Any] | None:
    best_action: dict[str, Any] | None = None
    best_dist = -1.0
    for action in actions:
        action_id = str(action["action_id"])
        if action_id in selected_ids:
            continue
        dist = min(
            finite_feature_metric(action["features"], center["features"])
            for center in centers
        )
        if dist > best_dist:
            best_dist = float(dist)
            best_action = action
    return best_action


def _first_zero_rho(curve: list[Mapping[str, Any]]) -> int | None:
    for point in curve:
        try:
            if float(point.get("rho") or 0.0) <= 1e-12:
                return int(point.get("candidate_count") or 0)
        except (TypeError, ValueError):
            continue
    return None


def _features(profiles: Mapping[str, int], members: tuple[TaskSetMember, ...]) -> Mapping[str, Any]:
    return _action_features(_action_id(profiles), profiles, members)


def _fmt(value: Any) -> str:
    try:
        out = float(value)
        return f"{out:.9f}" if math.isfinite(out) else "NA"
    except (TypeError, ValueError):
        return "NA"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_global_fabric_cover_calibration(
        tasksets=tuple(x.strip() for x in args.tasksets.split(",") if x.strip()),
        cover_points=tuple(int(x.strip()) for x in args.cover_points.split(",") if x.strip()),
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.global_fabric_cover_calibration")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Calibrate global measured finite-fabric cover")
    build.add_argument("--tasksets", default=",".join(DEFAULT_TASKSETS))
    build.add_argument("--cover-points", default=",".join(str(x) for x in DEFAULT_COVER_POINTS))
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "global_fabric_cover_calibration.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "global_fabric_cover_calibration.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
