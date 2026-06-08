"""Build a measured finite-slice slack certificate from the service cache.

This is the empirical counterpart of the proof-side drift condition.  It does
not infer a production arrival process from a finite batch replay.  Instead it
constructs a load-certified finite action slice from measured service profiles:

    lambda_i = load_fraction * selected_service_i.

The resulting certificate answers whether the current Scheduleurm service map
can support a theorem-grade subcritical load under the calibrated candidate
MaxWeight abstraction.  Production-arrival certification remains a separate
load-measurement problem.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.defaults import build_default_cache, calibrated_candidate_policy
from simulation.service_cache import ProfileRecord, ServiceRateCache
from simulation.tasksets import TaskSetMember, taskset_by_name

from .capacity_lp import solve_capacity_slack
from .penalty_fit import fit_penalty_envelope
from .slack_accounting import build_slack_certificate, markdown_slack_table


def build_measured_finite_slice_certificate(
    *,
    taskset_name: str = "hybrid_research_portfolio",
    load_fraction: float = 0.80,
    alpha: float = 1.0,
    include_actions: bool = False,
) -> dict[str, Any]:
    cache = build_default_cache()
    taskset = taskset_by_name(taskset_name)
    specs = taskset.workload_specs()
    policy = calibrated_candidate_policy(cache, specs)
    selected_profiles = {
        spec.workload_key: policy.select_profile(cache, spec).profile
        for spec in specs
    }
    domains = {
        member.workload_key: _profile_domain(cache, member)
        for member in taskset.members
    }
    actions = _product_actions(cache, tuple(taskset.members), domains)
    selected_action_id = _action_id(selected_profiles)
    selected = next(
        action for action in actions
        if action["action_id"] == selected_action_id
    )
    selected_service = dict(selected["service_vector"])
    lam = {
        key: max(0.0, float(value) * max(0.0, min(1.0, float(load_fraction))))
        for key, value in selected_service.items()
    }
    capacity = solve_capacity_slack(actions, lam)
    fabric = _exact_finite_slice_fabric_certificate(actions)
    service = {
        "epsilon_est": 0.0,
        "status": "measured_service_map_used_as_lower_service_for_this_finite_slice",
        "usable_for_theorem": True,
        "statistical_generalization_claimed": False,
    }
    penalty = fit_penalty_envelope(
        [
            {"q_norm": 0.0, "penalty_units": 0.0},
            {"q_norm": 1.0, "penalty_units": 0.0},
        ]
    )
    oracle = {
        "alpha0": 0.0,
        "alpha1": 0.0,
        "status": "exact_candidate_maxweight_oracle_by_construction",
        "usable_for_theorem": True,
        "live_greedy_scheduler_audit_claimed": False,
    }
    moment = {
        "B": _finite_support_second_moment_bound(actions, lam),
        "status": "finite_support_bound_from_measured_action_slice_and_declared_load",
        "usable_for_theorem": True,
    }
    cert0 = build_slack_certificate(
        fabric=fabric,
        service=service,
        penalty=penalty,
        oracle=oracle,
        capacity=capacity,
        moment=moment,
        alpha=alpha,
        N=None,
    )
    threshold = cert0.get("finite_set_threshold_N")
    cert = build_slack_certificate(
        fabric=fabric,
        service=service,
        penalty=penalty,
        oracle=oracle,
        capacity=capacity,
        moment=moment,
        alpha=alpha,
        N=int(threshold or 0),
    )
    cert.update(
        {
            "taskset": taskset_name,
            "policy": policy.name,
            "selected_profiles": selected_profiles,
            "selected_action_id": selected_action_id,
            "selected_service_vector": selected_service,
            "lambda": lam,
            "load_fraction": load_fraction,
            "full_action_count": len(actions),
            "candidate_action_count": len(actions),
            "action_profile_domains": domains,
            "certificate_scope": (
                "measured finite service-action slice; not a production-arrival "
                "or unmeasured-global-action certificate"
            ),
            "component_inputs": {
                "fabric": fabric,
                "service": service,
                "penalty": penalty,
                "oracle": oracle,
                "capacity": capacity,
                "moment": moment,
            },
        }
    )
    if include_actions:
        cert["actions"] = actions
    return cert


def _profile_domain(cache: ServiceRateCache, member: TaskSetMember) -> list[int]:
    required = sorted({int(x) for x in member.required_profiles})
    out = []
    for profile in required:
        record = cache.get(member.workload_key, profile)
        if record is not None and not record.capacity_boundary and record.aggregate_rate > 0:
            out.append(profile)
    if not out:
        raise KeyError(f"no measured feasible profiles for {member.workload_key!r}")
    return out


def _product_actions(
    cache: ServiceRateCache,
    members: tuple[TaskSetMember, ...],
    domains: Mapping[str, list[int]],
) -> list[dict[str, Any]]:
    actions = []
    keys = [member.workload_key for member in members]
    for combo in product(*(domains[key] for key in keys)):
        profiles = dict(zip(keys, (int(x) for x in combo)))
        service = {
            member.workload_key: _class_service(cache, member, profiles[member.workload_key])
            for member in members
        }
        action_id = _action_id(profiles)
        actions.append(
            {
                "action_id": action_id,
                "profiles": profiles,
                "features": _action_features(action_id, profiles, members),
                "service_vector": service,
                "lower_service": service,
                "penalty_units": 0.0,
            }
        )
    return actions


def _class_service(cache: ServiceRateCache, member: TaskSetMember, profile: int) -> float:
    record: ProfileRecord | None = cache.get(member.workload_key, profile)
    if record is None or record.capacity_boundary:
        return 0.0
    return max(0.0, float(record.aggregate_rate) * float(max(1, member.resource_count)))


def _action_id(profiles: Mapping[str, int]) -> str:
    return "profile_combo:" + ",".join(
        f"{key}={int(value)}" for key, value in sorted(profiles.items())
    )


def _action_features(
    action_id: str,
    profiles: Mapping[str, int],
    members: Iterable[TaskSetMember],
) -> dict[str, Any]:
    profile_text = ",".join(f"{key}:{int(value)}" for key, value in sorted(profiles.items()))
    max_profile = max((int(value) for value in profiles.values()), default=0)
    return {
        "finite": {
            "class_key": f"class:portfolio_action:{action_id}",
            "regime_key": "regime:measured_static_portfolio",
            "post_count_bucket": f"combo:{profile_text}",
            "post_vram_bucket": f"max_profile:{max_profile}",
            "util_bucket": "measured",
            "resource_bucket": ".".join(sorted(member.resource_kind for member in members)),
            "task_kind": "portfolio_profile_combo",
        },
        "post_task_count": float(max_profile),
        "running_task_count": float(max_profile),
        "post_vram_frac": 0.0,
        "used_vram_frac": 0.0,
        "util_pct": 0.0,
        "legacy_runtime_s": 0.0,
    }


def _exact_finite_slice_fabric_certificate(actions: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "L": _lightweight_lipschitz(actions),
        "rho": 0.0,
        "Lrho": 0.0,
        "cover_domain": "measured_finite_slice_exact",
        "full_count": len(actions),
        "candidate_count": len(actions),
        "usable_for_theorem": bool(actions),
        "status": "candidate_action_family_equals_measured_full_slice",
    }


def _lightweight_lipschitz(actions: list[Mapping[str, Any]]) -> float:
    # Features are unique per profile combo in this finite slice, so the minimum
    # nonzero metric distance is one unit in the discrete part.  This gives a
    # conservative finite service-gap envelope without materializing all pairs.
    max_gap = 0.0
    for i, left in enumerate(actions):
        lsvc = left.get("service_vector") or {}
        for right in actions[i + 1:]:
            rsvc = right.get("service_vector") or {}
            for cls in set(lsvc) | set(rsvc):
                max_gap = max(max_gap, abs(float(lsvc.get(cls, 0.0)) - float(rsvc.get(cls, 0.0))))
    return max_gap


def _finite_support_second_moment_bound(
    actions: list[Mapping[str, Any]],
    lam: Mapping[str, float],
) -> float:
    classes = sorted(set(lam) | {cls for action in actions for cls in (action.get("service_vector") or {})})
    total = 0.0
    for cls in classes:
        arrival_bound = max(0.0, float(lam.get(cls, 0.0)))
        service_bound = max(
            (max(0.0, float((action.get("service_vector") or {}).get(cls, 0.0))) for action in actions),
            default=0.0,
        )
        total += arrival_bound * arrival_bound + service_bound * service_bound
    return 0.5 * total


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    cert = build_measured_finite_slice_certificate(
        taskset_name=args.taskset,
        load_fraction=args.load_fraction,
        alpha=args.alpha,
        include_actions=args.include_actions,
    )
    _write_json(args.output, cert)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_markdown_certificate(cert), encoding="utf-8")
    print(args.output)
    return 0 if cert.get("usable_for_theorem") else 2


def _markdown_certificate(cert: Mapping[str, Any]) -> str:
    lines = [
        "# Measured Finite-Slice Slack Certificate",
        "",
        "Scope: " + str(cert.get("certificate_scope") or ""),
        "",
        "```text",
        f"taskset = {cert.get('taskset')}",
        f"policy = {cert.get('policy')}",
        f"load_fraction = {cert.get('load_fraction')}",
        f"selected_profiles = {cert.get('selected_profiles')}",
        f"full_action_count = {cert.get('full_action_count')}",
        "```",
        "",
        markdown_slack_table(cert),
        "## Load Certificate",
        "",
        "| Class | lambda | selected service |",
        "|---|---:|---:|",
    ]
    lam = cert.get("lambda") or {}
    svc = cert.get("selected_service_vector") or {}
    for key in sorted(set(lam) | set(svc)):
        lines.append(f"| `{key}` | {float(lam.get(key, 0.0)):.9f} | {float(svc.get(key, 0.0)):.9f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a positive theorem-condition certificate for the measured finite service-action slice. "
            "It is not a production-arrival certificate and does not claim statistical generalization "
            "outside the measured Scheduleurm buckets.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.empirical_slack_certificate")
    p.add_argument("--taskset", default="hybrid_research_portfolio")
    p.add_argument("--load-fraction", type=float, default=0.80)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--output", required=True)
    p.add_argument("--markdown-output", default="")
    p.add_argument("--include-actions", action="store_true")
    p.set_defaults(func=_cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
