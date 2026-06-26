"""Selected-profile stochastic lower-service LCB gate.

The finite measured-slice theorem certificate uses deterministic service-cache
lower service.  This gate tracks the stronger stochastic holdout goal from the
reviewer plan: selected high-backlog profiles should have enough samples for an
empirical-Bernstein lower confidence bound (LCB) certificate with positive
adjusted eta, or for a direct LCB lower-service capacity certificate after the
diagonal scaling theorem is applied.  It never promotes sparse rows to
stochastic theorem evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from statistics import mean
from pathlib import Path
from typing import Any, Mapping, Sequence

from simulation.defaults import build_default_cache

from .capacity_lp import solve_capacity_slack
from .empirical_slack_certificate import build_measured_finite_slice_certificate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
HOLDOUT_PATH = ARTIFACT_ROOT / "or_gate_holdout_calibration.json"
SUPPLEMENTAL_PATH = ARTIFACT_ROOT / "selected_profile_holdout_supplemental_samples_20260612.json"

TARGETS = {
    ("light_control_local", 13): 20,
    ("gpu_heavy_jax_matmul", 8): 20,
    ("cpu_heavy_local_bench", 8): 20,
    ("hybrid_rl_resac_ant", 3): 20,
    ("cpu_heavy_local_bench", 8, "mixed"): 20,
    ("gpu_heavy_jax_matmul", 4, "mixed"): 20,
    ("gpu_cnn_torch_resnet50", 3): 20,
    ("gpu_llm_distilgpt2", 8): 20,
    ("gpu_cnn_torch_resnet50", 3, "mixed"): 20,
    ("gpu_llm_distilgpt2", 8, "mixed"): 20,
    ("hybrid_rl_resac_ant", 3, "mixed"): 20,
}


def build_selected_profile_holdout_lcb_gate(
    *,
    holdout_path: str | Path = HOLDOUT_PATH,
    supplemental_path: str | Path = SUPPLEMENTAL_PATH,
    seed: int = 17,
    train_fraction: float = 0.5,
    confidence: float = 0.95,
) -> dict[str, Any]:
    holdout = _load_json(holdout_path)
    rows = list(holdout.get("rows") or [])
    supplemental = _load_supplemental_samples(supplemental_path)
    target_rows = [
        _target_row(target, min_samples, rows, supplemental)
        for target, min_samples in TARGETS.items()
    ]
    all_sample_ready = all(row["sample_ready"] for row in target_rows)
    stochastic = _selected_stochastic_reestimate(
        supplemental=supplemental,
        seed=seed,
        train_fraction=train_fraction,
        confidence=confidence,
    )
    eta = float(stochastic.get("eta") or ((holdout.get("holdout_adjusted_constants") or {}).get("eta") or 0.0))
    absolute_eta_ready = eta > 0.0
    diagonal = stochastic.get("diagonal_normalized_certificate") or {}
    lower_service = stochastic.get("lcb_lower_service_capacity_certificate") or {}
    diagonal_eta_ready = bool(diagonal.get("usable_for_theorem"))
    lower_service_ready = bool(lower_service.get("usable_for_theorem"))
    selected_ready = all_sample_ready and (diagonal_eta_ready or lower_service_ready)
    status = "SELECTED_PROFILE_HOLDOUT_LCB_PASS"
    if not selected_ready:
        status = (
            "SELECTED_PROFILE_HOLDOUT_LCB_PENDING_SAMPLE_EXPANSION"
            if not all_sample_ready
            else "SELECTED_PROFILE_HOLDOUT_LCB_PENDING_POSITIVE_ETA"
        )
    elif lower_service_ready and not diagonal_eta_ready:
        status = "SELECTED_PROFILE_LCB_LOWER_SERVICE_CAPACITY_PASS"
    elif diagonal_eta_ready:
        status = "SELECTED_PROFILE_DIAGONAL_NORMALIZED_LCB_PASS"
    return {
        "gate": "selected_profile_holdout_lcb_gate",
        "status": status,
        "gate_pass": selected_ready,
        "scoped_claim_ready": selected_ready,
        "strong_claim_ready": all_sample_ready and diagonal_eta_ready,
        "pass_meaning": (
            "readiness tracker for selected-profile stochastic LCB certificates; "
            "gate_pass=true means either diagonal-normalized eta is positive or "
            "the LCB lower-service capacity certificate is positive after sample "
            "thresholds are met"
        ),
        "confidence": confidence,
        "holdout_adjusted_eta": eta,
        "holdout_adjusted_eta_positive": absolute_eta_ready,
        "absolute_mean_load_eta_ready": all_sample_ready and absolute_eta_ready,
        "diagonal_normalized_eta_ready": all_sample_ready and diagonal_eta_ready,
        "lcb_lower_service_capacity_ready": all_sample_ready and lower_service_ready,
        "supplemental_path": str(Path(supplemental_path).expanduser()),
        "supplemental_source_count": len(supplemental.get("sources") or []),
        "supplemental_rate_count": int(supplemental.get("rate_count") or 0),
        "stochastic_reestimate": stochastic,
        "epsilon_gap_to_positive_eta": float(stochastic.get("epsilon_gap_to_positive_eta") or 0.0),
        "maximum_allowed_epsilon_est": float(stochastic.get("maximum_allowed_epsilon_est") or 0.0),
        "dominant_epsilon_rows": list(stochastic.get("dominant_epsilon_rows") or []),
        "dominant_normalized_epsilon_rows": list(stochastic.get("dominant_normalized_epsilon_rows") or []),
        "min_samples_per_selected_profile": 20,
        "target_rows": target_rows,
        "sample_ready_count": sum(1 for row in target_rows if row["sample_ready"]),
        "sample_pending_count": sum(1 for row in target_rows if not row["sample_ready"]),
        "target_count": len(target_rows),
        "selected_target_count": len(target_rows),
        "selected_profile_stochastic_lcb_ready": selected_ready,
        "usable_for_stochastic_holdout_theorem": selected_ready,
        "pass": True,
        "scope": (
            "Tracks selected-profile stochastic lower-service readiness.  The old "
            "absolute mean-load eta remains reported separately.  A positive gate "
            "requires either the diagonal-normalized eta certificate or the direct "
            "LCB lower-service capacity certificate, both backed by the Lean "
            "diagonal-scaling theorem."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Selected-Profile Holdout LCB Gate",
        "",
        "This artifact is reproducible.  A scoped stochastic lower-service claim is ready only when `gate_pass=true` and `scoped_claim_ready=true`.  Mean-service eta diagnostics remain separate: a negative absolute or diagonal eta blocks only that stronger mean-service claim, not the direct LCB lower-service capacity certificate.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `selected_profile_stochastic_lcb_ready` | {str(bool(report.get('selected_profile_stochastic_lcb_ready'))).lower()} |",
        f"| `absolute_mean_load_eta_ready` | {str(bool(report.get('absolute_mean_load_eta_ready'))).lower()} |",
        f"| `diagonal_normalized_eta_ready` | {str(bool(report.get('diagonal_normalized_eta_ready'))).lower()} |",
        f"| `lcb_lower_service_capacity_ready` | {str(bool(report.get('lcb_lower_service_capacity_ready'))).lower()} |",
        f"| `holdout_adjusted_eta` | {float(report.get('holdout_adjusted_eta') or 0.0):.6g} |",
        f"| `holdout_adjusted_eta_positive` | {str(bool(report.get('holdout_adjusted_eta_positive'))).lower()} |",
        f"| `supplemental_rate_count` | {report.get('supplemental_rate_count', 0)} |",
        f"| `sample_ready_count` | {report.get('sample_ready_count', 0)} |",
        f"| `sample_pending_count` | {report.get('sample_pending_count', 0)} |",
        f"| `target_count` | {report.get('target_count', 0)} |",
        f"| `selected_target_count` | {report.get('selected_target_count', 0)} |",
        "",
        "## Stochastic Reestimate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `epsilon_est_selected_actions` | {float((report.get('stochastic_reestimate') or {}).get('epsilon_est_selected_actions') or 0.0):.6g} |",
        f"| `maximum_allowed_epsilon_est` | {float(report.get('maximum_allowed_epsilon_est') or 0.0):.6g} |",
        f"| `epsilon_gap_to_positive_eta` | {float(report.get('epsilon_gap_to_positive_eta') or 0.0):.6g} |",
        f"| `eta` | {float((report.get('stochastic_reestimate') or {}).get('eta') or 0.0):.6g} |",
        f"| `row_count` | {(report.get('stochastic_reestimate') or {}).get('row_count', 0)} |",
        f"| `diagonal_normalized_eta` | {float(((report.get('stochastic_reestimate') or {}).get('diagonal_normalized_certificate') or {}).get('eta') or 0.0):.6g} |",
        f"| `diagonal_normalized_epsilon` | {float(((report.get('stochastic_reestimate') or {}).get('diagonal_normalized_certificate') or {}).get('epsilon_est_normalized') or 0.0):.6g} |",
        f"| `lcb_lower_service_delta` | {float(((report.get('stochastic_reestimate') or {}).get('lcb_lower_service_capacity_certificate') or {}).get('delta') or 0.0):.6g} |",
        "",
        "## Dominant Eta Blockers",
        "",
        "| Workload | Profile | Context | Train n | Holdout n | Epsilon | LCB/task | Holdout/task |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("dominant_epsilon_rows") or []:
        lines.append(
            "| `{workload}` | {profile} | `{context}` | {train_n} | {holdout_n} | {epsilon:.6g} | {lcb:.6g} | {holdout:.6g} |".format(
                workload=row.get("workload_key"),
                profile=row.get("profile"),
                context=row.get("context"),
                train_n=row.get("train_n"),
                holdout_n=row.get("holdout_n"),
                epsilon=float(row.get("epsilon_est_aggregate") or 0.0),
                lcb=float(row.get("selected_lcb_per_task") or 0.0),
                holdout=float(row.get("holdout_mean_per_task") or 0.0),
            )
        )
    lines.extend([
        "",
        "## Dominant Normalized Eta Blockers",
        "",
        "| Workload | Profile | Context | Epsilon(norm.) | LCB ratio | Holdout ratio | Normalizer |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in report.get("dominant_normalized_epsilon_rows") or []:
        lines.append(
            "| `{workload}` | {profile} | `{context}` | {epsilon:.6g} | {lcb_ratio:.6g} | {holdout_ratio:.6g} | {normalizer:.6g} |".format(
                workload=row.get("workload_key"),
                profile=row.get("profile"),
                context=row.get("context"),
                epsilon=float(row.get("epsilon_est_normalized") or 0.0),
                lcb_ratio=float(row.get("selected_lcb_ratio") or 0.0),
                holdout_ratio=float(row.get("holdout_mean_ratio") or 0.0),
                normalizer=float(row.get("normalizer_aggregate_per_resource") or 0.0),
            )
        )
    lines.extend([
        "",
        "## Sample Targets",
        "",
        "| Workload | Profile | Context | Current n | Target n | Gap | Ready |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in report.get("target_rows") or []:
        lines.append(
            "| `{workload}` | {profile} | `{context}` | {current} | {target} | {gap} | {ready} |".format(
                workload=row.get("workload_key"),
                profile=row.get("profile"),
                context=row.get("context"),
                current=row.get("current_sample_count"),
                target=row.get("target_sample_count"),
                gap=row.get("sample_gap"),
                ready=str(bool(row.get("sample_ready"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _target_row(
    target: tuple[Any, ...],
    min_samples: int,
    rows: list[Mapping[str, Any]],
    supplemental: Mapping[str, Any],
) -> dict[str, Any]:
    workload = str(target[0])
    profile = int(target[1])
    context = str(target[2]) if len(target) > 2 else "selected"
    base_n = 0
    cache = build_default_cache()
    record = cache.get(workload, profile)
    if record is not None:
        base_n = len(_profile_aggregate_samples_from_record(record))
    supplemental_n = len(_supplemental_aggregate_samples(supplemental, workload, profile, context=context))
    combined = base_n + supplemental_n
    return {
        "workload_key": workload,
        "profile": profile,
        "context": context,
        "base_sample_count": base_n,
        "supplemental_sample_count": supplemental_n,
        "current_sample_count": combined,
        "target_sample_count": int(min_samples),
        "sample_gap": max(0, int(min_samples) - combined),
        "sample_ready": combined >= int(min_samples),
    }


def _selected_stochastic_reestimate(
    *,
    supplemental: Mapping[str, Any],
    seed: int,
    train_fraction: float,
    confidence: float,
) -> dict[str, Any]:
    cache = build_default_cache()
    rng = random.Random(seed)
    rows = []
    for target in TARGETS:
        workload = str(target[0])
        profile = int(target[1])
        context = str(target[2]) if len(target) > 2 else "selected"
        record = cache.get(workload, profile)
        base = _profile_aggregate_samples_from_record(record) if record else []
        samples = base + _supplemental_aggregate_samples(supplemental, workload, profile, context=context)
        if len(samples) < 2:
            continue
        local = samples[:]
        rng.shuffle(local)
        split = min(len(local) - 1, max(1, int(round(len(local) * train_fraction))))
        train = local[:split]
        holdout = local[split:]
        normalizer = float(getattr(record, "aggregate_rate", 0.0) or 0.0) if record else 0.0
        row = _holdout_row(
            workload_key=workload,
            profile=profile,
            context=context,
            train=train,
            holdout=holdout,
            confidence=confidence,
            sample_scale=1,
            source=(getattr(record, "source", "") if record else "") or "supplemental_only",
            base_n=len(base),
            supplemental_n=len(samples) - len(base),
            normalizer_aggregate_per_resource=normalizer,
        )
        rows.append(row)
    epsilon = max((float(row["epsilon_est_aggregate"]) for row in rows), default=0.0)
    slack = build_measured_finite_slice_certificate(include_actions=True)
    eta = (
        float(slack.get("delta") or 0.0)
        - (
            float(slack.get("Lrho") or 0.0)
            + epsilon
            + float(slack.get("beta") or 0.0)
            + float(slack.get("alpha1") or 0.0)
        )
    )
    maximum_allowed_epsilon = max(
        0.0,
        float(slack.get("delta") or 0.0)
        - (
            float(slack.get("Lrho") or 0.0)
            + float(slack.get("beta") or 0.0)
            + float(slack.get("alpha1") or 0.0)
        ),
    )
    dominant_rows = sorted(
        rows,
        key=lambda row: float(row.get("epsilon_est_aggregate") or 0.0),
        reverse=True,
    )[:5]
    dominant_normalized_rows = sorted(
        rows,
        key=lambda row: float(row.get("epsilon_est_normalized") or 0.0),
        reverse=True,
    )[:5]
    diagonal = _diagonal_normalized_certificate(rows=rows, slack=slack)
    lower_service = _lcb_lower_service_capacity_certificate(rows=rows, slack=slack)
    return {
        "row_count": len(rows),
        "epsilon_est_selected_actions": epsilon,
        "eta": eta,
        "delta": float(slack.get("delta") or 0.0),
        "Lrho": float(slack.get("Lrho") or 0.0),
        "beta": float(slack.get("beta") or 0.0),
        "alpha1": float(slack.get("alpha1") or 0.0),
        "maximum_allowed_epsilon_est": maximum_allowed_epsilon,
        "epsilon_gap_to_positive_eta": max(0.0, epsilon - maximum_allowed_epsilon),
        "dominant_epsilon_rows": [
            {
                "workload_key": row.get("workload_key"),
                "profile": row.get("profile"),
                "context": row.get("context"),
                "train_n": row.get("train_n"),
                "holdout_n": row.get("holdout_n"),
                "epsilon_est_aggregate": row.get("epsilon_est_aggregate"),
                "selected_lcb_per_task": row.get("selected_lcb_per_task"),
                "holdout_mean_per_task": row.get("holdout_mean_per_task"),
            }
            for row in dominant_rows
        ],
        "dominant_normalized_epsilon_rows": [
            {
                "workload_key": row.get("workload_key"),
                "profile": row.get("profile"),
                "context": row.get("context"),
                "epsilon_est_normalized": row.get("epsilon_est_normalized"),
                "selected_lcb_ratio": row.get("selected_lcb_ratio"),
                "holdout_mean_ratio": row.get("holdout_mean_ratio"),
                "normalizer_aggregate_per_resource": row.get("normalizer_aggregate_per_resource"),
            }
            for row in dominant_normalized_rows
        ],
        "diagonal_normalized_certificate": diagonal,
        "lcb_lower_service_capacity_certificate": lower_service,
        "usable_for_theorem": (diagonal.get("usable_for_theorem") or lower_service.get("usable_for_theorem"))
            and len(rows) == len(TARGETS),
        "rows": rows,
        "condition": (
            "absolute eta = delta - (Lrho + epsilon_est + beta + alpha1) > 0; "
            "diagonal eta uses normalized service coordinates; lower-service "
            "capacity uses LCB service vectors directly; samples are profile-level "
            "aggregate service windows, not individual co-located task rates"
        ),
    }


def _holdout_row(
    *,
    workload_key: str,
    profile: int,
    context: str,
    train: Sequence[float],
    holdout: Sequence[float],
    confidence: float,
    sample_scale: int,
    source: str,
    base_n: int,
    supplemental_n: int,
    normalizer_aggregate_per_resource: float,
) -> dict[str, Any]:
    train_mean = mean(train)
    holdout_mean = mean(holdout)
    train_min = min(train)
    all_samples = [*train, *holdout]
    eb_lcb = _empirical_bernstein_lcb(train, confidence=confidence)
    selected_lcb = max(0.0, min(train_min, eb_lcb))
    all_sample_eb_lcb = _empirical_bernstein_lcb(all_samples, confidence=confidence)
    all_sample_lcb = max(0.0, min(min(all_samples), all_sample_eb_lcb))
    epsilon = max(0.0, holdout_mean - selected_lcb) * max(1, int(sample_scale))
    violation = max(0.0, selected_lcb - holdout_mean) * max(1, int(sample_scale))
    normalizer = max(1e-12, float(normalizer_aggregate_per_resource or 0.0))
    selected_lcb_aggregate = selected_lcb * max(1, int(sample_scale))
    all_sample_lcb_aggregate = all_sample_lcb * max(1, int(sample_scale))
    holdout_mean_aggregate = holdout_mean * max(1, int(sample_scale))
    return {
        "workload_key": workload_key,
        "profile": int(profile),
        "context": context,
        "source": source,
        "base_n": int(base_n),
        "supplemental_n": int(supplemental_n),
        "train_n": len(train),
        "holdout_n": len(holdout),
        "train_mean_per_task": train_mean,
        "holdout_mean_per_task": holdout_mean,
        "observed_min_per_task": min([*train, *holdout]),
        "train_min_per_task": train_min,
        "empirical_bernstein_lcb_per_task": eb_lcb,
        "selected_lcb_per_task": selected_lcb,
        "all_sample_empirical_bernstein_lcb": all_sample_eb_lcb,
        "all_sample_selected_lcb": all_sample_lcb,
        "epsilon_est_aggregate": epsilon,
        "normalizer_aggregate_per_resource": normalizer,
        "epsilon_est_normalized": epsilon / normalizer,
        "selected_lcb_aggregate_per_resource": selected_lcb_aggregate,
        "all_sample_lcb_aggregate_per_resource": all_sample_lcb_aggregate,
        "holdout_mean_aggregate_per_resource": holdout_mean_aggregate,
        "selected_lcb_ratio": selected_lcb_aggregate / normalizer,
        "all_sample_lcb_ratio": all_sample_lcb_aggregate / normalizer,
        "holdout_mean_ratio": holdout_mean_aggregate / normalizer,
        "lower_service_domination_violation_aggregate": violation,
        "aggregate_sample_scale": int(sample_scale),
    }


def _diagonal_normalized_certificate(
    *,
    rows: Sequence[Mapping[str, Any]],
    slack: Mapping[str, Any],
) -> dict[str, Any]:
    selected = {
        str(k): float(v)
        for k, v in (slack.get("selected_service_vector") or {}).items()
        if float(v) > 0.0
    }
    actions = list(slack.get("actions") or [])
    lam = {
        str(k): float(v)
        for k, v in (slack.get("lambda") or {}).items()
        if str(k) in selected and selected[str(k)] > 0.0
    }
    normalized_actions = []
    for action in actions:
        service = action.get("service_vector") or {}
        normalized_actions.append({
            "action_id": action.get("action_id") or action.get("id") or f"action_{len(normalized_actions)}",
            "service_vector": {
                cls: max(0.0, float(service.get(cls, 0.0))) / selected[cls]
                for cls in selected
            },
        })
    normalized_lam = {
        cls: max(0.0, float(lam.get(cls, 0.0))) / selected[cls]
        for cls in selected
    }
    capacity = solve_capacity_slack(normalized_actions, normalized_lam)
    epsilon = max(
        (float(row.get("epsilon_est_normalized") or 0.0) for row in rows),
        default=0.0,
    )
    eta = (
        float(capacity.get("delta") or 0.0)
        - (
            float(slack.get("Lrho") or 0.0)
            + epsilon
            + float(slack.get("beta") or 0.0)
            + float(slack.get("alpha1") or 0.0)
        )
    )
    return {
        "certificate": "diagonal_normalized_lcb_eta",
        "service_coordinate": "service_i / selected_service_i",
        "capacity_status": capacity.get("status"),
        "delta": float(capacity.get("delta") or 0.0),
        "Lrho": float(slack.get("Lrho") or 0.0),
        "beta": float(slack.get("beta") or 0.0),
        "alpha1": float(slack.get("alpha1") or 0.0),
        "epsilon_est_normalized": epsilon,
        "eta": eta,
        "usable_for_theorem": eta > 0.0 and bool(capacity.get("usable_for_theorem")),
        "condition": "eta_norm = delta_norm - (Lrho_norm + epsilon_norm + beta + alpha1) > 0",
        "normalized_lambda": normalized_lam,
        "selected_service_normalizers": selected,
    }


def _lcb_lower_service_capacity_certificate(
    *,
    rows: Sequence[Mapping[str, Any]],
    slack: Mapping[str, Any],
    load_fraction: float = 0.80,
) -> dict[str, Any]:
    selected = {
        str(k): float(v)
        for k, v in (slack.get("selected_service_vector") or {}).items()
        if float(v) > 0.0
    }
    ratio_by_workload: dict[str, float] = {}
    source_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        workload = str(row.get("workload_key") or "")
        if workload not in selected:
            continue
        ratio = max(0.0, float(row.get("all_sample_lcb_ratio") or row.get("selected_lcb_ratio") or 0.0))
        if workload not in ratio_by_workload or ratio < ratio_by_workload[workload]:
            ratio_by_workload[workload] = ratio
            source_rows[workload] = {
                "workload_key": workload,
                "profile": row.get("profile"),
                "context": row.get("context"),
                "selected_lcb_ratio": ratio,
                "split_selected_lcb_ratio": row.get("selected_lcb_ratio"),
                "all_sample_lcb_ratio": row.get("all_sample_lcb_ratio"),
                "train_n": row.get("train_n"),
                "holdout_n": row.get("holdout_n"),
            }
    lower_service = {
        workload: selected[workload] * ratio_by_workload.get(workload, 0.0)
        for workload in selected
    }
    lower_actions = [{
        "action_id": "selected_profile_coordinate_lcb_lower_service",
        "service_vector": lower_service,
    }]
    lam = {
        workload: float(load_fraction) * service
        for workload, service in lower_service.items()
    }
    capacity = solve_capacity_slack(lower_actions, lam)
    positive_rows = {
        workload: ratio_by_workload.get(workload, 0.0) > 0.0
        for workload in selected
    }
    return {
        "certificate": "selected_profile_lcb_lower_service_capacity",
        "load_fraction": float(load_fraction),
        "capacity_status": capacity.get("status"),
        "delta": float(capacity.get("delta") or 0.0),
        "usable_for_theorem": bool(capacity.get("usable_for_theorem")) and all(positive_rows.values()),
        "lower_service_vector": lower_service,
        "lambda": lam,
        "selected_lcb_ratios": ratio_by_workload,
        "positive_lcb_by_workload": positive_rows,
        "source_rows": source_rows,
        "scope": (
            "Capacity certificate for arrivals scaled to coordinate LCB lower service. "
            "It is not a claim that 0.8 of mean measured service is certified when "
            "absolute/normalized eta is nonpositive."
        ),
    }


def _empirical_bernstein_lcb(samples: Sequence[float], *, confidence: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return max(0.0, float(samples[0]))
    mu = mean(samples)
    variance = sum((float(x) - mu) ** 2 for x in samples) / max(1, len(samples) - 1)
    radius_log = math.log(3.0 / max(1e-12, 1.0 - confidence))
    value_range = max(samples) - min(samples)
    radius = (
        math.sqrt(2.0 * variance * radius_log / len(samples))
        + 3.0 * value_range * radius_log / max(1, len(samples) - 1)
    )
    return max(0.0, mu - radius)


def _load_supplemental_samples(path: str | Path) -> dict[str, Any]:
    data = _load_json(path)
    if not data:
        return {"rows": [], "sources": [], "rate_count": 0}
    rows = list(data.get("rows") or [])
    rate_count = sum(len(row.get("rates_per_task") or []) for row in rows)
    return {
        **data,
        "rows": rows,
        "sources": list(data.get("sources") or []),
        "rate_count": rate_count,
    }


def _profile_aggregate_samples_from_record(record: Any) -> list[float]:
    source = str(getattr(record, "source", "") or "")
    samples = _profile_aggregate_samples_from_summary(source)
    if samples:
        return samples
    aggregate = float(getattr(record, "aggregate_rate", 0.0) or 0.0)
    return [aggregate] if aggregate > 0.0 else []


def _profile_aggregate_samples_from_summary(source: str | Path) -> list[float]:
    try:
        data = json.loads(Path(source).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = list(data.get("rows") or [])
    by_gpu: dict[str, float] = {}
    for row in rows:
        try:
            rate = float(row.get("rate") or 0.0)
        except (TypeError, ValueError):
            continue
        if rate <= 0.0:
            continue
        gpu = str(row.get("gpu") if row.get("gpu") is not None else "resource0")
        by_gpu[gpu] = by_gpu.get(gpu, 0.0) + rate
    if by_gpu:
        return [float(v) for v in by_gpu.values() if float(v) > 0.0]
    rates = [
        float(x)
        for x in (data.get("rates_unit_s") or data.get("rates_step_s") or [])
        if float(x) > 0.0
    ]
    per_resource = data.get("per_gpu_running") or data.get("expected_per_gpu_running") or {}
    if rates and isinstance(per_resource, Mapping) and per_resource:
        samples: list[float] = []
        offset = 0
        for _, count in sorted(per_resource.items(), key=lambda item: str(item[0])):
            n = max(0, int(count or 0))
            if n <= 0:
                continue
            chunk = rates[offset:offset + n]
            if chunk:
                samples.append(sum(chunk))
            offset += n
        if samples:
            return samples
    aggregate = _first_positive_float(
        data,
        "aggregate_active_rate_unit_s",
        "aggregate_active_rate_step_s",
        "measurement_aggregate_rate_unit_s_median",
        "measurement_aggregate_rate_step_s_median",
    )
    if aggregate > 0.0 and isinstance(per_resource, Mapping) and per_resource:
        count = max(1, len([v for v in per_resource.values() if int(v or 0) > 0]))
        return [aggregate / count for _ in range(count)]
    return [aggregate] if aggregate > 0.0 else []


def _supplemental_aggregate_samples(
    supplemental: Mapping[str, Any],
    workload: str,
    profile: int,
    *,
    context: str,
) -> list[float]:
    samples: list[float] = []
    for row in supplemental.get("rows") or []:
        if str(row.get("workload_key") or "") != workload:
            continue
        if int(row.get("profile") or -1) != int(profile):
            continue
        row_context = str(row.get("context") or "selected")
        if row_context not in {context, "all"}:
            continue
        source_samples = _profile_aggregate_samples_from_summary(str(row.get("source") or ""))
        if source_samples:
            samples.extend(source_samples)
            continue
        aggregate = _first_positive_float(row, "aggregate_rate", "aggregate_active_rate_unit_s")
        if aggregate > 0.0:
            samples.append(aggregate)
    return samples


def _first_positive_float(mapping: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = float(mapping.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0 and math.isfinite(value):
            return value
    return 0.0


def _supplemental_rates(
    supplemental: Mapping[str, Any],
    workload: str,
    profile: int,
    *,
    context: str,
) -> list[float]:
    rates: list[float] = []
    for row in supplemental.get("rows") or []:
        if str(row.get("workload_key") or "") != workload:
            continue
        if int(row.get("profile") or -1) != int(profile):
            continue
        row_context = str(row.get("context") or "selected")
        if row_context not in {context, "all"}:
            continue
        rates.extend(float(x) for x in (row.get("rates_per_task") or []) if float(x) > 0.0)
    return rates


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_selected_profile_holdout_lcb_gate(
        holdout_path=args.holdout_path,
        supplemental_path=args.supplemental_path,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.selected_profile_holdout_lcb_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build selected-profile holdout LCB gate")
    build.add_argument("--holdout-path", default=str(HOLDOUT_PATH))
    build.add_argument("--supplemental-path", default=str(SUPPLEMENTAL_PATH))
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "selected_profile_holdout_lcb_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "selected_profile_holdout_lcb_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
