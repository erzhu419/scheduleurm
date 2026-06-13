"""SOTA-facing algorithm upgrade gate.

This gate implements the five algorithmic follow-ups requested after the SOTA
candidate-union closure:

1. a single adaptive-scalarized Scheduleurm+SOTA union policy;
2. a finer state-dependent marginal service cache;
3. a bounded global batch lookahead selector;
4. reusable online LCB/ETA estimates that avoid repeated probes;
5. additional SOTA policy families admitted into the finite action union.

It is a replay/certificate gate.  It does not mutate the production scheduler
or claim direct full-stack external-system superiority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.theorem_dispatch.eta_lcb import (
    OnlineServiceEstimator,
    ReusableEtaCache,
    service_observation_key,
)
from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.state_service import (
    StateDependentServiceCache,
    classify_load_state,
)
from simulation.defaults import build_default_cache
from simulation.sota_baselines import sota_baseline_specs

from .sota_candidate_union_gate import build_sota_candidate_union_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_sota_algorithm_upgrade_gate() -> dict[str, Any]:
    union = build_sota_candidate_union_gate()
    aggregate = union.get("aggregate") or {}
    adaptive = _adaptive_scalarized_certificate(union)
    state_cache = _state_cache_certificate()
    global_batch = _global_batch_lookahead_certificate()
    eta_reuse = _eta_reuse_certificate()
    sota_families = _expanded_sota_family_certificate()
    gate_pass = all(
        (
            adaptive["adaptive_scalarized_ready"],
            state_cache["state_dependent_marginal_cache_ready"],
            global_batch["lookahead_global_batch_ready"],
            eta_reuse["eta_reuse_ready"],
            sota_families["expanded_sota_families_ready"],
        )
    )
    return {
        "gate": "sota_algorithm_upgrade_gate",
        "status": "SOTA_ALGORITHM_UPGRADE_PASS" if gate_pass else "SOTA_ALGORITHM_UPGRADE_REVIEW",
        "pass": bool(gate_pass),
        "scoped_claim_ready": bool(gate_pass),
        "strong_claim_ready": False,
        "sota_candidate_union_summary": {
            "status": union.get("status"),
            "scenario_count": union.get("scenario_count"),
            "union_makespan_beats_sota_envelope_all": aggregate.get("union_makespan_beats_sota_envelope_all"),
            "union_mean_flow_beats_sota_envelope_all": aggregate.get("union_mean_flow_beats_sota_envelope_all"),
            "adaptive_scalarized_union_not_pareto_dominated_all": aggregate.get("adaptive_scalarized_union_not_pareto_dominated_all"),
            "adaptive_scalarized_union_beats_both_envelopes_all": aggregate.get("adaptive_scalarized_union_beats_both_envelopes_all"),
        },
        "adaptive_scalarized_single_policy": adaptive,
        "state_dependent_marginal_cache": state_cache,
        "global_batch_lookahead": global_batch,
        "eta_reuse": eta_reuse,
        "expanded_sota_policy_families": sota_families,
        "scope": (
            "Replay/certificate evidence for an opt-in algorithm upgrade.  It "
            "admits SOTA-style action families into Scheduleurm's measured-cache "
            "candidate set and adds bounded lookahead/ETA reuse, but it is not a "
            "production launch trace and not direct full-stack SOTA superiority."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    adaptive = report.get("adaptive_scalarized_single_policy") or {}
    state = report.get("state_dependent_marginal_cache") or {}
    batch = report.get("global_batch_lookahead") or {}
    eta = report.get("eta_reuse") or {}
    families = report.get("expanded_sota_policy_families") or {}
    lines = [
        "# SOTA Algorithm Upgrade Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `adaptive_scalarized_ready` | {str(bool(adaptive.get('adaptive_scalarized_ready'))).lower()} |",
        f"| `state_dependent_marginal_cache_ready` | {str(bool(state.get('state_dependent_marginal_cache_ready'))).lower()} |",
        f"| `lookahead_global_batch_ready` | {str(bool(batch.get('lookahead_global_batch_ready'))).lower()} |",
        f"| `eta_reuse_ready` | {str(bool(eta.get('eta_reuse_ready'))).lower()} |",
        f"| `expanded_sota_families_ready` | {str(bool(families.get('expanded_sota_families_ready'))).lower()} |",
        "",
        "## Adaptive Scalarized Union",
        "",
        "| Policy | Sum makespan | Weighted mean flow | vs best SOTA makespan envelope | vs best SOTA flow envelope |",
        "|---|---:|---:|---:|---:|",
    ]
    row = adaptive.get("aggregate_row") or {}
    lines.append(
        "| `{policy}` | {makespan:.6g} | {flow:.6g} | {ms:.6g} | {mf:.6g} |".format(
            policy=row.get("policy") or "",
            makespan=float(row.get("sum_makespan_s") or 0.0),
            flow=float(row.get("job_weighted_mean_flow_s") or 0.0),
            ms=float(adaptive.get("worst_vs_sota_makespan_envelope") or 0.0),
            mf=float(adaptive.get("worst_vs_sota_mean_flow_envelope") or 0.0),
        )
    )
    lines.extend([
        "",
        "## Marginal Service States",
        "",
        "| Workload | State | Certified | Relative to empty |",
        "|---|---|---:|---:|",
    ])
    for row in state.get("marginal_rows") or []:
        lines.append(
            "| `{workload}` | `{state}` | {certified} | {ratio:.6g} |".format(
                workload=row.get("workload_key"),
                state=row.get("load_state"),
                certified=str(bool(row.get("certified"))).lower(),
                ratio=float(row.get("relative_to_reference") or 0.0),
            )
        )
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _adaptive_scalarized_certificate(union_report: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = [row for row in union_report.get("scenarios") or [] if row.get("replayable", True)]
    aggregate_rows = (union_report.get("aggregate") or {}).get("policy_aggregate") or []
    adaptive_row = next(
        (
            row for row in aggregate_rows
            if row.get("policy") == "scheduleurm_sota_union_adaptive_scalarized"
        ),
        {},
    )
    worst_ms = min(
        (
            float(row.get("adaptive_union_vs_sota_best_makespan") or 0.0)
            for row in scenarios
        ),
        default=0.0,
    )
    worst_flow = min(
        (
            float(row.get("adaptive_union_vs_sota_best_mean_flow") or 0.0)
            for row in scenarios
        ),
        default=0.0,
    )
    dominated = [
        row for row in scenarios
        if row.get("adaptive_union_pareto_dominated_by_sota")
    ]
    return {
        "adaptive_scalarized_ready": bool(adaptive_row and not dominated),
        "aggregate_row": adaptive_row,
        "dominated_scenario_count": len(dominated),
        "worst_vs_sota_makespan_envelope": worst_ms,
        "worst_vs_sota_mean_flow_envelope": worst_flow,
        "beats_both_envelopes_all": bool(
            (union_report.get("aggregate") or {}).get("adaptive_scalarized_union_beats_both_envelopes_all")
        ),
        "not_pareto_dominated_all": bool(
            (union_report.get("aggregate") or {}).get("adaptive_scalarized_union_not_pareto_dominated_all")
        ),
    }


def _state_cache_certificate() -> dict[str, Any]:
    state_cache = StateDependentServiceCache.load(base_cache=build_default_cache())
    marginal_rows = (
        state_cache.marginal_opportunity_rows("marginal_cuda")
        + state_cache.marginal_opportunity_rows("marginal_cpu")
    )
    classifications = {
        "gpu_empty": classify_load_state(
            resource_kind="gpu", running_task_count=0, used_mb=100, total_mb=12000
        ),
        "gpu_half": classify_load_state(
            resource_kind="gpu", running_task_count=2, used_mb=4000, total_mb=12000
        ),
        "gpu_high_vram": classify_load_state(
            resource_kind="gpu", running_task_count=4, used_mb=10000, total_mb=12000
        ),
        "cpu_full": classify_load_state(
            resource_kind="cpu", cpu_workers=180, total_cpu=192
        ),
    }
    ready = (
        all(row["certified"] for row in marginal_rows)
        and {"empty", "high_vram_resident"} <= set(state_cache.available_states().get("marginal_cuda", []))
        and {"empty", "cpu_half_resident", "cpu_full_resident"} <= set(state_cache.available_states().get("marginal_cpu", []))
    )
    return {
        "state_dependent_marginal_cache_ready": bool(ready),
        "available_states": state_cache.available_states(),
        "load_state_classifications": classifications,
        "marginal_rows": marginal_rows,
    }


def _global_batch_lookahead_certificate() -> dict[str, Any]:
    rows = [
        _candidate_row("a_slow", eta=120.0),
        _candidate_row("b_fast", eta=20.0),
    ]
    queue = {"gpu_heavy_jax_matmul": 8.0}
    without = select_global_action(rows, queue, max_batch_size=1, lookahead_weight=0.0)
    with_lookahead = select_global_action(rows, queue, max_batch_size=1, lookahead_weight=1.0)
    return {
        "lookahead_global_batch_ready": (
            without.selected_action_ids == ("a_slow",)
            and with_lookahead.selected_action_ids == ("b_fast",)
            and with_lookahead.oracle_gap_alpha0 == 0.0
            and with_lookahead.oracle_gap_alpha1 == 0.0
        ),
        "without_lookahead": without.snapshot(),
        "with_lookahead": with_lookahead.snapshot(),
        "semantics": "bounded lookahead only affects tie/near-tie global configurations; robust lower-service scoring remains certified",
    }


def _eta_reuse_certificate() -> dict[str, Any]:
    cache = ReusableEtaCache(
        OnlineServiceEstimator(confidence=0.80, min_samples=3)
    )
    key = service_observation_key(
        workload_key="gpu_heavy_jax_matmul",
        profile=1,
        load_state="high_vram_resident",
        node_bucket="node007:12gb",
        command_fingerprint="jax_matmul_size8192_v1",
        algorithm_mode="sota_adaptive_scalarized_v1",
    )
    for idx, rate in enumerate((9.7, 9.8, 9.9, 9.85)):
        cache.observe(
            key=key,
            units_done=rate * 10.0,
            elapsed_s=10.0,
            remaining_units=100.0,
            source=f"sample{idx}",
        )
    first = cache.lookup(key, remaining_units=100.0)
    second = cache.lookup(key, remaining_units=50.0)
    missing = cache.lookup(
        service_observation_key(
            workload_key="gpu_heavy_jax_matmul",
            profile=2,
            load_state="high_vram_resident",
            node_bucket="node007:12gb",
            command_fingerprint="jax_matmul_size8192_v1",
            algorithm_mode="sota_adaptive_scalarized_v1",
        ),
        remaining_units=50.0,
    )
    snap = cache.snapshot()
    return {
        "eta_reuse_ready": bool(
            first.certified
            and second.certified
            and first.reused_from_cache
            and second.reused_from_cache
            and missing.probe_required
            and snap["reuse_count"] == 2
            and snap["probe_required_count"] == 1
        ),
        "first_lookup": first.snapshot(),
        "second_lookup": second.snapshot(),
        "missing_lookup": missing.snapshot(),
        "cache": snap,
    }


def _expanded_sota_family_certificate() -> dict[str, Any]:
    specs = sota_baseline_specs()
    systems = sorted(
        {
            system
            for spec in specs
            for system in spec.representative_systems
        }
    )
    names = [spec.name for spec in specs]
    required = {"Gavel", "Pollux", "Sia", "IADeep", "Salus"}
    return {
        "expanded_sota_families_ready": bool(
            len(specs) >= 7 and required <= set(systems)
        ),
        "family_count": len(specs),
        "families": names,
        "representative_systems": systems,
    }


def _candidate_row(action_id: str, *, eta: float) -> dict[str, Any]:
    return {
        "task_id": action_id,
        "action_id": action_id,
        "resource_ids": (f"gpu:{action_id}",),
        "workload_key": "gpu_heavy_jax_matmul",
        "service_workload_key": "gpu_heavy_jax_matmul",
        "lower_service": {"gpu_heavy_jax_matmul": 1.0},
        "selected_class_lower_service": 1.0,
        "penalty_units": 0.0,
        "eta_lcb_s": float(eta),
        "oldest_wait_s": 600.0,
        "score_semantics": "robust_maxweight_lower_service",
        "theorem_ready": True,
    }


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_algorithm_upgrade_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_algorithm_upgrade_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build the SOTA-facing algorithm upgrade gate")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "sota_algorithm_upgrade_gate_20260613.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "sota_algorithm_upgrade_gate_20260613.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
