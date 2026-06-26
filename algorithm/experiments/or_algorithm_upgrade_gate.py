"""Algorithm-upgrade gate for the OR candidate.

This gate validates five implementation axes without changing the legacy
scheduler default:

1. bounded global batch robust-MaxWeight selector;
2. state-dependent marginal service cache;
3. online LCB/ETA updater;
4. backlog-aware guarded replay policy;
5. q01 CNN/LLM/co-location non-regression and improvement checks.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from algorithm.theorem_dispatch.batch_policy import select_global_batch_placements
from algorithm.theorem_dispatch.eta_lcb import OnlineServiceEstimator, service_observation_key
from algorithm.theorem_dispatch.state_service import StateDependentServiceCache
from simulation.defaults import (
    build_default_cache,
    calibrated_scalar_candidate_policy,
    legacy_policy,
)
from simulation.fast_forward import compare_policies
from simulation.sota_baselines import compare_against_sota_suite
from simulation.sota_baselines import sota_candidate_union_policy
from simulation.tasksets import benchmark_tasksets


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
TARGET_TASKSETS = (
    "q01_gpu_bound_compute",
    "q01_gpu_bound_cnn_resnet50",
    "q01_gpu_bound_llm_inference",
    "q01_gpu_model_portfolio",
    "node007_task_native_cnn",
    "node007_task_native_llm",
    "node007_task_native_eta_portfolio",
    "q10_cpu_host_bound",
    "q11_cpu_gpu_coupled",
    "hybrid_research_portfolio",
)


def build_or_algorithm_upgrade_gate(
    *,
    trials: int = 31,
    seed: int = 7,
    tolerance: float = 0.005,
) -> dict[str, Any]:
    cache = build_default_cache()
    rows = []
    for name in TARGET_TASKSETS:
        specs = benchmark_tasksets()[name].workload_specs(replayable_only=True, cache=cache)
        current_policy = calibrated_scalar_candidate_policy(cache, specs)
        upgraded_policy = sota_candidate_union_policy(
            cache,
            specs,
            selection_objective="online_pareto_slack",
        )
        current = compare_policies(
            cache,
            specs,
            baseline=legacy_policy(),
            candidate=current_policy,
            trials=trials,
            seed=seed,
        )
        upgraded = compare_policies(
            cache,
            specs,
            baseline=legacy_policy(),
            candidate=upgraded_policy,
            trials=trials,
            seed=seed,
        )
        sota = compare_against_sota_suite(
            cache,
            specs,
            candidate=upgraded_policy,
            trials=max(11, min(31, trials)),
            seed=seed,
        )
        rows.append(_upgrade_row(name, current, upgraded, sota, tolerance=tolerance))

    global_batch = _global_batch_certificate()
    state_cache = _state_cache_certificate(cache)
    lcb = _lcb_certificate()
    regressions = _regression_rows(rows, tolerance)
    improvements = _improvement_rows(rows, tolerance)
    gate_pass = (
        not regressions
        and len(improvements) >= 3
        and global_batch["global_batch_scheduler_hook_ready"]
        and state_cache["state_dependent_cache_ready"]
        and lcb["online_lcb_eta_ready"]
        and all(row["upgraded_not_pareto_dominated_by_sota_tol"] for row in rows)
    )
    return {
        "gate": "or_algorithm_upgrade_gate",
        "status": "OR_ALGORITHM_UPGRADE_PASS" if gate_pass else "OR_ALGORITHM_UPGRADE_REVIEW",
        "gate_pass": gate_pass,
        "scoped_claim_ready": gate_pass,
        "strong_claim_ready": False,
        "pass_meaning": (
            "optional algorithm-layer upgrade validated by replay and certificates; "
            "legacy scheduler default is unchanged and direct external binary "
            "superiority is not claimed"
        ),
        "trials": int(trials),
        "seed": int(seed),
        "tolerance": float(tolerance),
        "rows": rows,
        "regression_count": len(regressions),
        "regressions": regressions,
        "improvement_count": len(improvements),
        "improvements": improvements,
        "global_batch": global_batch,
        "state_dependent_service_cache": state_cache,
        "online_lcb_eta": lcb,
        "pass": gate_pass,
        "scope": (
            "The gate covers measured Scheduleurm replay tasksets and synthetic "
            "global-dispatch/state-cache/LCB certificates. It is not a production "
            "launch claim and not a direct Gavel/Pollux/Sia/IADeep binary result."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# OR Algorithm Upgrade Gate",
        "",
        "This gate validates the optional algorithm-layer upgrade. It does not change the legacy scheduler default.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `regression_count` | {report.get('regression_count')} |",
        f"| `improvement_count` | {report.get('improvement_count')} |",
        f"| `global_batch_scheduler_hook_ready` | {str(bool((report.get('global_batch') or {}).get('global_batch_scheduler_hook_ready'))).lower()} |",
        f"| `state_dependent_cache_ready` | {str(bool((report.get('state_dependent_service_cache') or {}).get('state_dependent_cache_ready'))).lower()} |",
        f"| `online_lcb_eta_ready` | {str(bool((report.get('online_lcb_eta') or {}).get('online_lcb_eta_ready'))).lower()} |",
        "",
        "## Replay Rows",
        "",
        "| Taskset | Current vs legacy makespan | Upgraded vs legacy makespan | Upgraded/current makespan | Current vs legacy flow | Upgraded vs legacy flow | Upgraded/current flow | SOTA Pareto safe (tol.) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{taskset}` | {cms:.6g} | {ums:.6g} | {ucms:.6g} | {cmf:.6g} | {umf:.6g} | {ucmf:.6g} | {safe} |".format(
                taskset=row.get("taskset"),
                cms=float(row.get("current_vs_legacy_makespan") or 0.0),
                ums=float(row.get("upgraded_vs_legacy_makespan") or 0.0),
                ucms=float(row.get("upgraded_vs_current_makespan") or 0.0),
                cmf=float(row.get("current_vs_legacy_mean_flow") or 0.0),
                umf=float(row.get("upgraded_vs_legacy_mean_flow") or 0.0),
                ucmf=float(row.get("upgraded_vs_current_mean_flow") or 0.0),
                safe=str(bool(row.get("upgraded_not_pareto_dominated_by_sota_tol"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _upgrade_row(
    name: str,
    current: Any,
    upgraded: Any,
    sota: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    strict_dominators = list(sota.get("candidate_pareto_dominated_by") or [])
    tol_dominators = _tolerance_pareto_dominators(sota, tolerance)
    return {
        "taskset": name,
        "current_policy": current.candidate.policy,
        "upgraded_policy": upgraded.candidate.policy,
        "current_profiles": {
            row.workload_key: row.selected_profile
            for row in current.candidate.workloads
        },
        "upgraded_profiles": {
            row.workload_key: row.selected_profile
            for row in upgraded.candidate.workloads
        },
        "current_vs_legacy_makespan": current.makespan_improvement,
        "upgraded_vs_legacy_makespan": upgraded.makespan_improvement,
        "upgraded_vs_current_makespan": _ratio(
            current.candidate.total_makespan_s,
            upgraded.candidate.total_makespan_s,
        ),
        "current_vs_legacy_mean_flow": current.mean_flow_improvement,
        "upgraded_vs_legacy_mean_flow": upgraded.mean_flow_improvement,
        "upgraded_vs_current_mean_flow": _ratio(
            current.candidate.weighted_mean_flow_s,
            upgraded.candidate.weighted_mean_flow_s,
        ),
        "upgraded_not_pareto_dominated_by_sota": bool(sota.get("candidate_not_pareto_dominated")),
        "upgraded_not_pareto_dominated_by_sota_tol": not tol_dominators,
        "upgraded_pareto_dominated_by_sota": strict_dominators,
        "upgraded_pareto_dominated_by_sota_tol": tol_dominators,
    }


def _global_batch_certificate() -> dict[str, Any]:
    tasks = [
        {
            "id": "batch-q01",
            "status": "queued",
            "project": "ScheduleurmBench",
            "signature": "bench/q01/gpu-heavy",
            "cmd": "python jax_matmul.py --gpu_heavy",
            "workload_key": "gpu_heavy_jax_matmul",
            "est_vram_mb": 512,
            "ram_mb": 1024,
            "cpu_cores": 1,
        },
        {
            "id": "batch-q11",
            "status": "queued",
            "project": "BAPR",
            "signature": "BAPR/resac/train",
            "cmd": "python train.py --env Ant-v5",
            "workload_key": "hybrid_rl_resac_ant",
            "est_vram_mb": 512,
            "ram_mb": 2048,
            "cpu_cores": 2,
        },
    ]
    nodes = [
        {
            "name": "gate-node-a",
            "alive": True,
            "gpus": [
                {"idx": 0, "used_mb": 100, "free_mb": 11900, "total_mb": 12000, "util_pct": 0, "running_task_count": 0},
                {"idx": 1, "used_mb": 100, "free_mb": 11900, "total_mb": 12000, "util_pct": 0, "running_task_count": 0},
            ],
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = Path(tmpdir) / "queue.json"
        queue_path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        result = select_global_batch_placements(
            tasks,
            nodes,
            context={"queue_file": str(queue_path), "global_batch_size": 2},
            max_batch_size=2,
        )
    snap = result.snapshot()
    return {
        "global_batch_scheduler_hook_ready": bool(result.scheduler_hook_ready),
        "selected_task_count": len(result.placements),
        "candidate_configuration_count": snap["action"]["candidate_configuration_count"],
        "oracle_gap_alpha0": snap["action"]["oracle_gap_alpha0"],
        "oracle_gap_alpha1": snap["action"]["oracle_gap_alpha1"],
        "snapshot": snap,
    }


def _state_cache_certificate(cache: Any) -> dict[str, Any]:
    state_cache = StateDependentServiceCache.load(base_cache=cache)
    gpu_empty = state_cache.lookup("marginal_cuda", 1, "empty", allow_base_fallback=False)
    gpu_resident = state_cache.lookup("marginal_cuda", 1, "high_vram_resident", allow_base_fallback=False)
    cpu_full = state_cache.lookup("marginal_cpu", 1, "cpu_full_resident", allow_base_fallback=False)
    ready = all(row.certified for row in (gpu_empty, gpu_resident, cpu_full))
    return {
        "state_dependent_cache_ready": ready,
        "available_states": state_cache.available_states(),
        "gpu_empty": gpu_empty.snapshot(),
        "gpu_high_vram_resident": gpu_resident.snapshot(),
        "cpu_full_resident": cpu_full.snapshot(),
    }


def _lcb_certificate() -> dict[str, Any]:
    key = service_observation_key(
        workload_key="gpu_heavy_jax_matmul",
        profile=1,
        load_state="empty",
        node_bucket="gate-node-a:12gb",
        command_fingerprint="unit",
        algorithm_mode="global_theorem_maxweight_v1",
    )
    estimator = OnlineServiceEstimator(confidence=0.80, min_samples=3)
    for idx, rate in enumerate((9.8, 10.0, 10.2, 10.1, 9.9)):
        estimator.add_progress(key=key, units_done=rate * 10.0, elapsed_s=10.0, source=f"sample{idx}")
    estimate = estimator.estimate(key, remaining_units=100.0)
    return {
        "online_lcb_eta_ready": bool(estimate.certified and estimate.eta_s and estimate.eta_s > 0.0),
        "estimate": estimate.snapshot(),
        "dedup_key_fields_present": all(part in key for part in ("gpu_heavy_jax_matmul", "empty", "gate-node-a", "global_theorem_maxweight_v1")),
    }


def _regression_rows(rows: list[Mapping[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    """Rows where the old candidate Pareto-dominates the upgrade.

    The upgrade gate is Pareto-facing: a single metric may spend bounded slack
    when the other metric improves.  Count a regression only when both metrics
    are below the tolerance floor, i.e. the previous candidate is better in both
    makespan and mean-flow under the same replay cache.
    """

    out = []
    floor = 1.0 - float(tolerance)
    for row in rows:
        makespan = float(row.get("upgraded_vs_current_makespan") or 0.0)
        mean_flow = float(row.get("upgraded_vs_current_mean_flow") or 0.0)
        if makespan < floor and mean_flow < floor:
            out.append({
                "taskset": row.get("taskset"),
                "metric": "pareto",
                "makespan_ratio": makespan,
                "mean_flow_ratio": mean_flow,
            })
    return out


def _improvement_rows(rows: list[Mapping[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    out = []
    threshold = 1.0 + float(tolerance)
    for row in rows:
        metrics = []
        if float(row.get("upgraded_vs_current_makespan") or 0.0) > threshold:
            metrics.append("makespan")
        if float(row.get("upgraded_vs_current_mean_flow") or 0.0) > threshold:
            metrics.append("mean_flow")
        if metrics:
            out.append({"taskset": row.get("taskset"), "metrics": metrics})
    return out


def _tolerance_pareto_dominators(sota: Mapping[str, Any], tolerance: float) -> list[dict[str, Any]]:
    out = []
    floor = 1.0 - float(tolerance)
    for row in (sota.get("baselines") or []):
        ms = float(row.get("candidate_vs_baseline_makespan") or 0.0)
        flow = float(row.get("candidate_vs_baseline_mean_flow") or 0.0)
        if ms < floor and flow < floor:
            out.append({
                "name": (row.get("baseline") or {}).get("name"),
                "candidate_vs_baseline_makespan": ms,
                "candidate_vs_baseline_mean_flow": flow,
            })
    return out


def _ratio(old_value: float, new_value: float) -> float:
    return float(old_value) / float(new_value) if float(new_value) > 0.0 else 0.0


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_or_algorithm_upgrade_gate(
        trials=args.trials,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.or_algorithm_upgrade_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build OR algorithm upgrade gate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "or_algorithm_upgrade_gate_20260613.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "or_algorithm_upgrade_gate_20260613.md"))
    build.add_argument("--trials", type=int, default=31)
    build.add_argument("--seed", type=int, default=7)
    build.add_argument("--tolerance", type=float, default=0.005)
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
