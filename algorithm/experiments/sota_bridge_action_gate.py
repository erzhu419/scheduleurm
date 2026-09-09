"""Bridge-action gate for measured-cache external-policy frontiers.

This gate attacks the remaining strict Pareto rows from two directions:

1. search the current measured service cache for finite phase-switch actions;
2. prepare and, only when explicitly allowed and safe, launch real workload
   probes for the missing bridge service rows.

It never changes the legacy scheduler default and it refuses GPU launch when
the target devices already carry scheduler tasks or high utilization.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shlex
import time
from typing import Any, Mapping

from simulation.defaults import build_default_cache
from simulation.fast_forward import ReplayPolicy
from simulation.sota_baselines import sota_baseline_specs, sota_candidate_union_policy
from simulation.tasksets import taskset_by_name
from simulation.trace_benchmark import build_task_trace, replay_trace

from .remote_workload_selected_profile_probe import build_remote_workload_selected_profile_probe
from .sota_strict_dominance_frontier import build_sota_strict_dominance_frontier


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
SCHEDULER_PATH = REPO_ROOT / "skill" / "scheduler.py"

TARGET_GPU_NODES: dict[str, tuple[int, ...]] = {
    "jtl110gpu": (0, 1),
    "jtl110gpu2": (0, 1),
    "node007": (0, 1, 2, 3),
}

PROBE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "hybrid_rl_p2_p3_bridge",
        "workload_key": "hybrid_rl_resac_ant",
        "profiles": (2, 3),
        "template": "algorithm/experiments/templates/resac_ant_real_venv.cmd.tpl",
        "cwd": "/home/erzhu419/mine_code/RE-SAC",
        "output_root_suffix": "hybrid_rl_p2_p3_bridge",
        "max_iters": 80,
        "timeout_s": 420,
        "unit": "iter",
        "terminate_on_stable": True,
        "stable_windows": 3,
        "min_rate_samples": 5,
        "stable_cv": 0.20,
        "stable_rel_delta": 0.25,
        "stable_skip_samples": 1,
        "coordinated_profile_launch": True,
        "required_action": (
            "measure a p2/p3 bridge under the current resident-state fabric; "
            "admit only if lower-confidence service gives p3-level makespan with "
            "p2-level tail flow"
        ),
    },
    {
        "name": "cnn_tail_drain_bridge",
        "workload_key": "gpu_cnn_torch_resnet50",
        "profiles": (1, 2, 3),
        "template": "algorithm/experiments/templates/torch_cnn_resnet50.cmd.tpl",
        "cwd": "/home/erzhu419/mine_code/scheduleurm",
        "output_root_suffix": "cnn_tail_drain_bridge",
        "max_iters": 120,
        "timeout_s": 300,
        "unit": "step",
        "terminate_on_stable": True,
        "stable_windows": 3,
        "min_rate_samples": 5,
        "stable_cv": 0.12,
        "stable_rel_delta": 0.10,
        "stable_skip_samples": 1,
        "coordinated_profile_launch": False,
        "required_action": (
            "measure CNN p1/p2/p3 tail-drain candidates and co-location states; "
            "admit only if the tail-drain flow gain does not exceed the current "
            "makespan envelope"
        ),
    },
)


@dataclass(frozen=True)
class TargetTailSwitchPolicy(ReplayPolicy):
    """Wrap the Pareto-slack policy and switch one workload in the tail."""

    wrapped: ReplayPolicy | None = None
    workload_key: str = ""
    base_profile: int = 3
    tail_profile: int = 2
    threshold: int = 0

    def select_profile(self, cache, spec):  # type: ignore[override]
        if spec.workload_key == self.workload_key:
            profile = (
                self.tail_profile
                if int(spec.task_count) <= int(self.threshold)
                else self.base_profile
            )
            record = cache.get(spec.workload_key, profile)
            if record is None or record.capacity_boundary or record.aggregate_rate <= 0:
                raise KeyError(f"missing usable bridge profile {spec.workload_key}:{profile}")
            return record
        if self.wrapped is None:
            raise KeyError(f"no wrapped policy for {spec.workload_key}")
        return self.wrapped.select_profile(cache, spec)


def build_sota_bridge_action_gate(
    *,
    allow_launch: bool = False,
    run_id: str | None = None,
    max_gpu_util_pct: float = 20.0,
    max_gpu_mem_used_frac: float = 0.20,
) -> dict[str, Any]:
    run_id = run_id or time.strftime("sota_bridge_action_%Y%m%d_%H%M%S")
    frontier = build_sota_strict_dominance_frontier()
    phase_search = _phase_switch_search()
    resources = _gpu_resource_audit(
        max_gpu_util_pct=max_gpu_util_pct,
        max_gpu_mem_used_frac=max_gpu_mem_used_frac,
    )
    launch_plan = _launch_plan(resources, run_id=run_id)
    launch_results: list[dict[str, Any]] = []
    if allow_launch and launch_plan["launch_safe"]:
        launch_results = _launch_bridge_probes(launch_plan, run_id=run_id)
    status = _status(frontier, phase_search, resources, allow_launch, launch_results)
    return {
        "gate": "sota_bridge_action_gate",
        "status": status,
        "run_id": run_id,
        "allow_launch": bool(allow_launch),
        "measured_cache_external_policy_frontier": {
            "strict_pareto_ready": bool(frontier.get("strict_pareto_ready")),
            "within_tolerance_ready": bool(frontier.get("within_tolerance_ready")),
            "frontier_count": int(frontier.get("frontier_count") or 0),
            "frontier": frontier.get("frontier") or [],
            "status": frontier.get("status"),
        },
        "strict_frontier": {
            "strict_pareto_ready": bool(frontier.get("strict_pareto_ready")),
            "within_tolerance_ready": bool(frontier.get("within_tolerance_ready")),
            "frontier_count": int(frontier.get("frontier_count") or 0),
            "frontier": frontier.get("frontier") or [],
            "status": frontier.get("status"),
        },
        "phase_switch_search": phase_search,
        "resource_audit": resources,
        "launch_plan": launch_plan,
        "launch_results": launch_results,
        "current_cache_phase_switch_strict_ready": bool(
            phase_search.get("current_cache_phase_switch_strict_ready")
        ),
        "resource_launch_ready": bool(resources.get("launch_ready")),
        "real_probe_launched": bool(launch_results),
        "strict_bridge_ready": bool(frontier.get("strict_pareto_ready")),
        "scoped_claim_ready": True,
        "strong_claim_ready": False,
        "pass": True,
        "scope": (
            "This gate diagnoses and prepares bridge actions for strict "
            "measured-cache external-policy closure. It is not direct "
            "full-stack SOTA binary superiority and it does not mutate the legacy "
            "scheduler default."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    frontier = report.get("strict_frontier") or {}
    resources = report.get("resource_audit") or {}
    phase = report.get("phase_switch_search") or {}
    lines = [
        "# Measured-Cache External-Policy Bridge-Action Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `within_tolerance_ready` | {str(bool(frontier.get('within_tolerance_ready'))).lower()} |",
        f"| `strict_pareto_ready` | {str(bool(frontier.get('strict_pareto_ready'))).lower()} |",
        f"| `frontier_count` | {frontier.get('frontier_count', 0)} |",
        f"| `current_cache_phase_switch_strict_ready` | {str(bool(phase.get('current_cache_phase_switch_strict_ready'))).lower()} |",
        f"| `resource_launch_ready` | {str(bool(resources.get('launch_ready'))).lower()} |",
        f"| `real_probe_launched` | {str(bool(report.get('real_probe_launched'))).lower()} |",
        "",
        "## Phase-Switch Search",
        "",
    ]
    if frontier.get("strict_pareto_ready"):
        lines.extend(
            [
                "The measured-cache external-policy frontier is already closed by the accepted trajectory-action union. "
                "The phase-switch rows below are diagnostic only; they do not include the "
                "deterministic shuffle-static tail-drain bridge that closes the q01 CNN frontier.",
                "",
            ]
        )
    lines.extend(
        [
            "| Target | Best worst ratio | Strict ready | Best action | Diagnosis |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in phase.get("targets") or []:
        best = row.get("best_action") or {}
        lines.append(
            "| `{target}` | {worst:.9g} | {ready} | `{action}` | {diagnosis} |".format(
                target=row.get("target"),
                worst=float(row.get("best_worst_ratio") or 0.0),
                ready=str(bool(row.get("strict_ready"))).lower(),
                action=best.get("action_id") or "",
                diagnosis=row.get("diagnosis") or "",
            )
        )
    lines.extend([
        "",
        "## Resource Audit",
        "",
        "| Node | GPU | Used / total MiB | Util % | Scheduler blockers | Safe |",
        "|---|---:|---:|---:|---|---:|",
    ])
    for row in resources.get("gpu_rows") or []:
        lines.append(
            "| `{node}` | {gpu} | {used:.0f}/{total:.0f} | {util:.0f} | {blockers} | {safe} |".format(
                node=row.get("node"),
                gpu=row.get("gpu"),
                used=float(row.get("memory_used_mb") or 0.0),
                total=float(row.get("memory_total_mb") or 0.0),
                util=float(row.get("util_pct") or 0.0),
                blockers=", ".join(row.get("scheduler_blockers") or []),
                safe=str(bool(row.get("safe_for_bridge_probe"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Launch Plan",
        "",
        "| Probe | Launch safe | Command | Required action |",
        "|---|---:|---|---|",
    ])
    for row in (report.get("launch_plan") or {}).get("probes") or []:
        lines.append(
            "| `{name}` | {safe} | `{cmd}` | {required} |".format(
                name=row.get("name"),
                safe=str(bool(row.get("launch_safe"))).lower(),
                cmd=row.get("command") or "",
                required=row.get("required_action") or "",
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _phase_switch_search() -> dict[str, Any]:
    cache = build_default_cache()
    targets = [
        _search_target(
            cache,
            taskset_name="q01_gpu_bound_cnn_resnet50",
            workload_key="gpu_cnn_torch_resnet50",
            tail_profiles=(1, 2, 3),
            thresholds=(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96),
        ),
        _search_target(
            cache,
            taskset_name="hybrid_research_portfolio",
            workload_key="hybrid_rl_resac_ant",
            tail_profiles=(1, 2, 3, 4),
            thresholds=(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96),
        ),
    ]
    return {
        "targets": targets,
        "current_cache_phase_switch_strict_ready": all(
            bool(row.get("strict_ready")) for row in targets
        ),
        "meaning": (
            "finite phase-switch actions were searched on the current measured "
            "cache before launching new probes"
        ),
    }


def _search_target(
    cache,
    *,
    taskset_name: str,
    workload_key: str,
    tail_profiles: tuple[int, ...],
    thresholds: tuple[int, ...],
) -> dict[str, Any]:
    taskset = taskset_by_name(taskset_name)
    specs = taskset.workload_specs()
    wrapped = sota_candidate_union_policy(cache, specs, selection_objective="pareto_slack")
    baselines = [spec.policy for spec in sota_baseline_specs()]
    rows = []
    for tail_profile in tail_profiles:
        record = cache.get(workload_key, int(tail_profile))
        if record is None or record.capacity_boundary or record.aggregate_rate <= 0:
            continue
        for threshold in thresholds:
            policy = TargetTailSwitchPolicy(
                name=f"bridge_{workload_key}_p3_to_p{tail_profile}_T{threshold}",
                statewise=True,
                statewise_resource_kinds=("gpu_cnn", "hybrid_rl"),
                statewise_service_dominance_guard=False,
                wrapped=wrapped,
                workload_key=workload_key,
                base_profile=3,
                tail_profile=int(tail_profile),
                threshold=int(threshold),
            )
            metrics = []
            for arrival in ("static", "poisson"):
                trace = build_task_trace(taskset, arrival_mode=arrival, seed=42)
                sota = [replay_trace(cache, trace, base, seed=7) for base in baselines]
                best_ms = min(float(result.makespan_s) for result in sota)
                best_flow = min(float(result.mean_flow_s) for result in sota)
                result = replay_trace(cache, trace, policy, seed=7)
                metrics.append(
                    {
                        "arrival": arrival,
                        "makespan_ratio": best_ms / float(result.makespan_s),
                        "mean_flow_ratio": best_flow / float(result.mean_flow_s),
                        "makespan_s": float(result.makespan_s),
                        "mean_flow_s": float(result.mean_flow_s),
                        "profiles": result.profiles,
                    }
                )
            rows.append(
                {
                    "action_id": policy.name,
                    "tail_profile": int(tail_profile),
                    "threshold": int(threshold),
                    "worst_ratio": min(
                        min(float(item["makespan_ratio"]), float(item["mean_flow_ratio"]))
                        for item in metrics
                    ),
                    "strict_ready": all(
                        float(item["makespan_ratio"]) >= 1.0
                        and float(item["mean_flow_ratio"]) >= 1.0
                        for item in metrics
                    ),
                    "metrics": metrics,
                }
            )
    best = max(rows, key=lambda row: float(row["worst_ratio"])) if rows else {}
    strict = [row for row in rows if row.get("strict_ready")]
    return {
        "target": taskset_name,
        "workload_key": workload_key,
        "searched_action_count": len(rows),
        "strict_ready": bool(strict),
        "strict_action_count": len(strict),
        "best_worst_ratio": float(best.get("worst_ratio") or 0.0),
        "best_action": best,
        "diagnosis": (
            "current measured profiles do not contain a strict bridge action"
            if not strict
            else "current measured profiles contain a strict bridge action"
        ),
    }


def _gpu_resource_audit(
    *,
    max_gpu_util_pct: float,
    max_gpu_mem_used_frac: float,
) -> dict[str, Any]:
    scheduler = _load_scheduler_module()
    active = _active_gpu_tasks(scheduler)
    rows = []
    for node, target_gpus in TARGET_GPU_NODES.items():
        for gpu in _query_node_gpus(scheduler, node):
            if int(gpu["gpu"]) not in set(target_gpus):
                continue
            blockers = active.get((node, int(gpu["gpu"])), [])
            total = max(1.0, float(gpu.get("memory_total_mb") or 1.0))
            mem_frac = float(gpu.get("memory_used_mb") or 0.0) / total
            util = float(gpu.get("util_pct") or 0.0)
            safe = (
                not blockers
                and util <= float(max_gpu_util_pct)
                and mem_frac <= float(max_gpu_mem_used_frac)
            )
            rows.append(
                {
                    **gpu,
                    "scheduler_blockers": blockers,
                    "memory_used_frac": mem_frac,
                    "safe_for_bridge_probe": safe,
                }
            )
    safe_rows = [row for row in rows if row.get("safe_for_bridge_probe")]
    return {
        "max_gpu_util_pct": float(max_gpu_util_pct),
        "max_gpu_mem_used_frac": float(max_gpu_mem_used_frac),
        "gpu_rows": rows,
        "safe_gpu_count": len(safe_rows),
        "launch_ready": bool(safe_rows),
        "safe_gpus": safe_rows,
    }


def _launch_plan(resources: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    safe_gpus = list(resources.get("safe_gpus") or [])
    if not safe_gpus:
        selected = None
    else:
        selected = safe_gpus[0]
    probes = []
    for spec in PROBE_SPECS:
        if selected is None:
            node = "<waiting-for-safe-gpu>"
            gpu = "<gpu>"
            launch_safe = False
        else:
            node = str(selected["node"])
            gpu = str(int(selected["gpu"]))
            launch_safe = True
        probe_run_id = f"{run_id}_{spec['name']}"
        output_root = f"/tmp/scheduleurm_bridge_probes/{probe_run_id}/{spec['output_root_suffix']}"
        cmd = (
            "PYTHONPATH=. python3 -m algorithm.experiments.remote_workload_selected_profile_probe build "
            f"--run-id {shlex.quote(probe_run_id)} "
            f"--node {shlex.quote(node)} "
            f"--gpus {shlex.quote(gpu)} "
            f"--profiles {shlex.quote(','.join(str(x) for x in spec['profiles']))} "
            f"--cwd {shlex.quote(str(spec['cwd']))} "
            f"--cmd-template-file {shlex.quote(str(REPO_ROOT / spec['template']))} "
            f"--output-root {shlex.quote(output_root)} "
            f"--max-iters {int(spec['max_iters'])} "
            f"--timeout-s {int(spec['timeout_s'])} "
            f"--unit {shlex.quote(str(spec['unit']))} "
            f"{'--terminate-on-stable ' if bool(spec.get('terminate_on_stable')) else ''}"
            f"--stable-windows {int(spec.get('stable_windows', 3))} "
            f"--min-rate-samples {int(spec.get('min_rate_samples', 3))} "
            f"--stable-cv {float(spec.get('stable_cv', 0.08))} "
            f"--stable-rel-delta {float(spec.get('stable_rel_delta', 0.05))} "
            f"--stable-skip-samples {int(spec.get('stable_skip_samples', 0))} "
            f"{'--coordinated-profile-launch' if bool(spec.get('coordinated_profile_launch')) else ''}"
        )
        probes.append(
            {
                "name": spec["name"],
                "workload_key": spec["workload_key"],
                "node": node,
                "gpus": [gpu],
                "profiles": list(spec["profiles"]),
                "run_id": probe_run_id,
                "output_root": output_root,
                "command": cmd,
                "launch_safe": launch_safe,
                "required_action": spec["required_action"],
            }
        )
    return {
        "launch_safe": bool(selected is not None),
        "selected_gpu": selected,
        "probes": probes,
    }


def _launch_bridge_probes(launch_plan: Mapping[str, Any], *, run_id: str) -> list[dict[str, Any]]:
    results = []
    for probe in launch_plan.get("probes") or []:
        if not probe.get("launch_safe"):
            continue
        spec = next(item for item in PROBE_SPECS if item["name"] == probe["name"])
        result = build_remote_workload_selected_profile_probe(
            run_id=str(probe["run_id"]),
            node=str(probe["node"]),
            gpus=[int(x) for x in probe["gpus"]],
            profiles=[int(x) for x in probe["profiles"]],
            cwd=str(spec["cwd"]),
            cmd_template_file=str(REPO_ROOT / spec["template"]),
            output_root=str(probe["output_root"]),
            max_iters=int(spec["max_iters"]),
            timeout_s=int(spec["timeout_s"]),
            unit=str(spec["unit"]),
            terminate_on_stable=bool(spec.get("terminate_on_stable")),
            stable_windows=int(spec.get("stable_windows", 3)),
            min_rate_samples=int(spec.get("min_rate_samples", 3)),
            stable_cv=float(spec.get("stable_cv", 0.08)),
            stable_rel_delta=float(spec.get("stable_rel_delta", 0.05)),
            stable_skip_samples=int(spec.get("stable_skip_samples", 0)),
            coordinated_profile_launch=bool(spec.get("coordinated_profile_launch")),
        )
        results.append(result)
    return results


def _status(
    frontier: Mapping[str, Any],
    phase_search: Mapping[str, Any],
    resources: Mapping[str, Any],
    allow_launch: bool,
    launch_results: list[Mapping[str, Any]],
) -> str:
    if frontier.get("strict_pareto_ready"):
        return "MEASURED_CACHE_EXTERNAL_POLICY_BRIDGE_FRONTIER_ALREADY_CLOSED"
    if phase_search.get("current_cache_phase_switch_strict_ready"):
        return "MEASURED_CACHE_EXTERNAL_POLICY_BRIDGE_CURRENT_CACHE_ACTION_FOUND"
    if launch_results:
        return "MEASURED_CACHE_EXTERNAL_POLICY_BRIDGE_PROBES_LAUNCHED"
    if allow_launch and not resources.get("launch_ready"):
        return "MEASURED_CACHE_EXTERNAL_POLICY_BRIDGE_WAIT_RESOURCE"
    return "MEASURED_CACHE_EXTERNAL_POLICY_BRIDGE_PROBE_PLAN_READY"


def _load_scheduler_module():
    spec = importlib.util.spec_from_file_location("scheduleurm_scheduler_for_bridge_gate", SCHEDULER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduler module from {SCHEDULER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _active_gpu_tasks(scheduler) -> dict[tuple[str, int], list[str]]:
    state = scheduler.load_state()
    active_statuses = {"queued", "launching", "running"}
    blockers: dict[tuple[str, int], list[str]] = {}
    for task in state.get("tasks") or []:
        if str(task.get("status")) not in active_statuses:
            continue
        node = str(task.get("node") or "")
        gpu = task.get("gpu_idx")
        if gpu is None:
            continue
        key = (node, int(gpu))
        blockers.setdefault(key, []).append(str(task.get("id") or "?"))
    return blockers


def _query_node_gpus(scheduler, node: str) -> list[dict[str, Any]]:
    command = (
        "NVS=nvidia-smi; "
        "if [ -x /cm/local/apps/cuda-driver/libs/535.261.03/bin/nvidia-smi ]; then "
        "NVS=/cm/local/apps/cuda-driver/libs/535.261.03/bin/nvidia-smi; fi; "
        "$NVS --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null"
    )
    try:
        rc, out, err = scheduler.run_on(node, command, timeout=20, check=False)
    except Exception as exc:
        return [
            {
                "node": node,
                "gpu": -1,
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "util_pct": 100.0,
                "query_error": repr(exc),
            }
        ]
    rows = []
    if rc != 0:
        return [
            {
                "node": node,
                "gpu": -1,
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "util_pct": 100.0,
                "query_error": (err or out or f"rc={rc}")[:500],
            }
        ]
    for line in (out or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "node": node,
                    "gpu": int(parts[0]),
                    "memory_used_mb": float(parts[1]),
                    "memory_total_mb": float(parts[2]),
                    "util_pct": float(parts[3]),
                    "query_error": "",
                }
            )
        except ValueError:
            continue
    return rows


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_bridge_action_gate(
        allow_launch=bool(args.allow_launch),
        run_id=args.run_id,
        max_gpu_util_pct=args.max_gpu_util_pct,
        max_gpu_mem_used_frac=args.max_gpu_mem_used_frac,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_bridge_action_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build bridge-action search/resource/probe gate")
    build.add_argument("--allow-launch", action="store_true")
    build.add_argument("--run-id", default=None)
    build.add_argument("--max-gpu-util-pct", type=float, default=20.0)
    build.add_argument("--max-gpu-mem-used-frac", type=float, default=0.20)
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "sota_bridge_action_gate_20260613.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "sota_bridge_action_gate_20260613.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
