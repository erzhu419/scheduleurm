"""Cross-node theorem-dispatch shadow gate.

The original multi-node full-stack claim needs real launches through the
original control plane.  This gate is a safer intermediate certificate: it
audits whether the theorem-facing action family can enumerate certified
lower-service candidates across multiple visible nodes and select a global
robust-MaxWeight configuration without launching work.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.service_registry import bind_service, default_service_cache

from .multinode_original_deployment_gate import DEFAULT_NODES, _probe_node


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "multinode_theorem_shadow_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "multinode_theorem_shadow_gate_20260614.md"

SYNTHETIC_WORKLOADS = (
    {
        "task_id": "shadow-q01-gpu-heavy",
        "theorem_workload_key": "gpu_heavy_jax_matmul",
        "resource": "gpu",
        "queue_weight": 8.0,
        "est_vram_mb": 1024,
        "cpu": 1,
    },
    {
        "task_id": "shadow-q01-cnn",
        "theorem_workload_key": "gpu_cnn_torch_progress_stack",
        "resource": "gpu",
        "queue_weight": 6.0,
        "est_vram_mb": 2048,
        "cpu": 2,
    },
    {
        "task_id": "shadow-q01-llm",
        "theorem_workload_key": "gpu_llm_torch_decoder_stack",
        "resource": "gpu",
        "queue_weight": 5.0,
        "est_vram_mb": 3072,
        "cpu": 2,
    },
    {
        "task_id": "shadow-q11-hybrid",
        "theorem_workload_key": "hybrid_rl_resac_ant",
        "resource": "gpu",
        "queue_weight": 7.0,
        "est_vram_mb": 1536,
        "cpu": 4,
    },
    {
        "task_id": "shadow-q10-cpu",
        "theorem_workload_key": "cpu_heavy_local_bench",
        "resource": "cpu",
        "queue_weight": 4.0,
        "est_vram_mb": 0,
        "cpu": 16,
    },
    {
        "task_id": "shadow-q00-light",
        "theorem_workload_key": "light_control_local",
        "resource": "cpu",
        "queue_weight": 2.0,
        "est_vram_mb": 0,
        "cpu": 1,
    },
)


def build_multinode_theorem_shadow_gate(
    *,
    probe_remotes: bool = False,
    nodes: tuple[str, ...] = DEFAULT_NODES,
    max_batch_size: int = 6,
) -> dict[str, Any]:
    inventory = [_probe_node(node) for node in nodes] if probe_remotes else [
        {
            "node": node,
            "probed": False,
            "reachable": None,
            "cpu_count": None,
            "gpu_count": None,
            "note": "probe disabled",
        }
        for node in nodes
    ]
    candidate_rows = _candidate_rows(inventory)
    queue = {
        str(item["theorem_workload_key"]): float(item["queue_weight"])
        for item in SYNTHETIC_WORKLOADS
    }
    result = select_global_action(
        candidate_rows,
        queue,
        max_batch_size=max_batch_size,
        max_configurations=50000,
        lookahead_weight=0.05,
    )
    reachable = [row for row in inventory if row.get("reachable")]
    gpu_nodes = [row for row in reachable if int(row.get("gpu_count") or 0) > 0]
    cpu_nodes = [row for row in reachable if int(row.get("cpu_count") or 0) > 0]
    selected_nodes = {
        str(row.get("node") or "")
        for row in candidate_rows
        if row.get("action_id") in set(result.selected_action_ids)
    }
    selected_nodes.discard("")
    theorem_ready_count = sum(1 for row in candidate_rows if row.get("theorem_ready"))
    cross_node_candidate_ready = len(gpu_nodes) >= 2 and theorem_ready_count > 0
    cross_node_selected_ready = len(selected_nodes) >= 2
    shadow_ready = bool(
        cross_node_candidate_ready
        and result.candidate_configuration_count > 0
        and result.oracle_gap_alpha0 == 0.0
        and result.oracle_gap_alpha1 == 0.0
    )
    return {
        "gate": "multinode_theorem_shadow_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "MULTINODE_THEOREM_SHADOW_CLOSED"
            if shadow_ready else "MULTINODE_THEOREM_SHADOW_PENDING"
        ),
        "probe_remotes": bool(probe_remotes),
        "remote_inventory_rows": inventory,
        "reachable_node_count": len(reachable),
        "reachable_gpu_node_count": len(gpu_nodes),
        "reachable_cpu_node_count": len(cpu_nodes),
        "candidate_row_count": len(candidate_rows),
        "theorem_candidate_row_count": theorem_ready_count,
        "candidate_nodes": sorted({str(row.get("node") or "") for row in candidate_rows}),
        "selected_nodes": sorted(selected_nodes),
        "selected_action_ids": list(result.selected_action_ids),
        "selected_task_ids": list(result.selected_task_ids),
        "candidate_configuration_count": int(result.candidate_configuration_count),
        "oracle_gap_alpha0": float(result.oracle_gap_alpha0),
        "oracle_gap_alpha1": float(result.oracle_gap_alpha1),
        "cross_node_candidate_family_ready": bool(cross_node_candidate_ready),
        "cross_node_selected_configuration_ready": bool(cross_node_selected_ready),
        "multinode_theorem_shadow_ready": bool(shadow_ready),
        "multinode_original_launch_claim_ready": False,
        "candidate_resource_counts": dict(Counter(str(row.get("resource_kind") or "") for row in candidate_rows)),
        "candidate_rows": candidate_rows[:500],
        "scoped_claim_ready": bool(shadow_ready),
        "strong_claim_ready": False,
        "pass": bool(shadow_ready),
        "blocker": "" if shadow_ready else _blocker(inventory, candidate_rows),
        "scope": (
            "Read-only cross-node theorem candidate-family shadow.  It shows "
            "that certified lower-service candidates can be enumerated and "
            "globally scored across visible nodes.  It is not an original "
            "multi-node full-stack launched-completion comparison."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Multi-Node Theorem Shadow Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `probe_remotes` | {str(bool(report.get('probe_remotes'))).lower()} |",
        f"| `reachable_node_count` | {report.get('reachable_node_count')} |",
        f"| `reachable_gpu_node_count` | {report.get('reachable_gpu_node_count')} |",
        f"| `reachable_cpu_node_count` | {report.get('reachable_cpu_node_count')} |",
        f"| `candidate_row_count` | {report.get('candidate_row_count')} |",
        f"| `theorem_candidate_row_count` | {report.get('theorem_candidate_row_count')} |",
        f"| `candidate_configuration_count` | {report.get('candidate_configuration_count')} |",
        f"| `cross_node_candidate_family_ready` | {str(bool(report.get('cross_node_candidate_family_ready'))).lower()} |",
        f"| `cross_node_selected_configuration_ready` | {str(bool(report.get('cross_node_selected_configuration_ready'))).lower()} |",
        f"| `multinode_theorem_shadow_ready` | {str(bool(report.get('multinode_theorem_shadow_ready'))).lower()} |",
        f"| `multinode_original_launch_claim_ready` | {str(bool(report.get('multinode_original_launch_claim_ready'))).lower()} |",
        "",
        "## Selected Actions",
        "",
        ", ".join(f"`{item}`" for item in report.get("selected_action_ids") or []) or "none",
        "",
        "## Node Inventory",
        "",
        "| Node | Probed | Reachable | CPUs | GPUs | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report.get("remote_inventory_rows") or []:
        lines.append(
            "| `{node}` | {probed} | {reachable} | `{cpu}` | `{gpu}` | {note} |".format(
                node=row.get("node"),
                probed=str(bool(row.get("probed"))).lower(),
                reachable=str(bool(row.get("reachable"))).lower() if row.get("reachable") is not None else "NA",
                cpu=row.get("cpu_count"),
                gpu=row.get("gpu_count"),
                note=_md(row.get("note") or row.get("stderr") or ""),
            )
        )
    lines.extend(["", "## Blocker", "", str(report.get("blocker") or "none"), "", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _candidate_rows(inventory: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cache = default_service_cache()
    rows: list[dict[str, Any]] = []
    for node in inventory:
        if not node.get("reachable"):
            continue
        node_name = str(node.get("node") or "")
        gpu_count = int(node.get("gpu_count") or 0)
        cpu_count = int(node.get("cpu_count") or 0)
        for task in SYNTHETIC_WORKLOADS:
            resource = str(task["resource"])
            if resource == "gpu":
                for gpu_idx in range(max(0, gpu_count)):
                    rows.append(
                        _candidate_row(
                            task,
                            node=node_name,
                            resource_id=f"{node_name}:gpu:{gpu_idx}",
                            post_task_count=1,
                            resource_kind="gpu",
                            cache=cache,
                        )
                    )
            elif cpu_count > 0:
                rows.append(
                    _candidate_row(
                        task,
                        node=node_name,
                        resource_id=f"{node_name}:cpu",
                        post_task_count=max(1, min(64, int(task.get("cpu") or 1))),
                        resource_kind="cpu",
                        cache=cache,
                    )
                )
    return [row for row in rows if row.get("theorem_ready")]


def _candidate_row(
    task: Mapping[str, Any],
    *,
    node: str,
    resource_id: str,
    post_task_count: int,
    resource_kind: str,
    cache: Any,
) -> dict[str, Any]:
    features = {
        "post_task_count": int(post_task_count),
        "running_task_count": max(0, int(post_task_count) - 1),
        "candidate_bucket": f"{node}:{resource_kind}",
        "class_key": task.get("theorem_workload_key"),
        "regime_key": "multinode_shadow",
    }
    binding = bind_service(task, features, cache=cache, workload_key=str(task["theorem_workload_key"]))
    lower = binding.lower_service_vector()
    q = float(task.get("queue_weight") or 0.0)
    penalty = 0.0
    score = q * float(binding.lower_service) - penalty
    return {
        "action_id": f"{task['task_id']}|{resource_id}|p{binding.profile}",
        "task_id": task["task_id"],
        "node": node,
        "resource_id": resource_id,
        "resource_ids": [resource_id],
        "resource_kind": resource_kind,
        "workload_key": binding.workload_key,
        "lower_service": lower,
        "selected_class_lower_service": float(binding.lower_service),
        "penalty_units": penalty,
        "robust_maxweight_score": float(score),
        "score_semantics": (
            "robust_maxweight_lower_service"
            if binding.certified and lower else "scheduler_sort_key_minimization"
        ),
        "theorem_ready": bool(binding.certified and lower),
        "service_binding": binding.snapshot(),
    }


def _blocker(inventory: list[Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> str:
    reachable_gpu = [
        row for row in inventory
        if row.get("reachable") and int(row.get("gpu_count") or 0) > 0
    ]
    if len(reachable_gpu) < 2:
        return "fewer than two reachable GPU nodes under SSH BatchMode; run again when multi-node fabric is reachable"
    if not rows:
        return "no theorem-certified service rows for the synthetic cross-node workload mix"
    return "global robust-MaxWeight cross-node candidate enumeration did not produce a closed shadow certificate"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:900]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    nodes = tuple(args.nodes.split(",")) if args.nodes else DEFAULT_NODES
    report = build_multinode_theorem_shadow_gate(
        probe_remotes=bool(args.probe_remotes),
        nodes=nodes,
        max_batch_size=args.max_batch_size,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.multinode_theorem_shadow_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build read-only cross-node theorem shadow")
    build.add_argument("--probe-remotes", action="store_true")
    build.add_argument("--nodes", default="")
    build.add_argument("--max-batch-size", type=int, default=6)
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
