"""Multi-node original-deployment closure gate.

The existing external-runtime evidence is same-host and scoped.  This gate audits
whether there is evidence for broader multi-node original deployments and, when
requested, performs only read-only SSH inventory probes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .multinode_history_completion_gate import build_multinode_history_completion_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "multinode_original_deployment_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "multinode_original_deployment_gate_20260614.md"
SOTA_FULLSTACK = ARTIFACT_ROOT / "sota_fullstack_superiority_gate_20260613.json"
DEFAULT_NODES = (
    "jtl110gpu",
    "jtl110gpu2",
    "node001",
    "node002",
    "node003",
    "node004",
    "node005",
    "node006",
    "node007",
)


def build_multinode_original_deployment_gate(
    *,
    probe_remotes: bool = False,
    nodes: tuple[str, ...] = DEFAULT_NODES,
) -> dict[str, Any]:
    fullstack = _load_json(SOTA_FULLSTACK)
    same_host_ready = bool(fullstack.get("named_same_host_runtime_probe_ready"))
    history = build_multinode_history_completion_gate()
    inventory_rows = [_probe_node(node) for node in nodes] if probe_remotes else [
        {"node": node, "probed": False, "reachable": None, "gpu_count": None, "cpu_count": None, "note": "probe disabled"}
        for node in nodes
    ]
    reachable = [row for row in inventory_rows if row.get("reachable")]
    gpu_nodes = [row for row in reachable if int(row.get("gpu_count") or 0) > 0]
    cpu_nodes = [row for row in reachable if int(row.get("cpu_count") or 0) > 0]
    infrastructure_two_node_visible = len(reachable) >= 2
    gpu_two_node_visible = len(gpu_nodes) >= 2
    scheduleurm_multinode_history_ready = bool(
        history.get("scheduleurm_multinode_launched_completion_ready")
    )
    original_deployment_rows_ready = scheduleurm_multinode_history_ready
    return {
        "gate": "multinode_original_deployment_gate",
        "status": (
            "SCHEDULEURM_MULTINODE_HISTORY_READY_EXTERNAL_ORIGINAL_PENDING"
            if same_host_ready and scheduleurm_multinode_history_ready else
            "SAME_HOST_READY_MULTINODE_ORIGINAL_PENDING"
            if same_host_ready else
            "SAME_HOST_AND_MULTINODE_PENDING"
        ),
        "probe_remotes": bool(probe_remotes),
        "same_host_named_runtime_probe_ready": same_host_ready,
        "same_host_named_fullstack_ready": False,
        "remote_inventory_rows": inventory_rows,
        "reachable_node_count": len(reachable),
        "reachable_gpu_node_count": len(gpu_nodes),
        "reachable_cpu_node_count": len(cpu_nodes),
        "infrastructure_two_node_visible": infrastructure_two_node_visible,
        "gpu_two_node_visible": gpu_two_node_visible,
        "scheduleurm_multinode_history_completion_ready": scheduleurm_multinode_history_ready,
        "scheduleurm_multinode_history_snapshot": {
            "status": history.get("status"),
            "strict_launched_count": history.get("strict_launched_count"),
            "strict_completed_count": history.get("strict_completed_count"),
            "strict_unadmitted_count": history.get("strict_unadmitted_count"),
            "node_count": history.get("node_count"),
            "gpu_node_count": history.get("gpu_node_count"),
            "workload_domain_count": history.get("workload_domain_count"),
        },
        "original_multinode_deployment_rows_ready": original_deployment_rows_ready,
        "multinode_original_deployment_superiority_ready": False,
        "external_sota_original_multinode_superiority_ready": False,
        "scoped_claim_ready": bool(same_host_ready and scheduleurm_multinode_history_ready),
        "strong_claim_ready": False,
        "pass": True,
        "scope": (
            "Scheduleurm-native multi-node launched/completion history is "
            "certified separately from external SOTA original-deployment "
            "superiority.  Same-host external runtime-probe rows and Scheduleurm "
            "multi-node history do not imply that every external scheduler has "
            "been run through its original multi-node worker/control-plane path."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Multi-Node Original Deployment Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `probe_remotes` | {str(bool(report.get('probe_remotes'))).lower()} |",
        f"| `same_host_named_runtime_probe_ready` | {str(bool(report.get('same_host_named_runtime_probe_ready'))).lower()} |",
        f"| `same_host_named_fullstack_ready` | {str(bool(report.get('same_host_named_fullstack_ready'))).lower()} |",
        f"| `reachable_node_count` | {report.get('reachable_node_count')} |",
        f"| `reachable_gpu_node_count` | {report.get('reachable_gpu_node_count')} |",
        f"| `reachable_cpu_node_count` | {report.get('reachable_cpu_node_count')} |",
        f"| `infrastructure_two_node_visible` | {str(bool(report.get('infrastructure_two_node_visible'))).lower()} |",
        f"| `gpu_two_node_visible` | {str(bool(report.get('gpu_two_node_visible'))).lower()} |",
        f"| `scheduleurm_multinode_history_completion_ready` | {str(bool(report.get('scheduleurm_multinode_history_completion_ready'))).lower()} |",
        f"| `original_multinode_deployment_rows_ready` | {str(bool(report.get('original_multinode_deployment_rows_ready'))).lower()} |",
        f"| `multinode_original_deployment_superiority_ready` | {str(bool(report.get('multinode_original_deployment_superiority_ready'))).lower()} |",
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
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _probe_node(node: str) -> dict[str, Any]:
    script = (
        "set -e; "
        "echo HOST=$(hostname); "
        "echo CPU=$(nproc 2>/dev/null || echo 0); "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "echo GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l); "
        "else echo GPU=0; fi"
    )
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "node": node,
            "probed": True,
            "reachable": False,
            "cpu_count": None,
            "gpu_count": None,
            "note": str(exc),
        }
    parsed = _parse_probe(proc.stdout)
    return {
        "node": node,
        "probed": True,
        "reachable": proc.returncode == 0,
        "cpu_count": parsed.get("CPU"),
        "gpu_count": parsed.get("GPU"),
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-1000:],
        "note": "" if proc.returncode == 0 else (proc.stderr or proc.stdout)[-300:],
    }


def _parse_probe(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"CPU", "GPU"}:
            try:
                out[key] = int(str(value).strip())
            except ValueError:
                out[key] = 0
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:800]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    nodes = tuple(args.nodes.split(",")) if args.nodes else DEFAULT_NODES
    report = build_multinode_original_deployment_gate(
        probe_remotes=bool(args.probe_remotes),
        nodes=nodes,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.multinode_original_deployment_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build multi-node original-deployment gate")
    build.add_argument("--probe-remotes", action="store_true")
    build.add_argument("--nodes", default="")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
