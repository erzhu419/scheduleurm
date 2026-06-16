"""Remote full-stack runtime readiness gate for external SOTA baselines.

This gate records whether a remote machine can run Kubernetes GPU jobs for
direct SOTA runtime experiments.  It is intentionally narrower than a SOTA
superiority claim: a ready runtime base means Docker/Kubernetes/GPU plumbing is
usable, not that Gavel/Pollux/Sia/IADeep/Salus have been beaten full-stack.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .iadeep_fullstack_same_workload_gate import build_iadeep_fullstack_same_workload_gate
from .pollux_fullstack_same_workload_gate import build_pollux_fullstack_same_workload_gate
from .salus_fullstack_same_workload_gate import build_salus_fullstack_same_workload_gate
from .sia_fullstack_same_workload_gate import build_sia_fullstack_same_workload_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "fullstack_runtime_readiness_jtl110gpu_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "fullstack_runtime_readiness_jtl110gpu_20260613.md"
DEFAULT_REMOTE = "huiwei@jtl110gpu"
DEFAULT_KUBECONFIG = "/tmp/scheduleurm-k3s-docker/kubeconfig"


def build_fullstack_runtime_readiness_gate(
    *,
    remote: str = DEFAULT_REMOTE,
    kubeconfig: str = DEFAULT_KUBECONFIG,
    timeout_s: int = 30,
) -> dict[str, Any]:
    snapshot = _remote_snapshot(remote=remote, kubeconfig=kubeconfig, timeout_s=timeout_s)
    daemon = _parse_json(snapshot.get("docker_daemon_json", ""))
    node_json = _parse_json(snapshot.get("kubectl_nodes_json", ""))
    pods_json = _parse_json(snapshot.get("kubectl_pods_json", ""))
    runtime_json = _parse_json(snapshot.get("kubectl_runtimeclasses_json", ""))

    node = (node_json.get("items") or [{}])[0] if isinstance(node_json, dict) else {}
    allocatable = ((node.get("status") or {}).get("allocatable") or {}) if isinstance(node, dict) else {}
    capacity = ((node.get("status") or {}).get("capacity") or {}) if isinstance(node, dict) else {}
    conditions = ((node.get("status") or {}).get("conditions") or []) if isinstance(node, dict) else []
    ready_condition = next((c for c in conditions if c.get("type") == "Ready"), {})
    pods = pods_json.get("items") or [] if isinstance(pods_json, dict) else []
    runtime_classes = [
        str(item.get("metadata", {}).get("name"))
        for item in (runtime_json.get("items") or [])
        if item.get("metadata", {}).get("name")
    ] if isinstance(runtime_json, dict) else []
    device_plugin_pods = [
        _pod_summary(pod)
        for pod in pods
        if "nvidia-device-plugin" in str(pod.get("metadata", {}).get("name", ""))
    ]
    system_pods = [_pod_summary(pod) for pod in pods]
    gpu_smoke_log = snapshot.get("gpu_smoke_log", "")

    docker_default_runtime = str(daemon.get("default-runtime") or "")
    docker_has_nvidia_runtime = "nvidia" in (daemon.get("runtimes") or {})
    alloc_gpu = _int_quantity(allocatable.get("nvidia.com/gpu"))
    cap_gpu = _int_quantity(capacity.get("nvidia.com/gpu"))
    node_ready = ready_condition.get("status") == "True"
    device_plugin_running = any(pod.get("phase") == "Running" and pod.get("ready") for pod in device_plugin_pods)
    gpu_smoke_completed = "NVIDIA GeForce" in gpu_smoke_log and "RTX 3080 Ti" in gpu_smoke_log
    runtime_base_ready = all([
        node_ready,
        docker_default_runtime == "nvidia",
        docker_has_nvidia_runtime,
        cap_gpu >= 1,
        alloc_gpu >= 1,
        device_plugin_running,
        gpu_smoke_completed,
    ])
    pollux_fullstack = build_pollux_fullstack_same_workload_gate()
    pollux_ready = bool(pollux_fullstack.get("scoped_pollux_fullstack_same_workload_ready"))
    sia_fullstack = build_sia_fullstack_same_workload_gate()
    sia_ready = bool(sia_fullstack.get("scoped_sia_fullstack_same_workload_ready"))
    iadeep_fullstack = build_iadeep_fullstack_same_workload_gate()
    iadeep_ready = bool(iadeep_fullstack.get("scoped_iadeep_fullstack_same_workload_ready"))
    salus_fullstack = build_salus_fullstack_same_workload_gate()
    salus_ready = bool(salus_fullstack.get("scoped_salus_fullstack_same_workload_ready"))
    salus_blocker = "; ".join(salus_fullstack.get("blockers") or []) or "Salus server plus TensorFlow-Salus client workload binary are not yet completed on this host"

    return {
        "gate": "fullstack_runtime_readiness",
        "remote": remote,
        "kubeconfig": kubeconfig,
        "status": "K8S_GPU_RUNTIME_READY" if runtime_base_ready else "K8S_GPU_RUNTIME_NOT_READY",
        "runtime_base_ready": bool(runtime_base_ready),
        "direct_fullstack_sota_superiority_ready": False,
        "strong_claim_ready": False,
        "scoped_claim_ready": bool(runtime_base_ready),
        "pass": True,
        "runtime": {
            "hostname": snapshot.get("hostname", "").strip(),
            "nvidia_smi_query": snapshot.get("nvidia_smi_query", "").strip(),
            "docker_daemon_default_runtime": docker_default_runtime,
            "docker_has_nvidia_runtime": bool(docker_has_nvidia_runtime),
            "kubernetes_node_ready": bool(node_ready),
            "kubernetes_version": (node.get("status") or {}).get("nodeInfo", {}).get("kubeletVersion"),
            "container_runtime": (node.get("status") or {}).get("nodeInfo", {}).get("containerRuntimeVersion"),
            "capacity": capacity,
            "allocatable": allocatable,
            "runtime_classes": runtime_classes,
            "nvidia_device_plugin_running": bool(device_plugin_running),
            "gpu_smoke_completed": bool(gpu_smoke_completed),
        },
        "system_pods": system_pods,
        "device_plugin_pods": device_plugin_pods,
        "gpu_smoke_log_tail": "\n".join(gpu_smoke_log.splitlines()[-20:]),
        "full_stack_sota_rows": [
            {
                "adapter": "pollux_adaptdl_scheduler",
                "runtime_base_ready": bool(runtime_base_ready),
                "same_workload_runtime_ready": bool(pollux_ready),
                "blocker": "" if pollux_ready else "AdaptDL/Pollux scheduler image/deployment and same-workload AdaptDLJob completion are not yet certified by this gate",
                "evidence_artifact": str(ARTIFACT_ROOT / "pollux_fullstack_same_workload_gate_20260613.json"),
            },
            {
                "adapter": "sia_goodput_scheduler",
                "runtime_base_ready": bool(runtime_base_ready),
                "same_workload_runtime_ready": bool(sia_ready),
                "blocker": "" if sia_ready else "Sia physical-cluster path still requires AdaptDL deployment with Sia/mip policy and same-workload run_workload completion",
                "evidence_artifact": str(ARTIFACT_ROOT / "sia_fullstack_same_workload_gate_20260613.json"),
            },
            {
                "adapter": "iadeep_kubernetes_extender",
                "runtime_base_ready": bool(runtime_base_ready),
                "same_workload_runtime_ready": bool(iadeep_ready),
                "blocker": "" if iadeep_ready else "IADeep extender/coordinator/tuner and gpushare resource stack are not yet deployed on this K3s test cluster",
                "evidence_artifact": str(ARTIFACT_ROOT / "iadeep_fullstack_same_workload_gate_20260613.json"),
            },
            {
                "adapter": "salus_gpu_sharing",
                "runtime_base_ready": bool(runtime_base_ready),
                "same_workload_runtime_ready": bool(salus_ready),
                "blocker": "" if salus_ready else salus_blocker,
                "evidence_artifact": str(ARTIFACT_ROOT / "salus_fullstack_same_workload_gate_20260613.json"),
            },
        ],
        "scope": (
            "Certifies the remote Docker/K3s/NVIDIA device-plugin substrate for "
            "direct full-stack SOTA experiments.  It does not claim direct "
            "full-stack superiority over any SOTA system until that system's own "
            "runtime completes the same workload and is compared against "
            "Scheduleurm on JCT/makespan."
        ),
        "raw_snapshot": snapshot,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    runtime = report.get("runtime") or {}
    lines = [
        "# Full-Stack Runtime Readiness",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `runtime_base_ready` | {str(bool(report.get('runtime_base_ready'))).lower()} |",
        f"| `direct_fullstack_sota_superiority_ready` | {str(bool(report.get('direct_fullstack_sota_superiority_ready'))).lower()} |",
        f"| `remote` | `{report.get('remote')}` |",
        f"| `hostname` | `{_md(runtime.get('hostname'))}` |",
        f"| `docker_default_runtime` | `{_md(runtime.get('docker_daemon_default_runtime'))}` |",
        f"| `kubernetes_node_ready` | {str(bool(runtime.get('kubernetes_node_ready'))).lower()} |",
        f"| `nvidia_device_plugin_running` | {str(bool(runtime.get('nvidia_device_plugin_running'))).lower()} |",
        f"| `gpu_smoke_completed` | {str(bool(runtime.get('gpu_smoke_completed'))).lower()} |",
        "",
        "## GPU Capacity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| `capacity.nvidia.com/gpu` | `{_md((runtime.get('capacity') or {}).get('nvidia.com/gpu'))}` |",
        f"| `allocatable.nvidia.com/gpu` | `{_md((runtime.get('allocatable') or {}).get('nvidia.com/gpu'))}` |",
        f"| `nvidia-smi` | `{_md(runtime.get('nvidia_smi_query'))}` |",
        "",
        "## SOTA Runtime Rows",
        "",
        "| Adapter | Runtime base | Same-workload runtime | Blocker |",
        "|---|---:|---:|---|",
    ]
    for row in report.get("full_stack_sota_rows") or []:
        lines.append(
            "| `{adapter}` | {base} | {ready} | {blocker} |".format(
                adapter=row.get("adapter"),
                base=str(bool(row.get("runtime_base_ready"))).lower(),
                ready=str(bool(row.get("same_workload_runtime_ready"))).lower(),
                blocker=_md(row.get("blocker")),
            )
        )
    lines.extend([
        "",
        "## GPU Smoke Log",
        "",
        "```text",
        str(report.get("gpu_smoke_log_tail") or "").strip(),
        "```",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _remote_snapshot(*, remote: str, kubeconfig: str, timeout_s: int) -> dict[str, str]:
    commands = {
        "hostname": "hostname",
        "os_release": "cat /etc/os-release 2>/dev/null || true",
        "nvidia_smi_query": "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true",
        "docker_daemon_json": "cat /etc/docker/daemon.json 2>/dev/null || true",
        "kubectl_nodes_json": f"KUBECONFIG={_sh(kubeconfig)} kubectl get nodes -o json 2>/dev/null || true",
        "kubectl_pods_json": f"KUBECONFIG={_sh(kubeconfig)} kubectl get pods -A -o json 2>/dev/null || true",
        "kubectl_runtimeclasses_json": f"KUBECONFIG={_sh(kubeconfig)} kubectl get runtimeclass -o json 2>/dev/null || true",
        "gpu_smoke_log": f"KUBECONFIG={_sh(kubeconfig)} kubectl logs job/scheduleurm-gpu-smoke 2>/dev/null || true",
    }
    return {
        key: _run_ssh(remote, command, timeout_s=timeout_s)
        for key, command in commands.items()
    }


def _run_ssh(remote: str, command: str, *, timeout_s: int) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", remote, command],
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def _int_quantity(value: Any) -> int:
    try:
        return int(str(value))
    except Exception:
        return 0


def _pod_summary(pod: Mapping[str, Any]) -> dict[str, Any]:
    status = pod.get("status") or {}
    container_statuses = status.get("containerStatuses") or []
    ready = bool(container_statuses) and all(bool(item.get("ready")) for item in container_statuses)
    return {
        "namespace": pod.get("metadata", {}).get("namespace"),
        "name": pod.get("metadata", {}).get("name"),
        "phase": status.get("phase"),
        "ready": ready,
        "node": (pod.get("spec") or {}).get("nodeName"),
    }


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " / ")


def _sh(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_fullstack_runtime_readiness_gate(
        remote=args.remote,
        kubeconfig=args.kubeconfig,
        timeout_s=args.timeout_s,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.fullstack_runtime_readiness_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build remote full-stack runtime readiness certificate")
    build.add_argument("--remote", default=DEFAULT_REMOTE)
    build.add_argument("--kubeconfig", default=DEFAULT_KUBECONFIG)
    build.add_argument("--timeout-s", type=int, default=30)
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
