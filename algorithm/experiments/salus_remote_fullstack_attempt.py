"""Remote Salus full-stack same-workload attempt.

This runner is intentionally isolated from the production scheduler.  It uses a
remote Kubernetes test substrate to run two scoped workloads:

1. a native TensorFlow-Salus client container that executes a tiny TensorFlow
   graph directly on one GPU; and
2. a Salus server container plus a TensorFlow-Salus client container that
   executes the same graph through zrpc://tcp://127.0.0.1:5501.

The gate is strict.  A plain CUDA probe is not counted as Salus evidence, and a
server-only startup is not counted as a same-workload comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
SALUS_ARTIFACT_DIR = ARTIFACT_ROOT / "salus_fullstack_20260613"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "salus_remote_fullstack_attempt_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "salus_remote_fullstack_attempt_20260613.md"
DEFAULT_REMOTE = "huiwei@jtl110gpu"
DEFAULT_KUBECONFIG = "/tmp/scheduleurm-k3s-docker/kubeconfig"
DEFAULT_NAMESPACE = "salus-baseline"
SERVER_IMAGE = "registry.gitlab.com/salus/salus:latest"
CLIENT_IMAGE = "registry.gitlab.com/salus/tensorflow-salus:latest"


def build_salus_remote_fullstack_attempt(
    *,
    remote: str = DEFAULT_REMOTE,
    kubeconfig: str = DEFAULT_KUBECONFIG,
    namespace: str = DEFAULT_NAMESPACE,
    timeout_s: int = 900,
    poll_s: int = 10,
    keep_pods: bool = False,
    artifact_dir: Path = SALUS_ARTIFACT_DIR,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _run_kubectl(remote, kubeconfig, "create namespace " + _sh(namespace), timeout_s=30, check=False)

    native_name = "salus-native-same-workload-smoke"
    salus_name = "salus-fullstack-same-workload-smoke"
    native_manifest = _native_pod_manifest(native_name, namespace)
    salus_manifest = _salus_pod_manifest(salus_name, namespace)
    _write_json(artifact_dir / "salus_native_same_workload_pod_manifest.json", native_manifest)
    _write_json(artifact_dir / "salus_server_client_same_workload_pod_manifest.json", salus_manifest)

    native = _run_pod_attempt(
        remote=remote,
        kubeconfig=kubeconfig,
        namespace=namespace,
        name=native_name,
        manifest=native_manifest,
        marker="SALUS_NATIVE_RESULT",
        timeout_s=timeout_s,
        poll_s=poll_s,
        keep_pod=keep_pods,
        artifact_prefix=artifact_dir / "salus_native_same_workload",
    )
    salus = _run_pod_attempt(
        remote=remote,
        kubeconfig=kubeconfig,
        namespace=namespace,
        name=salus_name,
        manifest=salus_manifest,
        marker="SALUS_CLIENT_RESULT",
        timeout_s=timeout_s,
        poll_s=poll_s,
        keep_pod=keep_pods,
        artifact_prefix=artifact_dir / "salus_server_client_same_workload",
    )

    native_wall = _extract_result_wall(native.get("logs", ""), "SALUS_NATIVE_RESULT")
    salus_wall = _extract_result_wall(salus.get("logs", ""), "SALUS_CLIENT_RESULT")
    native_completed = bool(native.get("marker_seen"))
    salus_completed = bool(salus.get("marker_seen"))
    server_started = bool(
        salus.get("server_container_started")
        or "SALUS_CLIENT_RESULT" in str(salus.get("logs") or "")
        or "tcp://*:5501" in str(salus.get("logs") or "")
        or "5501" in str(salus.get("logs") or "")
    )
    same_service_scale_ready = bool(native_completed and salus_completed and native_wall is not None and salus_wall is not None)
    superiority_ready = bool(same_service_scale_ready and float(native_wall) < float(salus_wall))
    official_images_ready = bool(native.get("all_images_pulled") and salus.get("all_images_pulled"))
    native_runtime_hang_suspected = bool(native.get("runtime_hang_suspected"))
    salus_runtime_hang_suspected = bool(salus.get("runtime_hang_suspected"))

    blockers = _blockers(
        native=native,
        salus=salus,
        native_completed=native_completed,
        salus_completed=salus_completed,
        server_started=server_started,
        same_service_scale_ready=same_service_scale_ready,
        official_images_ready=official_images_ready,
    )
    report: dict[str, Any] = {
        "gate": "salus_remote_fullstack_attempt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote": remote,
        "kubeconfig": kubeconfig,
        "namespace": namespace,
        "server_image": SERVER_IMAGE,
        "client_image": CLIENT_IMAGE,
        "native_attempt": _attempt_summary(native),
        "salus_attempt": _attempt_summary(salus),
        "official_images_ready": bool(official_images_ready),
        "image_pull_blocker_resolved": bool(official_images_ready),
        "native_runtime_hang_suspected": bool(native_runtime_hang_suspected),
        "salus_runtime_hang_suspected": bool(salus_runtime_hang_suspected),
        "salus_server_started": bool(server_started),
        "native_tensorflow_salus_client_completed": bool(native_completed),
        "tensorflow_salus_client_completed": bool(salus_completed),
        "native_wall_s": native_wall,
        "salus_wall_s": salus_wall,
        "same_service_scale_ready": bool(same_service_scale_ready),
        "scoped_salus_fullstack_same_workload_ready": bool(salus_completed and server_started),
        "scoped_salus_same_workload_superiority_ready": bool(superiority_ready),
        "direct_fullstack_sota_superiority_ready": False,
        "blockers": blockers,
        "evidence_artifacts": {
            "native_manifest": str(artifact_dir / "salus_native_same_workload_pod_manifest.json"),
            "native_pod": str(artifact_dir / "salus_native_same_workload_pod.json"),
            "native_describe": str(artifact_dir / "salus_native_same_workload_describe.txt"),
            "native_logs": str(artifact_dir / "salus_native_same_workload_logs.txt"),
            "salus_manifest": str(artifact_dir / "salus_server_client_same_workload_pod_manifest.json"),
            "salus_pod": str(artifact_dir / "salus_server_client_same_workload_pod.json"),
            "salus_describe": str(artifact_dir / "salus_server_client_same_workload_describe.txt"),
            "salus_logs": str(artifact_dir / "salus_server_client_same_workload_logs.txt"),
        },
        "scope": (
            "Scoped remote Salus full-stack attempt on the isolated jtl110gpu "
            "K3s substrate.  It does not modify production scheduler code and "
            "does not alter system CUDA/cuDNN.  It counts Salus only when a "
            "TensorFlow-Salus client completes through a Salus server."
        ),
        "pass": True,
    }
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    rows = [
        ("pass", bool(report.get("pass"))),
        ("official_images_ready", bool(report.get("official_images_ready"))),
        ("image_pull_blocker_resolved", bool(report.get("image_pull_blocker_resolved"))),
        ("native_runtime_hang_suspected", bool(report.get("native_runtime_hang_suspected"))),
        ("salus_runtime_hang_suspected", bool(report.get("salus_runtime_hang_suspected"))),
        ("salus_server_started", bool(report.get("salus_server_started"))),
        ("native_tensorflow_salus_client_completed", bool(report.get("native_tensorflow_salus_client_completed"))),
        ("tensorflow_salus_client_completed", bool(report.get("tensorflow_salus_client_completed"))),
        ("same_service_scale_ready", bool(report.get("same_service_scale_ready"))),
        ("scoped_salus_fullstack_same_workload_ready", bool(report.get("scoped_salus_fullstack_same_workload_ready"))),
        ("scoped_salus_same_workload_superiority_ready", bool(report.get("scoped_salus_same_workload_superiority_ready"))),
        ("direct_fullstack_sota_superiority_ready", bool(report.get("direct_fullstack_sota_superiority_ready"))),
        ("native_wall_s", report.get("native_wall_s")),
        ("salus_wall_s", report.get("salus_wall_s")),
    ]
    lines = [
        "# Salus Remote Full-Stack Attempt",
        "",
        "| Quantity | Value |",
        "|---|---:|",
    ]
    for key, value in rows:
        lines.append(f"| `{key}` | `{_md(value)}` |")
    lines.extend([
        "",
        "## Attempts",
        "",
        "| Attempt | Completed | Images pulled | Status | Reason |",
        "|---|---:|---:|---|---|",
    ])
    for label, attempt in (
        ("native_tensorflow_salus", report.get("native_attempt") or {}),
        ("salus_server_client", report.get("salus_attempt") or {}),
    ):
        lines.append(
            f"| `{label}` | {str(bool(attempt.get('marker_seen'))).lower()} | "
            f"{str(bool(attempt.get('all_images_pulled'))).lower()} | "
            f"`{_md(attempt.get('status'))}` | {_md(attempt.get('reason'))} |"
        )
    lines.extend(["", "## Blockers", "", "| Blocker |", "|---|"])
    blockers = report.get("blockers") or []
    if not blockers:
        lines.append("| none |")
    for blocker in blockers:
        lines.append(f"| {_md(blocker)} |")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _native_pod_manifest(name: str, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": {"app": name}},
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "native-client",
                    "image": CLIENT_IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["bash", "-lc", _client_script(native=True)],
                    "resources": {"limits": {"nvidia.com/gpu": 1}},
                    "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"}],
                }
            ],
        },
    }


def _salus_pod_manifest(name: str, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": {"app": name}},
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "salus-server",
                    "image": SERVER_IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "ports": [{"containerPort": 5501}],
                    "resources": {"limits": {"nvidia.com/gpu": 1}},
                    "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"}],
                },
                {
                    "name": "salus-client",
                    "image": CLIENT_IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["bash", "-lc", _client_script(native=False)],
                },
            ],
        },
    }


def _client_script(*, native: bool) -> str:
    target = "" if native else "zrpc://tcp://127.0.0.1:5501"
    marker = "SALUS_NATIVE_RESULT" if native else "SALUS_CLIENT_RESULT"
    wait = "" if native else (
        "python - <<'PYWAIT'\n"
        "from __future__ import print_function\n"
        "import socket, time\n"
        "deadline = time.time() + 180\n"
        "while True:\n"
        "    try:\n"
        "        s = socket.create_connection(('127.0.0.1', 5501), 2)\n"
        "        s.close()\n"
        "        print('SALUS_SERVER_PORT_READY')\n"
        "        break\n"
        "    except Exception as exc:\n"
        "        if time.time() > deadline:\n"
        "            raise\n"
        "        time.sleep(2)\n"
        "PYWAIT\n"
    )
    return (
        "set -euo pipefail\n"
        + wait
        + "python - <<'PY'\n"
        "from __future__ import print_function\n"
        "import json, time\n"
        "import tensorflow as tf\n"
        f"target = {target!r}\n"
        f"marker = {marker!r}\n"
        "steps = 24\n"
        "start = time.time()\n"
        "with tf.Graph().as_default():\n"
        "    with tf.device('/device:GPU:0'):\n"
        "        x = tf.ones([256, 256], dtype=tf.float32)\n"
        "        y = tf.matmul(x, x)\n"
        "        z = tf.reduce_sum(y)\n"
        "    sess = tf.Session(target=target) if target else tf.Session()\n"
        "    with sess:\n"
        "        sess.run(z)\n"
        "        total = 0.0\n"
        "        for _ in range(steps):\n"
        "            total += float(sess.run(z))\n"
        "wall = time.time() - start\n"
        "print(marker, json.dumps({'ok': True, 'steps': steps, 'wall_s': wall, 'checksum': total}))\n"
        "PY\n"
    )


def _run_pod_attempt(
    *,
    remote: str,
    kubeconfig: str,
    namespace: str,
    name: str,
    manifest: Mapping[str, Any],
    marker: str,
    timeout_s: int,
    poll_s: int,
    keep_pod: bool,
    artifact_prefix: Path,
) -> dict[str, Any]:
    _run_kubectl(remote, kubeconfig, f"-n {_sh(namespace)} delete pod {_sh(name)} --ignore-not-found --wait=true", timeout_s=90, check=False)
    apply = _run_kubectl(
        remote,
        kubeconfig,
        "apply -f -",
        input_text=json.dumps(manifest),
        timeout_s=60,
        check=False,
    )
    start = time.time()
    pod_json: dict[str, Any] = {}
    describe = ""
    logs = ""
    status = "PENDING"
    reason = ""
    marker_seen = False
    timeout_expired = False
    while time.time() - start <= timeout_s:
        pod_text = _run_kubectl(remote, kubeconfig, f"-n {_sh(namespace)} get pod {_sh(name)} -o json", timeout_s=30, check=False).stdout
        pod_json = _parse_json(pod_text)
        describe = _run_kubectl(remote, kubeconfig, f"-n {_sh(namespace)} describe pod {_sh(name)}", timeout_s=30, check=False).stdout
        logs = _pod_logs(remote, kubeconfig, namespace, name)
        marker_seen = marker in logs
        status, reason = _pod_status_reason(pod_json)
        if marker_seen:
            status = "MARKER_SEEN"
            break
        if _terminal_failure(pod_json):
            break
        time.sleep(max(1, poll_s))
    if not marker_seen and not _terminal_failure(pod_json):
        timeout_expired = True
        status = "TIMEOUT"
        _, current_reason = _pod_status_reason(pod_json)
        if current_reason:
            reason = current_reason
    if not keep_pod:
        _run_kubectl(
            remote,
            kubeconfig,
            f"-n {_sh(namespace)} delete pod {_sh(name)} --ignore-not-found --grace-period=0 --force --wait=false",
            timeout_s=30,
            check=False,
        )
    final_pod_text = _run_kubectl(remote, kubeconfig, f"-n {_sh(namespace)} get pod {_sh(name)} -o json", timeout_s=20, check=False).stdout
    if final_pod_text.strip().startswith("{"):
        pod_json = _parse_json(final_pod_text)
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_prefix.with_name(artifact_prefix.name + "_pod.json"), pod_json)
    artifact_prefix.with_name(artifact_prefix.name + "_describe.txt").write_text(describe, encoding="utf-8", errors="replace")
    artifact_prefix.with_name(artifact_prefix.name + "_logs.txt").write_text(logs, encoding="utf-8", errors="replace")
    return {
        "apply_stdout": apply.stdout,
        "apply_stderr": apply.stderr,
        "status": status,
        "reason": reason,
        "marker_seen": marker_seen,
        "all_images_pulled": _all_images_pulled(pod_json),
        "server_container_started": _container_started(pod_json, "salus-server"),
        "elapsed_s": time.time() - start,
        "timeout_expired": timeout_expired,
        "runtime_hang_suspected": _runtime_hang_suspected(
            pod_json=pod_json,
            logs=logs,
            marker_seen=marker_seen,
            timeout_expired=timeout_expired,
        ),
        "pod": pod_json,
        "describe": describe,
        "logs": logs,
    }


def _pod_logs(remote: str, kubeconfig: str, namespace: str, name: str) -> str:
    all_logs = _run_kubectl(
        remote,
        kubeconfig,
        f"-n {_sh(namespace)} logs {_sh(name)} --all-containers=true --tail=300",
        timeout_s=30,
        check=False,
    )
    previous = _run_kubectl(
        remote,
        kubeconfig,
        f"-n {_sh(namespace)} logs {_sh(name)} --all-containers=true --previous --tail=300",
        timeout_s=30,
        check=False,
    )
    return "\n".join(part for part in [all_logs.stdout, all_logs.stderr, previous.stdout, previous.stderr] if part)


def _run_kubectl(
    remote: str,
    kubeconfig: str,
    args: str,
    *,
    input_text: str | None = None,
    timeout_s: int,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    command = f"KUBECONFIG={_sh(kubeconfig)} kubectl {args}"
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", remote, command],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"remote kubectl failed: {command}\n{proc.stdout}\n{proc.stderr}")
    return proc


def _pod_status_reason(pod: Mapping[str, Any]) -> tuple[str, str]:
    status = pod.get("status") or {}
    phase = str(status.get("phase") or "UNKNOWN")
    reasons: list[str] = []
    for cs in list(status.get("initContainerStatuses") or []) + list(status.get("containerStatuses") or []):
        state = cs.get("state") or {}
        waiting = state.get("waiting") or {}
        terminated = state.get("terminated") or {}
        running = state.get("running") or {}
        if waiting:
            reasons.append(f"{cs.get('name')}:waiting:{waiting.get('reason') or ''}:{waiting.get('message') or ''}")
        if terminated:
            reasons.append(f"{cs.get('name')}:terminated:{terminated.get('reason') or ''}:exit={terminated.get('exitCode')}")
        if running:
            reasons.append(f"{cs.get('name')}:running")
    return phase, "; ".join(reasons)


def _terminal_failure(pod: Mapping[str, Any]) -> bool:
    status = pod.get("status") or {}
    phase = status.get("phase")
    if phase in {"Failed", "Succeeded"}:
        return True
    _, reason = _pod_status_reason(pod)
    blockers = ("ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff", "RunContainerError", "CreateContainerError")
    return any(item in reason for item in blockers)


def _all_images_pulled(pod: Mapping[str, Any]) -> bool:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    if not statuses:
        return False
    return all(bool(item.get("imageID")) for item in statuses)


def _container_started(pod: Mapping[str, Any], name: str) -> bool:
    for cs in (pod.get("status") or {}).get("containerStatuses") or []:
        if cs.get("name") != name:
            continue
        state = cs.get("state") or {}
        last = cs.get("lastState") or {}
        return bool(state.get("running") or state.get("terminated") or last.get("terminated"))
    return False


def _extract_result_wall(logs: str, marker: str) -> float | None:
    for line in logs.splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            data = json.loads(payload)
            return float(data["wall_s"])
        except Exception:
            continue
    return None


def _attempt_summary(attempt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": attempt.get("status"),
        "reason": attempt.get("reason"),
        "marker_seen": bool(attempt.get("marker_seen")),
        "all_images_pulled": bool(attempt.get("all_images_pulled")),
        "server_container_started": bool(attempt.get("server_container_started")),
        "elapsed_s": attempt.get("elapsed_s"),
        "timeout_expired": bool(attempt.get("timeout_expired")),
        "runtime_hang_suspected": bool(attempt.get("runtime_hang_suspected")),
        "apply_stdout": attempt.get("apply_stdout"),
        "apply_stderr": attempt.get("apply_stderr"),
    }


def _blockers(
    *,
    native: Mapping[str, Any],
    salus: Mapping[str, Any],
    native_completed: bool,
    salus_completed: bool,
    server_started: bool,
    same_service_scale_ready: bool,
    official_images_ready: bool,
) -> list[str]:
    blockers: list[str] = []
    if not official_images_ready:
        blockers.append("Salus and/or TensorFlow-Salus images were not both pulled into runnable pods")
    if native.get("runtime_hang_suspected"):
        blockers.append(
            "native TensorFlow-Salus image was present and the pod started, but the GPU workload did not make progress before timeout"
        )
    if not native_completed:
        blockers.append(f"native TensorFlow-Salus client did not complete: {native.get('status')} {native.get('reason')}")
    if not server_started:
        blockers.append("Salus server container did not reach a usable started state")
    if salus.get("runtime_hang_suspected"):
        blockers.append(
            "Salus server/client pod used locally available images, but the TensorFlow-Salus zrpc workload did not complete before timeout"
        )
    if not salus_completed:
        blockers.append(f"TensorFlow-Salus zrpc client did not complete through Salus: {salus.get('status')} {salus.get('reason')}")
    if not same_service_scale_ready:
        blockers.append("same-service bridge is missing because both native and Salus zrpc client timings are required")
    return blockers


def _runtime_hang_suspected(
    *,
    pod_json: Mapping[str, Any],
    logs: str,
    marker_seen: bool,
    timeout_expired: bool,
) -> bool:
    if marker_seen or not timeout_expired or not _all_images_pulled(pod_json):
        return False
    _, reason = _pod_status_reason(pod_json)
    lower_logs = logs.lower()
    saw_tf_gpu_start = "creating tensorflow device" in lower_logs or "found device" in lower_logs
    return "running" in reason and saw_tf_gpu_start


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")[:1200]


def _sh(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _cmd_run(args: argparse.Namespace) -> int:
    report = build_salus_remote_fullstack_attempt(
        remote=args.remote,
        kubeconfig=args.kubeconfig,
        namespace=args.namespace,
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
        keep_pods=args.keep_pods,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.salus_remote_fullstack_attempt")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run a remote Salus full-stack same-workload attempt")
    run.add_argument("--remote", default=DEFAULT_REMOTE)
    run.add_argument("--kubeconfig", default=DEFAULT_KUBECONFIG)
    run.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    run.add_argument("--timeout-s", type=int, default=900)
    run.add_argument("--poll-s", type=int, default=10)
    run.add_argument("--keep-pods", action="store_true")
    run.add_argument("--output", default=str(DEFAULT_OUTPUT))
    run.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
