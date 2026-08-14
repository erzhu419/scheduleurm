from __future__ import annotations

import hashlib
import json
from typing import Callable


def task_runtime_payload(task: dict, *, conflict_path_key: Callable[[object], str]) -> dict:
    """Fields that make runtime/walltime materially different."""
    extra_env = task.get("extra_env") or {}
    if not isinstance(extra_env, dict):
        extra_env = {}
    return {
        "signature": task.get("signature") or "",
        "project": task.get("project") or "",
        "description": task.get("description") or "",
        "cmd": task.get("cmd") or "",
        "cwd": conflict_path_key(task.get("cwd")),
        "env_spec": task.get("env_spec") or "none",
        "image": task.get("image") or "",
        "extra_env": extra_env,
    }


def task_runtime_keys(
    task: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
) -> list:
    payload = task_runtime_payload(task)
    exact_payload = {
        "cmd": payload.get("cmd") or "",
        "cwd": payload.get("cwd") or "",
        "env_spec": payload.get("env_spec") or "none",
        "image": payload.get("image") or "",
        "extra_env": payload.get("extra_env") or {},
    }
    keys = []
    if exact_payload["cmd"]:
        raw = json.dumps(exact_payload, sort_keys=True, separators=(",", ":"))
        exact = "exact:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
        keys.append((exact, "exact", payload))
    sig = payload.get("signature")
    if sig:
        keys.append(("sig:" + sig, "signature", payload))
    return keys


def runtime_device_kind_for_task(
    task: dict,
    gpu_idx=None,
    *,
    task_launch_cpu_mode: Callable[[dict], bool],
) -> str:
    if gpu_idx is None or task_launch_cpu_mode(task):
        return "cpu"
    return "gpu"


def runtime_node_bucket_key(
    node: str,
    device_kind: str,
    *,
    canonical_node_name: Callable[[str], str],
) -> str:
    node = canonical_node_name(node) if node else "?"
    return f"{node or '?'}:{device_kind or '?'}"


def runtime_node_bucket_keys(
    node: str,
    device_kind: str,
    *,
    canonical_node_name: Callable[[str], str],
    node_name_aliases: dict,
) -> list[str]:
    canon = canonical_node_name(node) if node else "?"
    names = [canon]
    names.extend(
        alias for alias, target in node_name_aliases.items()
        if target == canon and alias not in names
    )
    return [
        runtime_node_bucket_key(
            name,
            device_kind,
            canonical_node_name=canonical_node_name,
        )
        for name in names
    ]


def candidate_runtime_seconds(
    task: dict,
    node: str,
    gpu_idx=None,
    *,
    load_runtime_history: Callable[[], dict],
    task_runtime_keys: Callable[[dict], list],
    runtime_device_kind_for_task: Callable[[dict, object], str],
    runtime_node_bucket_keys: Callable[[str, str], list[str]],
) -> int:
    """Return node/device-specific runtime history for placement scoring, or 0 if unknown."""
    history = load_runtime_history()
    device = runtime_device_kind_for_task(task, gpu_idx)
    buckets = runtime_node_bucket_keys(node, device)
    for key, _kind, _payload in task_runtime_keys(task):
        rec = history.get(key)
        if not isinstance(rec, dict):
            continue
        node_runtime = rec.get("node_runtime") or {}
        bucket_rec = next(
            (
                node_runtime.get(bucket)
                for bucket in buckets
                if isinstance(node_runtime.get(bucket), dict)
            ),
            None,
        )
        if isinstance(bucket_rec, dict) and int(bucket_rec.get("total_s") or 0) > 0:
            return int(bucket_rec.get("total_s") or 0)
    return 0
