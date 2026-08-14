from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class DockerPreloadDeps:
    env_deploy: Any
    load_state: Callable[[], dict]
    node_configs: dict
    node_is_windows: Callable[[str], bool]
    run_on: Callable[..., tuple[int, str, str]]
    notify: Callable[..., Any]
    record_conda_sync_ok: Callable[[str, str], Any]
    record_conda_sync_failed: Callable[[str, str], Any]


def _target_ids(task_ids: Optional[set]) -> set[str]:
    return {str(task_id) for task_id in (task_ids or set()) if str(task_id)}


def _queued_tasks(state: dict, task_ids: Optional[set]) -> list[dict]:
    target_ids = _target_ids(task_ids)
    out = []
    for task in state.get("tasks", []):
        if task.get("status") != "queued":
            continue
        if target_ids and str(task.get("id") or "") not in target_ids:
            continue
        out.append(task)
    return out


def _candidate_nodes(task: dict, deps: DockerPreloadDeps) -> list[str]:
    require = task.get("require_node")
    return [require] if require else list(deps.node_configs.keys())


def _collect_preload_needs(
    state: dict,
    task_ids: Optional[set],
    deps: DockerPreloadDeps,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    needed_docker: set[tuple[str, str]] = set()
    needed_conda: set[tuple[str, str]] = set()
    for task in _queued_tasks(state, task_ids):
        spec = task.get("env_spec") or "none"
        if spec == "none":
            continue
        try:
            kind, spec_payload = deps.env_deploy.parse_env_spec(spec)
        except ValueError:
            continue
        candidates = _candidate_nodes(task, deps)
        if kind == "docker" or (kind == "auto" and (spec_payload or task.get("image"))):
            image = spec_payload or (task.get("image") or "")
            if not image:
                continue
            for node in candidates:
                if deps.node_is_windows(node):
                    continue
                needed_docker.add((node, image))
        elif kind == "conda":
            if not spec_payload:
                continue
            path = Path(spec_payload)
            if not path.is_absolute():
                continue
            if not path.is_dir():
                continue
            for node in candidates:
                if deps.node_is_windows(node):
                    continue
                if deps.node_configs.get(node, {}).get("host") is None:
                    continue
                needed_conda.add((node, spec_payload))
    return needed_docker, needed_conda


def _preload_docker_images(needed_docker: set[tuple[str, str]], deps: DockerPreloadDeps) -> None:
    local_digests: dict[str, Optional[str]] = {}
    for _, image in needed_docker:
        if image not in local_digests:
            local_digests[image] = deps.env_deploy.get_image_digest(deps.run_on, "local", image)
    for node, image in needed_docker:
        try:
            if not deps.env_deploy.has_docker(deps.run_on, node, timeout=8):
                continue
            if deps.env_deploy.has_image(deps.run_on, node, image, local_digest=local_digests.get(image)):
                continue
            node_host = deps.node_configs.get(node, {}).get("host")
            ok, msg = deps.env_deploy.push_image(node_host, image, timeout_s=1800)
            event = "preload_image_ok" if ok else "preload_image_failed"
            deps.notify(
                event,
                {"node": node, "image": image, "msg": msg[:200] if not ok else ""},
                feishu_enabled=False,
            )
        except Exception as exc:
            deps.notify(
                "preload_image_error",
                {"node": node, "image": image, "error": str(exc)[:200]},
                feishu_enabled=False,
            )


def _preload_conda_envs(needed_conda: set[tuple[str, str]], deps: DockerPreloadDeps) -> None:
    for node, env_path in needed_conda:
        try:
            node_host = deps.node_configs.get(node, {}).get("host")
            ok, msg = deps.env_deploy.push_conda_env(
                node_host,
                env_path,
                env_path,
                timeout_s=3600,
            )
            if ok:
                deps.record_conda_sync_ok(node, env_path)
            else:
                deps.record_conda_sync_failed(node, env_path)
            event = "preload_conda_ok" if ok else "preload_conda_failed"
            deps.notify(
                event,
                {"node": node, "env_path": env_path, "msg": msg[:300] if not ok else ""},
                feishu_enabled=False,
            )
        except Exception as exc:
            deps.record_conda_sync_failed(node, env_path)
            deps.notify(
                "preload_conda_error",
                {"node": node, "env_path": env_path, "error": str(exc)[:200]},
                feishu_enabled=False,
            )


def preload_docker_images_outside_lock(
    task_ids: Optional[set] = None,
    *,
    deps: DockerPreloadDeps,
) -> None:
    """Preload queued task environments without taking the scheduler state lock."""
    if deps.env_deploy is None:
        return
    try:
        state = deps.load_state()
    except Exception:
        return
    needed_docker, needed_conda = _collect_preload_needs(state, task_ids, deps)
    _preload_docker_images(needed_docker, deps)
    _preload_conda_envs(needed_conda, deps)
