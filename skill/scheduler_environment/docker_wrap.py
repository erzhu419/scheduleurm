from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class DockerWrapDeps:
    env_deploy: Any
    node_configs: dict
    run_on: Callable[..., tuple[int, str, str]]
    conda_sync_ok: Callable[[str, str], bool]


def maybe_wrap_docker(
    task: dict,
    inner: str,
    cwd: str,
    *,
    gpu_runtime_env: Optional[str] = None,
    deps: DockerWrapDeps,
) -> tuple[str, Optional[str]]:
    """If task's env_spec resolves to docker, wrap inner cmd in `docker run ...`.

    Returns (wrapped_or_inner, error_or_None). When error is non-None, caller
    must fail the launch with that message. Explicit docker requests are fatal
    when docker/image checks fail; auto mode falls back to the host command.
    """
    env_deploy = deps.env_deploy
    if env_deploy is None:
        spec = (task.get("env_spec") or "none").lower()
        if spec.startswith("docker") and spec != "auto":
            return (inner, "env_deploy module not loadable; cannot honor --env-spec docker")
        return (inner, None)

    spec = task.get("env_spec") or "none"
    image = task.get("image") or ""
    try:
        kind, spec_image = env_deploy.parse_env_spec(spec)
    except ValueError as exc:
        return (inner, f"invalid env_spec {spec!r}: {exc}")
    if kind == "none":
        return (inner, None)

    node = task.get("node")
    node_host = deps.node_configs.get(node, {}).get("host") if node else None
    if kind == "conda":
        if spec_image and Path(spec_image).is_absolute() and not Path(spec_image).is_dir():
            return (
                inner,
                f"--env-spec conda:{spec_image} but the env path does not "
                f"exist locally - preload skipped (nothing to rsync); "
                f"launching now would risk running a stale remote env at "
                f"the same path. Create/deploy the env locally first, "
                f"then resubmit.",
            )
        if (
            node_host
            and spec_image
            and Path(spec_image).is_absolute()
            and not deps.conda_sync_ok(node, spec_image)
        ):
            return (
                inner,
                f"--env-spec conda:{spec_image} but the latest sync to "
                f"{node} did not succeed (or has not run yet) - refusing "
                f"to launch; would risk running a stale remote env. Wait "
                f"for the next dispatch cycle's preload, or check "
                f"~/.claude/scheduler/logs/watcher.log for "
                f"preload_conda_failed events to diagnose.",
            )
        return (inner, None)

    chosen_image = spec_image or image
    explicit = kind == "docker"
    if not chosen_image:
        if explicit:
            return (inner, "--env-spec docker requires --image (or 'docker:IMAGE' inline)")
        return (inner, None)

    if not env_deploy.has_docker(deps.run_on, node, timeout=8):
        if explicit:
            return (inner, f"--env-spec docker requested but `docker info` failed on {node}")
        return (inner, None)

    local_digest = env_deploy.get_image_digest(deps.run_on, "local", chosen_image)
    if explicit and local_digest is None:
        return (
            inner,
            f"--env-spec docker:{chosen_image} but image not present "
            f"locally (no digest available); refusing to launch - would "
            f"risk running a stale or unintended image. Build/pull the "
            f"image locally first, then resubmit.",
        )
    if not explicit and local_digest is None:
        return (inner, None)

    if node_host:
        if not env_deploy.has_image(
            deps.run_on,
            node,
            chosen_image,
            local_digest=local_digest,
        ):
            err = (
                f"docker image {chosen_image} not present (or digest "
                f"drift) on {node} at launch - preload not yet "
                f"successful. Will retry next dispatch cycle; if this "
                f"keeps happening, check ~/.claude/scheduler/logs/"
                f"watcher.log for preload_image_failed events."
            )
            if explicit:
                return (inner, err)
            return (inner, None)

    container_name = f"sched-{task.get('id') or 'unknown'}"
    task["container_name"] = container_name
    wrapped = env_deploy.wrap_cmd_docker(
        inner=inner,
        image=chosen_image,
        cwd=cwd,
        gpu_idx=task.get("gpu_idx"),
        extra_env=task.get("extra_env") or {},
        container_name=container_name,
        memory_mb=task.get("ram_mb"),
        cpus=task.get("cpu_cores"),
        gpu_runtime_env=gpu_runtime_env,
    )
    return (wrapped, None)
