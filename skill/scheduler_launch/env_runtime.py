"""Runtime-bound launch environment, docker, and git-precheck wrappers."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

try:
    from scheduler_environment.docker_wrap import maybe_wrap_docker as _maybe_wrap_docker_impl
    from scheduler_migration.policy import compute_node_load_seconds as _compute_node_load_seconds_impl
    from scheduler_runtime.wiring import build_docker_wrap_deps as _build_docker_wrap_deps
except ModuleNotFoundError:
    from ..scheduler_environment.docker_wrap import maybe_wrap_docker as _maybe_wrap_docker_impl
    from ..scheduler_migration.policy import compute_node_load_seconds as _compute_node_load_seconds_impl
    from ..scheduler_runtime_wiring import build_docker_wrap_deps as _build_docker_wrap_deps

from .command import inject_python_u as _inject_python_u_impl
from .env import (
    apply_node_cmd_rewrites as _apply_node_cmd_rewrites_impl,
    launch_extra_env as _launch_extra_env_impl,
    node_launch_extra_env as _node_launch_extra_env_impl,
    parse_env as _parse_env_impl,
    project_from_path as _project_from_path_impl,
    project_from_pid as _project_from_pid_impl,
    safe_extra_env_items as _safe_extra_env_items_impl,
    task_needs_jax_launch_env as _task_needs_jax_launch_env_impl,
)


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_launch_env_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {
        "_ENV_KEY_RE": re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$"),
        "_RESERVED_ENV_KEYS": frozenset({
            "CUDA_VISIBLE_DEVICES",
            "SCHEDULEURM_TASK_ID",
            "SCHEDULEURM_LAUNCH_TOKEN",
        }),
    }

    def precheck_git(task):
        """Verify git repo is clean locally and synced with target node."""
        repo = task.get("git_repo")
        if not repo:
            return True, "no git check requested"
        subprocess_mod = _ns(namespace, "subprocess")
        shlex_mod = _ns(namespace, "shlex")
        try:
            local_hash = subprocess_mod.check_output(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                text=True,
                timeout=5,
            ).strip()
            local_dirty = bool(subprocess_mod.check_output(
                ["git", "-C", repo, "status", "--porcelain"],
                text=True,
                timeout=5,
            ).strip())
        except Exception as exc:
            return False, f"local git check failed: {exc}"

        warnings = []
        if local_dirty:
            warnings.append(
                f"warn: local repo {repo} is dirty (uncommitted changes - run will use working tree state, not the committed commit)"
            )
        node = task["node"]
        if _ns(namespace, "NODES")[node]["host"] is None:
            return True, "; ".join(warnings) or "ok"
        if _ns(namespace, "_node_is_windows")(node):
            warnings.append(
                f"warn: git precheck skipped on Windows node {node} "
                "(path mapping / git availability is project-specific)"
            )
            return True, "; ".join(warnings) or "ok"
        try:
            rc, out, _ = _ns(namespace, "run_on")(
                node,
                f"git -C {shlex_mod.quote(repo)} rev-parse HEAD",
                timeout=10,
                check=False,
            )
            if rc != 0:
                return False, f"remote repo {repo} missing on {node} (run git clone first)"
            remote_hash = out.strip()
            if remote_hash != local_hash:
                return False, f"git mismatch: local={local_hash[:8]} {node}={remote_hash[:8]} - push & pull first"
        except Exception as exc:
            return False, f"remote git check failed: {exc}"
        return True, "; ".join(warnings) or "ok"

    def _inject_python_u(cmd: str) -> str:
        return _inject_python_u_impl(cmd)

    def _docker_wrap_deps():
        return _build_docker_wrap_deps(namespace)

    def _maybe_wrap_docker(
        task: dict,
        inner: str,
        cwd: str,
        gpu_runtime_env: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        return _maybe_wrap_docker_impl(
            task,
            inner,
            cwd,
            gpu_runtime_env=gpu_runtime_env,
            deps=_ns(namespace, "_docker_wrap_deps")(),
        )

    def _project_from_path(path):
        return _project_from_path_impl(path)

    def _project_from_pid(node, pid):
        return _project_from_pid_impl(
            node,
            pid,
            node_is_windows=_ns(namespace, "_node_is_windows"),
            run_on=_ns(namespace, "run_on"),
        )

    def _safe_extra_env_items(extra_env):
        return _safe_extra_env_items_impl(
            extra_env,
            env_key_re=_ns(namespace, "_ENV_KEY_RE"),
            reserved_env_keys=_ns(namespace, "_RESERVED_ENV_KEYS"),
        )

    def _apply_node_cmd_rewrites(node, cmd):
        return _apply_node_cmd_rewrites_impl(
            node,
            cmd,
            node_configs=_ns(namespace, "NODES"),
            canonical_node_name=_ns(namespace, "_canonical_node_name"),
        )

    def _task_needs_jax_launch_env(task: dict) -> bool:
        return _task_needs_jax_launch_env_impl(
            task,
            default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
            task_launch_cpu_mode=_ns(namespace, "_task_launch_cpu_mode"),
        )

    def _launch_extra_env(task):
        return _launch_extra_env_impl(
            task,
            default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
            task_launch_cpu_mode=_ns(namespace, "_task_launch_cpu_mode"),
        )

    def _node_launch_extra_env(task):
        return _node_launch_extra_env_impl(
            task,
            node_configs=_ns(namespace, "NODES"),
            rewrite_command_paths_for_node=_ns(namespace, "_rewrite_command_paths_for_node"),
            default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
            task_launch_cpu_mode=_ns(namespace, "_task_launch_cpu_mode"),
        )

    def _parse_env(pairs):
        return _parse_env_impl(
            pairs,
            env_key_re=_ns(namespace, "_ENV_KEY_RE"),
            reserved_env_keys=_ns(namespace, "_RESERVED_ENV_KEYS"),
        )

    def compute_node_load_seconds(state: dict) -> dict:
        return _compute_node_load_seconds_impl(
            state,
            node_configs=_ns(namespace, "NODES"),
        )

    exports.update({
        "precheck_git": precheck_git,
        "_inject_python_u": _inject_python_u,
        "_docker_wrap_deps": _docker_wrap_deps,
        "_maybe_wrap_docker": _maybe_wrap_docker,
        "_project_from_path": _project_from_path,
        "_project_from_pid": _project_from_pid,
        "_safe_extra_env_items": _safe_extra_env_items,
        "_apply_node_cmd_rewrites": _apply_node_cmd_rewrites,
        "_task_needs_jax_launch_env": _task_needs_jax_launch_env,
        "_launch_extra_env": _launch_extra_env,
        "_node_launch_extra_env": _node_launch_extra_env,
        "_parse_env": _parse_env,
        "compute_node_load_seconds": compute_node_load_seconds,
    })
    return exports
