from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Mapping


ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RESERVED_ENV_KEYS = frozenset({
    "CUDA_VISIBLE_DEVICES",
    "SCHEDULEURM_TASK_ID",
    "SCHEDULEURM_LAUNCH_TOKEN",
})


def project_from_path(path) -> str:
    """Heuristic: project name = last path component."""
    if not path:
        return ""
    return os.path.basename(str(path).rstrip("/")) or ""


def project_from_pid(
    node,
    pid,
    *,
    node_is_windows: Callable[[str], bool],
    run_on: Callable,
) -> str:
    """Read /proc/<pid>/cwd on the node via readlink."""
    if node_is_windows(node):
        return ""
    try:
        rc, out, _ = run_on(node, f"readlink /proc/{pid}/cwd 2>/dev/null", timeout=8, check=False)
        return project_from_path(out.strip()) if rc == 0 else ""
    except Exception:
        return ""


def safe_extra_env_items(
    extra_env,
    *,
    env_key_re=ENV_KEY_RE,
    reserved_env_keys=RESERVED_ENV_KEYS,
):
    """Yield valid user-provided env items, skipping legacy unsafe keys."""
    for k, v in (extra_env or {}).items():
        if not env_key_re.match(k):
            continue
        if k in reserved_env_keys:
            continue
        yield (k, v)


def apply_node_cmd_rewrites(
    node,
    cmd,
    *,
    node_configs: Mapping,
    canonical_node_name: Callable[[str], str],
):
    """Apply deterministic node-local command rewrites before launch."""
    if not node or not cmd:
        return cmd
    node = canonical_node_name(node)
    for old, new in (node_configs.get(node, {}).get("cmd_rewrites") or []):
        old = str(old or "")
        new = str(new or "")
        if old and new and old in cmd:
            cmd = cmd.replace(old, new)
    return cmd


def task_needs_jax_launch_env(
    task: dict,
    *,
    default_vram_mb: int,
    task_launch_cpu_mode: Callable[[dict], bool],
) -> bool:
    cmd = task.get("cmd") or ""
    project = str(task.get("project") or "").lower()
    is_gpu = int(task.get("est_vram_mb", default_vram_mb) or 0) > 0 and not task_launch_cpu_mode(task)
    return bool(is_gpu and (
        project == "re-sac"
        or project == "bapr"
        or "run_seed.sh" in cmd
        or "jax_experiments" in cmd
        or "jax" in cmd.lower()
    ))


def launch_extra_env(
    task: dict,
    *,
    default_vram_mb: int,
    task_launch_cpu_mode: Callable[[dict], bool],
) -> dict:
    """Launch-time env after applying conservative defaults for known JAX GPU tasks."""
    merged = {}
    if task_needs_jax_launch_env(
        task,
        default_vram_mb=default_vram_mb,
        task_launch_cpu_mode=task_launch_cpu_mode,
    ):
        merged["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        merged["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
        if task.get("project") == "RE-SAC" and task.get("cwd"):
            merged["PYTHONPATH"] = task["cwd"]
    merged.update(task.get("extra_env") or {})
    return merged


def node_launch_extra_env(
    task: dict,
    *,
    node_configs: Mapping,
    rewrite_command_paths_for_node: Callable[[str, str], str],
    default_vram_mb: int,
    task_launch_cpu_mode: Callable[[dict], bool],
) -> dict:
    """Merge node-local launch env with task/JAX defaults."""
    node = task.get("node") or ""
    node_env = dict((node_configs.get(node, {}) or {}).get("launch_extra_env") or {})
    merged = node_env if task_needs_jax_launch_env(
        task,
        default_vram_mb=default_vram_mb,
        task_launch_cpu_mode=task_launch_cpu_mode,
    ) else {}
    merged.update(launch_extra_env(
        task,
        default_vram_mb=default_vram_mb,
        task_launch_cpu_mode=task_launch_cpu_mode,
    ))
    if node:
        merged = {
            k: rewrite_command_paths_for_node(node, str(v))
            for k, v in merged.items()
        }
    return merged


def parse_env(
    pairs: Iterable[str] | None,
    *,
    env_key_re=ENV_KEY_RE,
    reserved_env_keys=RESERVED_ENV_KEYS,
) -> dict:
    if not pairs:
        return {}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--env expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        if not env_key_re.match(k):
            raise SystemExit(
                f"--env key {k!r} is not a valid POSIX env var name "
                f"(must match ^[A-Za-z_][A-Za-z0-9_]*$); refusing to launch "
                f"with a key that would break the export shell line.")
        if k in reserved_env_keys:
            raise SystemExit(
                f"--env key {k!r} is reserved by scheduleurm (set by the "
                f"launch path to honor GPU pinning / scheduling decisions); "
                f"user override would break VRAM accounting + the 1/3 "
                f"packing rule.")
        out[k] = v
    return out
