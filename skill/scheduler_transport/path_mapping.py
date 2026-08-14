from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


def ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def windows_path_for_node(
    node: str,
    path: str,
    *,
    node_configs: dict,
    node_is_windows: Callable[[str], bool],
) -> str:
    if not node_is_windows(node) or not path:
        return path
    text = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return text.replace("/", "\\")
    norm = os.path.normpath(text)
    root = (node_configs.get(node, {}) or {}).get("windows_workspace_root") or r"F:\\"
    root = root.rstrip("\\/") + "\\"
    prefixes = [
        str(Path.home() / "mine_code"),
        str(Path.home()),
        "/home/erzhu419/mine_code",
        "/home/erzhu419",
    ]
    for pref in prefixes:
        pref = os.path.normpath(pref)
        try:
            if os.path.commonpath([norm, pref]) != pref:
                continue
        except Exception:
            continue
        rel = os.path.relpath(norm, pref)
        parts = rel.split(os.sep)
        if not parts or parts[0] in (".", ".."):
            continue
        return root + "\\".join(parts)
    return text.replace("/", "\\")


def remote_path_for_node(
    node: str,
    path: str,
    *,
    node_configs: dict,
    canonical_node_name: Callable[[str], str],
    node_is_windows: Callable[[str], bool],
    windows_path_mapper: Callable[[str, str], str],
) -> str:
    if not path:
        return path
    node = canonical_node_name(node)
    if node_is_windows(node):
        return windows_path_mapper(node, path)
    info = node_configs.get(node, {}) or {}
    root = str(info.get("remote_workspace_root") or "").strip()
    if not root:
        return path
    text = str(path)
    if not text.startswith("/"):
        return text
    norm = os.path.normpath(text)
    prefixes = list(info.get("remote_path_prefixes") or [])
    if not prefixes:
        prefixes = [
            str(Path.home() / "mine_code"),
            "/home/erzhu419/mine_code",
        ]
    root = root.rstrip("/")
    for pref in prefixes:
        pref = os.path.normpath(os.path.expanduser(str(pref)))
        try:
            if os.path.commonpath([norm, pref]) != pref:
                continue
        except Exception:
            continue
        rel = os.path.relpath(norm, pref)
        if not rel or rel == ".":
            return root
        parts = rel.split(os.sep)
        if parts[0] == "..":
            continue
        return root + "/" + "/".join(parts)
    return text


def rewrite_command_paths_for_node(
    node: str,
    cmd: str,
    *,
    node_configs: dict,
    node_is_windows: Callable[[str], bool],
    remote_path_mapper: Callable[[str, str], str],
) -> str:
    if not node or not cmd or node_is_windows(node):
        return cmd
    info = node_configs.get(node, {}) or {}
    if not info.get("remote_workspace_root"):
        return cmd
    out = str(cmd)
    for pref in (info.get("remote_path_prefixes") or []):
        pref = os.path.normpath(os.path.expanduser(str(pref)))
        mapped = remote_path_mapper(node, pref)
        if mapped and mapped != pref:
            out = out.replace(pref, mapped)
    return out


def windows_rewrite_token(
    node: str,
    token: str,
    *,
    node_configs: dict,
    windows_path_mapper: Callable[[str, str], str],
) -> str:
    if not token:
        return token
    base = os.path.basename(token.replace("\\", "/")).lower()
    if base in ("python", "python.exe", "python3", "python3.exe") or base.startswith("python3."):
        return (node_configs.get(node, {}) or {}).get("windows_python") or token
    if token.startswith("/home/") or token.startswith(str(Path.home())):
        return windows_path_mapper(node, token)
    return token


@dataclass(frozen=True)
class WindowsPrepareCommandDeps:
    node_configs: dict
    inject_python_u: Callable[[str], str]
    task_launch_cpu_mode: Callable[[dict], bool]
    windows_path_for_node: Callable[[str, str], str]
    windows_rewrite_token: Callable[[str, str], str]


def windows_prepare_command(task: dict, *, deps: WindowsPrepareCommandDeps) -> dict:
    node = task.get("node")
    inner = deps.inject_python_u(task.get("cmd") or "")
    resume_path = task.get("resume_from")
    resume_flag = task.get("resume_flag") or ""
    if resume_path and resume_flag:
        inner = f"{inner} {resume_flag} {shlex.quote(resume_path)}"
    shell_meta = bool(re.search(r"[;&|<>]", inner))
    if not shell_meta:
        try:
            toks = shlex.split(inner)
        except Exception:
            toks = []
        if toks:
            env = {}
            while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
                k, v = toks.pop(0).split("=", 1)
                if k.upper() == "PYTHONPATH":
                    parts = [deps.windows_path_for_node(node, p) for p in v.split(":") if p]
                    v = ";".join(parts)
                elif v.startswith("/home/") or v.startswith(str(Path.home())):
                    v = deps.windows_path_for_node(node, v)
                env[k] = v
            if not toks:
                return {"argv": [], "env": env}
            toks[0] = deps.windows_rewrite_token(node, toks[0])
            toks = [deps.windows_rewrite_token(node, tok) for tok in toks]
            if deps.task_launch_cpu_mode(task):
                device_aware = "--device" in toks or any("jax_experiments" in tok for tok in toks)
                if "--device" in toks:
                    idx = toks.index("--device")
                    if idx + 1 < len(toks):
                        toks[idx + 1] = "cpu"
                elif device_aware:
                    toks.extend(["--device", "cpu"])
            return {"argv": toks, "env": env}
    mapped = inner
    for pref in (str(Path.home() / "mine_code"), str(Path.home()), "/home/erzhu419/mine_code", "/home/erzhu419"):
        if pref in mapped:
            mapped = mapped.replace(pref, deps.windows_path_for_node(node, pref))
    py = (deps.node_configs.get(node, {}) or {}).get("windows_python") or "python"
    mapped = re.sub(
        r"^(python(?:3(?:\.\d+)?)?(?:\.exe)?)(\s|$)",
        lambda match: py + match.group(2),
        mapped,
        count=1,
        flags=re.IGNORECASE,
    )
    return {"cmdline": mapped}


def windows_env_spec_error(task: dict) -> Optional[str]:
    spec = (task.get("env_spec") or "none").strip()
    low = spec.lower()
    if low in ("", "none", "auto"):
        return None
    if low.startswith("docker") or low.startswith("conda"):
        return (
            f"WindowsBackend does not support explicit env_spec={spec!r}; "
            "preinstall/use the node's configured windows_python, or run this "
            "task on a Linux node where docker/conda env sync is available."
        )
    return f"unsupported env_spec={spec!r} for WindowsBackend"
