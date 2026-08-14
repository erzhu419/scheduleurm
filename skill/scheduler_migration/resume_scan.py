"""Resume-checkpoint discovery across local, remote, and Windows nodes."""

from __future__ import annotations

import json
import os
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional, Pattern


@dataclass(frozen=True)
class ResumeScanDeps:
    node_configs: dict
    ckpt_exts: tuple[str, ...]
    resume_safe_name_re: Pattern
    resume_unsafe_name_re: Pattern
    resume_scan_ttl_s: int
    resume_scan_workers: int
    task_gpu_capability: Callable[[dict], Optional[str]]
    node_is_windows: Callable[[str], bool]
    windows_path_for_node: Callable[[str, str], str]
    ps_quote: Callable[[str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    remote_path_for_node: Callable[[str, str], str]
    run_on: Callable[..., tuple[int, str, str]]
    cmd_has_resume_flag: Callable[[str], bool]
    now: Callable[[], float]


def resume_candidate_is_safe(path: str, explicit_glob: bool = False, *, deps: ResumeScanDeps) -> bool:
    base = os.path.basename(path or "")
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    if ext not in deps.ckpt_exts:
        return False
    if explicit_glob:
        return True
    if deps.resume_unsafe_name_re.search(base):
        return False
    return bool(deps.resume_safe_name_re.search(base))


def _normalized_relative_glob(pattern: str) -> str:
    """Return a portable ckpt glob that cannot escape ckpt_dir."""
    normalized = str(pattern or "*").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if normalized.startswith("/") or not parts or any(part == ".." for part in parts):
        raise RuntimeError(f"ckpt_glob must be relative to ckpt_dir: {pattern!r}")
    return "/".join(parts)


def resume_node_names(
    task: Optional[dict] = None,
    nodes: Optional[list] = None,
    *,
    deps: ResumeScanDeps,
) -> list:
    names = []
    live_names = None
    if nodes is not None:
        live_names = {node.get("name") for node in nodes if node.get("name") and node.get("alive")}

    def add(name, *, force: bool = False):
        if not name or name not in deps.node_configs or name in names:
            return
        if live_names is not None and name not in live_names and not force:
            return
        names.append(name)

    known_resume_nodes = []
    if task:
        for loc in task.get("resume_locations") or []:
            node = loc.get("node") if isinstance(loc, dict) else None
            if node and node not in known_resume_nodes:
                known_resume_nodes.append(node)
        for key in ("resume_checkpoint_node", "staged_node", "node", "assigned_node"):
            node = task.get(key)
            if node and node not in known_resume_nodes:
                known_resume_nodes.append(node)

        explicit = []
        require_node = task.get("require_node")
        if require_node:
            explicit.append(require_node)
        for node in task.get("allowed_nodes") or []:
            if node and node not in explicit:
                explicit.append(node)

        for node in explicit:
            add(str(node))
        for node in known_resume_nodes:
            add(str(node), force=True)
        if explicit and names:
            # Placement constraints describe where the task may run, not where
            # an input checkpoint may live. Always inspect local as a possible
            # staging source so CPU/GPU tasks pinned to remote nodes can consume
            # locally collected checkpoints without weakening placement rules.
            add("local", force=True)
            return names

        preferred = []
        for node in [task.get("preferred_node")] + list(task.get("resume_preferred_nodes") or []):
            if node and node not in preferred:
                preferred.append(node)
        for node in preferred:
            add(str(node))

    fallback = [node.get("name") for node in (nodes or []) if node.get("name")] or list(deps.node_configs.keys())
    gpu_task = bool(task and int(task.get("est_vram_mb") or 0) > 0)
    need_cap = deps.task_gpu_capability(task or {}) if gpu_task else None
    for name in fallback:
        info = deps.node_configs.get(name, {})
        if gpu_task:
            if deps.node_is_windows(name) or info.get("max_vram_per_task") == 0:
                continue
            caps = {str(cap).lower() for cap in (info.get("capabilities") or [])}
            if caps and need_cap not in caps and "cuda" not in caps:
                continue
        add(name)
    return names


def resume_scan_key(task: dict, nodes: Optional[list] = None, *, deps: ResumeScanDeps) -> list:
    return [
        task.get("ckpt_dir"),
        task.get("ckpt_glob", "*") or "*",
        sorted(resume_node_names(task, nodes, deps=deps)),
    ]


def task_requires_resume_scan(task: dict, *, deps: ResumeScanDeps) -> bool:
    if task.get("skip_resume_scan"):
        return False
    if not task.get("ckpt_dir"):
        return False
    return bool(
        task.get("resume_flag")
        or task.get("resume_managed_by_cmd")
        or deps.cmd_has_resume_flag(task.get("cmd") or "")
    )


def allow_initial_resume_scan_error(task: dict) -> bool:
    if not task.get("allow_initial_resume_scan_error"):
        return False
    if task.get("parent_id") or int(task.get("retry_count") or 0) > 0:
        return False
    if task.get("started_at") or task.get("actual_started_at") or task.get("log_path"):
        return False
    if task.get("remote_pids") or task.get("process_group") or task.get("slurm_job_id"):
        return False
    if task.get("resume_locations") or task.get("resume_checkpoint_node") or task.get("resume_from"):
        return False
    return True


def cached_resume_locations_for_task(
    task: dict,
    nodes: list,
    *,
    deps: ResumeScanDeps,
) -> tuple[bool, list, dict]:
    """Return only a fresh, already-persisted resume scan result.

    This helper is deliberately I/O free so it is safe to call while the
    scheduler state lock is held.
    """
    if not task_requires_resume_scan(task, deps=deps):
        return True, [], {}
    key = resume_scan_key(task, nodes, deps=deps)
    try:
        completed_at = float(
            task.get("resume_scan_completed_at")
            or task.get("resume_scan_at")
            or 0
        )
    except Exception:
        completed_at = 0.0
    age = deps.now() - completed_at
    fresh = (
        "resume_locations" in task
        and task.get("resume_scan_key") == key
        and completed_at > 0
        and 0 <= age <= deps.resume_scan_ttl_s
    )
    if not fresh:
        return False, [], {}
    return (
        True,
        list(task.get("resume_locations") or []),
        dict(task.get("resume_scan_errors") or {}),
    )


def apply_resume_scan_result(
    task: dict,
    *,
    scan_key: list,
    locations: list,
    errors: dict,
    completed_at: float,
    source: str = "outside_state_lock",
) -> None:
    """Persist one completed scan and its derived placement hints on a task."""
    task["resume_scan_at"] = completed_at
    task["resume_scan_completed_at"] = completed_at
    task["resume_scan_key"] = scan_key
    task["resume_scan_source"] = source
    task["resume_locations"] = list(locations or [])
    if errors:
        task["resume_scan_errors"] = dict(errors)
    else:
        task.pop("resume_scan_errors", None)
    if locations:
        ordered_nodes = []
        for loc in locations:
            node = loc.get("node")
            if node and node not in ordered_nodes:
                ordered_nodes.append(node)
        task["resume_preferred_nodes"] = ordered_nodes
        best = locations[0]
        task["resume_checkpoint_node"] = best.get("node")
        task["resume_from"] = best.get("path")
    else:
        task.pop("resume_preferred_nodes", None)
        task.pop("resume_checkpoint_node", None)
        task.pop("resume_from", None)


def _find_resume_on_windows(task: dict, node: str, *, deps: ResumeScanDeps) -> Optional[dict]:
    win_dir = deps.windows_path_for_node(node, task.get("ckpt_dir"))
    pattern = _normalized_relative_glob(
        task.get("ckpt_glob", "*") or "*")
    ext_list = ",".join(deps.ckpt_exts)
    safe_pat = deps.resume_safe_name_re.pattern
    unsafe_pat = deps.resume_unsafe_name_re.pattern
    explicit = "1" if pattern != "*" else "0"
    ps = rf'''
$dir = {deps.ps_quote(win_dir)}
$relative = {deps.ps_quote(pattern)}.Replace('/', '\\')
$parent = Split-Path -Path $relative -Parent
$pattern = Split-Path -Path $relative -Leaf
$scanDir = if ($parent) {{ Join-Path -Path $dir -ChildPath $parent }} else {{ $dir }}
$explicit = ({deps.ps_quote(explicit)} -eq '1')
$exts = ({deps.ps_quote(ext_list)}).Split(',')
$safe = {deps.ps_quote(safe_pat)}
$unsafe = {deps.ps_quote(unsafe_pat)}
if (-not (Test-Path -LiteralPath $scanDir)) {{ exit 0 }}
$files = Get-ChildItem -LiteralPath $scanDir -File -Filter $pattern -ErrorAction SilentlyContinue | Where-Object {{
  $ext = $_.Extension.TrimStart('.').ToLowerInvariant()
  if ($exts -notcontains $ext) {{ return $false }}
  if ($explicit) {{ return $true }}
  if ($_.Name -match $unsafe) {{ return $false }}
  return ($_.Name -match $safe)
}} | Sort-Object LastWriteTimeUtc -Descending
if ($files -and $files.Count -gt 0) {{
  $f = $files[0]
  [pscustomobject]@{{ path=$f.FullName; mtime=([DateTimeOffset]$f.LastWriteTimeUtc).ToUnixTimeSeconds(); size=$f.Length }} | ConvertTo-Json -Compress
}}
'''
    try:
        rc, out, err = deps.run_windows_ps(node, ps, timeout=10, check=False)
    except Exception as exc:
        raise RuntimeError(str(exc))
    if rc != 0:
        raise RuntimeError((err or out or f"rc={rc}").strip()[:300])
    out = (out or "").strip()
    if not out:
        return None
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"bad Windows checkpoint scan output: {exc}")
    data["node"] = node
    return data


def _find_resume_on_remote(task: dict, node: str, *, deps: ResumeScanDeps) -> Optional[dict]:
    scan_dir = deps.remote_path_for_node(node, task.get("ckpt_dir"))
    pattern = _normalized_relative_glob(
        task.get("ckpt_glob", "*") or "*")
    explicit_glob = pattern != "*"
    marker_begin = "__SCHEDULEURM_RESUME_SCAN_JSON_BEGIN__"
    marker_end = "__SCHEDULEURM_RESUME_SCAN_JSON_END__"
    # Compute only portable file metadata remotely.  Candidate filtering and
    # newest-file selection stay local, so old HPC nodes do not need Python 3.
    if "/" in pattern:
        max_depth = len(pattern.split("/"))
        matcher = (
            f"-maxdepth {max_depth} -type f -path "
            f"{shlex.quote(scan_dir.rstrip('/') + '/' + pattern)}")
    else:
        matcher = (
            f"-maxdepth 1 -type f -name {shlex.quote(pattern)}")
    cmd = (
        f"printf '%s\\n' {shlex.quote(marker_begin)}; "
        "cd / >/dev/null 2>&1 || true; rc=0; "
        f"if [ -d {shlex.quote(scan_dir)} ]; then "
        f"LC_ALL=C find {shlex.quote(scan_dir)} {matcher} "
        "-printf '%T@\\t%s\\t%p\\n' || rc=$?; "
        f"fi; printf '%s\\n' {shlex.quote(marker_end)}; exit $rc"
    )
    try:
        rc, out, err = deps.run_on(node, cmd, timeout=10, check=False)
    except Exception as exc:
        raise RuntimeError(str(exc))
    if rc != 0:
        raise RuntimeError((err or out or f"rc={rc}").strip()[:300])
    text = out or ""
    if marker_begin in text and marker_end in text:
        text = text.split(marker_begin, 1)[1].split(marker_end, 1)[0]
    out = text.strip()
    if not out:
        return None
    candidates = []
    malformed = []
    for line in out.splitlines():
        if line.lstrip().startswith("{"):
            try:
                legacy = json.loads(line)
                path = str(legacy["path"])
                mtime = float(legacy["mtime"])
                size = int(legacy["size"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                malformed.append(line)
                continue
            if resume_candidate_is_safe(path, explicit_glob, deps=deps):
                candidates.append({
                    "path": path,
                    "mtime": mtime,
                    "size": size,
                    "node": node,
                })
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            malformed.append(line)
            continue
        try:
            mtime = float(parts[0])
            size = int(parts[1])
        except (TypeError, ValueError):
            malformed.append(line)
            continue
        path = parts[2]
        if not resume_candidate_is_safe(path, explicit_glob, deps=deps):
            continue
        candidates.append({
            "path": path,
            "mtime": mtime,
            "size": size,
            "node": node,
        })
    if not candidates:
        if malformed:
            raise RuntimeError(
                f"bad checkpoint scan output: {malformed[-1][:200]}"
            )
        return None
    return max(candidates, key=lambda item: item["mtime"])


def find_resume_on_node(task: dict, node: str, *, deps: ResumeScanDeps) -> Optional[dict]:
    if not task.get("ckpt_dir"):
        return None
    if deps.node_is_windows(node):
        return _find_resume_on_windows(task, node, deps=deps)
    return _find_resume_on_remote(task, node, deps=deps)


def scan_resume_locations(
    task: dict,
    nodes: Optional[list] = None,
    cache: Optional[dict] = None,
    *,
    deps: ResumeScanDeps,
) -> tuple:
    if not task_requires_resume_scan(task, deps=deps):
        return [], {}
    node_names = resume_node_names(task, nodes, deps=deps)
    key = (task.get("ckpt_dir"), task.get("ckpt_glob", "*") or "*", tuple(node_names))
    if cache is not None and key in cache:
        locs, errs = cache[key]
        return list(locs), dict(errs)
    locations = []
    errors = {}

    def scan_one(node):
        try:
            return node, find_resume_on_node(task, node, deps=deps), None
        except Exception as exc:
            return node, None, str(exc)[:300]

    workers = min(deps.resume_scan_workers, len(node_names))
    if workers <= 1 or len(node_names) <= 1:
        results = [scan_one(node) for node in node_names]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(scan_one, node): node for node in node_names}
            for fut in as_completed(futures):
                results.append(fut.result())

    for node, found, err in results:
        if err:
            errors[node] = err
        elif found:
            locations.append(found)
    locations.sort(key=lambda item: (float(item.get("mtime") or 0), str(item.get("node") or "")), reverse=True)
    if cache is not None:
        cache[key] = (list(locations), dict(errors))
    return locations, errors


def refresh_resume_locations_for_task(task: dict, nodes: list, cache: dict, *, deps: ResumeScanDeps) -> tuple:
    if not task_requires_resume_scan(task, deps=deps):
        for key in (
            "resume_locations",
            "resume_scan_errors",
            "resume_preferred_nodes",
            "resume_checkpoint_node",
            "resume_scan_at",
            "resume_scan_completed_at",
            "resume_scan_key",
            "resume_scan_source",
            "resume_scan_inflight",
        ):
            task.pop(key, None)
        return [], {}
    fresh, locations, errors = cached_resume_locations_for_task(task, nodes, deps=deps)
    if fresh:
        return locations, errors
    key = resume_scan_key(task, nodes, deps=deps)
    locations, errors = scan_resume_locations(task, nodes=nodes, cache=cache, deps=deps)
    apply_resume_scan_result(
        task,
        scan_key=key,
        locations=locations,
        errors=errors,
        completed_at=deps.now(),
        source="synchronous_compat",
    )
    return locations, errors
