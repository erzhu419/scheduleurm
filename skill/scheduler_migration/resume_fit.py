"""Resume-checkpoint staging checks and fit-wait ETA estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ResumeFitDeps:
    node_configs: dict
    staging_cap_exceeded: dict
    staging_fails: dict
    staging_ttl_s: int
    staging_fail_cooldown_s: int
    resume_ckpt_migrate_overhead_s: int
    resume_ckpt_migrate_est_mbps: float
    remote_path_for_node: Callable[[str, str], str]
    staging_cache_hit: Callable[[tuple], bool]
    gpu_fits: Callable[[dict, dict, dict], bool]
    pick_placement: Callable[..., tuple | None]
    task_required_gpu_idx: Callable[[dict], int | None]
    compute_node_load_seconds: Callable[[dict], dict]
    now: Callable[[], float]


def resume_ckpt_stage_key(
    source_node: str,
    target_node: str,
    ckpt_dir: str,
    source_location: Optional[dict] = None,
) -> tuple:
    """Key staging by checkpoint generation, not only by directory path."""
    source_location = source_location or {}
    mtime = source_location.get("mtime")
    size = source_location.get("size")
    if mtime is None and size is None:
        return ("resume_ckpt", source_node, target_node, ckpt_dir)
    try:
        mtime_key = round(float(mtime or 0.0), 6)
    except (TypeError, ValueError):
        mtime_key = 0.0
    try:
        size_key = int(size or 0)
    except (TypeError, ValueError):
        size_key = 0
    fingerprinted_path = (ckpt_dir, mtime_key, size_key)
    return ("resume_ckpt", source_node, target_node, fingerprinted_path)


def _location_freshness(location: dict) -> tuple[float, int]:
    try:
        mtime = float(location.get("mtime") or 0.0)
    except (TypeError, ValueError):
        mtime = 0.0
    try:
        size = int(location.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    return mtime, size


def resume_location_for_target(source_loc: dict, target_node: str, *, deps: ResumeFitDeps) -> dict:
    staged = dict(source_loc or {})
    staged["node"] = target_node
    if staged.get("path"):
        staged["path"] = deps.remote_path_for_node(target_node, staged["path"])
    return staged


def _route_sources_for_target(
    locations: list,
    target_node: str,
    *,
    deps: ResumeFitDeps,
) -> list:
    """Return checkpoint sources supported by launch-time staging, in preference order."""
    target_info = deps.node_configs.get(target_node, {}) or {}
    target_host = target_info.get("host")
    candidates = [loc for loc in locations if loc.get("node") != target_node]
    if target_info.get("windows"):
        return [loc for loc in candidates if loc.get("node") == "local"]
    if target_host:
        local = [loc for loc in candidates if loc.get("node") == "local"]
        remote_linux = [
            loc for loc in candidates
            if loc.get("node")
            and (deps.node_configs.get(loc.get("node"), {}) or {}).get("host")
            and not (deps.node_configs.get(loc.get("node"), {}) or {}).get("windows")
        ]
        return local + remote_linux
    return [
        loc for loc in candidates
        if loc.get("node")
        and (deps.node_configs.get(loc.get("node"), {}) or {}).get("host")
        and not (deps.node_configs.get(loc.get("node"), {}) or {}).get("windows")
    ]


def resume_stage_source_for_target(
    task: dict,
    target_node: str,
    locations: Optional[list] = None,
    *,
    deps: ResumeFitDeps,
) -> Optional[dict]:
    """Pick a checkpoint source that can be rsync'd to target for launch staging."""
    locations = list(locations if locations is not None else (task.get("resume_locations") or []))
    if not target_node or not locations:
        return None
    newest = max(locations, key=_location_freshness)
    target_locations = [
        loc for loc in locations if loc.get("node") == target_node
    ]
    if target_locations:
        target_location = max(target_locations, key=_location_freshness)
        newest_mtime, newest_size = _location_freshness(newest)
        target_mtime, target_size = _location_freshness(target_location)
        metadata_available = newest_mtime > 0.0 or newest_size > 0
        target_is_current = (
            not metadata_available
            or (target_mtime >= newest_mtime and target_size == newest_size)
        )
        if target_is_current:
            # A failed rsync can leave the globbed checkpoint file on the
            # target with the same mtime/size while the rest of ckpt_dir is
            # incomplete.  When an equivalent generation is still available
            # through a supported local-side route, trust the target only if
            # the generation-specific persistent staging marker exists.
            current_freshness = _location_freshness(target_location)
            equivalent = [
                loc for loc in locations
                if loc.get("node") != target_node
                and _location_freshness(loc) == current_freshness
            ]
            route_sources = _route_sources_for_target(
                equivalent, target_node, deps=deps)
            if route_sources and task.get("ckpt_dir"):
                source = route_sources[0]
                key = resume_ckpt_stage_key(
                    source.get("node"), target_node, task["ckpt_dir"], source)
                if not deps.staging_cache_hit(key):
                    return dict(source)
            return dict(target_location)

    # A staging route is useful only when it carries the newest generation.
    # Falling back to an older local copy can silently resume from stale state.
    newest_freshness = _location_freshness(newest)
    current_generation = [
        loc for loc in locations
        if _location_freshness(loc) == newest_freshness
    ]
    route_sources = _route_sources_for_target(
        current_generation, target_node, deps=deps)
    if route_sources:
        return dict(route_sources[0])
    return None


def resume_checkpoint_stage_check(
    task: dict,
    target_node: str,
    locations: Optional[list] = None,
    *,
    deps: ResumeFitDeps,
) -> tuple:
    """Cache-only check for launch-time checkpoint staging."""
    locations = list(locations if locations is not None else (task.get("resume_locations") or []))
    if not locations:
        return "ready", None, "no existing checkpoint"
    src = resume_stage_source_for_target(task, target_node, locations, deps=deps)
    if not src:
        return "unsupported", None, "no supported checkpoint staging route"
    if src.get("node") == target_node:
        return "ready", dict(src), "checkpoint already on target"
    ckpt_dir = task.get("ckpt_dir")
    if not ckpt_dir:
        return "ready", src, "no ckpt_dir"
    key = resume_ckpt_stage_key(
        src.get("node"), target_node, ckpt_dir, src)
    if deps.staging_cache_hit(key):
        staged = resume_location_for_target(src, target_node, deps=deps)
        return "ready", staged, "checkpoint staged to target"
    ts = deps.staging_cap_exceeded.get(key)
    if ts is not None:
        if (deps.now() - ts) > deps.staging_ttl_s:
            deps.staging_cap_exceeded.pop(key, None)
        else:
            return "cap_exceeded", src, "checkpoint directory exceeds staging cap"
    fail = deps.staging_fails.get(key)
    if fail is not None:
        fail_ts = fail[0] if isinstance(fail, tuple) else fail
        if (deps.now() - fail_ts) > deps.staging_fail_cooldown_s:
            deps.staging_fails.pop(key, None)
        else:
            msg = fail[1] if isinstance(fail, tuple) and len(fail) > 1 else "unknown"
            return "stage_failed", src, msg
    return "needs_stage", src, "checkpoint not yet staged to target"


def task_wait_eta_seconds(task: dict) -> int:
    try:
        return max(0, int(task.get("eta_seconds") or 0))
    except Exception:
        return 0


def resume_ckpt_stage_estimate_seconds(
    stage_state: str,
    source_loc: Optional[dict],
    *,
    deps: ResumeFitDeps,
) -> int:
    if stage_state == "ready":
        return 0
    if not source_loc:
        return 10 ** 9
    try:
        size_bytes = max(0, int(source_loc.get("size") or 0))
    except Exception:
        size_bytes = 0
    size_mb = max(1, (size_bytes + 1024 * 1024 - 1) // (1024 * 1024))
    return int(deps.resume_ckpt_migrate_overhead_s + (size_mb / deps.resume_ckpt_migrate_est_mbps) + 0.999)


def running_tasks_for_node(state: dict, node_name: str, gpu_idx=None) -> list:
    out = []
    for task in state.get("tasks", []):
        if task.get("status") != "running" or task.get("node") != node_name:
            continue
        if gpu_idx is not None:
            try:
                if int(task.get("gpu_idx")) != int(gpu_idx):
                    continue
            except Exception:
                continue
        out.append(task)
    return out


def task_vram_pressure_mb(task: dict) -> int:
    vals = []
    for key in ("current_vram_mb", "peak_vram_mb", "est_vram_mb"):
        try:
            vals.append(max(0, int(task.get(key) or 0)))
        except Exception:
            pass
    return max(vals or [0])


def estimate_gpu_fit_wait_seconds(
    task: dict,
    gpu_state: dict,
    node_info: dict,
    running_gpu_tasks: list,
    *,
    deps: ResumeFitDeps,
) -> Optional[int]:
    ordered = [(task_wait_eta_seconds(rt), rt) for rt in running_gpu_tasks]
    ordered = [(eta, rt) for eta, rt in ordered if eta > 0]
    if not ordered:
        return None
    ordered.sort(key=lambda item: item[0])
    synthetic = dict(gpu_state)
    try:
        total = max(1, int(synthetic.get("total_mb") or 1))
        used = max(0, int(synthetic.get("used_mb") or 0))
    except Exception:
        total, used = 1, 0
    running_count = int(synthetic.get("running_task_count") or len(running_gpu_tasks))
    for eta, running_task in ordered:
        used = max(0, used - task_vram_pressure_mb(running_task))
        running_count = max(0, running_count - 1)
        synthetic["used_mb"] = used
        synthetic["free_mb"] = max(0, total - used)
        synthetic["running_task_count"] = running_count
        if deps.gpu_fits(task, synthetic, node_info):
            return eta
    return ordered[-1][0]


def estimate_node_fit_wait_seconds(
    task: dict,
    node_state: dict,
    state: dict,
    *,
    deps: ResumeFitDeps,
) -> Optional[int]:
    if not node_state or not node_state.get("alive"):
        return None
    node_name = node_state.get("name")
    if not node_name:
        return None
    if deps.pick_placement(task, [node_state], extra_allowed_nodes=[node_name]) is not None:
        return 0
    node_info = deps.node_configs.get(node_name, {}) or {}
    if int(task.get("est_vram_mb") or 0) > 0 and node_state.get("gpus"):
        waits = []
        require_gpu_idx = deps.task_required_gpu_idx(task)
        for gpu_state in node_state.get("gpus") or []:
            try:
                gpu_idx = int(gpu_state.get("idx"))
            except Exception:
                continue
            if require_gpu_idx is not None and gpu_idx != require_gpu_idx:
                continue
            running_gpu_tasks = running_tasks_for_node(state, node_name, gpu_idx)
            wait = estimate_gpu_fit_wait_seconds(
                task, gpu_state, node_info, running_gpu_tasks, deps=deps)
            if wait is not None:
                waits.append(wait)
        if waits:
            return min(waits)
    running_node_tasks = running_tasks_for_node(state, node_name)
    waits = [task_wait_eta_seconds(rt) for rt in running_node_tasks
             if task_wait_eta_seconds(rt) > 0]
    if waits:
        return min(waits)
    load = int(deps.compute_node_load_seconds(state).get(node_name) or 0)
    return load if load > 0 else None


def record_staged_resume_location(
    task: dict,
    target_node: str,
    source_loc: Optional[dict],
    *,
    deps: ResumeFitDeps,
) -> None:
    """Record that source_loc's checkpoint path is now available on target_node."""
    if not source_loc or not target_node:
        return
    staged_loc = resume_location_for_target(source_loc, target_node, deps=deps)
    locs = [loc for loc in (task.get("resume_locations") or [])
            if loc.get("node") != target_node]
    locs.append(staged_loc)
    locs.sort(key=lambda item: (float(item.get("mtime") or 0), str(item.get("node") or "")), reverse=True)
    task["resume_locations"] = locs
    ordered_nodes = []
    for loc in locs:
        node = loc.get("node")
        if node and node not in ordered_nodes:
            ordered_nodes.append(node)
    task["resume_preferred_nodes"] = ordered_nodes
    best = locs[0] if locs else staged_loc
    task["resume_checkpoint_node"] = best.get("node")
    task["resume_from"] = (
        next((loc.get("path") for loc in locs
              if loc.get("node") == target_node and loc.get("path")), None)
        or best.get("path")
        or task.get("resume_from")
    )
