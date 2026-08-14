"""Active-state conflict checks for a submitted task."""

from __future__ import annotations

from typing import Any

try:
    from .task_common import SubmitTaskDeps, _refuse
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .task_common import SubmitTaskDeps, _refuse


def _validate_allowed_nodes(allowed_nodes: list[str], deps: SubmitTaskDeps) -> None:
    unknown_allowed = [n for n in allowed_nodes if n not in deps.known_nodes]
    if unknown_allowed:
        _refuse(
            f"REFUSED: --allowed-node contains unknown node(s): {unknown_allowed}. "
            f"Known nodes: {list(deps.known_nodes.keys())}",
        )


def _submit_identity_payload(
    args: Any,
    *,
    sig: str,
    extra_env: dict,
    allowed_nodes: list[str],
    parallel: dict,
) -> dict:
    return {
        "signature": sig,
        "cmd": args.cmd,
        "cwd": args.cwd,
        "extra_env": extra_env,
        "env_spec": getattr(args, "env_spec", None) or "none",
        "image": getattr(args, "image", None) or "",
        "ckpt_dir": args.ckpt_dir,
        "ckpt_glob": args.ckpt_glob,
        "resume_flag": args.resume_flag or "",
        "result_dir": getattr(args, "result_dir", None) or None,
        "local_result_dir": getattr(args, "local_result_dir", None) or None,
        "cpu_parallel_items": parallel["items"],
        "cpu_parallel_total_items": parallel["total_items"],
        "cpu_parallel_logical_items": parallel["logical_items"],
        "cpu_parallel_item_multiplier": parallel["item_multiplier"],
        "cpu_parallel_start": parallel["start"],
        "cpu_parallel_end": parallel["end"],
        "allowed_nodes": allowed_nodes,
    }


def _validate_duplicate_identity(
    state: dict,
    args: Any,
    *,
    sig: str,
    extra_env: dict,
    allowed_nodes: list[str],
    parallel: dict,
    deps: SubmitTaskDeps,
) -> None:
    if getattr(args, "allow_duplicate", False):
        return
    submit_identity = deps.task_run_identity(
        _submit_identity_payload(
            args,
            sig=sig,
            extra_env=extra_env,
            allowed_nodes=allowed_nodes,
            parallel=parallel,
        )
    )
    for existing in state["tasks"]:
        if deps.task_run_identity(existing) != submit_identity:
            continue
        if existing.get("status") in ("queued", "running", "launching"):
            _refuse(
                f"DUPLICATE: {existing['id']} ({existing['status']}) has identical run identity",
                f"  signature: {sig}",
                f"  cmd: {args.cmd[:120]}",
                f"  cwd: {args.cwd}",
                "  pass --allow-duplicate to override",
                stream="stdout",
            )
        if deps.task_has_recorded_launch_artifacts(existing):
            launch_state, launch_reason = deps.recorded_launch_safety_state(existing)
            if launch_state in ("alive", "unknown"):
                _refuse(
                    f"DUPLICATE: {existing['id']} ({existing.get('status')}) "
                    f"has {launch_state} launch artifacts for identical run identity",
                    f"  reason: {launch_reason}",
                    f"  signature: {sig}",
                    f"  cmd: {args.cmd[:120]}",
                    f"  cwd: {args.cwd}",
                    "  pass --allow-duplicate to override",
                    stream="stdout",
                )


def _validate_ckpt_conflict(
    state: dict,
    args: Any,
    *,
    sig: str,
    deps: SubmitTaskDeps,
) -> None:
    if not args.ckpt_dir or getattr(args, "allow_shared_ckpt_dir", False):
        return
    for existing in state["tasks"]:
        if existing.get("status") not in ("queued", "running", "launching"):
            continue
        if not deps.same_declared_path(existing.get("ckpt_dir"), args.ckpt_dir):
            continue
        _refuse(
            "REFUSED: --ckpt-dir already in use by an active task.",
            f"  conflicting task: {existing['id']} ({existing['status']}) sig={existing.get('signature','')!r}",
            f"  this submit:      sig={sig!r}",
            f"  shared ckpt-dir:  {args.ckpt_dir}",
            "  reason: concurrent procs writing the same ckpt path corrupt each other",
            "  Either:",
            "    (a) cancel/wait for the existing task, OR",
            "    (b) point this submit to a different --ckpt-dir, OR",
            "    (c) pass --allow-shared-ckpt-dir if you know what you're doing",
        )


def _validate_result_conflict(state: dict, args: Any, deps: SubmitTaskDeps) -> None:
    submit_local_result_dir = (
        (getattr(args, "local_result_dir", None) or None)
        or (getattr(args, "result_dir", None) or None)
    )
    if not submit_local_result_dir or getattr(args, "allow_shared_result_dir", False):
        return
    for existing in state["tasks"]:
        if not deps.active_or_unsynced_result_status(existing):
            continue
        existing_dst = existing.get("local_result_dir") or existing.get("result_dir")
        if not deps.same_declared_path(existing_dst, submit_local_result_dir):
            continue
        _refuse(
            "REFUSED: --local-result-dir destination already in use by an unfinished/unsynced task.",
            f"  conflicting task: {existing['id']} ({existing.get('status')}) sig={existing.get('signature','')!r}",
            f"  destination:      {submit_local_result_dir}",
            "  reason: result sync rsyncs remote result_dir contents into this directory; "
            "two tasks sharing it can merge or overwrite files.",
            "  Use a per-task subdirectory, or pass --allow-shared-result-dir if the "
            "layout is intentionally collision-free.",
        )
