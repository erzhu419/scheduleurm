from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class DoctorDeps:
    cmd_flag_values: Callable[[list, set], dict]
    arg_value: Callable[[list, str], Optional[str]]
    safe_read_text: Callable[[str], str]
    queued_wait_for_file_block_reason: Callable[[dict], Optional[str]]
    release_task_claims_and_intents: Callable[..., Any]
    queued_has_stale_live_eta: Callable[[dict], bool]
    clear_live_eta_fields: Callable[..., bool]
    seed_pending_eta_from_history: Callable[[dict], Any]
    state_lock: Callable[[], Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]


def flag_values_any(tokens: list, flags: set, *, deps: DoctorDeps) -> list:
    vals = []
    parsed = deps.cmd_flag_values(tokens, flags)
    for flag in flags:
        vals.extend(parsed.get(flag, []))
    return vals


def flag_present_or_true(tokens: list, flag: str) -> bool:
    for tok in tokens:
        if tok == flag:
            return True
        if tok.startswith(flag + "="):
            val = tok.split("=", 1)[1].strip().lower()
            return val not in ("0", "false", "no", "off")
    return False


def simple_sac_large_data_reason(cmd: str, cwd: str, *, deps: DoctorDeps) -> Optional[str]:
    """Return a reason when SimpleSAC should stay local due to external data."""
    text = cmd or ""
    if "h2o+_bus_main.py" not in text:
        return None
    try:
        tokens = shlex.split(text)
    except Exception:
        tokens = text.split()

    dataset_dir_vals = flag_values_any(tokens, {"--dataset_dir", "--dataset-dir"}, deps=deps)
    if any(str(v).strip() for v in dataset_dir_vals):
        return ("SimpleSAC bus training uses --dataset_dir/per-policy HDF5 data "
                "outside cwd; local bus_h2o/datasets_v2 is about 122GB and is not "
                "part of scheduler cwd staging")

    snapshot_disabled = any(tok == "--nouse_snapshot_reset" for tok in tokens)
    for flag in ("--use_snapshot_reset", "--use-snapshot-reset"):
        for tok in tokens:
            if tok.startswith(flag + "="):
                val = tok.split("=", 1)[1].strip().lower()
                if val in ("0", "false", "no", "off"):
                    snapshot_disabled = True

    if not snapshot_disabled:
        return ("SimpleSAC snapshot reset is enabled by default; snapshot-capable "
                "runs may require the external bus_h2o/datasets_v2 archive "
                "(about 122GB), which scheduler does not rsync with cwd")

    return None


def doctor_path_key(path: str) -> str:
    return os.path.normpath(os.path.expandvars(os.path.expanduser(str(path or ""))))


def resolve_cmd_path(raw: str, cwd: str = "") -> str:
    raw = os.path.expandvars(os.path.expanduser(str(raw or "")))
    if not raw:
        return ""
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(cwd or os.getcwd(), raw))


def task_wait_files(task: dict) -> list:
    waits = task.get("wait_for_files") or []
    if isinstance(waits, str):
        waits = [waits]
    return [str(w) for w in waits if str(w or "").strip()]


def task_has_wait_file(task: dict, path: str) -> bool:
    want = doctor_path_key(path)
    return any(doctor_path_key(w) == want for w in task_wait_files(task))


def ready_local_file(path: str) -> bool:
    try:
        key = doctor_path_key(path)
        st = os.stat(key)
        return os.path.isfile(key) and st.st_size > 0
    except OSError:
        return False


def simple_sac_ckpt_for_method(script_path: str, method: str, *, deps: DoctorDeps) -> Optional[str]:
    """Infer run_multiseed_eval.sh's checkpoint path for a method tag."""
    script_path = resolve_cmd_path(script_path)
    if not method or not os.path.isfile(script_path):
        return None
    src = deps.safe_read_text(script_path)
    if not src:
        return None
    pat = r"(?ms)^\s*" + re.escape(method) + r"\)\s*(.*?)^\s*;;"
    match = re.search(pat, src)
    if not match:
        return None
    block = match.group(1)
    ckpt_match = re.search(r"CKPT=(?:\"([^\"]*)\"|'([^']*)'|([^\s#;]+))", block)
    if not ckpt_match:
        return None
    raw = next((group for group in ckpt_match.groups() if group is not None), "")
    raw = raw.strip()
    if not raw:
        return None
    here = os.path.dirname(script_path)
    h2o_root = os.path.dirname(here)
    replacements = {
        "$H2O_ROOT": h2o_root,
        "${H2O_ROOT}": h2o_root,
        "$HERE": here,
        "${HERE}": here,
        "$HOME": str(Path.home()),
        "${HOME}": str(Path.home()),
    }
    for key, val in replacements.items():
        raw = raw.replace(key, val)
    return resolve_cmd_path(raw, here)


def simple_sac_eval_prereq_files(cmd: str, cwd: str, *, deps: DoctorDeps) -> list:
    """Return checkpoint files an eval command needs before dispatch."""
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = (cmd or "").split()
    if not toks:
        return []

    prereqs = []
    for idx, tok in enumerate(toks):
        if os.path.basename(tok) != "run_multiseed_eval.sh":
            continue
        if idx + 1 >= len(toks):
            continue
        script = resolve_cmd_path(tok, cwd)
        ckpt = simple_sac_ckpt_for_method(script, toks[idx + 1], deps=deps)
        if ckpt:
            prereqs.append(ckpt)

    lower = " ".join(toks).lower()
    if "eval" in lower:
        ckpt_vals = flag_values_any(
            toks,
            {"--checkpoint", "--ckpt", "--ckpt_path", "--ckpt-path"},
            deps=deps,
        )
        for value in ckpt_vals:
            path = resolve_cmd_path(value, cwd)
            if path:
                prereqs.append(path)

    out = []
    seen = set()
    for path in prereqs:
        key = doctor_path_key(path)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def simple_sac_train_best_ckpt_from_cmd(cmd: str, cwd: str, *, deps: DoctorDeps) -> Optional[str]:
    """Infer h2o+_bus_main.py's final best checkpoint path from seed/current_time."""
    if "h2o+_bus_main.py" not in (cmd or ""):
        return None
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = (cmd or "").split()
    seed = deps.arg_value(toks, "--seed")
    current_time = deps.arg_value(toks, "--current_time") or deps.arg_value(toks, "--current-time")
    if not seed or not current_time:
        return None
    h2o_root = os.path.dirname(os.path.normpath(cwd or os.getcwd()))
    return os.path.join(
        h2o_root,
        "experiment_output",
        f"h2oplus_bus_seed{seed}_{current_time}",
        "checkpoint_best.pt",
    )


def doctor_issue(task: Optional[dict], code: str, severity: str,
                 message: str, fix: str = "", fixed: bool = False,
                 path: str = "") -> dict:
    rec = {
        "code": code,
        "severity": severity,
        "message": message,
        "fix": fix,
        "fixed": bool(fixed),
    }
    if task:
        rec["task_id"] = task.get("id")
        rec["status"] = task.get("status")
        rec["project"] = task.get("project")
        rec["signature"] = task.get("signature")
    if path:
        rec["path"] = path
    return rec


def doctor_scan_state(state: dict, fix: bool = False, project: str = "",
                      *, deps: DoctorDeps) -> tuple[list, int]:
    """Audit active queue invariants and optionally repair safe queued-task fields."""
    def selected(task: dict) -> bool:
        if not project:
            return True
        return fnmatch.fnmatch(task.get("project") or "", project)

    tasks = [task for task in state.get("tasks", []) if selected(task)]
    active = [task for task in tasks if task.get("status") in ("queued", "launching", "running")]
    issues = []
    changed = 0

    needed_paths = {}
    producer_by_path = {}
    for task in active:
        train_ckpt = simple_sac_train_best_ckpt_from_cmd(
            task.get("cmd") or "",
            task.get("cwd") or "",
            deps=deps,
        )
        if train_ckpt:
            producer_by_path[doctor_path_key(train_ckpt)] = task

    for task in active:
        prereqs = simple_sac_eval_prereq_files(
            task.get("cmd") or "",
            task.get("cwd") or "",
            deps=deps,
        )
        for path in prereqs:
            needed_paths.setdefault(doctor_path_key(path), []).append(task)
        missing_waits = [path for path in prereqs if not task_has_wait_file(task, path)]
        if not missing_waits:
            continue
        if task.get("status") == "queued":
            if fix:
                waits = task_wait_files(task)
                waits.extend(
                    path for path in missing_waits
                    if not task_has_wait_file({"wait_for_files": waits}, path)
                )
                task["wait_for_files"] = waits
                task["last_block_reason"] = (
                    deps.queued_wait_for_file_block_reason(task)
                    or task.get("last_block_reason")
                )
                changed += 1
                issues.append(doctor_issue(
                    task,
                    "eval_missing_wait_for_file",
                    "fixed",
                    "queued eval was missing checkpoint prerequisite gating",
                    fixed=True,
                    path=", ".join(missing_waits),
                ))
            else:
                issues.append(doctor_issue(
                    task,
                    "eval_missing_wait_for_file",
                    "fixable",
                    "queued eval can dispatch before its checkpoint exists",
                    "doctor --fix will add wait_for_files",
                    path=", ".join(missing_waits),
                ))
        else:
            issues.append(doctor_issue(
                task,
                "eval_already_running_without_wait_for_file",
                "warn",
                "eval is already launching/running without scheduler-level checkpoint gating",
                "inspect log; cancel/requeue manually if it is burning time on a missing ckpt",
                path=", ".join(missing_waits),
            ))

    for task in active:
        reason = simple_sac_large_data_reason(
            task.get("cmd") or "",
            task.get("cwd") or "",
            deps=deps,
        )
        if not reason or task.get("allow_remote_large_data"):
            continue
        if task.get("status") == "queued":
            if task.get("require_node") == "local":
                continue
            if fix:
                old_node = task.get("node")
                task["require_node"] = "local"
                if task.get("preferred_node") != "local":
                    task["preferred_node"] = None
                task["node"] = None
                task["gpu_idx"] = None
                task["remote_pids"] = []
                task["last_block_reason"] = f"doctor: forced local for SimpleSAC large data: {reason}"
                try:
                    deps.release_task_claims_and_intents(
                        task,
                        extra_nodes=[old_node] if old_node else None,
                    )
                except Exception:
                    pass
                changed += 1
                issues.append(doctor_issue(
                    task,
                    "simple_sac_large_data_not_local",
                    "fixed",
                    reason,
                    fixed=True,
                ))
            else:
                issues.append(doctor_issue(
                    task,
                    "simple_sac_large_data_not_local",
                    "fixable",
                    reason,
                    "doctor --fix will set require_node=local",
                ))
        elif task.get("node") != "local":
            issues.append(doctor_issue(
                task,
                "simple_sac_large_data_running_remote",
                "error",
                reason,
                "running task cannot be safely rewritten; cancel manually if this is wrong",
            ))

    for path, eval_tasks in sorted(needed_paths.items()):
        producer = producer_by_path.get(path)
        if producer and producer.get("status") == "queued" and producer.get("priority") != "high":
            if fix:
                producer["priority"] = "high"
                producer["last_block_reason"] = (
                    f"doctor: promoted training because eval task(s) wait for {path}"
                )
                changed += 1
                issues.append(doctor_issue(
                    producer,
                    "train_priority_below_dependent_eval",
                    "fixed",
                    "training produces a checkpoint needed by queued evals but was not high priority",
                    fixed=True,
                    path=path,
                ))
            else:
                issues.append(doctor_issue(
                    producer,
                    "train_priority_below_dependent_eval",
                    "fixable",
                    "training produces a checkpoint needed by queued evals but is not high priority",
                    "doctor --fix will set priority=high",
                    path=path,
                ))
        if not ready_local_file(path) and not producer:
            ids = ", ".join(task.get("id", "?") for task in eval_tasks[:4])
            more = "" if len(eval_tasks) <= 4 else f" (+{len(eval_tasks) - 4} more)"
            issues.append(doctor_issue(
                None,
                "eval_checkpoint_missing_no_active_producer",
                "warn",
                f"checkpoint is missing and no active SimpleSAC producer was inferred; eval tasks: {ids}{more}",
                "submit/restore the matching train task, or remove the evals if obsolete",
                path=path,
            ))

    for task in active:
        if not deps.queued_has_stale_live_eta(task):
            continue
        msg = (
            f"queued task carries stale live ETA/progress "
            f"(eta_source={task.get('eta_source')!r}, runtime_est_source={task.get('runtime_est_source')!r}); "
            "queued ETA must be reseeded from local-test/runtime history"
        )
        if task.get("status") == "queued" and fix:
            deps.clear_live_eta_fields(task, clear_runtime_projection=True)
            deps.seed_pending_eta_from_history({"tasks": [task]})
            changed += 1
            issues.append(doctor_issue(
                task,
                "queued_stale_live_eta",
                "fixed",
                msg,
                fixed=True,
            ))
        else:
            issues.append(doctor_issue(
                task,
                "queued_stale_live_eta",
                "fixable",
                msg,
                "doctor --fix will clear live ETA fields and reseed from history",
            ))

    return issues, changed


def cmd_doctor(args, *, deps: DoctorDeps) -> None:
    with deps.state_lock():
        state = deps.load_state()
        issues, changed = doctor_scan_state(
            state,
            fix=bool(getattr(args, "fix", False)),
            project=getattr(args, "project", "") or "",
            deps=deps,
        )
        if changed:
            deps.save_state(state)
    result = {"ok": True, "fixed": changed, "issues": issues}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    if not issues:
        print("doctor: no active queue invariant issues found")
        return
    print(f"doctor: {len(issues)} issue(s), {changed} fixed")
    for issue in issues:
        tid = issue.get("task_id") or "-"
        path = f" path={issue['path']}" if issue.get("path") else ""
        print(f"  [{issue['severity']}] {issue['code']} {tid}: {issue['message']}{path}")
        if issue.get("fix") and not issue.get("fixed"):
            print(f"      fix: {issue['fix']}")
