"""CPU-only production workload service-curve runner.

This is the runner intended for Module53 sub-buckets such as
``freqduet_cpu_ablation|c_17_32`` and ``sumo_eval_cpu|c_le2``.  It mirrors the
existing service-curve runners but does not request or pin GPUs.

The module also exposes a dry-run plan builder so probe batches can be reviewed
before any Scheduleurm tasks are submitted.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping

from .cpu_service_curve_validation import _annotate_summary, _build_verdict, _phase_name, _write_markdown
from .service_curve_validation import profile_progress_count, select_measurement_summary
from .sweetspot_ab_validation import (
    _cancel_all,
    _now_stamp,
    _parse_submit_id,
    _record_event,
    _record_observation,
    _repo_root,
    _run_cmd,
    _scheduler_cmd,
    _status_refresh,
    _tasks_by_ids,
    _watcher_status,
    _write_json,
)
from .trace_export import init_run


REMOTE_PROGRESS_WRAPPER_ROOT = "/tmp/scheduleurm_progress_wrapper_pkg"
REMOTE_PROGRESS_WRAPPER = (
    f"{REMOTE_PROGRESS_WRAPPER_ROOT}/algorithm/experiments/progress_wrapper.py"
)
REMOTE_CSV_PROGRESS_WRAPPER = (
    f"{REMOTE_PROGRESS_WRAPPER_ROOT}/algorithm/experiments/csv_progress_wrapper.py"
)
WINDOWS_PROGRESS_WRAPPER_ROOT = (
    r"F:\erzhu419_smoke\.scheduleurm\progress_wrapper_pkg"
)
WINDOWS_PROGRESS_WRAPPER = (
    WINDOWS_PROGRESS_WRAPPER_ROOT + r"\algorithm\experiments\progress_wrapper.py"
)
WINDOWS_CSV_PROGRESS_WRAPPER = (
    WINDOWS_PROGRESS_WRAPPER_ROOT + r"\algorithm\experiments\csv_progress_wrapper.py"
)
WINDOWS_NODES = {
    "jtl110cpu": {
        "host": "tf290q6n.zjz-service.cn",
        "port": 22945,
        "user": "erzhu419",
        "identity": str(Path.home() / ".ssh" / "id_ed25519"),
    },
    "jtl110cpu2": {
        "host": "tf290q6n.zjz-service.cn",
        "port": 23565,
        "user": "erzhu419",
        "identity": str(Path.home() / ".ssh" / "id_ed25519"),
    },
}
_SCHEDULEURM_PROGRESS_RE = re.compile(
    r"ScheduleurmProgress\s+(\w+)\s+(\d+)(?:/(\d+))?"
    r"(?:.*?\brate=([0-9.eE+-]+)\s+(\w+)/s)?",
    re.IGNORECASE,
)


def render_cpu_workload_command(
    template: str,
    *,
    run_id: str,
    sub_bucket: str,
    phase: str,
    profile: int,
    index: int,
    seed_base: int,
    total_units: int,
    output_root: str,
    node: str,
    cpu_cores: int,
    work_items_per_task: int = 1,
) -> dict[str, Any]:
    """Render a CPU production workload command template.

    Supported placeholders are explicit so typos fail before submission.
    """

    work_items = max(1, int(work_items_per_task))
    seed = int(seed_base) + int(index) * work_items
    seeds = [seed + offset for offset in range(work_items)]
    safe_sub_bucket = sub_bucket.replace("|", "_").replace("/", "_")
    run_name = f"{run_id}_{safe_sub_bucket}_{phase}_{index}"
    values: dict[str, Any] = {
        "run_id": run_id,
        "sub_bucket": sub_bucket,
        "safe_sub_bucket": safe_sub_bucket,
        "phase": phase,
        "profile": int(profile),
        "index": int(index),
        "seed": seed,
        "seed_csv": ",".join(str(x) for x in seeds),
        "seed_start": seeds[0],
        "seed_end": seeds[-1] + 1,
        "work_items": work_items,
        "total_units": int(total_units),
        "progress_total_units": int(total_units) * work_items,
        "output_root": output_root.rstrip("/"),
        "run_name": run_name,
        "node": node,
        "cpu_cores": int(cpu_cores),
    }
    unknown = sorted(_template_fields(template) - set(values))
    if unknown:
        raise ValueError(f"unknown CPU workload template placeholders: {unknown}")
    return {
        "cmd": template.format(**values),
        "run_name": run_name,
        "seed": seed,
        "values": values,
    }


def build_submission_plan(
    *,
    run_id: str,
    sub_bucket: str,
    profiles: Iterable[int],
    cmd_template: str,
    node: str,
    cwd: str,
    output_root: str,
    seed_base: int,
    total_units: int,
    work_items_per_task: int = 1,
    cpu_cores: int,
    ram_mb: int,
    project: str,
    signature_prefix: str,
    progress_unit: str,
    progress_wrapper: str = REMOTE_PROGRESS_WRAPPER,
    progress_wrapper_kind: str = "line",
    progress_csv_glob: str = "",
    progress_poll_s: float = 5.0,
    child_shell: str = "bash",
) -> dict[str, Any]:
    profile_rows = []
    for profile in sorted({int(p) for p in profiles if int(p) > 0}):
        phase = _phase_name(profile)
        tasks = []
        for index in range(profile):
            rendered = render_cpu_workload_command(
                cmd_template,
                run_id=run_id,
                sub_bucket=sub_bucket,
                phase=phase,
                profile=profile,
                index=index,
                seed_base=seed_base,
                total_units=total_units,
                output_root=output_root,
                node=node,
                cpu_cores=cpu_cores,
                work_items_per_task=work_items_per_task,
            )
            csv_glob = _render_progress_csv_glob(
                progress_csv_glob,
                rendered["values"],
            )
            wrapped = _wrap_with_progress(
                rendered["cmd"],
                progress_wrapper=progress_wrapper,
                progress_wrapper_kind=progress_wrapper_kind,
                progress_unit=progress_unit,
                total_units=int(rendered["values"]["progress_total_units"]),
                progress_csv_glob=csv_glob,
                progress_poll_s=progress_poll_s,
                child_shell=child_shell,
            )
            tasks.append({
                "index": index,
                "seed": rendered["seed"],
                "seed_csv": rendered["values"]["seed_csv"],
                "work_items": int(rendered["values"]["work_items"]),
                "run_name": rendered["run_name"],
                "cmd": wrapped,
                "cwd": cwd,
                "signature": f"{signature_prefix}/{sub_bucket}/{run_id}/{phase}/{index}",
                "description": f"Production CPU curve {sub_bucket} {phase} task {index}",
                "vram_mb": 0,
                "ram_mb": int(ram_mb),
                "cpu_cores": int(cpu_cores),
                "project": project,
                "require_node": node,
            })
        profile_rows.append({"profile": profile, "phase": phase, "tasks": tasks})
    return {
        "run_id": run_id,
        "sub_bucket": sub_bucket,
        "node": node,
        "cwd": cwd,
        "output_root": output_root,
        "progress_unit": progress_unit,
        "total_units": int(total_units),
        "work_items_per_task": int(work_items_per_task),
        "progress_total_units_per_task": int(total_units) * max(1, int(work_items_per_task)),
        "cpu_cores": int(cpu_cores),
        "ram_mb": int(ram_mb),
        "project": project,
        "signature_prefix": signature_prefix,
        "progress_wrapper_kind": progress_wrapper_kind,
        "progress_csv_glob": progress_csv_glob,
        "progress_poll_s": float(progress_poll_s),
        "child_shell": child_shell,
        "profiles": profile_rows,
        "task_count": sum(len(row["tasks"]) for row in profile_rows),
        "theorem_status": "plan_only_not_measured",
    }


def markdown_plan(plan: Mapping[str, Any]) -> str:
    lines = [
        "# Production CPU Workload Service-Curve Plan",
        "",
        "```text",
        f"run_id = {plan.get('run_id')}",
        f"sub_bucket = {plan.get('sub_bucket')}",
        f"node = {plan.get('node')}",
        f"task_count = {plan.get('task_count')}",
        f"progress_unit = {plan.get('progress_unit')}",
        f"total_units = {plan.get('total_units')}",
        f"theorem_status = {plan.get('theorem_status')}",
        "```",
        "",
        "| Profile | Tasks | CPU/task | RAM MB/task |",
        "|---:|---:|---:|---:|",
    ]
    for row in plan.get("profiles") or []:
        tasks = list(row.get("tasks") or [])
        cpu = tasks[0].get("cpu_cores") if tasks else ""
        ram = tasks[0].get("ram_mb") if tasks else ""
        lines.append(f"| {row.get('profile')} | {len(tasks)} | {cpu} | {ram} |")
    lines.extend([
        "",
        "## First Commands",
        "",
    ])
    for row in (plan.get("profiles") or [])[:2]:
        for task in (row.get("tasks") or [])[:2]:
            lines.extend([
                f"### `{task.get('signature')}`",
                "",
                "```bash",
                str(task.get("cmd") or ""),
                "```",
                "",
            ])
    lines.extend([
        "## Interpretation",
        "",
        "This is a dry-run submission plan.  It is not a service certificate until",
        "the runner submits the tasks, observes stable progress, writes profile",
        "summaries, and those summaries are loaded into the service cache.",
        "",
    ])
    return "\n".join(lines)


def _template_fields(template: str) -> set[str]:
    return {
        str(field)
        for _, field, _, _ in Formatter().parse(template or "")
        if field
    }


def _wrap_with_progress(
    command: str,
    *,
    progress_wrapper: str,
    progress_wrapper_kind: str,
    progress_unit: str,
    total_units: int,
    progress_csv_glob: str,
    progress_poll_s: float,
    child_shell: str,
) -> str:
    kind = str(progress_wrapper_kind or "line").strip().lower()
    outer = [
        "python3",
        "-u",
        str(progress_wrapper),
        "--unit",
        str(progress_unit),
        "--total",
        str(int(total_units)),
    ]
    if kind == "csv":
        if not progress_csv_glob:
            raise ValueError("progress_csv_glob is required for csv progress wrapper")
        outer.extend([
            "--poll-s",
            str(float(progress_poll_s)),
            "--csv-glob",
            str(progress_csv_glob),
        ])
    elif kind != "line":
        raise ValueError(f"unknown progress wrapper kind: {progress_wrapper_kind!r}")

    if str(child_shell or "bash").lower() == "argv":
        child = shlex.split(command)
    else:
        child = ["bash", "-lc", command]
    return shlex.join(outer + ["--"] + child)


def _render_progress_csv_glob(template: str, values: Mapping[str, Any]) -> str:
    raw = str(template or "").strip()
    if not raw:
        raw = "{output_root}/{run_name}/*/diagnostics.csv"
    unknown = sorted(_template_fields(raw) - set(values))
    if unknown:
        raise ValueError(f"unknown progress CSV glob placeholders: {unknown}")
    return raw.format(**values)


def _read_template(args: argparse.Namespace) -> str:
    if args.cmd_template_file:
        return Path(args.cmd_template_file).read_text(encoding="utf-8").strip()
    return str(args.cmd_template or "").strip()


def _deploy_progress_wrapper(node: str, raw_dir: Path) -> str:
    local_dir = _repo_root() / "algorithm" / "experiments"
    if _is_windows_node(node):
        return _deploy_windows_progress_package(node, raw_dir)
    if node == "local":
        target_root = Path("/tmp/scheduleurm_progress_wrapper_pkg")
        target_dir = target_root / "algorithm" / "experiments"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_root / "algorithm" / "__init__.py").touch()
        (target_dir / "__init__.py").touch()
        for name in ("progress_wrapper.py", "progress_units.py", "csv_progress_wrapper.py"):
            shutil.copy2(local_dir / name, target_dir / name)
        return str(target_dir / "progress_wrapper.py")
    remote_dir = f"{REMOTE_PROGRESS_WRAPPER_ROOT}/algorithm/experiments"
    _run_cmd(
        [
            "ssh",
            node,
            (
                f"mkdir -p {remote_dir} "
                f"&& touch {REMOTE_PROGRESS_WRAPPER_ROOT}/algorithm/__init__.py "
                f"{REMOTE_PROGRESS_WRAPPER_ROOT}/algorithm/experiments/__init__.py"
            ),
        ],
        cwd=_repo_root(),
        raw_dir=raw_dir,
        label="deploy_cpu_progress_wrapper_mkdir",
    )
    _run_cmd(
        [
            "scp",
            str(local_dir / "progress_wrapper.py"),
            str(local_dir / "progress_units.py"),
            str(local_dir / "csv_progress_wrapper.py"),
            f"{node}:{remote_dir}/",
        ],
        cwd=_repo_root(),
        raw_dir=raw_dir,
        label="deploy_cpu_progress_wrapper_scp",
    )
    return REMOTE_PROGRESS_WRAPPER


def _mkdir_target(node: str, cwd: str, raw_dir: Path) -> None:
    if _is_windows_node(node):
        win_cwd = _windows_path_for_node(node, cwd)
        rc, out, err = _run_windows_ps(
            node,
            (
                f"if (Test-Path -LiteralPath {_ps_quote(win_cwd)} -PathType Container) "
                "{ 'OK' } else { 'MISSING' }"
            ),
            timeout=15,
            check=False,
        )
        if rc != 0 or "OK" not in out:
            raise SystemExit(
                f"Windows target cwd missing on {node}: {win_cwd}; "
                f"{(err or out).strip()[:160]}"
            )
        return
    if node == "local":
        _run_cmd(["mkdir", "-p", cwd], cwd=_repo_root(), raw_dir=raw_dir, label="local_mkdir_cwd")
    else:
        _run_cmd(["ssh", node, "mkdir", "-p", cwd], cwd=_repo_root(), raw_dir=raw_dir, label="remote_mkdir_cwd")


def _submit_profile_from_plan(*, run_dir: Path, phase: str, tasks: list[Mapping[str, Any]]) -> list[str]:
    raw_dir = run_dir / "raw"
    ids: list[str] = []
    for task in tasks:
        proc = _scheduler_cmd(
            [
                "submit",
                "--description",
                str(task["description"]),
                "--cmd",
                str(task["cmd"]),
                "--cwd",
                str(task["cwd"]),
                "--signature",
                str(task["signature"]),
                "--vram",
                "0",
                "--ram-mb",
                str(int(task["ram_mb"])),
                "--cpu",
                str(int(task["cpu_cores"])),
                "--priority",
                "high",
                "--project",
                str(task["project"]),
                "--require-node",
                str(task["require_node"]),
                "--allow-no-ckpt",
                "--allow-no-resume",
                "--allow-duplicate",
            ],
            raw_dir=raw_dir,
            label=f"{phase}_submit_{task['index']}",
            timeout_s=600,
        )
        ids.append(_parse_submit_id(proc.stdout))
    _record_event(run_dir, "production_cpu_profile_submitted", phase=phase, ids=ids)
    return ids


def _record_observation_with_log_progress(
    *,
    run_dir: Path,
    raw_dir: Path,
    phase: str,
    stage: str,
    ids: list[str],
    sample_idx: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks, summary = _record_observation(
        run_dir=run_dir,
        raw_dir=raw_dir,
        phase=phase,
        stage=stage,
        ids=ids,
        sample_idx=sample_idx,
    )
    return tasks, _augment_summary_from_log_progress(tasks, summary)


def _augment_summary_from_log_progress(
    tasks: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(summary)
    progress_rows = []
    for task in tasks:
        progress = _latest_task_log_progress(task)
        if progress:
            progress_rows.append(progress)
    if not progress_rows:
        return out
    rates = [float(row["rate"]) for row in progress_rows if float(row.get("rate") or 0.0) > 0.0]
    units = sorted({str(row.get("unit") or "unit") for row in progress_rows if str(row.get("unit") or "")})
    current_units = [int(row["current"]) for row in progress_rows if int(row.get("current") or 0) > 0]
    out["log_progress_count"] = len(progress_rows)
    out["runtime_current_units"] = sorted(set(list(out.get("runtime_current_units") or []) + current_units))
    if rates and int(out.get("running_with_rate_count") or 0) < len(rates):
        out["running_with_rate_count"] = len(rates)
        out["rates_unit_s"] = rates
        out["rates_step_s"] = rates
        out["rate_units"] = units
        out["aggregate_active_rate_unit_s"] = sum(rates)
        out["mean_active_rate_unit_s"] = sum(rates) / float(len(rates))
        out["min_active_rate_unit_s"] = min(rates)
        out["max_active_rate_unit_s"] = max(rates)
        out["aggregate_active_rate_step_s"] = sum(rates)
        out["mean_active_rate_step_s"] = sum(rates) / float(len(rates))
        out["min_active_rate_step_s"] = min(rates)
        out["max_active_rate_step_s"] = max(rates)
        out["completed_log_progress_rate_fallback"] = True
    if int(out.get("running_count") or 0) < len(progress_rows):
        out["running_count"] = len(progress_rows)
        out["completed_log_progress_count_fallback"] = True
    return out


def _latest_task_log_progress(task: Mapping[str, Any]) -> dict[str, Any] | None:
    text = _task_log_tail(task, max_lines=240)
    if not text:
        return None
    latest_current = None
    latest_total = None
    latest_unit = None
    latest_rate = None
    for line in text.splitlines():
        match = _SCHEDULEURM_PROGRESS_RE.search(line)
        if not match:
            continue
        latest_unit = str(match.group(1) or "unit").lower()
        latest_current = int(match.group(2))
        latest_total = int(match.group(3)) if match.group(3) else latest_total
        if match.group(4):
            latest_rate = float(match.group(4))
            if match.group(5):
                latest_unit = str(match.group(5)).lower()
    if latest_current is None:
        return None
    return {
        "task_id": task.get("id"),
        "current": latest_current,
        "total": latest_total,
        "rate": latest_rate or 0.0,
        "unit": latest_unit or "unit",
    }


def _task_log_tail(task: Mapping[str, Any], *, max_lines: int = 200) -> str:
    node = str(task.get("node") or "")
    path = str(task.get("log_path") or "")
    if not path:
        return ""
    if _is_windows_node(node):
        rc, out, _err = _run_windows_ps(
            node,
            (
                f"$p={_ps_quote(path)}; "
                "if (Test-Path -LiteralPath $p) { "
                f"Get-Content -LiteralPath $p -Tail {int(max_lines)} "
                "}"
            ),
            timeout=15,
            check=False,
        )
        return out if rc == 0 else ""
    p = Path(path)
    if p.exists():
        return "\n".join(p.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:])
    return ""


def _wait_for_profile_progress_with_log_fallback(
    *,
    phase: str,
    run_dir: Path,
    ids: list[str],
    required: int,
    timeout_s: int,
    poll_s: int,
    min_runtime_unit: int = 1,
) -> None:
    raw_dir = run_dir / "raw"
    deadline = time.time() + max(1, int(timeout_s))
    attempt = 0
    while True:
        attempt += 1
        _status_refresh(raw_dir, f"{phase}_warmup_status_{attempt:03d}", ids)
        tasks, summary = _record_observation_with_log_progress(
            run_dir=run_dir,
            raw_dir=raw_dir,
            phase=phase,
            stage="warmup",
            ids=ids,
            sample_idx=attempt,
        )
        progressed = profile_progress_count(summary, min_runtime_unit=min_runtime_unit)
        _record_event(
            run_dir,
            "production_cpu_warmup",
            phase=phase,
            attempt=attempt,
            progressed=progressed,
            required=required,
            min_runtime_unit=max(1, int(min_runtime_unit)),
            summary=summary,
        )
        if progressed >= required:
            return
        queued_count = int(summary.get("queued_count") or 0)
        if progressed > 0 and queued_count > 0 and progressed < required and attempt >= 3:
            raise RuntimeError(
                f"{phase} capacity boundary: progressed={progressed}, "
                f"queued={queued_count}, required={required}, "
                f"statuses={summary.get('status_counts')}"
            )
        if int(summary.get("blocked_count") or 0) > 0 and progressed < required:
            raise RuntimeError(
                f"{phase} capacity boundary: blocked={summary.get('blocked')}, "
                f"progressed={progressed}, required={required}"
            )
        active = [t for t in tasks if t.get("status") in ("queued", "launching", "running")]
        if not active and progressed <= 0:
            raise RuntimeError(
                f"{phase} ended before measurable progress: "
                f"statuses={summary.get('status_counts')}"
            )
        if time.time() >= deadline:
            statuses = {str(t.get("id")): t.get("status") for t in tasks}
            raise RuntimeError(
                f"{phase} warmup timed out: progressed={progressed}, required={required}, "
                f"min_runtime_unit={max(1, int(min_runtime_unit))}, statuses={statuses}"
            )
        time.sleep(max(1, int(poll_s)))


def _measure_profile_with_log_fallback(
    *,
    phase: str,
    run_dir: Path,
    ids: list[str],
    measure_s: int,
    poll_s: int,
) -> dict[str, Any]:
    raw_dir = run_dir / "raw"
    end = time.time() + max(0, int(measure_s))
    sample_idx = 0
    sample_summaries: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    while True:
        sample_idx += 1
        _status_refresh(raw_dir, f"{phase}_measure_status_{sample_idx:03d}", ids)
        tasks, last_summary = _record_observation_with_log_progress(
            run_dir=run_dir,
            raw_dir=raw_dir,
            phase=phase,
            stage="measure",
            ids=ids,
            sample_idx=sample_idx,
        )
        sample_summary = dict(last_summary)
        sample_summary["measurement_sample_idx"] = sample_idx
        sample_summaries.append(sample_summary)
        _record_event(
            run_dir,
            "production_cpu_measure",
            phase=phase,
            sample_idx=sample_idx,
            remaining_s=max(0.0, end - time.time()),
            summary=last_summary,
        )
        active = [t for t in tasks if t.get("status") in ("queued", "launching", "running")]
        if time.time() >= end or not active:
            final_summary = select_measurement_summary(sample_summaries) or dict(last_summary)
            _write_json(run_dir / "reports" / f"{phase}_summary.json", final_summary)
            return final_summary
        time.sleep(min(max(1, int(poll_s)), max(0.0, end - time.time())))


def run_plan(plan: Mapping[str, Any], *, args: argparse.Namespace) -> int:
    run_id = str(plan["run_id"])
    manifest_path = init_run(run_id)
    run_dir = manifest_path.parent
    raw_dir = run_dir / "raw"
    _write_json(run_dir / "reports" / "runner_config.json", dict(vars(args)) | {"plan": dict(plan)})

    watcher = _watcher_status()
    _record_event(run_dir, "preflight_watcher_status", watcher=watcher)
    if watcher["active"] and not args.allow_active_watcher:
        raise SystemExit("scheduler watcher is active; stop it before clean production CPU curve validation")

    _mkdir_target(str(plan["node"]), str(plan["cwd"]), raw_dir)
    if not args.progress_wrapper:
        _deploy_progress_wrapper(str(plan["node"]), raw_dir)
    summaries: dict[int, dict[str, Any]] = {}
    all_ids: list[str] = []
    try:
        for row in plan.get("profiles") or []:
            profile = int(row["profile"])
            phase = str(row["phase"])
            tasks = list(row.get("tasks") or [])
            _record_event(run_dir, "production_cpu_profile_start", phase=phase, profile=profile)
            ids = _submit_profile_from_plan(run_dir=run_dir, phase=phase, tasks=tasks)
            all_ids.extend(ids)
            _record_observation_with_log_progress(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_submit", ids=ids)
            _scheduler_cmd(
                ["dispatch", "--algorithm", "legacy", "--hard-rule-mode", args.hard_rule_mode],
                raw_dir=raw_dir,
                label=f"{phase}_dispatch",
                timeout_s=1200,
            )
            _record_observation_with_log_progress(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_dispatch", ids=ids)
            try:
                _wait_for_profile_progress_with_log_fallback(
                    phase=phase,
                    run_dir=run_dir,
                    ids=ids,
                    required=len(ids),
                    timeout_s=args.warmup_timeout_s,
                    poll_s=args.poll_s,
                    min_runtime_unit=args.warmup_min_unit,
                )
            except RuntimeError as exc:
                if not args.stop_on_capacity_boundary:
                    raise
                _status_refresh(raw_dir, f"{phase}_capacity_boundary_status", ids)
                _tasks, summary = _record_observation_with_log_progress(
                    run_dir=run_dir,
                    raw_dir=raw_dir,
                    phase=phase,
                    stage="capacity_boundary",
                    ids=ids,
                )
                summary = _annotate_summary(summary, expected=len(ids))
                summary["capacity_boundary"] = True
                summary["boundary_reasons"] = [str(exc)]
                _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
                summaries[profile] = summary
                _cancel_all(raw_dir, phase, ids)
                break
            summary = _measure_profile_with_log_fallback(
                phase=phase,
                run_dir=run_dir,
                ids=ids,
                measure_s=args.measure_s,
                poll_s=args.poll_s,
            )
            summary = _annotate_summary(summary, expected=len(ids))
            _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
            summaries[profile] = summary
            _cancel_all(raw_dir, phase, ids)
            _status_refresh(raw_dir, f"{phase}_post_cancel_status", ids)
            _record_observation_with_log_progress(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_cancel", ids=ids)

        verdict = _build_verdict(
            run_id=run_id,
            node=str(plan["node"]),
            steps=int(plan["total_units"]),
            mode=str(plan["sub_bucket"]),
            summaries=summaries,
        )
        _write_json(run_dir / "reports" / "service_curve_verdict.json", verdict)
        _write_markdown(run_dir / "reports" / "service_curve.md", verdict)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["pass"] else 2
    finally:
        active = [
            str(t.get("id"))
            for t in _tasks_by_ids(all_ids)
            if t.get("status") in ("queued", "launching", "running")
        ]
        if active:
            _record_event(run_dir, "final_cleanup_start", active_ids=active)
            _cancel_all(raw_dir, "final_cleanup", active)
            _record_event(run_dir, "final_cleanup_done", active_ids=active)
        time.sleep(0.1)


def _build_plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cmd_template = _read_template(args)
    if not cmd_template:
        raise SystemExit("--cmd-template or --cmd-template-file is required")
    run_id = args.run_id or f"module54_production_cpu_curve_{_now_stamp()}"
    profiles = [int(x) for x in str(args.profiles).split(",") if x.strip()]
    if not profiles:
        raise SystemExit("--profiles must be nonempty")
    return build_submission_plan(
        run_id=run_id,
        sub_bucket=args.sub_bucket,
        profiles=profiles,
        cmd_template=cmd_template,
        node=args.node,
        cwd=args.cwd,
        output_root=args.output_root,
        seed_base=args.seed_base,
        total_units=args.total_units,
        work_items_per_task=args.work_items_per_task,
        cpu_cores=args.cpu,
        ram_mb=args.ram_mb,
        project=args.project,
        signature_prefix=args.signature_prefix,
        progress_unit=args.progress_unit,
        progress_wrapper=args.progress_wrapper or _default_progress_wrapper_for_node(
            args.node,
            args.progress_wrapper_kind,
        ),
        progress_wrapper_kind=args.progress_wrapper_kind,
        progress_csv_glob=args.progress_csv_glob,
        progress_poll_s=args.progress_poll_s,
        child_shell=_child_shell_for_node(args.node, args.child_shell),
    )


def _is_windows_node(node: str) -> bool:
    return str(node or "") in WINDOWS_NODES


def _default_progress_wrapper_for_node(node: str, kind: str) -> str:
    csv = str(kind or "").strip().lower() == "csv"
    if _is_windows_node(node):
        return WINDOWS_CSV_PROGRESS_WRAPPER if csv else WINDOWS_PROGRESS_WRAPPER
    return REMOTE_CSV_PROGRESS_WRAPPER if csv else REMOTE_PROGRESS_WRAPPER


def _child_shell_for_node(node: str, requested: str) -> str:
    raw = str(requested or "auto").strip().lower()
    if raw in ("bash", "argv"):
        return raw
    if raw != "auto":
        raise ValueError(f"unknown child shell mode: {requested!r}")
    return "argv" if _is_windows_node(node) else "bash"


def _windows_path_for_node(node: str, path: str) -> str:
    if not _is_windows_node(node) or not path:
        return path
    text = str(path)
    if len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        return text.replace("/", "\\")
    prefixes = [
        str(Path.home() / "mine_code"),
        str(Path.home()),
        "/home/erzhu419/mine_code",
        "/home/erzhu419",
    ]
    norm = str(Path(text).expanduser()) if text.startswith("~") else text
    for pref in prefixes:
        pref = str(Path(pref))
        if norm == pref or norm.startswith(pref.rstrip("/") + "/"):
            rel = norm[len(pref):].lstrip("/")
            parts = [p for p in rel.split("/") if p]
            return r"F:\erzhu419_smoke" + ("\\" + "\\".join(parts) if parts else "")
    return text.replace("/", "\\")


def _ssh_base_args(node: str) -> list[str]:
    info = WINDOWS_NODES[node]
    args = [
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "BatchMode=yes",
        "-i",
        str(info["identity"]),
        "-p",
        str(info["port"]),
        f"{info['user']}@{info['host']}",
    ]
    return args


def _ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_windows_ps(
    node: str,
    ps_script: str,
    *,
    timeout: int = 30,
    check: bool = True,
    input_data: bytes | None = None,
) -> tuple[int, str, str]:
    encoded = base64.b64encode((ps_script or "").encode("utf-16le")).decode("ascii")
    proc = subprocess.run(
        _ssh_base_args(node) + [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        input=input_data,
        capture_output=True,
        timeout=timeout,
    )
    stdout = (proc.stdout or b"").decode("utf-8", "replace")
    stderr = (proc.stderr or b"").decode("utf-8", "replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"[{node}] powershell failed: {(stderr or stdout).strip()[:240]}")
    return int(proc.returncode), stdout, stderr


def _deploy_windows_progress_package(node: str, raw_dir: Path) -> str:
    dest = WINDOWS_PROGRESS_WRAPPER_ROOT
    ps = (
        f"$dest={_ps_quote(dest)}; "
        "[IO.Directory]::CreateDirectory($dest) | Out-Null; "
        "tar -xf - -C $dest; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
        "Write-Output 'READY'"
    )
    encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    files = [
        "algorithm/__init__.py",
        "algorithm/experiments/__init__.py",
        "algorithm/experiments/progress_wrapper.py",
        "algorithm/experiments/progress_units.py",
        "algorithm/experiments/csv_progress_wrapper.py",
    ]
    tar_proc = subprocess.Popen(
        ["tar", "-h", "-C", str(_repo_root()), "-cf", "-", *files],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ssh_proc = subprocess.run(
        _ssh_base_args(node) + [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        stdin=tar_proc.stdout,
        capture_output=True,
        timeout=120,
    )
    if tar_proc.stdout:
        tar_proc.stdout.close()
    _out, tar_err = tar_proc.communicate(timeout=30)
    tar_rc = int(tar_proc.returncode or 0)
    out = (ssh_proc.stdout or b"").decode("utf-8", "replace")
    err = (ssh_proc.stderr or b"").decode("utf-8", "replace")
    _write_json(
        raw_dir / f"deploy_windows_progress_wrapper_{node}.json",
        {
            "node": node,
            "dest": dest,
            "tar_rc": tar_rc,
            "ssh_rc": int(ssh_proc.returncode),
            "stdout": out[-1000:],
            "stderr": err[-1000:],
            "tar_stderr": (tar_err or b"").decode("utf-8", "replace")[-1000:],
        },
    )
    if tar_rc != 0 or ssh_proc.returncode != 0 or "READY" not in out:
        raise RuntimeError(
            f"Windows progress wrapper deploy failed on {node}: "
            f"{(err or out).strip()[:200]}"
        )
    return WINDOWS_PROGRESS_WRAPPER


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sub-bucket", required=True)
    parser.add_argument("--node", default="local")
    parser.add_argument("--profiles", default="1,2,4,8")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--cmd-template", default="")
    parser.add_argument("--cmd-template-file", default="")
    parser.add_argument("--output-root", default="scheduleurm_production_cpu_curve_runs")
    parser.add_argument("--seed-base", type=int, default=100000)
    parser.add_argument("--total-units", type=int, default=100)
    parser.add_argument("--work-items-per-task", type=int, default=1)
    parser.add_argument("--progress-unit", default="episode")
    parser.add_argument("--progress-wrapper", default="")
    parser.add_argument("--progress-wrapper-kind", choices=["line", "csv"], default="line")
    parser.add_argument("--progress-csv-glob", default="")
    parser.add_argument("--progress-poll-s", type=float, default=5.0)
    parser.add_argument("--child-shell", choices=["auto", "bash", "argv"], default="auto")
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--project", default="ScheduleurmBench")
    parser.add_argument("--signature-prefix", default="ScheduleurmBench/production_cpu_workload_curve")
    parser.add_argument("--warmup-timeout-s", type=int, default=600)
    parser.add_argument("--warmup-min-unit", type=int, default=1)
    parser.add_argument("--measure-s", type=int, default=120)
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--hard-rule-mode", default="clean_bench")
    parser.add_argument("--allow-active-watcher", action="store_true")
    parser.add_argument("--stop-on-capacity-boundary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", default="")
    parser.add_argument("--plan-markdown-output", default="")
    args = parser.parse_args()

    plan = _build_plan_from_args(args)
    if args.plan_output:
        _write_json(Path(args.plan_output), plan)
    if args.plan_markdown_output:
        path = Path(args.plan_markdown_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_plan(plan), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return run_plan(plan, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
