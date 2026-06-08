"""CPU-only production workload service-curve runner.

This is the runner intended for Module53 sub-buckets such as
``freqduet_cpu_ablation|c_17_32`` and ``sumo_eval_cpu|c_le2``.  It mirrors the
existing service-curve runners but does not request or pin GPUs.

The module also exposes a dry-run plan builder so probe batches can be reviewed
before any Scheduleurm tasks are submitted.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import time
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping

from .cpu_service_curve_validation import _annotate_summary, _build_verdict, _phase_name, _write_markdown
from .service_curve_validation import _measure_profile, _wait_for_profile_progress
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
) -> dict[str, Any]:
    """Render a CPU production workload command template.

    Supported placeholders are explicit so typos fail before submission.
    """

    seed = int(seed_base) + int(index)
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
        "total_units": int(total_units),
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
    cpu_cores: int,
    ram_mb: int,
    project: str,
    signature_prefix: str,
    progress_unit: str,
    progress_wrapper: str = REMOTE_PROGRESS_WRAPPER,
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
            )
            wrapped = _wrap_with_progress(
                rendered["cmd"],
                progress_wrapper=progress_wrapper,
                progress_unit=progress_unit,
                total_units=total_units,
            )
            tasks.append({
                "index": index,
                "seed": rendered["seed"],
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
        "cpu_cores": int(cpu_cores),
        "ram_mb": int(ram_mb),
        "project": project,
        "signature_prefix": signature_prefix,
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
    progress_unit: str,
    total_units: int,
) -> str:
    return (
        f"python3 -u {shlex.quote(progress_wrapper)} "
        f"--unit {shlex.quote(progress_unit)} --total {int(total_units)} -- "
        f"bash -lc {shlex.quote(command)}"
    )


def _read_template(args: argparse.Namespace) -> str:
    if args.cmd_template_file:
        return Path(args.cmd_template_file).read_text(encoding="utf-8").strip()
    return str(args.cmd_template or "").strip()


def _deploy_progress_wrapper(node: str, raw_dir: Path) -> str:
    local_dir = _repo_root() / "algorithm" / "experiments"
    if node == "local":
        target_root = Path("/tmp/scheduleurm_progress_wrapper_pkg")
        target_dir = target_root / "algorithm" / "experiments"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_root / "algorithm" / "__init__.py").touch()
        (target_dir / "__init__.py").touch()
        for name in ("progress_wrapper.py", "progress_units.py"):
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
            f"{node}:{remote_dir}/",
        ],
        cwd=_repo_root(),
        raw_dir=raw_dir,
        label="deploy_cpu_progress_wrapper_scp",
    )
    return REMOTE_PROGRESS_WRAPPER


def _mkdir_target(node: str, cwd: str, raw_dir: Path) -> None:
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
            _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_submit", ids=ids)
            _scheduler_cmd(
                ["dispatch", "--algorithm", "legacy", "--hard-rule-mode", args.hard_rule_mode],
                raw_dir=raw_dir,
                label=f"{phase}_dispatch",
                timeout_s=1200,
            )
            _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_dispatch", ids=ids)
            try:
                _wait_for_profile_progress(
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
                _tasks, summary = _record_observation(
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
            summary = _measure_profile(
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
            _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_cancel", ids=ids)

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
        cpu_cores=args.cpu,
        ram_mb=args.ram_mb,
        project=args.project,
        signature_prefix=args.signature_prefix,
        progress_unit=args.progress_unit,
        progress_wrapper=args.progress_wrapper or REMOTE_PROGRESS_WRAPPER,
    )


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
    parser.add_argument("--progress-unit", default="episode")
    parser.add_argument("--progress-wrapper", default="")
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
