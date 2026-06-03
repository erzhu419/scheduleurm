"""Fixed-profile service-curve runner for real workload commands.

This is the production-workload counterpart of ``service_curve_validation``.
It keeps Scheduleurm's scheduler code unchanged: tasks are submitted through the
existing CLI, pinned with ``edit --require-gpu``, dispatched, measured, and then
cancelled after a stable progress window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import Formatter
from typing import Any

from .service_curve_validation import (
    annotate_expected_placement,
    _gpu_plan,
    _measure_profile,
    _profile_phase_name,
    _wait_for_profile_progress,
    _write_curve_markdown,
    build_service_curve_verdict,
)
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


def _template_fields(template: str) -> set[str]:
    return {
        str(field)
        for _, field, _, _ in Formatter().parse(template or "")
        if field
    }


def render_workload_command(
    template: str,
    *,
    run_id: str,
    phase: str,
    count_per_gpu: int,
    index: int,
    gpu_idx: int,
    seed_base: int,
    max_iters: int,
    output_root: str,
    node: str,
) -> dict[str, Any]:
    """Render a real-workload command template plus its run identity.

    Supported placeholders are intentionally explicit.  A typo should fail
    before tasks are submitted rather than silently producing duplicate jobs.
    """

    run_name = f"{run_id}_{phase}_gpu{gpu_idx}_{index}"
    seed = int(seed_base) + int(index)
    values: dict[str, Any] = {
        "run_id": run_id,
        "phase": phase,
        "profile": count_per_gpu,
        "count_per_gpu": count_per_gpu,
        "index": index,
        "gpu_idx": gpu_idx,
        "seed": seed,
        "max_iters": int(max_iters),
        "output_root": output_root.rstrip("/"),
        "run_name": run_name,
        "node": node,
    }
    unknown = sorted(_template_fields(template) - set(values))
    if unknown:
        raise ValueError(f"unknown command-template placeholders: {unknown}")
    return {
        "cmd": template.format(**values),
        "run_name": run_name,
        "seed": seed,
        "values": values,
    }


def _submit_profile(
    *,
    run_id: str,
    phase: str,
    run_dir: Path,
    node: str,
    gpu_plan: list[int],
    cwd: str,
    cmd_template: str,
    output_root: str,
    seed_base: int,
    max_iters: int,
    vram_mb: int,
    ram_mb: int,
    cpu: int,
    project: str,
    signature_prefix: str,
    description_prefix: str,
) -> list[str]:
    raw_dir = run_dir / "raw"
    count_per_gpu = max(0, len(gpu_plan) // max(1, len(set(gpu_plan))))
    ids: list[str] = []
    for i, gpu_idx in enumerate(gpu_plan):
        rendered = render_workload_command(
            cmd_template,
            run_id=run_id,
            phase=phase,
            count_per_gpu=count_per_gpu,
            index=i,
            gpu_idx=gpu_idx,
            seed_base=seed_base,
            max_iters=max_iters,
            output_root=output_root,
            node=node,
        )
        proc = _scheduler_cmd(
            [
                "submit",
                "--description",
                f"{description_prefix} {phase} task {i}",
                "--cmd",
                rendered["cmd"],
                "--cwd",
                cwd,
                "--signature",
                f"{signature_prefix}/{run_id}/{phase}/{i}",
                "--vram",
                str(vram_mb),
                "--ram-mb",
                str(ram_mb),
                "--cpu",
                str(cpu),
                "--priority",
                "high",
                "--project",
                project,
                "--require-node",
                node,
                "--allow-no-ckpt",
                "--allow-no-resume",
                "--allow-duplicate",
            ],
            raw_dir=raw_dir,
            label=f"{phase}_submit_{i}",
            timeout_s=600,
        )
        tid = _parse_submit_id(proc.stdout)
        ids.append(tid)
        _scheduler_cmd(
            [
                "edit",
                tid,
                "--require-gpu",
                str(gpu_idx),
                "--allow-gpu-over-one-third",
            ],
            raw_dir=raw_dir,
            label=f"{phase}_pin_gpu_{gpu_idx}_{i}",
            timeout_s=120,
        )
        _record_event(
            run_dir,
            "workload_task_submitted",
            phase=phase,
            task_id=tid,
            gpu_idx=gpu_idx,
            run_name=rendered["run_name"],
            seed=rendered["seed"],
        )
    _record_event(run_dir, "workload_profile_submitted", phase=phase, ids=ids, gpu_plan=gpu_plan)
    return ids


def _read_template(args: argparse.Namespace) -> str:
    if args.cmd_template_file:
        return Path(args.cmd_template_file).read_text(encoding="utf-8").strip()
    return str(args.cmd_template or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--node", default="jtl110gpu2")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--profiles", default="1,2,3,4,5")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--cmd-template", default="")
    parser.add_argument("--cmd-template-file", default="")
    parser.add_argument("--output-root", default="scheduleurm_service_curve_runs")
    parser.add_argument("--project", default="ScheduleurmBench")
    parser.add_argument("--signature-prefix", default="ScheduleurmBench/workload_service_curve")
    parser.add_argument("--description-prefix", default="Scheduleurm workload service curve")
    parser.add_argument("--vram-mb", type=int, default=1024)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=100000)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--total-units", type=int, default=0)
    parser.add_argument("--warmup-timeout-s", type=int, default=600)
    parser.add_argument("--measure-s", type=int, default=120)
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--hard-rule-mode", default="clean_bench")
    parser.add_argument("--expected-sweetspot-count", type=int, default=0)
    parser.add_argument("--total-jobs-for-proxy", type=int, default=0)
    parser.add_argument("--proxy-job-counts", default="")
    parser.add_argument("--min-two-vs-three-gain", type=float, default=0.0)
    parser.add_argument("--allow-active-watcher", action="store_true")
    parser.add_argument(
        "--stop-on-capacity-boundary",
        action="store_true",
        help="Record a blocked/partial profile as the saturation boundary instead of timing out the run.",
    )
    args = parser.parse_args()

    cmd_template = _read_template(args)
    if not cmd_template:
        raise SystemExit("--cmd-template or --cmd-template-file is required")

    run_id = args.run_id or f"module8_workload_curve_{_now_stamp()}"
    manifest_path = init_run(run_id)
    run_dir = manifest_path.parent
    raw_dir = run_dir / "raw"
    gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()]
    profiles = [int(x) for x in str(args.profiles).split(",") if x.strip()]
    if not gpus or not profiles:
        raise SystemExit("--gpus and --profiles must be nonempty")

    total_units = int(args.total_units or args.max_iters)
    total_jobs_for_proxy = int(args.total_jobs_for_proxy or max(profiles) * len(gpus))
    proxy_job_counts = [int(x) for x in str(args.proxy_job_counts).split(",") if x.strip()]
    if not proxy_job_counts:
        proxy_job_counts = [total_jobs_for_proxy, total_jobs_for_proxy * 2, total_jobs_for_proxy * 4]

    _write_json(
        run_dir / "reports" / "runner_config.json",
        {k: getattr(args, k) for k in sorted(vars(args))} | {
            "run_id": run_id,
            "cmd_template": cmd_template,
            "gpus_list": gpus,
            "profiles_list": profiles,
            "total_units_effective": total_units,
            "total_jobs_for_proxy_effective": total_jobs_for_proxy,
            "proxy_job_counts_list": proxy_job_counts,
        },
    )

    watcher = _watcher_status()
    _record_event(run_dir, "preflight_watcher_status", watcher=watcher)
    if watcher["active"] and not args.allow_active_watcher:
        raise SystemExit("scheduler watcher is active; stop it before clean workload service-curve validation")

    _run_cmd(["mkdir", "-p", args.cwd], cwd=_repo_root(), raw_dir=raw_dir, label="local_mkdir_cwd")
    _run_cmd(["ssh", args.node, "mkdir", "-p", args.cwd], cwd=_repo_root(), raw_dir=raw_dir, label="remote_mkdir_cwd")

    summaries: dict[int, dict[str, Any]] = {}
    all_ids: list[str] = []
    try:
        for count in profiles:
            phase = _profile_phase_name(count)
            plan = _gpu_plan(gpus, count)
            _record_event(run_dir, "workload_profile_start", phase=phase, count_per_gpu=count, gpu_plan=plan)
            ids = _submit_profile(
                run_id=run_id,
                phase=phase,
                run_dir=run_dir,
                node=args.node,
                gpu_plan=plan,
                cwd=args.cwd,
                cmd_template=cmd_template,
                output_root=args.output_root,
                seed_base=args.seed_base,
                max_iters=args.max_iters,
                vram_mb=args.vram_mb,
                ram_mb=args.ram_mb,
                cpu=args.cpu,
                project=args.project,
                signature_prefix=args.signature_prefix,
                description_prefix=args.description_prefix,
            )
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
                )
            except RuntimeError as exc:
                if not args.stop_on_capacity_boundary:
                    raise
                _status_refresh(raw_dir, f"{phase}_capacity_boundary_status")
                _tasks, summary = _record_observation(
                    run_dir=run_dir,
                    raw_dir=raw_dir,
                    phase=phase,
                    stage="capacity_boundary",
                    ids=ids,
                )
                summary = annotate_expected_placement(summary, plan)
                summary["capacity_boundary"] = True
                summary["boundary_reasons"] = [str(exc)]
                _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
                summaries[count] = summary
                _record_event(
                    run_dir,
                    "workload_profile_capacity_boundary",
                    phase=phase,
                    count_per_gpu=count,
                    summary=summary,
                    reason=str(exc),
                )
                _cancel_all(raw_dir, phase, ids)
                _status_refresh(raw_dir, f"{phase}_post_cancel_status")
                _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_cancel", ids=ids)
                break
            else:
                summary = _measure_profile(
                    phase=phase,
                    run_dir=run_dir,
                    ids=ids,
                    measure_s=args.measure_s,
                    poll_s=args.poll_s,
                )
                summary = annotate_expected_placement(summary, plan)
                _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
                summaries[count] = summary
                _cancel_all(raw_dir, phase, ids)
                _status_refresh(raw_dir, f"{phase}_post_cancel_status")
                _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_cancel", ids=ids)
                _record_event(run_dir, "workload_profile_done", phase=phase, count_per_gpu=count, summary=summary)

        verdict = build_service_curve_verdict(
            run_id=run_id,
            node=args.node,
            steps=total_units,
            total_jobs_for_proxy=total_jobs_for_proxy,
            proxy_job_counts=proxy_job_counts,
            expected_sweetspot_count=args.expected_sweetspot_count,
            min_two_vs_three_gain=args.min_two_vs_three_gain,
            summaries=summaries,
        )
        _write_json(run_dir / "reports" / "service_curve_verdict.json", verdict)
        _write_curve_markdown(run_dir / "reports" / "service_curve.md", verdict)
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


if __name__ == "__main__":
    raise SystemExit(main())
