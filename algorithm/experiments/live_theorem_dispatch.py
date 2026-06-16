"""Run a controlled live dispatch through the theorem MaxWeight hook.

This module is deliberately outside ``skill/scheduler.py``.  It submits a small
gated workload, releases the gate only for a one-shot theorem-policy dispatch,
and then audits the emitted candidate-family trace against realized scheduler
records.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from algorithm.oracle_trace import audit_scheduler_score_trace, load_trace_slots
from algorithm.experiments.live_trace_realization_bridge import build_realization_bridge
from algorithm.experiments.theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = REPO_ROOT / "skill" / "scheduler.py"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def run_live_theorem_dispatch(
    *,
    run_id: str,
    require_node: str = "local",
    preferred_node: str = "",
    allowed_nodes: Sequence[str] | None = None,
    steps: int = 24,
    size: int = 512,
    timeout_s: int = 600,
    task_count: int = 2,
    max_gpu_util_pct: int | None = None,
    max_tasks_per_gpu: int = 9,
    max_dispatch_passes: int = 2,
    python_bin: str = "python3",
) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = _safe_run_id(run_id)
    trace_path = ARTIFACT_ROOT / f"{run_id}_trace.jsonl"
    report_path = ARTIFACT_ROOT / f"{run_id}.json"
    realization_path = ARTIFACT_ROOT / f"{run_id}_realization.json"
    markdown_path = REPO_ROOT / "md" / f"{run_id}.md"
    gate_path = ARTIFACT_ROOT / f"{run_id}.ready"
    trace_path.unlink(missing_ok=True)
    gate_path.unlink(missing_ok=True)

    task_ids = []
    submit_rows = []
    for idx, workload_key in enumerate(_workload_mix(task_count)):
        row = _submit_task(
            run_id=run_id,
            idx=idx,
            workload_key=workload_key,
            gate_path=gate_path,
            require_node=require_node,
            preferred_node=preferred_node,
            allowed_nodes=list(allowed_nodes or []),
            steps=steps,
            size=size,
            python_bin=python_bin,
        )
        task_ids.append(row["task_id"])
        submit_rows.append(row)

    gate_path.write_text(f"ready {time.time()}\n", encoding="utf-8")
    dispatch_env = {
        "SCHEDULEURM_ORACLE_TRACE_PATH": str(trace_path),
        "SCHEDULEURM_THEOREM_UNCERTIFIED_MODE": "block",
        "SCHEDULEURM_THEOREM_QUEUE_TTL_S": "0",
        "SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU": str(int(max_tasks_per_gpu)),
        "SCHEDULEURM_ALGO_MAX_POST_VRAM_FRAC": "0.95",
        "SCHEDULEURM_ALGO_GPU_SWEET_SPOT_TASKS": "3",
        "SCHEDULEURM_THEOREM_PROFILE_PENALTY": "0",
    }
    if max_gpu_util_pct is not None:
        dispatch_env["SCHEDULEURM_ALGO_MAX_GPU_UTIL_PCT"] = str(int(max_gpu_util_pct))
    dispatches = []
    for pass_idx in range(max(1, int(max_dispatch_passes or 1))):
        dispatch = _run(
            [
                "python3",
                str(SCHEDULER),
                "dispatch",
                "--algorithm",
                "theorem_maxweight_v1",
                "--hard-rule-mode",
                "clean_bench",
                *[part for tid in task_ids for part in ("--task-id", tid)],
            ],
            env=dispatch_env,
            check=False,
            timeout=timeout_s,
        )
        statuses = _task_statuses(task_ids)
        dispatches.append({
            "pass_index": pass_idx + 1,
            "returncode": dispatch.returncode,
            "stdout_tail": _tail(dispatch.stdout),
            "stderr_tail": _tail(dispatch.stderr),
            "task_statuses": statuses,
        })
        if dispatch.returncode != 0:
            break
        queued = [tid for tid, status in statuses.items() if status == "queued"]
        if not queued:
            break
        if pass_idx + 1 < max(1, int(max_dispatch_passes or 1)):
            time.sleep(5)
    wait = _run(
        [
            "python3",
            str(SCHEDULER),
            "wait-for",
            "--task-id",
            *task_ids,
            "--poll",
            "5",
            "--timeout",
            str(max(30, int(timeout_s))),
        ],
        check=False,
        timeout=max(45, int(timeout_s) + 30),
    )

    slots = load_trace_slots(trace_path)
    scheduler_audit = audit_scheduler_score_trace(slots)
    theorem_bridge = build_theorem_oracle_audit_from_trace(slots)
    realization = build_realization_bridge(trace_path=trace_path)
    _write_json(realization_path, realization)
    report = {
        "gate": "controlled_live_theorem_dispatch",
        "run_id": run_id,
        "scope": (
            "controlled local live dispatch smoke using scheduler.py dispatch "
            "--algorithm theorem_maxweight_v1. The scheduler queue is mutated, "
            "tasks are actually launched, candidate-family traces are emitted, "
            "and realization is matched against queue/archive records. This is "
            "not yet the full global q01/q11 production experiment."
        ),
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "realization_path": str(realization_path),
        "markdown_path": str(markdown_path),
        "require_node": require_node,
        "preferred_node": preferred_node,
        "allowed_nodes": list(allowed_nodes or []),
        "max_gpu_util_pct": max_gpu_util_pct,
        "max_tasks_per_gpu": int(max_tasks_per_gpu),
        "max_dispatch_passes": max_dispatch_passes,
        "python_bin": python_bin,
        "task_ids": task_ids,
        "submitted": submit_rows,
        "dispatches": dispatches,
        "dispatch_returncode": max([int(d.get("returncode") or 0) for d in dispatches] or [1]),
        "dispatch_stdout_tail": "\n".join(
            f"--- dispatch pass {d.get('pass_index')} ---\n{d.get('stdout_tail') or ''}"
            for d in dispatches
        )[-12000:],
        "dispatch_stderr_tail": "\n".join(
            f"--- dispatch pass {d.get('pass_index')} ---\n{d.get('stderr_tail') or ''}"
            for d in dispatches
        )[-12000:],
        "wait_returncode": wait.returncode,
        "wait_stdout_tail": _tail(wait.stdout),
        "wait_stderr_tail": _tail(wait.stderr),
        "trace_slot_count": len(slots),
        "candidate_count_total": sum(len(slot.get("candidates") or []) for slot in slots),
        "theorem_slot_count": sum(
            1 for slot in slots
            if slot.get("score_semantics") == "robust_maxweight_lower_service"
        ),
        "scheduler_score_audit": scheduler_audit,
        "theorem_oracle_bridge": theorem_bridge,
        "realization": realization,
        "pass": (
            dispatch.returncode == 0
            and all(int(d.get("returncode") or 0) == 0 for d in dispatches)
            and wait.returncode == 0
            and bool(scheduler_audit.get("usable_for_theorem"))
            and bool(theorem_bridge.get("usable_for_theorem"))
            and bool(realization.get("usable_for_live_progress_claim"))
        ),
    }
    _write_json(report_path, report)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    bridge = report.get("theorem_oracle_bridge") or {}
    realization = report.get("realization") or {}
    lines = [
        "# Controlled Live Theorem Dispatch",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| run_id | `{report.get('run_id')}` |",
        f"| require_node | `{report.get('require_node')}` |",
        f"| preferred_node | `{report.get('preferred_node')}` |",
        f"| allowed_nodes | `{','.join(report.get('allowed_nodes') or [])}` |",
        f"| max_gpu_util_pct | {report.get('max_gpu_util_pct')} |",
        f"| max_tasks_per_gpu | {report.get('max_tasks_per_gpu')} |",
        f"| dispatch_pass_count | {len(report.get('dispatches') or [])} |",
        f"| task_count | {len(report.get('task_ids') or [])} |",
        f"| trace_slot_count | {report.get('trace_slot_count', 0)} |",
        f"| theorem_slot_count | {report.get('theorem_slot_count', 0)} |",
        f"| candidate_count_total | {report.get('candidate_count_total', 0)} |",
        f"| dispatch_returncode | {report.get('dispatch_returncode')} |",
        f"| wait_returncode | {report.get('wait_returncode')} |",
        f"| alpha0 | {_fmt(bridge.get('alpha0'))} |",
        f"| alpha1 | {_fmt(bridge.get('alpha1'))} |",
        f"| realization_status | `{realization.get('status')}` |",
        f"| live_completion_claim | {str(bool(realization.get('usable_for_live_completion_claim'))).lower()} |",
        f"| live_progress_claim | {str(bool(realization.get('usable_for_live_progress_claim'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Submitted Tasks",
        "",
        "| task | workload | signature |",
        "|---|---|---|",
    ]
    for row in report.get("submitted") or []:
        lines.append(
            f"| `{row.get('task_id')}` | `{row.get('workload_key')}` | "
            f"`{row.get('signature')}` |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- trace: `{report.get('trace_path')}`",
        f"- report: `{report.get('report_path')}`",
        f"- realization: `{report.get('realization_path')}`",
        "",
    ])
    return "\n".join(lines)


def _submit_task(
    *,
    run_id: str,
    idx: int,
    workload_key: str,
    gate_path: Path,
    require_node: str,
    preferred_node: str,
    allowed_nodes: Sequence[str],
    steps: int,
    size: int,
    python_bin: str,
) -> dict[str, Any]:
    label = f"{run_id}-{idx:02d}-{workload_key}"
    signature_token = (
        "jax_matmul_size8192"
        if workload_key == "gpu_heavy_jax_matmul"
        else "resac_ant_real"
    )
    description = (
        "q01 GPU-heavy JAX matmul controlled live theorem dispatch"
        if workload_key == "gpu_heavy_jax_matmul"
        else "q11 RE-SAC/BAPR-like hybrid RL controlled live theorem dispatch"
    )
    cmd = (
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 "
        f"{shlex.quote(python_bin)} -u algorithm/experiments/gpu_progress_benchmark.py "
        f"--steps {int(steps)} --size {int(size)} --label {label} && echo DONE"
    )
    signature = f"ScheduleurmBench/{signature_token}/controlled_live_theorem/{run_id}/{idx:02d}"
    argv = [
        "python3",
        str(SCHEDULER),
        "submit",
        "--description",
        description,
        "--cmd",
        cmd,
        "--cwd",
        str(REPO_ROOT),
        "--signature",
        signature,
        "--project",
        "ScheduleurmBench",
        "--vram",
        "512",
        "--ram-mb",
        "768",
        "--cpu",
        "1",
        "--priority",
        "high",
        "--wait-for-file",
        str(gate_path),
        "--allow-duplicate",
    ]
    if require_node:
        argv.extend(["--require-node", require_node])
    if preferred_node:
        argv.extend(["--preferred-node", preferred_node])
    for node in allowed_nodes:
        argv.extend(["--allowed-node", str(node)])
    completed = _run(
        argv,
        check=True,
        timeout=120,
    )
    task_id = _parse_submitted_task_id(completed.stdout)
    return {
        "task_id": task_id,
        "workload_key": workload_key,
        "signature": signature,
        "description": description,
        "stdout": completed.stdout,
    }


def _workload_mix(task_count: int) -> list[str]:
    base = ["gpu_heavy_jax_matmul", "hybrid_rl_resac_ant"]
    count = max(1, int(task_count))
    return [base[i % len(base)] for i in range(count)]


def _task_statuses(task_ids: Sequence[str]) -> dict[str, str]:
    wanted = {str(tid) for tid in task_ids}
    if not wanted:
        return {}
    completed = _run(
        ["python3", str(SCHEDULER), "show", *([] if len(wanted) != 1 else [next(iter(wanted))])],
        check=False,
        timeout=30,
    )
    if len(wanted) == 1 and completed.returncode == 0:
        try:
            row = json.loads(completed.stdout)
            return {str(row.get("id") or next(iter(wanted))): str(row.get("status") or "")}
        except json.JSONDecodeError:
            pass
    queue_path = Path.home() / ".claude" / "scheduler" / "queue.json"
    try:
        state = json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        return {tid: "unknown" for tid in wanted}
    out = {}
    for task in state.get("tasks", []):
        tid = str(task.get("id") or "")
        if tid in wanted:
            out[tid] = str(task.get("status") or "")
    for tid in wanted:
        out.setdefault(tid, "missing")
    return out


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    check: bool,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    completed = subprocess.run(
        list(argv),
        cwd=str(REPO_ROOT),
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return completed


def _parse_submitted_task_id(stdout: str) -> str:
    match = re.search(r"\bsubmitted\s+(t\d+)\b", stdout or "")
    if not match:
        raise RuntimeError(f"could not parse submitted task id from stdout: {stdout!r}")
    return match.group(1)


def _safe_run_id(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        text = "live_theorem_dispatch_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or "live_theorem_dispatch"


def _tail(text: str, max_lines: int = 80) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-max(1, int(max_lines)):])


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return "NA"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.live_theorem_dispatch"
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--require-node", default="local")
    parser.add_argument("--no-require-node", action="store_true")
    parser.add_argument("--preferred-node", default="")
    parser.add_argument(
        "--allowed-node",
        dest="allowed_nodes",
        action="append",
        default=[],
        help=(
            "Restrict scheduler candidate family to this node while leaving "
            "the final node/GPU choice to theorem_maxweight_v1. Repeatable."
        ),
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--task-count", type=int, default=2)
    parser.add_argument("--max-gpu-util-pct", type=int, default=None)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=9)
    parser.add_argument("--max-dispatch-passes", type=int, default=2)
    parser.add_argument("--python-bin", default=os.environ.get("SCHEDULEURM_LIVE_DISPATCH_PYTHON", "python3"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_node = "" if args.no_require_node else args.require_node
    report = run_live_theorem_dispatch(
        run_id=args.run_id,
        require_node=require_node,
        preferred_node=args.preferred_node,
        allowed_nodes=args.allowed_nodes,
        steps=args.steps,
        size=args.size,
        timeout_s=args.timeout_s,
        task_count=args.task_count,
        max_gpu_util_pct=args.max_gpu_util_pct,
        max_tasks_per_gpu=args.max_tasks_per_gpu,
        max_dispatch_passes=args.max_dispatch_passes,
        python_bin=args.python_bin,
    )
    print(report["report_path"])
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
