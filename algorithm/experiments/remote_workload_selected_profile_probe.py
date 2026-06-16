"""Direct remote selected-profile probe for real workload command templates.

This is the real-workload counterpart of ``remote_gpu_selected_profile_probe``.
It intentionally bypasses the live scheduler queue and runs only measurement
windows on explicitly named idle GPUs.  The generated summaries use the same
shape as service-curve reports, so selected-profile LCB gates can consume them
without changing the production scheduler.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from statistics import mean
from string import Formatter
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "skill" / "scheduler.py"
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
REMOTE_PROGRESS_WRAPPER_ROOT = "/tmp/scheduleurm_progress_wrapper_pkg"
REMOTE_PROGRESS_WRAPPER_DIR = f"{REMOTE_PROGRESS_WRAPPER_ROOT}/algorithm/experiments"
_NUM_RE = r"[0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?"
RATE_RE = re.compile(rf"\brate=({_NUM_RE})\s+([A-Za-z]+)/s\b", re.IGNORECASE)
STABLE_RE = re.compile(
    r"ScheduleurmStableRate\s+unit=([A-Za-z]+)\s+"
    rf"mean_rate=({_NUM_RE})\s+"
    rf"last_rate=({_NUM_RE})\s+"
    rf"cv=({_NUM_RE})\s+"
    rf"last_two_relative_delta=({_NUM_RE})\s+"
    r"samples=(\d+)\s+tail=([0-9.,-]*)"
)
_SCHEDULER_MODULE = None


def build_remote_workload_selected_profile_probe(
    *,
    run_id: str,
    node: str,
    gpus: list[int],
    profiles: list[int],
    cwd: str,
    cmd_template_file: str,
    output_root: str,
    seed_base: int = 700000,
    max_iters: int = 80,
    timeout_s: int = 300,
    unit: str = "iter",
    terminate_on_stable: bool = False,
    stable_windows: int = 3,
    min_rate_samples: int = 3,
    stable_cv: float = 0.08,
    stable_rel_delta: float = 0.05,
    stable_skip_samples: int = 0,
    coordinated_profile_launch: bool = False,
) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    raw_dir = run_dir / "raw"
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    template = Path(cmd_template_file).read_text(encoding="utf-8").strip()
    _deploy_progress_wrapper(node, raw_dir, cwd=cwd, output_root=output_root)
    summaries: dict[int, dict[str, Any]] = {}
    for profile in profiles:
        summaries[int(profile)] = _measure_profile(
            run_id=run_id,
            run_dir=run_dir,
            node=node,
            gpus=gpus,
            profile=int(profile),
            cwd=cwd,
            cmd_template=template,
            output_root=output_root.rstrip("/"),
            seed_base=seed_base,
            max_iters=max_iters,
            timeout_s=timeout_s,
            unit=unit,
            terminate_on_stable=terminate_on_stable,
            stable_windows=stable_windows,
            min_rate_samples=min_rate_samples,
            stable_cv=stable_cv,
            stable_rel_delta=stable_rel_delta,
            stable_skip_samples=stable_skip_samples,
            coordinated_profile_launch=coordinated_profile_launch,
        )
    result = {
        "gate": "remote_workload_selected_profile_probe",
        "run_id": run_id,
        "node": node,
        "gpus": list(gpus),
        "profiles": list(profiles),
        "summary_paths": [
            str(report_dir / f"profile_{int(profile)}_per_gpu_summary.json")
            for profile in profiles
        ],
        "rate_count": sum(len(summary.get("rates_unit_s") or []) for summary in summaries.values()),
        "stable_rate_ready_count": sum(
            int(summary.get("stable_rate_ready_count") or 0)
            for summary in summaries.values()
        ),
        "pass": all(bool(summary.get("measurement_valid")) for summary in summaries.values()),
    }
    (report_dir / "probe_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _measure_profile(
    *,
    run_id: str,
    run_dir: Path,
    node: str,
    gpus: list[int],
    profile: int,
    cwd: str,
    cmd_template: str,
    output_root: str,
    seed_base: int,
    max_iters: int,
    timeout_s: int,
    unit: str,
    terminate_on_stable: bool,
    stable_windows: int,
    min_rate_samples: int,
    stable_cv: float,
    stable_rel_delta: float,
    stable_skip_samples: int,
    coordinated_profile_launch: bool,
) -> dict[str, Any]:
    phase = f"profile_{int(profile)}_per_gpu"
    report_dir = run_dir / "reports"
    raw_dir = run_dir / "raw" / phase
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    launch_rows: list[tuple[int, int, dict[str, Any], str]] = []
    procs: list[tuple[int, int, dict[str, Any], subprocess.Popen[str]]] = []
    global_index = 0
    for gpu in gpus:
        for local_index in range(profile):
            rendered = _render_workload_command(
                cmd_template,
                run_id=run_id,
                phase=phase,
                count_per_gpu=profile,
                index=global_index,
                gpu_idx=int(gpu),
                seed_base=seed_base,
                max_iters=max_iters,
                output_root=output_root,
                node=node,
                terminate_on_stable=terminate_on_stable,
                stable_windows=stable_windows,
                min_rate_samples=min_rate_samples,
                stable_cv=stable_cv,
                stable_rel_delta=stable_rel_delta,
                stable_skip_samples=stable_skip_samples,
            )
            remote = (
                f"cd {shlex.quote(cwd)} && "
                f"CUDA_VISIBLE_DEVICES={int(gpu)} "
                f"PYTHONUNBUFFERED=1 "
                f"timeout --foreground {int(timeout_s)}s "
                f"{rendered['cmd']}"
            )
            launch_rows.append((int(gpu), local_index, rendered, remote))
            if not coordinated_profile_launch:
                proc = _remote_popen(node, remote, timeout_s=int(timeout_s) + 90)
                procs.append((int(gpu), local_index, rendered, proc))
            global_index += 1

    rows: list[dict[str, Any]] = []
    if coordinated_profile_launch:
        rows = _run_coordinated_profile(
            node=node,
            run_id=run_id,
            phase=phase,
            launch_rows=launch_rows,
            raw_dir=raw_dir,
            timeout_s=timeout_s,
            unit=unit,
        )
    else:
        for gpu, local_index, rendered, proc in procs:
            try:
                output, _ = proc.communicate(timeout=int(timeout_s) + 45)
            except subprocess.TimeoutExpired:
                proc.kill()
                output, _ = proc.communicate(timeout=10)
            rows.append(_row_from_output(
                gpu=gpu,
                local_index=local_index,
                rendered=rendered,
                returncode=int(proc.returncode or 0),
                output=output or "",
                raw_dir=raw_dir,
                unit=unit,
            ))

    if terminate_on_stable:
        failed_names = [
            str(row["run_name"])
            for row in rows
            if not bool(row.get("stable_rate_ready")) or int(row.get("returncode") or 0) != 0
        ]
        if failed_names:
            _cleanup_remote_run_names(
                node,
                failed_names,
                raw_dir / "cleanup_unstable_remote_runs",
            )

    rates = [float(row["rate"]) for row in rows if float(row["rate"]) > 0.0]
    stable_rates = [
        float(row["stable_rate"])
        for row in rows
        if bool(row.get("stable_rate_ready")) and float(row.get("stable_rate") or 0.0) > 0.0
    ]
    expected = {str(int(gpu)): int(profile) for gpu in gpus}
    actual: dict[str, int] = {}
    for row in rows:
        if float(row["rate"]) > 0.0:
            key = str(int(row["gpu"]))
            actual[key] = actual.get(key, 0) + 1
    observed_units = sorted({str(row.get("unit") or unit) for row in rows if float(row["rate"]) > 0.0})
    returncode_valid_count = sum(1 for row in rows if int(row.get("returncode") or 0) == 0)
    all_stable_rate_ready = len(stable_rates) == len(rows) if rows else False
    measurement_valid = (
        actual == expected
        and returncode_valid_count == len(rows)
        and (all_stable_rate_ready if terminate_on_stable else len(rates) == len(rows))
    )
    summary = {
        "phase": phase,
        "probe": "remote_workload_selected_profile_probe",
        "run_id": run_id,
        "node": node,
        "gpus": list(gpus),
        "profile": int(profile),
        "max_iters": int(max_iters),
        "timeout_s": int(timeout_s),
        "terminate_on_stable": bool(terminate_on_stable),
        "coordinated_profile_launch": bool(coordinated_profile_launch),
        "stable_windows": int(stable_windows),
        "min_rate_samples": int(min_rate_samples),
        "stable_cv_threshold": float(stable_cv),
        "stable_rel_delta_threshold": float(stable_rel_delta),
        "stable_skip_samples": int(stable_skip_samples),
        "elapsed_wall_s": time.time() - started,
        "running_count": len(rows),
        "running_with_rate_count": len(rates),
        "stable_rate_ready_count": len(stable_rates),
        "all_stable_rate_ready": all_stable_rate_ready,
        "returncode_valid_count": returncode_valid_count,
        "per_gpu_running": actual,
        "expected_per_gpu_running": expected,
        "rates_unit_s": rates,
        "stable_rates_unit_s": stable_rates,
        "rate_units": observed_units or [unit],
        "aggregate_active_rate_unit_s": sum(rates),
        "aggregate_stable_rate_unit_s": sum(stable_rates),
        "mean_active_rate_unit_s": mean(rates) if rates else 0.0,
        "mean_stable_rate_unit_s": mean(stable_rates) if stable_rates else 0.0,
        "measurement_aggregate_rate_unit_s_median": sum(rates),
        "blocked_count": 0,
        "eviction_count": 0,
        "placement_valid": actual == expected,
        "measurement_valid": measurement_valid,
        "placement_issues": [] if actual == expected else [f"actual per-GPU running {actual} != expected {expected}"],
        "status_counts": {"sampled": len(rates), "missing_rate": max(0, len(rows) - len(rates))},
        "capacity_boundary": False,
        "rows": rows,
    }
    summary_path = report_dir / f"{phase}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / f"{phase}_summary.md").write_text(_markdown(summary, summary_path), encoding="utf-8")
    return summary


def _render_workload_command(
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
    terminate_on_stable: bool,
    stable_windows: int,
    min_rate_samples: int,
    stable_cv: float,
    stable_rel_delta: float,
    stable_skip_samples: int,
) -> dict[str, Any]:
    run_name = f"{run_id}_{phase}_gpu{gpu_idx}_{index}"
    seed = int(seed_base) + int(index)
    values = {
        "run_id": run_id,
        "phase": phase,
        "profile": int(count_per_gpu),
        "count_per_gpu": int(count_per_gpu),
        "index": int(index),
        "gpu_idx": int(gpu_idx),
        "seed": seed,
        "max_iters": int(max_iters),
        "output_root": output_root.rstrip("/"),
        "run_name": run_name,
        "node": node,
        "terminate_on_stable_arg": "--terminate-on-stable" if terminate_on_stable else "",
        "stable_windows": int(stable_windows),
        "min_rate_samples": int(min_rate_samples),
        "stable_cv": float(stable_cv),
        "stable_rel_delta": float(stable_rel_delta),
        "stable_skip_samples": int(stable_skip_samples),
    }
    unknown = sorted(_template_fields(template) - set(values))
    if unknown:
        raise ValueError(f"unknown command-template placeholders: {unknown}")
    return {"cmd": template.format(**values), "run_name": run_name, "seed": seed, "values": values}


def _row_from_output(
    *,
    gpu: int,
    local_index: int,
    rendered: Mapping[str, Any],
    returncode: int,
    output: str,
    raw_dir: Path,
    unit: str,
) -> dict[str, Any]:
    log_path = raw_dir / f"gpu{int(gpu)}_{int(local_index)}.log"
    log_path.write_text(output or "", encoding="utf-8")
    rate, parsed_unit = _last_rate(output or "")
    stable = _last_stable_rate(output or "")
    return {
        "gpu": int(gpu),
        "idx": int(local_index),
        "global_index": (rendered.get("values") or {}).get("index"),
        "seed": rendered.get("seed"),
        "run_name": rendered.get("run_name"),
        "returncode": int(returncode),
        "rate": rate,
        "unit": parsed_unit or unit,
        "stable_rate_ready": bool(stable.get("ready")),
        "stable_rate": float(stable.get("last_rate") or 0.0),
        "stable_mean_rate": float(stable.get("mean_rate") or 0.0),
        "stable_tail_rates": stable.get("tail") or [],
        "stable_cv": float(stable.get("cv") or 0.0),
        "stable_last_two_relative_delta": float(stable.get("last_two_relative_delta") or 0.0),
        "stable_samples": int(stable.get("samples") or 0),
        "log_path": str(log_path),
    }


def _run_coordinated_profile(
    *,
    node: str,
    run_id: str,
    phase: str,
    launch_rows: list[tuple[int, int, dict[str, Any], str]],
    raw_dir: Path,
    timeout_s: int,
    unit: str,
) -> list[dict[str, Any]]:
    token = _safe_token(f"{run_id}_{phase}")
    remote_dir = f"/tmp/scheduleurm_workload_probe/{token}"
    script_path = f"/tmp/scheduleurm_workload_probe/{token}.sh"
    rewritten_launch_rows = [
        (
            gpu,
            local_index,
            rendered,
            _rewrite_remote_command(node, remote) if _scheduler_node(node) else remote,
        )
        for gpu, local_index, rendered, remote in launch_rows
    ]
    script = _coordinated_remote_script(remote_dir, rewritten_launch_rows)
    _write_remote_text(node, script_path, script, raw_dir / "coordinated_profile_script")
    rc, out, err = _run_remote_capture(
        node,
        f"bash {shlex.quote(script_path)}",
        raw_dir / "coordinated_profile_run",
        timeout_s=int(timeout_s) + 300,
    )
    (raw_dir / "coordinated_profile_run.combined.log").write_text(
        (out or "") + ("\n__STDERR__\n" + err if err else ""),
        encoding="utf-8",
    )
    logs, returncodes = _parse_coordinated_output(out or "")
    rows = []
    for gpu, local_index, rendered, _remote in rewritten_launch_rows:
        key = f"gpu{int(gpu)}_{int(local_index)}"
        rows.append(_row_from_output(
            gpu=gpu,
            local_index=local_index,
            rendered=rendered,
            returncode=int(returncodes.get(key, rc)),
            output=logs.get(key, ""),
            raw_dir=raw_dir,
            unit=unit,
        ))
    return rows


def _coordinated_remote_script(
    remote_dir: str,
    launch_rows: list[tuple[int, int, dict[str, Any], str]],
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set +e",
        f"REMOTE_DIR={shlex.quote(remote_dir)}",
        'rm -rf "$REMOTE_DIR"',
        'mkdir -p "$REMOTE_DIR"',
    ]
    keys: list[str] = []
    for gpu, local_index, _rendered, remote in launch_rows:
        key = f"gpu{int(gpu)}_{int(local_index)}"
        keys.append(key)
        lines.extend([
            f"KEY={shlex.quote(key)}",
            f"({remote}) > \"$REMOTE_DIR/$KEY.log\" 2>&1 &",
            'echo $! > "$REMOTE_DIR/$KEY.pid"',
        ])
    lines.append("OVERALL_RC=0")
    for key in keys:
        lines.extend([
            f"KEY={shlex.quote(key)}",
            'PID=$(cat "$REMOTE_DIR/$KEY.pid" 2>/dev/null || echo "")',
            'if [ -n "$PID" ]; then wait "$PID"; RC=$?; else RC=255; fi',
            'echo "$RC" > "$REMOTE_DIR/$KEY.rc"',
            'if [ "$RC" -ne 0 ]; then OVERALL_RC="$RC"; fi',
        ])
    for key in keys:
        lines.extend([
            f"KEY={shlex.quote(key)}",
            'echo "__SCHEDULEURM_LOG_BEGIN__ $KEY"',
            'cat "$REMOTE_DIR/$KEY.log" 2>/dev/null || true',
            'echo "__SCHEDULEURM_LOG_END__ $KEY"',
            'RC=$(cat "$REMOTE_DIR/$KEY.rc" 2>/dev/null || echo 255)',
            'echo "__SCHEDULEURM_RC__ $KEY $RC"',
        ])
    lines.append("exit 0")
    return "\n".join(lines) + "\n"


def _parse_coordinated_output(output: str) -> tuple[dict[str, str], dict[str, int]]:
    logs: dict[str, list[str]] = {}
    returncodes: dict[str, int] = {}
    current: str | None = None
    for line in (output or "").splitlines():
        if line.startswith("__SCHEDULEURM_LOG_BEGIN__ "):
            current = line.split(None, 1)[1].strip()
            logs.setdefault(current, [])
            continue
        if line.startswith("__SCHEDULEURM_LOG_END__ "):
            current = None
            continue
        if line.startswith("__SCHEDULEURM_RC__ "):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    returncodes[str(parts[1])] = int(parts[2])
                except ValueError:
                    returncodes[str(parts[1])] = 255
            continue
        if current is not None:
            logs.setdefault(current, []).append(line)
    return {key: "\n".join(value) + ("\n" if value else "") for key, value in logs.items()}, returncodes


def _safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or ""))[:180]


def _template_fields(template: str) -> set[str]:
    return {
        str(field)
        for _, field, _, _ in Formatter().parse(template or "")
        if field
    }


def _deploy_progress_wrapper(node: str, raw_dir: Path, *, cwd: str, output_root: str) -> None:
    remote_mkdir = (
        f"mkdir -p {shlex.quote(cwd)} {shlex.quote(output_root)} {REMOTE_PROGRESS_WRAPPER_DIR} "
        f"&& touch {REMOTE_PROGRESS_WRAPPER_ROOT}/algorithm/__init__.py "
        f"{REMOTE_PROGRESS_WRAPPER_DIR}/__init__.py"
    )
    _run_remote(node, remote_mkdir, raw_dir / "deploy_progress_wrapper_mkdir", timeout_s=45)
    local_dir = REPO_ROOT / "algorithm" / "experiments"
    files = (
        "progress_wrapper.py",
        "progress_units.py",
        "torch_cnn_progress_benchmark.py",
        "torch_llm_progress_benchmark.py",
        "jax_cnn_progress_benchmark.py",
    )
    if _scheduler_node(node):
        for name in files:
            _write_remote_text(
                node,
                f"{REMOTE_PROGRESS_WRAPPER_DIR}/{name}",
                (local_dir / name).read_text(encoding="utf-8"),
                raw_dir / f"deploy_{name}",
            )
    else:
        _run(
            [
                "scp",
                *(str(local_dir / name) for name in files),
                f"{node}:{REMOTE_PROGRESS_WRAPPER_DIR}/",
            ],
            raw_dir / "deploy_progress_wrapper_scp",
        )


def _cleanup_remote_run_names(node: str, run_names: list[str], prefix: Path) -> None:
    patterns = [name for name in run_names if str(name).strip()]
    if not patterns:
        return
    clauses = []
    for name in patterns:
        quoted = shlex.quote(str(name))
        clauses.append(
            "for p in $(pgrep -f " + quoted + " 2>/dev/null || true); do "
            '[ "$p" = "$$" ] && continue; '
            '[ "$p" = "$PPID" ] && continue; '
            "kill $p >/dev/null 2>&1 || true; "
            "done"
        )
    shell_cmd = "; ".join(clauses) + "; sleep 1; " + "; ".join(
        "for p in $(pgrep -f " + shlex.quote(str(name)) + " 2>/dev/null || true); do "
        '[ "$p" = "$$" ] && continue; '
        '[ "$p" = "$PPID" ] && continue; '
        "kill -9 $p >/dev/null 2>&1 || true; "
        "done"
        for name in patterns
    )
    try:
        _run_remote(node, shell_cmd, prefix, timeout_s=60)
    except Exception as exc:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".cleanup_error").write_text(repr(exc) + "\n", encoding="utf-8")


def _run(cmd: list[str], prefix: Path) -> None:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".stdout")).write_text(proc.stdout or "", encoding="utf-8")
    (prefix.with_suffix(".stderr")).write_text(proc.stderr or "", encoding="utf-8")
    (prefix.with_suffix(".meta.json")).write_text(
        json.dumps({"cmd": cmd, "returncode": proc.returncode}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")


def _run_remote(node: str, shell_cmd: str, prefix: Path, *, timeout_s: int) -> None:
    if _scheduler_node(node):
        scheduler = _scheduler_module()
        rendered = _rewrite_remote_command(node, shell_cmd)
        rc, out, err = _scheduler_run_on_retry(scheduler, node, rendered, timeout_s=timeout_s)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".stdout").write_text(out or "", encoding="utf-8")
        prefix.with_suffix(".stderr").write_text(err or "", encoding="utf-8")
        prefix.with_suffix(".meta.json").write_text(
            json.dumps({"cmd": ["scheduler.run_on", node, rendered], "returncode": rc}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if rc != 0:
            raise RuntimeError(f"remote command failed rc={rc} on {node}: {err.strip()[:300]}")
        return
    _run(["ssh", node, shell_cmd], prefix)


def _run_remote_capture(node: str, shell_cmd: str, prefix: Path, *, timeout_s: int) -> tuple[int, str, str]:
    if _scheduler_node(node):
        scheduler = _scheduler_module()
        rendered = _rewrite_remote_command(node, shell_cmd)
        rc, out, err = _scheduler_run_on_retry(scheduler, node, rendered, timeout_s=timeout_s)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".stdout").write_text(out or "", encoding="utf-8")
        prefix.with_suffix(".stderr").write_text(err or "", encoding="utf-8")
        prefix.with_suffix(".meta.json").write_text(
            json.dumps({"cmd": ["scheduler.run_on", node, rendered], "returncode": rc}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return int(rc), out or "", err or ""
    proc = subprocess.run(
        ["ssh", node, shell_cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_s,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".stdout").write_text(proc.stdout or "", encoding="utf-8")
    prefix.with_suffix(".stderr").write_text(proc.stderr or "", encoding="utf-8")
    prefix.with_suffix(".meta.json").write_text(
        json.dumps({"cmd": ["ssh", node, shell_cmd], "returncode": proc.returncode}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _scheduler_run_on_retry(scheduler, node: str, rendered: str, *, timeout_s: int) -> tuple[int, str, str]:
    attempts = 4
    last = (255, "", "")
    for attempt in range(1, attempts + 1):
        rc, out, err = scheduler.run_on(node, rendered, timeout=timeout_s, check=False)
        last = (int(rc), out or "", err or "")
        if int(rc) == 0 or not _transient_ssh_error(out, err):
            return last
        time.sleep(min(8, 2 * attempt))
    return last


def _transient_ssh_error(out: str | None, err: str | None) -> bool:
    text = f"{out or ''}\n{err or ''}".lower()
    needles = (
        "connection timed out during banner exchange",
        "connection refused",
        "could not resolve hostname",
        "temporary failure in name resolution",
        "kex_exchange_identification",
        "connection closed by remote host",
        "connection closed by unknown port",
        "mux_client_request_stdio_fwd",
        "broken pipe",
        "server not responding",
    )
    return any(needle in text for needle in needles)


def _remote_popen(node: str, shell_cmd: str, *, timeout_s: int) -> subprocess.Popen[str]:
    if _scheduler_node(node):
        scheduler = _scheduler_module()
        rendered = _rewrite_remote_command(node, shell_cmd)
        remote = scheduler._remote_bash_command_for_node(node, rendered, timeout=timeout_s)
        argv = scheduler._ssh_no_stdin_args(node) + [remote]
        return subprocess.Popen(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    return subprocess.Popen(
        ["ssh", node, shell_cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _write_remote_text(node: str, remote_path: str, text: str, prefix: Path) -> None:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    command = (
        f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && "
        f"printf %s {shlex.quote(payload)} | base64 -d > {shlex.quote(remote_path)}"
    )
    _run_remote(node, command, prefix, timeout_s=45)


def _rewrite_remote_command(node: str, shell_cmd: str) -> str:
    if not _scheduler_node(node):
        return shell_cmd
    scheduler = _scheduler_module()
    rewritten = scheduler._rewrite_command_paths_for_node(node, shell_cmd)
    rewritten = scheduler._apply_node_cmd_rewrites(node, rewritten)
    extra_env = (getattr(scheduler, "NODES", {}).get(node, {}) or {}).get("launch_extra_env") or {}
    exports = []
    for key, value in sorted(extra_env.items()):
        text_key = str(key or "")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text_key):
            continue
        exports.append(f"export {text_key}={shlex.quote(str(value))};")
    if exports:
        rewritten = " ".join(exports) + " " + rewritten
    return rewritten


def _scheduler_node(node: str) -> bool:
    try:
        scheduler = _scheduler_module()
        return str(node) in getattr(scheduler, "NODES", {}) and not scheduler._node_is_windows(str(node))
    except Exception:
        return False


def _scheduler_module():
    global _SCHEDULER_MODULE
    if _SCHEDULER_MODULE is None:
        spec = importlib.util.spec_from_file_location("scheduleurm_scheduler_remote_probe", SCHEDULER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load scheduler module from {SCHEDULER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SCHEDULER_MODULE = module
    return _SCHEDULER_MODULE


def _last_rate(output: str) -> tuple[float, str]:
    matches = RATE_RE.findall(output or "")
    if not matches:
        return 0.0, ""
    rate, unit = matches[-1]
    return float(rate), str(unit)


def _last_stable_rate(output: str) -> dict[str, Any]:
    matches = STABLE_RE.findall(output or "")
    if not matches:
        return {"ready": False}
    unit, mean_rate, last_rate, cv, rel, samples, tail = matches[-1]
    tail_rates = [
        float(part)
        for part in str(tail or "").split(",")
        if part.strip()
    ]
    return {
        "ready": True,
        "unit": str(unit),
        "mean_rate": float(mean_rate),
        "last_rate": float(last_rate),
        "cv": float(cv),
        "last_two_relative_delta": float(rel),
        "samples": int(samples),
        "tail": tail_rates,
    }


def _markdown(summary: Mapping[str, Any], summary_path: Path) -> str:
    return "\n".join([
        "# Remote Workload Selected-Profile Probe",
        "",
        f"Summary: `{summary_path}`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `phase` | `{summary.get('phase')}` |",
        f"| `node` | `{summary.get('node')}` |",
        f"| `running_count` | {summary.get('running_count')} |",
        f"| `running_with_rate_count` | {summary.get('running_with_rate_count')} |",
        f"| `stable_rate_ready_count` | {summary.get('stable_rate_ready_count')} |",
        f"| `aggregate_active_rate_unit_s` | {float(summary.get('aggregate_active_rate_unit_s') or 0.0):.6g} |",
        f"| `aggregate_stable_rate_unit_s` | {float(summary.get('aggregate_stable_rate_unit_s') or 0.0):.6g} |",
        f"| `mean_active_rate_unit_s` | {float(summary.get('mean_active_rate_unit_s') or 0.0):.6g} |",
        f"| `mean_stable_rate_unit_s` | {float(summary.get('mean_stable_rate_unit_s') or 0.0):.6g} |",
        f"| `placement_valid` | {str(bool(summary.get('placement_valid'))).lower()} |",
        "",
    ])


def _cmd_build(args: argparse.Namespace) -> int:
    result = build_remote_workload_selected_profile_probe(
        run_id=args.run_id,
        node=args.node,
        gpus=[int(x) for x in str(args.gpus).split(",") if x.strip()],
        profiles=[int(x) for x in str(args.profiles).split(",") if x.strip()],
        cwd=args.cwd,
        cmd_template_file=args.cmd_template_file,
        output_root=args.output_root,
        seed_base=args.seed_base,
        max_iters=args.max_iters,
        timeout_s=args.timeout_s,
        unit=args.unit,
        terminate_on_stable=args.terminate_on_stable,
        stable_windows=args.stable_windows,
        min_rate_samples=args.min_rate_samples,
        stable_cv=args.stable_cv,
        stable_rel_delta=args.stable_rel_delta,
        stable_skip_samples=args.stable_skip_samples,
        coordinated_profile_launch=args.coordinated_profile_launch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.remote_workload_selected_profile_probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Measure real workload selected profiles without using scheduler queue")
    build.add_argument("--run-id", required=True)
    build.add_argument("--node", required=True)
    build.add_argument("--gpus", default="0,1")
    build.add_argument("--profiles", required=True)
    build.add_argument("--cwd", required=True)
    build.add_argument("--cmd-template-file", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--seed-base", type=int, default=700000)
    build.add_argument("--max-iters", type=int, default=80)
    build.add_argument("--timeout-s", type=int, default=300)
    build.add_argument("--unit", default="iter")
    build.add_argument("--terminate-on-stable", action="store_true")
    build.add_argument("--stable-windows", type=int, default=3)
    build.add_argument("--min-rate-samples", type=int, default=3)
    build.add_argument("--stable-cv", type=float, default=0.08)
    build.add_argument("--stable-rel-delta", type=float, default=0.05)
    build.add_argument("--stable-skip-samples", type=int, default=0)
    build.add_argument("--coordinated-profile-launch", action="store_true")
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
