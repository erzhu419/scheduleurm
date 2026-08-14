"""Terminal task log diagnosis heuristics."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .analysis import log_pattern_matches
from .final_model_success import FinalModelSuccessDeps, terminal_final_model_success
from .local_oom_detection import LocalOomKillDetectionDeps, detect_oom_kills_local
from .log_success_scan import FullLogSuccessScanDeps, scan_full_log_for_success


@dataclass(frozen=True)
class TerminalDiagnosisDeps:
    node_configs: dict
    crash_patterns: list
    training_markers: list
    early_death_seconds: int
    short_live_seconds: int
    node_is_windows: Callable[[str], bool]
    fetch_windows_text_tail: Callable[..., tuple[str, int]]
    run_on: Callable[..., tuple]
    success_patterns_for_task: Callable[[dict], list]
    cmd_looks_like_eval_or_benchmark: Callable[[str], bool]
    scan_full_log_for_success: Callable[[dict], list]
    terminal_final_model_success: Callable[[dict], str | None]
    terminal_result_artifact_success: Callable[[dict], str | None]
    now: Callable[[], float]


def _read_task_log_tail(task: dict, log_path: str, deps: TerminalDiagnosisDeps) -> tuple[str, int]:
    evidence = task.get("_terminal_evidence") or {}
    if (
        isinstance(evidence, dict)
        and str(evidence.get("log_path") or "") == str(log_path or "")
    ):
        return str(evidence.get("tail") or ""), int(evidence.get("log_size") or 0)
    try:
        node = task.get("node")
        if node and deps.node_configs.get(node, {}).get("host") is None:
            path = Path(log_path)
            if path.exists():
                log_size = path.stat().st_size
                with open(path, "rb") as fh:
                    fh.seek(max(0, log_size - 4096))
                    return fh.read().decode("utf-8", errors="replace"), log_size
        elif node and deps.node_is_windows(node):
            return deps.fetch_windows_text_tail(task, log_path, max_bytes=4096)
        elif node:
            rc, out, _ = deps.run_on(
                node,
                f"tail -c 4096 {shlex.quote(log_path)} 2>/dev/null; "
                f"echo '___SZ___'; wc -c < {shlex.quote(log_path)} 2>/dev/null",
                timeout=10,
                check=False,
            )
            if rc == 0 and out:
                if "___SZ___" in out:
                    body, _, size_text = out.rpartition("___SZ___")
                    try:
                        return body, int(size_text.strip())
                    except ValueError:
                        return body, 0
                return out, 0
    except Exception:
        pass
    return "", 0


def _redirect_log_path(cmd: str) -> str | None:
    match = re.search(r"(?:^|\s)(?:&>|>)\s*([^\s|;&)<>]+)", cmd)
    return match.group(1) if match else None


def _recover_redirected_log(
    task: dict,
    cmd_str: str,
    log_size: int,
    deps: TerminalDiagnosisDeps,
) -> tuple[str | None, int, bool]:
    cmd_has_own_redirect = (
        bool(re.search(r"(?<!\d)(?:&>|>>|2>&1|>&|>)\s*[^\s|;&)]+", cmd_str))
        or "2>&1" in cmd_str
    )
    if not (cmd_has_own_redirect and log_size == 0):
        return None, log_size, True
    real_log_path = _redirect_log_path(cmd_str)
    tail_text = ""
    if real_log_path:
        tail_text, log_size = _read_task_log_tail(task, real_log_path, deps)
    return tail_text, log_size, log_size > 0


def _disk_full_reason(task: dict, log_path: str, deps: TerminalDiagnosisDeps) -> str | None:
    try:
        log_dir = log_path.rsplit("/", 1)[0] if "/" in log_path else "/tmp"
        if deps.node_is_windows(task.get("node") or ""):
            return None
        rc, out, _ = deps.run_on(
            task.get("node") or "local",
            f"df -P {shlex.quote(log_dir)} | tail -n +2 | awk '{{print $5}}' | tr -d %",
            timeout=5,
            check=False,
        )
        if rc == 0:
            pct = int((out.strip() or "0").splitlines()[-1])
            if pct >= 95:
                return f"DISK_FULL: {log_dir} at {pct}% on {task.get('node')}"
    except Exception:
        pass
    return None


def _normal_result(task: dict, log_path: str | None, lifetime: float) -> dict:
    return {
        "is_crash": False,
        "reason": "auto-adopted (no scheduler log; cannot diagnose)",
        "tail": "(no log)",
        "lifetime_s": int(lifetime),
        "log_size": 0,
        "log_path": log_path,
        "success_marker": None,
    }


def diagnose_terminal(task: dict, *, deps: TerminalDiagnosisDeps) -> dict:
    """Inspect a just-finished task and classify normal completion vs crash."""
    log_path = task.get("log_path")
    started = task.get("started_at") or 0
    finished = task.get("finished_at") or deps.now()
    lifetime = max(0, finished - started)
    if task.get("auto_adopted") or not log_path:
        return _normal_result(task, log_path, lifetime)

    batch_probe_only = bool(task.get("_terminal_batch_probe_only"))
    terminal_evidence = task.get("_terminal_evidence") or {}
    tail_text, log_size = _read_task_log_tail(task, log_path, deps)
    cmd_str = task.get("cmd") or ""
    if batch_probe_only:
        redirect_tail, redirect_size, log_trusted = None, log_size, log_size > 0
    else:
        redirect_tail, redirect_size, log_trusted = _recover_redirected_log(
            task, cmd_str, log_size, deps)
    if redirect_tail is not None:
        tail_text, log_size = redirect_tail, redirect_size

    err_matched = [
        pattern for pattern in deps.crash_patterns
        if log_pattern_matches(tail_text, pattern)
    ]
    evidence_failure = str(terminal_evidence.get("failure_reason") or "")
    if evidence_failure:
        err_matched.append(evidence_failure)
    success_matched = [
        pattern for pattern in deps.success_patterns_for_task(task)
        if pattern in tail_text
    ]
    is_eval_or_benchmark = deps.cmd_looks_like_eval_or_benchmark(cmd_str)
    if not err_matched and not success_matched:
        result_artifact_success = str(terminal_evidence.get("success_marker") or "")
        if (
            not result_artifact_success
            and not terminal_evidence.get("artifact_checked")
            and not batch_probe_only
        ):
            result_artifact_success = deps.terminal_result_artifact_success(task)
        if result_artifact_success:
            success_matched = [result_artifact_success]
    if batch_probe_only and not err_matched and not success_matched:
        return {
            "is_crash": False,
            "reason": "terminal evidence is ambiguous; deep diagnosis required",
            "tail": tail_text[-600:].strip() if tail_text else "(no log)",
            "lifetime_s": int(lifetime),
            "log_size": log_size,
            "log_path": log_path,
            "success_marker": None,
            "needs_deep_terminal_diagnosis": True,
        }
    if not err_matched and log_trusted and not success_matched:
        full_log_success = deps.scan_full_log_for_success(task)
        if full_log_success:
            success_matched = full_log_success
    if not err_matched and not success_matched:
        final_model_success = deps.terminal_final_model_success(task)
        if final_model_success:
            success_matched = [final_model_success]

    reasons = []
    is_crash = False
    if err_matched:
        is_crash = True
        reasons.append("err_pattern: " + ", ".join(err_matched[:3]))
    elif success_matched:
        pass
    elif 0 < lifetime < deps.early_death_seconds:
        is_crash = True
        reasons.append(f"died after only {lifetime:.0f}s with no success marker")
    elif 0 < lifetime < deps.short_live_seconds:
        is_crash = True
        reasons.append(
            f"finished in {lifetime:.0f}s with no success marker (expected DONE/Saved/complete)"
        )
    elif log_trusted and log_size < 500 and lifetime > 60:
        is_crash = True
        reasons.append(f"log only {log_size}B after {lifetime:.0f}s (very suspicious)")
        disk_full = _disk_full_reason(task, log_path, deps)
        if disk_full:
            reasons.append(disk_full)
    else:
        has_training = any(marker in tail_text for marker in deps.training_markers)
        peak_vram = int(task.get("peak_vram_mb") or 0)
        if (
            log_trusted
            and not has_training
            and peak_vram == 0
            and lifetime > deps.short_live_seconds
            and not is_eval_or_benchmark
        ):
            is_crash = True
            reasons.append(
                f"never entered training: lifetime {lifetime:.0f}s but peak_vram=0 and no "
                f"training markers in log tail (likely OOM/silent-kill mid-init)"
            )
        elif (
            log_trusted
            and not has_training
            and peak_vram > 0
            and not is_eval_or_benchmark
        ):
            is_crash = True
            reasons.append(
                f"GPU work observed (peak_vram={peak_vram}MB) but no success marker after "
                f"{lifetime:.0f}s — kill mid-execution (training markers may have been "
                f"rotated out of tail or use non-standard format); auto-requeue will "
                f"resume from latest ckpt if --resume-flag is set"
            )
        elif log_trusted and has_training and not is_eval_or_benchmark:
            is_crash = True
            reasons.append(
                f"training markers present but no success marker after {lifetime:.0f}s — "
                f"task killed mid-training (likely SIGKILL/OOM/host reboot); "
                f"auto-requeue will resume from latest ckpt if --resume-flag is set"
            )

    return {
        "is_crash": is_crash,
        "reason": (
            "; ".join(reasons) or "normal exit (success marker found)"
            if success_matched
            else "; ".join(reasons) or "ambiguous; assumed normal"
        ),
        "tail": tail_text[-600:].strip() if tail_text else "(no log)",
        "lifetime_s": int(lifetime),
        "log_size": log_size,
        "log_path": log_path,
        "success_marker": success_matched[0] if success_matched else None,
    }
