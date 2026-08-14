"""Log-tail crash scanning and failure-category classification."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CRASH_PATTERNS = [
    "Traceback (most recent call",
    "Error:", "Exception:",
    "Killed", "Segmentation fault",
    "out of memory", "OOM", "CUDA out of memory",
    "ModuleNotFoundError", "ImportError", "FileNotFoundError",
    "ConnectionError", "AssertionError", "RuntimeError",
]

SUCCESS_PATTERNS = [
    "Training complete",
    "DONE",
    " complete!",
    "Final model saved",
    # Some training entrypoints use the shorter form after atomically writing
    # their only terminal checkpoint (for example ``model_final.pt``).
    "Model saved",
    "Saved final",
    "Eval complete",
    "Saved: ",
    "Saving final",
    "[Done]",
    "Done. Results in",
    "Results saved to",
    "Best composite=",
    "Best headway:",
    "node-worker complete failures=0",
    "Running 0 checkpoints",
    "Nothing to ",
    "no checkpoints to",
    "All ckpts already",
    "PAIR COMPLETE:",
    "PAIR ALREADY COMPLETE:",
    "FINALIZE-ONLY COMPLETE:",
    "RECOVERY COMPLETE:",
    "INDEPENDENT SPECIALIST AUDIT COMPLETE:",
    "Complete valid specialist audit already exists:",
    "INDEPENDENT SPECIALIST ANALYSIS COMPLETE:",
]

EVAL_SUCCESS_PATTERNS = [
    '"summary_rows"',
    "'summary_rows'",
    '"summary_csv"',
    "'summary_csv'",
    '"schema_version"',
    "'schema_version'",
    "posterior_solve_share_mean",
    "candidate_source_mix",
    '"summary_paths"',
    "'summary_paths'",
    "Wrote ",
    # offline-sumo's revision evaluator writes durable per-episode CSV rows
    # and finishes with this marker.  It is a CPU simulation evaluation, not
    # a training job, so an exit code of zero plus this marker is conclusive.
    "[eval_revision_metrics] complete",
    "[eval_revision_metrics] complete: no pending tasks",
]

TRAINING_MARKERS = [
    "Epoch ", "[Epoch", "epoch ",
    "step=", "[Step", "Step ",
    "iter=", "iteration ",
    "loss=", "loss ",
    "Train: ", "[Train",
    "val_", "eval_",
]

EARLY_DEATH_SECONDS = 120
SHORT_LIVE_SECONDS = 600


def log_pattern_matches(text: str, pattern: str) -> bool:
    """Match crash markers without treating embedded acronyms as failures."""
    if pattern == "OOM":
        return re.search(r"(?<![A-Za-z0-9_])OOM(?![A-Za-z0-9_])", text) is not None
    return pattern in text


@dataclass(frozen=True)
class LogTailDeps:
    node_configs: dict
    node_is_windows: Callable[[str], bool]
    fetch_windows_text_tail: Callable[..., tuple[str, int]]
    run_on: Callable[..., tuple[int, str, str]]


@dataclass(frozen=True)
class CompletedLogCrashScanDeps:
    fetch_log_tail: Callable[[dict], tuple[str, int]]
    crash_patterns: list


@dataclass(frozen=True)
class FailureClassificationDeps:
    env_missing_patterns: list
    python_import_patterns: list
    cuda_runtime_patterns: list
    invalid_flag_patterns: list
    disk_full_patterns: list
    oom_patterns: list


def fetch_log_tail(task: dict, *, deps: LogTailDeps) -> tuple[str, int]:
    """Fetch the last chunk of a scheduler log from local, Windows, or remote Linux."""
    log_path = task.get("log_path")
    if not log_path or task.get("auto_adopted"):
        return ("", 0)
    try:
        node = task.get("node")
        if node and deps.node_configs.get(node, {}).get("host") is None:
            path = Path(log_path)
            if not path.exists():
                return ("", 0)
            size = path.stat().st_size
            with open(path, "rb") as fh:
                fh.seek(max(0, size - 4096))
                return (fh.read().decode("utf-8", errors="replace"), size)
        if node and deps.node_is_windows(node):
            return deps.fetch_windows_text_tail(task, log_path, max_bytes=4096)
        if node:
            rc, out, _ = deps.run_on(
                node,
                f"tail -c 4096 {shlex.quote(log_path)} 2>/dev/null; "
                f"echo '___SZ___'; wc -c < {shlex.quote(log_path)} 2>/dev/null",
                timeout=10,
                check=False,
            )
            if rc != 0 or not out:
                return ("", 0)
            if "___SZ___" in out:
                body, _, size_text = out.rpartition("___SZ___")
                try:
                    return (body, int(size_text.strip()))
                except ValueError:
                    return (body, 0)
            return (out, 0)
    except Exception:
        return ("", 0)
    return ("", 0)


def scan_completed_log_for_crash(
    task: dict,
    *,
    deps: CompletedLogCrashScanDeps,
) -> tuple[bool, str]:
    """Scan legacy externally completed logs only for explicit crash patterns."""
    tail_text, _log_size = deps.fetch_log_tail(task)
    if not tail_text:
        return (False, "")
    matched = [
        pattern for pattern in deps.crash_patterns
        if log_pattern_matches(tail_text, pattern)
    ]
    if matched:
        return (
            True,
            "legacy external scheduler reported COMPLETED but log tail contains crash "
            f"pattern(s): {', '.join(matched[:3])}",
        )
    return (False, "")


def cmd_looks_like_eval_or_benchmark(
    cmd: str,
    *,
    cmd_looks_like_training: Callable[[str], bool] | None = None,
) -> bool:
    lower = (cmd or "").lower()
    if not lower:
        return False
    try:
        if cmd_looks_like_training is not None and cmd_looks_like_training(cmd):
            return False
    except Exception:
        pass
    return any(p in lower for p in (
        "benchmark_",
        "validate_performance.py",
        "/performance/",
        " performance/",
        "final_task_sweep",
        "pytest ",
        "python -m pytest",
        "lake build",
        "eval_revision_metrics.py",
    ))


def cmd_has_repeated_subrun_done(cmd: str) -> bool:
    lower = (cmd or "").lower()
    return "benchmark_traffic_ingolstadt21.py" in lower


def success_patterns_for_task(
    task: dict,
    *,
    cmd_looks_like_training: Callable[[str], bool] | None = None,
) -> list[str]:
    cmd = (task or {}).get("cmd", "")
    patterns = list(SUCCESS_PATTERNS)
    if "run_training_dispatch_v7.py" in cmd:
        # The formal transit host runner prints its atomically written report
        # path as the only terminal line. A nonzero backend exit still wins in
        # lifecycle finalization, so this marker only closes successful runs.
        patterns.append("/host_execution.json")
    if cmd_has_repeated_subrun_done(cmd):
        patterns = [p for p in patterns if p != "DONE"]
    if cmd_looks_like_eval_or_benchmark(cmd, cmd_looks_like_training=cmd_looks_like_training):
        patterns.extend(EVAL_SUCCESS_PATTERNS)
    return patterns


def cached_task_success_marker(
    task: dict,
    *,
    success_patterns_for_task_fn: Callable[[dict], list[str]],
) -> str:
    diag = task.get("_diagnosis") or {}
    texts = [
        task.get("last_progress_line") or "",
        diag.get("success_marker") or "",
        diag.get("tail") or "",
    ]
    for text in texts:
        if not text:
            continue
        for marker in success_patterns_for_task_fn(task):
            if marker and marker in str(text):
                return marker
    return ""


def classify_failure(diag: dict, *, deps: FailureClassificationDeps) -> str:
    """Categorize terminal diagnostics for retry/escalation routing."""
    if not diag or not diag.get("is_crash"):
        return "NORMAL"
    tail = diag.get("tail", "") or ""
    reason = diag.get("reason", "") or ""
    haystack = (tail + " " + reason).lower()
    # Python raises FileNotFoundError for missing task inputs, checkpoints and
    # generated artifacts as well as for environment files.  The generic
    # "No such file" pattern must not turn those task-local failures into a
    # node-wide ENV_MISSING block.  Missing interpreters/commands do not carry
    # this Python exception marker and continue through the environment rules.
    if "filenotfounderror" in haystack:
        return "ARTIFACT_MISSING"
    for pattern in deps.env_missing_patterns:
        if pattern.lower() in haystack:
            return "ENV_MISSING"
    for pattern in deps.python_import_patterns:
        if pattern.lower() in haystack:
            return "PYTHON_IMPORT"
    for pattern in deps.cuda_runtime_patterns:
        if pattern.lower() in haystack:
            return "CUDA_RUNTIME"
    for pattern in deps.invalid_flag_patterns:
        if pattern.lower() in haystack:
            return "INVALID_FLAG"
    for pattern in deps.disk_full_patterns:
        if pattern.lower() in haystack:
            return "DISK_FULL"
    for pattern in deps.oom_patterns:
        if pattern.lower() in haystack:
            return "OOM"
    if "Traceback" in tail and diag.get("lifetime_s", 0) > 60:
        return "APP_BUG"
    return "UNKNOWN"
