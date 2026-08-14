"""Runtime profile parsing and projection helpers."""

from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class RuntimeProfileDeps:
    runtime_profile_from_log: Callable[..., dict | None]
    apply_runtime_projection: Callable[[dict, dict | None], Any]
    runtime_history_record: Callable[[dict], Any]
    expanduser: Callable[[str], str] = os.path.expanduser


def load_eta_tracker_module(skill_dir: str | Path):
    try:
        from . import eta_tracker  # type: ignore
        return eta_tracker
    except Exception:
        try:
            spec = importlib.util.spec_from_file_location(
                "eta_tracker", str(Path(skill_dir) / "eta_tracker.py")
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        except Exception:
            return None
    return None


def runtime_total_units_from_cmd(
    cmd: str,
    *,
    load_eta_tracker_module_fn: Callable[[], Any],
) -> int:
    eta_tracker = load_eta_tracker_module_fn()
    if eta_tracker and hasattr(eta_tracker, "_extract_total_from_cmd"):
        try:
            return int(eta_tracker._extract_total_from_cmd(cmd) or 0)
        except Exception:
            return 0
    return 0


def apply_runtime_projection(
    task: dict,
    projection: Optional[dict],
    *,
    now: Callable[[], float] = time.time,
) -> None:
    if not projection:
        return
    total_s = int(projection.get("total_s") or 0)
    if total_s <= 0:
        return
    task["runtime_total_s_est"] = total_s
    task["runtime_est_source"] = projection.get("source") or "progress"
    task["runtime_progress_at"] = int(now())
    if projection.get("eta_s") is not None:
        task["runtime_eta_s_est"] = int(projection.get("eta_s") or 0)
    if projection.get("current") is not None:
        task["runtime_current_unit"] = int(projection.get("current") or 0)
    if projection.get("total_units") is not None:
        task["runtime_total_units"] = int(projection.get("total_units") or 0)
    if projection.get("unit_s") is not None:
        task["runtime_unit_s_est"] = float(projection.get("unit_s") or 0.0)


def read_text_tail(path: str, max_bytes: int = 2 * 1024 * 1024) -> str:
    try:
        with open(os.path.expanduser(path), "rb") as handle:
            try:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            except OSError:
                pass
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""


def runtime_profile_from_log(
    log_path: str,
    cmd: str = "",
    observed_duration_s: float = 0,
    *,
    load_eta_tracker_module_fn: Callable[[], Any],
    read_text_tail_fn: Callable[[str], str] = read_text_tail,
):
    eta_tracker = load_eta_tracker_module_fn()
    if not eta_tracker or not hasattr(eta_tracker, "runtime_projection_from_log"):
        return None
    text = read_text_tail_fn(log_path)
    if not text:
        return None
    try:
        return eta_tracker.runtime_projection_from_log(
            text, cmd=cmd, observed_duration_s=observed_duration_s)
    except Exception:
        return None


def apply_test_log_runtime_profile(task: dict, test_log: str, *, deps: RuntimeProfileDeps) -> bool:
    profile = deps.runtime_profile_from_log(test_log, task.get("cmd") or "")
    if not profile:
        return False
    profile = dict(profile)
    expanded = deps.expanduser(test_log)
    profile["source"] = (profile.get("source") or "local_test_log") + ":" + expanded
    deps.apply_runtime_projection(task, profile)
    task["test_log_path"] = expanded
    deps.runtime_history_record(task)
    return True
