#!/usr/bin/env python3
"""Multi-resource scheduler across local (4060 8GB / 16c / 64GB) + jtl110gpu
(2x 3080Ti 12GB / 12c / auto-probed RAM) + jtl110gpu2 (same) + jtl311linux
(2x 2080 8GB / 8c / auto-probed RAM) + jtl110cpu/jtl110cpu2 (Windows CPU-only /
128 physical cores / 512GB each).

Resource model: each task declares cpu_cores, ram_mb, vram_mb (or vram=0 for CPU-only). Placement requires
ALL three to fit on the chosen node + GPU. Per-task resource needs are auto-learned from history (peak
VRAM and peak RAM, cores user-declared) so re-runs of the same signature use accurate budgets.

Subcommands:
  submit    Add a task to the queue (no launch yet).
  dispatch  Probe nodes, pick placements for queued tasks, launch what fits. Same call doubles as rebalance.
  status    Show node telemetry + task table. Updates running-task health and peak VRAM/RAM.
  doctor    Audit queue invariants; --fix applies safe queued-task repairs.
  profile-local Run a local preflight directly and record resource/runtime history.
  results   Find inferred result artifacts in queue + archive.
  cancel    Cancel one or many tasks; with --force, kill selected running tasks.
  forget    Drop a task record (never touches processes — for fixing wrong adopts).
  clear-queue   Cancel ALL queued tasks (running tasks untouched). Requires --confirm.
  record-vram   Manually record peak VRAM for a signature (auto-tracked too via status).
  history   Show recorded peak VRAM / RAM per signature.
  show      Print one task's full record (incl. log path, resume_from, etc.).
  adopt     Register externally-launched PIDs as a tracked task.
  watch     Background daemon: probe + dispatch every --interval s; auto-adopt external GPU tasks.

State lives in ~/.claude/scheduler/{queue.json, vram_history.json, logs/}. Inter-process safe via flock.
"""
import signal
import sys
from pathlib import Path

_BOOTSTRAP_SKILL_DIR = Path(__file__).resolve().parent
if str(_BOOTSTRAP_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_SKILL_DIR))

from scheduler_facade_bootstrap import prepare_scheduler_facade

env_deploy, _SKILL_DIR, _SCHEDULEURM_ROOT = prepare_scheduler_facade(__file__)

from scheduler_compat_imports import *
from scheduler_config import build_scheduler_config_exports as _build_scheduler_config_exports
from scheduler_eta.state import (
    clear_live_eta_fields as _clear_live_eta_fields_impl,
    eta_confidence_for_source as _eta_confidence_for_source_impl,
    eta_source_base as _eta_source_base_impl,
    eta_source_tag as _eta_source_tag_impl,
    fmt_eta_seconds as _fmt_eta_seconds_impl,
    format_task_eta as _format_task_eta_impl,
    history_fallback_eta_seconds as _history_fallback_eta_seconds_impl,
    is_live_remaining_eta_source as _is_live_remaining_eta_source_impl,
    is_live_runtime_projection_source as _is_live_runtime_projection_source_impl,
    queued_has_stale_live_eta as _queued_has_stale_live_eta_impl,
)
from scheduler_failure.analysis import (
    CRASH_PATTERNS,
    EARLY_DEATH_SECONDS,
    EVAL_SUCCESS_PATTERNS,
    SHORT_LIVE_SECONDS,
    SUCCESS_PATTERNS,
    TRAINING_MARKERS,
    cached_task_success_marker as _cached_task_success_marker_impl,
    cmd_has_repeated_subrun_done as _cmd_has_repeated_subrun_done_impl,
    cmd_looks_like_eval_or_benchmark as _cmd_looks_like_eval_or_benchmark_impl,
    success_patterns_for_task as _success_patterns_for_task_impl,
)
from scheduler_runtime.bootstrap import (
    build_scheduler_runtime_exports as _build_scheduler_runtime_exports,
)
from scheduler_watch.shutdown import terminate_active_control_subprocesses

globals().update(_build_scheduler_config_exports())
# ---------- node inventory ----------
# The concrete node table lives in scheduler_node/inventory.py. Keep this section for the
# scheduler.py compatibility wrappers that historically owned node name handling.

def _canonical_node_name(name) -> str:
    return _canonical_node_name_impl(name, NODE_NAME_ALIASES)


def _canonicalize_node_list(values) -> list:
    """Canonicalize node aliases while preserving the user's node set.

    This deliberately does not add compatibility siblings. In particular,
    `node007-direct` may canonicalize to `node007`, but an explicit
    --allowed-node list that omitted node007 must stay omitted.
    """
    return _canonicalize_node_list_impl(values, NODE_NAME_ALIASES)


def _canonicalize_state_node_names(state: dict) -> bool:
    return _canonicalize_state_node_names_impl(state, NODE_NAME_ALIASES)


_LAST_AUTO_ADOPT_AT = 0.0

# ---------- runtime compatibility exports ----------
globals().update(_build_scheduler_runtime_exports(globals()))


# ---------- arg parsing ----------
def _scheduler_cli_deps() -> _SchedulerCliDeps:
    return _build_scheduler_cli_deps(globals())


def main(argv=None):
    _reset_watcher_shutdown()
    previous_handlers = {}

    def _control_signal(sig, frame):
        _request_watcher_shutdown(sig, frame)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _control_signal)
    except ValueError:
        previous_handlers = {}
    try:
        return _run_scheduler_cli(_scheduler_cli_deps(), argv=argv)
    finally:
        terminate_active_control_subprocesses(signal.SIGTERM)
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass

if __name__ == "__main__":
    main()
