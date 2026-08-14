"""eta_tracker: live ETA estimation from log tail.

Phase 3.0.1: parses tqdm / epoch / iter progress patterns out of running tasks'
log tails and computes a rate-based remaining-seconds estimate. Falls back to
history EWMA when no pattern is found.

Two reasons this matters:
  1. The history EWMA (`dur_s_ewma`) is a per-signature aggregate — it doesn't
     reflect THIS run's actual progress. A task that's 90% done shouldn't be
     accounted as "EWMA seconds remaining"; it should be "ETA: 10% × EWMA".
  2. Phase 3.0's load-balanced migration needs per-node load = sum of remaining
     ETAs of in-flight tasks. Without live ETA, we'd over-estimate load on a
     node whose tasks are nearly finished + under-estimate on a node where
     fresh tasks just kicked off.

Output: parse_eta(tail_text, elapsed_s, fallback_ewma_s) → int seconds.

Patterns are tried in priority order; the LAST match in the tail wins (most
recent progress line). Tolerant to missing patterns (returns fallback) and to
absurd values (clamps current ≤ total, eta ≥ 0).
"""
from __future__ import annotations

import json
import re
import shlex
import statistics
from datetime import datetime
from typing import Optional, Tuple


# Patterns (priority order). Each returns groups (current, total). Optional
# third group (rate) is parsed but currently unused — we recompute rate from
# current/elapsed because that's more reliable than parsing the displayed unit
# (some tools show "it/s" for steps, others for episodes; ambiguous).
_TQDM_UNIT_RE = r'(?:it|iter|iters|step|steps|epoch|epochs|episode|episodes|update|updates|sample|samples)'
_ETA_PATTERNS = [
    # tqdm: "  47%|████▋     | 1234/5678 [00:42<03:21, 12.34it/s]"
    # also catches simpler tqdm without percent prefix: "1234/5678 [..., 12.34it/s]"
    re.compile(r'(\d+)\s*/\s*(\d+)\s*\[[^\]]*?(\d+(?:\.\d+)?)\s*' + _TQDM_UNIT_RE + r'/s'),
    # tqdm slow form: "1234/5678 [..., 1.23s/it]"
    re.compile(r'(\d+)\s*/\s*(\d+)\s*\[[^\]]*?(\d+(?:\.\d+)?)\s*s/' + _TQDM_UNIT_RE),
    # explicit "[Epoch N/M]" e.g. "[Epoch 23/200]"
    re.compile(r'\[Epoch\s+(\d+)\s*/\s*(\d+)\]'),
    # explicit "Epoch: N/M" or "Epoch N/M"
    re.compile(r'(?:^|\s)Epoch[:\s]+(\d+)\s*/\s*(\d+)\b'),
    # logging-style progress: "epoch=174/190" (H2O+) or lower-case variants
    re.compile(
        r'(?:^|[^\w])epoch\s*(?:[:=]\s*|\s+)(\d+)\s*/\s*(\d+)\b',
        re.IGNORECASE,
    ),
    # "Iter 100/1000", "Step 100/1000", "iter 100/1000", "step 100 of 1000"
    re.compile(r'(?:^|\s)(?:Iter|Step|step|iter)[:\s]+(\d+)\s*(?:/|of)\s*(\d+)\b'),
    # "100/1000 done" / "(100/1000)" trailing or in parens
    re.compile(r'(?:^|[\s(])(\d+)\s*/\s*(\d+)\s*(?:done|complete|completed|\))'),
    # batch/eval counters: "[16/103] OK ..." (offline-sumo style)
    re.compile(r'^\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]'),
]


# tqdm pre-computed remaining time. Format: "[<elapsed><<remaining>, <rate>it/s]"
# Examples:
#   "[00:42<03:21, 12.34it/s]"          → remaining=03:21 (m:s)
#   "[1:14:32<5:23:11, 3.21it/s]"       → remaining=5:23:11 (h:m:s)
#   "[02:00<00:00, 1.50s/it]"           → remaining=00:00 (effectively done)
#   "[02:00<?, ?it/s]"                  → remaining='?' (tqdm doesn't know)
_TQDM_ETA_RE = re.compile(
    r'\[\s*(\S+?)\s*<\s*(\S+?)\s*,\s*[\d.?]+\s*(?:'
    + _TQDM_UNIT_RE
    + r'/s|s/'
    + _TQDM_UNIT_RE
    + r')(?:\s*,[^\]]*)?\s*\]'
)

# Explicit ETA in free-form progress lines, e.g.
#   "[16/103] ... (874.7m, ETA 4756.0m)"
_INLINE_ETA_RE = re.compile(
    r'(?:^|[\s,(])ETA\s*[:=]?\s*(\d+(?:\.\d+)?)\s*([smhd])\b',
    re.IGNORECASE,
)

_SECONDS_PER_UNIT_RE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*s\s*/\s*(?:it|iter|iters|step|steps|epoch|epochs|episode|episodes)\b',
    re.IGNORECASE,
)

_LOG_TIMESTAMP_RE = re.compile(
    r'(?<!\d)(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[,.](\d{1,6}))?'
)

_ITER_TIME_RE = re.compile(
    r'(?:^|[^\w])(?:Iter|Iteration|Step|step)\s+(\d+)\b.*?\bTime:\s*(\d+(?:\.\d+)?)\s*s\b',
    re.IGNORECASE,
)

_BAPR_FORK_PROTOCOL_ENTRYPOINTS = (
    "jax_experiments.analysis.run_bapr_v3_budget_matched_fork",
    "jax_experiments.analysis.run_bapr_v3_stochastic_headroom",
    "jax_experiments.analysis.run_bapr_v3_structured_channel_headroom",
)
_BAPR_INDEPENDENT_SPECIALIST_ENTRYPOINTS = (
    "jax_experiments.analysis.launch_bapr_v3_stochastic_independent_specialist",
    "jax_experiments.analysis.run_bapr_v3_independent_specialist",
)
_BAPR_INDEPENDENT_SPECIALIST_FINAL_NEXT_ITERATION = 1400
_BAPR_INDEPENDENT_SPECIALIST_TERMINAL_GAP_S = 120.0
_BAPR_TRAIN_SUBPROCESS_PREFIX = "TRAIN SUBPROCESS:"
_HIDDEN_PROTOCOL_TOTALS_BY_MODULE_PROFILE = {
    (
        "jax_experiments.analysis.run_bapr_v4_persistent_option",
        "formal",
    ): 1400,
    (
        "jax_experiments.analysis.run_bapr_v4_persistent_option",
        "smoke",
    ): 4,
    (
        "jax_experiments.analysis.run_bapr_v5_hard_option",
        "formal",
    ): 1400,
    (
        "jax_experiments.analysis.run_bapr_v5_hard_option",
        "smoke",
    ): 4,
    (
        "jax_experiments.analysis.run_bapr_v6_balanced_option",
        "formal",
    ): 1400,
    (
        "jax_experiments.analysis.run_bapr_v6_balanced_option",
        "smoke",
    ): 4,
    (
        "jax_experiments.analysis.run_bapr_v7_data_equivalent_option",
        "formal",
    ): 5600,
    (
        "jax_experiments.analysis.run_bapr_v7_data_equivalent_option",
        "smoke",
    ): 4,
    (
        "jax_experiments.analysis.run_bapr_v8_seed_controller",
        "",
    ): 1400,
}

_FREQDUET_SHARD_RE = re.compile(r'\bShard jobs \[(\d+),\s*(\d+)\) of (\d+)\b')
_FREQDUET_JOB_START_RE = re.compile(r'--job-start(?:=|\s+)(\d+)\b')
_FREQDUET_JOB_END_RE = re.compile(r'--job-end(?:=|\s+)(\d+)\b')
_FREQDUET_SHARD_NAME_RE = re.compile(r'\bshard[_-](\d+)[_-](\d+)\b')
_FREQDUET_EPISODES_RE = re.compile(r'(?:--episodes(?:=|\s+)|(?:^|\s)EPISODES=)(\d+)\b')
_FREQDUET_EXTERNAL_EP_RE = re.compile(
    r'^(?P<variant>\S+)\s+(?P<config>\S+)\s+seed=(?P<seed>\d+)\s+ep=(?P<ep>\d+)\b'
)
_FREQDUET_ABLATION_EP_RE = re.compile(
    r'^(?P<config>\S+)\s+seed=(?P<seed>\d+)\s+ep=(?P<ep>\d+)\b'
)
_FREQDUET_DONE_PAYLOAD_RE = re.compile(r'^(?:DONE|SKIP)\s+(.+?)(?::\s|$)')
_FREQDUET_DIRECT_EXTERNAL_DONE_RE = re.compile(
    r'^(?:DONE|SKIP)\s+(?P<config>\S+)\s+(?P<variant>\S+)\s+seed=(?P<seed>\d+)\b'
)
_FREQDUET_DIRECT_ABLATION_DONE_RE = re.compile(
    r'^(?:DONE|SKIP)\s+(?P<config>\S+)\s+seed=(?P<seed>\d+)\b'
)
_KG_BENCHMARK_RE = re.compile(r'(?:^|[/\s])benchmark_sota\.py\b')
_KG_N_RE = re.compile(r'--N(?:=|\s+)(\d+)\b')
_BOTORCH_SAASBO_RE = re.compile(
    r'(?:--method(?:=|\s+)|\bmethod=)botorch_saasbo\b',
    re.IGNORECASE,
)
_SAAS_NATIVE_PROGRESS_RE = re.compile(
    r'\bIter\s+(\d+)\s*/\s*(\d+)\s+\[botorch-canonical\]',
    re.IGNORECASE,
)
_ETA_MODEL_RE = re.compile(r'\beta_model=([A-Za-z0-9_+.-]+)\b')
_SCOLHKG_PROGRESS_PREFIX = "SCOLHKG_PROGRESS "
_KG_INNER_PROGRESS_RE = re.compile(
    r'\b(?:Step|Iter)\s+(\d+)\s*/\s*(\d+)\b', re.IGNORECASE)
_KG_INNER_STAGE_RE = re.compile(r'\bstage=(\d+)\s*/\s*(\d+)\b')
_KG_INNER_ELAPSED_RE = re.compile(r'\belapsed=(\d+(?:\.\d+)?)s\b')
_KG_INNER_STEP_ELAPSED_RE = re.compile(r'\bstep_elapsed=(\d+(?:\.\d+)?)s\b')
_KG_INNER_EVAL_RE = re.compile(r'\beval=(\d+(?:\.\d+)?)s\b')
_KG_INNER_DEFAULT_EVAL_INTERVAL = 5
_KG_INNER_TERMINAL_GAP_FLOOR_S = 120.0


def _parse_tqdm_time(s: str) -> Optional[int]:
    """Parse tqdm's "MM:SS" / "HH:MM:SS" / "D:HH:MM:SS" → seconds. '?' → None."""
    if not s or s == '?':
        return None
    parts = s.split(':')
    try:
        if len(parts) == 1:
            return int(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 4:
            return (int(parts[0]) * 86400 + int(parts[1]) * 3600
                    + int(parts[2]) * 60 + int(parts[3]))
    except ValueError:
        return None
    return None


def parse_tqdm_eta(tail_text: str) -> Optional[int]:
    """Extract tqdm's own pre-computed remaining-seconds from the tail.

    tqdm's remaining is more accurate than rate-from-current/elapsed because
    tqdm uses a smoothed (windowed) rate that adapts to phase changes —
    e.g. JAX warmup compile is one slow step then fast steady-state, but
    cumulative-average rate would project wildly long ETAs throughout
    steady-state. tqdm's smoothed rate handles this correctly.

    Returns int seconds (>= 0) or None if no tqdm output found in tail.
    """
    if not tail_text:
        return None
    last = None
    for line in tail_text.splitlines():
        for m in _TQDM_ETA_RE.finditer(line):
            remaining = _parse_tqdm_time(m.group(2))
            if remaining is not None and remaining >= 0:
                last = remaining
    return last


def parse_tqdm_elapsed_remaining(tail_text: str) -> Optional[Tuple[int, int]]:
    """Extract tqdm's own elapsed + remaining seconds from the latest bar line.

    This is for submit-time/local preflight logs. Unlike a scheduler-running
    task, a static preflight log should trust tqdm's internal loop clock:
    `total ~= tqdm_elapsed + tqdm_remaining`.
    """
    if not tail_text:
        return None
    last = None
    for line in tail_text.splitlines():
        for m in _TQDM_ETA_RE.finditer(line):
            elapsed = _parse_tqdm_time(m.group(1))
            remaining = _parse_tqdm_time(m.group(2))
            if elapsed is not None and remaining is not None and elapsed >= 0 and remaining >= 0:
                last = (elapsed, remaining)
    return last


def parse_inline_eta(tail_text: str) -> Optional[int]:
    """Extract explicit free-form ETA like "ETA 4756.0m" from the latest line."""
    if not tail_text:
        return None
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    last = None
    for line in tail_text.splitlines():
        for m in _INLINE_ETA_RE.finditer(line):
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            unit = (m.group(2) or "s").lower()
            if value >= 0 and unit in mult:
                last = int(value * mult[unit])
    return last


def parse_seconds_per_unit(tail_text: str) -> Optional[float]:
    """Extract explicit task-native speed like ``12.0s/iter`` from the latest line."""
    if not tail_text:
        return None
    last = None
    for line in tail_text.splitlines():
        for m in _SECONDS_PER_UNIT_RE.finditer(line):
            try:
                seconds = float(m.group(1))
            except ValueError:
                continue
            if seconds > 0:
                last = seconds
    return last


def iter_time_window_projection(tail_text: str,
                                elapsed_s: float,
                                cmd: Optional[str] = None,
                                min_points: int = 3,
                                max_points: int = 80) -> Optional[dict]:
    """Estimate ETA from recent cumulative ``Iter N ... Time: Xs`` slope.

    Some training logs print a per-line ``s/iter`` for the last iteration only.
    BAPR/RE-SAC periodically do evaluation/checkpoint work, so the latest line
    can swing between ~20s/iter and ~110s/iter. The cumulative Time counter is
    steadier: the recent window slope includes both cheap train iterations and
    periodic expensive iterations.
    """
    if not tail_text or not cmd:
        return None
    total = _extract_total_from_cmd(cmd)
    if not total or total <= 0:
        return None

    points = []
    for line in tail_text.splitlines():
        m = _ITER_TIME_RE.search(line)
        if not m:
            continue
        try:
            current = int(m.group(1))
            time_s = float(m.group(2))
        except ValueError:
            continue
        if current < 0 or current > total or time_s < 0:
            continue
        if points and (current < points[-1][0] or time_s < points[-1][1]):
            points = []
        if points and current == points[-1][0]:
            points[-1] = (current, time_s)
        else:
            points.append((current, time_s))

    if len(points) < max(2, int(min_points)):
        return None
    window = points[-max(2, int(max_points)):]
    first_current, first_time = window[0]
    last_current, last_time = window[-1]
    if last_current <= first_current or last_time <= first_time:
        return None
    if last_current >= total:
        return {
            "source": "iter_time_window",
            "eta_s": 0,
            "total_s": int(max(float(elapsed_s or 0), last_time)),
            "current": int(last_current),
            "total_units": int(total),
            "unit_s": 0.0,
        }

    unit_s = (last_time - first_time) / float(last_current - first_current)
    if unit_s <= 0:
        return None
    eta_s = int(max(0, (float(total) - float(last_current)) * unit_s))
    elapsed = max(0.0, float(elapsed_s or 0))
    total_s = int(max(elapsed, elapsed + eta_s, last_time + eta_s))
    return {
        "source": "iter_time_window",
        "eta_s": eta_s,
        "total_s": total_s,
        "current": int(last_current),
        "total_units": int(total),
        "unit_s": float(unit_s),
    }


def _command_arg(command: str, flag: str) -> Optional[str]:
    try:
        tokens = shlex.split(command or "")
    except Exception:
        tokens = (command or "").split()
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return str(tokens[index + 1])
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _recent_iter_window(tail_text: str, *, max_points: int = 80) -> Optional[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for line in (tail_text or "").splitlines():
        match = _ITER_TIME_RE.search(line)
        if not match:
            continue
        try:
            current = int(match.group(1))
            cumulative_s = float(match.group(2))
        except ValueError:
            continue
        if current < 0 or cumulative_s < 0:
            continue
        if points and (
            current < points[-1][0] or cumulative_s < points[-1][1]
        ):
            points = []
        if points and current == points[-1][0]:
            points[-1] = (current, cumulative_s)
        else:
            points.append((current, cumulative_s))
    window = points[-max(3, int(max_points)):]
    if len(window) < 3:
        return None
    first_current, first_time = window[0]
    last_current, last_time = window[-1]
    if last_current <= first_current or last_time <= first_time:
        return None
    return last_current, (last_time - first_time) / float(last_current - first_current)


def kg_inner_progress_projection(
    tail_text: str,
    elapsed_s: float,
    cmd: Optional[str] = None,
) -> Optional[dict]:
    """Project SC-OLH-KG from completed inner-stage wall times.

    The task-native ETA divides the whole inner-loop clock by the number of
    completed post-initial-design stages.  That clock also contains one-time
    initialization, so early estimates replay setup on every future stage (for
    example, 585 seconds over two stages becomes a spurious 2340-second ETA).
    ``step_elapsed`` is the actual stage cost.  Model its regular and periodic
    evaluation portions separately, then subtract time already spent in the
    currently active stage.
    """
    records = []
    for raw_line in (tail_text or "").splitlines():
        line = raw_line.strip()
        if "[kg-inner]" not in line or "kind=iteration_done" not in line:
            continue
        progress_match = _KG_INNER_PROGRESS_RE.search(line)
        stage_match = _KG_INNER_STAGE_RE.search(line)
        elapsed_match = _KG_INNER_ELAPSED_RE.search(line)
        step_match = _KG_INNER_STEP_ELAPSED_RE.search(line)
        if not all((progress_match, stage_match, elapsed_match, step_match)):
            continue
        try:
            current = int(progress_match.group(1))
            total_units = int(progress_match.group(2))
            stage = int(stage_match.group(1))
            total_stages = int(stage_match.group(2))
            run_elapsed_s = float(elapsed_match.group(1))
            step_elapsed_s = float(step_match.group(1))
            eval_match = _KG_INNER_EVAL_RE.search(line)
            eval_elapsed_s = float(eval_match.group(1)) if eval_match else 0.0
        except (TypeError, ValueError):
            continue
        if (
            total_units <= 0
            or current < 0
            or current > total_units
            or total_stages <= 0
            or stage < 0
            or stage >= total_stages
            or run_elapsed_s < 0
            or step_elapsed_s <= 0
            or eval_elapsed_s < 0
        ):
            continue
        record = {
            "current": current,
            "total_units": total_units,
            "stage": stage,
            "total_stages": total_stages,
            "run_elapsed_s": run_elapsed_s,
            "step_elapsed_s": step_elapsed_s,
            "eval_elapsed_s": min(eval_elapsed_s, step_elapsed_s),
        }
        if records and (
            stage <= records[-1]["stage"]
            or current <= records[-1]["current"]
            or run_elapsed_s < records[-1]["run_elapsed_s"]
        ):
            # ETA snapshots concatenate a progress grep and a byte tail, and a
            # resumed task can also restart its inner clock. Keep only the most
            # recent monotonic segment in either case.
            records = []
        records.append(record)

    if not records:
        return None
    latest = records[-1]
    remaining_stages = max(0, latest["total_stages"] - latest["stage"] - 1)
    window = records[-10:]
    base_steps = [
        record["step_elapsed_s"] - record["eval_elapsed_s"]
        for record in window
        if record["step_elapsed_s"] - record["eval_elapsed_s"] > 0
    ]
    if not base_steps:
        return None
    base_step_s = float(statistics.median(base_steps[-6:]))
    eval_steps = [
        record["eval_elapsed_s"]
        for record in window
        if record["eval_elapsed_s"] > 0
    ]
    eval_step_s = float(statistics.median(eval_steps)) if eval_steps else 0.0

    try:
        n0 = int(_command_arg(cmd or "", "--n0") or records[0]["stage"])
    except ValueError:
        n0 = int(records[0]["stage"])
    interval_arg = (
        _command_arg(cmd or "", "--evaluate-interval")
        or _command_arg(cmd or "", "--evaluate_interval")
    )
    try:
        eval_interval = int(interval_arg) if interval_arg is not None else None
    except ValueError:
        eval_interval = None
    observed_eval_stages = [
        int(record["stage"])
        for record in records
        if record["eval_elapsed_s"] > 0
    ]
    if eval_interval is None and len(observed_eval_stages) >= 2:
        gaps = [
            right - left
            for left, right in zip(observed_eval_stages, observed_eval_stages[1:])
            if right > left
        ]
        if gaps:
            eval_interval = max(1, int(round(statistics.median(gaps))))
    if eval_interval is None:
        eval_interval = _KG_INNER_DEFAULT_EVAL_INTERVAL if eval_steps else 0

    future_stage_ids = range(latest["stage"] + 1, latest["total_stages"])
    if eval_interval > 0 and eval_step_s > 0:
        future_eval_stages = {
            stage
            for stage in future_stage_ids
            if (stage - n0) % eval_interval == 0
            or stage == latest["total_stages"] - 1
        }
    else:
        future_eval_stages = set()
    future_eval_count = len(future_eval_stages)
    modeled_remaining_s = (
        base_step_s * float(remaining_stages)
        + eval_step_s * float(future_eval_count)
    )

    elapsed = max(0.0, float(elapsed_s or 0.0))
    # Step N/N is not terminal for this workload. It still performs the final
    # posterior recommendation, diagnostics, result serialization, process
    # exit, and watcher reconciliation. Across completed production KG runs,
    # ``scheduler_duration - final_inner_elapsed`` is roughly 3 minutes and
    # grows with expensive evaluation phases. Project the full wall runtime
    # directly so time spent in the active stage or terminal tail naturally
    # counts down instead of resetting ETA to zero at Step N/N.
    terminal_gap_s = max(
        _KG_INNER_TERMINAL_GAP_FLOOR_S,
        2.0 * eval_step_s,
    )
    projected_total_s = (
        float(latest["run_elapsed_s"])
        + modeled_remaining_s
        + terminal_gap_s
    )
    eta_s = int(max(0.0, projected_total_s - elapsed))
    total_s = int(max(elapsed, projected_total_s))
    inline_eta = parse_inline_eta(tail_text or "")
    accounted_step_s = sum(record["step_elapsed_s"] for record in records)
    one_time_setup_s = max(0.0, latest["run_elapsed_s"] - accounted_step_s)
    return {
        "source": "kg_inner_progress",
        "eta_s": eta_s,
        "total_s": total_s,
        "current": int(latest["current"]),
        "total_units": int(latest["total_units"]),
        "unit_s": float(total_s) / float(latest["total_units"]),
        "stage": int(latest["stage"]),
        "total_stages": int(latest["total_stages"]),
        "base_step_s": base_step_s,
        "eval_step_s": eval_step_s,
        "eval_interval": int(eval_interval),
        "future_eval_count": int(future_eval_count),
        "terminal_gap_s": float(terminal_gap_s),
        "one_time_setup_s": float(one_time_setup_s),
        "inline_eta_s": inline_eta,
    }


def bapr_fork_protocol_projection(
    tail_text: str,
    elapsed_s: float,
    cmd: Optional[str] = None,
) -> Optional[dict]:
    """Project the three-stage BAPR fork protocol from its active subprocess.

    The outer protocol command does not expose ``--max_iters``. It trains a
    shared base, then two equal-budget branches, and logs each concrete child
    command with ``TRAIN SUBPROCESS:``. Account for the not-yet-started branch
    instead of reporting only the active child's remaining time.
    """
    outer_cmd = cmd or ""
    if not any(entrypoint in outer_cmd for entrypoint in _BAPR_FORK_PROTOCOL_ENTRYPOINTS):
        return None

    train_commands = [
        line.split(_BAPR_TRAIN_SUBPROCESS_PREFIX, 1)[1].strip()
        for line in (tail_text or "").splitlines()
        if line.startswith(_BAPR_TRAIN_SUBPROCESS_PREFIX)
    ]
    if not train_commands:
        return None
    active_command = train_commands[-1]
    stage = (_command_arg(active_command, "--run_name") or "").strip()
    if stage not in {"shared_base", "robust_long", "oracle_direct"}:
        return None
    try:
        stage_end = int(_command_arg(active_command, "--max_iters") or 0)
        stage_start = int(_command_arg(active_command, "--min_resume_iteration") or 0)
    except ValueError:
        return None
    if stage_end <= stage_start:
        return None

    recent = _recent_iter_window(tail_text)
    if recent is None:
        return None
    stage_current, unit_s = recent
    if unit_s <= 0 or stage_current < stage_start:
        return None

    if stage == "shared_base":
        base_units = stage_end
        branch_units = base_units
        stage_start = 0
        completed_before_stage = 0
        future_stages = 2
    else:
        base_units = stage_start
        branch_units = stage_end - stage_start
        completed_before_stage = base_units
        future_stages = 1
        if stage == "oracle_direct":
            completed_before_stage += branch_units
            future_stages = 0

    protocol_total = base_units + 2 * branch_units
    stage_progress = max(0, min(stage_current, stage_end) - stage_start)
    protocol_current = min(protocol_total, completed_before_stage + stage_progress)
    eta_s = int(max(0.0, (protocol_total - protocol_current) * unit_s))
    elapsed = max(0.0, float(elapsed_s or 0.0))
    return {
        "source": "bapr_fork_protocol",
        "eta_s": eta_s,
        "total_s": int(max(elapsed, elapsed + eta_s)),
        "current": int(protocol_current),
        "total_units": int(protocol_total),
        "unit_s": float(unit_s),
        "protocol_stage": stage,
        "stage_current": int(stage_current),
        "stage_start": int(stage_start),
        "stage_end": int(stage_end),
        "future_stages": int(future_stages),
    }


def bapr_independent_specialist_projection(
    tail_text: str,
    elapsed_s: float,
    cmd: Optional[str] = None,
) -> Optional[dict]:
    """Project a BAPR independent-specialist wrapper's hidden train budget.

    These launchers expose only ``--target-next-iteration`` on the outer
    command and normally omit it because the protocol default is 1400.  The
    child ``--max_iters`` command is written to a separate workload log, so a
    generic scheduler sees ``Iter N`` without a total and otherwise reports an
    unknown ``elapsed+`` ETA.  Use the protocol budget plus a recent cumulative
    ``Time`` slope; the window includes periodic evaluation and checkpoint
    iterations that the latest ``s/iter`` value alone misses.
    """
    outer_cmd = cmd or ""
    if not any(
        entrypoint in outer_cmd
        for entrypoint in _BAPR_INDEPENDENT_SPECIALIST_ENTRYPOINTS
    ):
        return None

    target_arg = _command_arg(outer_cmd, "--target-next-iteration")
    try:
        target = int(target_arg) if target_arg is not None else (
            _BAPR_INDEPENDENT_SPECIALIST_FINAL_NEXT_ITERATION
        )
    except ValueError:
        return None
    if target <= 0:
        return None

    recent = _recent_iter_window(tail_text, max_points=120)
    if recent is not None:
        last_iteration, unit_s = recent
        rate_source = "recent_time_window"
    else:
        last_iteration = _extract_current_only_from_tail(tail_text)
        unit_s = parse_seconds_per_unit(tail_text)
        rate_source = "last_seconds_per_iter"
    if last_iteration is None or unit_s is None or unit_s <= 0:
        return None
    if last_iteration < 0 or last_iteration >= target:
        return None

    # The workload logs the zero-based iteration just completed; its target is
    # the exclusive next-iteration boundary used by range(..., max_iters).
    completed_next_iteration = min(target, int(last_iteration) + 1)
    remaining_train_units = max(0, target - completed_next_iteration)
    try:
        tokens = shlex.split(outer_cmd)
    except Exception:
        tokens = outer_cmd.split()
    terminal_gap_s = (
        30.0
        if "--smoke" in tokens
        else _BAPR_INDEPENDENT_SPECIALIST_TERMINAL_GAP_S
    )
    eta_s = int(max(
        0.0,
        float(remaining_train_units) * float(unit_s) + terminal_gap_s,
    ))
    elapsed = max(0.0, float(elapsed_s or 0.0))
    return {
        "source": "bapr_independent_specialist",
        "eta_s": eta_s,
        "total_s": int(max(elapsed, elapsed + eta_s)),
        "current": int(completed_next_iteration),
        "total_units": int(target),
        "unit_s": float(unit_s),
        "last_iteration": int(last_iteration),
        "remaining_train_units": int(remaining_train_units),
        "terminal_gap_s": float(terminal_gap_s),
        "rate_source": rate_source,
    }


def growing_iter_cost_projection(tail_text: str,
                                 elapsed_s: float,
                                 cmd: Optional[str] = None,
                                 min_points: int = 12,
                                 max_segments: int = 24) -> Optional[dict]:
    """Project SAASBO runtime when model-fit cost grows with observations.

    Canonical SAASBO refits a fully Bayesian GP after every new observation.
    Its per-iteration cost therefore grows with the training-set size.  The
    benchmark's inline ETA is a cumulative-average projection and can remain
    almost constant for hours.  Fit a non-negative linear trend to smoothed
    per-iteration costs and integrate that trend over the remaining calls.

    This intentionally applies only to commands that explicitly select
    ``botorch_saasbo``.  Generic task-native ETA remains authoritative for all
    other workloads.
    """
    if not tail_text or not cmd or not _BOTORCH_SAASBO_RE.search(cmd):
        return None
    progress = parse_progress(tail_text, cmd=cmd)
    if progress is None:
        return None
    current, total = progress
    if current <= 0 or current > total:
        return None

    points = []
    for line in tail_text.splitlines():
        match = _ITER_TIME_RE.search(line)
        if not match:
            continue
        try:
            unit = int(match.group(1))
            cumulative_s = float(match.group(2))
        except ValueError:
            continue
        if unit < 0 or unit > total or cumulative_s < 0:
            continue
        if points and (unit < points[-1][0] or cumulative_s < points[-1][1]):
            points = []
        if points and unit == points[-1][0]:
            points[-1] = (unit, cumulative_s)
        else:
            points.append((unit, cumulative_s))

    if len(points) < 2:
        return None
    last_unit, last_time = points[-1]
    if last_unit != current:
        current = last_unit
    if current <= 0 or current > total:
        return None

    # Initial-design simulations are nearly free in the synthetic gates.
    # Their near-zero cost must not be regressed against the later NUTS
    # refits: doing so turned a four-hour remainder into a 10-100 hour ETA.
    n0_arg = _command_arg(cmd or "", "--n0")
    try:
        adaptive_floor = max(0, int(n0_arg)) if n0_arg is not None else 0
    except ValueError:
        adaptive_floor = 0
    fit_points = [
        point for point in points if int(point[0]) >= adaptive_floor
    ]
    if len(fit_points) < 2:
        return None
    adjacent_rates = []
    for left, right in zip(fit_points, fit_points[1:]):
        delta_units = int(right[0]) - int(left[0])
        delta_s = float(right[1]) - float(left[1])
        if delta_units > 0 and delta_s > 0:
            adjacent_rates.append(delta_s / float(delta_units))
    if not adjacent_rates:
        return None
    recent_rate = statistics.median(
        adjacent_rates[-min(5, len(adjacent_rates)):])
    remaining_refits = float(max(0, total - current) + 1)

    def recent_projection() -> dict:
        eta_s = int(max(0.0, recent_rate * remaining_refits))
        anchor_elapsed = max(float(elapsed_s or 0.0), float(last_time))
        return {
            "source": "recent_adaptive_cost_plus_terminal",
            "eta_s": eta_s,
            "total_s": int(max(anchor_elapsed, anchor_elapsed + eta_s)),
            "current": int(current),
            "total_units": int(total),
            "unit_s": float(recent_rate),
            "remaining_refits": int(remaining_refits),
            "terminal_refits": 1,
        }

    if len(fit_points) < max(4, int(min_points)):
        return recent_projection()
    span_units = fit_points[-1][0] - fit_points[0][0]
    if span_units < max(6, int(0.05 * total)):
        return recent_projection()

    stride = max(2, len(fit_points) // max(4, int(max_segments)))
    rates = []
    for start in range(0, len(fit_points) - stride, stride):
        left = fit_points[start]
        right = fit_points[min(len(fit_points) - 1, start + stride)]
        delta_units = right[0] - left[0]
        delta_s = right[1] - left[1]
        if delta_units <= 0 or delta_s <= 0:
            continue
        rates.append(((left[0] + right[0]) / 2.0, delta_s / delta_units))
    if len(rates) < 4:
        return recent_projection()

    rate_center = statistics.median(y for _, y in rates)
    robust_rates = [
        (x, y) for x, y in rates
        if y <= max(1e-9, 3.0 * rate_center)
    ]
    if len(robust_rates) >= 4:
        rates = robust_rates

    # NUTS jobs are bursty under node contention.  Ordinary least squares
    # turns one stalled fit into an enormous positive slope and then integrates
    # that scheduling accident over every remaining BO call.  The Theil-Sen
    # slope plus median intercept preserves genuine smooth growth while being
    # insensitive to a minority of delayed iterations.
    pair_slopes = []
    for left_index, (left_x, left_y) in enumerate(rates):
        for right_x, right_y in rates[left_index + 1:]:
            delta_x = right_x - left_x
            if delta_x > 0:
                pair_slopes.append((right_y - left_y) / delta_x)
    if not pair_slopes:
        return recent_projection()
    slope = max(0.0, statistics.median(pair_slopes))
    intercept = statistics.median(
        y - slope * x for x, y in rates
    )
    recent_rate = statistics.median(
        adjacent_rates[-min(5, len(adjacent_rates)):])
    fitted_current_rate = intercept + slope * float(current)
    fitted_current_rate = max(1e-9, fitted_current_rate)
    # A short-lived resource-contention regime may make every point in the
    # small recent window slow.  Permit a 1.5x slowdown in the projection, but do
    # not reinterpret a 10-40x jump as intrinsic GP complexity growth.
    recent_rate_cap = 1.5 * fitted_current_rate
    recent_rate_was_capped = recent_rate > recent_rate_cap
    current_rate = max(
        fitted_current_rate,
        min(recent_rate, recent_rate_cap),
    )

    # Keep a noisy tail from extrapolating a pathological growth curve.  This
    # cap still permits quadratic cumulative runtime, the expected upper shape
    # for the observed canonical SAASBO runs.
    slope = min(slope, 2.0 * current_rate / max(1.0, float(current)))
    remaining_units = remaining_refits
    eta_s = int(max(
        0.0,
        current_rate * remaining_units
        + 0.5 * slope * remaining_units * remaining_units,
    ))
    anchor_elapsed = max(float(elapsed_s or 0.0), float(last_time))
    return {
        "source": "growing_iter_cost",
        "eta_s": eta_s,
        "total_s": int(max(anchor_elapsed, anchor_elapsed + eta_s)),
        "current": int(current),
        "total_units": int(total),
        "unit_s": float(current_rate),
        "unit_growth_s": float(slope),
        "remaining_refits": int(remaining_refits),
        "terminal_refits": 1,
        "recent_rate_s": float(recent_rate),
        "recent_rate_was_capped": bool(recent_rate_was_capped),
    }


def saas_native_eta_projection(tail_text: str,
                               elapsed_s: float,
                               cmd: Optional[str] = None) -> Optional[dict]:
    """Use the benchmark's current-run SAAS projection when it is mature.

    The benchmark estimates canonical SAAS cost from completed refits in this
    run. Replacing that value with a contention-trimmed fit can make a real
    20-40 hour remainder look like 3-5 hours. History remains appropriate
    before the first adaptive refit, but once the log has a positive
    post-initial-design ETA, the task-native estimate is the primary signal.
    """
    if not tail_text or not cmd or not _BOTORCH_SAASBO_RE.search(cmd):
        return None
    try:
        n0 = max(0, int(_command_arg(cmd, "--n0") or 0))
    except ValueError:
        n0 = 0
    latest = None
    for line in tail_text.splitlines():
        progress_match = _SAAS_NATIVE_PROGRESS_RE.search(line)
        eta_match = _INLINE_ETA_RE.search(line)
        model_match = _ETA_MODEL_RE.search(line)
        if not all((progress_match, eta_match, model_match)):
            continue
        try:
            current = int(progress_match.group(1))
            total = int(progress_match.group(2))
            eta_value = float(eta_match.group(1))
        except (TypeError, ValueError):
            continue
        unit = (eta_match.group(2) or "s").lower()
        multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit)
        if multiplier is None:
            continue
        eta_s = eta_value * multiplier
        if (
            total <= 0
            or current <= n0
            or current >= total
            or eta_s <= 0
        ):
            continue
        latest = {
            "current": current,
            "total_units": total,
            "eta_s": int(eta_s),
            "eta_model": model_match.group(1),
        }
    if latest is None:
        return None
    eta_s = int(latest["eta_s"])
    remaining_units = max(
        1, int(latest["total_units"]) - int(latest["current"])
    )
    elapsed = max(0.0, float(elapsed_s or 0.0))
    return {
        "source": "saas_native_eta",
        "eta_s": eta_s,
        "total_s": int(max(elapsed, elapsed + eta_s)),
        "current": int(latest["current"]),
        "total_units": int(latest["total_units"]),
        "unit_s": float(eta_s) / float(remaining_units),
        "eta_model": str(latest["eta_model"]),
    }


def scolhkg_progress_projection(tail_text: str,
                                elapsed_s: float,
                                cmd: Optional[str] = None) -> Optional[dict]:
    """Project transfer-benchmark ETA from its structured progress records."""
    latest = None
    for raw_line in (tail_text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith(_SCOLHKG_PROGRESS_PREFIX):
            continue
        try:
            payload = json.loads(line[len(_SCOLHKG_PROGRESS_PREFIX):])
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            done = int(payload.get("done"))
            total = int(payload.get("total"))
        except (TypeError, ValueError):
            continue
        if total <= 0 or done < 0 or done > total:
            continue
        latest = (payload, done, total)
    if latest is None:
        return None

    payload, done, total = latest
    if done >= total:
        eta_s = 0
    else:
        try:
            explicit_eta = float(payload.get("eta_seconds"))
        except (TypeError, ValueError):
            explicit_eta = -1.0
        if explicit_eta >= 0:
            eta_s = int(explicit_eta)
        else:
            try:
                phase_elapsed = float(payload.get("phase_elapsed_s"))
            except (TypeError, ValueError):
                phase_elapsed = 0.0
            method = str(payload.get("method") or "").lower()
            # Old FSBO logs expose only target-call counts after a large hidden
            # source-training phase.  Using whole-process elapsed per target
            # call would inflate the remaining ETA by orders of magnitude; let
            # runtime history handle those legacy records instead.
            if phase_elapsed <= 0 and method == "fsbo_cbo":
                return None
            rate_elapsed = phase_elapsed if phase_elapsed > 0 else float(elapsed_s or 0)
            if done <= 0 or rate_elapsed <= 0:
                return None
            eta_s = int(max(0.0, rate_elapsed / float(done) * float(total - done)))

    elapsed = max(0.0, float(elapsed_s or 0))
    return {
        "source": "scolhkg_progress",
        "eta_s": int(eta_s),
        "total_s": int(max(elapsed, elapsed + eta_s)),
        "current": int(done),
        "total_units": int(total),
        "unit_s": (
            float(eta_s) / float(total - done)
            if total > done else 0.0
        ),
        "progress_kind": str(payload.get("kind") or ""),
        "progress_label": str(payload.get("label") or ""),
    }


def _extract_freqduet_shard_span(tail_text: str, cmd: Optional[str]) -> Optional[Tuple[int, int]]:
    text = "\n".join(part for part in (tail_text or "", cmd or "") if part)
    last_span = None
    for m in _FREQDUET_SHARD_RE.finditer(text):
        try:
            start = int(m.group(1))
            end = int(m.group(2))
        except ValueError:
            continue
        if end >= start:
            last_span = (start, end)
    if last_span is not None:
        return last_span

    start_m = _FREQDUET_JOB_START_RE.search(cmd or "")
    end_m = _FREQDUET_JOB_END_RE.search(cmd or "")
    if start_m and end_m:
        try:
            start = int(start_m.group(1))
            end = int(end_m.group(1))
            if end >= start:
                return (start, end)
        except ValueError:
            pass

    for m in _FREQDUET_SHARD_NAME_RE.finditer(text):
        try:
            start = int(m.group(1))
            end = int(m.group(2))
        except ValueError:
            continue
        if end >= start:
            last_span = (start, end)
    return last_span


def _extract_freqduet_episodes(cmd: Optional[str]) -> int:
    if not cmd:
        return 100
    m = _FREQDUET_EPISODES_RE.search(cmd)
    if not m:
        return 100
    try:
        return max(1, int(m.group(1)))
    except ValueError:
        return 100


def freqduet_shard_projection(tail_text: str,
                              elapsed_s: float,
                              cmd: Optional[str] = None) -> Optional[dict]:
    """Project FreqDuet shard ETA from DONE/SKIP count plus active episode progress."""
    haystack = f"{tail_text or ''}\n{cmd or ''}".lower()
    if "freqduet" not in haystack and "run_freqduet_" not in haystack:
        return None
    span = _extract_freqduet_shard_span(tail_text, cmd)
    if span is None:
        return None
    start, end = span
    total_jobs = max(0, int(end) - int(start))
    if total_jobs <= 0:
        return None

    episodes = _extract_freqduet_episodes(cmd)
    completed_payloads = set()
    completed_keys = set()
    active = {}

    for raw_line in (tail_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        payload_m = _FREQDUET_DONE_PAYLOAD_RE.match(line)
        if payload_m:
            completed_payloads.add(payload_m.group(1).strip())
            m = _FREQDUET_DIRECT_EXTERNAL_DONE_RE.match(line)
            if m:
                completed_keys.add((
                    "external",
                    m.group("config"),
                    m.group("variant"),
                    int(m.group("seed")),
                ))
            else:
                m = _FREQDUET_DIRECT_ABLATION_DONE_RE.match(line)
                if m:
                    completed_keys.add(("ablation", m.group("config"), "", int(m.group("seed"))))
            continue

        m = _FREQDUET_EXTERNAL_EP_RE.match(line)
        if m and m.group("variant") not in {"RUN", "DONE", "SKIP", "ERR", "Shard"}:
            try:
                key = ("external", m.group("config"), m.group("variant"), int(m.group("seed")))
                ep = int(m.group("ep"))
            except ValueError:
                continue
            active[key] = max(active.get(key, -1), ep)
            continue

        m = _FREQDUET_ABLATION_EP_RE.match(line)
        if m and m.group("config") not in {"RUN", "DONE", "SKIP", "ERR", "Shard"}:
            try:
                key = ("ablation", m.group("config"), "", int(m.group("seed")))
                ep = int(m.group("ep"))
            except ValueError:
                continue
            active[key] = max(active.get(key, -1), ep)

    # Parallel external mode prints "DONE {run_dir.name}: ..." where run_dir.name
    # is "{config}_{variant}_seed{seed}". Map those names back onto active keys
    # so completed jobs do not also contribute fractional active progress.
    for payload in completed_payloads:
        name = payload.split()[0] if payload else ""
        for mode, config, variant, seed in active:
            if mode == "external":
                expected = f"{config}_{variant}_seed{seed}"
            else:
                expected = f"{config}_seed{seed}"
            if name == expected:
                completed_keys.add((mode, config, variant, seed))

    completed_jobs = min(total_jobs, len(completed_payloads))
    active_fraction = 0.0
    active_jobs = 0
    for key, ep in active.items():
        if key in completed_keys:
            continue
        active_jobs += 1
        active_fraction += min(1.0, max(0.0, (float(ep) + 1.0) / float(episodes)))

    progress = min(float(total_jobs), float(completed_jobs) + active_fraction)
    if progress <= 0:
        return None

    elapsed = max(1.0, float(elapsed_s or 0))
    unit_s = elapsed / progress
    eta_s = int(max(0, (float(total_jobs) - progress) * unit_s))
    total_s = int(max(elapsed, unit_s * float(total_jobs)))
    return {
        "source": "freqduet_shard",
        "eta_s": eta_s,
        "total_s": total_s,
        "current": int(progress),
        "total_units": int(total_jobs),
        "unit_s": float(unit_s),
        "completed_jobs": int(completed_jobs),
        "active_jobs": int(active_jobs),
    }


def parse_progress(tail_text: str, cmd: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """Walk all patterns over every line of tail_text. Return the LATEST
    (current, total) found. None if nothing matches.

    "Latest" = most recent line containing a progress pattern. tqdm rewrites
    the same line repeatedly so we want the LAST line's number, not the first.

    Tier 2 fallback (when cmd is provided): some training scripts log only
    `Iter N` / `step N` / `Epoch N` with no total. Extract the total from the
    cmd's `--max_iters N` / `--n_epochs N` / `--max_steps N` / `--epochs N` /
    `--total_steps N` flag and pair with the latest current-only marker in the
    tail. Without this fallback ETAs would be 0 for any framework that doesn't
    print N/M (RE-SAC, many torch examples, hand-rolled training loops).
    """
    if not tail_text:
        # Empty tail can still benefit from cmd parsing — but no current-step
        # signal means no rate, so still None. The caller's EWMA fallback owns
        # the no-tail case.
        return None
    last = None
    # Tier 1: full-form patterns (current AND total in the same line)
    for line in tail_text.splitlines():
        for pat in _ETA_PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            groups = m.groups()
            try:
                current = int(groups[0])
                total = int(groups[1])
            except (ValueError, IndexError):
                continue
            if total <= 0 or current < 0 or current > total:
                continue
            last = (current, total)
            break
    if last is not None:
        return last

    # Tier 2: tail has "Iter N" / "step N" / "Epoch N" alone, cmd has --max_iters / etc.
    if cmd:
        total = _extract_total_from_cmd(cmd)
        if total and total > 0:
            current = _extract_current_only_from_tail(tail_text)
            if current is not None and 0 <= current <= total:
                return (current, total)
    return None


def _log_timestamp_seconds(line: str) -> Optional[float]:
    match = _LOG_TIMESTAMP_RE.search(line or "")
    if not match:
        return None
    try:
        stamp = datetime.strptime(match.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S")
        fraction = (match.group(2) or "").ljust(6, "0")[:6]
        return float(stamp.toordinal() * 86400 + stamp.hour * 3600 + stamp.minute * 60
                     + stamp.second) + (int(fraction or 0) / 1_000_000.0)
    except (TypeError, ValueError):
        return None


def timestamped_progress_window_projection(
    tail_text: str,
    elapsed_s: float,
    cmd: Optional[str] = None,
    *,
    min_points: int = 3,
    max_points: int = 30,
) -> Optional[dict]:
    """Project ETA from recent timestamped ``current/total`` log records.

    This path is intended for ordinary logging output such as H2O+'s
    ``2026-... INFO epoch=174/190``.  A recent wall-clock slope reflects
    checkpoint and evaluation pauses while avoiding one-time startup cost.
    """
    del cmd  # Kept for the same projection-call interface as other estimators.
    points: list[tuple[int, int, float]] = []
    for line in (tail_text or "").splitlines():
        timestamp_s = _log_timestamp_seconds(line)
        progress = parse_progress(line)
        if timestamp_s is None or progress is None:
            continue
        current, total = progress
        if total <= 0 or current < 0 or current > total:
            continue
        if points and (
            total != points[-1][1]
            or current < points[-1][0]
            or timestamp_s < points[-1][2]
        ):
            points = []
        if points and current == points[-1][0]:
            points[-1] = (current, total, timestamp_s)
        else:
            points.append((current, total, timestamp_s))

    window = points[-max(2, int(max_points)):]
    if len(window) < max(2, int(min_points)):
        return None
    first_current, _first_total, first_timestamp = window[0]
    current, total, timestamp = window[-1]
    completed = current - first_current
    elapsed_window = timestamp - first_timestamp
    if completed < 2 or elapsed_window <= 0:
        return None
    unit_s = elapsed_window / float(completed)
    eta_s = int(max(0.0, (float(total) - float(current)) * unit_s))
    elapsed = max(0.0, float(elapsed_s or 0.0))
    return {
        "source": "progress_window",
        "eta_s": eta_s,
        "total_s": int(max(elapsed, elapsed + eta_s)),
        "current": int(current),
        "total_units": int(total),
        "unit_s": float(unit_s),
        "window_points": len(window),
        "window_elapsed_s": float(elapsed_window),
    }


# Cmd-line flags that declare the task's total step/iter count. Order doesn't
# matter; we take the first match. Common across PyTorch, JAX, scikit, etc.
_SCHEDULEURM_ETA_TOTAL_RE = re.compile(
    r'\bSCHEDULEURM_ETA_TOTAL_UNITS=(\d+)\b')
_CMD_TOTAL_PATTERNS = [
    re.compile(r'--max[_-]?iters[=\s]+(\d+)'),
    re.compile(r'--n[_-]?epochs[=\s]+(\d+)'),
    re.compile(r'--num[_-]?epochs[=\s]+(\d+)'),
    re.compile(r'--epochs[=\s]+(\d+)'),
    re.compile(r'--max[_-]?steps[=\s]+(\d+)'),
    re.compile(r'--total[_-]?steps[=\s]+(\d+)'),
    re.compile(r'--n[_-]?steps[=\s]+(\d+)'),
    re.compile(r'--num[_-]?steps[=\s]+(\d+)'),
    re.compile(r'--num[_-]?iters[=\s]+(\d+)'),
    re.compile(r'--iterations[=\s]+(\d+)'),
]


def _extract_total_from_cmd(cmd: str) -> Optional[int]:
    """Look for --max_iters / --n_epochs / --max_steps / etc. flags in the cmd
    string. Returns the first match as int, or None if no recognized flag."""
    if not cmd:
        return None
    marker = _SCHEDULEURM_ETA_TOTAL_RE.search(cmd)
    if marker:
        try:
            value = int(marker.group(1))
            if value > 0:
                return value
        except ValueError:
            pass
    if _KG_BENCHMARK_RE.search(cmd):
        m = _KG_N_RE.search(cmd)
        if m:
            try:
                v = int(m.group(1))
                if v > 0:
                    return v
            except ValueError:
                pass
    for pat in _CMD_TOTAL_PATTERNS:
        m = pat.search(cmd)
        if m:
            try:
                v = int(m.group(1))
                if v > 0:
                    return v
            except ValueError:
                pass
    module = _command_arg(cmd, "-m")
    profile = _command_arg(cmd, "--profile")
    hidden_total = _HIDDEN_PROTOCOL_TOTALS_BY_MODULE_PROFILE.get(
        (module or "", profile or "")
    )
    if hidden_total:
        return hidden_total
    # BAPR / RE-SAC helper scripts use positional args:
    #   ./run_seed.sh <algo> <env> <seed> <max_iters> <dwell> ...
    # The second integer after the script is the total iteration count.
    try:
        import os as _os
        import shlex as _shlex
        toks = _shlex.split(cmd or "")
        for i, tok in enumerate(toks):
            if _os.path.basename(tok) != "run_seed.sh":
                continue
            ints = []
            for part in toks[i + 1:]:
                if part.isdigit():
                    ints.append(int(part))
                    if len(ints) >= 2:
                        return ints[1] if ints[1] > 0 else None
            break
    except Exception:
        pass
    return None


# Per-line "current step only" patterns — anchored to common log formats.
# We keep these conservative to avoid false positives from random integers
# in the log (e.g. "Reward: 4739.1" wouldn't match because Reward isn't in
# the prefix list).
_CURRENT_ONLY_PATTERNS = [
    re.compile(r'(?:^|[^\w])Iter\s+(\d+)(?:\s|$|[|,])'),
    re.compile(r'(?:^|[^\w])Iteration\s+(\d+)(?:\s|$|[|,])'),
    re.compile(r'(?:^|[^\w])Epoch\s+(\d+)(?:\s|$|[|,:])'),
    re.compile(r'(?:^|[^\w])Step\s+(\d+)(?:\s|$|[|,])'),
    re.compile(r'(?:^|[^\w])step\s+(\d+)(?:\s|$|[|,])'),
]


def _extract_current_only_from_tail(tail_text: str) -> Optional[int]:
    """Find the LATEST `Iter N` / `Epoch N` / `Step N` in the tail that doesn't
    have a `/total` immediately after. Returns int current or None.

    Why not also check tqdm "47%|" alone? Because tqdm always shows total
    alongside (its progress bar is meaningless without it); tasks using tqdm
    are already covered by tier-1 patterns. The current-only fallback is for
    hand-rolled "Iter N" loggers that DON'T do tqdm at all (RE-SAC, many
    JAX experiments).
    """
    last = None
    for line in tail_text.splitlines():
        for pat in _CURRENT_ONLY_PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            try:
                current = int(m.group(1))
            except ValueError:
                continue
            if current < 0:
                continue
            last = current
            break
    return last


def _min_progress_for_rate(total: int, configured: Optional[int] = None) -> int:
    """Minimum progress count before cumulative rate math is trusted.

    Startup-heavy JAX jobs can spend thousands of seconds in compilation/import
    before the first few iterations. Projecting total runtime from Iter 1/2000
    turns that warmup into a multi-thousand-hour ETA. Use a small adaptive
    threshold: roughly 1% of the run, capped at 20 units, with a floor of 3 for
    short jobs. Explicit callers can still pass a stricter threshold.
    """
    try:
        total_i = max(1, int(total))
    except Exception:
        total_i = 1
    if configured is not None:
        try:
            return max(1, min(total_i, int(configured)))
        except Exception:
            return 1
    adaptive = max(3, min(20, (total_i + 99) // 100))
    return max(1, min(total_i, adaptive))


def compute_eta_seconds(tail_text: str,
                        elapsed_s: float,
                        fallback_ewma_s: float = 0,
                        min_progress_for_rate: Optional[int] = None,
                        cmd: Optional[str] = None) -> int:
    """Returns ETA (remaining seconds) as int. 0 means unknown / done / no signal.

    Strategy:
      1. Parse latest (current, total) from tail. If found AND current ≥
         min_progress_for_rate AND elapsed > 0:
            rate = current / elapsed  (steps per second observed THIS run)
            eta  = (total - current) / rate
         Caps at 0 lower bound; no upper cap (a 30-day projection IS the right
         answer if the task's that slow).
      2. Else: fallback to (fallback_ewma_s - elapsed_s) clamped at 0. This is
         the per-signature historical estimate minus how long we've already run.
      3. If neither signal available, return 0.

    Why current/elapsed instead of parsing the displayed it/s? The displayed
    rate jumps wildly during warmup (JAX compilation, cuDNN init, etc.) and
    converges to a steady-state. current/elapsed is the average-since-start,
    which is more stable and matches "if it keeps going at the same overall
    pace, how much longer". For warmup-heavy workloads (BAPR JAX) this
    over-estimates ETA in the first few minutes — acceptable; under-estimating
    would be worse (would trigger premature migration).
    """
    elapsed = max(1.0, float(elapsed_s))  # avoid div-by-zero

    # Tier 0 (highest priority): tqdm's own pre-computed remaining-seconds.
    # tqdm uses a smoothed windowed rate that adapts to warmup vs steady-state
    # better than our cumulative current/elapsed; trust it when available.
    tqdm_eta = parse_tqdm_eta(tail_text)
    if tqdm_eta is not None:
        return int(tqdm_eta)
    saas_native_projection = saas_native_eta_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if saas_native_projection is not None:
        return int(saas_native_projection.get("eta_s") or 0)
    growing_projection = growing_iter_cost_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if growing_projection is not None:
        return int(growing_projection.get("eta_s") or 0)
    kg_inner_projection = kg_inner_progress_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if kg_inner_projection is not None:
        return int(kg_inner_projection.get("eta_s") or 0)
    scolhkg_projection = scolhkg_progress_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if scolhkg_projection is not None:
        return int(scolhkg_projection.get("eta_s") or 0)
    bapr_protocol_projection = bapr_fork_protocol_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if bapr_protocol_projection is not None:
        return int(bapr_protocol_projection.get("eta_s") or 0)
    bapr_specialist_projection = bapr_independent_specialist_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if bapr_specialist_projection is not None:
        return int(bapr_specialist_projection.get("eta_s") or 0)
    inline_eta = parse_inline_eta(tail_text)
    if inline_eta is not None and (
        inline_eta > 0 or not _BOTORCH_SAASBO_RE.search(cmd or "")
    ):
        return int(inline_eta)

    freqduet_projection = freqduet_shard_projection(tail_text, elapsed_s=elapsed, cmd=cmd)
    if freqduet_projection is not None:
        return int(freqduet_projection.get("eta_s") or 0)

    iter_window_projection = iter_time_window_projection(tail_text, elapsed_s=elapsed, cmd=cmd)
    if iter_window_projection is not None:
        return int(iter_window_projection.get("eta_s") or 0)

    progress_window_projection = timestamped_progress_window_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if progress_window_projection is not None:
        return int(progress_window_projection.get("eta_s") or 0)

    progress = parse_progress(tail_text, cmd=cmd)
    if progress is not None and _BOTORCH_SAASBO_RE.search(cmd or ""):
        try:
            saas_n0 = max(0, int(_command_arg(cmd or "", "--n0") or 0))
        except ValueError:
            saas_n0 = 0
        if int(progress[0]) <= saas_n0:
            progress = None
    if progress is not None:
        current, total = progress
        seconds_per_unit = parse_seconds_per_unit(tail_text)
        if seconds_per_unit is not None and total > 0 and current <= total:
            remaining = (float(total) - float(current)) * float(seconds_per_unit)
            return int(max(0, remaining))
        if current >= _min_progress_for_rate(total, min_progress_for_rate):
            rate = current / elapsed
            if rate > 0:
                remaining = (total - current) / rate
                return int(max(0, remaining))

    # Fallback: EWMA-based projection
    if fallback_ewma_s > 0:
        return int(max(0, fallback_ewma_s - elapsed))

    return 0


def runtime_projection(tail_text: str,
                       elapsed_s: float,
                       cmd: Optional[str] = None,
                       min_progress_for_rate: Optional[int] = None) -> Optional[dict]:
    """Project total runtime from the latest progress signal.

    Returns a small dict:
      {
        "source": "tqdm" | "progress_rate",
        "eta_s": remaining seconds,
        "total_s": projected total seconds for the whole task,
        "current": current unit (when known),
        "total_units": total unit count (when known),
        "unit_s": projected seconds per unit (when total_units known),
      }

    `elapsed_s` is scheduler-observed task elapsed time, not tqdm's internal
    loop elapsed. That intentionally includes startup/import/checkpoint overhead
    in the projected walltime. For tqdm workloads, the remaining time comes from
    tqdm's smoothed estimate, then `total_s = elapsed_s + remaining_s`.
    """
    elapsed = max(0.0, float(elapsed_s or 0))
    progress = parse_progress(tail_text, cmd=cmd)

    tqdm_eta = parse_tqdm_eta(tail_text)
    if tqdm_eta is not None:
        total_s = int(max(0, elapsed + tqdm_eta))
        out = {"source": "tqdm", "eta_s": int(tqdm_eta), "total_s": total_s}
        if progress is not None:
            current, total = progress
            out["current"] = int(current)
            out["total_units"] = int(total)
            if total > 0:
                out["unit_s"] = float(total_s) / float(total)
        return out

    saas_native_projection = saas_native_eta_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if saas_native_projection is not None:
        tracker_projection = growing_iter_cost_projection(
            tail_text, elapsed_s=elapsed, cmd=cmd)
        if tracker_projection is not None:
            saas_native_projection["robust_eta_s"] = int(
                tracker_projection.get("eta_s") or 0
            )
        return saas_native_projection

    growing_projection = growing_iter_cost_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if growing_projection is not None:
        return growing_projection

    kg_inner_projection = kg_inner_progress_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if kg_inner_projection is not None:
        return kg_inner_projection

    scolhkg_projection = scolhkg_progress_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if scolhkg_projection is not None:
        return scolhkg_projection

    bapr_protocol_projection = bapr_fork_protocol_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if bapr_protocol_projection is not None:
        return bapr_protocol_projection

    bapr_specialist_projection = bapr_independent_specialist_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if bapr_specialist_projection is not None:
        return bapr_specialist_projection

    inline_eta = parse_inline_eta(tail_text)
    if inline_eta is not None and (
        inline_eta > 0 or not _BOTORCH_SAASBO_RE.search(cmd or "")
    ):
        total_s = int(max(0, elapsed + inline_eta))
        out = {"source": "inline_eta", "eta_s": int(inline_eta), "total_s": total_s}
        if progress is not None:
            current, total = progress
            out["current"] = int(current)
            out["total_units"] = int(total)
            if total > 0:
                out["unit_s"] = float(total_s) / float(total)
        return out

    freqduet_projection = freqduet_shard_projection(tail_text, elapsed_s=elapsed, cmd=cmd)
    if freqduet_projection is not None:
        return freqduet_projection

    iter_window_projection = iter_time_window_projection(tail_text, elapsed_s=elapsed, cmd=cmd)
    if iter_window_projection is not None:
        return iter_window_projection

    progress_window_projection = timestamped_progress_window_projection(
        tail_text, elapsed_s=elapsed, cmd=cmd)
    if progress_window_projection is not None:
        return progress_window_projection

    if progress is not None and _BOTORCH_SAASBO_RE.search(cmd or ""):
        try:
            saas_n0 = max(0, int(_command_arg(cmd or "", "--n0") or 0))
        except ValueError:
            saas_n0 = 0
        if int(progress[0]) <= saas_n0:
            progress = None

    if progress is not None:
        current, total = progress
        seconds_per_unit = parse_seconds_per_unit(tail_text)
        if seconds_per_unit is not None and total > 0 and current <= total:
            eta_s = int(max(0, (float(total) - float(current)) * float(seconds_per_unit)))
            total_s = int(max(elapsed, elapsed + eta_s))
            return {
                "source": "seconds_per_unit",
                "eta_s": eta_s,
                "total_s": total_s,
                "current": int(current),
                "total_units": int(total),
                "unit_s": float(seconds_per_unit),
            }
        if (current >= _min_progress_for_rate(total, min_progress_for_rate)
                and total > 0 and elapsed > 0):
            unit_s = float(elapsed) / float(current)
            total_s = int(max(0, unit_s * float(total)))
            eta_s = int(max(0, total_s - elapsed))
            return {
                "source": "progress_rate",
                "eta_s": eta_s,
                "total_s": total_s,
                "current": int(current),
                "total_units": int(total),
                "unit_s": unit_s,
            }
    return None


def runtime_projection_from_log(log_text: str,
                                cmd: Optional[str] = None,
                                observed_duration_s: float = 0) -> Optional[dict]:
    """Project total runtime from a local preflight/test log.

    Priority:
      1. tqdm elapsed+remaining from the log itself. This is the intended path
         for "run locally first, then submit to scheduleurm".
      2. progress N/M plus observed wall duration, if the caller monitored the
         preflight process.
    """
    tqdm_times = parse_tqdm_elapsed_remaining(log_text)
    progress = parse_progress(log_text, cmd=cmd)
    if tqdm_times is not None:
        elapsed, remaining = tqdm_times
        total_s = int(elapsed + remaining)
        out = {
            "source": "local_test_tqdm",
            "eta_s": int(remaining),
            "total_s": total_s,
        }
        if progress is not None:
            current, total = progress
            out["current"] = int(current)
            out["total_units"] = int(total)
            if total > 0:
                out["unit_s"] = float(total_s) / float(total)
        return out

    if observed_duration_s > 0 and progress is not None:
        current, total = progress
        if current > 0 and total > 0:
            unit_s = float(observed_duration_s) / float(current)
            total_s = int(max(0, unit_s * float(total)))
            return {
                "source": "local_test_progress",
                "eta_s": int(max(0, total_s - observed_duration_s)),
                "total_s": total_s,
                "current": int(current),
                "total_units": int(total),
                "unit_s": unit_s,
            }
    return None


def format_eta(seconds: int) -> str:
    """Pretty-print ETA for status / TUI. Mirrors _fmt_min/_fmt_eta pattern in tui.py
    but lives here so the watcher can use the same format."""
    if seconds <= 0:
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"
