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

import re
from typing import Optional, Tuple


# Patterns (priority order). Each returns groups (current, total). Optional
# third group (rate) is parsed but currently unused — we recompute rate from
# current/elapsed because that's more reliable than parsing the displayed unit
# (some tools show "it/s" for steps, others for episodes; ambiguous).
_ETA_PATTERNS = [
    # tqdm: "  47%|████▋     | 1234/5678 [00:42<03:21, 12.34it/s]"
    # also catches simpler tqdm without percent prefix: "1234/5678 [..., 12.34it/s]"
    re.compile(r'(\d+)\s*/\s*(\d+)\s*\[[^\]]*?(\d+(?:\.\d+)?)\s*it/s'),
    # tqdm slow form: "1234/5678 [..., 1.23s/it]"
    re.compile(r'(\d+)\s*/\s*(\d+)\s*\[[^\]]*?(\d+(?:\.\d+)?)\s*s/it'),
    # explicit "[Epoch N/M]" e.g. "[Epoch 23/200]"
    re.compile(r'\[Epoch\s+(\d+)\s*/\s*(\d+)\]'),
    # explicit "Epoch: N/M" or "Epoch N/M"
    re.compile(r'(?:^|\s)Epoch[:\s]+(\d+)\s*/\s*(\d+)\b'),
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
    r'\[\s*(\S+?)\s*<\s*(\S+?)\s*,\s*[\d.?]+\s*(?:it/s|s/it)(?:\s*,[^\]]*)?\s*\]'
)

# Explicit ETA in free-form progress lines, e.g.
#   "[16/103] ... (874.7m, ETA 4756.0m)"
_INLINE_ETA_RE = re.compile(
    r'(?:^|[\s,(])ETA\s*[:=]?\s*(\d+(?:\.\d+)?)\s*([smhd])\b',
    re.IGNORECASE,
)


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


# Cmd-line flags that declare the task's total step/iter count. Order doesn't
# matter; we take the first match. Common across PyTorch, JAX, scikit, etc.
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
    for pat in _CMD_TOTAL_PATTERNS:
        m = pat.search(cmd)
        if m:
            try:
                v = int(m.group(1))
                if v > 0:
                    return v
            except ValueError:
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
    inline_eta = parse_inline_eta(tail_text)
    if inline_eta is not None:
        return int(inline_eta)

    progress = parse_progress(tail_text, cmd=cmd)
    if progress is not None:
        current, total = progress
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

    inline_eta = parse_inline_eta(tail_text)
    if inline_eta is not None:
        total_s = int(max(0, elapsed + inline_eta))
        out = {"source": "inline_eta", "eta_s": int(inline_eta), "total_s": total_s}
        if progress is not None:
            current, total = progress
            out["current"] = int(current)
            out["total_units"] = int(total)
            if total > 0:
                out["unit_s"] = float(total_s) / float(total)
        return out

    if progress is not None:
        current, total = progress
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
