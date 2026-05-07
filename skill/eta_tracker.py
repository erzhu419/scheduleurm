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
]


def parse_progress(tail_text: str) -> Optional[Tuple[int, int]]:
    """Walk all patterns over every line of tail_text. Return the LATEST
    (current, total) found. None if nothing matches.

    "Latest" = most recent line containing a progress pattern. tqdm rewrites
    the same line repeatedly so we want the LAST line's number, not the first.
    """
    if not tail_text:
        return None
    last = None
    # Iterate lines in original order; last successful match wins (so most
    # recent progress, since logs are append-only and tail is the newest N bytes).
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
            # Sanity: total must be positive, current must fit in [0, total].
            # Some training scripts log "0/0" before init — reject that.
            if total <= 0 or current < 0 or current > total:
                continue
            last = (current, total)
            break  # this line matched; move to next line in the tail
    return last


def compute_eta_seconds(tail_text: str,
                        elapsed_s: float,
                        fallback_ewma_s: float = 0,
                        min_progress_for_rate: int = 1) -> int:
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

    progress = parse_progress(tail_text)
    if progress is not None:
        current, total = progress
        if current >= min_progress_for_rate:
            rate = current / elapsed
            if rate > 0:
                remaining = (total - current) / rate
                return int(max(0, remaining))

    # Fallback: EWMA-based projection
    if fallback_ewma_s > 0:
        return int(max(0, fallback_ewma_s - elapsed))

    return 0


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
