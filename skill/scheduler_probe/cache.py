"""Short-lived node probe cache used when live probing is unavailable."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

try:
    from scheduler_state.io import atomic_write_json, load_json
except ImportError:  # package import path used by pytest
    from skill.scheduler_state_io import atomic_write_json, load_json


def save_node_probe_cache(
    cache_file: str | Path,
    states: dict,
    *,
    now_fn: Callable[[], float] = time.time,
) -> None:
    if not states:
        return
    try:
        atomic_write_json(cache_file, {
            "ts": now_fn(),
            "states": states,
        })
    except Exception:
        pass


def load_node_probe_cache(
    cache_file: str | Path,
    *,
    default_ttl_s: int,
    max_age_s: Optional[int] = None,
    now_fn: Callable[[], float] = time.time,
) -> dict:
    ttl = int(default_ttl_s if max_age_s is None else max_age_s)
    if ttl <= 0:
        return {}
    try:
        data = load_json(cache_file, {}, now_fn=now_fn)
    except Exception:
        return {}
    try:
        ts = float(data.get("ts") or 0)
    except Exception:
        return {}
    age = now_fn() - ts
    if age < 0 or age > ttl:
        return {}
    states = data.get("states") or {}
    if not isinstance(states, dict):
        return {}
    out = {}
    for name, state in states.items():
        if not isinstance(state, dict):
            continue
        rec = dict(state)
        rec["probe_fallback"] = "stale_node_probe_cache"
        rec["probe_cache_age_s"] = int(age)
        out[name] = rec
    return out
