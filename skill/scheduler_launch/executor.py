"""Deferred launch-intent execution helpers."""

from __future__ import annotations

import copy
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable


def execute_launch_intents(
    events: list,
    *,
    max_workers: int,
    launch_fn: Callable,
    commit_fn: Callable,
    batch_commit_fn: Callable | None = None,
    slot_lock_factory: Callable,
    notify_fn: Callable,
    commit_batch_size: int = 8,
    commit_max_delay_s: float = 0.5,
) -> list:
    """Execute launch_intent events, commit their results, and preserve event order.

    The scheduler writes durable launch leases while holding the state lock.
    This helper runs the slow launch calls outside that lock and replaces each
    launch_intent with the commit events produced by commit_fn.
    """
    if not any(ev.get("type") == "launch_intent" for ev in events):
        return events
    launch_intents = [
        (idx, ev) for idx, ev in enumerate(events)
        if ev.get("type") == "launch_intent"
    ]

    def _launch_one(idx: int, ev: dict):
        task = copy.deepcopy(ev.get("task") or {})
        node_state = copy.deepcopy(ev.get("node_state"))
        node = task.get("node") or (node_state or {}).get("name") or ""
        try:
            with slot_lock_factory(str(node)):
                ok, msg = launch_fn(task, node_state=node_state)
        except Exception as e:
            ok, msg = False, f"launch exception: {e}"
        return {
            "idx": idx,
            "intent": ev,
            "task": task,
            "ok": ok,
            "msg": msg,
            "node": node,
        }

    def _commit_one(record: dict):
        idx = record["idx"]
        ev = record["intent"]
        task = record["task"]
        ok = record["ok"]
        msg = record["msg"]
        node = record["node"]
        try:
            committed = commit_fn(ev, task, ok, msg)
        except Exception as e:
            notify_fn(
                "deferred_launch_commit_error",
                {
                    "task_id": (task or {}).get("id"),
                    "node": node,
                    "error": str(e)[:300],
                },
                feishu_enabled=False,
            )
            committed = [{
                "type": "launch_commit_error",
                "task_id": (task or {}).get("id"),
                "task": task,
                "error": str(e)[:300],
            }]
        return idx, committed

    results_by_idx: dict[int, list] = {}

    def _commit_records(records: list[dict]) -> None:
        if not records:
            return
        records.sort(key=lambda r: r["idx"])
        if batch_commit_fn is not None:
            try:
                committed = batch_commit_fn(records)
            except Exception as e:
                notify_fn(
                    "deferred_launch_commit_error",
                    {
                        "count": len(records),
                        "error": str(e)[:300],
                    },
                    feishu_enabled=False,
                )
                committed = {
                    record["idx"]: [{
                        "type": "launch_commit_error",
                        "task_id": ((record.get("task") or {}).get("id")),
                        "task": record.get("task"),
                        "error": str(e)[:300],
                    }]
                    for record in records
                }
            results_by_idx.update(committed)
            return
        for record in records:
            out_idx, committed = _commit_one(record)
            results_by_idx[out_idx] = committed

    workers = max(1, min(int(max_workers or 1), len(launch_intents)))
    if workers <= 1:
        for idx, ev in launch_intents:
            _commit_records([_launch_one(idx, ev)])
    else:
        batch_size = max(1, int(commit_batch_size or 1))
        max_delay = max(0.05, float(commit_max_delay_s or 0.0))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(_launch_one, idx, ev)
                for idx, ev in launch_intents
            }
            buffered: list[dict] = []
            last_commit_at = time.monotonic()
            while pending:
                done, pending = wait(
                    pending,
                    timeout=max_delay,
                    return_when=FIRST_COMPLETED,
                )
                buffered.extend(future.result() for future in done)
                while len(buffered) >= batch_size:
                    _commit_records(buffered[:batch_size])
                    del buffered[:batch_size]
                    last_commit_at = time.monotonic()
                if buffered and (
                    not pending or time.monotonic() - last_commit_at >= max_delay
                ):
                    _commit_records(buffered)
                    buffered = []
                    last_commit_at = time.monotonic()
            _commit_records(buffered)

    final_events: list = []
    for idx, ev in enumerate(events):
        if ev.get("type") == "launch_intent":
            final_events.extend(results_by_idx.get(idx) or [])
        else:
            final_events.append(ev)
    return final_events
