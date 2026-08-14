"""wait-for command implementation."""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WaitForDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    recover_stale_launching_tasks: Callable[[dict], Any]
    update_running_tasks: Callable[..., Any]
    running_probe_snapshot_outside_lock: Callable[[str], tuple]
    eta_tail_snapshot_outside_lock: Callable[[str], tuple]
    refresh_writer_lease: Callable[[], Any]
    now: Callable[[], float] = time.time
    sleep: Callable[[float], Any] = time.sleep
    print_fn: Callable[..., Any] = print


def _matching_tasks(state: dict, *, task_ids: set, signature_glob: str | None) -> list:
    matches = []
    for task in state.get("tasks", []):
        hit = False
        if task_ids and task["id"] in task_ids:
            hit = True
        elif signature_glob and fnmatch.fnmatch(task.get("signature", ""), signature_glob):
            hit = True
        if hit:
            matches.append(task)
    return matches


def cmd_wait_for(args, *, deps: WaitForDeps) -> int:
    """Block until all matching tasks reach a terminal state."""
    deadline = deps.now() + args.timeout if args.timeout > 0 else float("inf")
    seen_ids = set()
    last_print = 0.0
    task_ids = set(getattr(args, "task_ids", None) or [])
    signature_glob = getattr(args, "signature", None)

    while True:
        if getattr(args, "refresh", False):
            # The watcher is the sole running-state writer while it is alive.
            # A refresh wait-for process only takes over while the watcher
            # lifetime lease is unowned, preserving refresh as a failover path
            # without creating a second long-lived control loop.
            with deps.refresh_writer_lease() as owns_refresh:
                if owns_refresh:
                    running_probe_results, running_probe_ids = (
                        deps.running_probe_snapshot_outside_lock(
                            "wait-for:running-probe"
                        )
                    )
                    eta_tail_outputs, eta_tail_ids = deps.eta_tail_snapshot_outside_lock(
                        "wait-for:eta-tail"
                    )
                    with deps.state_lock(purpose="wait-for:update"):
                        state = deps.load_state()
                        deps.recover_stale_launching_tasks(state)
                        deps.update_running_tasks(
                            state,
                            probe_results=running_probe_results,
                            probed_task_ids=running_probe_ids,
                            eta_tail_outputs=eta_tail_outputs,
                            eta_probed_task_ids=eta_tail_ids,
                        )
                        deps.save_state(state)
                else:
                    with deps.state_lock(shared=True, purpose="wait-for:snapshot"):
                        state = deps.load_state()
        else:
            with deps.state_lock(shared=True, purpose="wait-for:snapshot"):
                state = deps.load_state()

        matches = _matching_tasks(
            state,
            task_ids=task_ids,
            signature_glob=signature_glob,
        )
        if not matches:
            if seen_ids:
                deps.print_fn(
                    f"[wait-for] all {len(seen_ids)} previously-matched tasks gone - "
                    "assumed terminal"
                )
                return 0
            if deps.now() > deadline:
                deps.print_fn("[wait-for] timeout: no matching tasks ever found")
                return 2
            deps.sleep(args.poll)
            continue

        seen_ids.update(task["id"] for task in matches)
        terminal_states = ("done", "failed", "cancelled")
        terminal = [task for task in matches if task["status"] in terminal_states]
        running = [task for task in matches if task["status"] == "running"]
        launching = [task for task in matches if task["status"] == "launching"]
        queued = [task for task in matches if task["status"] == "queued"]

        if len(terminal) == len(matches):
            done_n = sum(1 for task in terminal if task["status"] == "done")
            fail_n = sum(1 for task in terminal if task["status"] == "failed")
            canc_n = sum(1 for task in terminal if task["status"] == "cancelled")
            tag = signature_glob or f"{len(task_ids)} ids"
            deps.print_fn(
                f"[wait-for {tag}] all {len(matches)} terminal: "
                f"{done_n} done, {fail_n} failed, {canc_n} cancelled"
            )
            return 0

        if deps.now() > deadline:
            tag = signature_glob or f"{len(task_ids)} ids"
            deps.print_fn(
                f"[wait-for {tag}] timeout: {len(running)} running, "
                f"{len(launching)} launching, {len(queued)} queued, "
                f"{len(terminal)} terminal"
            )
            return 1

        now = deps.now()
        if args.verbose and now - last_print >= 300:
            tag = signature_glob or f"{len(task_ids)} ids"
            deps.print_fn(
                f"[wait-for {tag}] {len(running)} running, {len(launching)} launching, "
                f"{len(queued)} queued, {len(terminal)}/{len(matches)} terminal"
            )
            last_print = now
        deps.sleep(args.poll)
