"""Remote disconnect/reconnect state-sync regressions.

The runner imports this module with the already-loaded scheduler module and a
check() callback. Tests are side-effect free: no real ssh or scheduler state.
"""

from __future__ import annotations

import os
import tempfile
import time


class _SeqBackend:
    name = "seq"

    def __init__(self, results):
        self.results = list(results)
        self.i = 0

    def batch_probe(self, state):
        if not self.results:
            return {}
        res = self.results[min(self.i, len(self.results) - 1)]
        self.i += 1
        return {
            t["id"]: dict(res)
            for t in state.get("tasks", [])
            if t.get("status") == "running"
        }


def _task(log_path, **overrides):
    base = {
        "id": "toff",
        "status": "running",
        "node": "local",
        "gpu_idx": 0,
        "project": "offline-sync",
        "signature": "offline-sync/train",
        "description": "offline sync test",
        "cmd": "python train.py --seed 1",
        "cwd": tempfile.gettempdir(),
        "extra_env": {},
        "env_spec": "none",
        "image": "",
        "priority": "normal",
        "remote_pids": [12345],
        "alive_pids": [12345],
        "log_path": log_path,
        "started_at": time.time() - 1000,
        "finished_at": None,
        "peak_vram_mb": 900,
        "peak_ram_mb": 512,
        "est_vram_mb": 1000,
        "ram_mb": 1000,
        "cpu_cores": 1,
        "retry_count": 0,
        "notified_done": False,
    }
    base.update(overrides)
    return base


def run(check, sch):
    print("\n[external] remote offline/reconnect sync")

    saved_backend = sch._BACKEND
    saved_notify = sch.notify
    sch.notify = lambda *a, **k: None
    try:
        with tempfile.TemporaryDirectory() as td:
            log_ok = os.path.join(td, "ok.log")
            with open(log_ok, "w") as f:
                f.write("Epoch 1/1\nTraining complete\n")
            state = {"tasks": [_task(log_ok)], "next_id": 9000}
            sch._BACKEND = _SeqBackend([
                {"state": "unknown", "alive_pids": [], "vram_mb": 0,
                 "ram_mb": 0, "pcpu": 0.0, "error": "ssh timeout"},
                {"state": "dead", "alive_pids": [], "vram_mb": 0,
                 "ram_mb": 0, "pcpu": 0.0},
            ])
            sch.update_running_tasks(state)
            t = state["tasks"][0]
            check("remote offline unknown keeps task running",
                  t["status"] == "running" and t.get("probe_unknown_since"))
            check("remote offline unknown records probe reason/count",
                  t.get("probe_unknown_count") == 1
                  and "ssh timeout" in t.get("last_probe_unknown_reason", ""))
            t["probe_unknown_since"] = time.time() - 120
            sch.update_running_tasks(state)
            check("reconnect dead+success syncs terminal done",
                  t["status"] == "done" and not t.get("probe_unknown_since"))
            check("reconnect terminal records offline sync duration",
                  t.get("last_probe_unknown_duration_s", 0) >= 100
                  and t.get("_diagnosis", {}).get("offline_sync_s", 0) >= 100)
            check("terminal diagnosis says it was synced after unknown probe",
                  "synced after remote probe was unknown"
                  in t.get("_diagnosis", {}).get("reason", ""))

            log_fail = os.path.join(td, "fail.log")
            with open(log_fail, "w") as f:
                f.write("Epoch 3/10 loss=1.23\n")
            parent = _task(log_fail, id="tfail", signature="offline-sync/fail")
            state = {"tasks": [parent], "next_id": 9000}
            sch._BACKEND = _SeqBackend([
                {"state": "unknown", "alive_pids": [], "vram_mb": 0,
                 "ram_mb": 0, "pcpu": 0.0, "error": "node offline"},
                {"state": "dead", "alive_pids": [], "vram_mb": 0,
                 "ram_mb": 0, "pcpu": 0.0},
            ])
            sch.update_running_tasks(state)
            parent["probe_unknown_since"] = time.time() - 180
            sch.update_running_tasks(state)
            retry = next((x for x in state["tasks"] if x.get("parent_id") == "tfail"), None)
            check("reconnect dead+no-success syncs failed parent",
                  parent["status"] == "failed"
                  and parent.get("_diagnosis", {}).get("offline_sync_s", 0) >= 100)
            check("failed terminal after offline creates retry clone",
                  retry is not None and retry.get("status") == "queued")
            check("retry clone does not inherit offline-probe markers",
                  retry is not None
                  and "probe_unknown_since" not in retry
                  and "last_probe_unknown_duration_s" not in retry)
    finally:
        sch._BACKEND = saved_backend
        sch.notify = saved_notify
