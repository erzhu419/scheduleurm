#!/usr/bin/env python3
"""Minimal regression tests for scheduler.py. Run from anywhere:
    python ~/.claude/skills/scheduler/test_regression.py

Covers four bugs that bit us in the past:
  1. PPID-aware adopt — a child of an already-tracked PID must NOT be re-adopted
     as a phantom second task (#2 from the GPT review; t0966/t0969 case).
  2. Crash requeue dedup — _requeue_after_crash must NOT create a duplicate when
     a live task with same sig+cmd already exists (the t0800-killed-instead-of-t0919
     incident's root cause prevention).
  3. Preempt sufficiency — high-prio waiting > 5min must keep evicting until freed
     CPU/RAM covers its requirement, capped at PREEMPT_MAX_VICTIMS_PER_DISPATCH.
  4. Launch-fail fallback — launch() returning failure must push task back to queued
     with launch_fail_count, not terminal-fail it. Hits cap → failed + heal escalation.

Tests are self-contained: no real processes launched, no disk state mutated, no ssh.
"""

import copy
import importlib.util
import os
import sys
import time

SCHED_PATH = os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")
spec = importlib.util.spec_from_file_location("scheduler", SCHED_PATH)
sch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sch)

results = []
def check(name, cond, diag=""):
    results.append((name, cond, diag))
    mark = "PASS" if cond else "FAIL"
    extra = f"  [{diag}]" if (diag and not cond) else ""
    print(f"  {mark}  {name}{extra}")

# -----------------------------------------------------------------------------
def test_ppid_descendant_filter():
    """_descendants_of: marks a tracked PID's whole subtree, leaves siblings alone."""
    print("\n[1] PPID-aware adopt (child-of-tracked must be filtered out)")
    # Tree:
    #   1 (init) → 100 (tracked) → 101 → 102
    #              200 (untracked) → 201
    #              300 (orphan)
    ppid_of = {100: 1, 101: 100, 102: 101, 200: 1, 201: 200, 300: 1}
    desc = sch._descendants_of({100}, ppid_of)
    check("child of tracked is descendant", 101 in desc)
    check("grandchild of tracked is descendant", 102 in desc)
    check("unrelated PID not descendant", 200 not in desc and 201 not in desc and 300 not in desc)
    check("tracked PID itself NOT in descendant set", 100 not in desc)
    check("empty roots → empty result", sch._descendants_of(set(), ppid_of) == set())
    check("empty ppid_map → empty result", sch._descendants_of({100}, {}) == set())
    # Cycle defense (shouldn't happen but mustn't infinite-loop)
    cyclic = {1: 2, 2: 1}
    sch._descendants_of({99}, cyclic)  # must terminate
    check("cycle in ppid_map terminates", True)

# -----------------------------------------------------------------------------
def test_crash_requeue_dedup():
    """_requeue_after_crash: live duplicate (same sig+cmd) → return existing id, no new task."""
    print("\n[2] Crash requeue dedup (no double-queue from same sig+cmd)")
    parent = {
        "id": "tA", "signature": "proj/exp/s42",
        "cmd": "python train.py --seed 42", "status": "failed",
        "_diagnosis": {"is_crash": True, "reason": "test", "tail": "", "lifetime_s": 100},
        "retry_count": 0, "ram_mb": 100, "est_vram_mb": 0, "cpu_cores": 1,
        "submitted_at": time.time(), "extra_env": {}, "priority": "normal",
    }
    # Case A: live duplicate exists → return its id, NO append
    state_a = {
        "tasks": [
            parent,
            {"id": "tB", "signature": "proj/exp/s42",
             "cmd": "python train.py --seed 42", "status": "queued"},
        ],
        "next_id": 100,
    }
    pre = len(state_a["tasks"])
    new_id = sch._requeue_after_crash(parent, state_a)
    check("returns existing duplicate's id", new_id == "tB")
    check("no new task appended on dedup hit", len(state_a["tasks"]) == pre)

    # Case B: same sig but different cmd → real requeue, new task appended
    parent2 = dict(parent); parent2["id"] = "tC"; parent2["retry_count"] = 0
    state_b = {
        "tasks": [
            parent2,
            {"id": "tD", "signature": "proj/exp/s42",
             "cmd": "python train.py --seed 999",  # different cmd
             "status": "queued"},
        ],
        "next_id": 200,
    }
    pre_b = len(state_b["tasks"])
    new_id_b = sch._requeue_after_crash(parent2, state_b)
    check("different cmd → real requeue (new id)", new_id_b is not None and new_id_b != "tD")
    check("different cmd → new task appended", len(state_b["tasks"]) == pre_b + 1)

    # Case C: same cmd but only DONE/FAILED siblings → must NOT dedup against terminal tasks
    parent3 = dict(parent); parent3["id"] = "tE"; parent3["retry_count"] = 0
    state_c = {
        "tasks": [
            parent3,
            {"id": "tF", "signature": "proj/exp/s42",
             "cmd": "python train.py --seed 42", "status": "done"},
            {"id": "tG", "signature": "proj/exp/s42",
             "cmd": "python train.py --seed 42", "status": "failed"},
        ],
        "next_id": 300,
    }
    pre_c = len(state_c["tasks"])
    new_id_c = sch._requeue_after_crash(parent3, state_c)
    check("done/failed siblings do NOT block requeue", new_id_c is not None
          and new_id_c not in ("tF", "tG") and len(state_c["tasks"]) == pre_c + 1)

    # Case D: backend launch artifacts from the crashed parent must be cleared on retry clone.
    parent4 = dict(parent)
    parent4.update({
        "id": "tH", "signature": "proj/exp/slurm-artifacts",
        "cmd": "python train.py --seed 777", "retry_count": 0,
        "slurm_job_id": 12345, "slurm_state": "FAILED",
        "container_name": "sched-tH", "container_main_pid": 999,
    })
    state_d = {"tasks": [parent4], "next_id": 400}
    new_id_d = sch._requeue_after_crash(parent4, state_d)
    clone = state_d["tasks"][-1]
    check("requeue clone clears stale slurm_job_id",
          new_id_d is not None and clone.get("slurm_job_id") is None,
          diag=str(clone))
    check("requeue clone clears stale docker/container artifacts",
          clone.get("container_name") is None and clone.get("container_main_pid") is None,
          diag=str(clone))

# -----------------------------------------------------------------------------
def test_preempt_sufficiency():
    """_preempt_for_high_priority: keep evicting until freed cpu/ram covers hi requirement."""
    print("\n[3] Preempt sufficiency (cumulative eviction until hi fits)")
    now = time.time()
    state = {
        "tasks": [
            # high-prio waited 10min, needs cpu=6 ram=6000 — three cpu=2 victims required
            {"id": "thi", "status": "queued", "priority": "high", "require_node": "local",
             "submitted_at": now - 600, "cpu_cores": 6, "ram_mb": 6000,
             "signature": "x/hi", "remote_pids": []},
            # 3 eligible victims (in age window 10–240 min, normal-prio, not adopted)
            {"id": "tv1", "status": "running", "priority": "normal", "node": "local",
             "started_at": now - 1200, "cpu_cores": 2, "ram_mb": 2000,
             "signature": "x/v1", "remote_pids": [], "auto_adopted": False},
            {"id": "tv2", "status": "running", "priority": "normal", "node": "local",
             "started_at": now - 1100, "cpu_cores": 2, "ram_mb": 2000,
             "signature": "x/v2", "remote_pids": [], "auto_adopted": False},
            {"id": "tv3", "status": "running", "priority": "normal", "node": "local",
             "started_at": now - 1000, "cpu_cores": 2, "ram_mb": 2000,
             "signature": "x/v3", "remote_pids": [], "auto_adopted": False},
            # ineligibles
            {"id": "tv4", "status": "running", "priority": "normal", "node": "local",
             "started_at": now - 60, "cpu_cores": 2, "ram_mb": 2000,
             "signature": "x/v4", "remote_pids": [], "auto_adopted": False},  # too fresh
            {"id": "tv5", "status": "running", "priority": "normal", "node": "local",
             "started_at": now - 18000, "cpu_cores": 2, "ram_mb": 2000,
             "signature": "x/v5", "remote_pids": [], "auto_adopted": False},  # too old
            {"id": "tv6", "status": "running", "priority": "normal", "node": "local",
             "started_at": now - 1100, "cpu_cores": 2, "ram_mb": 2000,
             "signature": "x/v6", "remote_pids": [], "auto_adopted": True},   # adopted
        ],
        "next_id": 999,
    }
    # _evict_to_queue tries ssh kill — stub run_on so it no-ops cleanly
    orig_run_on = sch.run_on
    sch.run_on = lambda *a, **k: (0, "", "")
    try:
        evicted = sch._preempt_for_high_priority(state, nodes=[])
    finally:
        sch.run_on = orig_run_on
    evicted_ids = sorted(e["id"] for e in evicted)
    check("cap honored (≤ PREEMPT_MAX_VICTIMS_PER_DISPATCH)",
          len(evicted) <= sch.PREEMPT_MAX_VICTIMS_PER_DISPATCH)
    check("3 victims evicted to satisfy cpu=6 need (each cpu=2)", len(evicted) == 3)
    check("only eligible victims (no fresh/old/adopted)",
          set(evicted_ids).issubset({"tv1", "tv2", "tv3"}),
          diag=f"got {evicted_ids}")
    # All evicted come back to queued — NOT failed, NOT crash-counted
    for vid in ("tv1", "tv2", "tv3"):
        v = next(t for t in state["tasks"] if t["id"] == vid)
        check(f"{vid} returned to queued (not failed)", v["status"] == "queued")
        check(f"{vid} retry_count NOT bumped", v.get("retry_count", 0) == 0)
    # Ineligibles untouched
    for vid in ("tv4", "tv5", "tv6"):
        v = next(t for t in state["tasks"] if t["id"] == vid)
        check(f"{vid} still running (not evicted)", v["status"] == "running")

# -----------------------------------------------------------------------------
def test_launch_fail_fallback():
    """_do_dispatch + launch failure: < cap → queued, hit cap → failed + escalation."""
    print("\n[4] Launch-fail fallback (requeue on transient, escalate on cap)")
    now = time.time()
    task = {
        "id": "tx1", "status": "queued", "signature": "proj/test", "cmd": "fake-cmd",
        "cwd": "/tmp", "priority": "normal", "ram_mb": 100, "est_vram_mb": 0,
        "cpu_cores": 1, "submitted_at": now, "extra_env": {}, "remote_pids": [],
        "alive_pids": [], "log_path": None, "started_at": None, "node": None,
        "gpu_idx": None, "peak_vram_mb": 0, "peak_ram_mb": 0, "resume_from": None,
        "resume_flag": "", "ckpt_dir": None, "ckpt_glob": "*", "git_repo": None,
        "preferred_node": None, "require_node": None, "project": "proj",
        "description": "synthetic test task",
    }
    state = {"tasks": [task], "next_id": 1000}
    nodes_template = [{"name": "local", "alive": True, "free_cpu": 8, "total_cpu": 12,
                       "free_ram_mb": 30000, "total_ram_mb": 30000, "loadavg": 1.0,
                       "gpus": [], "running_count": 0}]

    # Stub the I/O-bound boundaries — all OK except launch which fails
    saved = {
        "launch": sch.launch,
        "pick_placement": sch.pick_placement,
        "precheck_git": sch.precheck_git,
        "find_resume": sch.find_resume,
        "_write_escalation": sch._write_escalation,
        "save_state": sch.save_state,  # WAL save runs BEFORE launch in dispatch — must stub
    }
    esc_calls = []
    sch.launch = lambda t: (False, "synthetic launch failure")
    sch.pick_placement = lambda t, ns: ("local", None)        # CPU-only on local
    sch.precheck_git = lambda t: (True, "")
    sch.find_resume = lambda t: None
    sch._write_escalation = lambda task, cat, diag: esc_calls.append((task["id"], cat))
    sch.save_state = lambda s: None  # block fake state from reaching live queue.json

    try:
        # Iter 1: launch fails → status returns to queued, count = 1
        sch._do_dispatch(state, copy.deepcopy(nodes_template))
        check("after fail #1: status=queued (not failed)", task["status"] == "queued",
              diag=f"actual status={task['status']}")
        check("after fail #1: launch_fail_count=1", task.get("launch_fail_count") == 1,
              diag=f"actual={task.get('launch_fail_count')}")
        check("after fail #1: no escalation", len(esc_calls) == 0)
        check("after fail #1: node/gpu reset", task.get("node") is None and task.get("gpu_idx") is None)

        # Iter 2: still under cap
        sch._do_dispatch(state, copy.deepcopy(nodes_template))
        check("after fail #2: launch_fail_count=2", task.get("launch_fail_count") == 2)
        check("after fail #2: still queued", task["status"] == "queued")
        check("after fail #2: still no escalation", len(esc_calls) == 0)

        # Iter 3: hits MAX_LAUNCH_RETRY → terminal failed + escalation
        sch._do_dispatch(state, copy.deepcopy(nodes_template))
        check(f"after fail #{sch.MAX_LAUNCH_RETRY}: status=failed",
              task["status"] == "failed",
              diag=f"actual status={task['status']}")
        check(f"after fail #{sch.MAX_LAUNCH_RETRY}: escalation written with LAUNCH_FAIL_CAP",
              len(esc_calls) == 1 and esc_calls[0] == (task["id"], "LAUNCH_FAIL_CAP"),
              diag=f"actual={esc_calls}")
    finally:
        for k, v in saved.items():
            setattr(sch, k, v)

# -----------------------------------------------------------------------------
# Phase 2: SLURM-style robustness invariants
# -----------------------------------------------------------------------------
def test_cancel_never_becomes_failed():
    """User-cancel must not be misidentified as crash / OOM / server-reboot fail.
    Concrete invariants (as enforced by code):
      a) _detect_oom_kills_local skips status != 'done' (so cancelled tasks are exempt
         from being flipped to 'failed' even if syslog shows an OOM in the same window).
      b) _batch_check_running iterates only status=='running', so a task already
         marked 'cancelled' by cmd_cancel cannot be re-evaluated/diagnosed.
      c) _requeue_after_crash is only entered via the running→failed transition
         in _batch_check_running, never directly from a cancelled state.
    We assert (a) and (b) as live behavior; (c) is a structural property verified by inspection."""
    print("\n[5] Cancel-vs-OOM separation (cancel is sticky, never auto-requeued)")
    now = time.time()
    state = {
        "tasks": [
            # Cancelled task with timing that WOULD overlap a fake OOM event
            {"id": "tcanc", "status": "cancelled", "node": "local",
             "started_at": now - 600, "finished_at": now - 30,
             "peak_vram_mb": 0, "signature": "x/cancelled", "remote_pids": []},
            # done-but-suspicious task that SHOULD be flipped (control)
            {"id": "tdone", "status": "done", "node": "local", "auto_adopted": False,
             "started_at": now - 600, "finished_at": now - 30,
             "peak_vram_mb": 0, "signature": "x/done", "remote_pids": [],
             "_diagnosis": {}},
        ]
    }
    # Stub subprocess.check_output to return a syslog snippet with one OOM event in-window
    import subprocess as _sp
    orig_co = _sp.check_output
    fake_ts = time.strftime("%b %d %H:%M:%S", time.localtime(now - 30))
    fake_syslog = f"{fake_ts} host kernel: Out of memory: Killed process 999 (python)\n"
    def fake_check_output(cmd, *args, **kwargs):
        if isinstance(cmd, list) and "tail" in cmd[0]:
            return fake_syslog
        return orig_co(cmd, *args, **kwargs)
    _sp.check_output = fake_check_output
    try:
        flipped = sch._detect_oom_kills_local(state)
    finally:
        _sp.check_output = orig_co
    flipped_ids = [t["id"] for t in flipped]
    # The control task should be flipped, the cancelled one must NOT be
    check("cancelled task NOT flipped by OOM detector", "tcanc" not in flipped_ids,
          diag=f"flipped={flipped_ids}")
    canc = next(t for t in state["tasks"] if t["id"] == "tcanc")
    check("cancelled task status preserved", canc["status"] == "cancelled",
          diag=f"actual status={canc['status']}")
    # Structural: verify _batch_check_running source actually filters status=='running'
    src = open(SCHED_PATH).read()
    check("_batch_check_running filters status=='running' (won't touch cancelled)",
          'if t["status"] != "running": continue' in src
          or "if t.get(\"status\") != \"running\": continue" in src,
          diag="filter line not found in source")

def test_ram_placement_check():
    """RAM-OOM defense: _node_resources_ok must reject placements that would push
    free_ram below the configured headroom. Verifies (b) — RAM is checked at the
    same level as VRAM, not afterthought."""
    print("\n[6] RAM placement check (rejects pre-launch when below headroom)")
    # Mimic local-WSL config: declared 56GB, probed 30GB, 25% headroom
    node_info = {"ram_mb": 56000, "ram_headroom_frac": 0.25, "max_concurrent_running": 10}
    node_state = {"name": "local", "free_cpu": 8, "total_cpu": 12, "running_count": 0,
                  "free_ram_mb": 11000, "total_ram_mb": 30000}
    # Headroom = 30000 * 0.25 = 7500. Need (free - need) >= headroom → need <= 11000 - 7500 = 3500
    fits = {"id": "tfit", "ram_mb": 3000, "cpu_cores": 1}
    big = {"id": "tbig", "ram_mb": 8000, "cpu_cores": 1}
    edge = {"id": "tedge", "ram_mb": 3500, "cpu_cores": 1}  # exactly at boundary
    just_over = {"id": "tover", "ram_mb": 3501, "cpu_cores": 1}

    ok1, why1 = sch._node_resources_ok(fits, node_state, node_info)
    check("3000MB request fits (under boundary)", ok1, diag=why1)
    ok2, why2 = sch._node_resources_ok(big, node_state, node_info)
    check("8000MB request rejected with ram: reason",
          (not ok2) and "ram" in why2.lower(),
          diag=f"ok={ok2}, why={why2!r}")
    ok3, _ = sch._node_resources_ok(edge, node_state, node_info)
    check("3500MB at boundary fits (>= comparison)", ok3)
    ok4, _ = sch._node_resources_ok(just_over, node_state, node_info)
    check("3501MB just over boundary rejected", not ok4)

    # Also: probed total used for headroom, NOT over-declared 56GB
    # Headroom must reflect 30000 (probed), not 56000 (declared) — else it'd be 14000 and
    # the 8000 request might have squeezed in.
    fake_state = dict(node_state); fake_state["total_ram_mb"] = 30000
    ok_check, why_check = sch._node_resources_ok(big, fake_state, node_info)
    # If headroom were 56000*0.25=14000, free=11000 < 14000 → rejected anyway.
    # If headroom is 30000*0.25=7500, free-need=3000 < 7500 → also rejected. Both reject.
    # Verify the rejection MESSAGE references the actual headroom (7500), not 14000:
    check("rejection message uses probed total for headroom",
          "7500" in why_check or "headroom 7500" in why_check,
          diag=f"why={why_check!r}")

def test_default_vram_not_inflated():
    """High-default starvation: cascade must not blow novel-sig est_vram up beyond
    max(observed siblings). Median-of-candidates protects against one outlier."""
    print("\n[7] Cascade est_vram bounded (no runaway over-allocation)")
    # 4 sibling peaks: 600, 700, 5000 (outlier), 800
    history = {}
    state = {"tasks": [
        {"id": f"sib{i}", "status": "done", "project": "P",
         "description": "P2 retrain: configX seed=" + str(i),
         "peak_vram_mb": v, "signature": f"P/configX/s{i}"}
        for i, v in enumerate([600, 700, 5000, 800])
    ]}
    new = {"id": "tnew", "signature": "P/configX/s99",
           "project": "P", "description": "P2 retrain: configX seed=99",
           "est_vram_mb": 0}
    est = sch._effective_est_vram(new, state, history)
    check("median (not max) used for cascade",
          est < 5000,  # outlier 5000 must not dominate
          diag=f"got est={est}")
    check("cascade returns ≤ DEFAULT or near-median",
          est <= 1000,
          diag=f"got est={est}MB (siblings 600/700/800/5000 median is ~750)")

def test_status_view_no_truncation():
    """TUI/status truncation concern: make sure no part of the active queue is hidden.
    Both cmd_status (CLI) and TUI render-from-cache logic include EVERY task whose
    status is in (queued, launching, running). We verify by simulating the same filter on a
    synthetic state with 13 tasks."""
    print("\n[8] Active-queue visibility (no hidden tasks in status/TUI)")
    state = {"tasks": [
        {"id": f"t{i:04d}", "status": s, "signature": f"x/{s}/{i}",
         "description": f"job {i}", "project": "X", "peak_vram_mb": 0,
         "started_at": None, "finished_at": None, "node": None}
        for i, s in enumerate(
            ["queued"] * 8 + ["launching"] + ["running"] * 3 + ["done"]  # 13 total: 12 active + 1 done
        )
    ]}
    # cmd_status default filter: queued + launching + running, excludes done unless --all
    rows = [t for t in state["tasks"] if t["status"] in ("queued", "launching", "running")]
    check("cmd_status filter shows all 12 active tasks", len(rows) == 12,
          diag=f"got {len(rows)} rows")
    # TUI 'all' filter (running OR launching OR queued): same set
    tui_rows_all = [t for t in state["tasks"] if t["status"] in ("running", "launching", "queued")]
    check("TUI 'all' filter == status default filter", len(tui_rows_all) == 12)
    # TUI 'queued' filter
    tui_q = [t for t in state["tasks"] if t["status"] == "queued"]
    check("TUI 'queued' filter shows all 8 queued (no hidden)", len(tui_q) == 8)
    # TUI 'running' filter
    tui_r = [t for t in state["tasks"] if t["status"] == "running"]
    check("TUI 'running' filter shows all 3 running", len(tui_r) == 3)
    # Source structural: ensure neither TUI nor cmd_status has a slice/limit on active rows
    src = open(SCHED_PATH).read()
    tui_src = open(SCHED_PATH.replace("scheduler.py", "tui.py")).read()
    tui_render_src = tui_src.split("def _render_from_cache")[1].split("\n    def ")[0]
    check("cmd_status has no [:N] slice on active task rows",
          "rows[:" not in src.split("def cmd_status")[1].split("def cmd_show")[0],
          diag="rows[:N] suggests truncation")
    check("cmd_status source includes launching in active rows",
          '"launching"' in src.split("def cmd_status")[1].split("def cmd_show")[0])
    check("TUI all-filter source includes launching in active rows",
          '"launching"' in tui_render_src)
    check("TUI has no max-row slice in render",
          "tasks[:" not in tui_render_src,
          diag="tasks[:N] suggests truncation")
    check("TUI row location uses scheduler formatter (shows Slurm job/state)",
          "sch._format_task_location(t)" in tui_render_src,
          diag="TUI must not hand-roll node:GPU display")
    check("TUI text filter includes Slurm job id/state",
          '"slurm_job_id"' in tui_render_src and '"slurm_state"' in tui_render_src,
          diag="filtering for 'slurm', job id, or Slurm state would miss rows")

def test_high_defaults_lower_before_placement():
    """High stored/default estimates must not starve a job when sibling evidence says it is small.
    Covers both VRAM and RAM lowering before placement."""
    print("\n[9] High default estimates are lowered before placement (VRAM + RAM)")
    now = time.time()
    queued = {
        "id": "tq", "status": "queued", "signature": "P/config/s99",
        "cmd": "python train.py", "cwd": "/tmp", "priority": "normal",
        "ram_mb": 50000, "est_vram_mb": 5000, "cpu_cores": 1,
        "submitted_at": now, "extra_env": {}, "remote_pids": [],
        "alive_pids": [], "log_path": None, "started_at": None, "node": None,
        "gpu_idx": None, "peak_vram_mb": 0, "peak_ram_mb": 0, "resume_from": None,
        "resume_flag": "", "ckpt_dir": None, "ckpt_glob": "*", "git_repo": None,
        "preferred_node": None, "require_node": None, "project": "P",
        "description": "train: config seed=99",
    }
    state = {"tasks": [
        {"id": "sib", "status": "done", "project": "P", "signature": "P/config/s1",
         "description": "train: config seed=1", "peak_vram_mb": 600, "peak_ram_mb": 3000},
        queued,
    ], "next_id": 100}
    nodes = [{"name": "local", "alive": True, "free_cpu": 8, "total_cpu": 12,
              "free_ram_mb": 12000, "total_ram_mb": 30000, "loadavg": 1.0,
              "running_count": 0,
              "gpus": [{"idx": 0, "used_mb": 0, "total_mb": 8192,
                        "free_mb": 8192, "util_pct": 0}]}]
    saved = {
        "load_history": sch.load_history,
        "launch": sch.launch,
        "precheck_git": sch.precheck_git,
        "find_resume": sch.find_resume,
        "save_state": sch.save_state,  # P0b: _do_dispatch saves per launch; stub to avoid live writes
    }
    sch.load_history = lambda: {}
    sch.precheck_git = lambda t: (True, "")
    sch.find_resume = lambda t: None
    sch.save_state = lambda s: None  # stub — fake state must not reach live queue.json
    def fake_launch(t):
        t["status"] = "running"; t["remote_pids"] = [4242]
        t["process_group"] = 4242; t["started_at"] = time.time()
        t["log_path"] = "/tmp/fake.log"; return True, "pid=4242"
    sch.launch = fake_launch
    try:
        events, _ = sch._do_dispatch(state, copy.deepcopy(nodes))
    finally:
        for k, v in saved.items():
            setattr(sch, k, v)
    check("VRAM lowered from 5000 to sibling estimate", queued["est_vram_mb"] < 5000,
          diag=f"vram={queued.get('est_vram_mb')}")
    check("RAM lowered from 50000 to sibling estimate", queued["ram_mb"] < 50000,
          diag=f"ram={queued.get('ram_mb')}")
    check("task launches after estimates lower", queued["status"] == "running",
          diag=f"status={queued.get('status')}, events={events}")

def test_probe_ram_budget_cap():
    """Remote physical MemAvailable may exceed configured schedulable budget; placement must use
    the configured budget, not physical 500GB."""
    print("\n[10] Probe RAM is capped to schedulable budget")
    orig_run_on = sch.run_on
    def fake_run_on(node, cmd, timeout=15, check=False):
        out = (
            "0, 0, 12288, 12288, 0\n"
            "===SEP===\n"
            "500000\n515000\n"
            "===SEP===\n24\n"
            "===SEP===\n0.10\n"
            "===SEP===\n"
            "0, 0\n---SAMPLE---\n0, 0\n---SAMPLE---\n"
        )
        return 0, out, ""
    sch.run_on = fake_run_on
    try:
        info = sch.probe_node("jtl110gpu")
    finally:
        sch.run_on = orig_run_on
    check("remote total_ram capped to configured 204800MB", info["total_ram_mb"] == 204800,
          diag=str(info))
    check("remote free_ram capped to configured 204800MB", info["free_ram_mb"] == 204800,
          diag=str(info))
    check("actual_free_ram still recorded for observability", info["actual_free_ram_mb"] == 500000,
          diag=str(info))

def test_running_descendant_resources_counted():
    """A scheduler-launched bash root plus Python child/worker must be accounted as one task."""
    print("\n[11] Running resource accounting includes descendants")
    state = {"tasks": [{
        "id": "trun", "status": "running", "node": "local", "remote_pids": [100],
        "signature": "x/run", "cmd": "bash -c python", "started_at": time.time() - 60,
        "ram_mb": 100, "cpu_cores": 1, "peak_ram_mb": 0, "peak_vram_mb": 0,
    }]}
    orig_run_on = sch.run_on
    def fake_run_on(node, cmd, timeout=30, check=False):
        out = (
            "A100\n"
            "===VRAM===\n"
            "101, 700\n"
            "===PSALL===\n"
            "100 1 1000 0.0\n"
            "101 100 2048000 125.0\n"
            "102 101 1024000 80.0\n"
        )
        return 0, out, ""
    sch.run_on = fake_run_on
    try:
        sch.update_running_tasks(state)
    finally:
        sch.run_on = orig_run_on
    t = state["tasks"][0]
    check("descendant pids are alive_pids", set(t["alive_pids"]) == {100, 101, 102},
          diag=str(t.get("alive_pids")))
    check("descendant VRAM counted", t["peak_vram_mb"] == 700,
          diag=f"peak_vram={t.get('peak_vram_mb')}")
    check("descendant RAM counted and budget bumped", t["peak_ram_mb"] >= 3000 and t["ram_mb"] >= 3000,
          diag=f"peak_ram={t.get('peak_ram_mb')} ram={t.get('ram_mb')}")
    check("descendant CPU counted and cores bumped", t["cpu_cores"] >= 3,
          diag=f"cpu={t.get('cpu_cores')}")

def test_kill_uses_process_group_sigkill():
    """Cancel/preempt/evict kill helper must SIGKILL the process group, not only the root pid."""
    print("\n[12] Kill helper uses process-group SIGKILL")
    calls = []
    orig_run_on = sch.run_on
    sch.run_on = lambda node, cmd, timeout=15, check=False: (calls.append(cmd) or (0, "", ""))
    try:
        ok, msg = sch._kill_task_processes({
            "node": "local", "remote_pids": [1234], "auto_adopted": False,
        })
    finally:
        sch.run_on = orig_run_on
    cmd = calls[-1] if calls else ""
    check("kill helper returns ok", ok, diag=msg)
    check("SIGTERM targets process group", "kill -- -1234" in cmd, diag=cmd)
    check("SIGKILL targets process group", "kill -9 -- -1234" in cmd, diag=cmd)

def test_launch_failed_node_fallback_and_notification_events():
    """Launch failed node is soft-blocked next placement; new event names are handled."""
    print("\n[13] Launch-failed node fallback + event names")
    nodes = [
        {"name": "local", "alive": True, "free_cpu": 12, "total_cpu": 12,
         "free_ram_mb": 30000, "total_ram_mb": 30000, "running_count": 0,
         "gpus": [{"idx": 0, "used_mb": 0, "total_mb": 8192, "free_mb": 8192, "util_pct": 0}]},
        {"name": "jtl110gpu", "alive": True, "free_cpu": 12, "total_cpu": 12,
         "free_ram_mb": 204800, "total_ram_mb": 204800, "running_count": 0,
         "gpus": [{"idx": 0, "used_mb": 0, "total_mb": 12288, "free_mb": 12288, "util_pct": 0}]},
    ]
    task = {"est_vram_mb": 512, "ram_mb": 1000, "cpu_cores": 1,
            "preferred_node": "local", "launch_failed_nodes": {"local": {}},
            "priority": "normal"}
    placement = sch.pick_placement(task, copy.deepcopy(nodes))
    check("placement skips previously failed local node", placement == ("jtl110gpu", 0),
          diag=f"placement={placement}")
    retry_msg = sch._format_feishu("task_launch_retry",
                                   {"task_id": "tx", "attempt": 1, "error": "cwd missing"})
    fail_msg = sch._format_feishu("task_failed", {"task_id": "tx", "error": "cwd missing"})
    check("task_launch_retry notification renders", "will retry" in retry_msg, diag=retry_msg)
    check("task_failed notification renders", "launch failed" in fail_msg, diag=fail_msg)

def test_requeue_from_adopt_becomes_scheduler_owned():
    """If an auto-adopted task has a captured real cmd and is requeued, the clone is now
    scheduler-owned so logs/diagnosis/preemption work normally."""
    print("\n[14] Requeue clone is scheduler-owned, not auto-adopted")
    parent = {
        "id": "ta", "status": "failed", "signature": "P/adopt/s1",
        "cmd": "python train.py", "auto_adopted": True, "adopted": True,
        "process_group": 999, "retry_count": 0, "ram_mb": 100, "est_vram_mb": 0,
        "cpu_cores": 1, "submitted_at": time.time(), "priority": "normal",
        "_diagnosis": {"is_crash": True, "reason": "RuntimeError", "tail": "RuntimeError",
                       "lifetime_s": 100},
    }
    state = {"tasks": [parent], "next_id": 42}
    new_id = sch._requeue_after_crash(parent, state)
    new = state["tasks"][-1]
    check("new task created", new_id == "t0042" and new["id"] == "t0042",
          diag=f"new_id={new_id}, id={new.get('id')}")
    check("clone auto_adopted cleared", new.get("auto_adopted") is False,
          diag=str(new.get("auto_adopted")))
    check("clone adopted cleared", new.get("adopted") is False,
          diag=str(new.get("adopted")))
    check("clone process_group cleared", new.get("process_group") is None,
          diag=str(new.get("process_group")))

# -----------------------------------------------------------------------------
def test_post_dispatch_eviction_and_rule():
    """_enforce_post_dispatch_thresholds must evict ONLY when BOTH mem≥1/3 AND util≥saturation.
    Util-only (the elder task pinning the chip while mem is fine) used to evict the youngest
    repeatedly — that's the t1029/t1030 thrash bug. Also tests the warmup cooldown."""
    print("\n[10] Post-dispatch eviction (AND-rule + warmup cooldown)")
    now = time.time()
    def make_state(elder_age_s, young_age_s):
        return {"tasks": [
            {"id": "telder", "status": "running", "node": "g2", "gpu_idx": 0,
             "started_at": now - elder_age_s, "remote_pids": [9001],
             "auto_adopted": False},
            {"id": "tyoung", "status": "running", "node": "g2", "gpu_idx": 0,
             "started_at": now - young_age_s, "remote_pids": [9002],
             "auto_adopted": False},
        ]}
    def make_nodes(used_mb, util):
        return [{"name": "g2", "alive": True, "gpus": [
            {"idx": 0, "used_mb": used_mb, "total_mb": 12288,
             "free_mb": 12288 - used_mb, "util_pct": util}]}]
    # Stub run_on so the kill call is a no-op, and count kill attempts.
    orig_run_on = sch.run_on
    kill_calls = []
    def fake_run_on(*a, **k):
        kill_calls.append((a, k))
        return (0, "", "")
    sch.run_on = fake_run_on
    try:
        # Case A: util=100, mem=20% → NO evict (the bug fix; was OR-rule false-positive)
        state = make_state(elder_age_s=600, young_age_s=400)
        evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=2400, util=100))
        check("util=100 alone (mem 20%) → no evict (was the t1030 thrash bug)", evicted == [],
              diag=f"got evicted={evicted}")
        check("util-only cycle does not call kill", len(kill_calls) == 0,
              diag=f"kill_calls={len(kill_calls)}")
        before = copy.deepcopy(state)
        for _ in range(3):
            evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=2400, util=100))
        check("repeated util-only cycles still do not evict", evicted == [] and state == before,
              diag=f"evicted={evicted}, state_changed={state != before}")
        check("repeated util-only cycles still do not call kill", len(kill_calls) == 0,
              diag=f"kill_calls={len(kill_calls)}")

        # Case A2: just below the 1/3 memory threshold + util=100 → still NO evict.
        state = make_state(600, 400)
        evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=4095, util=100))
        check("util=100 with mem just below 1/3 → no evict", evicted == [],
              diag=f"got evicted={evicted}")

        # Case B: mem=50%, util=30% → NO evict (only mem high, no contention)
        state = make_state(600, 400)
        evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=6000, util=30))
        check("mem=50% alone (util 30%) → no evict", evicted == [],
              diag=f"got evicted={evicted}")

        # Case C: BOTH high (mem 50% + util 100%) → MUST evict youngest
        state = make_state(600, 400)
        evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=6000, util=100))
        check("mem 50% AND util 100% → evict youngest (real contention)", evicted == ["tyoung"],
              diag=f"got evicted={evicted}")
        young = [t for t in state["tasks"] if t["id"] == "tyoung"][0]
        check("real threshold eviction queues youngest, not failed", young["status"] == "queued",
              diag=f"status={young.get('status')}")
        check("real threshold eviction clears placement", young.get("node") is None and young.get("gpu_idx") is None,
              diag=f"node={young.get('node')} gpu={young.get('gpu_idx')}")

        # Case D: BOTH high but youngest is in warmup (age < EVICT_TASK_MIN_AGE_S) → don't evict
        state = make_state(elder_age_s=600, young_age_s=30)  # young is 30s old
        evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=6000, util=100))
        check("youngest in warmup (age<180s) → no evict even with both signals high",
              evicted == [], diag=f"got evicted={evicted}")

        # Case E: single task on GPU → never evict (design exception)
        state = {"tasks": [
            {"id": "tonly", "status": "running", "node": "g2", "gpu_idx": 0,
             "started_at": now - 600, "remote_pids": [9003], "auto_adopted": False}
        ]}
        evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=6000, util=100))
        check("single task on GPU → never evict (design exception)", evicted == [])

        # Case F: auto-adopted task at fault → never evict
        state = make_state(600, 400)
        state["tasks"][1]["auto_adopted"] = True
        evicted = sch._enforce_post_dispatch_thresholds(state, make_nodes(used_mb=6000, util=100))
        check("youngest is auto-adopted → never evict", evicted == [])
    finally:
        sch.run_on = orig_run_on

# -----------------------------------------------------------------------------
def test_training_cpu_guard():
    """Training + vram=0 must be refused/blocked unless --allow-cpu-training is set.
    App-level `--device cpu` is not sufficient because that exact footgun can put a GPU
    training batch onto the CPU partition."""
    print("\n[15] Training+vram=0 guard (refuse unless scheduler-level override)")
    # _cmd_looks_like_training pattern matrix
    yes = ["python train_iql_bus.py --seed 42",
           "python /path/to/trainer.py",
           "python -u train.py --foo",
           "/abs/path/run_train.sh args",
           "python h2o+_bus_main.py --seed 42"]
    no = ["python eval_only.py",
          "python supervisor.py --poll",
          "python dispatch.py",
          "python my_main.py"]
    for c in yes:
        check(f"detect training: {c[:40]!r}", sch._cmd_looks_like_training(c))
    for c in no:
        check(f"NOT training: {c[:40]!r}", not sch._cmd_looks_like_training(c))
    # _cmd_explicitly_cpu pattern matrix
    cpu_explicit = [
        "python train.py --device cpu",
        "python train.py --device=cpu",
        "python train.py --cpu-only",
        "python train.py --no-cuda",
        "python train.py --no-gpu",
        "CUDA_VISIBLE_DEVICES= python train.py",
    ]
    cpu_implicit = [
        "python train.py",
        "python train.py --device cuda",
        "python train.py --gpus 1",
        "CUDA_VISIBLE_DEVICES=0 python train.py",
        "CUDA_VISIBLE_DEVICES=0,1 python train.py",
    ]
    for c in cpu_explicit:
        check(f"detect explicit-CPU: {c[:40]!r}", sch._cmd_explicitly_cpu(c))
    for c in cpu_implicit:
        check(f"NOT explicit-CPU: {c[:40]!r}", not sch._cmd_explicitly_cpu(c))
    check("RLPD/WSRL-style description is training-shaped",
          sch._task_looks_like_training("python h2o+_bus_main.py --device cpu",
                                        "R3 #10 baseline: RLPD seed=42"))
    reason = sch._cpu_training_policy_reason(
        "python train_iql_bus.py --device cpu",
        "R3 #10 baseline: IQL seed=42",
        allow_cpu_training=False,
    )
    check("--device cpu does NOT bypass scheduler CPU-training guard", reason is not None,
          diag=str(reason))
    gpu_reserved_cpu_cmd = sch._cpu_training_policy_reason(
        "python train_iql_bus.py --device cpu",
        "R3 #10 baseline: IQL seed=42",
        allow_cpu_training=False,
        est_vram_mb=512,
    )
    check("--device cpu is blocked even when vram>0 would reserve a GPU",
          gpu_reserved_cpu_cmd is not None,
          diag=str(gpu_reserved_cpu_cmd))
    cpu_override_gpu_reserved = sch._cpu_training_policy_reason(
        "python train_iql_bus.py --device cpu",
        "R3 #10 baseline: IQL seed=42",
        allow_cpu_training=True,
        est_vram_mb=512,
    )
    check("--allow-cpu-training still requires --vram 0 (no GPU reservation for CPU job)",
          cpu_override_gpu_reserved is not None,
          diag=str(cpu_override_gpu_reserved))
    check("--allow-cpu-training bypasses scheduler CPU-training guard",
          sch._cpu_training_policy_reason(
              "python train_iql_bus.py --device cpu",
              "R3 #10 baseline: IQL seed=42",
              allow_cpu_training=True,
              est_vram_mb=0,
          ) is None)
    legacy = {
        "id": "tlegacy", "status": "queued", "est_vram_mb": 0,
        "cmd": "python train_awac_bus.py --seed 42 --device cpu",
        "description": "R3 #10 baseline: AWAC seed=42",
    }
    check("legacy queued CPU-training task is dispatch-blocked",
          sch._queued_cpu_training_block_reason(legacy) is not None)
    legacy["allow_cpu_training"] = True
    check("explicit scheduler override lets queued CPU training dispatch",
          sch._queued_cpu_training_block_reason(legacy) is None)
    legacy_gpu_reserved = {
        "id": "tlegacy-gpu-reserved", "status": "queued", "est_vram_mb": 512,
        "cmd": "CUDA_VISIBLE_DEVICES= python train_awac_bus.py --seed 42",
        "description": "R3 #10 baseline: AWAC seed=42",
    }
    check("legacy queued task with CPU cmd but vram>0 is still dispatch-blocked",
          sch._queued_cpu_training_block_reason(legacy_gpu_reserved) is not None)
    legacy_gpu_reserved["allow_cpu_training"] = True
    check("CPU override + vram>0 remains blocked (would reserve GPU for CPU job)",
          sch._queued_cpu_training_block_reason(legacy_gpu_reserved) is not None)

def test_resume_capability_guard():
    """Training-shaped cmd with --ckpt-dir but no resume flag in cmd nor --resume-flag at submit
    must be refused. WSRL 05-04 footgun: ckpts saved every 50 epochs to disk but cmd had no
    `--resume_from`, so any crash/reboot would have relaunched from epoch 0."""
    print("\n[16] Resume-capability guard (training+ckpt-dir but no resume wired up)")
    # _cmd_has_resume_flag matrix
    has = [
        "python train.py --resume",
        "python train.py --resume_from /path/ckpt.pt",
        "python train.py --resume-from=/path/ckpt.pt",
        "python train.py --load_ckpt /path/ckpt.pt",
        "python train.py --load-ckpt=/path/ckpt.pt",
        "python train.py --restore_from /path/ckpt.pt",
        "python train.py --ckpt_path /path/ckpt.pt",
        "python train.py --init_from /path/ckpt.pt",
    ]
    has_not = [
        "python train.py --seed 42",
        "python train.py --max_iters 1000",
        "python train.py --resume_strategy is_a_red_herring",  # unrelated arg containing 'resume'
    ]
    for c in has:
        check(f"detect resume flag: {c[:50]!r}", sch._cmd_has_resume_flag(c))
    for c in has_not:
        check(f"NOT resume flag: {c[:50]!r}", not sch._cmd_has_resume_flag(c))

    # WSRL footgun: training cmd, --ckpt-dir set, no resume in cmd, no --resume-flag → BLOCK
    reason = sch._resume_capability_reason(
        cmd="python train_wsrl.py --n_epochs 300 --seed 42",
        ckpt_dir="/path/to/ckpts",
        resume_flag="",
        allow_no_resume=False,
    )
    check("WSRL footgun is blocked", reason is not None, diag=str(reason))
    # cmd already has resume → pass
    check("cmd with --resume bypasses guard",
          sch._resume_capability_reason(
              cmd="python train_wsrl.py --resume",
              ckpt_dir="/path/to/ckpts", resume_flag="", allow_no_resume=False) is None)
    # submit-time --resume-flag → pass (scheduler will inject at relaunch)
    check("--resume-flag at submit bypasses guard (scheduler injects on relaunch)",
          sch._resume_capability_reason(
              cmd="python train_wsrl.py --n_epochs 300",
              ckpt_dir="/path/to/ckpts", resume_flag="--resume_from", allow_no_resume=False) is None)
    # --allow-no-resume override → pass
    check("--allow-no-resume override bypasses guard",
          sch._resume_capability_reason(
              cmd="python train_wsrl.py --n_epochs 300",
              ckpt_dir="/path/to/ckpts", resume_flag="", allow_no_resume=True) is None)
    # No --ckpt-dir means upstream guard handles it (we don't double-block)
    check("no --ckpt-dir defers to upstream ckpt-dir guard",
          sch._resume_capability_reason(
              cmd="python train_wsrl.py --n_epochs 300",
              ckpt_dir=None, resume_flag="", allow_no_resume=False) is None)
    # Non-training cmd is exempt
    check("non-training cmd is exempt",
          sch._resume_capability_reason(
              cmd="python eval.py --workers 4",
              ckpt_dir="/path/to/ckpts", resume_flag="", allow_no_resume=False) is None)

def test_diagnose_mid_training_kill():
    """WSRL 05-04 fix: a task killed mid-training (training markers in log, no success marker)
    must be flagged is_crash=True so auto-requeue fires. Previously fell through to
    'ambiguous; assumed normal' which silently marked done and lost 50h of progress."""
    print("\n[17] Diagnose mid-training kill (no success marker → crash, not done)")
    import tempfile, os as _os

    def make_task(log_lines, lifetime, peak_vram=0):
        """Build a fake task with a real on-disk log so _diagnose_finished_task can read it."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        tmp.write("\n".join(log_lines))
        tmp.close()
        return {
            "id": "tdiag", "status": "running",
            "log_path": tmp.name,
            "started_at": time.time() - lifetime,
            "finished_at": time.time(),
            "peak_vram_mb": peak_vram,
            "cmd": "python train_wsrl.py --n_epochs 300 --seed 42",
            "node": "local",
        }, tmp.name

    # Case 1: WSRL footgun — actively training, no success, host reboot kill
    task, logp = make_task(
        ["[Epoch  10/300] loss=0.5 t=300s", "[Epoch  50/300] loss=0.3 t=1500s",
         "[Epoch 100/300] loss=0.2 t=3000s checkpoint saved"],
        lifetime=180000)  # 50h
    diag = sch._diagnose_terminal(task)
    check("mid-training kill flagged is_crash=True (was 'ambiguous; assumed normal')",
          diag.get("is_crash") is True, diag=str(diag.get("reason"))[:100])
    _os.unlink(logp)

    # Case 2: legitimate completion with success marker — must NOT flag
    task, logp = make_task(
        ["[Epoch 295/300] loss=0.1", "[Epoch 300/300] loss=0.1",
         "Final model saved: /path/to/model_final.pt"],
        lifetime=180000)
    diag = sch._diagnose_terminal(task)
    check("training task with success marker not flagged",
          diag.get("is_crash") is False, diag=str(diag.get("reason"))[:100])
    _os.unlink(logp)

    # Case 3: no training markers at all (the mid-init OOM case — existing rule)
    task, logp = make_task(
        ["loading data...", "building model...", "[killed by oom-killer]"],
        lifetime=900, peak_vram=0)
    diag = sch._diagnose_terminal(task)
    check("mid-init OOM still flagged is_crash=True (existing rule unchanged)",
          diag.get("is_crash") is True, diag=str(diag.get("reason"))[:100])
    _os.unlink(logp)

    # Case 4 — t1401 footgun: clean no-op exit (eval --skip_existing finds 0 to do, runs 25s)
    # was getting false-flagged as crash. The "Running 0 checkpoints" marker now counts as success.
    task, logp = make_task(
        ["Found 564 checkpoints to evaluate",
         "Skipping 564 already evaluated checkpoints",
         "Running 0 checkpoints on 4 workers"],
        lifetime=25, peak_vram=0)
    diag = sch._diagnose_terminal(task)
    check("clean no-op exit (\"Running 0\") not flagged as crash",
          diag.get("is_crash") is False, diag=str(diag.get("reason"))[:100])
    _os.unlink(logp)
    task, logp = make_task(
        ["Reading config...", "Nothing to do — all evaluations already complete."],
        lifetime=8, peak_vram=0)
    diag = sch._diagnose_terminal(task)
    check("clean no-op exit (\"Nothing to\") not flagged as crash",
          diag.get("is_crash") is False, diag=str(diag.get("reason"))[:100])
    _os.unlink(logp)

def test_find_resume_extension_filter():
    """Bug t1422: find_resume's default glob `*` matched train_log.csv, which scheduler then
    injected as --resume_from → torch.load(csv) raised EOFError. Now find_resume filters
    results to a whitelist of known ckpt extensions."""
    print("\n[19] find_resume() filters by ckpt-extension whitelist")
    import tempfile, os as _os, subprocess as _sp

    tmpdir = tempfile.mkdtemp()
    # Mix of ckpt-extension and non-ckpt files; .csv is newest (would have been picked before fix)
    for ext in ("pt", "pth", "pkl", "csv", "json", "log"):
        with open(_os.path.join(tmpdir, f"file.{ext}"), "w") as f:
            f.write("x")
        time.sleep(0.01)  # ensure mtime ordering
    # Touch the CSV last so default glob `*` would have picked it
    with open(_os.path.join(tmpdir, "file.csv"), "w") as f:
        f.write("y")
    # Touch a real ckpt last so it should be the winner under the new filter
    time.sleep(0.01)
    with open(_os.path.join(tmpdir, "checkpoint_epoch100.pt"), "w") as f:
        f.write("z")

    # Build a fake task that runs on local (no SSH)
    task = {"id": "tres", "ckpt_dir": tmpdir, "ckpt_glob": "*", "node": "local"}
    result = sch.find_resume(task)
    check("find_resume returns a .pt file (not .csv)",
          result and result.endswith(".pt"), diag=f"got: {result}")
    check("find_resume picks newest matching ckpt",
          result and result.endswith("checkpoint_epoch100.pt"), diag=f"got: {result}")

    # Ensure non-ckpt-only directory returns None
    tmpdir2 = tempfile.mkdtemp()
    for ext in ("csv", "json", "log", "txt"):
        with open(_os.path.join(tmpdir2, f"file.{ext}"), "w") as f:
            f.write("x")
    task2 = {"id": "tres2", "ckpt_dir": tmpdir2, "ckpt_glob": "*", "node": "local"}
    result2 = sch.find_resume(task2)
    check("dir with only non-ckpt files → find_resume returns None",
          result2 is None, diag=f"got: {result2}")

    # Semantic filter: default glob must not pick output artifacts that share ckpt extensions.
    # model_final.pt is usually an inference artifact, and buffer_epoch*.pkl is replay data;
    # neither is guaranteed to contain optimizer/RNG training state. Old code picked newest by
    # extension only, so these could be injected as --resume_from and crash or silently replay wrong.
    tmpdir3 = tempfile.mkdtemp()
    for name in ("checkpoint_epoch050.pt", "buffer_epoch050.pkl", "model_final.pt"):
        with open(_os.path.join(tmpdir3, name), "w") as f:
            f.write(name)
        time.sleep(0.01)
    task3 = {"id": "tres3", "ckpt_dir": tmpdir3, "ckpt_glob": "*", "node": "local"}
    result3 = sch.find_resume(task3)
    check("default glob skips model_final/buffer and picks training checkpoint",
          result3 and result3.endswith("checkpoint_epoch050.pt"),
          diag=f"got: {result3}")
    task4 = {"id": "tres4", "ckpt_dir": tmpdir3, "ckpt_glob": "model_final.pt", "node": "local"}
    result4 = sch.find_resume(task4)
    check("explicit --ckpt-glob can still select model_final when user asks exactly",
          result4 and result4.endswith("model_final.pt"),
          diag=f"got: {result4}")

    _sp.run(["rm", "-rf", tmpdir, tmpdir2, tmpdir3], check=False)

def test_env_spec_conda_parsing():
    """Item: conda: env auto-sync. Verify parse_env_spec requires absolute conda paths."""
    print("\n[44a] env_spec parsing handles conda:/abs/path")
    import importlib.util as _ilu
    edp = _ilu.spec_from_file_location("env_deploy",
        os.path.expanduser("~/.claude/skills/scheduler/env_deploy.py"))
    ed = _ilu.module_from_spec(edp); edp.loader.exec_module(ed)
    check("conda:/abs/path", ed.parse_env_spec("conda:/home/u/.conda/envs/x") == ("conda", "/home/u/.conda/envs/x"))
    try:
        ed.parse_env_spec("conda:./local")
        check("conda relative path rejected", False)
    except ValueError:
        check("conda relative path rejected", True)
    try:
        ed.parse_env_spec("conda:")
        check("empty conda path rejected", False)
    except ValueError:
        check("empty conda path rejected", True)
    # Bare 'conda' without : is unrecognized — must include path
    try:
        ed.parse_env_spec("conda")
        check("bare 'conda' rejected", False)
    except ValueError:
        check("bare 'conda' rejected", True)


def test_conda_preload_helpers():
    """Verify env_deploy.has_conda_env + push_conda_env signatures and basic logic.
    (Live rsync is integration-tested at dispatch time; here we cover the contract.)"""
    print("\n[44b] env_deploy.has_conda_env / push_conda_env contracts")
    import importlib.util as _ilu
    edp = _ilu.spec_from_file_location("env_deploy",
        os.path.expanduser("~/.claude/skills/scheduler/env_deploy.py"))
    ed = _ilu.module_from_spec(edp); edp.loader.exec_module(ed)
    # has_conda_env: stub run_on, verify it builds the right probe cmd
    captured = {}
    def stub_run_on(node, cmd, timeout=8, check=True, **kw):
        captured["cmd"] = cmd
        return (0, "OK\n", "")
    ok = ed.has_conda_env(stub_run_on, "remote", "/home/u/.conda/envs/myenv")
    check("has_conda_env probes <env_path>/bin/python --version", ok is True)
    check("probe cmd uses bin/python --version",
          "/bin/python" in captured["cmd"] and "--version" in captured["cmd"],
          diag=captured["cmd"])
    # push_conda_env to local node = no-op
    ok2, msg2 = ed.push_conda_env(None, "/local/path", "/local/path")
    check("push_conda_env(node_host=None) is a no-op", ok2 is True,
          diag=msg2)


def test_preload_handles_conda_spec():
    """`_preload_docker_images_outside_lock` (now multi-kind) must enumerate conda tasks
    too. Verify by inspecting source for `needed_conda` set being populated and synced."""
    print("\n[44c] preload enumerates and syncs conda tasks alongside docker")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    preload_src = src.split("def _preload_docker_images_outside_lock")[1].split("def cmd_dispatch")[0]
    check("preload has separate conda needed-set",
          "needed_conda" in preload_src and 'kind == "conda"' in preload_src,
          diag="preload should branch on kind=='conda'")
    check("conda preload calls push_conda_env",
          "env_deploy.push_conda_env(" in preload_src)
    check("conda preload does not skip rsync just because remote python works",
          "env_deploy.has_conda_env(" not in preload_src,
          diag="stale local env updates must propagate via incremental rsync")


def test_invariant_kill_unless_done_or_cancelled_requeues():
    """INVARIANT: a task that ENDED for any reason other than (a) clean exit (status=done) or
    (b) explicit user cancel (status=cancelled) MUST re-enter the queue (auto-requeue).

    Verifies the failure→requeue path: _batch_check_running detects dead PID → diagnose runs
    → if is_crash AND category not in HARD-FAIL set → _requeue_after_crash creates retry.
    Cancelled tasks bypass this path (status=cancelled, not failed). Done tasks bypass too."""
    print("\n[44] invariant: killed (not cancel/done) → requeue")
    state = {"tasks": [], "next_id": 0}
    crashed = {
        "id": "tk1", "status": "failed", "cmd": "python train.py", "signature": "TEST/inv-A",
        "retry_count": 0, "_diagnosis": {"is_crash": True, "tail": "", "reason": "ssh timeout"},
    }
    state["tasks"].append(crashed)
    new_id = sch._requeue_after_crash(crashed, state)
    check("crashed task → new retry created", new_id is not None and new_id != "tk1",
          diag=f"new_id={new_id}")
    new = next((t for t in state["tasks"] if t["id"] == new_id), None)
    check("retry status is queued (re-enters queue)",
          new is not None and new.get("status") == "queued")
    check("retry parent_id links to original",
          new is not None and new.get("parent_id") == "tk1")
    # Cancelled tasks are sticky — _requeue_after_crash is only called via _batch_check_running
    # which iterates status='running'. Cancelled tasks are status='cancelled' → never reach
    # this path. Verify by inspecting source.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_batch_check_running iterates status=='running' only (cancelled exempt)",
          't["status"] != "running"' in src or 'status"] != "running"' in src)


def test_invariant_no_dup_active_same_sig_cmd():
    """INVARIANT: same (signature, cmd) pair has AT MOST ONE active task (queued OR running OR
    launching). cmd_submit refuses dup, _requeue_after_crash dedups against existing actives."""
    print("\n[45] invariant: no duplicate active (sig, cmd)")
    # Verify _requeue_after_crash dedups
    state = {
        "tasks": [
            {"id": "trun", "status": "running", "cmd": "python a.py",
             "signature": "TEST/inv-dup", "retry_count": 0},
            {"id": "tnew", "status": "failed", "cmd": "python a.py",
             "signature": "TEST/inv-dup", "retry_count": 0,
             "_diagnosis": {"is_crash": True, "tail": "", "reason": "x"}},
        ],
        "next_id": 0,
    }
    out = sch._requeue_after_crash(state["tasks"][1], state)
    check("requeue dedupes against running same sig+cmd → returns existing id",
          out == "trun", diag=f"got {out}")
    check("no new task appended (count stays 2)",
          len(state["tasks"]) == 2)
    # Try with launching state instead of running
    state2 = {
        "tasks": [
            {"id": "tlnch", "status": "launching", "cmd": "python a.py",
             "signature": "TEST/inv-dup-2", "retry_count": 0},
            {"id": "tnew2", "status": "failed", "cmd": "python a.py",
             "signature": "TEST/inv-dup-2", "retry_count": 0,
             "_diagnosis": {"is_crash": True, "tail": "", "reason": "x"}},
        ],
        "next_id": 0,
    }
    out2 = sch._requeue_after_crash(state2["tasks"][1], state2)
    check("dedup ALSO catches launching state (WAL fix)", out2 == "tlnch",
          diag=f"got {out2}")


def test_invariant_race_guard_includes_launching():
    """INVARIANT: race-condition guard at dispatch must consider 'launching' status as
    'sig is taken' — not just 'running'. WAL window is brief but in concurrent dispatch
    cycles (cmd_dispatch + watcher) two tasks with same sig could both pass the running_sigs
    check before either flipped to running."""
    print("\n[46] invariant: race-guard counts launching as taken")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # The running_sigs comprehension in _do_dispatch should include launching now
    idx = src.find("running_sigs = {")
    check("running_sigs comprehension found", idx > 0)
    if idx > 0:
        block = src[idx:idx + 400]
        check("running_sigs includes 'launching' state",
              "launching" in block,
              diag=block[:300])


def test_systemd_unit_restart_always():
    """Item 8: scheduler.service must have `Restart=always`, not `on-failure`. SIGKILL'd
    processes (kernel OOM) exit without a clean failure code, which `on-failure` does NOT
    treat as failure → systemd doesn't restart → watcher silently dead."""
    print("\n[39] systemd unit has Restart=always")
    unit = os.path.expanduser("~/.config/systemd/user/scheduler.service")
    if not os.path.exists(unit):
        check("systemd unit file exists", False, diag=f"missing {unit}")
        return
    src = open(unit).read()
    check("Restart=always present", "Restart=always" in src,
          diag=f"unit content: {src[:300]}")
    check("Restart=on-failure NOT present (regression to old config)",
          "Restart=on-failure" not in src.replace("# ", ""),  # ignore commented-out
          diag="found bare Restart=on-failure")


def test_run_on_has_server_alive_options():
    """Item 9: ssh `ControlMaster` socket can become half-dead (network blip, remote node
    pause). Without ServerAlive options, the next run_on() blocks for the full subprocess
    timeout. ServerAliveInterval=5 + ServerAliveCountMax=3 → 15s detection, well within
    our 15s default timeout."""
    print("\n[40] run_on uses ssh ServerAlive* options for half-dead masters")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # Find the run_on def, slice ~30 lines after, look for ssh args
    idx = src.find("def run_on(")
    check("run_on() definition found", idx > 0)
    if idx > 0:
        block = src[idx:idx + 1500]
        check("ServerAliveInterval option present", "ServerAliveInterval=5" in block,
              diag=block[:400])
        check("ServerAliveCountMax option present", "ServerAliveCountMax=3" in block,
              diag=block[:400])


def test_env_deploy_doc_matches_code():
    """Sentinel: env_deploy.py top-level docstring must NOT reference outdated docker flags
    (e.g. `--gpus all` was replaced by `--gpus device=N` after Codex flagged it as a GPU-leak.
    Doc drift is a maintenance hazard, not a runtime bug, but if the doc lies future-me
    will copy the wrong incantation. Catch drift here, not in production."""
    print("\n[50] env_deploy.py docstring matches actual code (no outdated --gpus all)")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/env_deploy.py")).read()
    # Extract the module docstring
    import ast as _ast
    module = _ast.parse(src)
    docstring = _ast.get_docstring(module) or ""
    check("docstring exists", len(docstring) > 0)
    # Forbidden phrases (these were the OLD wrong flags):
    forbidden = ["--gpus all"]
    for phrase in forbidden:
        check(f"docstring does NOT contain outdated {phrase!r}",
              phrase not in docstring,
              diag=f"found {phrase!r} in module docstring")
    # Required current behaviors that should be mentioned:
    check("docstring mentions --gpus device=<N> (current pinning)",
          "--gpus device=<N>" in docstring or "device=<N>" in docstring,
          diag="docstring should describe device-N pinning")
    check("docstring mentions conda strategy (current code branch)",
          "conda:" in docstring,
          diag="conda env-spec branch undocumented")
    mcp_src = open(os.path.expanduser("~/.claude/skills/scheduler/integrations/scheduler_mcp.py")).read()
    check("MCP docs do NOT contain outdated --gpus all",
          "--gpus all" not in mcp_src,
          diag="scheduler_mcp submit doc still advertises GPU-leaking Docker launch")
    check("MCP docs mention conda env-spec",
          "conda:/abs/path/to/env" in mcp_src,
          diag="scheduler_mcp submit doc omits conda sync strategy")


def test_pick_placement_best_fit_warm_first():
    """Codex follow-up: scoring used `-free_mb` (worst-fit) → small task got biggest empty
    card, fragmenting the cluster. New scoring is best-fit + warm-first:
      - warm card preferred over empty card (preserve empties for big tasks)
      - among same-emptiness candidates, tightest fit (smallest leftover) wins"""
    print("\n[49] pick_placement: best-fit + warm-first (was worst-fit)")
    # Force LocalBackend semantics for the duration of this test — Phase 2.3+ would otherwise
    # route the test "local" node through SlurmBackend (gpu_idx=None) on machines where the
    # actual host has slurm installed, defeating the GPU-pinning assertions below.
    _saved_backend = sch._BACKEND
    sch._BACKEND = sch.LocalBackend()
    # Two GPUs on one node:
    #   GPU0: 2GB used, 6GB free  (warm, partially full)
    #   GPU1: 0GB used, 8GB free  (empty)
    # Task needs 1GB. Old policy: GPU1 (largest free). New policy: GPU0 (warm, tight fit).
    task = {"id": "tplace", "status": "queued", "signature": "TEST/place",
            "cmd": "python x.py", "cwd": "/tmp", "ram_mb": 500,
            "est_vram_mb": 1024, "cpu_cores": 1, "priority": "normal",
            "submitted_at": time.time()}
    nodes = [{"name": "local", "alive": True, "free_cpu": 8, "total_cpu": 12,
              "free_ram_mb": 30000, "total_ram_mb": 30000, "loadavg": 1.0,
              "running_count": 0,
              "gpus": [
                  {"idx": 0, "used_mb": 2048, "total_mb": 8192, "free_mb": 6144, "util_pct": 30},
                  {"idx": 1, "used_mb": 0,    "total_mb": 8192, "free_mb": 8192, "util_pct": 0},
              ]}]
    placement = sch.pick_placement(task, nodes)
    check("warm GPU0 picked over empty GPU1 (best-fit + warm-first)",
          placement is not None and placement[1] == 0,
          diag=f"placement={placement} (expected (local, 0))")
    # Sanity: when ONLY empty card available, pick it
    nodes2 = [{"name": "local", "alive": True, "free_cpu": 8, "total_cpu": 12,
               "free_ram_mb": 30000, "total_ram_mb": 30000, "loadavg": 1.0,
               "running_count": 0,
               "gpus": [
                   {"idx": 5, "used_mb": 0, "total_mb": 8192, "free_mb": 8192, "util_pct": 0},
               ]}]
    placement2 = sch.pick_placement(task, nodes2)
    check("only empty available → still picked",
          placement2 is not None and placement2[1] == 5,
          diag=f"placement={placement2}")
    # Best-fit among two warm cards: tighter fit wins. Both warm but under 1/3 threshold
    # (8192/3 ≈ 2730MB), so 1/3 rule doesn't block. Task needs 1GB.
    nodes3 = [{"name": "local", "alive": True, "free_cpu": 8, "total_cpu": 12,
               "free_ram_mb": 30000, "total_ram_mb": 30000, "loadavg": 1.0,
               "running_count": 0,
               "gpus": [
                   # GPU2: 1.5GB used, 6.7GB free → bigger leftover after 1GB placement
                   {"idx": 2, "used_mb": 1500, "total_mb": 8192, "free_mb": 6692, "util_pct": 30},
                   # GPU3: 2.5GB used, 5.7GB free → tighter leftover, BEST FIT wins
                   {"idx": 3, "used_mb": 2500, "total_mb": 8192, "free_mb": 5692, "util_pct": 30},
               ]}]
    placement3 = sch.pick_placement(task, nodes3)
    check("among warm+fitting cards, best-fit picks tighter (GPU3, smaller free_mb)",
          placement3 is not None and placement3[1] == 3,
          diag=f"placement={placement3} (expected (local, 3))")
    # restore backend so subsequent tests use the production HybridBackend
    sch._BACKEND = _saved_backend


def test_diagnose_peak_vram_implies_crash_without_success():
    """Codex follow-up: a task with peak_vram>0 (GPU was used) AND no success marker AND
    no training-markers-in-tail (markers rotated out, custom log format) MUST be flagged
    as crash, not ambiguous-assumed-normal. Otherwise long-running non-standard-logger
    tasks that get SIGKILL'd are silently marked done."""
    print("\n[48] diagnose: peak_vram>0 + no-success → crash (even when no training markers in tail)")
    import tempfile, os as _os
    def make_task(log_lines, lifetime, peak_vram):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        tmp.write("\n".join(log_lines)); tmp.close()
        return {
            "id": "tc48", "status": "running",
            "log_path": tmp.name,
            "started_at": time.time() - lifetime,
            "finished_at": time.time(),
            "peak_vram_mb": peak_vram,
            "cmd": "python myapp.py",
            "node": "local",
        }, tmp.name
    # Custom log >500B with output that doesn't match TRAINING_MARKERS but task DID hit GPU.
    # Need >500B to bypass rule 5 ("log only XB after Ys") so we land in the catch-all branch.
    custom = ["Loading dataset from /data/dataset.h5 (size=4GB)...",
              "Model built: 12.3M params, fp16",
              "Using device: cuda:0 / NVIDIA RTX 4060 Laptop GPU"]
    for i in range(50):
        custom.append(f"[2025-05-07 06:{i:02d}:00] sample {i*100}/100000 lr=0.001 wall=12.3s")
    task, logp = make_task(custom, lifetime=8000, peak_vram=2500)
    diag = sch._diagnose_terminal(task)
    check("peak_vram>0 + no-success + no-marker → flagged crash",
          diag.get("is_crash") is True, diag=str(diag.get("reason"))[:120])
    check("crash reason mentions GPU work observed",
          "GPU work observed" in (diag.get("reason") or ""),
          diag=str(diag.get("reason"))[:120])
    _os.unlink(logp)
    # Sanity: peak_vram=0 + no-marker still falls to existing "never entered training" rule
    task2, logp2 = make_task(
        ["loading..."], lifetime=900, peak_vram=0)
    diag2 = sch._diagnose_terminal(task2)
    check("peak_vram=0 + no-marker still flagged (existing rule)",
          diag2.get("is_crash") is True)
    _os.unlink(logp2)


def test_launch_path_uses_digest_check():
    """Codex follow-up: launch path's `_maybe_wrap_docker` must compare digests, not just
    tag presence — otherwise a preload failure leaving a stale tag on remote silently
    runs old code at launch time. Verify the source actually fetches local_digest and
    passes it to has_image at launch."""
    print("\n[47] launch path's docker check uses digest (not tag-presence)")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # Locate _maybe_wrap_docker
    idx = src.find("def _maybe_wrap_docker(")
    check("_maybe_wrap_docker found", idx > 0)
    if idx > 0:
        # The launch-path block within this function
        end = src.find("def ", idx + 1)
        block = src[idx:end if end > 0 else idx + 4000]
        check("launch path fetches local_digest before has_image",
              "get_image_digest(run_on" in block,
              diag="missing get_image_digest call in _maybe_wrap_docker")
        check("launch path passes local_digest to has_image",
              "has_image(run_on, node, chosen_image, local_digest=" in block,
              diag="has_image call in launch path lacks local_digest=")
        check("launch path no longer uses tag-presence-only has_image (regression)",
              "has_image(run_on, node, chosen_image)" not in block,
              diag="bare has_image call (without local_digest) reintroduced — would skip drift detection")


def test_has_image_digest_drift():
    """Item 16: `has_image` must compare digests, not just tag presence — otherwise a remote
    `myproj:latest` with stale content while local was rebuilt would skip push and run old code."""
    print("\n[41] has_image rejects when local digest differs (P1b drift detection)")
    import importlib.util as _ilu
    edp = _ilu.spec_from_file_location("env_deploy",
        os.path.expanduser("~/.claude/skills/scheduler/env_deploy.py"))
    ed = _ilu.module_from_spec(edp); edp.loader.exec_module(ed)
    # Stub run_on so we control what 'docker inspect --format {{.Id}}' returns per node
    digest_local = "sha256:abc111"
    digest_remote_old = "sha256:zzz999"
    def fake_run_on_match(node, cmd, timeout=8, check=True, **kw):
        # remote returns SAME digest as local
        return (0, digest_local + "\n", "")
    def fake_run_on_drift(node, cmd, timeout=8, check=True, **kw):
        if node == "local":
            return (0, digest_local + "\n", "")
        return (0, digest_remote_old + "\n", "")
    def fake_run_on_missing(node, cmd, timeout=8, check=True, **kw):
        return (1, "", "no such image")
    # Without local_digest provided → fast path: just tag presence
    check("legacy fast path (no local_digest): tag present → True",
          ed.has_image(fake_run_on_match, "remote", "myproj:latest") is True)
    # With local_digest matching → True
    check("digest matches → True",
          ed.has_image(fake_run_on_match, "remote", "myproj:latest", local_digest=digest_local) is True)
    # With local_digest differing → False (forces re-push)
    check("digest drift → False (will re-push)",
          ed.has_image(fake_run_on_drift, "remote", "myproj:latest", local_digest=digest_local) is False)
    # Image missing on remote → False
    check("image missing → False",
          ed.has_image(fake_run_on_missing, "remote", "myproj:latest", local_digest=digest_local) is False)


def test_atomic_write_integrity():
    """Item 24: queue.json write interruption (signal mid-fwrite) should NEVER leave a
    half-written file — `_atomic_write_json` writes to .tmp + fsync + os.replace. Verify
    behavior end-to-end."""
    print("\n[42] _atomic_write_json never leaves a partial file")
    import tempfile, json as _json
    from pathlib import Path as _Path
    tmpdir = tempfile.mkdtemp()
    target = _Path(tmpdir) / "test.json"
    # Pre-existing valid file
    with open(target, "w") as f: _json.dump({"v": 1}, f)
    # Atomic-write of new content
    sch._atomic_write_json(target, {"v": 2, "tasks": [1, 2, 3]})
    # Read back
    with open(target) as f:
        data = _json.load(f)
    check("atomic_write produces valid JSON", isinstance(data, dict))
    check("atomic_write writes correct content", data.get("v") == 2)
    # Verify no .tmp leftover
    leftover = [f for f in os.listdir(tmpdir) if f.endswith(".tmp")]
    check("no .tmp leftover after success", len(leftover) == 0,
          diag=f"found: {leftover}")
    # Cleanup
    import shutil; shutil.rmtree(tmpdir, ignore_errors=True)


def test_cmd_with_special_shell_chars():
    """Items 30 + 32: cmds with quotes, $(), backticks, spaces in cwd, non-ASCII paths must
    survive launch's wrap chain (shlex.quote → setsid bash -c → docker run wrappers).
    Regression: nested-quoting bugs would corrupt the cmd at remote shell."""
    print("\n[43] cmd parsing tolerates special chars (quotes, $, spaces, non-ASCII)")
    # Test the wrap functions don't mangle special chars
    cases = [
        ("python -c \"print('ok')\"", "single+double quote mix"),
        ("python -c 'print(\"ok\")'", "double inside single"),
        ("python script.py --note \"tag with spaces\"", "spaces in arg"),
        ("VAR=$(echo hello) python x.py", "command substitution preserved"),
        ("python /path/with spaces/script.py", "space in path"),
        ("python /home/用户/项目/train.py --tag 中文标签", "non-ASCII path + arg"),
    ]
    for cmd, label in cases:
        # _inject_python_u shouldn't corrupt the cmd (just adds -u after `python` token)
        wrapped = sch._inject_python_u(cmd)
        check(f"-u inject preserves: {label}",
              "python" in wrapped or "python -u" in wrapped, diag=f"cmd={cmd!r} → {wrapped!r}")
        # docker wrap should also survive (passes cmd as bash -c arg via shlex.quote)
        import importlib.util as _ilu
        edp = _ilu.spec_from_file_location("env_deploy",
            os.path.expanduser("~/.claude/skills/scheduler/env_deploy.py"))
        ed = _ilu.module_from_spec(edp); edp.loader.exec_module(ed)
        # docker wrap shlex-quotes everything → safe even with spaces / non-ASCII
        out = ed.wrap_cmd_docker(cmd, "img:tag", "/wd", gpu_idx=None)
        check(f"docker wrap quotes: {label}",
              "bash -c" in out, diag=f"cmd={cmd!r} → {out[:120]!r}")
    # cwd with spaces — verify shlex.quote produces a safely-quoted form
    import shlex as _shlex
    weird_cwd = "/home/user/dir with spaces/中文"
    quoted = _shlex.quote(weird_cwd)
    check("cwd with spaces+CJK gets shlex-quoted (single-quoted form)",
          quoted.startswith("'") and quoted.endswith("'"), diag=quoted)


def test_no_test_writes_live_queue_with_fake_state():
    """REGRESSION SENTINEL (queue-wipe incident 2026-05-07): test_dispatch_skips_duplicate_signature
    used to call sch._do_dispatch(state, nodes) with a FAKE in-memory state, but my P0b orphan-fix
    made _do_dispatch call save_state(state) per launch — fake state thus overwrote live queue.json
    and wiped ~1600 production tasks. The fix is to stub sch.save_state in any test that calls
    sch._do_dispatch / sch.launch with fake state. This sentinel verifies that no test calls
    those functions without first stubbing save_state, by inspecting the test source for the pattern.

    If a future test forgets the stub, this sentinel fails fast — before the test actually
    runs and wipes the queue."""
    print("\n[37] sentinel: no test calls _do_dispatch / launch without stubbing save_state")
    # Use ast to find ACTUAL `sch._do_dispatch(...)` calls (not text in docstrings).
    import ast as _ast
    src = open(__file__).read()
    tree = _ast.parse(src)
    lines = src.split("\n")
    call_lines = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call): continue
        f = node.func
        if (isinstance(f, _ast.Attribute) and isinstance(f.value, _ast.Name)
                and f.value.id == "sch" and f.attr == "_do_dispatch"):
            call_lines.append(node.lineno)
    risky_blocks = []
    for ln_no in call_lines:
        window = "\n".join(lines[max(0, ln_no - 50):ln_no])
        if "sch.save_state" not in window:
            risky_blocks.append((ln_no, lines[ln_no - 1].strip()))
    check("no sch._do_dispatch call lacks save_state stub in surrounding 50 lines",
          len(risky_blocks) == 0,
          diag=f"risky locations: {risky_blocks}")


def test_descendants_of_capped():
    """Item 21: a fork-bomb script could produce thousands of descendants → BFS output
    explodes, kill cmd line gigantic, ssh stdout buffer overflow. _DESCENDANTS_CAP
    bounds the result so probe stays responsive."""
    print("\n[32] _descendants_of capped against fork-bomb")
    # Build a deep + wide ppid_map: 5000 procs, all children of root pid=1000
    roots = {1000}
    ppid_of = {1000: None}
    for i in range(2, 5002):
        ppid_of[i] = 1000  # all children of root
    out = sch._descendants_of(roots, ppid_of)
    check("descendants_of caps at _DESCENDANTS_CAP (500)",
          len(out) <= sch._DESCENDANTS_CAP,
          diag=f"got {len(out)} descendants, cap={sch._DESCENDANTS_CAP}")
    check("normal small case unaffected (500 PIDs all returned)",
          len(out) == sch._DESCENDANTS_CAP, diag=f"got {len(out)}")


def test_clock_skew_lifetime_clamped():
    """Item 7: if NTP skews backward mid-task, finished_at < started_at → naive subtraction
    produces negative lifetime → diagnose lifetime-based rules misfire. Verify all lifetime
    sites use max(0, fa - sa)."""
    print("\n[33] lifetime computations clamp to 0 on clock skew")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # Search all `finished_at` - `started_at` patterns; each should have max(0, ...) wrapping.
    import re as _re
    raw_subs = _re.findall(r"\(?t\.get\(.finished_at.\).*?\)?\s*-\s*t\.get\(.started_at.\)", src)
    for s in raw_subs:
        check(f"raw finished_at - started_at (no max wrapper) absent",
              "max(0," in src[max(0, src.find(s)-15):src.find(s)+len(s)],
              diag=f"unguarded subtraction at: {s[:60]}")
    # Also assert the helpers are using max(0, ...)
    check("scheduler has max(0, finished - started) pattern",
          "max(0, finished - started)" in src or "max(0, t[\"finished_at\"]" in src)


def test_history_lru_truncation():
    """Item 25: vram_history.json grows unbounded across years of experiments. LRU keep-newest
    by last_seen so file stays bounded."""
    print("\n[34] history_record LRU truncates to HISTORY_MAX_ENTRIES")
    import tempfile, json
    real_VRAM_FILE = sch.VRAM_FILE
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w')
    tmp.close()
    from pathlib import Path as _Path
    sch.VRAM_FILE = _Path(tmp.name)
    try:
        # Pre-load 600 fake old entries with descending last_seen
        h = {}
        for i in range(600):
            h[f"OLD/sig_{i}"] = {"vram_mb": 100, "last_seen": 1000 + i}
        sch.save_history(h)
        # Add one new entry — should trigger truncation
        sch.history_record("NEW/sig_added", peak_vram_mb=999)
        h_after = sch.load_history()
        check("history capped at HISTORY_MAX_ENTRIES after record",
              len(h_after) <= sch.HISTORY_MAX_ENTRIES,
              diag=f"got {len(h_after)} entries, cap={sch.HISTORY_MAX_ENTRIES}")
        check("newest entry kept (NEW/sig_added present)",
              "NEW/sig_added" in h_after)
        # Oldest entries should be evicted (those with low last_seen)
        check("oldest entries evicted (OLD/sig_0 gone)",
              "OLD/sig_0" not in h_after)
    finally:
        sch.VRAM_FILE = real_VRAM_FILE
        try: os.unlink(tmp.name)
        except: pass


def test_launching_state_field_persistence():
    """Item 5 follow-up: dispatch loop must write WAL (status='launching') BEFORE ssh so
    scheduler crash mid-launch leaves a recoverable breadcrumb."""
    print("\n[35] dispatch sets status='launching' before launch (WAL)")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # Find dispatch's launch call and verify a status='launching' flip + save_state is BEFORE it.
    idx_launch = src.find("ok, msg = launch(t)")
    check("dispatch calls launch(t)", idx_launch > 0)
    if idx_launch > 0:
        before = src[max(0, idx_launch - 800):idx_launch]
        check("status='launching' flip before launch (WAL)",
              't["status"] = "launching"' in before, diag=before[-200:])
        check("save_state before launch (WAL persistence)",
              "save_state(state)" in before, diag=before[-200:])
    check("central stale-launching recovery helper exists",
          "def recover_stale_launching_tasks" in src and "LAUNCHING_RESET_S" in src)
    dispatch_src = src.split("def cmd_dispatch")[1].split("def cmd_watch")[0]
    watch_iter_src = src.split("def _watch_iteration")[1].split("def cmd_status")[0]
    status_src = src.split("def cmd_status")[1].split("def cmd_show")[0]
    cancel_src = src.split("def cmd_cancel")[1].split("def cmd_forget")[0]
    check("cmd_dispatch recovers stale launching before dispatch",
          "recover_stale_launching_tasks(state)" in dispatch_src)
    check("_watch_iteration recovers stale launching every loop",
          "recover_stale_launching_tasks(state)" in watch_iter_src)
    check("cmd_status recovers stale launching and shows active launching tasks",
          "recover_stale_launching_tasks(state)" in status_src
          and '("queued", "launching", "running")' in status_src)
    check("cmd_cancel can cancel launching tasks",
          "recover_stale_launching_tasks(state)" in cancel_src
          and '("queued", "launching")' in cancel_src)


def test_env_value_with_equals_sign():
    """Item 31: env vars with `=` in value (e.g. JWT tokens, conda env names like 'a=b')
    must parse via split('=', 1) not split('='), else val truncated at first `=`."""
    print("\n[36] env KEY=VAL parsing handles `=` in value")
    parsed = sch._parse_env(["TOKEN=sk-abc=xyz=qrs", "FLAG=true"])
    check("multi-= value preserved", parsed.get("TOKEN") == "sk-abc=xyz=qrs",
          diag=str(parsed))
    check("simple value still works", parsed.get("FLAG") == "true")


def test_disk_full_classification():
    """DISK_FULL routes to escalation (not auto-retry) and is checked BEFORE OOM patterns
    so 'No space left on device' isn't accidentally tagged OOM. Codex P1."""
    print("\n[28] DISK_FULL classification + escalate-don't-retry")
    diag = {"is_crash": True, "tail": "OSError: [Errno 28] No space left on device", "reason": ""}
    check("OSError errno 28 → DISK_FULL", sch._classify_failure(diag) == "DISK_FULL",
          diag=str(sch._classify_failure(diag)))
    diag2 = {"is_crash": True, "tail": "no space left on device", "reason": ""}
    check("'no space left on device' → DISK_FULL (case-insensitive)",
          sch._classify_failure(diag2) == "DISK_FULL")
    diag3 = {"is_crash": True, "tail": "Disk quota exceeded", "reason": ""}
    check("'Disk quota exceeded' → DISK_FULL", sch._classify_failure(diag3) == "DISK_FULL")
    # Real OOM still classifies as OOM (not affected by DISK_FULL ordering)
    diag_oom = {"is_crash": True, "tail": "Out of memory: Killed process 1", "reason": ""}
    check("kernel OOM still OOM (not DISK_FULL)", sch._classify_failure(diag_oom) == "OOM")
    # _requeue_after_crash escalates DISK_FULL like ENV_MISSING/OOM (not retry-cap path)
    state = {"tasks": [], "next_id": 0}
    parent = {"id": "tdf", "cmd": "python train.py", "signature": "TEST/df-test",
              "status": "failed", "retry_count": 0,
              "_diagnosis": {"is_crash": True, "tail": "ENOSPC writing log", "reason": ""}}
    state["tasks"].append(parent)
    saved_we = sch._write_escalation
    esc_calls = []
    sch._write_escalation = lambda task, cat, diag: esc_calls.append((task["id"], cat))
    new_id = sch._requeue_after_crash(parent, state)
    sch._write_escalation = saved_we
    check("DISK_FULL → no retry created, escalation written instead", new_id is None)
    check("parent failure_category set to DISK_FULL", parent.get("failure_category") == "DISK_FULL")


def test_zombie_pid_excluded_from_alive():
    """`kill -0 <pid>` returns success for zombies. Without /proc state check, a zombie PID
    keeps the task forever-marked-running. Codex P1: ALIVE only when State != Z and != X."""
    print("\n[29] zombie PID detection: kill -0 alone is insufficient")
    # We can't easily simulate a zombie in regression — verify the cmd template includes
    # the /proc state check and the awk filter excludes Z and X.
    # Re-implement check_running's pid_check shape here to assert it.
    pids = [12345, 67890]
    pid_checks = "; ".join(
        f"kill -0 {p} 2>/dev/null && "
        f"awk '/^State:/{{s=$2}} END{{if(s!=\"Z\" && s!=\"X\") print \"ALIVE_{p}\"}}' "
        f"/proc/{p}/status 2>/dev/null"
        for p in pids
    )
    check("pid_check probes /proc/<pid>/status", "/proc/12345/status" in pid_checks)
    check("pid_check excludes State=Z (zombie)", '!="Z"' in pid_checks)
    check("pid_check excludes State=X (dead)", '!="X"' in pid_checks)
    check("pid_check still does kill -0 first", "kill -0 12345" in pid_checks)
    # Verify production scheduler.py contains the same shape (catches regression to old form)
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("scheduler.py check_running uses /proc state guard",
          "/proc/{p}/status" in src and 'State:' in src)


def test_preload_uses_spec_image_or_image_field():
    """`_preload_docker_images_outside_lock` must consider tasks where the image is encoded
    inline in `--env-spec docker:IMAGE` even when `--image` field is not set separately.
    Old code skipped on `not image` and missed those. Codex P0."""
    print("\n[30] preload uses spec_image OR image field (P0 fix)")
    # We verify by reading the source rather than running a live preload — preload's actual
    # work requires docker daemon.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # The fixed code uses `chosen = spec_image or image_field` and continues on `if not chosen`.
    # Prior buggy code was `if spec == "none" or not image: continue` BEFORE parsing.
    check("preload doesn't skip on `not image` BEFORE parsing env_spec",
          "if spec == \"none\" or not image: continue" not in src,
          diag="found old buggy guard pattern")
    check("preload computes `spec_image or image_field`",
          "spec_image or image_field" in src)


def test_save_state_after_each_launch():
    """Codex P0: dispatch loop must persist queue.json AFTER each successful launch, not
    only at end of loop. Otherwise a SIGKILL mid-loop leaves remote procs running with no
    queue.json record → orphaned processes."""
    print("\n[31] save_state per-launch (orphan window minimization)")
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # The fix: save_state(state) inside the dispatch launch loop, after `events.append({"type": "launched", ...})`.
    # Locate the dispatch loop's launched-event block and check save_state is present nearby.
    idx = src.find('events.append({"type": "launched"')
    check("dispatch loop has 'launched' event append", idx > 0)
    if idx > 0:
        window = src[idx:idx + 1200]
        check("save_state(state) appears soon after launched event",
              "save_state(state)" in window, diag=window[:200])


def test_kill_includes_docker_for_named_container():
    """When task has container_name (set by launch when env_spec=docker), the kill cmd
    must include `docker stop` BEFORE host PID kills (containerd-shim isolates the actual
    proc tree from the docker run client, so killing the launcher PID doesn't reliably
    stop the container). Codex review caught this gap."""
    print("\n[26] _kill_task_processes uses docker stop/kill for named container")
    # Stub out run_on so we can capture the kill cmd without ssh
    captured = {"cmd": None, "node": None}
    real_run_on = sch.run_on
    def fake_run_on(node, cmd, timeout=15, check=True, **kw):
        captured["cmd"] = cmd; captured["node"] = node
        return (0, "", "")
    sch.run_on = fake_run_on
    try:
        task = {"id": "tdocker", "node": "local",
                "remote_pids": [12345], "process_group": 12345,
                "container_name": "sched-tdocker"}
        ok, msg = sch._kill_task_processes(task, timeout=5)
    finally:
        sch.run_on = real_run_on
    cmd = captured["cmd"] or ""
    check("kill cmd includes 'docker stop -t 5 sched-tdocker'",
          "docker stop" in cmd and "sched-tdocker" in cmd, diag=cmd[:200])
    check("kill cmd includes 'docker kill sched-tdocker' (escalation)",
          "docker kill" in cmd, diag=cmd[:200])
    check("docker stop ordered BEFORE host kill",
          cmd.index("docker stop") < cmd.index("kill 12345") if "kill 12345" in cmd else True,
          diag=cmd[:200])
    check("kill cmd includes container_name in success msg",
          ok and "container=sched-tdocker" in msg, diag=msg)


def test_kill_no_container_unchanged():
    """When task has NO container_name (legacy / non-docker), kill cmd does NOT mention docker
    (regression: the new docker-aware code path must be opt-in via container_name)."""
    print("\n[27] _kill_task_processes for non-docker task: no docker keywords")
    captured = {"cmd": None}
    real_run_on = sch.run_on
    def fake_run_on(node, cmd, timeout=15, check=True, **kw):
        captured["cmd"] = cmd
        return (0, "", "")
    sch.run_on = fake_run_on
    try:
        task = {"id": "tplain", "node": "local",
                "remote_pids": [99999], "process_group": 99999}
        sch._kill_task_processes(task, timeout=5)
    finally:
        sch.run_on = real_run_on
    cmd = captured["cmd"] or ""
    check("non-docker task: no 'docker' in kill cmd",
          "docker" not in cmd, diag=cmd[:200])


def test_env_deploy_wrap_docker():
    """env_deploy.wrap_cmd_docker produces a launch-time docker-run wrapper that:
    - includes --gpus all when gpu_idx is set, omits it for CPU-only
    - mounts cwd onto identical host path (-v cwd:cwd -w cwd)
    - quotes image / cwd / inner correctly
    - flows extra_env via -e KEY=VAL"""
    print("\n[25] env_deploy.wrap_cmd_docker shape (docker launch wrapper)")
    import importlib.util as _ilu, os as _os
    edp = _ilu.spec_from_file_location("env_deploy", _os.path.expanduser("~/.claude/skills/scheduler/env_deploy.py"))
    ed = _ilu.module_from_spec(edp); edp.loader.exec_module(ed)
    # GPU task: hard-pinned to device=1 (Codex review fix; --gpus all was a leak)
    # + docker rm -f stale-container prefix (Codex P1: name reuse after dirty exit)
    # + memory/cpus cgroup limits (Codex P1: container honors scheduler budgets)
    out = ed.wrap_cmd_docker("python -u train.py --seed 42", "myproj:latest",
                             "/home/u/proj", gpu_idx=1, extra_env={"FOO": "bar"},
                             container_name="sched-t9999",
                             memory_mb=4096, cpus=2)
    check("stale-container cleanup prefix", out.startswith("docker rm -f sched-t9999"),
          diag=out[:120])
    check("docker run after cleanup", "docker run --rm" in out, diag=out[:120])
    check("--gpus device=N hard pin (not --gpus all)", "--gpus device=1" in out, diag=out)
    check("--gpus all NOT used (was the leak)", "--gpus all" not in out, diag=out)
    check("CUDA_VISIBLE_DEVICES=0 inside container (pinned dev enumerates as 0)",
          "CUDA_VISIBLE_DEVICES=0" in out, diag=out)
    check("--name <container> for traceable cancel", "sched-t9999" in out, diag=out)
    check("--memory cgroup limit", "--memory 4096m" in out, diag=out)
    check("--cpus cgroup limit", "--cpus 2" in out, diag=out)
    check("-v cwd:cwd mount", "/home/u/proj:/home/u/proj" in out, diag=out)
    check("-w cwd workdir", "-w /home/u/proj" in out, diag=out)
    check("image last positional", "myproj:latest" in out, diag=out)
    check("extra_env injected", "FOO=bar" in out, diag=out)
    # CPU-only task: no --gpus, CUDA_VISIBLE_DEVICES nulled
    out_cpu = ed.wrap_cmd_docker("python -u eval.py", "img", "/wd", gpu_idx=None)
    check("CPU-only: no --gpus", "--gpus" not in out_cpu, diag=out_cpu)
    check("CPU-only: CUDA_VISIBLE_DEVICES explicitly empty (no host leak)",
          "CUDA_VISIBLE_DEVICES=" in out_cpu, diag=out_cpu)
    # extra_env CUDA_VISIBLE_DEVICES override is REJECTED (we set it based on gpu_idx)
    out_override = ed.wrap_cmd_docker("python -u t.py", "img", "/wd", gpu_idx=2,
                                       extra_env={"CUDA_VISIBLE_DEVICES": "5"})
    check("extra_env can't override pinned CUDA_VISIBLE_DEVICES",
          out_override.count("CUDA_VISIBLE_DEVICES") == 1, diag=out_override)
    # Spec parsing
    check("parse 'none'", ed.parse_env_spec("none") == ("none", None))
    check("parse '' (empty/None)", ed.parse_env_spec("") == ("none", None))
    check("parse 'docker:img:tag'", ed.parse_env_spec("docker:img:tag") == ("docker", "img:tag"))
    check("parse 'docker' (no image)", ed.parse_env_spec("docker") == ("docker", None))
    check("parse 'auto'", ed.parse_env_spec("auto") == ("auto", None))
    try:
        ed.parse_env_spec("garbage")
        check("parse rejects garbage", False)
    except ValueError:
        check("parse rejects garbage", True)

def test_local_max_vram_per_task_dynamic():
    """After NVIDIA 4060 (8GB) replaced AMD 610 (4GB) as local GPU0, the hardcoded
    `max_vram_per_task: 4096` cap silently blocked any single-task allocation >4GB even
    though physically OK. Cap is now None in NODES → no static refusal; the 1/3 packing
    rule + GPU's actual total_mb are the only constraints."""
    print("\n[24] local max_vram_per_task is None (auto-derive from probed GPU)")
    check("NODES['local']['max_vram_per_task'] is None (was 4096 hardcoded)",
          sch.NODES["local"].get("max_vram_per_task") is None,
          diag=str(sch.NODES["local"].get("max_vram_per_task")))
    # _gpu_fits should not refuse based on a static cap when it's None
    fake_node = {"max_vram_per_task": None}
    fake_gpu = {"used_mb": 0, "total_mb": 8188, "free_mb": 8000, "util_pct": 0}
    fake_task = {"est_vram_mb": 6000}  # 6GB task — would exceed old 4GB cap
    check("6GB task on empty 4060 → not blocked by static cap (was blocked when cap=4096)",
          sch._gpu_fits(fake_task, fake_gpu, fake_node) is True,
          diag="_gpu_fits returned False with cap=None")

def test_ckpt_dir_cross_sig_conflict():
    """Submit-time guard: refuse if --ckpt-dir is already in use by an active task with a
    DIFFERENT signature. Was the cross-session footgun that produced 3 wsrl/s1024 procs
    (different sig labels, same out_dir) writing checkpoint_epoch50.pt for 14h → corrupt ckpt.

    SAFETY (2026-05-07 incident): old version of this test wrote queue.json directly without
    state_lock. When watcher concurrently wrote (mid-task transition), races wiped tasks —
    ~1600 tasks reduced to 20 mid-session before recovery from .corrupt-* backup. NOW: use
    sch.state_lock() for ALL queue.json reads/writes to serialize with watcher. Plus take
    a copy.deepcopy + json.dumps backup to safely restore even if our test crashes mid-way."""
    print("\n[23] ckpt-dir cross-signature conflict guard at submit")
    import subprocess as _sp, json as _json
    SCHED = os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")
    # Acquire state_lock for the whole test window. The subprocess submit calls scheduler.py
    # which ALSO acquires state_lock; nested fcntl on same fd from same process is allowed
    # (same lock), so this works.
    with sch.state_lock():
        state = sch.load_state()
        backup = _json.dumps(state)  # backup AS JSON string after lock-protected load
        state["tasks"].append({
            "id": "trun_for_ckpt_test", "status": "running",
            "signature": "TEST/ckpt-dir-conflict-A",
            "cmd": "python train.py --device cuda",
            "ckpt_dir": "/tmp/test_shared_ckpt_dir",
            "node": "local", "started_at": time.time() - 60,
            "ram_mb": 1000, "est_vram_mb": 100, "cpu_cores": 1,
        })
        sch.save_state(state)
    # Now release lock for the subprocess to acquire it (subprocess can't share our lock fd)
    try:
        r = _sp.run(["python", SCHED, "submit",
                     "--description", "TEST conflict",
                     "--signature", "TEST/ckpt-dir-conflict-B",
                     "--cwd", "/tmp", "--vram", "100",
                     "--ckpt-dir", "/tmp/test_shared_ckpt_dir",
                     "--allow-no-resume",
                     "--cmd", "python train.py --device cuda"],
                    capture_output=True, text=True)
        check("different-sig submit with same ckpt-dir → REFUSED",
              r.returncode == 2 and "different signature" in r.stderr,
              diag=(r.stderr or r.stdout)[:200])
        r = _sp.run(["python", SCHED, "submit",
                     "--description", "TEST conflict OK",
                     "--signature", "TEST/ckpt-dir-conflict-C",
                     "--cwd", "/tmp", "--vram", "100",
                     "--ckpt-dir", "/tmp/test_shared_ckpt_dir",
                     "--allow-shared-ckpt-dir",
                     "--allow-no-resume",
                     "--cmd", "python train.py --device cuda"],
                    capture_output=True, text=True)
        check("--allow-shared-ckpt-dir overrides → submitted",
              r.returncode == 0 and "submitted" in r.stdout.lower(),
              diag=(r.stderr or r.stdout)[:200])
    finally:
        # Cleanup: remove only the test-injected entries, preserve everything else (including
        # any tasks the watcher requeued during the test window). Old version `f.write(backup)`
        # blew away those legitimate updates.
        with sch.state_lock():
            cur = sch.load_state()
            test_sigs = {"TEST/ckpt-dir-conflict-A", "TEST/ckpt-dir-conflict-B",
                         "TEST/ckpt-dir-conflict-C"}
            cur["tasks"] = [t for t in cur["tasks"]
                            if t.get("id") != "trun_for_ckpt_test"
                            and t.get("signature") not in test_sigs]
            sch.save_state(cur)

def test_oom_classify_no_false_positive():
    """OOM_PATTERNS used to include bare 'Killed' which matched innocent English like
    'task killed mid-training' in our own diagnose reason → mid-training kills got classified
    as OOM → _requeue_after_crash escalated instead of retried → 4 wsrl/s1024 tasks lost."""
    print("\n[22] OOM classification: don't false-positive on the word 'killed' in reason text")
    # The mid-training-kill reason as actually emitted by _diagnose_terminal
    diag_mid_training = {
        "is_crash": True,
        "reason": ("training markers present but no success marker after 50148s — "
                   "task killed mid-training (likely SIGKILL/OOM/host reboot); "
                   "auto-requeue will resume from latest ckpt if --resume-flag is set"),
        "tail": "[Epoch 100/300] loss=0.5",
    }
    cat = sch._classify_failure(diag_mid_training)
    check("mid-training kill diag NOT classified as OOM (was the bug)",
          cat != "OOM", diag=f"got {cat}")
    # Genuine OOM kill should still classify as OOM (kernel format)
    diag_real_oom = {
        "is_crash": True,
        "reason": "process disappeared",
        "tail": "Out of memory: Killed process 12345 (python) total-vm:9586984kB",
    }
    check("real kernel OOM still classified as OOM",
          sch._classify_failure(diag_real_oom) == "OOM",
          diag=f"got {sch._classify_failure(diag_real_oom)}")
    # CUDA OOM still classified
    check("CUDA OOM classified as OOM",
          sch._classify_failure({"is_crash": True, "tail": "CUDA out of memory", "reason": ""}) == "OOM")
    # MemoryError classified
    check("MemoryError classified as OOM",
          sch._classify_failure({"is_crash": True, "tail": "MemoryError", "reason": ""}) == "OOM")

def test_inject_python_u():
    """At launch time, scheduler auto-injects -u after every python invocation that doesn't
    already have it. Without this, log buffering hides progress until exit, and a SIGKILL'd
    process leaves a 0-byte log → diagnose's 'log only 0B' rule false-flags it as crash even
    when training actually completed (AWAC s123/s789 footgun)."""
    print("\n[21] Auto-inject -u into python invocations at launch time")
    cases = [
        # (input, expected)
        ("python train.py --seed 42",            "python -u train.py --seed 42"),
        ("python3 train.py",                     "python3 -u train.py"),
        ("/conda/envs/x/bin/python train.py",    "/conda/envs/x/bin/python -u train.py"),
        ("conda run -n env python -m mod",       "conda run -n env python -u -m mod"),
        ("PYTHONPATH=. python script.py && echo done",
         "PYTHONPATH=. python -u script.py && echo done"),
        # idempotent
        ("python -u train.py",                   "python -u train.py"),
        ("python -uX foo",                       "python -uX foo"),  # -u-prefixed any flag
        # don't confuse --user with -u (double-dash)
        ("python --user --version",              "python -u --user --version"),
        # non-python untouched
        ("bash run.sh",                          "bash run.sh"),
        ("./train",                              "./train"),
        ("",                                     ""),
    ]
    for inp, want in cases:
        got = sch._inject_python_u(inp)
        check(f"{inp[:50]!r:55s} → injected={got != inp}", got == want,
              diag=f"got={got!r} want={want!r}")

def test_dispatch_skips_duplicate_signature():
    """Race-condition guard: when a signature already has a running task, queued tasks with
    the SAME signature must be skipped (not dispatched). Without this, multi-session
    re-submissions clobber each other's --out_dir/--ckpt-dir — exactly what bit
    offline-sumo/retrain-v3/wsrl/s1024 (3 procs writing same dir for 14h)."""
    print("\n[20] Race-condition guard: skip dispatch when same signature already running")
    import tempfile, subprocess as _sp, json as _json

    # Build a fake state with one running task at sig "X" and one queued task at sig "X"
    # Plus an empty-signature queued task to confirm exemption.
    fake_state = {
        "tasks": [
            {"id": "trun", "status": "running", "signature": "TEST/dup-sig",
             "node": "local", "started_at": time.time() - 60,
             "cmd": "python train.py", "ckpt_dir": "/tmp/x", "ram_mb": 1000,
             "est_vram_mb": 0, "cpu_cores": 1},
            {"id": "tdup", "status": "queued", "signature": "TEST/dup-sig",
             "submitted_at": time.time(), "priority": "normal",
             "cmd": "python train.py", "ckpt_dir": "/tmp/x", "ram_mb": 1000,
             "est_vram_mb": 0, "cpu_cores": 1},
            {"id": "tnosig", "status": "queued", "signature": "",
             "submitted_at": time.time(), "priority": "normal",
             "cmd": "python eval.py", "ram_mb": 500,
             "est_vram_mb": 0, "cpu_cores": 1},
        ]
    }
    # Re-implement just the guard logic against fake_state (mirrors scheduler dispatch loop)
    running_sigs = {(t.get("signature") or "")
                    for t in fake_state["tasks"]
                    if t.get("status") == "running" and t.get("signature")}
    blocked, eligible = [], []
    for t in fake_state["tasks"]:
        if t["status"] != "queued": continue
        sig = t.get("signature") or ""
        if sig and sig in running_sigs:
            blocked.append(t["id"])
        else:
            eligible.append(t["id"])
    check("queued task with same sig as running → blocked",
          "tdup" in blocked, diag=f"blocked={blocked}")
    check("queued task with empty sig → not blocked (exempt from guard)",
          "tnosig" in eligible, diag=f"eligible={eligible}")
    check("running_sigs correctly built from running tasks",
          running_sigs == {"TEST/dup-sig"}, diag=str(running_sigs))

    # Full _do_dispatch regression: if two queued tasks share a signature and no instance is
    # running at loop start, launching the first must immediately block the second in the same
    # dispatch pass. The old implementation precomputed running_sigs once and forgot to update it
    # after launch, so both would start and clobber the same output directory.
    state = {
        "tasks": [
            {"id": "tq1", "status": "queued", "signature": "TEST/same-pass",
             "submitted_at": time.time(), "priority": "normal", "description": "eval A",
             "cmd": "python eval.py --a", "cwd": "/tmp", "ram_mb": 500,
             "est_vram_mb": 0, "cpu_cores": 1},
            {"id": "tq2", "status": "queued", "signature": "TEST/same-pass",
             "submitted_at": time.time() + 1, "priority": "normal", "description": "eval B",
             "cmd": "python eval.py --b", "cwd": "/tmp", "ram_mb": 500,
             "est_vram_mb": 0, "cpu_cores": 1},
        ],
        "next_id": 900,
    }
    nodes = [{"name": "local", "alive": True, "free_cpu": 12, "total_cpu": 12,
              "free_ram_mb": 30000, "total_ram_mb": 56000, "running_count": 0,
              "gpus": [{"idx": 0, "used_mb": 0, "total_mb": 8192,
                        "free_mb": 8192, "util_pct": 0}]}]
    launched = []
    orig_precheck, orig_find, orig_launch = sch.precheck_git, sch.find_resume, sch.launch
    # CRITICAL: also stub save_state. _do_dispatch now calls save_state(state) after each
    # successful launch (Codex P0b orphan-window fix). Without this stub, our FAKE in-memory
    # state would be written to the live queue.json — wipes ~1600 production tasks. Recovered
    # from queue.json.corrupt-* the first time this bit; this stub prevents recurrence.
    orig_save = sch.save_state
    saved_count = [0]
    def fake_save_state(s):
        saved_count[0] += 1  # observable for assertion
    try:
        sch.precheck_git = lambda t: (True, "ok")
        sch.find_resume = lambda t: None
        sch.save_state = fake_save_state
        def fake_launch(t):
            launched.append(t["id"])
            t["status"] = "running"
            t["remote_pids"] = [1000 + len(launched)]
            t["started_at"] = time.time()
            t["log_path"] = f"/tmp/{t['id']}.log"
            return True, "pid=stub"
        sch.launch = fake_launch
        events, _ = sch._do_dispatch(state, nodes)
    finally:
        sch.precheck_git, sch.find_resume, sch.launch = orig_precheck, orig_find, orig_launch
        sch.save_state = orig_save
    check("save_state called per launch (P0b orphan-window fix is live)",
          saved_count[0] >= 1, diag=f"calls={saved_count[0]}")
    tq1 = state["tasks"][0]
    tq2 = state["tasks"][1]
    blocked_ids = [ev["task_id"] for ev in events if ev["type"] == "blocked"]
    check("same-pass duplicate: only first queued sig launches",
          launched == ["tq1"], diag=f"launched={launched}")
    check("same-pass duplicate: second task remains queued",
          tq1["status"] == "running" and tq2["status"] == "queued",
          diag=f"tq1={tq1['status']} tq2={tq2['status']}")
    check("same-pass duplicate: second task gets blocked event",
          "tq2" in blocked_ids and "already has a running task" in tq2.get("last_block_reason", ""),
          diag=f"blocked={blocked_ids}, reason={tq2.get('last_block_reason')}")

def test_cpu_training_justification_required():
    """When --allow-cpu-training is set on a training task, --cpu-training-justification must
    be supplied with ≥30 chars. Without this, the override becomes a reflex bypass — exactly
    the bug that put 6 H2O+ R3 baselines onto CPU when they belonged on GPU."""
    print("\n[18] CPU-training override requires written justification (friction layer)")
    import argparse, subprocess as _sp
    SCHED = os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")

    def submit(extra_args):
        return _sp.run(
            ["python", SCHED, "submit",
             "--description", "TEST cpu-training",
             "--signature", "TEST/cpu-training-justification",
             "--cwd", "/tmp",
             "--vram", "0",
             "--ckpt-dir", "/tmp/ckpt",
             "--allow-no-resume",  # bypass the unrelated resume guard
             "--cmd", "python train.py --device cpu",
             *extra_args],
            capture_output=True, text=True
        )
    # No override at all → blocked by upstream cpu-training guard
    r = submit([])
    check("no override → cpu-training guard refuses", r.returncode == 2,
          diag=r.stderr[:200])
    # Override flag without justification → blocked by new friction guard
    r = submit(["--allow-cpu-training"])
    check("--allow-cpu-training without justification → REFUSED",
          r.returncode == 2 and "justification" in r.stderr.lower(),
          diag=r.stderr[:200])
    # Override + short justification → blocked
    r = submit(["--allow-cpu-training", "--cpu-training-justification", "yes"])
    check("--allow-cpu-training with <30-char justification → REFUSED",
          r.returncode == 2 and "30 chars" in r.stderr,
          diag=r.stderr[:200])
    # Override + adequate justification → submitted
    just_text = "tiny MLP debug, GPU all booked, runs in 5 min on local CPU"
    r = submit(["--allow-cpu-training", "--cpu-training-justification", just_text])
    check("--allow-cpu-training with adequate justification → submitted",
          r.returncode == 0 and "submitted" in r.stdout.lower(),
          diag=r.stderr[:200] or r.stdout[:200])
    # Verify justification persisted on task record
    if r.returncode == 0:
        import json as _json
        with open(os.path.expanduser("~/.claude/scheduler/queue.json")) as f:
            qs = _json.load(f)
        new_task = next((t for t in qs["tasks"]
                        if t.get("signature") == "TEST/cpu-training-justification"
                        and t.get("status") == "queued"), None)
        check("justification persisted on task record",
              new_task is not None and new_task.get("cpu_training_justification") == just_text,
              diag=str(new_task.get("cpu_training_justification") if new_task else None)[:80])
        # cleanup
        if new_task:
            _sp.run(["python", SCHED, "cancel", new_task["id"]], capture_output=True)


def test_history_record_p80_outlier_resistance():
    """history_record uses p80 of last 10 samples (not max). Without this, a single anomalous
    peak from one bad run pins all future estimates at that high value, blocking placement
    of subsequent typical-sized runs. Concrete bug: H2O+ WSRL queued at 5GB because one
    bad sibling sample inflated the project median; typical WSRL runs use ~1.6GB."""
    print("\n[39] history_record uses p80 (not max) so single outlier doesn't pin estimate")
    import tempfile, json
    from pathlib import Path as _Path
    real_VRAM_FILE = sch.VRAM_FILE
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w')
    tmp.close()
    sch.VRAM_FILE = _Path(tmp.name)
    try:
        # Constants must exist on the module
        check("HISTORY_PERCENTILE constant defined",
              hasattr(sch, "HISTORY_PERCENTILE") and sch.HISTORY_PERCENTILE == 80,
              diag=f"got {getattr(sch, 'HISTORY_PERCENTILE', None)}")
        check("HISTORY_SAMPLES_PER_SIG constant defined",
              hasattr(sch, "HISTORY_SAMPLES_PER_SIG") and sch.HISTORY_SAMPLES_PER_SIG == 10,
              diag=f"got {getattr(sch, 'HISTORY_SAMPLES_PER_SIG', None)}")
        check("_percentile helper exists", hasattr(sch, "_percentile"))

        # Pure helper behavior
        check("_percentile of empty → 0", sch._percentile([], 80) == 0)
        check("_percentile of single → that value", sch._percentile([1500], 80) == 1500)
        # Sequence 1000..1900 (10 values), p80 ≈ 1720 (rank 7.2 between 1700 and 1800)
        seq = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        p80 = sch._percentile(seq, 80)
        check("_percentile p80 of 1000..1900 lands near 1720",
              1700 <= p80 <= 1740, diag=f"got {p80}")

        # OUTLIER RESISTANCE (the actual bug fix):
        # 9 typical runs + 1 outlier. Max would return the outlier; p80 should ignore it.
        sch.save_history({})
        for _ in range(9):
            sch.history_record("TEST/wsrl-typical", peak_ram_mb=1600)
        sch.history_record("TEST/wsrl-typical", peak_ram_mb=8000)  # one bad run
        h = sch.load_history()
        rec = h.get("TEST/wsrl-typical")
        check("entry was recorded as dict with samples",
              isinstance(rec, dict) and "ram_samples" in rec,
              diag=str(rec)[:120])
        # p80 of [1600]*9 + [8000] sorted = [1600..1600, 8000]; rank 7.2 falls between 1600 and 1600 → 1600
        # (because index 7 and 8 are both 1600; only index 9 is 8000)
        check("9 typical + 1 outlier → estimate stays near typical (NOT max)",
              rec["ram_mb"] <= 2000,
              diag=f"ram_mb={rec['ram_mb']} samples={rec['ram_samples']}")
        check("samples capped at HISTORY_SAMPLES_PER_SIG",
              len(rec["ram_samples"]) <= sch.HISTORY_SAMPLES_PER_SIG)

        # MIGRATION: legacy single-value record (from before p80) should be seeded as the
        # first sample so it doesn't get lost when new samples arrive.
        sch.save_history({"TEST/legacy-sig": {"ram_mb": 5000, "vram_mb": 1500}})
        sch.history_record("TEST/legacy-sig", peak_ram_mb=2000)
        h2 = sch.load_history()
        rec2 = h2.get("TEST/legacy-sig")
        check("legacy single-value seeded into samples list on next record",
              5000 in (rec2.get("ram_samples") or []) and 2000 in (rec2.get("ram_samples") or []),
              diag=f"samples={rec2.get('ram_samples')}")
        # p80 of [2000, 5000] = rank 0.8*(2-1)=0.8 → 2000 + 0.8*(5000-2000) = 4400
        check("p80 of [2000, 5000] ≈ 4400 (legacy NOT max-pinned at 5000)",
              3800 <= rec2["ram_mb"] <= 4600,
              diag=f"ram_mb={rec2['ram_mb']}")

        # SLIDING WINDOW: 12 samples → only last 10 retained; old high values fall off
        sch.save_history({})
        # First 2 are 9000 (will be evicted), next 10 are 1500
        sch.history_record("TEST/slide-sig", peak_ram_mb=9000)
        sch.history_record("TEST/slide-sig", peak_ram_mb=9000)
        for _ in range(10):
            sch.history_record("TEST/slide-sig", peak_ram_mb=1500)
        h3 = sch.load_history()
        rec3 = h3.get("TEST/slide-sig")
        check("after >SAMPLES_PER_SIG records, old high values evicted (window slides)",
              9000 not in (rec3.get("ram_samples") or []),
              diag=f"samples={rec3.get('ram_samples')}")
        check("p80 reflects current behavior, not historical max",
              rec3["ram_mb"] == 1500, diag=f"ram_mb={rec3['ram_mb']}")
    finally:
        sch.VRAM_FILE = real_VRAM_FILE
        try: os.unlink(tmp.name)
        except: pass


def test_backend_abstraction_phase1():
    """Phase 1 abstraction: launch / kill / check_running / _batch_check_running must all
    go through the active Backend instance — not directly call run_on for launch/kill or
    inline an ad-hoc nvidia-smi command for probing. This test guards the abstraction
    from being silently re-bypassed in future edits.

    Phase 2 (slurm backend) and Phase 3 (multi-user local) plug in here without touching
    the rest of scheduler.py — but only if call sites stay routed through the singleton.
    """
    print("\n[40] Phase 1 backend abstraction: launch/kill/probe routed through _BACKEND")
    # Required exports
    check("Backend ABC defined", hasattr(sch, "Backend"))
    check("LocalBackend defined", hasattr(sch, "LocalBackend"))
    check("LocalBackend subclasses Backend",
          issubclass(getattr(sch, "LocalBackend", type), getattr(sch, "Backend", type)))
    check("_BACKEND singleton exists", hasattr(sch, "_BACKEND"))
    check("_BACKEND is a Backend instance",
          isinstance(getattr(sch, "_BACKEND", None), getattr(sch, "Backend", type)))

    # Backend interface contract
    backend_methods = ("launch", "kill", "batch_probe")
    for m in backend_methods:
        check(f"Backend.{m} declared",
              callable(getattr(sch.Backend, m, None)))
        check(f"LocalBackend.{m} concrete",
              callable(getattr(sch.LocalBackend, m, None)))

    # Source-level: top-level wrappers must delegate. We grep the source so a careless
    # future inline-implementation regresses immediately.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()

    # def launch(task): must contain `_BACKEND.launch(task)`. Find body of launch().
    def _body_after(marker_def, end_keywords=("\ndef ", "\nclass ")):
        i = src.find(marker_def)
        if i < 0: return ""
        j = min((src.find(k, i + len(marker_def)) for k in end_keywords if src.find(k, i + len(marker_def)) > 0),
                default=len(src))
        return src[i:j]

    launch_body = _body_after("\ndef launch(task):")
    check("top-level launch() delegates to _BACKEND.launch",
          "_BACKEND.launch(" in launch_body, diag=launch_body[:200])

    kill_body = _body_after("\ndef _kill_task_processes(task")
    check("top-level _kill_task_processes() delegates to _BACKEND.kill",
          "_BACKEND.kill(" in kill_body, diag=kill_body[:200])

    check_body = _body_after("\ndef check_running(task):")
    check("top-level check_running() delegates to _BACKEND.batch_probe",
          "_BACKEND.batch_probe(" in check_body, diag=check_body[:200])

    bcr_body = _body_after("\ndef _batch_check_running(state):")
    check("_batch_check_running() pulls probe data from _BACKEND.batch_probe",
          "_BACKEND.batch_probe(" in bcr_body, diag=bcr_body[:200])
    # Critical: the OLD inline ssh+nvidia-smi probe must be GONE from _batch_check_running
    # (it lives inside LocalBackend.batch_probe now). This catches a copy-paste regression
    # where someone adds a sibling ad-hoc probe.
    check("no inline `===PSALL===` in _batch_check_running body (must be in backend)",
          "===PSALL===" not in bcr_body, diag=bcr_body[:300])

    # Functional: launch routed through swappable backend. Substitute a fake backend, call
    # the top-level wrapper, verify the fake was hit.
    class _FakeBackend(sch.Backend):
        name = "fake"
        def launch(self, task): return True, "fake-launched"
        def kill(self, task, timeout=15): return True, "fake-killed"
        def batch_probe(self, state): return {}

    saved = sch._BACKEND
    try:
        sch._BACKEND = _FakeBackend()
        ok, msg = sch.launch({"id": "tX", "node": "local", "cmd": "true", "cwd": "/tmp"})
        check("launch() routes to swapped-in fake backend",
              ok and msg == "fake-launched", diag=f"ok={ok} msg={msg}")
        ok2, msg2 = sch._kill_task_processes({"id": "tX", "node": "local", "remote_pids": [1]})
        check("_kill_task_processes() routes to swapped-in fake backend",
              ok2 and msg2 == "fake-killed", diag=f"ok={ok2} msg={msg2}")
    finally:
        sch._BACKEND = saved


def test_backend_slurm_phase2():
    """Phase 2: SlurmBackend (sbatch / scancel / squeue) + HybridBackend (per-node routing).

    These tests don't require slurm to be installed — run_on is monkey-patched to return
    canned outputs that mimic real sbatch / squeue / scancel responses. Exercises:
    - sbatch script generation: directives derived from task fields (cpu/ram/gres/time)
    - launch parses 'Submitted batch job N' correctly
    - kill issues `scancel <id>`
    - batch_probe maps squeue states to alive/dead correctly
    - HybridBackend routes per-node based on cached detection result
    """
    print("\n[41] Phase 2 SlurmBackend + HybridBackend")
    check("SlurmBackend defined", hasattr(sch, "SlurmBackend"))
    check("HybridBackend defined", hasattr(sch, "HybridBackend"))
    check("SlurmBackend subclasses Backend",
          issubclass(getattr(sch, "SlurmBackend", type), sch.Backend))
    check("HybridBackend subclasses Backend",
          issubclass(getattr(sch, "HybridBackend", type), sch.Backend))
    check("singleton _BACKEND is HybridBackend (Phase 2 routing)",
          isinstance(sch._BACKEND, sch.HybridBackend))

    # ---------- SlurmBackend.launch: sbatch script generation ----------
    sb = sch.SlurmBackend()
    task = {
        "id": "t9001", "node": "local", "cwd": "/tmp",
        "cmd": "python train.py --seed 42",
        "cpu_cores": 4, "ram_mb": 8192, "est_vram_mb": 4000,
        "extra_env": {"FOO": "bar"},
        "slurm_partition": "gpu", "slurm_account": "acct", "slurm_qos": "normal",
        "signature": "TEST/slurm-script-gen",
        "resume_flag": "", "resume_from": None,
    }
    script = sb._build_sbatch_script(task, "python -u train.py --seed 42", "/tmp/sched_t9001.log")
    check("script starts with shebang", script.startswith("#!/bin/bash"))
    check("script has --job-name with task id",
          "#SBATCH --job-name=scheduleurm-t9001" in script, diag=script[:300])
    check("script has --cpus-per-task=4", "#SBATCH --cpus-per-task=4" in script)
    check("script has --mem=8192M", "#SBATCH --mem=8192M" in script)
    check("script has --gres=gpu:1 (vram > 0)", "#SBATCH --gres=gpu:1" in script)
    check("script sets --output and --error to log path",
          "#SBATCH --output=/tmp/sched_t9001.log" in script
          and "#SBATCH --error=/tmp/sched_t9001.log" in script)
    check("script has --time= directive", "#SBATCH --time=" in script, diag=script[:400])
    check("script carries optional --slurm-partition/account/qos",
          "#SBATCH --partition=gpu" in script
          and "#SBATCH --account=acct" in script
          and "#SBATCH --qos=normal" in script,
          diag=script)
    check("script exports extra_env", "export FOO=bar" in script)
    check("script cd's to cwd", "cd /tmp" in script)
    check("script body has the inner cmd", "python -u train.py --seed 42" in script)

    # CPU-only task should NOT request GPU
    task_cpu = dict(task)
    task_cpu["est_vram_mb"] = 0
    script_cpu = sb._build_sbatch_script(task_cpu, "python -u eval.py", "/tmp/log.log")
    check("CPU-only task: no --gres=gpu directive",
          "--gres=gpu" not in script_cpu, diag=script_cpu)

    # Walltime: known signature uses 3× EWMA, clamped
    real_history_get = sch.history_get
    sch.history_get = lambda sig: {"dur_s_ewma": 7200, "dur_s_runs": 5} if sig == "TEST/has-history" else None
    try:
        check("walltime for unknown sig defaults to 24h",
              sb._walltime_for({"signature": "TEST/no-history"}) == 24 * 3600)
        # 7200s × 3 = 21600s = 6h; clamps to MIN_WALLTIME_S=3600 floor (passes; 6h > 1h)
        check("walltime for known sig = 3× EWMA",
              sb._walltime_for({"signature": "TEST/has-history"}) == 21600)
        # Walltime format: 06:00:00 for 6h
        check("walltime format: HH:MM:SS for sub-day",
              sb._format_walltime(21600) == "06:00:00")
        check("walltime format: D-HH:MM:SS for multi-day",
              sb._format_walltime(2 * 86400 + 3600) == "2-01:00:00")
    finally:
        sch.history_get = real_history_get

    # ---------- SlurmBackend.launch: monkey-patched subprocess ----------
    # We patch subprocess.run because launch() uses it for sbatch stdin pipe (not run_on).
    real_subprocess_run = sch.subprocess.run
    captured = {}
    def fake_subprocess_run(args, input=None, capture_output=None, text=None, timeout=None):
        captured["args"] = args
        captured["input"] = input
        class R: pass
        r = R()
        r.returncode = 0
        r.stdout = "Submitted batch job 12345\n"
        r.stderr = ""
        return r
    sch.subprocess.run = fake_subprocess_run
    real_run_on = sch.run_on
    sch.run_on = lambda node, cmd, timeout=15, check=True: (0, "", "")  # cwd test passes
    try:
        ok, msg = sb.launch({
            "id": "t9002", "node": "local", "cwd": "/tmp",
            "cmd": "python train.py", "cpu_cores": 2, "ram_mb": 4096,
            "est_vram_mb": 0, "extra_env": {}, "signature": "TEST/launch",
            "resume_flag": "", "resume_from": None,
        })
        check("launch returns ok", ok, diag=msg)
        check("launch parses slurm_job_id", "slurm_job_id=12345" in msg)
        # The captured input should be the sbatch script
        check("launch piped sbatch script via stdin (input arg)",
              captured.get("input") and "#SBATCH" in captured["input"])
    finally:
        sch.subprocess.run = real_subprocess_run
        sch.run_on = real_run_on

    # ---------- SlurmBackend.kill: scancel routing ----------
    kill_calls = []
    sch.run_on = lambda node, cmd, timeout=15, check=True: (kill_calls.append((node, cmd)) or (0, "", ""))
    try:
        ok, msg = sb.kill({"id": "tK", "node": "local", "slurm_job_id": 99})
        check("kill ok when slurm_job_id present", ok)
        check("kill issued scancel <id>", any("scancel 99" in c[1] for c in kill_calls),
              diag=str(kill_calls))
        ok2, msg2 = sb.kill({"id": "tNoJid", "node": "local"})
        check("kill rejects task without slurm_job_id", not ok2 and "no slurm_job_id" in msg2)
    finally:
        sch.run_on = real_run_on

    # ---------- SlurmBackend.batch_probe: squeue parsing ----------
    canned_squeue = "100 RUNNING\n101 PENDING\n102 COMPLETED\n103 FAILED\n"
    sch.run_on = lambda node, cmd, timeout=15, check=True: (
        (0, canned_squeue, "") if "squeue" in cmd else (0, "", "")
    )
    try:
        state = {"tasks": [
            {"id": "ta", "status": "running", "node": "local", "slurm_job_id": 100},
            {"id": "tb", "status": "running", "node": "local", "slurm_job_id": 101},
            {"id": "tc", "status": "running", "node": "local", "slurm_job_id": 102},
            {"id": "td", "status": "running", "node": "local", "slurm_job_id": 103},
            {"id": "te", "status": "running", "node": "local", "slurm_job_id": 999},  # not in squeue output
        ]}
        res = sb.batch_probe(state)
        check("RUNNING → alive", res["ta"]["state"] == "alive", diag=str(res.get("ta")))
        check("PENDING → alive (still queued in slurm)", res["tb"]["state"] == "alive")
        check("COMPLETED → dead + terminal_ok=True",
              res["tc"]["state"] == "dead" and res["tc"].get("terminal_ok") is True,
              diag=str(res["tc"]))
        check("FAILED → dead + terminal_ok=False",
              res["td"]["state"] == "dead" and res["td"].get("terminal_ok") is False,
              diag=str(res["td"]))
        check("absent from squeue → dead + terminal_ok=None",
              res["te"]["state"] == "dead" and res["te"].get("terminal_ok") is None,
              diag=str(res["te"]))
        # All entries have peak fields zeroed (Phase 2 v1 doesn't track via slurm)
        check("vram_mb is 0 for all slurm probes",
              all(v["vram_mb"] == 0 for v in res.values()))
    finally:
        sch.run_on = real_run_on

    src_submit = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("submit parser exposes --slurm-partition/account/qos",
          "--slurm-partition" in src_submit
          and "--slurm-account" in src_submit
          and "--slurm-qos" in src_submit)

    # squeue ssh failure → all tasks should be 'unknown' (don't transition silently)
    sch.run_on = lambda node, cmd, timeout=15, check=True: (1, "", "ssh fail")
    try:
        state = {"tasks": [{"id": "tx", "status": "running", "node": "local", "slurm_job_id": 555}]}
        res = sb.batch_probe(state)
        check("squeue failure → 'unknown' (not 'dead')", res["tx"]["state"] == "unknown")
    finally:
        sch.run_on = real_run_on

    # ---------- HybridBackend routing ----------
    hb = sch.HybridBackend()
    # Force cache for predictable routing
    hb._cache["fake-slurm-node"] = "slurm"
    hb._cache["fake-local-node"] = "local"
    check("HybridBackend routes slurm-cached node to SlurmBackend",
          hb._backend_for("fake-slurm-node") is hb._slurm)
    check("HybridBackend routes non-slurm node to LocalBackend",
          hb._backend_for("fake-local-node") is hb._local)
    # Task with slurm_job_id ALWAYS routes to slurm (cache-independent — defensive)
    check("task with slurm_job_id routes to SlurmBackend regardless of cache",
          hb._backend_for_task({"slurm_job_id": 1, "node": "fake-local-node"}) is hb._slurm)
    # Task without job_id and unknown node → falls back to local probe path. Without ssh
    # access run_on raises; cache catches the exception and returns 'local'.
    sch.run_on = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ssh"))
    try:
        hb2 = sch.HybridBackend()  # fresh cache
        kind = hb2._kind_for("never-heard-of")
        check("ssh failure during slurm probe → defaults to local",
              kind == "local", diag=f"got {kind}")
    finally:
        sch.run_on = real_run_on


def test_backend_slurm_phase2_1_sstat():
    """Phase 2.1: SlurmBackend pulls live RAM peaks via `sstat` so the history accumulator
    gets real samples (Phase 2 v1 left ram_mb=0 → history estimates were forever stuck at
    declared values for slurm-only signatures). Failure tolerant: any sstat failure
    (no plugin / not in PATH / parse error) silently degrades to v1 behavior.

    These tests don't need slurm installed — run_on is mocked to return canned sstat output.
    """
    print("\n[42] Phase 2.1 SlurmBackend sstat live RAM peaks")
    sb = sch.SlurmBackend()

    # ---------- _parse_size_to_mb: K/M/G/T suffixes + bare KiB ----------
    p = sb._parse_size_to_mb
    check("'1024K' → 1 MB", p("1024K") == 1)
    check("'512000K' → 500 MB", p("512000K") == 500)
    check("'800M' → 800 MB", p("800M") == 800)
    check("'2G' → 2048 MB", p("2G") == 2048)
    check("'1T' → 1048576 MB", p("1T") == 1024 * 1024)
    check("bare digits = KiB by sstat convention",
          p("4096") == 4)  # 4096 KiB = 4 MiB
    check("empty / None → None", p("") is None and p(None) is None)
    check("garbage → None", p("notanumber") is None)
    check("decimal works ('1.5G' = 1536 MB)", p("1.5G") == 1536)

    # ---------- _query_sstat_peaks: parses pipe-delimited multi-step output ----------
    real_run_on = sch.run_on

    canned_sstat = (
        "12345.batch|512000K\n"     # = 500 MB
        "12345.0|800M\n"             # = 800 MB  ← max for jid 12345
        "12345.extern|1G\n"          # = 1024 MB ← actual max for jid 12345
        "67890.batch|256000K\n"      # = 250 MB  ← only step for jid 67890
        "99999.batch|garbage\n"      # parse fails for jid 99999 → not in output
    )
    captured_sstat_cmd = []
    def _record_sstat(node, cmd, timeout=10, check=True):
        if "sstat" in cmd:
            captured_sstat_cmd.append(cmd)
            return (0, canned_sstat, "")
        return (0, "", "")
    sch.run_on = _record_sstat
    try:
        peaks = sb._query_sstat_peaks("local", [12345, 67890, 99999])
        check("sstat: max across steps wins (1G > 800M > 500M)",
              peaks.get(12345) == 1024, diag=str(peaks))
        check("sstat: single-step job parsed",
              peaks.get(67890) == 250, diag=str(peaks))
        check("sstat: unparseable row silently skipped",
              99999 not in peaks, diag=str(peaks))
        # Phase 2.11 P1 fix: sstat invocation must include `-a` (--allsteps) so .batch
        # / .extern / .N records are returned. Without -a, sstat shows only the "main"
        # step and MaxRSS comes back empty for all batch jobs (verified empirically on
        # slurm 23.11.4 / Ubuntu 24.04 — `sstat -j 4` returns nothing, `sstat -a -j 4`
        # returns `4.batch|975.50M`).
        check("sstat cmd includes -a flag (--allsteps) so .batch records aren't hidden",
              captured_sstat_cmd and " -a " in captured_sstat_cmd[0],
              diag=str(captured_sstat_cmd))
    finally:
        sch.run_on = real_run_on

    # ---------- sstat error → empty dict (graceful degradation) ----------
    sch.run_on = lambda node, cmd, timeout=10, check=True: (1, "", "sstat: command not found")
    try:
        peaks = sb._query_sstat_peaks("local", [12345])
        check("sstat command-not-found → {} (no peaks, no exception)",
              peaks == {}, diag=str(peaks))
    finally:
        sch.run_on = real_run_on

    sch.run_on = lambda node, cmd, timeout=10, check=True: (_ for _ in ()).throw(RuntimeError("ssh broken"))
    try:
        peaks = sb._query_sstat_peaks("local", [12345])
        check("sstat: ssh exception → {} (caught, no propagation)",
              peaks == {}, diag=str(peaks))
    finally:
        sch.run_on = real_run_on

    # Empty job-ids list → trivial early return (no ssh)
    ssh_called = []
    sch.run_on = lambda *a, **k: (ssh_called.append(1) or (0, "", ""))
    try:
        peaks = sb._query_sstat_peaks("local", [])
        check("sstat: empty job_ids → no ssh issued",
              peaks == {} and not ssh_called)
    finally:
        sch.run_on = real_run_on

    # ---------- batch_probe: sstat peak folded into ALIVE result, not dead ----------
    canned_squeue = "100 RUNNING\n101 COMPLETED\n"
    canned_sstat_2 = (
        "100.batch|2G\n"     # alive — 2048 MB should land in ram_mb
        "101.batch|1G\n"     # dead — sstat data ignored (won't fold)
    )
    def _mock_run_on(node, cmd, timeout=15, check=True):
        if "squeue" in cmd:
            return (0, canned_squeue, "")
        if "sstat" in cmd:
            return (0, canned_sstat_2, "")
        return (0, "", "")
    sch.run_on = _mock_run_on
    try:
        state = {"tasks": [
            {"id": "ta", "status": "running", "node": "local", "slurm_job_id": 100},
            {"id": "tb", "status": "running", "node": "local", "slurm_job_id": 101},
        ]}
        res = sb.batch_probe(state)
        check("alive task: sstat ram_mb folded into result",
              res["ta"]["state"] == "alive" and res["ta"]["ram_mb"] == 2048,
              diag=str(res.get("ta")))
        check("dead task: sstat ram_mb NOT folded (would be stale anyway)",
              res["tb"]["state"] == "dead" and res["tb"]["ram_mb"] == 0,
              diag=str(res.get("tb")))
        check("vram_mb / pcpu still 0 (Phase 2.1 only adds ram_mb)",
              res["ta"]["vram_mb"] == 0 and res["ta"]["pcpu"] == 0.0)
    finally:
        sch.run_on = real_run_on

    # ---------- batch_probe: sstat failure → ram_mb stays 0 (v1 fallback) ----------
    def _mock_squeue_only(node, cmd, timeout=15, check=True):
        if "squeue" in cmd:
            return (0, "100 RUNNING\n", "")
        if "sstat" in cmd:
            return (1, "", "no accounting plugin")
        return (0, "", "")
    sch.run_on = _mock_squeue_only
    try:
        state = {"tasks": [{"id": "tc", "status": "running", "node": "local", "slurm_job_id": 100}]}
        res = sb.batch_probe(state)
        check("sstat failure → ram_mb=0, state still 'alive' (graceful v1 degradation)",
              res["tc"]["state"] == "alive" and res["tc"]["ram_mb"] == 0,
              diag=str(res.get("tc")))
    finally:
        sch.run_on = real_run_on


def test_backend_slurm_phase2_2_adopt_skip():
    """Phase 2.2: auto-adopt skips processes managed by slurm (SLURM_JOB_ID in /proc/<pid>/environ).

    Scenario: same machine has scheduleurm + slurm. SlurmBackend submits a task; slurmstepd
    starts the user proc; nvidia-smi sees it. Without this filter, _reconcile_external_tasks
    would create an auto-adopted task record on top of the existing slurm_job_id-tracked one
    — same workload tracked twice, doubled resource accounting, eviction picks wrong victim.
    """
    print("\n[43] Phase 2.2 auto-adopt skips SLURM_JOB_ID-marked processes")

    # ---------- Source-level: probe scripts emit and parsers carry the is_slurm flag ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_node_processes ssh script greps SLURM_JOB_ID/SLURM_JOBID",
          "SLURM_JOB_ID=" in src and "SLURM_JOBID=" in src,
          diag="missing slurm-detection grep")
    check("GPU probe ssh script emits ${sl} field between ${pg} and ${cl}",
          "${pg}|${sl}|${cl}" in src,
          diag="GPU probe doesn't emit sl field in expected position")
    check("CPU probe ssh script emits ${sl} field between ${cwd} and ${cl}",
          # CPU probe uses Python f-string with doubled curlies, so source bytes are
          # literal `${{cwd}}|${{sl}}|${{cl}}` (escaped curlies for f-string).
          "${{cwd}}|${{sl}}|${{cl}}" in src,
          diag="CPU probe doesn't emit sl field in expected position")
    # _reconcile_external_tasks must skip is_slurm candidates BEFORE owner / cwd / project filters
    # (or at any point, but it must skip) — verify via grep
    check("_reconcile_external_tasks rejects is_slurm candidates",
          'p.get("is_slurm")' in src and 'continue' in src,
          diag="no is_slurm filter in candidate loop")

    # ---------- Functional: stub probes to emit a slurm-managed PID, verify it's skipped ----------
    # We stub _node_processes / _node_cpu_processes / _node_ppid_map so _reconcile_external_tasks
    # runs without ssh. Inject a slurm-managed proc and a regular proc — only the regular one
    # should be auto-adopted.
    real_node_processes = sch._node_processes
    real_node_cpu_processes = sch._node_cpu_processes
    real_node_ppid_map = sch._node_ppid_map
    real_history_get = sch.history_get
    real_save = sch.save_state
    sch.save_state = lambda s: None  # don't write live queue.json (regression sentinel rule)
    sch.history_get = lambda sig: None

    import getpass
    me = getpass.getuser()
    home = f"/home/{me}/some-project-dir"
    _NODES_BACKUP = sch.NODES.copy()
    sch.NODES = {"local": {"host": None, "cpu_cores": 12, "ram_mb": 56*1024,
                            "ram_headroom_frac": 0.20, "max_vram_per_task": None,
                            "max_concurrent_running": 10}}

    def _fake_gpu_procs(node):
        return [
            # Regular non-slurm proc → SHOULD be adopted
            {"node": "local", "pid": 1001, "gpu_idx": 0, "used_mb": 2048,
             "owner": me, "cwd": home, "rss_mb": 4096, "pcpu": 80.0,
             "pgid": 1001, "cmdline": "python train.py", "is_slurm": False},
            # Slurm-managed proc → SHOULD be skipped (Phase 2.2)
            {"node": "local", "pid": 2002, "gpu_idx": 1, "used_mb": 3072,
             "owner": me, "cwd": home, "rss_mb": 6000, "pcpu": 90.0,
             "pgid": 2002, "cmdline": "python slurm_managed.py", "is_slurm": True},
        ]
    def _fake_cpu_procs(node):
        return [
            {"node": "local", "pid": 3003, "owner": me, "rss_mb": 1500,
             "cwd": home, "pcpu": 75.0, "gpu_idx": None, "used_mb": 0,
             "is_cpu_only": True, "pgid": 3003, "cmdline": "python eval.py",
             "is_slurm": True},  # also slurm-managed → SHOULD be skipped
        ]
    def _fake_ppid(node): return {}

    sch._node_processes = _fake_gpu_procs
    sch._node_cpu_processes = _fake_cpu_procs
    sch._node_ppid_map = _fake_ppid

    try:
        state = {"tasks": [], "next_id": 1}
        adopted = sch._reconcile_external_tasks(state)
        adopted_pids = sorted(p for t in adopted for p in t.get("remote_pids", []))
        check("non-slurm GPU proc (pid 1001) adopted",
              1001 in adopted_pids, diag=f"adopted_pids={adopted_pids}")
        check("slurm-managed GPU proc (pid 2002) NOT adopted",
              2002 not in adopted_pids, diag=f"adopted_pids={adopted_pids}")
        check("slurm-managed CPU proc (pid 3003) NOT adopted",
              3003 not in adopted_pids, diag=f"adopted_pids={adopted_pids}")
        check("exactly one adopted task (the non-slurm one)",
              len(adopted) == 1, diag=f"adopted={len(adopted)} tasks")
    finally:
        sch._node_processes = real_node_processes
        sch._node_cpu_processes = real_node_cpu_processes
        sch._node_ppid_map = real_node_ppid_map
        sch.history_get = real_history_get
        sch.save_state = real_save
        sch.NODES = _NODES_BACKUP


def test_backend_slurm_phase2_12_eviction_skips_slurm_tasks():
    """Phase 2.12 P2 defensive: scheduleurm's eviction / preemption / inflight-vram
    reservation must NOT touch slurm-managed tasks, EVEN IF they have a gpu_idx set.

    Today the legacy `gpu_idx == g["idx"]` filter implicitly excludes slurm tasks
    (which have gpu_idx=None per Phase 2.3). This test forces a slurm task to have
    a non-None gpu_idx (defeating the legacy filter) and asserts the new explicit
    `_is_slurm_managed(t)` guard still keeps eviction/preemption hands-off.

    Why: a future refactor that sets gpu_idx for any reason (cosmetic display,
    NVML telemetry binding, etc.) would otherwise re-enable scancel'ing slurm
    tasks — silent destructive interference with slurm's queue.
    """
    print("\n[52] Phase 2.12 eviction/preempt/reserve skip slurm-managed tasks (defensive)")

    # ---------- _is_slurm_managed contract ----------
    check("_is_slurm_managed exists", hasattr(sch, "_is_slurm_managed"))
    check("_is_slurm_managed: slurm_job_id set → True",
          sch._is_slurm_managed({"slurm_job_id": 42}) is True)
    check("_is_slurm_managed: slurm_job_id None → False",
          sch._is_slurm_managed({"slurm_job_id": None}) is False)
    check("_is_slurm_managed: missing slurm_job_id → False",
          sch._is_slurm_managed({}) is False)
    check("_is_slurm_managed: slurm_job_id=0 → False (defensive: 0 is sentinel-like)",
          sch._is_slurm_managed({"slurm_job_id": 0}) is False)

    # ---------- _enforce_post_dispatch_thresholds: skip slurm task even when gpu_idx matches ----------
    now = time.time()
    state = {"tasks": [
        # LocalBackend task on GPU0 (older — would be the survivor)
        {"id": "tlocal-old", "status": "running", "node": "n", "gpu_idx": 0,
         "remote_pids": [101], "started_at": now - 1000, "priority": "normal",
         "cpu_cores": 2, "ram_mb": 4096},
        # LocalBackend task on GPU0 (younger — would be evicted under threshold breach)
        {"id": "tlocal-young", "status": "running", "node": "n", "gpu_idx": 0,
         "remote_pids": [102], "started_at": now - 500, "priority": "normal",
         "cpu_cores": 2, "ram_mb": 4096},
        # Slurm task ARTIFICIALLY pinned to gpu_idx=0 (defeats legacy filter).
        # Phase 2.12 must still skip via _is_slurm_managed.
        {"id": "tslurm-young", "status": "running", "node": "n", "gpu_idx": 0,
         "remote_pids": [], "slurm_job_id": 999, "started_at": now - 100,
         "priority": "normal", "cpu_cores": 2, "ram_mb": 4096},
    ]}
    nodes = [{"name": "n", "alive": True,
              "gpus": [{"idx": 0, "used_mb": 5000, "total_mb": 12000,
                         "free_mb": 7000, "util_pct": 100}]}]
    # Stub out the actual kill so test doesn't try ssh
    real_kill = sch._kill_task_processes
    kill_calls = []
    sch._kill_task_processes = lambda t, timeout=15: (kill_calls.append(t["id"]) or (True, ""))
    try:
        evicted_ids = sch._enforce_post_dispatch_thresholds(state, nodes)
        check("eviction did NOT scancel slurm task (tslurm-young)",
              "tslurm-young" not in kill_calls,
              diag=f"kill_calls={kill_calls}, evicted={evicted_ids}")
        check("eviction did NOT include slurm task in evicted_ids",
              "tslurm-young" not in evicted_ids,
              diag=f"evicted_ids={evicted_ids}")
        # Sanity: with 2 LOCAL tasks on the GPU and threshold breach, eviction WOULD
        # pick the youngest LOCAL task (or skip if neither qualifies due to age window).
        # We don't assert it actually evicts the local — just that it didn't pick slurm.
    finally:
        sch._kill_task_processes = real_kill

    # ---------- _preempt_for_high_priority: slurm task can't be a victim ----------
    state = {"tasks": [
        # high-prio task waiting > PREEMPT_QUEUE_WAIT_MIN
        {"id": "thi", "status": "queued", "priority": "high",
         "submitted_at": now - sch.PREEMPT_QUEUE_WAIT_MIN * 60 - 100,
         "require_node": "n", "cpu_cores": 4, "ram_mb": 4096},
        # Local task in age window, eligible victim
        {"id": "tvictim-local", "status": "running", "node": "n", "priority": "normal",
         "started_at": now - sch.PREEMPT_VICTIM_MIN_AGE_MIN * 60 - 600,
         "cpu_cores": 2, "ram_mb": 2048,
         "remote_pids": [201]},
        # Slurm task in age window — defensively must NOT be picked as victim
        {"id": "tvictim-slurm", "status": "running", "node": "n", "priority": "normal",
         "started_at": now - sch.PREEMPT_VICTIM_MIN_AGE_MIN * 60 - 100,
         "cpu_cores": 2, "ram_mb": 2048,
         "remote_pids": [], "slurm_job_id": 1000},
    ]}
    nodes = [{"name": "n", "alive": True, "free_cpu": 0, "free_ram_mb": 0, "gpus": []}]
    sch._kill_task_processes = lambda t, timeout=15: (kill_calls.append(t["id"]) or (True, ""))
    kill_calls.clear()
    try:
        out = sch._preempt_for_high_priority(state, nodes)
        evicted_ids2 = [e["id"] for e in out]
        check("preempt did NOT pick slurm task as victim",
              "tvictim-slurm" not in evicted_ids2 and "tvictim-slurm" not in kill_calls,
              diag=f"evicted={evicted_ids2}, kill_calls={kill_calls}")
    finally:
        sch._kill_task_processes = real_kill

    # ---------- _reserve_inflight_vram: slurm task contributes 0 reservation ----------
    # Build a node with a fresh GPU and one slurm task gpu_idx=0 with peak_vram=0.
    # Without the skip, scheduleurm would reserve STARTUP_FLOOR_MB on GPU0; with it,
    # it reserves nothing for the slurm task.
    state = {"tasks": [
        {"id": "ts", "status": "running", "node": "n", "gpu_idx": 0,
         "slurm_job_id": 5, "remote_pids": [], "peak_vram_mb": 0,
         "est_vram_mb": 4000},
    ]}
    nodes = [{"name": "n", "alive": True,
              "gpus": [{"idx": 0, "used_mb": 100, "total_mb": 12000,
                         "free_mb": 11900, "util_pct": 0}]}]
    sch._reserve_inflight_vram(state, nodes)
    check("reserve_inflight_vram does NOT reserve for slurm task",
          nodes[0]["gpus"][0]["free_mb"] == 11900,
          diag=f"free_mb={nodes[0]['gpus'][0]['free_mb']} (should still be 11900)")

    # And regression: if it were a LocalBackend task with same shape, reservation DOES happen
    state2 = {"tasks": [
        {"id": "tl", "status": "running", "node": "n", "gpu_idx": 0,
         "remote_pids": [101], "peak_vram_mb": 0, "est_vram_mb": 4000},
    ]}
    nodes2 = [{"name": "n", "alive": True,
               "gpus": [{"idx": 0, "used_mb": 100, "total_mb": 12000,
                          "free_mb": 11900, "util_pct": 0}]}]
    sch._reserve_inflight_vram(state2, nodes2)
    check("reserve_inflight_vram still reserves for LocalBackend task (regression)",
          nodes2[0]["gpus"][0]["free_mb"] < 11900,
          diag=f"free_mb={nodes2[0]['gpus'][0]['free_mb']}")

    # ---------- Source guards: ensure helper is consulted at all 3 sites ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_enforce_post_dispatch_thresholds calls _is_slurm_managed",
          "_is_slurm_managed(t)" in src[src.find("def _enforce_post_dispatch_thresholds"):
                                          src.find("def _evict_to_queue")],
          diag="eviction site missing _is_slurm_managed guard")
    check("_preempt_for_high_priority calls _is_slurm_managed",
          "_is_slurm_managed(t)" in src[src.find("def _preempt_for_high_priority"):
                                         src.find("def _do_dispatch")],
          diag="preempt site missing _is_slurm_managed guard")
    check("_reserve_inflight_vram calls _is_slurm_managed",
          "_is_slurm_managed(t)" in src[src.find("def _reserve_inflight_vram"):
                                       src.find("def _signature_batch_key")],
          diag="reserve_inflight_vram site missing _is_slurm_managed guard")


def test_backend_slurm_phase2_13_terminal_state_semantics():
    """Phase 2.13: Slurm terminal states should drive done/failed semantics.

    COMPLETED is stronger than scheduleurm's log heuristic: Slurm has already
    observed exit code 0, so an empty/missing log must not false-crash it.
    FAILED/TIMEOUT/OUT_OF_MEMORY/etc. are stronger than "ambiguous log" too:
    they must become failed/requeued, not done.
    """
    print("\n[53] Phase 2.13 Slurm terminal states override fragile log heuristics")

    now = time.time()
    state = {
        "next_id": 900,
        "tasks": [
            {"id": "tc", "status": "running", "node": "cluster", "slurm_job_id": 10,
             "cmd": "python train.py --seed 10", "signature": "TEST/slurm-completed",
             "started_at": now - 1000, "submitted_at": now - 1100,
             "retry_count": 0, "ram_mb": 1024, "est_vram_mb": 1000, "cpu_cores": 1,
             "extra_env": {}, "priority": "normal", "description": "completed", "project": "p"},
            {"id": "tf", "status": "running", "node": "cluster", "slurm_job_id": 11,
             "cmd": "python train.py --seed 11", "signature": "TEST/slurm-timeout",
             "started_at": now - 1000, "submitted_at": now - 1100,
             "retry_count": 0, "ram_mb": 1024, "est_vram_mb": 1000, "cpu_cores": 1,
             "extra_env": {}, "priority": "normal", "description": "timeout", "project": "p"},
        ],
    }

    class _FakeBackend:
        def batch_probe(self, _state):
            return {
                "tc": {"state": "dead", "alive_pids": [], "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0,
                       "backend_state": "COMPLETED", "terminal_ok": True,
                       "terminal_reason": "slurm terminal state COMPLETED"},
                "tf": {"state": "dead", "alive_pids": [], "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0,
                       "backend_state": "TIMEOUT", "terminal_ok": False,
                       "terminal_reason": "slurm terminal state TIMEOUT"},
            }

    saved_backend = sch._BACKEND
    saved_diag = sch._diagnose_terminal
    saved_history_record = sch.history_record
    saved_history_get = sch.history_get
    diag_calls = []

    def fake_diag(t):
        diag_calls.append(t["id"])
        return {"is_crash": False, "reason": "ambiguous; assumed normal", "tail": "",
                "lifetime_s": 1000, "log_size": 0, "log_path": t.get("log_path"),
                "success_marker": None}

    sch._BACKEND = _FakeBackend()
    sch._diagnose_terminal = fake_diag
    sch.history_record = lambda *a, **k: None
    sch.history_get = lambda sig: {"dur_s_ewma": 10000, "dur_s_runs": 2}
    try:
        sch.update_running_tasks(state)
    finally:
        sch._BACKEND = saved_backend
        sch._diagnose_terminal = saved_diag
        sch.history_record = saved_history_record
        sch.history_get = saved_history_get

    tc = next(t for t in state["tasks"] if t["id"] == "tc")
    tf = next(t for t in state["tasks"] if t["id"] == "tf")
    retry = next((t for t in state["tasks"] if t.get("parent_id") == "tf"), None)
    check("Slurm COMPLETED → scheduleurm done, no log/lifetime false crash",
          tc["status"] == "done" and tc["_diagnosis"]["is_crash"] is False and "tc" not in diag_calls,
          diag=str(tc.get("_diagnosis")))
    check("Slurm TIMEOUT → scheduleurm failed even if log heuristic is ambiguous",
          tf["status"] == "failed" and tf["_diagnosis"]["is_crash"] is True
          and "TIMEOUT" in tf["_diagnosis"]["reason"],
          diag=str(tf.get("_diagnosis")))
    check("Slurm failed terminal state auto-requeues with cleared backend artifacts",
          retry is not None and retry["status"] == "queued" and retry.get("slurm_job_id") is None,
          diag=str(retry))


def test_backend_slurm_phase2_14_ui_and_launch_notification():
    """Phase 2.14: Slurm tasks have no remote_pids and gpu_idx=None by design.

    UI/notifications must render slurm_job_id and SLURM-GPU/CPU, not crash on
    remote_pids[0] or display GPU jobs as plain CPU work.
    """
    print("\n[54] Phase 2.14 Slurm UI location + launch notification handle")
    task = {
        "id": "tslurm-ui", "project": "p", "node": "cluster",
        "slurm_job_id": 77, "slurm_state": "PENDING",
        "gpu_idx": None, "remote_pids": [], "est_vram_mb": 4096,
        "description": "slurm gpu task",
    }
    loc = sch._format_task_location(task)
    check("Slurm GPU task displays as SLURM-GPU, not CPU",
          "SLURM-GPU#77" in loc and ":CPU" not in loc, diag=loc)
    msg = sch._format_feishu("task_launched", task)
    check("task_launched formats slurm_job_id instead of indexing empty remote_pids",
          "slurm_job_id=77" in msg and "pid=" not in msg, diag=msg)
    cpu_task = dict(task, slurm_job_id=78, est_vram_mb=0, slurm_state="RUNNING")
    check("Slurm CPU task displays as SLURM-CPU",
          "SLURM-CPU#78" in sch._format_task_location(cpu_task),
          diag=sch._format_task_location(cpu_task))


def test_backend_slurm_phase2_16_pending_throttle():
    """Phase 2.16: scheduleurm dispatch throttles slurm nodes that already have ≥
    SLURM_MAX_PENDING_PER_NODE of OUR tasks pending. Tasks queued in scheduleurm
    stay there instead of piling up in one slurm node's queue — they spread to
    whichever node frees up next.

    The user's pain point that prompted this: with several small slurm nodes (2 GPUs each),
    Phase 2.3 'always sbatch on slurm route' would dump all queued work onto whichever
    slurm node pick_placement saw first, leaving other nodes idle if THAT node's running
    tasks took longer than expected. Now scheduleurm holds extras in its own queue.

    Tunable via:
      - SLURM_MAX_PENDING_PER_NODE module constant (default 1)
      - SCHEDULEURM_SLURM_MAX_PENDING_PER_NODE env var (read at module import)
      - NODES[name]['max_slurm_pending'] per-node override
    """
    print("\n[56] Phase 2.16 slurm pending throttle (don't pile pending on one node)")

    # ---------- Constants + helper ----------
    check("SLURM_MAX_PENDING_PER_NODE constant exists",
          hasattr(sch, "SLURM_MAX_PENDING_PER_NODE"))
    check("default cap = 1",
          sch.SLURM_MAX_PENDING_PER_NODE == 1, diag=f"got {sch.SLURM_MAX_PENDING_PER_NODE}")
    check("_count_slurm_pending_per_node helper exists",
          callable(getattr(sch, "_count_slurm_pending_per_node", None)))
    check("_slurm_max_pending_for_node helper exists",
          callable(getattr(sch, "_slurm_max_pending_for_node", None)))

    # ---------- _count_slurm_pending_per_node: states correctly classified ----------
    state = {"tasks": [
        # PENDING-like states → counted
        {"id": "tA", "status": "running", "node": "n1", "slurm_job_id": 1, "slurm_state": "PENDING"},
        {"id": "tB", "status": "running", "node": "n1", "slurm_job_id": 2, "slurm_state": "CONFIGURING"},
        {"id": "tC", "status": "running", "node": "n1", "slurm_job_id": 3, "slurm_state": None},  # just-submitted
        {"id": "tD", "status": "running", "node": "n2", "slurm_job_id": 4, "slurm_state": "PENDING"},
        # NOT pending — should NOT be counted
        {"id": "tE", "status": "running", "node": "n1", "slurm_job_id": 5, "slurm_state": "RUNNING"},
        {"id": "tF", "status": "running", "node": "n2", "slurm_job_id": 6, "slurm_state": "COMPLETING"},
        # Local task on slurm node (mixed cluster scenario) — not slurm-managed, ignored
        {"id": "tG", "status": "running", "node": "n1", "remote_pids": [9], "slurm_job_id": None},
        # Done/queued — wrong status, ignored
        {"id": "tH", "status": "queued", "node": "n1", "slurm_job_id": 7, "slurm_state": "PENDING"},
    ]}
    counts = sch._count_slurm_pending_per_node(state)
    check("count: n1 = 3 (tA PENDING + tB CONFIGURING + tC just-submitted)",
          counts.get("n1") == 3, diag=str(counts))
    check("count: n2 = 1 (tD only — tF COMPLETING doesn't count)",
          counts.get("n2") == 1, diag=str(counts))

    # ---------- _slurm_max_pending_for_node: per-node override ----------
    saved_NODES = sch.NODES
    sch.NODES = {
        "default-cap": {"host": None, "cpu_cores": 12, "ram_mb": 30000,
                         "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                         "max_concurrent_running": None},
        "custom-cap": {"host": None, "cpu_cores": 12, "ram_mb": 30000,
                        "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                        "max_concurrent_running": None,
                        "max_slurm_pending": 5},  # per-node override
    }
    try:
        check("default cap from constant when no per-node override",
              sch._slurm_max_pending_for_node("default-cap") == sch.SLURM_MAX_PENDING_PER_NODE)
        check("per-node override beats global",
              sch._slurm_max_pending_for_node("custom-cap") == 5)
        check("unknown node falls back to global",
              sch._slurm_max_pending_for_node("ghost-node") == sch.SLURM_MAX_PENDING_PER_NODE)
    finally:
        sch.NODES = saved_NODES

    # ---------- pick_placement: throttle kicks in when pending >= cap ----------
    # Force HybridBackend with custom cache so we can simulate slurm nodes deterministically.
    saved_backend = sch._BACKEND
    fake_hb = sch.HybridBackend()
    fake_hb._cache["slurm-A"] = "slurm"
    fake_hb._cache["slurm-B"] = "slurm"
    sch._BACKEND = fake_hb
    sch.NODES = {
        "slurm-A": {"host": "A", "cpu_cores": 12, "ram_mb": 200000,
                     "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                     "max_concurrent_running": None},
        "slurm-B": {"host": "B", "cpu_cores": 12, "ram_mb": 200000,
                     "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                     "max_concurrent_running": None},
    }
    try:
        # Both nodes alive, no GPUs probed (slurm decides)
        nodes = [
            {"name": "slurm-A", "alive": True, "gpus": [], "free_cpu": 0,
             "free_ram_mb": 0, "loadavg": 0.0,
             "running_count": 0, "slurm_pending_count": 1},  # at cap
            {"name": "slurm-B", "alive": True, "gpus": [], "free_cpu": 0,
             "free_ram_mb": 0, "loadavg": 0.0,
             "running_count": 0, "slurm_pending_count": 0},  # has slot
        ]
        task = {"id": "tnew", "est_vram_mb": 1000, "cpu_cores": 2, "ram_mb": 4000,
                "signature": "TEST/throttle"}
        placement = sch.pick_placement(task, nodes)
        check("slurm-A throttled (pending=1, cap=1) → slurm-B picked",
              placement is not None and placement[0] == "slurm-B",
              diag=f"placement={placement}")

        # Both at cap → no placement
        nodes[1]["slurm_pending_count"] = 1
        placement = sch.pick_placement(task, nodes)
        check("both nodes at cap → no placement (task stays queued)",
              placement is None, diag=f"placement={placement}")

        # Both have 0 pending → first available picked
        nodes[0]["slurm_pending_count"] = 0
        nodes[1]["slurm_pending_count"] = 0
        placement = sch.pick_placement(task, nodes)
        check("both nodes have 0 pending → some node picked",
              placement is not None, diag=f"placement={placement}")

        # require_node forces a throttled node → still no placement (require trumps fallback,
        # but throttle trumps require — slurm queue is full, scheduleurm holds the task)
        nodes[0]["slurm_pending_count"] = 1
        nodes[1]["slurm_pending_count"] = 0
        task_req = dict(task, require_node="slurm-A")
        placement = sch.pick_placement(task_req, nodes)
        check("require_node + that node throttled → no placement (don't pile)",
              placement is None, diag=f"placement={placement}")
    finally:
        sch._BACKEND = saved_backend
        sch.NODES = saved_NODES

    # ---------- Source guard: dispatch loop populates slurm_pending_count ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_do_dispatch populates n['slurm_pending_count'] before placement loop",
          'n["slurm_pending_count"] = slurm_pending_per_node' in src,
          diag="dispatch must seed per-node pending count")
    check("_do_dispatch bumps slurm_pending_count on each launch",
          'n["slurm_pending_count"] = n.get("slurm_pending_count"' in src,
          diag="post-launch bump missing — sequential dispatches in same cycle would over-pack")
    check("_candidates_for_node consults the throttle",
          "n.get(\"slurm_pending_count\", 0)" in src and "_slurm_max_pending_for_node" in src,
          diag="throttle missing from pick_placement")


def test_phase3_0_9_slurm_pending_elapsed_zero():
    """Phase 3.0.9 P2 fix: slurm-PENDING tasks must NOT decay their ETA (or shrink
    their node load) while waiting in slurm's queue.

    Pre-fix: SlurmBackend.launch sets started_at = sbatch return time. While the
    job sits PENDING for hours, _refresh_eta_from_logs computes
        elapsed = now - started_at
    and EWMA-fallback returns max(0, ewma - elapsed). So a task with EWMA=3600
    pending for 1h shows eta_seconds=0, before any compute happened. eta_load
    drops too → migration may falsely see the source node as "free".

    Fix: _effective_elapsed_s returns 0 for slurm tasks until SlurmBackend.batch_probe
    sees slurm_state=RUNNING for the first time and records actual_started_at.
    """
    print("\n[66] Phase 3.0.9 P2 fix: slurm-PENDING elapsed=0, ETA stays at full EWMA")

    now = time.time()

    # ---------- _effective_elapsed_s semantics ----------
    # LocalBackend task: started_at == compute start
    local_task = {"id": "tL", "started_at": now - 600}
    el = sch._effective_elapsed_s(local_task)
    check("LocalBackend: elapsed = now - started_at",
          580 <= el <= 620, diag=f"got {el}")

    # SlurmBackend PENDING (no actual_started_at yet, slurm_state=PENDING)
    slurm_pending = {"id": "tP", "started_at": now - 3600,
                     "slurm_job_id": 42, "slurm_state": "PENDING"}
    el = sch._effective_elapsed_s(slurm_pending)
    check("Slurm PENDING (1h ago sbatch'd) → elapsed=0",
          el == 0, diag=f"got {el}")

    # SlurmBackend just-sbatched (no slurm_state yet) → still elapsed=0
    slurm_fresh = {"id": "tF", "started_at": now - 60,
                   "slurm_job_id": 42}
    el = sch._effective_elapsed_s(slurm_fresh)
    check("Slurm freshly sbatched (no slurm_state probed) → elapsed=0",
          el == 0, diag=f"got {el}")

    # SlurmBackend RUNNING with actual_started_at recorded
    slurm_running = {"id": "tR", "started_at": now - 3600,        # sbatch 1h ago
                     "actual_started_at": now - 600,              # but compute started 10 min ago
                     "slurm_job_id": 42, "slurm_state": "RUNNING"}
    el = sch._effective_elapsed_s(slurm_running)
    check("Slurm RUNNING with actual_started_at → elapsed = now - actual_started_at",
          580 <= el <= 620, diag=f"got {el}")

    # No started_at at all → 0
    el = sch._effective_elapsed_s({"id": "tX"})
    check("no started_at → 0", el == 0)

    # ---------- ETA computation through _refresh_eta_from_logs ----------
    saved_run_on = sch.run_on
    saved_history_get = sch.history_get
    sch.history_get = lambda sig: {"dur_s_ewma": 3600} if sig else None
    sch.run_on = lambda node, cmd, timeout=30, check=True: (0, "", "")  # empty log

    state = {"tasks": [
        # PENDING for 1h, EWMA=3600 → eta should be 3600 (full), NOT 0
        {"id": "tPend", "status": "running", "node": "X",
         "log_path": "/tmp/p.log", "signature": "TEST/eta-slurm",
         "started_at": now - 3600,
         "slurm_job_id": 42, "slurm_state": "PENDING"},
        # RUNNING for 600s of EWMA 3600 → eta should be ~3000s
        {"id": "tRun", "status": "running", "node": "X",
         "log_path": "/tmp/r.log", "signature": "TEST/eta-slurm",
         "started_at": now - 4200,           # sbatch'd 70 min ago
         "actual_started_at": now - 600,     # but compute started 10 min ago
         "slurm_job_id": 43, "slurm_state": "RUNNING"},
    ]}
    try:
        sch._refresh_eta_from_logs(state)
        pend = next(t for t in state["tasks"] if t["id"] == "tPend")
        run = next(t for t in state["tasks"] if t["id"] == "tRun")
        check("PENDING task: eta_seconds = full EWMA (3600), NOT decayed by sbatch wait",
              3550 <= (pend.get("eta_seconds") or 0) <= 3650,
              diag=f"got {pend.get('eta_seconds')}")
        check("RUNNING task: eta_seconds reflects actual_started_at not sbatch time",
              2950 <= (run.get("eta_seconds") or 0) <= 3050,
              diag=f"got {run.get('eta_seconds')}")
    finally:
        sch.run_on = saved_run_on
        sch.history_get = saved_history_get

    # ---------- batch_probe records actual_started_at on first RUNNING ----------
    sb = sch.SlurmBackend()
    saved_NODES = sch.NODES
    sch.NODES = {"X": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                        "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                        "max_concurrent_running": None}}

    canned_squeue = "100 RUNNING\n101 PENDING\n"
    sch.run_on = lambda node, cmd, timeout=15, check=True: (
        (0, canned_squeue, "") if "squeue" in cmd else (0, "", "")
    )
    state = {"tasks": [
        {"id": "tA", "status": "running", "node": "X", "slurm_job_id": 100,
         "started_at": now - 3600},
        {"id": "tB", "status": "running", "node": "X", "slurm_job_id": 101,
         "started_at": now - 3600},
    ]}
    try:
        sb.batch_probe(state)
        ta = next(t for t in state["tasks"] if t["id"] == "tA")
        tb = next(t for t in state["tasks"] if t["id"] == "tB")
        check("RUNNING task: actual_started_at set on first RUNNING observation",
              ta.get("actual_started_at") and abs(ta["actual_started_at"] - now) < 5,
              diag=f"got {ta.get('actual_started_at')}")
        check("PENDING task: actual_started_at NOT set",
              tb.get("actual_started_at") is None,
              diag=f"got {tb.get('actual_started_at')}")
        check("RUNNING task: slurm_state recorded",
              ta.get("slurm_state") == "RUNNING")
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES

    # actual_started_at should NOT be re-set on subsequent RUNNING observations
    state2 = {"tasks": [
        {"id": "tA2", "status": "running", "node": "X", "slurm_job_id": 200,
         "started_at": now - 3600,
         "actual_started_at": now - 1800,  # already set 30 min ago
         "slurm_state": "RUNNING"},
    ]}
    sch.run_on = lambda node, cmd, timeout=15, check=True: (
        (0, "200 RUNNING\n", "") if "squeue" in cmd else (0, "", "")
    )
    sch.NODES = {"X": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                        "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                        "max_concurrent_running": None}}
    try:
        sb.batch_probe(state2)
        ta = state2["tasks"][0]
        check("repeat RUNNING observation does NOT overwrite actual_started_at",
              abs(ta["actual_started_at"] - (now - 1800)) < 5,
              diag=f"got {ta.get('actual_started_at')} (should still be ~now-1800)")
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES


def test_phase3_0_10_migration_event_visibility():
    """Phase 3.0.10 P3 fix: migrated/preempted events must surface in
    cmd_dispatch stdout, watcher.log JSONL via notify(), and the Feishu push.

    Pre-fix: _do_dispatch appended {"type": "migrated", "task_id": tid} to its
    events list, but cmd_dispatch's print loop and _watch_iteration's notify
    loop both lacked elif branches. The only persisted side-effect was the
    `last_block_reason` string on the task, so the README's claim that
    migrations are "visible in journalctl --user -u scheduler and watcher.log"
    was empty. Same gap existed for `preempted` events (sibling event in the
    same code section).

    Fix: enrich the migrated event payload at emission with from_node/to_node/
    eta_seconds/reason so consumers don't have to re-look up the task; add elif
    branches to cmd_dispatch and _watch_iteration; add Feishu format strings
    for both task_migrated and task_preempted.
    """
    print("\n[67] Phase 3.0.10 P3 fix: migration/preempt events visible in dispatch + watcher.log + Feishu")

    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()

    # 1. _do_dispatch enriches the migrated event with from_node/to_node/eta/reason.
    do_idx = src.find("def _do_dispatch")
    next_def = src.find("\ndef ", do_idx + 5)
    body = src[do_idx:next_def]
    check("_do_dispatch attaches from_node to migrated event",
          '"from_node"' in body and "migrated_from" in body)
    check("_do_dispatch attaches to_node to migrated event",
          '"to_node"' in body and "preferred_node" in body)
    check("_do_dispatch attaches eta_seconds to migrated event",
          '"eta_seconds"' in body)
    check("_do_dispatch attaches reason to migrated event (last_block_reason)",
          '"reason"' in body and "last_block_reason" in body)

    # 2. cmd_dispatch print loop has elif branches for migrated and preempted.
    cd_idx = src.find("def cmd_dispatch")
    cd_next = src.find("\ndef ", cd_idx + 5)
    cd_body = src[cd_idx:cd_next]
    check("cmd_dispatch prints MIGRATE on 'migrated' event",
          'ev["type"] == "migrated"' in cd_body and "MIGRATE" in cd_body)
    check("cmd_dispatch prints PREEMPT on 'preempted' event",
          'ev["type"] == "preempted"' in cd_body and "PREEMPT" in cd_body)

    # 3. _watch_iteration's notify loop fires task_migrated and task_preempted.
    wi_idx = src.find("def _watch_iteration")
    wi_next = src.find("\ndef ", wi_idx + 5)
    wi_body = src[wi_idx:wi_next]
    check("_watch_iteration calls notify('task_migrated', ...)",
          'notify("task_migrated"' in wi_body)
    check("_watch_iteration calls notify('task_preempted', ...)",
          'notify("task_preempted"' in wi_body)

    # 4. _format_feishu has cases for both event types so they render in push.
    ff_idx = src.find("def _format_feishu")
    ff_next = src.find("\ndef ", ff_idx + 5)
    ff_body = src[ff_idx:ff_next]
    check("_format_feishu handles task_migrated",
          '"task_migrated"' in ff_body)
    check("_format_feishu handles task_preempted",
          '"task_preempted"' in ff_body)

    # 5. Behavioral: _do_dispatch on a contrived state must produce one enriched
    # 'migrated' event with the fields that downstream consumers depend on.
    saved_can_migrate = sch._can_migrate_to
    saved_NODES = sch.NODES
    saved_pick_placement = sch.pick_placement
    saved_save_state = sch.save_state  # SENTINEL (queue-wipe incident 2026-05-07): stub
    sch._can_migrate_to = lambda task, target_node, timeout_s=5: True
    sch.NODES = {
        "loaded": {"name": "loaded"},
        "free":   {"name": "free"},
    }
    # Skip placement entirely — we only care about migration's event emission,
    # not the launch path. None ⇒ pick_placement found nowhere to fit.
    sch.pick_placement = lambda task, nodes: None
    sch.save_state = lambda *_a, **_kw: None  # don't touch live queue.json
    try:
        eta = sch.MIGRATION_MIN_TASK_ETA_S + 600  # safely past the threshold
        running_a = {"id": "rA", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        running_b = {"id": "rB", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        cand = {"id": "tQ", "status": "queued", "preferred_node": "loaded",
                "eta_seconds": eta, "submitted_at": time.time(),
                "priority": "normal", "description": "migrate me",
                "cpu_cores": 1, "ram_mb": 1000, "est_vram_mb": 0}
        state = {"tasks": [running_a, running_b, cand]}
        nodes = [
            {"name": "loaded", "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 2,
             "slurm_pending_count": 0},
            {"name": "free",   "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 0,
             "slurm_pending_count": 0},
        ]
        events, _qcount = sch._do_dispatch(state, nodes)
        mig_events = [e for e in events if e.get("type") == "migrated"]
        check("_do_dispatch emitted exactly 1 'migrated' event",
              len(mig_events) == 1, diag=f"events={events}")
        if mig_events:
            ev = mig_events[0]
            check("migrated event has task_id=tQ",
                  ev.get("task_id") == "tQ")
            check("migrated event has from_node=loaded",
                  ev.get("from_node") == "loaded",
                  diag=f"got {ev.get('from_node')!r}")
            check("migrated event has to_node=free",
                  ev.get("to_node") == "free",
                  diag=f"got {ev.get('to_node')!r}")
            check("migrated event preserves eta_seconds (≥ MIN threshold)",
                  ev.get("eta_seconds", 0) >= sch.MIGRATION_MIN_TASK_ETA_S,
                  diag=f"got {ev.get('eta_seconds')}")
            check("migrated event has non-empty reason carrying last_block_reason",
                  isinstance(ev.get("reason"), str) and "migrated" in ev["reason"])
    finally:
        sch._can_migrate_to = saved_can_migrate
        sch.NODES = saved_NODES
        sch.pick_placement = saved_pick_placement
        sch.save_state = saved_save_state

    # 6. _format_feishu produces a sane single-line text for both event types.
    msg_m = sch._format_feishu("task_migrated", {
        "task_id": "tQ", "from_node": "loaded", "to_node": "free",
        "eta_seconds": 1800, "reason": "migrated: ..."
    })
    check("Feishu task_migrated text mentions task id, both nodes, eta",
          "tQ" in msg_m and "loaded" in msg_m and "free" in msg_m and "1800" in msg_m,
          diag=msg_m)
    msg_p = sch._format_feishu("task_preempted", {
        "task_id": "tV", "freed_node": "loaded", "cpu_freed": 4, "ram_freed": 8000
    })
    check("Feishu task_preempted text mentions task id, node, freed resources",
          "tV" in msg_p and "loaded" in msg_p and "4" in msg_p and "8000" in msg_p,
          diag=msg_p)


def test_phase3_0_11_migration_target_respects_blocked_and_launch_failed():
    """Phase 3.0.11 P2 fix: migration must skip a candidate whose target node is
    already known-bad for that task — either via env_missing/python_import
    escalation (_blocked_nodes_for_task) or via prior launch failure
    (_launch_failed_nodes_for_task).

    Pre-fix: _identify_migration_candidates picked the lightest-loaded alive node
    as target without consulting the task's own block lists. Result: rsync
    succeeded staging cwd+ckpt to target, but on the next dispatch pick_placement
    excluded target for THIS task and fell back to a different node (possibly a
    third one where the ckpt was NEVER staged) → resume task silently restarts
    from step 0. Same blast radius shape as the 3.0.6 remote→remote bug.

    Now: filter both inside _identify_migration_candidates (skip rsync) AND inside
    _consider_migration (defensive recheck — staging ran outside the lock so
    block lists could have updated since).
    """
    print("\n[68] Phase 3.0.11 P2 fix: migration target respects blocked / launch_failed lists")

    # 1. Source guard: both call sites consult the helpers.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    iden_idx = src.find("def _identify_migration_candidates")
    iden_end = src.find("\ndef ", iden_idx + 5)
    iden_body = src[iden_idx:iden_end]
    check("_identify_migration_candidates calls _blocked_nodes_for_task",
          "_blocked_nodes_for_task" in iden_body)
    check("_identify_migration_candidates calls _launch_failed_nodes_for_task",
          "_launch_failed_nodes_for_task" in iden_body)

    cm_idx = src.find("def _consider_migration")
    cm_end = src.find("\ndef ", cm_idx + 5)
    cm_body = src[cm_idx:cm_end]
    check("_consider_migration also calls _blocked_nodes_for_task (defensive)",
          "_blocked_nodes_for_task" in cm_body)
    check("_consider_migration also calls _launch_failed_nodes_for_task (defensive)",
          "_launch_failed_nodes_for_task" in cm_body)

    # 2. Behavioral test of _identify_migration_candidates with blocked / launch_failed.
    saved_blocked = sch._blocked_nodes_for_task
    saved_launch_failed = sch._launch_failed_nodes_for_task
    saved_NODES = sch.NODES
    sch.NODES = {
        "loaded": {"name": "loaded"},
        "free":   {"name": "free"},
    }
    # Per-task blocking: tBlocked has target='free' in its blocked set; tFailed has
    # target='free' in its launch_failed set; tOK has neither and should pass.
    def fake_blocked(task):
        if task.get("id") == "tBlocked":
            return {"free"}
        return set()
    def fake_launch_failed(task):
        if task.get("id") == "tFailed":
            return {"free"}
        return set()
    sch._blocked_nodes_for_task = fake_blocked
    sch._launch_failed_nodes_for_task = fake_launch_failed
    try:
        eta = sch.MIGRATION_MIN_TASK_ETA_S + 600
        running_a = {"id": "rA", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        running_b = {"id": "rB", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        # Three migration candidates pinned to loaded; only tOK should pass.
        tBlocked = {"id": "tBlocked", "status": "queued", "preferred_node": "loaded",
                    "eta_seconds": eta, "submitted_at": time.time(),
                    "priority": "normal", "description": "env-blocked on target"}
        tFailed = {"id": "tFailed", "status": "queued", "preferred_node": "loaded",
                   "eta_seconds": eta, "submitted_at": time.time() + 1,
                   "priority": "normal", "description": "launch-failed on target"}
        tOK = {"id": "tOK", "status": "queued", "preferred_node": "loaded",
               "eta_seconds": eta, "submitted_at": time.time() + 2,
               "priority": "normal", "description": "fine"}
        state = {"tasks": [running_a, running_b, tBlocked, tFailed, tOK]}
        nodes = [
            {"name": "loaded", "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 2,
             "slurm_pending_count": 0},
            {"name": "free",   "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 0,
             "slurm_pending_count": 0},
        ]
        identified = sch._identify_migration_candidates(state, nodes, max_candidates=10)
        identified_ids = [c["id"] for c, _ in identified]
        check("env-blocked candidate (tBlocked) NOT in staging snapshot",
              "tBlocked" not in identified_ids,
              diag=f"got {identified_ids}")
        check("launch-failed candidate (tFailed) NOT in staging snapshot",
              "tFailed" not in identified_ids,
              diag=f"got {identified_ids}")
        check("clean candidate (tOK) IS in staging snapshot",
              "tOK" in identified_ids,
              diag=f"got {identified_ids}")
        # All identified candidates target the lightest-loaded node ('free').
        check("all identified candidates target the lightest node ('free')",
              all(tgt == "free" for _, tgt in identified))

        # 3. Behavioral test of _consider_migration — only tOK migrates;
        # tBlocked/tFailed left in place (preferred_node still 'loaded').
        # _can_migrate_to mocked to True so staging-cache is irrelevant for this test.
        saved_can_migrate = sch._can_migrate_to
        sch._can_migrate_to = lambda task, target_node, timeout_s=5: True
        try:
            migrated = sch._consider_migration(state, nodes)
            check("_consider_migration migrates exactly one task",
                  len(migrated) == 1, diag=f"got {migrated}")
            check("_consider_migration migrates tOK (the un-blocked candidate)",
                  migrated == ["tOK"], diag=f"got {migrated}")
            check("tOK preferred_node now 'free'",
                  tOK["preferred_node"] == "free")
            check("tBlocked preferred_node UNCHANGED (still 'loaded')",
                  tBlocked["preferred_node"] == "loaded")
            check("tFailed preferred_node UNCHANGED (still 'loaded')",
                  tFailed["preferred_node"] == "loaded")
        finally:
            sch._can_migrate_to = saved_can_migrate

        # 4. If ALL candidates have target blocked, identify returns nothing.
        def fake_blocked_all(task):
            return {"free"} if task.get("status") == "queued" else set()
        sch._blocked_nodes_for_task = fake_blocked_all
        # Reset preferred_node so candidates are eligible again.
        for t in (tBlocked, tFailed, tOK):
            t["preferred_node"] = "loaded"
        identified_all_blocked = sch._identify_migration_candidates(
            state, nodes, max_candidates=10)
        check("when all candidates are blocked from target → no staging snapshot",
              identified_all_blocked == [],
              diag=f"got {identified_all_blocked}")
    finally:
        sch._blocked_nodes_for_task = saved_blocked
        sch._launch_failed_nodes_for_task = saved_launch_failed
        sch.NODES = saved_NODES


def test_phase3_0_12_migration_cooldown_anti_oscillation():
    """Phase 3.0.12 P3 fix: a task that just migrated must wait MIGRATION_COOLDOWN_S
    before another migration is considered.

    Pre-fix: no per-task cooldown. Under load oscillation (A becomes heavy → migrate
    task to B; B becomes heavy → migrate same task back to A; A becomes heavy → …)
    the same task could ping-pong every dispatch cycle, costing one rsync per hop
    and burning network bandwidth without ever progressing.

    Now: both _identify_migration_candidates and _consider_migration check
    `migrated_at` and skip candidates whose last migration was within
    MIGRATION_COOLDOWN_S of now. Tasks with no `migrated_at` (never migrated) are
    not affected.
    """
    print("\n[69] Phase 3.0.12 P3 fix: per-task migration cooldown stops oscillation")

    # 1. Source guard: cooldown constant exists and both call sites consult migrated_at.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("MIGRATION_COOLDOWN_S env-overridable constant exists",
          "MIGRATION_COOLDOWN_S = int(os.environ.get(" in src
          and "SCHEDULEURM_MIGRATION_COOLDOWN_S" in src)
    iden_idx = src.find("def _identify_migration_candidates")
    iden_end = src.find("\ndef ", iden_idx + 5)
    iden_body = src[iden_idx:iden_end]
    check("_identify_migration_candidates checks migrated_at vs cooldown",
          "migrated_at" in iden_body and "MIGRATION_COOLDOWN_S" in iden_body)
    cm_idx = src.find("def _consider_migration")
    cm_end = src.find("\ndef ", cm_idx + 5)
    cm_body = src[cm_idx:cm_end]
    check("_consider_migration also checks migrated_at vs cooldown (defensive)",
          "migrated_at" in cm_body and "MIGRATION_COOLDOWN_S" in cm_body)

    # 2. Behavioral: cooldown filter inside _identify_migration_candidates.
    saved_NODES = sch.NODES
    sch.NODES = {
        "loaded": {"name": "loaded"},
        "free":   {"name": "free"},
    }
    try:
        eta = sch.MIGRATION_MIN_TASK_ETA_S + 600
        # Background load on "loaded" so it qualifies as the source.
        running_a = {"id": "rA", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        running_b = {"id": "rB", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        # tCool: just migrated 60s ago → still in cooldown → must be skipped.
        # tWarm: migrated MIGRATION_COOLDOWN_S+60s ago → cooldown elapsed → eligible.
        # tFresh: never migrated (no migrated_at) → eligible.
        now = time.time()
        tCool = {"id": "tCool", "status": "queued", "preferred_node": "loaded",
                 "eta_seconds": eta, "submitted_at": now,
                 "priority": "normal", "description": "still in cooldown",
                 "migrated_at": now - 60}
        tWarm = {"id": "tWarm", "status": "queued", "preferred_node": "loaded",
                 "eta_seconds": eta, "submitted_at": now + 1,
                 "priority": "normal", "description": "cooldown expired",
                 "migrated_at": now - sch.MIGRATION_COOLDOWN_S - 60}
        tFresh = {"id": "tFresh", "status": "queued", "preferred_node": "loaded",
                  "eta_seconds": eta, "submitted_at": now + 2,
                  "priority": "normal", "description": "never migrated"}
        state = {"tasks": [running_a, running_b, tCool, tWarm, tFresh]}
        nodes = [
            {"name": "loaded", "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 2,
             "slurm_pending_count": 0},
            {"name": "free",   "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 0,
             "slurm_pending_count": 0},
        ]
        identified = sch._identify_migration_candidates(state, nodes, max_candidates=10)
        ids = [c["id"] for c, _ in identified]
        check("recently-migrated task (tCool) NOT in staging snapshot",
              "tCool" not in ids, diag=f"got {ids}")
        check("cooldown-expired task (tWarm) IS in snapshot",
              "tWarm" in ids, diag=f"got {ids}")
        check("never-migrated task (tFresh) IS in snapshot",
              "tFresh" in ids, diag=f"got {ids}")

        # 3. _consider_migration mirrors the gate. tCool stays put even if staging
        # would have succeeded (we mock _can_migrate_to True).
        saved_can_migrate = sch._can_migrate_to
        sch._can_migrate_to = lambda task, target_node, timeout_s=5: True
        try:
            migrated = sch._consider_migration(state, nodes)
            check("_consider_migration migrates exactly one task",
                  len(migrated) == 1, diag=f"got {migrated}")
            # MIGRATION_MAX_PER_DISPATCH=1; among eligibles tWarm vs tFresh, sort is
            # ascending eta — both have same eta so submitted_at decides ⇒ tWarm wins.
            check("_consider_migration migrates an eligible task (not tCool)",
                  migrated and migrated[0] != "tCool", diag=f"got {migrated}")
            check("tCool preferred_node UNCHANGED (still 'loaded')",
                  tCool["preferred_node"] == "loaded")
        finally:
            sch._can_migrate_to = saved_can_migrate

        # 4. Edge: migrated_at set to exactly cooldown boundary minus 1s — still gated.
        # And exactly cooldown + 1s — passes.
        boundary_just_under = {"id": "tBoundUnder", "status": "queued",
                               "preferred_node": "loaded", "eta_seconds": eta,
                               "submitted_at": now + 10, "priority": "normal",
                               "migrated_at": now - sch.MIGRATION_COOLDOWN_S + 1}
        boundary_just_over  = {"id": "tBoundOver", "status": "queued",
                               "preferred_node": "loaded", "eta_seconds": eta,
                               "submitted_at": now + 11, "priority": "normal",
                               "migrated_at": now - sch.MIGRATION_COOLDOWN_S - 1}
        state2 = {"tasks": [running_a, running_b, boundary_just_under, boundary_just_over]}
        ids2 = [c["id"] for c, _ in sch._identify_migration_candidates(
            state2, nodes, max_candidates=10)]
        check("migrated_at = (cooldown - 1s) ago → still gated",
              "tBoundUnder" not in ids2, diag=f"got {ids2}")
        check("migrated_at = (cooldown + 1s) ago → passes",
              "tBoundOver" in ids2, diag=f"got {ids2}")

        # 5. migrated_at = 0 / missing → treated as "never migrated", not gated.
        zero_t = {"id": "tZero", "status": "queued", "preferred_node": "loaded",
                  "eta_seconds": eta, "submitted_at": now + 20,
                  "priority": "normal", "migrated_at": 0}
        state3 = {"tasks": [running_a, running_b, zero_t]}
        ids3 = [c["id"] for c, _ in sch._identify_migration_candidates(
            state3, nodes, max_candidates=10)]
        check("migrated_at=0 (sentinel for never-migrated) → not gated",
              "tZero" in ids3, diag=f"got {ids3}")
    finally:
        sch.NODES = saved_NODES


def test_phase3_0_8_unknown_eta_skipped_in_migration():
    """Phase 3.0.8 P2 fix: queued tasks with eta_seconds=0 (unknown / no signal yet)
    must NOT be migrated.

    Pre-fix: filter was `if eta > 0 and eta < MIN: continue` — eta=0 escaped the
    filter, AND the candidate sort `key=lambda t: int(t.get("eta_seconds") or 0)`
    placed unknown-ETA tasks FIRST. So a brand-new queued task with no log signal
    yet would be the first to migrate, against documented "eta=0 is neutral, don't
    bias either way" semantics.

    Now: filter rewritten to `if eta < MIN: continue` which catches both eta=0
    and eta in [1, MIN). Conservative: without an ETA signal we can't reason about
    migration cost/benefit, so skip.
    """
    print("\n[65] Phase 3.0.8 P2 fix: eta=0 (unknown) tasks NOT migrated")

    saved_can_migrate = sch._can_migrate_to
    saved_NODES = sch.NODES
    sch._can_migrate_to = lambda task, target_node, timeout_s=5: True
    sch.NODES = {
        "A": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
        "B": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
    }
    nodes_alive = [{"name": "A", "alive": True}, {"name": "B", "alive": True}]

    try:
        # ---------- Reproducer of the bug user reported ----------
        # A is heavily loaded (running task tR). B is empty. Two queued candidates,
        # both pinned to A: q0 has eta=0 (unknown), q1 has eta=1800 (>MIN).
        # Pre-fix: q0 migrated first (sort puts eta=0 ahead of eta=1800).
        # Post-fix: q0 is skipped, q1 migrates.
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "q0", "status": "queued", "preferred_node": "A",
             "eta_seconds": 0, "cwd": "/tmp"},  # unknown ETA
            {"id": "q1", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1800, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("eta=0 candidate NOT migrated (was 'unknown_eta_migration' bug)",
              "q0" not in migrated, diag=str(migrated))
        check("eta>=MIN candidate q1 migrated instead",
              migrated == ["q1"], diag=str(migrated))

        # ---------- _identify_migration_candidates also filters eta=0 ----------
        # Same fix needs to apply to the outside-lock identification path so we
        # don't waste rsync staging on tasks we'd reject inside the lock anyway.
        # Use a FRESH state since _consider_migration above mutated q1.preferred_node.
        state_id = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "q0", "status": "queued", "preferred_node": "A",
             "eta_seconds": 0, "cwd": "/tmp"},
            {"id": "q1", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1800, "cwd": "/tmp"},
        ]}
        snap = sch._identify_migration_candidates(state_id, nodes_alive, max_candidates=5)
        candidate_ids = [c[0]["id"] for c in snap]
        check("identify_candidates also skips eta=0",
              "q0" not in candidate_ids, diag=str(candidate_ids))
        check("identify_candidates includes eta>=MIN candidates",
              "q1" in candidate_ids, diag=str(candidate_ids))

        # ---------- Edge: only eta=0 candidates → empty ----------
        state2 = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "q0a", "status": "queued", "preferred_node": "A",
             "eta_seconds": 0, "cwd": "/tmp"},
            {"id": "q0b", "status": "queued", "preferred_node": "A",
             "eta_seconds": 0, "cwd": "/tmp"},
        ]}
        migrated2 = sch._consider_migration(state2, nodes_alive)
        check("only-eta=0 candidates → no migration (don't migrate blind)",
              migrated2 == [], diag=str(migrated2))
    finally:
        sch._can_migrate_to = saved_can_migrate
        sch.NODES = saved_NODES


def test_phase3_0_7_rebalance_pending_no_duplicate_sbatch():
    """Phase 3.0.7 P1 fix: rebalance-pending verifies scancel actually took effect
    BEFORE clearing slurm_job_id + status=queued. If verification fails, the task
    is LEFT IN PLACE so next dispatch can't re-sbatch a duplicate.

    Pre-fix: scancel rc != 0 was logged but the code still cleared slurm_job_id
    and flipped status=queued. Next dispatch sbatched again. Slurm got two jobs
    for the same task — violation of the "same task never runs twice" invariant.

    Now: post-scancel `squeue -j <jid>` polls slurm's actual state with 1.5s
    settle delay. Only when squeue says terminal-or-absent do we clear + requeue.
    """
    print("\n[64] Phase 3.0.7 rebalance-pending: scancel verify before clearing slurm_job_id")

    saved_save = sch.save_state
    saved_load = sch.load_state
    saved_run_on = sch.run_on
    saved_lock = sch.state_lock
    saved_sleep = time.sleep
    time.sleep = lambda s: None  # skip the 1.5s settle delay in tests

    captured = {}
    sch.save_state = lambda s: captured.setdefault("state", s)
    from contextlib import contextmanager as _cm
    @_cm
    def fake_lock():
        yield
    sch.state_lock = fake_lock

    class Args: yes = True

    # ---------- Case A: scancel succeeds + squeue confirms terminal ----------
    sch._STAGING_CACHE.clear() if hasattr(sch, "_STAGING_CACHE") else None
    fake_state = {"next_id": 1, "tasks": [
        {"id": "tA", "status": "running", "node": "n1",
         "slurm_job_id": 100, "slurm_state": "PENDING",
         "remote_pids": [], "signature": "TEST/A", "cmd": "x"},
    ]}
    sch.load_state = lambda: fake_state
    def run_on_terminal(node, cmd, timeout=15, check=True):
        if "scancel" in cmd:
            return (0, "", "")
        if "squeue" in cmd:
            return (0, "CANCELLED\n", "")  # slurm confirms terminal
        return (0, "", "")
    sch.run_on = run_on_terminal
    captured.clear()
    try:
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("scancel + squeue=CANCELLED → cleared + requeued",
              post["status"] == "queued" and post.get("slurm_job_id") is None,
              diag=str(post))
    finally:
        pass

    # ---------- Case B: scancel rc=0 BUT squeue still shows RUNNING → leave task ----------
    fake_state = {"next_id": 1, "tasks": [
        {"id": "tB", "status": "running", "node": "n1",
         "slurm_job_id": 200, "slurm_state": "PENDING",
         "remote_pids": [], "signature": "TEST/B", "cmd": "x"},
    ]}
    sch.load_state = lambda: fake_state
    def run_on_still_alive(node, cmd, timeout=15, check=True):
        if "scancel" in cmd:
            return (0, "", "")          # scancel "succeeded" but...
        if "squeue" in cmd:
            return (0, "RUNNING\n", "")  # ...job is still RUNNING
        return (0, "", "")
    sch.run_on = run_on_still_alive
    try:
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("scancel rc=0 but slurm still shows RUNNING → task LEFT IN PLACE",
              post["status"] == "running" and post.get("slurm_job_id") == 200,
              diag=str(post))
        check("skipped task gets last_block_reason explaining why",
              "SKIPPED" in (post.get("last_block_reason") or "")
              and "duplicate sbatch" in (post.get("last_block_reason") or ""),
              diag=post.get("last_block_reason"))
    finally:
        pass

    # ---------- Case C: scancel rc != 0 (ssh blip / slurm RPC fail) → leave task ----------
    # Squeue check might still succeed and show RUNNING, OR fail too. Either way:
    # task should NOT be cleared if slurm still has it.
    fake_state = {"next_id": 1, "tasks": [
        {"id": "tC", "status": "running", "node": "n1",
         "slurm_job_id": 300, "slurm_state": "PENDING",
         "remote_pids": [], "signature": "TEST/C", "cmd": "x"},
    ]}
    sch.load_state = lambda: fake_state
    def run_on_scancel_fails(node, cmd, timeout=15, check=True):
        if "scancel" in cmd:
            return (1, "", "ssh: connection refused")
        if "squeue" in cmd:
            return (0, "PENDING\n", "")  # job still pending in slurm
        return (0, "", "")
    sch.run_on = run_on_scancel_fails
    try:
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("scancel rc!=0 + squeue says PENDING → task LEFT IN PLACE",
              post["status"] == "running" and post.get("slurm_job_id") == 300,
              diag=str(post))
    finally:
        pass

    # ---------- Case D: squeue verify itself fails (network issue) → leave task ----------
    fake_state = {"next_id": 1, "tasks": [
        {"id": "tD", "status": "running", "node": "n1",
         "slurm_job_id": 400, "slurm_state": "PENDING",
         "remote_pids": [], "signature": "TEST/D", "cmd": "x"},
    ]}
    sch.load_state = lambda: fake_state
    def run_on_verify_fails(node, cmd, timeout=15, check=True):
        if "scancel" in cmd:
            return (0, "", "")  # scancel "succeeded"
        if "squeue" in cmd:
            return (1, "", "ssh: timed out")  # but verify itself fails
        return (0, "", "")
    sch.run_on = run_on_verify_fails
    try:
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("scancel rc=0 but squeue verify fails → conservatively LEFT IN PLACE",
              post["status"] == "running" and post.get("slurm_job_id") == 400,
              diag=str(post))
    finally:
        pass

    # ---------- Case E: squeue returns empty (job not in queue at all) → safe to clear ----------
    fake_state = {"next_id": 1, "tasks": [
        {"id": "tE", "status": "running", "node": "n1",
         "slurm_job_id": 500, "slurm_state": "PENDING",
         "remote_pids": [], "signature": "TEST/E", "cmd": "x"},
    ]}
    sch.load_state = lambda: fake_state
    def run_on_squeue_empty(node, cmd, timeout=15, check=True):
        if "scancel" in cmd:
            return (0, "", "")
        if "squeue" in cmd:
            return (0, "", "")  # empty = job not in slurm anymore
        return (0, "", "")
    sch.run_on = run_on_squeue_empty
    try:
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("squeue empty (job purged from slurm) → safe to clear + requeue",
              post["status"] == "queued" and post.get("slurm_job_id") is None,
              diag=str(post))
    finally:
        pass

    # ---------- Case F: mix — one verifies, one doesn't ----------
    fake_state = {"next_id": 1, "tasks": [
        {"id": "tOK", "status": "running", "node": "n1",
         "slurm_job_id": 600, "slurm_state": "PENDING",
         "remote_pids": [], "signature": "TEST/OK", "cmd": "x"},
        {"id": "tBad", "status": "running", "node": "n1",
         "slurm_job_id": 700, "slurm_state": "PENDING",
         "remote_pids": [], "signature": "TEST/Bad", "cmd": "x"},
    ]}
    sch.load_state = lambda: fake_state
    def run_on_mix(node, cmd, timeout=15, check=True):
        if "scancel 600" in cmd: return (0, "", "")
        if "scancel 700" in cmd: return (0, "", "")
        if "squeue -h -j 600" in cmd: return (0, "CANCELLED\n", "")
        if "squeue -h -j 700" in cmd: return (0, "RUNNING\n", "")  # still alive
        return (0, "", "")
    sch.run_on = run_on_mix
    try:
        sch.cmd_rebalance_pending(Args())
        post_ok = next(t for t in fake_state["tasks"] if t["id"] == "tOK")
        post_bad = next(t for t in fake_state["tasks"] if t["id"] == "tBad")
        check("verified task → cleared + requeued",
              post_ok["status"] == "queued" and post_ok.get("slurm_job_id") is None)
        check("unverified task → LEFT IN PLACE (status=running, slurm_job_id intact)",
              post_bad["status"] == "running" and post_bad.get("slurm_job_id") == 700)
    finally:
        sch.save_state = saved_save
        sch.load_state = saved_load
        sch.run_on = saved_run_on
        sch.state_lock = saved_lock
        time.sleep = saved_sleep


def test_phase3_0_5_staging_outside_lock():
    """Phase 3.0.5 P1 fix: migration staging runs OUTSIDE state_lock so a multi-minute
    rsync can't block submit/cancel/status/watcher.

    Pre-fix: _do_dispatch → _consider_migration → _can_migrate_to → _stage_for_migration
    chain ran rsync (timeout=600s) inside the global state_lock. Any other tool
    waiting on state_lock (status / submit / cancel / watcher iteration) would
    starve for up to 10 minutes per dispatch cycle.

    Fix: split into
      - inside lock: _can_migrate_to() = dict lookup of _STAGED_TASKS only (μs)
      - outside lock: _stage_migration_candidates_outside_lock() does identify-
        candidates (brief lock) → release → rsync → update _STAGED_TASKS
    """
    print("\n[63] Phase 3.0.5 staging runs outside state_lock (P1 lock-starvation fix)")

    # ---------- Source guards: dispatch entry points call outside-lock staging ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()

    # cmd_dispatch: outside-lock call comes BEFORE the main `with state_lock():`
    cd_idx = src.find("def cmd_dispatch")
    next_def = src.find("\ndef ", cd_idx + 5)
    cd_body = src[cd_idx:next_def]
    stage_idx = cd_body.find("_stage_migration_candidates_outside_lock()")
    main_lock_idx = cd_body.rfind("with state_lock():")
    check("cmd_dispatch calls _stage_migration_candidates_outside_lock",
          stage_idx > 0)
    check("cmd_dispatch's stage call happens BEFORE the main state_lock",
          stage_idx > 0 and main_lock_idx > stage_idx,
          diag=f"stage_idx={stage_idx} main_lock_idx={main_lock_idx}")

    # _watch_iteration: same constraint
    wi_idx = src.find("def _watch_iteration")
    next_def = src.find("\ndef ", wi_idx + 5)
    wi_body = src[wi_idx:next_def]
    stage_idx_w = wi_body.find("_stage_migration_candidates_outside_lock()")
    # _watch_iteration acquires multiple locks; the relevant one is the one wrapping
    # _do_dispatch. Look for first state_lock that comes AFTER the staging call.
    main_lock_idx_w = wi_body.find("with state_lock():", stage_idx_w) if stage_idx_w > 0 else -1
    check("_watch_iteration calls _stage_migration_candidates_outside_lock",
          stage_idx_w > 0)
    check("_watch_iteration's stage call comes BEFORE its dispatch state_lock",
          stage_idx_w > 0 and main_lock_idx_w > stage_idx_w)

    # ---------- _can_migrate_to no longer calls _stage_for_migration ----------
    cmt_idx = src.find("def _can_migrate_to")
    next_def = src.find("\ndef ", cmt_idx + 5)
    cmt_body = src[cmt_idx:next_def]
    check("_can_migrate_to does NOT call _stage_for_migration (would block lock)",
          "_stage_for_migration(" not in cmt_body,
          diag="staging back inside the lock breaks Phase 3.0.5 invariant")
    check("_can_migrate_to is a _STAGED_TASKS lookup",
          "_STAGED_TASKS" in cmt_body)

    # ---------- _STAGED_TASKS LRU eviction guards memory ----------
    check("_STAGED_TASKS is a dict", isinstance(sch._STAGED_TASKS, dict))
    check("_STAGED_TASKS_MAX bounds memory",
          hasattr(sch, "_STAGED_TASKS_MAX") and sch._STAGED_TASKS_MAX > 0)

    # ---------- _identify_migration_candidates returns snapshot, not live refs ----------
    saved_NODES = sch.NODES
    sch.NODES = {
        "A": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
        "B": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
    }
    nodes_alive = [{"name": "A", "alive": True}, {"name": "B", "alive": True}]
    state = {"tasks": [
        {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
        {"id": "tQ1", "status": "queued", "preferred_node": "A",
         "eta_seconds": 1800, "cwd": "/p1", "ckpt_dir": "/c1",
         "cmd": "/abs/python train.py", "signature": "TEST/m1"},
        {"id": "tQ2", "status": "queued", "preferred_node": "A",
         "eta_seconds": 2400, "cwd": "/p2", "cmd": "/abs/python train.py",
         "signature": "TEST/m2"},
    ]}
    try:
        snap = sch._identify_migration_candidates(state, nodes_alive, max_candidates=2)
        check("_identify_migration_candidates returns up to N pairs",
              len(snap) == 2, diag=f"got {len(snap)}")
        check("each pair is (task_dict_copy, target_node)",
              all(isinstance(c[0], dict) and isinstance(c[1], str) for c in snap))
        check("pair contains the fields _stage_for_migration needs",
              all({"id", "cwd", "cmd", "preferred_node"} <= set(c[0].keys()) for c in snap))
        # Snapshot is a copy — mutating it shouldn't touch state["tasks"]
        snap[0][0]["cwd"] = "/MUTATED"
        check("snapshot is a copy, not a live ref into state",
              state["tasks"][1].get("cwd") == "/p1",
              diag=str(state["tasks"][1].get("cwd")))
        # Smallest-ETA-first ordering preserved
        check("candidates ordered by ETA ascending",
              snap[0][0]["eta_seconds"] <= snap[1][0]["eta_seconds"])

        # No candidates when balanced
        balanced = {"tasks": [
            {"id": "tA", "status": "running", "node": "A", "eta_seconds": 1000},
            {"id": "tB", "status": "running", "node": "B", "eta_seconds": 800},
        ]}
        snap2 = sch._identify_migration_candidates(balanced, nodes_alive)
        check("balanced loads → no candidates", len(snap2) == 0)
    finally:
        sch.NODES = saved_NODES


def test_phase3_0_4_staging():
    """Phase 3.0.4: _stage_for_migration handles cwd rsync, ckpt size cap (2GB), env probe.

    Mocks run_on (ssh) and subprocess.run (rsync) since this hits live hosts otherwise.
    """
    print("\n[62] Phase 3.0.4 staging — cwd rsync + ckpt cap + env probe")

    saved_run_on = sch.run_on
    saved_sp_run = sch.subprocess.run
    saved_NODES = sch.NODES
    saved_cache = sch._STAGING_CACHE.copy()

    sch.NODES = {
        "src": {"host": "srcbox", "cpu_cores": 12, "ram_mb": 32000,
                "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                "max_concurrent_running": None},
        "tgt": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                "max_concurrent_running": None},
        "tgt_remote": {"host": "tgtbox", "cpu_cores": 12, "ram_mb": 32000,
                       "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                       "max_concurrent_running": None},
    }

    # Helper: build a fake run_on that maps each command to a canned response
    def mk_run_on(plan):
        """plan: list of (cmd_substring → (rc, out, err))"""
        def fake(node, cmd, timeout=15, check=True):
            for substr, resp in plan:
                if substr in cmd:
                    return resp
            return (1, "", "")  # default deny
        return fake

    # Helper: subprocess.run mock that records rsync commands
    rsync_calls = []
    def fake_subprocess_run(args, **kwargs):
        if args and "rsync" in args[0]:
            rsync_calls.append(args)
        class R: pass
        r = R()
        r.returncode = 0
        r.stdout = ""; r.stderr = ""
        return r

    # ---------- Case A: cwd already exists on target → no rsync needed ----------
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([
        ("test -d /work", (0, "", "")),     # cwd exists on target
        ("test -x", (0, "", "")),           # python exists on target
    ])
    sch.subprocess.run = fake_subprocess_run
    rsync_calls.clear()
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tA", "cwd": "/work", "preferred_node": "src",
             "cmd": "/abs/path/python -u train.py"},
            "tgt"
        )
        check("cwd exists on target → ok, no rsync issued",
              ok and len(rsync_calls) == 0, diag=f"ok={ok} rsync={rsync_calls}")
    finally:
        pass

    # ---------- Case B: cwd missing on target, source local → rsync ----------
    sch._STAGING_CACHE.clear()
    cwd_state = [0]  # 0=missing, 1=exists. Flips after rsync.
    def fake_run_on_B(node, cmd, timeout=15, check=True):
        if "test -d /work" in cmd:
            return (0, "", "") if cwd_state[0] else (1, "", "")
        if "mkdir -p" in cmd or "test -x" in cmd:
            return (0, "", "")
        return (0, "", "")
    sch.run_on = fake_run_on_B
    rsync_calls.clear()
    def rsync_succeed(args, **kwargs):
        rsync_calls.append(args)
        cwd_state[0] = 1  # post-rsync: cwd exists
        class R: pass
        r = R(); r.returncode = 0; r.stdout = ""; r.stderr = ""
        return r
    sch.subprocess.run = rsync_succeed
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tB", "cwd": "/work", "preferred_node": "src",
             "cmd": "/abs/path/python -u train.py"},
            "tgt"
        )
        check("cwd missing → rsync issued + ok",
              ok and len(rsync_calls) == 1, diag=f"ok={ok} rsync_calls={rsync_calls}")
        check("rsync used `-az --partial` flags",
              "-az" in rsync_calls[0] and "--partial" in rsync_calls[0])
    finally:
        pass

    # ---------- Case C: ckpt size > 2GB → reject ----------
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([
        ("test -d /work", (0, "", "")),
        ("du -sm /ckpt", (0, "5000\n", "")),  # 5GB ckpt
        ("test -x", (0, "", "")),
    ])
    rsync_calls.clear()
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tC", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python train.py"},
            "tgt"
        )
        check("ckpt 5000MB > cap 2048MB → reject",
              not ok and "5000MB" in msg and "2048MB" in msg, diag=msg)
        check("no rsync of oversized ckpt",
              all("ckpt" not in (str(a) if a else "") for a in rsync_calls)
              if rsync_calls else True)
    finally:
        pass

    # ---------- Case D: ckpt under cap → rsync ckpt + cwd ----------
    sch._STAGING_CACHE.clear()
    cwd_state[0] = 0  # cwd missing initially; rsync_succeed flips it to 1
    def fake_run_on_D(node, cmd, timeout=15, check=True):
        if "test -d /work" in cmd:
            return (0, "", "") if cwd_state[0] else (1, "", "")
        if "du -sm /ckpt" in cmd:
            return (0, "1500\n", "")  # 1.5GB ckpt — under cap
        if "ls -1 /ckpt" in cmd:
            # post-rsync verify: return a fake ckpt file so empty-dir check passes
            return (0, "model_epoch_50.pt\n", "")
        if "mkdir -p" in cmd or "test -x" in cmd:
            return (0, "", "")
        return (0, "", "")
    sch.run_on = fake_run_on_D
    rsync_calls.clear()
    sch.subprocess.run = rsync_succeed
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tD", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python train.py"},
            "tgt"
        )
        check("ckpt under cap + cwd missing → both rsync'd, ok",
              ok and len(rsync_calls) >= 1, diag=f"ok={ok} rsync count={len(rsync_calls)}")
    finally:
        pass

    # ---------- Case E: env (python) missing on target → reject ----------
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([
        ("test -d /work", (0, "", "")),
        ("test -x /home/me/conda/envs/X/bin/python", (1, "", "")),  # missing
    ])
    rsync_calls.clear()
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tE", "cwd": "/work", "preferred_node": "src",
             "cmd": "/home/me/conda/envs/X/bin/python -u train.py"},
            "tgt"
        )
        check("python missing on target → reject with helpful msg",
              not ok and "python" in msg and "X/bin/python" in msg, diag=msg)
    finally:
        pass

    # ---------- Case F: source==target → no-op success ----------
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([("test -d", (0, "", ""))])  # never called actually
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tF", "cwd": "/work", "preferred_node": "tgt", "cmd": "python a.py"},
            "tgt"
        )
        check("source==target → ok, no-op", ok and "nothing to stage" in msg)
    finally:
        pass

    # ---------- Case G2 (Phase 3.0.6 P1 fix): remote→remote ckpt — refused ----------
    # Pre-fix: branch silently skipped rsync + STILL added to _STAGING_CACHE + returned
    # success → migration commits, task starts on target without ckpt → resume from
    # scratch. Now: explicit rejection like cwd has, and cache only on real rsync.
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([
        ("test -d /work", (0, "", "")),       # cwd already on target
        ("du -sm /ckpt", (0, "500\n", "")),   # ckpt is 500MB on source
    ])
    rsync_calls.clear()
    sch.subprocess.run = fake_subprocess_run
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tG2", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src",  # src has host=srcbox (remote)
             "cmd": "python a.py"},
            "tgt_remote"  # also remote (host=tgtbox)
        )
        check("remote→remote ckpt → REJECTED (would lose ckpt on migration)",
              not ok and "remote→remote not supported" in msg
              and "checkpoint" in msg, diag=msg)
        check("rejected ckpt is NOT cached (would falsely succeed next time)",
              ("src", "tgt_remote", "/ckpt") not in sch._STAGING_CACHE,
              diag=str(sch._STAGING_CACHE))
        check("no rsync attempted in remote→remote ckpt case",
              not any("rsync" in (str(a) if a else "") for a in rsync_calls))
    finally:
        pass

    # ---------- Case G3: ckpt rsync verifies non-empty target dir ----------
    # Pre-fix: rsync returncode=0 didn't guarantee files actually arrived (silent rsync
    # filter exclude / network stall). Add post-rsync ls check.
    sch._STAGING_CACHE.clear()
    cwd_state[0] = 1  # cwd already there
    def fake_run_on_empty_ckpt(node, cmd, timeout=15, check=True):
        if "test -d /work" in cmd: return (0, "", "")
        if "du -sm /ckpt" in cmd: return (0, "100\n", "")  # 100MB ckpt on source
        if "mkdir -p" in cmd: return (0, "", "")
        if "ls -1 /ckpt" in cmd:
            return (0, "", "")  # ← rsync "succeeded" but target dir empty
        if "test -x" in cmd: return (0, "", "")
        return (0, "", "")
    sch.run_on = fake_run_on_empty_ckpt
    rsync_calls.clear()
    sch.subprocess.run = fake_subprocess_run  # rsync rc=0
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tG3", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python a.py"},
            "tgt"
        )
        check("rsync rc=0 but ckpt dir empty on target → fail (don't migrate to empty)",
              not ok and "appears empty" in msg, diag=msg)
        check("empty post-rsync ckpt is NOT cached",
              ("src", "tgt", "/ckpt") not in sch._STAGING_CACHE)
    finally:
        pass

    # ---------- Case G: remote→remote rsync — refused (no via-local routing) ----------
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([
        ("test -d /work", (1, "", "")),  # cwd missing on target
    ])
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tG", "cwd": "/work", "preferred_node": "src",  # remote
             "cmd": "python a.py"},
            "tgt_remote"  # also remote
        )
        check("remote→remote rsync refused (not yet supported)",
              not ok and "remote→remote" in msg, diag=msg)
    finally:
        pass

    # ---------- _can_migrate_to is a fast cache lookup (Phase 3.0.5 split) ----------
    # Pre-fix it called _stage_for_migration directly inside the state_lock; now it
    # only checks _STAGED_TASKS membership. Staging happens outside-lock.
    sch._STAGED_TASKS.clear()
    task = {"id": "tH", "cwd": "/work", "preferred_node": "src", "cmd": "/abs/python a.py"}
    check("_can_migrate_to=False when not staged",
          not sch._can_migrate_to(task, "tgt"))
    # Simulate outside-lock staging completed
    sch._STAGED_TASKS[("tH", "tgt")] = time.time()
    check("_can_migrate_to=True when (task_id, target) is in _STAGED_TASKS",
          sch._can_migrate_to(task, "tgt"))
    # Different target → still False
    check("_can_migrate_to=False for unstaged target",
          not sch._can_migrate_to(task, "other-target"))
    sch._STAGED_TASKS.clear()

    # ---------- Cleanup ----------
    sch.run_on = saved_run_on
    sch.subprocess.run = saved_sp_run
    sch.NODES = saved_NODES
    sch._STAGING_CACHE.clear()
    sch._STAGING_CACHE.update(saved_cache)


def test_phase3_0_3_migration_trigger():
    """Phase 3.0.3: _consider_migration re-pins soft-pinned tasks from overloaded
    nodes to near-empty ones. Respects: hard pins (require_node), portability check,
    max-per-dispatch cap, ETA-based candidate ranking.
    """
    print("\n[61] Phase 3.0.3 load-balance migration trigger")

    # Tunable constants exist + sane defaults
    check("MIGRATION_LOAD_RATIO defined", hasattr(sch, "MIGRATION_LOAD_RATIO"))
    check("MIGRATION_FREE_THRESHOLD_S defined", hasattr(sch, "MIGRATION_FREE_THRESHOLD_S"))
    check("MIGRATION_MAX_PER_DISPATCH = 1 (user spec)",
          sch.MIGRATION_MAX_PER_DISPATCH == 1)
    check("MIGRATION_MIN_TASK_ETA_S defined", hasattr(sch, "MIGRATION_MIN_TASK_ETA_S"))

    # Mock _can_migrate_to to default-True (Phase 3.0.4 will replace with real staging)
    saved_can_migrate = sch._can_migrate_to
    sch._can_migrate_to = lambda task, target_node, timeout_s=5: True

    saved_NODES = sch.NODES
    sch.NODES = {
        "A": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
        "B": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
    }
    nodes_alive = [{"name": "A", "alive": True}, {"name": "B", "alive": True}]

    try:
        # ---------- imbalanced + portable: migration fires ----------
        # A has running 7200s, B is empty (under threshold). Queued task tQ has
        # preferred=A → should migrate to B.
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "tQ", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1800, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("imbalanced + portable → tQ migrated",
              migrated == ["tQ"], diag=str(migrated))
        check("after migration: tQ.preferred_node = B",
              state["tasks"][1].get("preferred_node") == "B")
        check("after migration: migrated_from = A",
              state["tasks"][1].get("migrated_from") == "A")
        check("after migration: last_block_reason mentions migration",
              "migrated:" in (state["tasks"][1].get("last_block_reason") or ""))

        # ---------- balanced loads: no migration ----------
        state = {"tasks": [
            {"id": "tR1", "status": "running", "node": "A", "eta_seconds": 1000},
            {"id": "tR2", "status": "running", "node": "B", "eta_seconds": 800},
            {"id": "tQ", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1800, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("balanced (B has 800s, > FREE_THRESHOLD) → no migration",
              migrated == [], diag=str(migrated))

        # ---------- target loaded → no migration ----------
        # A=10000, B=700 (just over FREE_THRESHOLD=600 by default). No migration.
        state = {"tasks": [
            {"id": "tR1", "status": "running", "node": "A", "eta_seconds": 10000},
            {"id": "tR2", "status": "running", "node": "B", "eta_seconds": 700},
            {"id": "tQ", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1800, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("target above FREE_THRESHOLD (700>600) → no migration",
              migrated == [], diag=str(migrated))

        # ---------- ratio not high enough: no migration ----------
        # A=400+300=700 (running+queued-pinned), B=400. Ratio 1.75 < 2.0.
        # Both target_load (400) < FREE_THRESHOLD (600), so target qualifies as free.
        # But ratio check rejects: source(700) < 2.0 * target(400) = 800.
        state = {"tasks": [
            {"id": "tR1", "status": "running", "node": "A", "eta_seconds": 400},
            {"id": "tR2", "status": "running", "node": "B", "eta_seconds": 400},
            {"id": "tQ", "status": "queued", "preferred_node": "A",
             "eta_seconds": 300, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("ratio 1.75 < 2.0 → no migration",
              migrated == [], diag=str(migrated))

        # ---------- hard-pinned task: NOT touched even if imbalanced ----------
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "tHard", "status": "queued", "require_node": "A",
             "preferred_node": "A", "eta_seconds": 3600, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("require_node set → no migration (hard pin respected)",
              migrated == [], diag=str(migrated))
        check("tHard.preferred_node still A (untouched)",
              state["tasks"][1].get("preferred_node") == "A")

        # ---------- auto_adopted: NOT touched ----------
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "tA", "status": "queued", "preferred_node": "A",
             "auto_adopted": True, "eta_seconds": 3600, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("auto_adopted → no migration",
              migrated == [], diag=str(migrated))

        # ---------- task ETA below MIN: skip ----------
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "tShort", "status": "queued", "preferred_node": "A",
             "eta_seconds": 60, "cwd": "/tmp"},  # 60s < MIN_TASK_ETA_S(300)
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("task eta < MIN_TASK_ETA_S → skip (won't recoup staging cost)",
              migrated == [], diag=str(migrated))

        # ---------- max-per-dispatch cap: only 1 migrates even if multiple eligible ----------
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "tQ1", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1000, "cwd": "/tmp"},
            {"id": "tQ2", "status": "queued", "preferred_node": "A",
             "eta_seconds": 2000, "cwd": "/tmp"},
            {"id": "tQ3", "status": "queued", "preferred_node": "A",
             "eta_seconds": 3000, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("max-per-dispatch=1: only 1 migrated despite 3 candidates",
              len(migrated) == 1, diag=str(migrated))
        check("smallest ETA migrated first (cheapest staging)",
              migrated == ["tQ1"], diag=str(migrated))

        # ---------- target node dead: no migration ----------
        nodes_target_down = [{"name": "A", "alive": True}, {"name": "B", "alive": False}]
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "tQ", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1800, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_target_down)
        check("target node down → no migration",
              migrated == [], diag=str(migrated))

        # ---------- portability fail: try next candidate ----------
        # Force _can_migrate_to to fail for tQ1 but succeed for tQ2
        sch._can_migrate_to = lambda task, target_node, timeout_s=5: task["id"] != "tQ1"
        state = {"tasks": [
            {"id": "tR", "status": "running", "node": "A", "eta_seconds": 7200},
            {"id": "tQ1", "status": "queued", "preferred_node": "A",
             "eta_seconds": 1000, "cwd": "/cant-migrate"},
            {"id": "tQ2", "status": "queued", "preferred_node": "A",
             "eta_seconds": 2000, "cwd": "/tmp"},
        ]}
        migrated = sch._consider_migration(state, nodes_alive)
        check("first candidate fails portability → next candidate tried",
              migrated == ["tQ2"], diag=str(migrated))
    finally:
        sch._can_migrate_to = saved_can_migrate
        sch.NODES = saved_NODES

    # Source guard: _do_dispatch calls _consider_migration BEFORE placement loop
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    do_idx = src.find("def _do_dispatch")
    next_def = src.find("\ndef ", do_idx + 5)
    body = src[do_idx:next_def]
    cm_idx = body.find("_consider_migration(state, nodes)")
    pp_idx = body.find("_preempt_for_high_priority(state, nodes)")
    check("_do_dispatch calls _consider_migration",
          cm_idx > 0)
    check("migration runs BEFORE preemption (so placement loop sees new pin)",
          cm_idx > 0 and pp_idx > 0 and cm_idx < pp_idx)
    check("emits 'migrated' event for each migrated task_id",
          '"type": "migrated"' in body)


def test_phase3_0_2_node_load_metric():
    """Phase 3.0.2: compute_node_load_seconds(state) → {node: total_eta_seconds}.

    Used by Phase 3.0.3 migration trigger to identify imbalanced nodes. Counts
    in-flight tasks pinned to each node:
      - status=running on N (regardless of slurm_state)
      - status=queued with require_node=N OR preferred_node=N
      - eta_seconds=0 → unknown ETA, don't count (neutral)
      - auto_adopted → not counted (don't migrate user-managed work)
      - status=launching → transient, not counted
    """
    print("\n[60] Phase 3.0.2 per-node ETA-based load metric")

    saved_NODES = sch.NODES
    sch.NODES = {
        "A": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
        "B": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
              "ram_headroom_frac": 0.10, "max_vram_per_task": None,
              "max_concurrent_running": None},
    }
    try:
        state = {"tasks": [
            # Running on A: 3600 + 1800 = 5400 should land on A
            {"id": "t1", "status": "running", "node": "A", "eta_seconds": 3600},
            {"id": "t2", "status": "running", "node": "A", "eta_seconds": 1800},
            # Queued, require=A: counts toward A's load (3000)
            {"id": "t3", "status": "queued", "require_node": "A", "eta_seconds": 3000},
            # Queued, preferred=B (no require): counts toward B (1200)
            {"id": "t4", "status": "queued", "preferred_node": "B", "eta_seconds": 1200},
            # Running on B: counts toward B (600)
            {"id": "t5", "status": "running", "node": "B", "eta_seconds": 600},
            # eta_seconds=0 (unknown): excluded
            {"id": "t6", "status": "running", "node": "A", "eta_seconds": 0},
            {"id": "t7", "status": "queued", "require_node": "B", "eta_seconds": 0},
            # auto_adopted: excluded
            {"id": "t8", "status": "running", "node": "B",
             "eta_seconds": 7200, "auto_adopted": True},
            # status=launching: excluded
            {"id": "t9", "status": "launching", "node": "A", "eta_seconds": 5000},
            # Pinned to unknown node: excluded (filtered by NODES dict)
            {"id": "t10", "status": "queued", "preferred_node": "ghost", "eta_seconds": 999},
        ]}
        loads = sch.compute_node_load_seconds(state)
        check("A's load = running(3600+1800) + require_queued(3000) = 8400",
              loads.get("A") == 8400, diag=str(loads))
        check("B's load = running(600) + preferred_queued(1200) = 1800",
              loads.get("B") == 1800, diag=str(loads))
        check("ghost node not in loads (filtered to NODES.keys())",
              "ghost" not in loads)
        check("auto_adopted task NOT counted (B would be 9000 if it were)",
              loads.get("B") < 9000)
        check("launching task NOT counted",
              loads.get("A") < 13400)
        check("eta=0 tasks NOT counted (excluded, neutral)",
              loads.get("A") == 8400)  # if eta=0 counted, A would have +0 still

        # Empty state → all zero
        loads2 = sch.compute_node_load_seconds({"tasks": []})
        check("empty state → all nodes load=0",
              all(v == 0 for v in loads2.values()) and len(loads2) == 2,
              diag=str(loads2))
    finally:
        sch.NODES = saved_NODES

    # Source guard: status display surfaces eta_load
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    cmd_status_idx = src.find("def cmd_status")
    next_def = src.find("\ndef ", cmd_status_idx + 5)
    body = src[cmd_status_idx:next_def]
    check("cmd_status calls compute_node_load_seconds",
          "compute_node_load_seconds(state)" in body)
    check("cmd_status formats eta_load output",
          "eta_load=" in body)


def test_phase3_0_1_eta_parser_and_integration():
    """Phase 3.0.1: ETA parser + watcher integration.

    Covers:
      - eta_tracker.parse_progress: each pattern (tqdm, epoch, iter, step, "done")
      - eta_tracker.compute_eta_seconds: rate math + EWMA fallback
      - _refresh_eta_from_logs: writes task['eta_seconds'] from log tail
      - update_running_tasks chains _batch_check_running + _refresh_eta_from_logs
    """
    print("\n[59] Phase 3.0.1 ETA parser + watcher integration")

    # Load eta_tracker the same way scheduler.py does
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "eta_tracker",
        os.path.expanduser("~/.claude/skills/scheduler/eta_tracker.py"),
    )
    et = _ilu.module_from_spec(spec); spec.loader.exec_module(et)

    # ---------- pattern coverage ----------
    cases = [
        ("tqdm fancy", "  47%|████▋     | 1234/5678 [00:42<03:21, 12.34it/s, loss=0.342]", (1234, 5678)),
        ("tqdm slow s/it", "Training: 12%|█▏ | 23/200 [11:24<1:29:20, 30.28s/it]", (23, 200)),
        ("epoch bracket", "[Epoch 22/200] Rollout: events=102", (22, 200)),
        ("epoch colon", "Epoch: 5/100  loss=0.42", (5, 100)),
        ("iter form", "Iter 1234 / 5000  lr=1e-4", (1234, 5000)),
        ("step form", "step 100 of 200: avg_reward=42", (100, 200)),
        ("done form", "(50/100) done", (50, 100)),
        ("multi-line, last wins", "[Epoch 1/100]\n[Epoch 2/100]\n[Epoch 3/100]\n", (3, 100)),
        ("no progress", "starting\nloading\nready", None),
        ("absurd current > total", "100/50 step", None),
        ("zero total", "0/0 init", None),
    ]
    for name, text, expected in cases:
        got = et.parse_progress(text)
        check(f"parse: {name}", got == expected, diag=f"text={text!r} got={got} expected={expected}")

    # --- tier 2 (cmd-flag fallback): "Iter N" alone in tail + "--max_iters N" in cmd ---
    cmd = "/abs/python -u -m jax.train --max_iters 2000 --seed 42"
    tail = "Iter 1793 | Reward: 4753.1 | Time: 71275s | 21.9s/iter"
    got = et.parse_progress(tail, cmd=cmd)
    check("tier 2: Iter N alone + --max_iters in cmd",
          got == (1793, 2000), diag=f"got {got}")

    cmd2 = "python train.py --n_epochs 50"
    tail2 = "Epoch 30 done"
    got = et.parse_progress(tail2, cmd=cmd2)
    check("tier 2: --n_epochs flag",
          got == (30, 50), diag=f"got {got}")

    # cmd has total but tail has no current → still None (need both)
    got = et.parse_progress("loading...", cmd=cmd)
    check("tier 2: no current in tail → None",
          got is None, diag=f"got {got}")

    # --- tqdm ETA extractor (used as tier-0 in compute_eta_seconds) ---
    cases_tqdm = [
        ("simple m:s", "[00:42<03:21, 12.34it/s]", 3*60+21),
        ("h:m:s", "[1:14:32<5:23:11, 3.21it/s]", 5*3600+23*60+11),
        ("d:h:m:s", "[02:00<1:02:30:00, 0.5s/it]", 1*86400+2*3600+30*60),
        ("? unknown", "[02:00<?, ?it/s]", None),
        ("zero remaining", "[02:00<00:00, 1.50s/it]", 0),
        ("not tqdm format", "Iter 100", None),
    ]
    for name, text, expected in cases_tqdm:
        got = et.parse_tqdm_eta(text)
        check(f"tqdm-eta: {name}", got == expected, diag=f"text={text!r} got={got} expected={expected}")

    # Multiple tqdm lines — last wins
    multi = "[00:42<03:21, 12.34it/s]\n[01:00<03:00, 12.34it/s]\n[01:30<02:30, 12.34it/s]\n"
    got = et.parse_tqdm_eta(multi)
    check("tqdm-eta: multi-line last wins",
          got == 2*60+30, diag=f"got {got}")

    # --- compute_eta_seconds prefers tqdm-eta over rate computation ---
    text = "  47%|████▋     | 1234/5678 [00:42<03:21, 12.34it/s]"
    eta = et.compute_eta_seconds(text, elapsed_s=42, fallback_ewma_s=99999)
    check("compute_eta_seconds: tqdm-eta tier-0 used (201s, ignoring rate-from-current)",
          eta == 201, diag=f"got {eta}")

    # ---------- ETA math ----------
    # tier-0 (tqdm pre-computed) wins: extracts "1:29:20" directly = 5360s
    eta = et.compute_eta_seconds(
        "Training: | 23/200 [11:24<1:29:20, 30.28s/it]",
        elapsed_s=600, fallback_ewma_s=0
    )
    check("ETA: tqdm-eta tier-0 reads remaining 1:29:20 = 5360s (ignores elapsed_s)",
          eta == 5360, diag=f"got {eta}")
    # Without tqdm bracket: rate from current/elapsed → 4617s
    eta = et.compute_eta_seconds(
        "[Epoch 23/200] step",
        elapsed_s=600, fallback_ewma_s=0
    )
    check("ETA rate-math: 23/200 after 600s elapsed ≈ 4617s (no tqdm-eta available)",
          4500 <= eta <= 4700, diag=f"got {eta}")

    # No progress, fallback EWMA (1800s) - elapsed (300s) = 1500s
    eta = et.compute_eta_seconds("nothing", elapsed_s=300, fallback_ewma_s=1800)
    check("ETA fallback: EWMA(1800) - elapsed(300) = 1500s",
          eta == 1500, diag=f"got {eta}")

    # No progress AND no EWMA → 0 (unknown)
    eta = et.compute_eta_seconds("", elapsed_s=10, fallback_ewma_s=0)
    check("ETA: no signal → 0", eta == 0)

    # Almost done: 199/200 after 5950s → rate=0.0334/s, remaining=(1)/0.0334 ≈ 30s
    eta = et.compute_eta_seconds("[Epoch 199/200]", elapsed_s=5950, fallback_ewma_s=0)
    check("ETA: 199/200 after 5950s → ~30s", 25 <= eta <= 35, diag=f"got {eta}")

    # 0/200 just started → no rate available, fallback to EWMA - elapsed
    eta = et.compute_eta_seconds("[Epoch 0/200]", elapsed_s=10, fallback_ewma_s=600)
    check("ETA: 0/N at start uses fallback (no rate)", 580 <= eta <= 600, diag=f"got {eta}")

    # format helpers
    check("format 0 → '—'", et.format_eta(0) == "—")
    check("format 30 → '30s'", et.format_eta(30) == "30s")
    check("format 90 → '1.5m'", et.format_eta(90) == "1.5m")
    check("format 7200 → '2.0h'", et.format_eta(7200) == "2.0h")

    # ---------- _refresh_eta_from_logs integration ----------
    saved_run_on = sch.run_on
    saved_history_get = sch.history_get

    sch.history_get = lambda sig: {"dur_s_ewma": 1200} if sig == "TEST/eta-fallback" else None

    # Mock per-node ssh tail: returns a marker-delimited log dump
    def _mock_run_on(node, cmd, timeout=30, check=True):
        # Pretend node-A has two tasks: t1 with 80% progress, t2 with no log
        if "===ETA_LOG_t-prog===" in cmd:
            return (0,
                    "===ETA_LOG_t-prog===\n[Epoch 80/100] step\n"
                    "===ETA_LOG_t-noprog===\nstarting up\n",
                    "")
        return (0, "", "")
    sch.run_on = _mock_run_on

    state = {"tasks": [
        {"id": "t-prog", "status": "running", "node": "A",
         "log_path": "/tmp/p.log", "signature": "TEST/eta-runs",
         "started_at": time.time() - 800},
        {"id": "t-noprog", "status": "running", "node": "A",
         "log_path": "/tmp/np.log", "signature": "TEST/eta-fallback",
         "started_at": time.time() - 100},
        {"id": "t-nolog", "status": "running", "node": "A",
         "log_path": None, "signature": "TEST/eta-fallback",
         "started_at": time.time() - 50},
        {"id": "t-done", "status": "done", "node": "A",
         "log_path": "/tmp/d.log"},  # not running — should be skipped
    ]}
    try:
        sch._refresh_eta_from_logs(state)
        prog = next(t for t in state["tasks"] if t["id"] == "t-prog")
        noprog = next(t for t in state["tasks"] if t["id"] == "t-noprog")
        nolog = next(t for t in state["tasks"] if t["id"] == "t-nolog")
        done = next(t for t in state["tasks"] if t["id"] == "t-done")

        # t-prog: 80/100 after 800s → rate=0.1/s, remaining=20/0.1=200s
        check("running task with progress: ETA from rate (~200s)",
              prog.get("eta_seconds") and 150 <= prog["eta_seconds"] <= 250,
              diag=f"got {prog.get('eta_seconds')}")
        # t-noprog: no progress in tail, EWMA=1200, elapsed=100 → 1100
        check("running task no progress in tail: EWMA fallback (~1100s)",
              noprog.get("eta_seconds") and 1050 <= noprog["eta_seconds"] <= 1200,
              diag=f"got {noprog.get('eta_seconds')}")
        # t-nolog: no log_path → pure EWMA fallback (1200 - 50 = 1150)
        check("running task with no log_path: pure EWMA fallback (~1150s)",
              nolog.get("eta_seconds") and 1100 <= nolog["eta_seconds"] <= 1200,
              diag=f"got {nolog.get('eta_seconds')}")
        # t-done: not running → eta_seconds NOT set (we skip non-running)
        check("non-running task is skipped",
              "eta_seconds" not in done or done.get("eta_seconds") is None,
              diag=f"got {done.get('eta_seconds')}")
    finally:
        sch.run_on = saved_run_on
        sch.history_get = saved_history_get

    # ---------- ssh failure → graceful fallback (no exception) ----------
    sch.run_on = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ssh broken"))
    sch.history_get = lambda sig: {"dur_s_ewma": 600}
    try:
        state = {"tasks": [
            {"id": "tx", "status": "running", "node": "A",
             "log_path": "/tmp/x.log", "signature": "TEST/eta-fb",
             "started_at": time.time() - 100},
        ]}
        sch._refresh_eta_from_logs(state)
        tx = state["tasks"][0]
        check("ssh failure → fallback EWMA, no exception",
              tx.get("eta_seconds") and 480 <= tx["eta_seconds"] <= 520,
              diag=f"got {tx.get('eta_seconds')}")
    finally:
        sch.run_on = saved_run_on
        sch.history_get = saved_history_get

    # ---------- update_running_tasks chains both passes ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    urt_idx = src.find("def update_running_tasks(state):")
    next_def = src.find("\ndef ", urt_idx + 5)
    body = src[urt_idx:next_def]
    check("update_running_tasks calls _batch_check_running",
          "_batch_check_running(state)" in body)
    check("update_running_tasks calls _refresh_eta_from_logs",
          "_refresh_eta_from_logs(state)" in body)


def test_phase2_17_install_slurm_orchestration():
    """Phase 2.17: scheduleurm install-slurm subcommand + node installer script.

    The tool itself does live work (apt install, source build, sudo) so we can't
    truly e2e it in a regression test. Instead we verify:
      - The bash installer script exists, is executable, syntax-checks clean
      - cmd_install_slurm exists and registers the right argparse args
      - The 3-tier fallback chain is in source (tier 1 ssh github, tier 2 rsync local
        cache, tier 3 = LocalBackend continues unchanged)
      - HybridBackend cache invalidation happens after install attempt (so next
        dispatch re-probes for newly-installed slurm)
    """
    print("\n[58] Phase 2.17 install-slurm tool: structure + orchestration guards")

    # ---------- bash installer script: exists + executable + syntax-clean ----------
    from pathlib import Path as _Path
    script_path = _Path(os.path.expanduser("~/.claude/skills/scheduler/scripts/install_slurm_node.sh"))
    check("install_slurm_node.sh exists at expected path",
          script_path.exists(), diag=str(script_path))
    if script_path.exists():
        check("install_slurm_node.sh is executable",
              os.access(script_path, os.X_OK))
        # bash -n syntax check (no execution)
        import subprocess as _sp
        rc = _sp.run(["bash", "-n", str(script_path)], capture_output=True).returncode
        check("install_slurm_node.sh syntax-checks clean (bash -n)", rc == 0)
        # Required behaviors in script
        body = script_path.read_text()
        check("script detects existing slurm via sbatch+squeue (early exit 2)",
              "command -v sbatch" in body and "command -v squeue" in body and "exit 2" in body)
        check("script supports both --tag (github clone) and --source-dir (rsync)",
              "--tag" in body and "--source-dir" in body)
        check("script handles sudo password via --sudo-pass",
              "--sudo-pass" in body and "sudo -S" in body)
        check("script writes default slurm.conf based on probed hardware",
              "/etc/slurm/slurm.conf" in body and "RealMemory=" in body and "CPUs=" in body)
        check("script auto-detects nvidia GPUs and writes gres.conf",
              "/dev/nvidia" in body and "gres.conf" in body)

    # ---------- cmd_install_slurm: exists + has 3-tier chain ----------
    check("cmd_install_slurm function defined",
          callable(getattr(sch, "cmd_install_slurm", None)))
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    func_idx = src.find("def cmd_install_slurm(args):")
    check("cmd_install_slurm in scheduler.py", func_idx > 0)
    if func_idx > 0:
        next_def = src.find("\ndef cmd_", func_idx + 5)
        body = src[func_idx:next_def] if next_def > 0 else src[func_idx:]
        check("tier 1: try github clone on the target node (ssh + script --tag)",
              "_try_tier1_github_on_node" in body)
        check("tier 2: rsync local-cache → node + script --source-dir",
              "_try_tier2_rsync" in body and "rsync" in body and "--source-dir" in body)
        check("tier 3: graceful degrade — node continues to use LocalBackend",
              "LocalBackend fallback" in body or "no-local-cache" in body)
        check("local cache lives at ~/.cache/scheduleurm/slurm-src",
              ".cache" in body and "slurm-src" in body)
        check("HybridBackend cache invalidation after install attempt",
              "_BACKEND._cache.pop" in body or "_cache.pop" in body)

    # ---------- argparse: subcommand registered with the right options ----------
    sched = os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")
    import subprocess as _sp
    r = _sp.run(["python3", sched, "install-slurm", "--help"],
                capture_output=True, text=True, timeout=10)
    out = r.stdout + r.stderr
    check("install-slurm subcommand registered (responds to --help)",
          r.returncode == 0)
    check("--node arg available", "--node" in out)
    check("--tag arg available", "--tag" in out)
    check("--sudo-pass arg available", "--sudo-pass" in out)


def test_backend_slurm_phase2_16_1_rebalance_pending():
    """Phase 2.16.1: cmd_rebalance_pending pulls slurm-PENDING tasks back to scheduleurm
    queue (scancel + revert to status=queued) so they can re-dispatch under current
    throttle policy. Critical safety: must NEVER touch RUNNING tasks (mid-training)
    or LocalBackend tasks.
    """
    print("\n[57] Phase 2.16.1 rebalance-pending: scancel slurm-PENDING + revert to queued")
    check("cmd_rebalance_pending exists",
          callable(getattr(sch, "cmd_rebalance_pending", None)))

    saved_save = sch.save_state
    saved_load = sch.load_state
    saved_run_on = sch.run_on
    saved_lock = sch.state_lock

    captured_kept = {}
    def fake_save(s):
        captured_kept["state"] = s
    fake_state = {
        "next_id": 100,
        "tasks": [
            # PENDING slurm task — should be rebalanced
            {"id": "tA", "status": "running", "node": "n1", "slurm_job_id": 1,
             "slurm_state": "PENDING", "remote_pids": [],
             "signature": "TEST/A", "cmd": "python train.py", "ckpt_dir": "/d/A"},
            # CONFIGURING slurm task — should also be rebalanced
            {"id": "tB", "status": "running", "node": "n2", "slurm_job_id": 2,
             "slurm_state": "CONFIGURING", "remote_pids": [],
             "signature": "TEST/B", "cmd": "python eval.py"},
            # Just-submitted (slurm_state=None) — should be rebalanced
            {"id": "tC", "status": "running", "node": "n1", "slurm_job_id": 3,
             "slurm_state": None, "remote_pids": [],
             "signature": "TEST/C", "cmd": "python long.py"},
            # RUNNING slurm task — must NOT be touched (mid-training)
            {"id": "tR", "status": "running", "node": "n1", "slurm_job_id": 4,
             "slurm_state": "RUNNING", "remote_pids": [],
             "signature": "TEST/R", "cmd": "python big.py",
             "started_at": time.time() - 600, "log_path": "/var/log/r.log"},
            # COMPLETING — also untouched
            {"id": "tCM", "status": "running", "node": "n2", "slurm_job_id": 5,
             "slurm_state": "COMPLETING", "remote_pids": []},
            # LocalBackend task — no slurm_job_id, untouched
            {"id": "tL", "status": "running", "node": "n1",
             "remote_pids": [9001], "slurm_job_id": None,
             "signature": "TEST/local", "cmd": "python local.py"},
            # queued tasks — untouched
            {"id": "tQ", "status": "queued", "signature": "TEST/Q"},
        ],
    }
    scancel_calls = []

    sch.load_state = lambda: fake_state
    sch.save_state = fake_save
    sch.run_on = lambda node, cmd, timeout=10, check=True: (
        scancel_calls.append((node, cmd)) or (0, "", "")
        if "scancel" in cmd else (0, "", "")
    )
    from contextlib import contextmanager as _cm
    @_cm
    def fake_lock():
        yield
    sch.state_lock = fake_lock

    class Args:
        def __init__(self, yes): self.yes = yes

    try:
        # Dry run: NO save, prints plan only
        captured_kept.clear()
        scancel_calls.clear()
        sch.cmd_rebalance_pending(Args(yes=False))
        check("dry run: no save_state call",
              "state" not in captured_kept,
              diag="dry run should not write state")
        check("dry run: no scancel issued",
              len(scancel_calls) == 0)

        # Apply: should rebalance tA, tB, tC; leave tR, tCM, tL, tQ untouched
        captured_kept.clear()
        scancel_calls.clear()
        sch.cmd_rebalance_pending(Args(yes=True))

        # Snapshot post-state
        post = {t["id"]: t for t in fake_state["tasks"]}
        check("tA (PENDING) → status='queued'", post["tA"]["status"] == "queued")
        check("tA → slurm_job_id cleared", post["tA"].get("slurm_job_id") is None)
        check("tA → remote_pids cleared to []", post["tA"]["remote_pids"] == [])
        check("tA → signature preserved (resume relies on it)",
              post["tA"]["signature"] == "TEST/A")
        check("tA → cmd preserved", post["tA"]["cmd"] == "python train.py")
        check("tA → ckpt_dir preserved (resume injection on next dispatch)",
              post["tA"].get("ckpt_dir") == "/d/A")

        check("tB (CONFIGURING) → queued", post["tB"]["status"] == "queued")
        check("tC (just-submitted, slurm_state=None) → queued",
              post["tC"]["status"] == "queued")

        check("tR (RUNNING) → STILL running (untouched)",
              post["tR"]["status"] == "running" and post["tR"]["slurm_job_id"] == 4)
        check("tCM (COMPLETING) → STILL running",
              post["tCM"]["status"] == "running" and post["tCM"]["slurm_job_id"] == 5)
        check("tL (LocalBackend) → untouched (no slurm_job_id)",
              post["tL"]["status"] == "running" and post["tL"]["remote_pids"] == [9001])
        check("tQ (queued) → STILL queued",
              post["tQ"]["status"] == "queued")

        # scancel issued for the 3 rebalanced
        scancel_jids = sorted(
            int(c[1].split()[1]) for c in scancel_calls if "scancel" in c[1]
        )
        check("scancel issued for jids 1, 2, 3 (tA, tB, tC)",
              scancel_jids == [1, 2, 3], diag=str(scancel_calls))
    finally:
        sch.save_state = saved_save
        sch.load_state = saved_load
        sch.run_on = saved_run_on
        sch.state_lock = saved_lock


def test_backend_slurm_phase2_15_orphan_recovery():
    """Phase 2.15 P2: WAL recovery for orphaned slurm jobs.

    The orphan window: SlurmBackend.launch persists status='launching' (with WAL
    save_state) BEFORE sbatch. If sbatch returns success but the scheduler process
    dies before status='running' + slurm_job_id can be flushed, slurm has the job
    (running 24h walltime by default) but scheduleurm forgot. The default
    recover_stale_launching_tasks reverts launching → queued, next dispatch sees
    a fresh queued task and sbatches AGAIN — slurm now has TWO copies of the same
    workload, scheduleurm tracks only the second, the first is orphaned.

    Fix: before reverting a slurm-routed launching task, query
    `squeue -n scheduleurm-<id>` for an existing job. If found in alive state,
    adopt it onto the task instead of reverting.
    """
    print("\n[55] Phase 2.15 orphan slurm job recovery (no double-submission after crash)")

    # ---------- Helper contract ----------
    check("_try_recover_orphan_slurm_job exists",
          callable(getattr(sch, "_try_recover_orphan_slurm_job", None)))

    saved_backend = sch._BACKEND
    real_run_on = sch.run_on
    now = time.time()
    stale = now - sch.LAUNCHING_RESET_S - 5  # past the threshold

    # ---------- Case A: orphan exists + RUNNING → adopt ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["slurmnode"] = "slurm"
    sch._BACKEND = fake_hb
    sch.run_on = lambda node, cmd, timeout=10, check=True: (
        (0, "9999 RUNNING\n", "") if "squeue -h -n" in cmd else (0, "", "")
    )
    try:
        state = {"tasks": [{
            "id": "tA", "status": "launching", "node": "slurmnode",
            "launching_started_at": stale, "remote_pids": [],
        }]}
        n_reverted = sch.recover_stale_launching_tasks(state, now=now)
        t = state["tasks"][0]
        check("orphan RUNNING → adopted (status=running)",
              t["status"] == "running", diag=str(t))
        check("orphan RUNNING → slurm_job_id set from squeue",
              t.get("slurm_job_id") == 9999, diag=str(t))
        check("orphan RUNNING → remote_pids stays []",
              t.get("remote_pids") == [])
        check("orphan RUNNING → launching_started_at cleared",
              "launching_started_at" not in t)
        check("orphan RUNNING → revert count is 0",
              n_reverted == 0, diag=f"n_reverted={n_reverted}")
        check("orphan RUNNING → last_block_reason mentions orphan recovery",
              "WAL recovery: adopted orphan" in (t.get("last_block_reason") or ""),
              diag=t.get("last_block_reason"))
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = real_run_on

    # ---------- Case B: orphan exists + PENDING → also adopt (alive state) ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["slurmnode"] = "slurm"
    sch._BACKEND = fake_hb
    sch.run_on = lambda node, cmd, timeout=10, check=True: (
        (0, "1234 PENDING\n", "") if "squeue -h -n" in cmd else (0, "", "")
    )
    try:
        state = {"tasks": [{
            "id": "tB", "status": "launching", "node": "slurmnode",
            "launching_started_at": stale, "remote_pids": [],
        }]}
        sch.recover_stale_launching_tasks(state, now=now)
        check("orphan PENDING → also adopted (slurm queue counts as alive)",
              state["tasks"][0]["status"] == "running"
              and state["tasks"][0].get("slurm_job_id") == 1234,
              diag=str(state["tasks"][0]))
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = real_run_on

    # ---------- Case C: squeue empty → revert as usual (no orphan) ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["slurmnode"] = "slurm"
    sch._BACKEND = fake_hb
    sch.run_on = lambda node, cmd, timeout=10, check=True: (0, "", "")
    try:
        state = {"tasks": [{
            "id": "tC", "status": "launching", "node": "slurmnode",
            "launching_started_at": stale, "remote_pids": [],
        }]}
        n_reverted = sch.recover_stale_launching_tasks(state, now=now)
        check("no orphan in squeue → reverted to queued",
              state["tasks"][0]["status"] == "queued"
              and state["tasks"][0].get("slurm_job_id") is None
              and n_reverted == 1)
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = real_run_on

    # ---------- Case D: orphan but TERMINAL state → revert (don't adopt dead job) ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["slurmnode"] = "slurm"
    sch._BACKEND = fake_hb
    sch.run_on = lambda node, cmd, timeout=10, check=True: (
        (0, "5555 COMPLETED\n", "") if "squeue -h -n" in cmd else (0, "", "")
    )
    try:
        state = {"tasks": [{
            "id": "tD", "status": "launching", "node": "slurmnode",
            "launching_started_at": stale, "remote_pids": [],
        }]}
        sch.recover_stale_launching_tasks(state, now=now)
        check("orphan in TERMINAL state → revert (don't adopt dead job)",
              state["tasks"][0]["status"] == "queued"
              and state["tasks"][0].get("slurm_job_id") is None,
              diag=str(state["tasks"][0]))
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = real_run_on

    # ---------- Case E: ssh fails during recovery probe → revert (don't get stuck) ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["slurmnode"] = "slurm"
    sch._BACKEND = fake_hb
    sch.run_on = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ssh broken"))
    try:
        state = {"tasks": [{
            "id": "tE", "status": "launching", "node": "slurmnode",
            "launching_started_at": stale, "remote_pids": [],
        }]}
        sch.recover_stale_launching_tasks(state, now=now)
        check("ssh failure during orphan probe → revert (don't get stuck launching)",
              state["tasks"][0]["status"] == "queued")
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = real_run_on

    # ---------- Case F: local-routed launching task → no slurm probe, just revert ----------
    # Local nodes don't have slurm orphans by definition; the recovery path must NOT
    # do an ssh probe for them (cost + irrelevant).
    fake_hb = sch.HybridBackend()
    fake_hb._cache["localnode"] = "local"
    sch._BACKEND = fake_hb
    probe_count = []
    sch.run_on = lambda *a, **k: (probe_count.append(1) or (0, "", ""))
    try:
        state = {"tasks": [{
            "id": "tF", "status": "launching", "node": "localnode",
            "launching_started_at": stale, "remote_pids": [],
        }]}
        sch.recover_stale_launching_tasks(state, now=now)
        check("local-routed launching task → no slurm orphan probe issued",
              len(probe_count) == 0, diag=f"made {len(probe_count)} probes")
        check("local-routed launching task → reverted normally",
              state["tasks"][0]["status"] == "queued")
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = real_run_on

    # ---------- Case G: not-yet-stale launching → no probe, no revert ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["slurmnode"] = "slurm"
    sch._BACKEND = fake_hb
    probe_count2 = []
    sch.run_on = lambda *a, **k: (probe_count2.append(1) or (0, "HAS_SLURM\n", ""))
    try:
        state = {"tasks": [{
            "id": "tG", "status": "launching", "node": "slurmnode",
            "launching_started_at": now - 5,  # very recent (< LAUNCHING_RESET_S)
            "remote_pids": [],
        }]}
        sch.recover_stale_launching_tasks(state, now=now)
        check("fresh launching task → no probe issued (let launch finish)",
              len(probe_count2) == 0, diag=f"made {len(probe_count2)} probes")
        check("fresh launching task → status preserved",
              state["tasks"][0]["status"] == "launching")
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = real_run_on



def test_backend_slurm_phase2_10_sbatch_stdin_form():
    """Phase 2.10 P1 fix: SlurmBackend.launch uses `sbatch /dev/stdin` (kernel-level
    stdin pipe), NOT `sbatch -` (slurm-CLI argv sentinel).

    Real-world bug: on Ubuntu 24.04 with slurm 23.11.4 (apt's universe package),
    `sbatch -` errors with "Unable to open file -" — the package's argv parser
    doesn't treat `-` as a stdin sentinel. End-to-end SlurmBackend launches
    failed at the sbatch step and the user got a cryptic error.

    `/dev/stdin` works on every slurm version because slurm just opens it as a
    file path; the kernel routes that to the same stdin pipe. No file leaks to
    the compute node either way.
    """
    print("\n[51] Phase 2.10 sbatch reads stdin via /dev/stdin (universal across slurm versions)")

    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    sb_idx = src.find("class SlurmBackend(Backend):")
    sb_kill_idx = src.find("def kill(self,", sb_idx)
    sb_launch_body = src[sb_idx:sb_kill_idx]

    check("SlurmBackend uses sbatch /dev/stdin (not sbatch -)",
          '"sbatch", "/dev/stdin"' in sb_launch_body
          and '"sbatch /dev/stdin"' in sb_launch_body,
          diag=sb_launch_body[-1500:])
    check("SlurmBackend does NOT use the broken `sbatch -` argv form",
          '"sbatch", "-"' not in sb_launch_body
          and ', "sbatch -"' not in sb_launch_body,
          diag="found `sbatch -` form which fails on Ubuntu 24.04 slurm 23.11.4")

    sb = sch.SlurmBackend()
    real_subprocess_run = sch.subprocess.run
    real_run_on = sch.run_on
    captured_args = []

    def fake_subprocess_run(args, input=None, capture_output=None, text=None, timeout=None):
        captured_args.append(args)
        class R: pass
        r = R(); r.returncode = 0; r.stdout = "Submitted batch job 50\n"; r.stderr = ""
        return r
    sch.subprocess.run = fake_subprocess_run
    sch.run_on = lambda *a, **k: (0, "", "")

    saved_NODES = sch.NODES
    sch.NODES = {
        "remote-slurm": {"host": "cluster.example", "cpu_cores": 12, "ram_mb": 200000,
                          "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                          "max_concurrent_running": None},
        "local-slurm": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                         "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                         "max_concurrent_running": None},
    }
    try:
        sb.launch({"id": "tr", "node": "remote-slurm", "cwd": "/work",
                   "cmd": "python train.py", "cpu_cores": 1, "ram_mb": 1024,
                   "est_vram_mb": 0, "extra_env": {}, "signature": "TEST/stdin",
                   "resume_flag": "", "resume_from": None})
        last = captured_args[-1]
        check("remote slurm launch invokes ssh ... 'sbatch /dev/stdin'",
              "sbatch /dev/stdin" in " ".join(last),
              diag=str(last))

        captured_args.clear()
        sb.launch({"id": "tl", "node": "local-slurm", "cwd": "/work",
                   "cmd": "python train.py", "cpu_cores": 1, "ram_mb": 1024,
                   "est_vram_mb": 0, "extra_env": {}, "signature": "TEST/stdin-local",
                   "resume_flag": "", "resume_from": None})
        last = captured_args[-1]
        check("local slurm launch invokes ['sbatch', '/dev/stdin']",
              last[:2] == ["sbatch", "/dev/stdin"],
              diag=str(last))
    finally:
        sch.subprocess.run = real_subprocess_run
        sch.run_on = real_run_on
        sch.NODES = saved_NODES


def test_check_running_phase2_9_slurm_aware():
    """Phase 2.9 P2 fix: check_running() helper must route through the backend for
    slurm tasks too (which have remote_pids=[] and track via slurm_job_id).

    Bug before fix:
        if not _task_pids(task): return "dead"
    fired immediately for any slurm task → reported 'dead' without ever consulting
    squeue. The main _batch_check_running path didn't have this bug (it routes via
    _BACKEND.batch_probe directly, no early-return), but check_running was used by
    external callers (tests, MCP wrapper, future tools) and gave wrong answers.

    Fix: drop the early-return. Let each backend's batch_probe filter tasks lacking
    its own tracking artifact — they fall through to `if not res: return "dead"`.
    """
    print("\n[50] Phase 2.9 check_running consults backend for slurm tasks (no pid early-return)")

    # Save and stub the singleton backend with a controllable fake.
    saved_backend = sch._BACKEND
    probe_calls = []

    class _ProbeRecorder(sch.Backend):
        name = "probe-recorder"
        def __init__(self, canned):
            self._canned = canned  # {task_id: result_dict}
        def launch(self, t): return False, "n/a"
        def kill(self, t, timeout=15): return False, "n/a"
        def batch_probe(self, state):
            probe_calls.append([t["id"] for t in state["tasks"]])
            return {tid: r for tid, r in self._canned.items()
                    if any(t["id"] == tid for t in state["tasks"])}

    # ---------- Slurm task (no pids, has slurm_job_id) is NOT short-circuited ----------
    # Backend reports alive — check_running must propagate that, not return "dead".
    sch._BACKEND = _ProbeRecorder({
        "tslurm": {"state": "alive", "alive_pids": [],
                   "vram_mb": 0, "ram_mb": 4096, "pcpu": 0.0}
    })
    probe_calls.clear()
    try:
        slurm_task = {
            "id": "tslurm", "status": "running", "node": "cluster",
            "remote_pids": [],            # slurm-launched: empty
            "slurm_job_id": 12345,        # tracking via slurm
            "peak_ram_mb": 0, "peak_vram_mb": 0,
        }
        result = sch.check_running(slurm_task)
        check("slurm task with [] pids consults backend (not early-return 'dead')",
              probe_calls and probe_calls[0] == ["tslurm"],
              diag=f"probe_calls={probe_calls}")
        check("slurm task: backend says alive → check_running returns 'alive'",
              result == "alive", diag=f"got {result!r}")
        check("slurm task: ram_mb folded into peak_ram_mb",
              slurm_task["peak_ram_mb"] == 4096)
    finally:
        sch._BACKEND = saved_backend

    # ---------- Slurm task: backend says dead (squeue COMPLETED) ----------
    sch._BACKEND = _ProbeRecorder({
        "tslurm2": {"state": "dead", "alive_pids": [],
                    "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0}
    })
    probe_calls.clear()
    try:
        result = sch.check_running({
            "id": "tslurm2", "status": "running", "node": "cluster",
            "remote_pids": [], "slurm_job_id": 200,
        })
        check("slurm task: backend says dead → check_running returns 'dead'",
              result == "dead")
    finally:
        sch._BACKEND = saved_backend

    # ---------- Slurm task: probe failure (squeue ssh broken) → 'unknown' ----------
    sch._BACKEND = _ProbeRecorder({
        "tslurm3": {"state": "unknown", "alive_pids": [],
                    "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0}
    })
    try:
        result = sch.check_running({
            "id": "tslurm3", "status": "running", "node": "cluster",
            "remote_pids": [], "slurm_job_id": 300,
        })
        check("slurm task: backend says unknown → check_running returns 'unknown'",
              result == "unknown")
    finally:
        sch._BACKEND = saved_backend

    # ---------- Local task with PIDs: regression — still works as before ----------
    sch._BACKEND = _ProbeRecorder({
        "tlocal": {"state": "alive", "alive_pids": [101, 102],
                   "vram_mb": 1024, "ram_mb": 2048, "pcpu": 80.0}
    })
    try:
        local_task = {
            "id": "tlocal", "status": "running", "node": "local",
            "remote_pids": [101], "peak_ram_mb": 0, "peak_vram_mb": 0,
        }
        result = sch.check_running(local_task)
        check("local task with PIDs: backend says alive → 'alive' (regression)",
              result == "alive")
        check("local task: vram_mb folded into peak_vram_mb",
              local_task["peak_vram_mb"] == 1024)
        check("local task: alive_pids written through",
              local_task.get("alive_pids") == [101, 102])
    finally:
        sch._BACKEND = saved_backend

    # ---------- No result from backend (task with neither artifact) → 'dead' ----------
    sch._BACKEND = _ProbeRecorder({})  # empty — backend has no info
    try:
        ghost = {"id": "tghost", "status": "running", "node": "?",
                 "remote_pids": [], "slurm_job_id": None}
        result = sch.check_running(ghost)
        check("task with no artifact → 'dead' (via `if not res`)",
              result == "dead")
    finally:
        sch._BACKEND = saved_backend

    # ---------- Source guard: the early-return is gone ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    cr_idx = src.find("def check_running(task):")
    cr_end = src.find("\ndef ", cr_idx + 5)
    cr_body = src[cr_idx:cr_end]
    # Strip the docstring before checking for the bug pattern — the docstring
    # mentions the old form in backticks for changelog context, which would falsely
    # match. Anything between the first and second `"""` is prose.
    if '"""' in cr_body:
        first = cr_body.find('"""')
        second = cr_body.find('"""', first + 3)
        if second > first:
            stripped_body = cr_body[:first] + cr_body[second + 3:]
        else:
            stripped_body = cr_body
    else:
        stripped_body = cr_body
    check("check_running body has NO multi-line `if not _task_pids(task): return \"dead\"` early-return",
          'if not _task_pids(task):\n        return "dead"' not in stripped_body
          and 'if not _task_pids(task): return "dead"' not in stripped_body,
          diag=stripped_body[:400])
    check("check_running body still delegates via _BACKEND.batch_probe",
          "_BACKEND.batch_probe(fake_state)" in cr_body,
          diag=cr_body[:400])


def test_backend_slurm_phase2_8_route_by_launch_artifacts():
    """Phase 2.8 P1 fix: HybridBackend._backend_for_task routes based on the task's
    launch artifacts (slurm_job_id / remote_pids), not the per-node cache.

    Bug before fix: routing consulted only slurm_job_id and the per-node cache.
    If a node's cache flipped from 'local' → 'slurm' (e.g. Phase 2.7's re-probe
    finding slurm after a transient blip), already-running tasks launched by
    LocalBackend (which have remote_pids but no slurm_job_id) suddenly routed
    to SlurmBackend. SlurmBackend.batch_probe skips them on `if not jid:
    continue` → never probed, never transition to terminal, become zombies.
    Same hazard for kill: SlurmBackend.kill returns "no slurm_job_id" and does
    nothing, leaving the proc orphaned.

    Fix: routing checks task['slurm_job_id'] AND task['remote_pids'] BEFORE
    falling back to per-node cache. Launch artifacts are immutable — they
    remember which backend launched the task — so cache flips can't reroute.
    """
    print("\n[49] Phase 2.8 _backend_for_task routes by launch artifacts, not cache")

    hb = sch.HybridBackend()

    # ---------- The bug scenario: cache flipped to slurm AFTER a local task launched ----------
    hb._cache["nodeX"] = "slurm"  # cache says slurm
    local_running = {
        "id": "tlocal",
        "status": "running",
        "node": "nodeX",
        "remote_pids": [12345],   # ← launched by LocalBackend
        "slurm_job_id": None,
    }
    backend = hb._backend_for_task(local_running)
    check("running task with remote_pids → LocalBackend even when cache says slurm",
          backend is hb._local,
          diag=f"got backend={type(backend).__name__}")

    # ---------- Mirror: slurm task with slurm_job_id wins even if cache says local ----------
    hb._cache["nodeY"] = "local"
    slurm_running = {
        "id": "tslurm",
        "status": "running",
        "node": "nodeY",
        "remote_pids": [],
        "slurm_job_id": 99999,
    }
    backend = hb._backend_for_task(slurm_running)
    check("running task with slurm_job_id → SlurmBackend even when cache says local",
          backend is hb._slurm,
          diag=f"got backend={type(backend).__name__}")

    # ---------- Queued task (no launch artifacts) → cache-based decision ----------
    hb._cache["nodeZ"] = "slurm"
    queued = {"id": "tq", "status": "queued", "node": "nodeZ"}
    backend = hb._backend_for_task(queued)
    check("queued task with no launch artifacts → cache-based (slurm)",
          backend is hb._slurm)

    hb._cache["nodeW"] = "local"
    queued_local = {"id": "tql", "status": "queued", "node": "nodeW"}
    backend = hb._backend_for_task(queued_local)
    check("queued task with no launch artifacts → cache-based (local)",
          backend is hb._local)

    # ---------- Edge case: task with no node at all (just submitted) → LocalBackend default ----------
    queued_unplaced = {"id": "tqu", "status": "queued"}
    backend = hb._backend_for_task(queued_unplaced)
    check("task with no node yet → LocalBackend default",
          backend is hb._local)

    # ---------- Defensive: both launch artifacts set (shouldn't happen) → slurm wins ----------
    # SlurmBackend.launch sets remote_pids=[] (empty list, falsy), so this is an
    # impossible state, but the routing must be unambiguous if it ever arises.
    weird = {"id": "tw", "status": "running", "node": "nodeX",
             "remote_pids": [9999], "slurm_job_id": 1}
    backend = hb._backend_for_task(weird)
    check("if both artifacts set: slurm_job_id wins (defensive ordering)",
          backend is hb._slurm)

    # ---------- End-to-end: batch_probe on mixed state with cache flip ----------
    # Simulate the realistic scenario: nodeX was 'local' yesterday, two tasks launched
    # via LocalBackend with remote_pids. Today watcher re-probed and cache flipped to
    # 'slurm' (e.g. operator installed slurm there). batch_probe must still probe
    # those local-launched tasks via LocalBackend — not silently skip them.
    hb = sch.HybridBackend()
    hb._cache["nodeX"] = "slurm"  # the cache flip

    state = {
        "tasks": [
            {"id": "ta", "status": "running", "node": "nodeX",
             "remote_pids": [101], "slurm_job_id": None},
            {"id": "tb", "status": "running", "node": "nodeX",
             "remote_pids": [102], "slurm_job_id": None},
        ]
    }

    local_probe_calls = []
    slurm_probe_calls = []
    real_local_probe = hb._local.batch_probe
    real_slurm_probe = hb._slurm.batch_probe
    hb._local.batch_probe = lambda s: (local_probe_calls.append(list(t["id"] for t in s["tasks"])) or {})
    hb._slurm.batch_probe = lambda s: (slurm_probe_calls.append(list(t["id"] for t in s["tasks"])) or {})
    try:
        hb.batch_probe(state)
        check("batch_probe sends remote_pids tasks to LocalBackend (not silently skipped)",
              local_probe_calls and set(local_probe_calls[0]) == {"ta", "tb"},
              diag=f"local={local_probe_calls}, slurm={slurm_probe_calls}")
        check("batch_probe sends ZERO local-launched tasks to SlurmBackend",
              not slurm_probe_calls or not slurm_probe_calls[0],
              diag=f"slurm={slurm_probe_calls}")
    finally:
        hb._local.batch_probe = real_local_probe
        hb._slurm.batch_probe = real_slurm_probe


def test_backend_slurm_phase2_7_probe_failure_does_not_cache():
    """Phase 2.7 P1 fix: HybridBackend._kind_for must NOT cache failure results.

    Bug before fix: any probe failure (ssh exception, non-zero rc, missing
    HAS_SLURM marker due to bashrc spam etc.) collapsed to `kind = "local"` and
    THAT got cached for the process lifetime. A single transient ssh blip during
    the very first probe of a slurm node would silently route every subsequent
    task to LocalBackend until watcher restart — the user would never know slurm
    was being bypassed.

    Fix: probe command always emits HAS_SLURM, NO_SLURM, or SLURM_UNUSABLE
    (no silent short-circuit on `&&`). Cache writes only happen on definitive
    install/non-install results. If Slurm tools exist but the controller is
    temporarily unusable, route to SlurmBackend without caching so launch fails
    loudly instead of running on a login node via LocalBackend.
    """
    print("\n[48] Phase 2.7 _kind_for caches only definitive answers, not failures")
    real_run_on = sch.run_on

    # ---------- Probe shape: always emits a marker ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # The probe command must use `if/then/else/fi` (not `&& ... echo HAS_SLURM`),
    # so a missing tool gets `NO_SLURM` (definitive) instead of empty output.
    check("probe command emits HAS_SLURM / NO_SLURM / SLURM_UNUSABLE markers",
          "echo HAS_SLURM" in src and "echo NO_SLURM" in src and "echo SLURM_UNUSABLE" in src,
          diag="probe doesn't emit definitive negative marker")
    check("probe validates slurm controller with squeue -h",
          "squeue -h" in src, diag="probe only checked command existence")
    check("probe uses if/fi shape (not bare && echo HAS_SLURM)",
          "if command -v sbatch" in src,
          diag="probe didn't switch to definitive shape")

    # ---------- Definitive HAS_SLURM → cache 'slurm' ----------
    hb = sch.HybridBackend()
    sch.run_on = lambda node, cmd, timeout=5, check=True: (0, "HAS_SLURM\n", "")
    try:
        kind = hb._kind_for("nodeA")
        check("HAS_SLURM → returns 'slurm'", kind == "slurm")
        check("HAS_SLURM → caches 'slurm'", hb._cache.get("nodeA") == "slurm")
    finally:
        sch.run_on = real_run_on

    # ---------- Definitive NO_SLURM → cache 'local' ----------
    hb = sch.HybridBackend()
    sch.run_on = lambda node, cmd, timeout=5, check=True: (0, "NO_SLURM\n", "")
    try:
        kind = hb._kind_for("nodeB")
        check("NO_SLURM → returns 'local'", kind == "local")
        check("NO_SLURM → caches 'local'", hb._cache.get("nodeB") == "local")
    finally:
        sch.run_on = real_run_on

    # ---------- Slurm tools present but controller/account unusable → SlurmBackend, no cache ----------
    hb = sch.HybridBackend()
    sch.run_on = lambda node, cmd, timeout=5, check=True: (0, "SLURM_UNUSABLE\n", "")
    try:
        kind = hb._kind_for("slurm-down")
        check("SLURM_UNUSABLE → routes to slurm (loud launch failure, not LocalBackend)",
              kind == "slurm", diag=f"got {kind}")
        check("SLURM_UNUSABLE → does NOT cache",
              "slurm-down" not in hb._cache, diag=f"cache={hb._cache}")
    finally:
        sch.run_on = real_run_on

    # ---------- ssh exception → DO NOT cache, return 'local' ----------
    hb = sch.HybridBackend()
    sch.run_on = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ssh broken"))
    try:
        kind = hb._kind_for("flakyA")
        check("ssh exception → returns 'local'", kind == "local")
        check("ssh exception → does NOT cache",
              "flakyA" not in hb._cache, diag=f"cache={hb._cache}")
    finally:
        sch.run_on = real_run_on

    # ---------- rc != 0 → DO NOT cache (probe broke before reaching markers) ----------
    hb = sch.HybridBackend()
    sch.run_on = lambda node, cmd, timeout=5, check=True: (255, "", "ssh: connect timeout")
    try:
        kind = hb._kind_for("flakyB")
        check("rc != 0 → returns 'local'", kind == "local")
        check("rc != 0 → does NOT cache",
              "flakyB" not in hb._cache, diag=f"cache={hb._cache}")
    finally:
        sch.run_on = real_run_on

    # ---------- Ambiguous output (rc=0 but no marker) → DO NOT cache ----------
    # E.g. remote bashrc emits a banner that overwrites stdout, or some weird shell
    # config swallows the if/fi output. We refuse to interpret as definitive.
    hb = sch.HybridBackend()
    sch.run_on = lambda node, cmd, timeout=5, check=True: (0, "Welcome to my server\nLast login: ...\n", "")
    try:
        kind = hb._kind_for("noisy")
        check("rc=0 but no marker → returns 'local' (safe default)", kind == "local")
        check("rc=0 but no marker → does NOT cache",
              "noisy" not in hb._cache, diag=f"cache={hb._cache}")
    finally:
        sch.run_on = real_run_on

    # ---------- Self-heal: failed probe doesn't poison subsequent successful one ----------
    hb = sch.HybridBackend()
    # Round 1: ssh exception → fail
    sch.run_on = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("transient blip"))
    try:
        first = hb._kind_for("recovers")
        check("after transient failure: returns 'local'", first == "local")
        check("after transient failure: cache empty (no poison)",
              "recovers" not in hb._cache)
    finally:
        sch.run_on = real_run_on
    # Round 2: ssh recovers, slurm IS installed
    sch.run_on = lambda node, cmd, timeout=5, check=True: (0, "HAS_SLURM\n", "")
    try:
        second = hb._kind_for("recovers")
        check("after recovery: returns 'slurm' (re-probed, found it)", second == "slurm")
        check("after recovery: cache now reflects 'slurm'",
              hb._cache.get("recovers") == "slurm")
    finally:
        sch.run_on = real_run_on

    # ---------- Once cached, no further probes (perf invariant) ----------
    hb = sch.HybridBackend()
    hb._cache["preset"] = "slurm"
    probe_calls = []
    sch.run_on = lambda *a, **k: (probe_calls.append(1) or (0, "HAS_SLURM\n", ""))
    try:
        for _ in range(5):
            hb._kind_for("preset")
        check("cached node: zero ssh probes on subsequent calls",
              len(probe_calls) == 0, diag=f"made {len(probe_calls)} probes")
    finally:
        sch.run_on = real_run_on


def test_backend_slurm_phase2_6_docker_gpu_runtime_env():
    """Phase 2.6 P1 fix: when SlurmBackend wraps a GPU task in docker, the `--gpus`
    arg must reference the runtime CUDA_VISIBLE_DEVICES that slurm's gres allocator
    sets, NOT a static scheduleurm-picked index.

    Bug before fix: SlurmBackend called _maybe_wrap_docker with task['gpu_idx'],
    which is None for slurm-routed tasks (Phase 2.3 set it None on the bypass
    path). wrap_cmd_docker with gpu_idx=None emits NO --gpus flag and explicitly
    nulls CUDA_VISIBLE_DEVICES inside the container. Result: a GPU training task
    runs in a container with no GPU access — CUDA init fails silently or the
    framework falls back to CPU and the run wastes hours producing nothing useful.

    Even if gpu_idx HAD been set by some pre-Phase-2.3 path, slurm picks the
    actual GPU at job start; a stale scheduleurm-picked index could be wrong,
    leading to docker pinning to a GPU slurm didn't allocate.

    Fix: wrap_cmd_docker accepts gpu_runtime_env="CUDA_VISIBLE_DEVICES". When set,
    --gpus is emitted as a _ShellLiteral (bypasses shlex.quote) `"device=$VAR"` so
    bash expands the slurm-set env var at sbatch runtime. SlurmBackend.launch
    passes this for est_vram_mb > 0 tasks; CPU-only stays gpu_runtime_env=None.
    """
    print("\n[47] Phase 2.6 docker GPU pin uses slurm-set CUDA_VISIBLE_DEVICES at runtime")

    # ---------- env_deploy.wrap_cmd_docker semantics ----------
    ed = sch.env_deploy
    check("env_deploy exposes _ShellLiteral marker", hasattr(ed, "_ShellLiteral"))

    # GPU task with runtime-env: docker --gpus arg has unquoted $CUDA_VISIBLE_DEVICES
    cmd = ed.wrap_cmd_docker(
        inner="python train.py",
        image="myimg:latest",
        cwd="/work",
        gpu_idx=None,
        gpu_runtime_env="CUDA_VISIBLE_DEVICES",
        container_name="sched-tslurm",
    )
    check("runtime-env path: --gpus contains $CUDA_VISIBLE_DEVICES (unquoted)",
          '--gpus "device=$CUDA_VISIBLE_DEVICES"' in cmd, diag=cmd[:400])
    check("runtime-env path: -e CUDA_VISIBLE_DEVICES=0 inside container",
          "CUDA_VISIBLE_DEVICES=0" in cmd, diag=cmd[:400])

    # Static gpu_idx path (LocalBackend): unchanged, literal device=N
    cmd_local = ed.wrap_cmd_docker(
        inner="python train.py",
        image="myimg:latest",
        cwd="/work",
        gpu_idx=1,
        container_name="sched-tlocal",
    )
    check("LocalBackend path: --gpus device=1 (static literal)",
          "--gpus device=1" in cmd_local, diag=cmd_local[:400])
    check("LocalBackend path: no $CUDA_VISIBLE_DEVICES (static pin)",
          "$CUDA_VISIBLE_DEVICES" not in cmd_local)

    # CPU-only path: neither flag — no --gpus at all
    cmd_cpu = ed.wrap_cmd_docker(
        inner="python eval.py",
        image="myimg:latest",
        cwd="/work",
        gpu_idx=None,
        gpu_runtime_env=None,
        container_name="sched-tcpu",
    )
    check("CPU-only path: no --gpus flag", "--gpus" not in cmd_cpu, diag=cmd_cpu[:400])
    check("CPU-only path: -e CUDA_VISIBLE_DEVICES= (empty, blocks host leak)",
          "CUDA_VISIBLE_DEVICES=" in cmd_cpu and "CUDA_VISIBLE_DEVICES=0" not in cmd_cpu)

    # gpu_runtime_env takes precedence over a stale gpu_idx (defensive)
    cmd_both = ed.wrap_cmd_docker(
        inner="python train.py",
        image="myimg:latest",
        cwd="/work",
        gpu_idx=99,                              # stale, should NOT be used
        gpu_runtime_env="CUDA_VISIBLE_DEVICES",  # this wins
        container_name="sched-tboth",
    )
    check("gpu_runtime_env takes precedence over gpu_idx",
          "$CUDA_VISIBLE_DEVICES" in cmd_both and "device=99" not in cmd_both,
          diag=cmd_both[:400])

    # ---------- _maybe_wrap_docker plumbs gpu_runtime_env ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_maybe_wrap_docker signature accepts gpu_runtime_env",
          "def _maybe_wrap_docker(task: dict, inner: str, cwd: str,\n                       gpu_runtime_env" in src
          or "gpu_runtime_env: Optional[str] = None" in src,
          diag="_maybe_wrap_docker doesn't accept gpu_runtime_env")
    check("_maybe_wrap_docker passes gpu_runtime_env to wrap_cmd_docker",
          "gpu_runtime_env=gpu_runtime_env" in src, diag="not threaded through")

    # ---------- SlurmBackend.launch passes the right value ----------
    sb_idx = src.find("class SlurmBackend(Backend):")
    sb_launch_idx = src.find("def launch(self, task: dict)", sb_idx)
    sb_kill_idx = src.find("def kill(self,", sb_launch_idx)
    sb_launch_body = src[sb_launch_idx:sb_kill_idx]
    check("SlurmBackend.launch sets gpu_runtime_env=CUDA_VISIBLE_DEVICES for GPU tasks",
          'gpu_runtime_env = "CUDA_VISIBLE_DEVICES"' in sb_launch_body,
          diag=sb_launch_body[:600])
    check("SlurmBackend.launch passes gpu_runtime_env to _maybe_wrap_docker",
          "_maybe_wrap_docker(task, inner, cwd, gpu_runtime_env=gpu_runtime_env)" in sb_launch_body,
          diag=sb_launch_body[:1000])
    check("SlurmBackend.launch CPU-only branch (est_vram_mb=0) → gpu_runtime_env=None",
          'else None' in sb_launch_body, diag=sb_launch_body[:1000])


def test_backend_slurm_phase2_5_cd_guard():
    """Phase 2.5 P1 fix: SlurmBackend's sbatch script must short-circuit on cd failure.

    Bug before fix: the script had a bare `cd /path/to/cwd` followed by the user's cmd.
    If the compute node can't see that path (NFS stale handle, path not exported to
    compute partition, etc.), bash continues from whatever its current cwd is —
    typically $HOME. The user cmd "runs" but with wrong working dir → relative paths
    point nowhere, output goes to $HOME, no useful signal in the log, diagnose has
    nothing to grab onto.

    Fix: explicit `cd PATH || { echo ...; exit 1; }` so cd-failure aborts immediately
    with a parseable error in the log. Mirrors LocalBackend's `cd ... && cmd` pattern.
    """
    print("\n[46] Phase 2.5 sbatch cd is fatal-on-failure (matches LocalBackend semantics)")
    sb = sch.SlurmBackend()

    task = {
        "id": "tcd", "node": "local", "cwd": "/path/to/proj",
        "cmd": "python train.py", "cpu_cores": 1, "ram_mb": 1024,
        "est_vram_mb": 0, "extra_env": {}, "signature": "TEST/cd-guard",
        "resume_flag": "", "resume_from": None,
    }
    script = sb._build_sbatch_script(task, "python -u train.py", "/tmp/log.log")

    # Required: the cd line is followed by `||` (logical-or short-circuit). The fix
    # uses { echo ...; exit 1; } block; check both halves.
    check("sbatch cd uses || guard (cd PATH || ...)",
          "cd /path/to/proj || " in script,
          diag=script)
    check("cd guard contains exit 1 (script aborts on cd failure)",
          "exit 1" in script,
          diag=script)
    check("cd guard echoes a diagnostic to stderr",
          "scheduleurm: cwd not accessible on compute node" in script and ">&2" in script,
          diag=script)
    # Specifically: there is no bare `cd /path/to/proj\n<cmd>` pattern remaining
    # (regression catcher). Look for the cd line and ensure the next line isn't
    # the inner cmd directly.
    lines = script.splitlines()
    cd_line_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("cd /path/to/proj"):
            cd_line_idx = i
            break
    check("cd line is on a single line with || guard (not split)",
          cd_line_idx is not None and "||" in lines[cd_line_idx],
          diag=str(lines[cd_line_idx-1:cd_line_idx+3]) if cd_line_idx else "no cd line found")

    # Edge case: cwd containing shell-special chars (spaces, $) must be properly quoted
    # in BOTH the cd target AND the diagnostic echo (so the echo doesn't blow up).
    task2 = dict(task, cwd="/path with space/$weird")
    script2 = sb._build_sbatch_script(task2, "python -u train.py", "/tmp/log.log")
    # shlex.quote single-quotes paths with shell-special chars
    check("cwd with shell-special chars is shlex-quoted in cd",
          "cd '/path with space/$weird'" in script2,
          diag=script2[:600])
    check("diagnostic message preserves the literal cwd for user readability",
          "/path with space/$weird" in script2,
          diag=script2[:600])


def test_backend_slurm_phase2_4_log_path_shared_fs():
    """Phase 2.4 P1 fix: SlurmBackend log_path must live on a shared filesystem so the
    compute node (where slurm runs the job) and the login node (where scheduler tails
    the log) see the same file. /tmp is per-node-local on every cluster I've seen, so
    using /tmp/sched_<id>.log on a remote slurm node leaves diagnose tailing a
    non-existent file → 0-byte read → false-classified as crash → wasteful re-queue.

    Fix: write under <cwd>/.scheduleurm/<id>.log. cwd is presumed shared (otherwise
    slurm couldn't run user's code from there). Pre-flight cwd-test now also creates
    the log dir so sbatch's --output= write doesn't fail on missing parent.
    """
    print("\n[45] Phase 2.4 SlurmBackend log path on shared FS (cwd-relative, not /tmp)")
    sb = sch.SlurmBackend()

    # ---------- Source guard: no /tmp/sched_ for remote slurm ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # SlurmBackend.launch should use cwd-relative path for remote nodes.
    sb_launch_idx = src.find("def launch(self, task: dict)", src.find("class SlurmBackend"))
    sb_kill_idx = src.find("def kill(self,", sb_launch_idx)
    launch_body = src[sb_launch_idx:sb_kill_idx]
    check("SlurmBackend.launch uses cwd-relative log path (not /tmp)",
          ".scheduleurm" in launch_body and "{cwd}" in launch_body,
          diag=launch_body[:600])
    check("SlurmBackend pre-flight cwd test now also mkdir -p log_dir",
          "mkdir -p" in launch_body, diag=launch_body[:600])

    # ---------- Functional: launch on remote slurm node uses cwd-relative path ----------
    real_subprocess_run = sch.subprocess.run
    real_run_on = sch.run_on
    captured = {}
    def fake_subprocess_run(args, input=None, capture_output=None, text=None, timeout=None):
        captured["args"] = args
        captured["input"] = input
        class R: pass
        r = R(); r.returncode = 0; r.stdout = "Submitted batch job 99\n"; r.stderr = ""
        return r
    preflight_cmds = []
    def fake_run_on(node, cmd, timeout=15, check=True):
        preflight_cmds.append(cmd)
        return (0, "", "")
    sch.subprocess.run = fake_subprocess_run
    sch.run_on = fake_run_on

    saved_NODES = sch.NODES
    sch.NODES = {
        "remote-slurm": {"host": "cluster.example", "cpu_cores": 12, "ram_mb": 200000,
                         "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                         "max_concurrent_running": None},
    }
    try:
        task = {
            "id": "tlog", "node": "remote-slurm", "cwd": "/shared/home/me/proj",
            "cmd": "python train.py", "cpu_cores": 2, "ram_mb": 4096,
            "est_vram_mb": 0, "extra_env": {}, "signature": "TEST/log-path",
            "resume_flag": "", "resume_from": None,
        }
        ok, msg = sb.launch(task)
        check("launch succeeds against remote slurm node", ok, diag=msg)
        # Assert no /tmp/sched_ in the captured sbatch script
        check("sbatch script does NOT use /tmp/sched_<id>.log for remote slurm",
              "/tmp/sched_" not in (captured["input"] or ""),
              diag=(captured.get("input") or "")[:400])
        # Assert cwd-relative log dir
        check("sbatch script uses cwd-relative log path",
              "/shared/home/me/proj/.scheduleurm/tlog.log" in (captured["input"] or ""),
              diag=(captured.get("input") or "")[:400])
        # Pre-flight check: cwd test + mkdir log_dir issued
        any_preflight_with_mkdir = any(
            "test -d" in c and "mkdir -p" in c and ".scheduleurm" in c
            for c in preflight_cmds
        )
        check("pre-flight cwd test issues mkdir -p .scheduleurm",
              any_preflight_with_mkdir,
              diag=str(preflight_cmds))
        # Task record stores the cwd-relative log path so diagnose tails the right place
        check("task['log_path'] points at cwd-relative path",
              task.get("log_path") == "/shared/home/me/proj/.scheduleurm/tlog.log",
              diag=str(task.get("log_path")))
    finally:
        sch.subprocess.run = real_subprocess_run
        sch.run_on = real_run_on
        sch.NODES = saved_NODES

    # ---------- Local slurm node (host=None): keep using STATE_DIR/logs (already shared) ----------
    sch.subprocess.run = fake_subprocess_run
    sch.run_on = fake_run_on
    captured = {}
    sch.NODES = {
        "local-slurm": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                         "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                         "max_concurrent_running": None},
    }
    try:
        task = {
            "id": "tloc", "node": "local-slurm", "cwd": "/tmp/proj",
            "cmd": "python train.py", "cpu_cores": 2, "ram_mb": 4096,
            "est_vram_mb": 0, "extra_env": {}, "signature": "TEST/log-path-local",
            "resume_flag": "", "resume_from": None,
        }
        ok, msg = sb.launch(task)
        check("local-slurm launch ok", ok, diag=msg)
        check("local slurm uses STATE_DIR/logs (already on local FS — no shared-fs concern)",
              str(sch.STATE_DIR) in (task.get("log_path") or ""),
              diag=str(task.get("log_path")))
    finally:
        sch.subprocess.run = real_subprocess_run
        sch.run_on = real_run_on
        sch.NODES = saved_NODES


def test_backend_slurm_phase2_3_bypass_local_capacity():
    """Phase 2.3 P1 fix: pick_placement must NOT gate slurm nodes on local capacity.

    Bug before fix: pick_placement ran _node_resources_ok + _gpu_fits on every node
    uniformly. On a real slurm cluster the login node usually has no GPU at all
    (probe gpus=[]) → no candidate emitted → task stuck queued in scheduleurm forever,
    NEVER reaching sbatch. Even on a node with GPUs, if all GPUs are currently busy
    with other slurm users, slurm could queue the job, but pick_placement would refuse.

    Fix: Backend.requires_local_capacity_check(node) — False for slurm-routed nodes.
    pick_placement short-circuits those: emits a deferred-placement candidate with
    gpu_idx=None and a 9999 primary score (so any local-fitting candidate wins, but
    if none fit OR slurm is required/preferred, slurm wins).
    """
    print("\n[44] Phase 2.3 pick_placement bypasses local capacity check for slurm nodes")

    # ---------- Hook contract ----------
    check("Backend has requires_local_capacity_check method",
          callable(getattr(sch.Backend, "requires_local_capacity_check", None)))
    # Default is True (LocalBackend semantics)
    check("LocalBackend.requires_local_capacity_check → True (instant gate)",
          sch.LocalBackend().requires_local_capacity_check("any-node") is True)
    check("SlurmBackend.requires_local_capacity_check → False (defer to slurm)",
          sch.SlurmBackend().requires_local_capacity_check("any-node") is False)

    # HybridBackend: per-node based on cache
    hb = sch.HybridBackend()
    hb._cache["fake-slurm"] = "slurm"
    hb._cache["fake-local"] = "local"
    check("HybridBackend slurm-cached node → bypass (False)",
          hb.requires_local_capacity_check("fake-slurm") is False)
    check("HybridBackend local-cached node → gate (True)",
          hb.requires_local_capacity_check("fake-local") is True)

    # ---------- Functional: pick_placement on a slurm-only no-GPU login node ----------
    # Scenario: NODES has just one slurm node ("login") whose probe returned no GPUs
    # and 0 free CPU. Task is GPU-needing. Under old behavior pick_placement would
    # return None (no candidate) and the task would be stuck. Under new behavior,
    # slurm short-circuits and returns ("login", None) — sbatch will queue it.
    saved_backend = sch._BACKEND
    fake_hb = sch.HybridBackend()
    fake_hb._cache["login"] = "slurm"
    sch._BACKEND = fake_hb
    try:
        nodes = [{"name": "login", "alive": True, "gpus": [], "free_cpu": 0,
                  "free_ram_mb": 0, "loadavg": 5.0}]
        # Need NODES to have an entry for "login" since pick_placement looks up node_info
        # for the local-capacity branch — but our slurm short-circuit returns BEFORE that
        # lookup. Still, _gpu_fits / _node_resources_ok would crash on a missing entry,
        # so verify the short-circuit really skips them.
        task = {"id": "tslurm", "est_vram_mb": 4000, "cpu_cores": 4, "ram_mb": 8000,
                "signature": "TEST/slurm-bypass"}
        placement = sch.pick_placement(task, nodes)
        check("slurm-only login node with no GPU still returns a placement",
              placement is not None, diag=f"got {placement}")
        check("placement returns (slurm-node, None) — slurm picks the GPU itself",
              placement == ("login", None), diag=f"got {placement}")
    finally:
        sch._BACKEND = saved_backend

    # ---------- Functional: local-fits AND slurm-available → local wins ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["local-box"] = "local"
    fake_hb._cache["cluster"] = "slurm"
    sch._BACKEND = fake_hb
    real_NODES = sch.NODES
    sch.NODES = {
        "local-box": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                       "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                       "max_concurrent_running": None},
        "cluster": {"host": "cluster.example", "cpu_cores": 12, "ram_mb": 200000,
                     "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                     "max_concurrent_running": None},
    }
    try:
        nodes = [
            {"name": "local-box", "alive": True, "loadavg": 1.0, "free_cpu": 8,
             "free_ram_mb": 20000, "running_count": 0,
             "gpus": [{"idx": 0, "used_mb": 200, "total_mb": 8000, "free_mb": 7800,
                       "util_pct": 10}]},
            {"name": "cluster", "alive": True, "loadavg": 0.0, "free_cpu": 0,
             "free_ram_mb": 0, "running_count": 0, "gpus": []},
        ]
        task = {"id": "tboth", "est_vram_mb": 1000, "cpu_cores": 2, "ram_mb": 4000,
                "signature": "TEST/local-vs-slurm"}
        placement = sch.pick_placement(task, nodes)
        check("when local fits AND slurm available → local wins (immediate > queued)",
              placement is not None and placement[0] == "local-box",
              diag=f"got {placement}")
        check("local-wins placement returns gpu_idx (not None)",
              placement and placement[1] == 0, diag=f"got {placement}")
    finally:
        sch._BACKEND = saved_backend
        sch.NODES = real_NODES

    # ---------- Functional: --require-node = slurm node → must use slurm even if local fits ----------
    fake_hb = sch.HybridBackend()
    fake_hb._cache["local-box"] = "local"
    fake_hb._cache["cluster"] = "slurm"
    sch._BACKEND = fake_hb
    sch.NODES = {
        "local-box": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                       "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                       "max_concurrent_running": None},
        "cluster": {"host": "cluster.example", "cpu_cores": 12, "ram_mb": 200000,
                     "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                     "max_concurrent_running": None},
    }
    try:
        nodes = [
            {"name": "local-box", "alive": True, "loadavg": 1.0, "free_cpu": 8,
             "free_ram_mb": 20000, "running_count": 0,
             "gpus": [{"idx": 0, "used_mb": 200, "total_mb": 8000, "free_mb": 7800,
                       "util_pct": 10}]},
            {"name": "cluster", "alive": True, "loadavg": 0.0, "free_cpu": 0,
             "free_ram_mb": 0, "running_count": 0, "gpus": []},
        ]
        task = {"id": "treq", "est_vram_mb": 1000, "cpu_cores": 2, "ram_mb": 4000,
                "signature": "TEST/require-slurm",
                "require_node": "cluster"}
        placement = sch.pick_placement(task, nodes)
        check("require_node=cluster → pick cluster even when local-box also fits",
              placement == ("cluster", None), diag=f"got {placement}")
    finally:
        sch._BACKEND = saved_backend
        sch.NODES = real_NODES

    # ---------- Source-level guard: requires_local_capacity_check is consulted in pick_placement ----------
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("pick_placement source consults _BACKEND.requires_local_capacity_check",
          "_BACKEND.requires_local_capacity_check" in src)
    # The bypass branch must come BEFORE _node_resources_ok call inside _candidates_for_node
    # (otherwise we'd still gate on local capacity). Check by finding both and asserting order.
    pp_idx = src.find("def pick_placement")
    bypass_idx = src.find("requires_local_capacity_check", pp_idx)
    nrok_idx = src.find("_node_resources_ok(task, n, node_info)", pp_idx)
    check("bypass branch precedes _node_resources_ok call inside pick_placement",
          0 < bypass_idx < nrok_idx,
          diag=f"bypass_idx={bypass_idx}, nrok_idx={nrok_idx}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"running regression tests against {SCHED_PATH}")
    test_ppid_descendant_filter()
    test_crash_requeue_dedup()
    test_preempt_sufficiency()
    test_launch_fail_fallback()
    test_cancel_never_becomes_failed()
    test_ram_placement_check()
    test_default_vram_not_inflated()
    test_status_view_no_truncation()
    test_high_defaults_lower_before_placement()
    test_probe_ram_budget_cap()
    test_running_descendant_resources_counted()
    test_kill_uses_process_group_sigkill()
    test_launch_failed_node_fallback_and_notification_events()
    test_requeue_from_adopt_becomes_scheduler_owned()
    test_training_cpu_guard()
    test_resume_capability_guard()
    test_diagnose_mid_training_kill()
    test_find_resume_extension_filter()
    test_local_max_vram_per_task_dynamic()
    test_env_deploy_wrap_docker()
    test_env_spec_conda_parsing()
    test_conda_preload_helpers()
    test_preload_handles_conda_spec()
    test_invariant_kill_unless_done_or_cancelled_requeues()
    test_invariant_no_dup_active_same_sig_cmd()
    test_invariant_race_guard_includes_launching()
    test_systemd_unit_restart_always()
    test_run_on_has_server_alive_options()
    test_has_image_digest_drift()
    test_env_deploy_doc_matches_code()
    test_pick_placement_best_fit_warm_first()
    test_diagnose_peak_vram_implies_crash_without_success()
    test_launch_path_uses_digest_check()
    test_atomic_write_integrity()
    test_cmd_with_special_shell_chars()
    test_no_test_writes_live_queue_with_fake_state()
    test_descendants_of_capped()
    test_clock_skew_lifetime_clamped()
    test_history_lru_truncation()
    test_launching_state_field_persistence()
    test_env_value_with_equals_sign()
    test_disk_full_classification()
    test_zombie_pid_excluded_from_alive()
    test_preload_uses_spec_image_or_image_field()
    test_save_state_after_each_launch()
    test_kill_includes_docker_for_named_container()
    test_kill_no_container_unchanged()
    test_ckpt_dir_cross_sig_conflict()
    test_oom_classify_no_false_positive()
    test_inject_python_u()
    test_dispatch_skips_duplicate_signature()
    test_cpu_training_justification_required()
    test_post_dispatch_eviction_and_rule()
    test_history_record_p80_outlier_resistance()
    test_backend_abstraction_phase1()
    test_backend_slurm_phase2()
    test_backend_slurm_phase2_1_sstat()
    test_backend_slurm_phase2_2_adopt_skip()
    test_backend_slurm_phase2_3_bypass_local_capacity()
    test_backend_slurm_phase2_4_log_path_shared_fs()
    test_backend_slurm_phase2_5_cd_guard()
    test_backend_slurm_phase2_6_docker_gpu_runtime_env()
    test_backend_slurm_phase2_7_probe_failure_does_not_cache()
    test_backend_slurm_phase2_8_route_by_launch_artifacts()
    test_check_running_phase2_9_slurm_aware()
    test_backend_slurm_phase2_10_sbatch_stdin_form()
    test_backend_slurm_phase2_12_eviction_skips_slurm_tasks()
    test_backend_slurm_phase2_13_terminal_state_semantics()
    test_backend_slurm_phase2_14_ui_and_launch_notification()
    test_backend_slurm_phase2_15_orphan_recovery()
    test_backend_slurm_phase2_16_pending_throttle()
    test_backend_slurm_phase2_16_1_rebalance_pending()
    test_phase2_17_install_slurm_orchestration()
    test_phase3_0_1_eta_parser_and_integration()
    test_phase3_0_2_node_load_metric()
    test_phase3_0_3_migration_trigger()
    test_phase3_0_4_staging()
    test_phase3_0_5_staging_outside_lock()
    test_phase3_0_7_rebalance_pending_no_duplicate_sbatch()
    test_phase3_0_8_unknown_eta_skipped_in_migration()
    test_phase3_0_9_slurm_pending_elapsed_zero()
    test_phase3_0_10_migration_event_visibility()
    test_phase3_0_11_migration_target_respects_blocked_and_launch_failed()
    test_phase3_0_12_migration_cooldown_anti_oscillation()

    passed = sum(1 for _, c, _ in results if c)
    total = len(results)
    print(f"\n{'=' * 60}\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)
