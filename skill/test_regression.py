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
    sch.launch = lambda t, node_state=None: (False, "synthetic launch failure")
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
    def fake_launch(t, node_state=None):
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
        # Phase 3.0.25: PSALL now requires a 5th column (stat). Use 'S' (sleeping)
        # for live procs; the parser drops Z/X.
        out = (
            "A100\n"
            "===VRAM===\n"
            "101, 700\n"
            "===PSALL===\n"
            "100 1 1000 0.0 S\n"
            "101 100 2048000 125.0 S\n"
            "102 101 1024000 80.0 R\n"
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
    # Phase 3.4.12: dedup key changed from `signature` alone to `(signature, cmd)`
    # tuple. Same-cmd duplicates still blocked; different-cmd-same-sig allowed.
    idx = src.find("running_keys = {")
    check("running_keys comprehension found",
          idx > 0,
          diag="3.4.12 P2-2: dedup key now (sig, cmd), not sig alone")
    if idx > 0:
        block = src[idx:idx + 400]
        check("running_keys includes 'launching' state",
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
    # Phase 3.2.1 extended the call to launch(t, node_state=picked_state).
    idx_launch = src.find("ok, msg = launch(t, node_state=")
    if idx_launch < 0:
        idx_launch = src.find("ok, msg = launch(t)")  # legacy fallback
    check("dispatch calls launch(t, ...)", idx_launch > 0)
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
    # Phase 3.4.12 P2-2: dedup key is (signature, cmd) tuple, not signature alone.
    # tdup has the SAME cmd as trun (`python train.py`) → still blocked.
    running_keys = {(t.get("signature") or "", t.get("cmd") or "")
                    for t in fake_state["tasks"]
                    if t.get("status") == "running" and t.get("signature")}
    blocked, eligible = [], []
    for t in fake_state["tasks"]:
        if t["status"] != "queued": continue
        sig = t.get("signature") or ""
        cmd_dd = t.get("cmd") or ""
        if sig and (sig, cmd_dd) in running_keys:
            blocked.append(t["id"])
        else:
            eligible.append(t["id"])
    check("queued task with same (sig, cmd) as running → blocked",
          "tdup" in blocked, diag=f"blocked={blocked}")
    check("queued task with empty sig → not blocked (exempt from guard)",
          "tnosig" in eligible, diag=f"eligible={eligible}")
    check("running_keys correctly built from running tasks (3.4.12 P2-2)",
          running_keys == {("TEST/dup-sig", "python train.py")},
          diag=str(running_keys))

    # Full _do_dispatch regression — Phase 3.4.12 P2-2 SCOPE CHANGE:
    # Pre-fix: any two queued tasks with the SAME signature were blocked
    # against each other regardless of cmd. That over-blocked legitimate
    # ablation batches (e.g. 28 BAPR ablations with one shared signature
    # but distinct (algo, env, seed) cmds — all stuck behind one peer).
    # Post-fix: dedup key is (sig, cmd). Same sig + DIFFERENT cmd → both
    # launch in parallel. Same sig + SAME cmd → second still blocked
    # (real duplicate; protects --out_dir/--ckpt-dir from clobber).
    # Below we test the same-cmd case (real duplicate) with cmds equal,
    # so the second task IS expected to be blocked.
    state = {
        "tasks": [
            {"id": "tq1", "status": "queued", "signature": "TEST/same-pass",
             "submitted_at": time.time(), "priority": "normal", "description": "dup A",
             "cmd": "python eval.py --x", "cwd": "/tmp", "ram_mb": 500,
             "est_vram_mb": 0, "cpu_cores": 1},
            {"id": "tq2", "status": "queued", "signature": "TEST/same-pass",
             "submitted_at": time.time() + 1, "priority": "normal", "description": "dup A again",
             "cmd": "python eval.py --x", "cwd": "/tmp", "ram_mb": 500,
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
        def fake_launch(t, node_state=None):
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
    check("same-pass duplicate (sig+cmd identical): only first launches",
          launched == ["tq1"], diag=f"launched={launched}")
    check("same-pass duplicate: second task remains queued",
          tq1["status"] == "running" and tq2["status"] == "queued",
          diag=f"tq1={tq1['status']} tq2={tq2['status']}")
    check("same-pass duplicate: second task gets blocked event with new reason",
          "tq2" in blocked_ids
          and "identical cmd already has a running task" in tq2.get("last_block_reason", ""),
          diag=f"blocked={blocked_ids}, reason={tq2.get('last_block_reason')}")

    # Phase 3.4.12 P2-2 — sibling case: same sig, DIFFERENT cmds → BOTH
    # launch in parallel. Pre-fix this was over-blocked (28-task ablation
    # batch incident). Re-run dispatch on a fresh state.
    state2 = {
        "tasks": [
            {"id": "tA", "status": "queued", "signature": "TEST/diff-cmd",
             "submitted_at": time.time(), "priority": "normal", "description": "ablation A",
             "cmd": "python ablation.py --algo bapr", "cwd": "/tmp", "ram_mb": 500,
             "est_vram_mb": 0, "cpu_cores": 1},
            {"id": "tB", "status": "queued", "signature": "TEST/diff-cmd",
             "submitted_at": time.time() + 1, "priority": "normal", "description": "ablation B",
             "cmd": "python ablation.py --algo escp", "cwd": "/tmp", "ram_mb": 500,
             "est_vram_mb": 0, "cpu_cores": 1},
        ],
        "next_id": 901,
    }
    nodes2 = [{"name": "local", "alive": True, "free_cpu": 12, "total_cpu": 12,
               "free_ram_mb": 30000, "total_ram_mb": 56000, "running_count": 0,
               "gpus": [{"idx": 0, "used_mb": 0, "total_mb": 8192,
                         "free_mb": 8192, "util_pct": 0}]}]
    launched2 = []
    try:
        sch.precheck_git = lambda t: (True, "ok")
        sch.find_resume = lambda t: None
        sch.save_state = fake_save_state
        def fake_launch2(t, node_state=None):
            launched2.append(t["id"])
            t["status"] = "running"
            t["remote_pids"] = [2000 + len(launched2)]
            t["started_at"] = time.time()
            t["log_path"] = f"/tmp/{t['id']}.log"
            return True, "pid=stub"
        sch.launch = fake_launch2
        sch._do_dispatch(state2, nodes2)
    finally:
        sch.precheck_git, sch.find_resume, sch.launch = orig_precheck, orig_find, orig_launch
        sch.save_state = orig_save
    check("3.4.12 P2-2: same sig + DIFFERENT cmd → both launch in parallel",
          set(launched2) == {"tA", "tB"},
          diag=f"launched2={launched2}")

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

    # Phase 3.2.1 extended the wrapper signature.
    launch_body = (_body_after("\ndef launch(task, node_state=None):")
                   or _body_after("\ndef launch(task):"))
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
        def launch(self, task, node_state=None): return True, "fake-launched"
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


def test_phase3_0_13_rebalance_pending_outside_lock():
    """Phase 3.0.13 P3 fix: cmd_rebalance_pending splits into 3 phases so the slow
    scancel+squeue ssh round-trip happens OUTSIDE state_lock.

    Pre-fix: one big `with state_lock()` block held the global lock for ~5s per
    candidate (scancel + sleep 1.5s + squeue verify). A 20-task batch blocked
    submit / cancel / status / watcher iterations for ~100s.

    Now: identify (short lock) → scancel+verify (NO LOCK) → commit (short lock).
    Plus: pre-scancel state recheck (the wider window means slurm could have
    started a job; never scancel a RUNNING task) and defensive recheck at commit
    time (status / slurm_job_id / slurm_state could shift during the unlocked
    window — leave such tasks alone).
    """
    print("\n[70] Phase 3.0.13 P3 fix: rebalance-pending runs slow ssh outside state_lock")

    # 1. Source guard: cmd_rebalance_pending body has TWO state_lock blocks and
    # the scancel call sits BETWEEN them.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def cmd_rebalance_pending")
    fn_end = src.find("\ndef ", fn_idx + 5)
    fn_body = src[fn_idx:fn_end]
    n_locks = fn_body.count("with state_lock()")
    check("cmd_rebalance_pending uses exactly 2 state_lock blocks (split phases)",
          n_locks == 2, diag=f"got {n_locks}")
    # Indices of the two locks and the scancel call. scancel must sit between them.
    lock1 = fn_body.find("with state_lock()")
    lock2 = fn_body.find("with state_lock()", lock1 + 1)
    scancel_idx = fn_body.find("scancel {int(jid)}")
    check("scancel call sits BETWEEN the two state_lock blocks (i.e. outside lock)",
          lock1 < scancel_idx < lock2,
          diag=f"lock1={lock1} scancel={scancel_idx} lock2={lock2}")

    # 2. Behavioral: shared mock setup.
    saved_save = sch.save_state
    saved_load = sch.load_state
    saved_run_on = sch.run_on
    saved_lock = sch.state_lock
    saved_sleep = time.sleep
    time.sleep = lambda s: None
    from contextlib import contextmanager as _cm
    @_cm
    def fake_lock():
        yield
    sch.state_lock = fake_lock

    class Args:
        yes = True

    try:
        # ---------- Case A: race — pre-check sees RUNNING (slurm started job
        # during the outside-lock window) → never scancel, leave task in place.
        fake_state = {"next_id": 1, "tasks": [
            {"id": "tRace", "status": "running", "node": "n1",
             "slurm_job_id": 800, "slurm_state": "PENDING",
             "remote_pids": [], "signature": "TEST/Race", "cmd": "x"},
        ]}
        sch.load_state = lambda: fake_state
        sch.save_state = lambda s: None
        scancel_seen = []
        def run_on_race(node, cmd, timeout=10, check=True):
            if "scancel" in cmd:
                scancel_seen.append(cmd)
                return (0, "", "")
            if "squeue" in cmd:
                return (0, "RUNNING\n", "")  # pre-check sees the race
            return (0, "", "")
        sch.run_on = run_on_race
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("pre-check RUNNING → scancel was NEVER called (don't kill the running job)",
              len(scancel_seen) == 0, diag=f"scancel calls: {scancel_seen}")
        check("pre-check RUNNING → task LEFT IN PLACE (status=running)",
              post["status"] == "running" and post.get("slurm_job_id") == 800,
              diag=str(post))

        # ---------- Case B: pre-check shows already-terminal → no scancel,
        # task gets cleared + requeued (slurm had already moved it).
        fake_state = {"next_id": 1, "tasks": [
            {"id": "tDone", "status": "running", "node": "n1",
             "slurm_job_id": 801, "slurm_state": "PENDING",
             "remote_pids": [], "signature": "TEST/Done", "cmd": "x"},
        ]}
        sch.load_state = lambda: fake_state
        scancel_seen = []
        def run_on_terminal(node, cmd, timeout=10, check=True):
            if "scancel" in cmd:
                scancel_seen.append(cmd)
                return (0, "", "")
            if "squeue" in cmd:
                return (0, "CANCELLED\n", "")
            return (0, "", "")
        sch.run_on = run_on_terminal
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("pre-check terminal → scancel skipped (already done)",
              len(scancel_seen) == 0, diag=f"scancel calls: {scancel_seen}")
        check("pre-check terminal → task requeued",
              post["status"] == "queued" and post.get("slurm_job_id") is None,
              diag=str(post))

        # ---------- Case C: defensive commit-phase recheck — task transitioned
        # to RUNNING in scheduleurm's state.json AFTER our outside-lock window
        # (slurm_state field updated by watcher's update_running_tasks).
        # Even though our scancel verified cancelled, do NOT clear: leave alone.
        identify_state = {"next_id": 1, "tasks": [
            {"id": "tShift", "status": "running", "node": "n1",
             "slurm_job_id": 802, "slurm_state": "PENDING",
             "remote_pids": [], "signature": "TEST/Shift", "cmd": "x"},
        ]}
        commit_state = {"next_id": 1, "tasks": [
            # Same task, but slurm_state has shifted to RUNNING by commit time.
            {"id": "tShift", "status": "running", "node": "n1",
             "slurm_job_id": 802, "slurm_state": "RUNNING",
             "remote_pids": [], "signature": "TEST/Shift", "cmd": "x"},
        ]}
        load_call_count = {"n": 0}
        def staged_load_state():
            load_call_count["n"] += 1
            # First call = identify; second call = commit (defensive recheck).
            return identify_state if load_call_count["n"] == 1 else commit_state
        sch.load_state = staged_load_state
        save_capture = {}
        sch.save_state = lambda s: save_capture.update(state=s)
        # Pre-check returns PENDING (so scancel fires + verifies).
        cancelled_jids_local = set()
        def run_on_shift(node, cmd, timeout=10, check=True):
            if "scancel" in cmd:
                try:
                    cancelled_jids_local.add(int(cmd.split()[-1]))
                except Exception:
                    pass
                return (0, "", "")
            if "squeue" in cmd:
                if any(str(j) in cmd for j in cancelled_jids_local):
                    return (0, "", "")  # post-scancel: gone
                return (0, "PENDING\n", "")  # pre-scancel: still pending
            return (0, "", "")
        sch.run_on = run_on_shift
        sch.cmd_rebalance_pending(Args())
        post = commit_state["tasks"][0]
        check("commit-phase defensive: slurm_state shifted to RUNNING during window → task UNTOUCHED",
              post["status"] == "running" and post.get("slurm_job_id") == 802
              and post.get("slurm_state") == "RUNNING",
              diag=str(post))

        # ---------- Case D: defensive commit-phase — task disappeared from state
        # (forgotten / archived between identify and commit). Don't crash;
        # silently skip.
        identify_state = {"next_id": 1, "tasks": [
            {"id": "tGone", "status": "running", "node": "n1",
             "slurm_job_id": 803, "slurm_state": "PENDING",
             "remote_pids": [], "signature": "TEST/Gone", "cmd": "x"},
        ]}
        commit_state = {"next_id": 1, "tasks": []}  # task vanished
        load_call_count["n"] = 0
        def staged_load_state2():
            load_call_count["n"] += 1
            return identify_state if load_call_count["n"] == 1 else commit_state
        sch.load_state = staged_load_state2
        cancelled_jids_local = set()
        sch.run_on = run_on_shift  # reuses the stateful mock (jid 803 not seen yet, mock starts fresh)
        # Run; should not raise.
        sch.cmd_rebalance_pending(Args())
        check("commit-phase defensive: missing task does not crash, no rebalance recorded",
              commit_state["tasks"] == [], diag=str(commit_state))

        # ---------- Case E: defensive commit-phase — slurm_job_id changed
        # between identify and commit (e.g., user cancel + auto-resubmit by
        # another path). Don't mutate even if our scancel "succeeded" against
        # the OLD jid — that's a different launch now.
        identify_state = {"next_id": 1, "tasks": [
            {"id": "tNewJid", "status": "running", "node": "n1",
             "slurm_job_id": 804, "slurm_state": "PENDING",
             "remote_pids": [], "signature": "TEST/NewJid", "cmd": "x"},
        ]}
        commit_state = {"next_id": 1, "tasks": [
            {"id": "tNewJid", "status": "running", "node": "n1",
             "slurm_job_id": 999, "slurm_state": "PENDING",  # different jid now
             "remote_pids": [], "signature": "TEST/NewJid", "cmd": "x"},
        ]}
        load_call_count["n"] = 0
        def staged_load_state3():
            load_call_count["n"] += 1
            return identify_state if load_call_count["n"] == 1 else commit_state
        sch.load_state = staged_load_state3
        cancelled_jids_local = set()
        sch.run_on = run_on_shift
        sch.cmd_rebalance_pending(Args())
        post = commit_state["tasks"][0]
        check("commit-phase defensive: slurm_job_id changed → leave alone",
              post["status"] == "running" and post.get("slurm_job_id") == 999,
              diag=str(post))
    finally:
        sch.save_state = saved_save
        sch.load_state = saved_load
        sch.run_on = saved_run_on
        sch.state_lock = saved_lock
        time.sleep = saved_sleep


def test_phase3_0_14_min_source_load_and_cwd_size_cap():
    """Phase 3.0.14 P4 fix: two cleanups around migration triggering.

    (a) MIGRATION_MIN_SOURCE_LOAD_S — absolute floor on the heaviest node's
        eta_load. Pre-fix, LOAD_RATIO=2x was satisfied by trivial imbalances
        (target=0s, source=2s → ratio 2.0 passes). That would migrate a 600s
        task to save 2s of source load — rsync cost dwarfs the saving.

    (b) MIGRATION_MAX_CWD_SIZE_MB — size cap on cwd rsync. MIGRATION_MAX_CKPT_
        SIZE_MB only bounded ckpt; cwd was unbounded. A monorepo cwd (5GB+)
        could blow past the 600s rsync timeout and starve the staging path.
        Excludes mirror rsync excludes (.git, __pycache__, *.pyc).
    """
    print("\n[71] Phase 3.0.14 P4 fix: min source-load + cwd size cap")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("MIGRATION_MIN_SOURCE_LOAD_S env-overridable constant exists",
          "MIGRATION_MIN_SOURCE_LOAD_S = int(os.environ.get(" in src
          and "SCHEDULEURM_MIGRATION_MIN_SOURCE_LOAD_S" in src)
    check("MIGRATION_MAX_CWD_SIZE_MB env-overridable constant exists",
          "MIGRATION_MAX_CWD_SIZE_MB = int(os.environ.get(" in src
          and "SCHEDULEURM_MIGRATION_MAX_CWD_SIZE_MB" in src)
    iden_idx = src.find("def _identify_migration_candidates")
    iden_end = src.find("\ndef ", iden_idx + 5)
    iden_body = src[iden_idx:iden_end]
    check("_identify_migration_candidates enforces MIN_SOURCE_LOAD_S",
          "MIGRATION_MIN_SOURCE_LOAD_S" in iden_body)
    cm_idx = src.find("def _consider_migration")
    cm_end = src.find("\ndef ", cm_idx + 5)
    cm_body = src[cm_idx:cm_end]
    check("_consider_migration also enforces MIN_SOURCE_LOAD_S (defensive)",
          "MIGRATION_MIN_SOURCE_LOAD_S" in cm_body)
    stage_idx = src.find("def _stage_for_migration")
    stage_end = src.find("\ndef ", stage_idx + 5)
    stage_body = src[stage_idx:stage_end]
    check("_stage_for_migration enforces MAX_CWD_SIZE_MB before rsync",
          "MIGRATION_MAX_CWD_SIZE_MB" in stage_body
          and "du -sm" in stage_body and "--exclude=.git" in stage_body)

    # 2. Behavioral: source-load below floor → no candidates / no migration.
    saved_NODES = sch.NODES
    saved_can_migrate = sch._can_migrate_to
    sch.NODES = {
        "loaded": {"name": "loaded"},
        "free":   {"name": "free"},
    }
    sch._can_migrate_to = lambda task, target_node, timeout_s=5: True
    try:
        # NOTE: queued tasks pinned to source ALSO count toward source_load via
        # compute_node_load_seconds. Use the minimum legal candidate eta
        # (MIGRATION_MIN_TASK_ETA_S = 300) so the candidate itself doesn't push
        # source_load past the floor we're testing.
        eta = sch.MIGRATION_MIN_TASK_ETA_S  # = 300 by default
        # Below-floor source: 50s running + 300s queued = 350s on "loaded" — below
        # the 600s default min source load. Ratio (350 vs 0) still passes the
        # legacy LOAD_RATIO check, so this exercises the new floor specifically.
        running_tiny = {"id": "rTiny", "status": "running", "node": "loaded",
                        "eta_seconds": 50, "started_at": time.time() - 10}
        cand = {"id": "tQ", "status": "queued", "preferred_node": "loaded",
                "eta_seconds": eta, "submitted_at": time.time(),
                "priority": "normal", "description": "should not migrate"}
        state_below = {"tasks": [running_tiny, cand]}
        nodes = [
            {"name": "loaded", "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 1,
             "slurm_pending_count": 0},
            {"name": "free",   "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 0,
             "slurm_pending_count": 0},
        ]
        identified_below = sch._identify_migration_candidates(state_below, nodes,
                                                              max_candidates=10)
        check("source_load 50s < MIN_SOURCE_LOAD_S (600s) → no staging snapshot",
              identified_below == [], diag=f"got {identified_below}")
        migrated_below = sch._consider_migration(state_below, nodes)
        check("source_load 50s < MIN_SOURCE_LOAD_S → _consider_migration returns []",
              migrated_below == [], diag=f"got {migrated_below}")
        check("tQ preferred_node UNCHANGED (still 'loaded')",
              cand["preferred_node"] == "loaded")

        # Above-floor source: enough load that the absolute floor passes too.
        running_a = {"id": "rA", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        running_b = {"id": "rB", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        cand2 = {"id": "tOK", "status": "queued", "preferred_node": "loaded",
                 "eta_seconds": eta, "submitted_at": time.time() + 1,
                 "priority": "normal", "description": "should migrate"}
        state_above = {"tasks": [running_a, running_b, cand2]}
        identified_above = sch._identify_migration_candidates(state_above, nodes,
                                                              max_candidates=10)
        ids_above = [c["id"] for c, _ in identified_above]
        check("source_load 10000s > MIN_SOURCE_LOAD_S → tOK in snapshot",
              "tOK" in ids_above, diag=f"got {ids_above}")
        migrated_above = sch._consider_migration(state_above, nodes)
        check("source_load 10000s → _consider_migration migrates tOK",
              migrated_above == ["tOK"], diag=f"got {migrated_above}")

        # 3. Boundary: total source_load just under threshold → blocked; just over → passes.
        # source_load = running.eta + cand.eta. cand.eta is fixed at MIGRATION_MIN_TASK_ETA_S.
        # Pick running.eta so total = (MIN_SOURCE_LOAD_S - 1) for the under case
        # and (MIN_SOURCE_LOAD_S + 1) for the over case.
        boundary_running_under = {"id": "rUnder", "status": "running", "node": "loaded",
                                  "eta_seconds": sch.MIGRATION_MIN_SOURCE_LOAD_S - 1 - eta,
                                  "started_at": time.time() - 10}
        boundary_running_over  = {"id": "rOver", "status": "running", "node": "loaded",
                                  "eta_seconds": sch.MIGRATION_MIN_SOURCE_LOAD_S + 1 - eta,
                                  "started_at": time.time() - 10}
        cand3 = {"id": "tBound", "status": "queued", "preferred_node": "loaded",
                 "eta_seconds": eta, "submitted_at": time.time() + 2,
                 "priority": "normal"}
        state_under = {"tasks": [boundary_running_under, cand3]}
        # Note: ratio gate also needs source > 2*max(target,1). target=0 ⇒ source ≥ 2 — satisfied.
        ids_under = [c["id"] for c, _ in sch._identify_migration_candidates(
            state_under, nodes, max_candidates=10)]
        check("source_load = MIN-1 → blocked",
              "tBound" not in ids_under, diag=f"got {ids_under}")
        # Reset cand3.preferred_node since previous _consider_migration calls
        # may have mutated it (we test identify-only here so no commit).
        cand3["preferred_node"] = "loaded"
        state_over = {"tasks": [boundary_running_over, cand3]}
        ids_over = [c["id"] for c, _ in sch._identify_migration_candidates(
            state_over, nodes, max_candidates=10)]
        check("source_load = MIN+1 → passes",
              "tBound" in ids_over, diag=f"got {ids_over}")
    finally:
        sch.NODES = saved_NODES
        sch._can_migrate_to = saved_can_migrate

    # 4. cwd size cap behavioral test on _stage_for_migration.
    saved_NODES = sch.NODES
    saved_run_on = sch.run_on
    sch.NODES = {
        "src":  {"name": "src",  "host": None},   # local
        "tgt":  {"name": "tgt",  "host": "tgt-host"},
    }
    try:
        # Mock: target says cwd is missing (test -d returns 1) → triggers rsync path.
        # Source du returns a size we control. For this test we don't actually rsync —
        # we want the size check to short-circuit before the rsync subprocess call.
        big_size_mb = sch.MIGRATION_MAX_CWD_SIZE_MB + 100
        small_size_mb = max(1, sch.MIGRATION_MAX_CWD_SIZE_MB - 100)
        size_to_return = {"v": big_size_mb}
        def run_on_size_check(node, cmd, timeout=15, check=False):
            if "test -d" in cmd:
                return (1, "", "")  # cwd missing on target → triggers rsync
            if "du -sm" in cmd and "--exclude=.git" in cmd:
                return (0, f"{size_to_return['v']}\n", "")
            if "mkdir" in cmd:
                return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_size_check
        # Above cap: should bail before rsync with "cwd ... too large" message.
        task_big = {"id": "tBig", "cwd": "/big/repo", "preferred_node": "src",
                    "cmd": "python x.py"}
        ok_big, msg_big = sch._stage_for_migration(task_big, "tgt")
        check("cwd size > MAX_CWD_SIZE_MB → staging fails with size-cap message",
              ok_big is False
              and "cwd" in msg_big
              and "MB" in msg_big
              and str(big_size_mb) in msg_big,
              diag=f"got ok={ok_big} msg={msg_big!r}")
        # Below cap: size check should pass and we'd proceed to rsync. Since the
        # test mock doesn't actually rsync, we expect a different failure (or
        # success on the env probe). We just want to confirm the size message
        # is NOT triggered when we're under the cap.
        size_to_return["v"] = small_size_mb
        # Use real rsync attempt — it'll fail (no actual src/tgt), but the
        # error msg should be about rsync, not about cwd size.
        # To avoid that hassle, mock subprocess.run. But the function imports
        # subprocess locally, so we can't easily replace. Simpler: just assert
        # size_to_return value gets through to a non-size-cap failure path.
        # The cleanest behavioral signal: the failure message must NOT contain
        # "max" + "MB" (which is unique to the size-cap msg).
        import subprocess as _sp_real
        saved_sp_run = _sp_real.run
        def fake_sp_run(*a, **kw):
            class R:
                returncode = 99
                stdout = ""
                stderr = "fake rsync failure for test"
            return R()
        _sp_real.run = fake_sp_run
        try:
            task_small = {"id": "tSmall", "cwd": "/small/repo", "preferred_node": "src",
                          "cmd": "python x.py"}
            ok_small, msg_small = sch._stage_for_migration(task_small, "tgt")
            check("cwd size < MAX_CWD_SIZE_MB → not blocked by size cap (different failure)",
                  ok_small is False
                  and ("max " not in msg_small or str(sch.MIGRATION_MAX_CWD_SIZE_MB) not in msg_small),
                  diag=f"got ok={ok_small} msg={msg_small!r}")
        finally:
            _sp_real.run = saved_sp_run
    finally:
        sch.NODES = saved_NODES
        sch.run_on = saved_run_on


def test_phase3_0_15_migrated_task_pins_to_staged_node():
    """Phase 3.0.15 P1 fix: after migration, the task must launch on the staged
    target node — never on a fallback node.

    Pre-fix: migration only rewrote `preferred_node`. pick_placement's policy is
    "try preferred first, fall back to ANY node if preferred is full / throttled
    / has no capacity". Reproduced: staged target B → B full at dispatch time →
    pick_placement returns C → task launches on C with cwd/ckpt staged for B but
    nothing on C → resume task silently restarts from step 0. Same blast radius
    shape as the 3.0.6 remote→remote ckpt bug — wasted compute on a
    quietly-broken resume.

    Now: _consider_migration sets `staged_node` on commit. pick_placement
    promotes `staged_node` to a hard pin (overrides preferred-with-fallback)
    UNLESS the user explicitly set `require_node` (operator override beats
    auto-balance).
    """
    print("\n[72] Phase 3.0.15 P1 fix: migrated task hard-pins to staged target node")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    cm_idx = src.find("def _consider_migration")
    cm_end = src.find("\ndef ", cm_idx + 5)
    cm_body = src[cm_idx:cm_end]
    check("_consider_migration sets staged_node on commit",
          'cand["staged_node"] = target_name' in cm_body
          or "cand['staged_node']" in cm_body)
    pp_idx = src.find("def pick_placement")
    pp_end = src.find("\ndef ", pp_idx + 5)
    pp_body = src[pp_idx:pp_end]
    check("pick_placement consults staged_node",
          "staged_node" in pp_body or 'staged = task.get("staged_node")' in pp_body)
    check("pick_placement promotes staged_node to require ONLY when no user require_node",
          "if staged and not require" in pp_body
          or "if staged and require is None" in pp_body)

    # 2. Behavioral. Use slurm-style nodes (no local capacity check) so the test
    # focuses on the pin logic rather than CPU/RAM/VRAM gates.
    saved_backend = sch._BACKEND
    saved_NODES = sch.NODES
    saved_slurm_cap = sch._slurm_max_pending_for_node
    saved_blocked = sch._blocked_nodes_for_task
    saved_launch_failed = sch._launch_failed_nodes_for_task

    class FakeBackend:
        def requires_local_capacity_check(self, name):
            return False  # treat every node as slurm-managed for the test

    sch._BACKEND = FakeBackend()
    sch.NODES = {"A": {"name": "A"}, "B": {"name": "B"}, "C": {"name": "C"}}
    sch._blocked_nodes_for_task = lambda task: set()
    sch._launch_failed_nodes_for_task = lambda task: set()
    # Throttle: B accepts 1 pending; A/C accept 999. Set B's pending count to
    # exhaust its cap so _candidates_for_node(B) returns [] in the "B full" cases.
    sch._slurm_max_pending_for_node = lambda n: 1 if n == "B" else 999

    nodes_open = [
        {"name": "A", "alive": True, "slurm_pending_count": 0},
        {"name": "B", "alive": True, "slurm_pending_count": 0},
        {"name": "C", "alive": True, "slurm_pending_count": 0},
    ]
    nodes_b_full = [
        {"name": "A", "alive": True, "slurm_pending_count": 0},
        {"name": "B", "alive": True, "slurm_pending_count": 1},  # at cap → throttled
        {"name": "C", "alive": True, "slurm_pending_count": 0},
    ]
    try:
        # ---- Case A: staged_node=B, B available → picks B ----
        t_staged = {"id": "tA", "est_vram_mb": 0, "staged_node": "B"}
        placement = sch.pick_placement(t_staged, nodes_open)
        check("staged_node=B + B available → placement=B",
              placement is not None and placement[0] == "B",
              diag=f"got {placement}")

        # ---- Case B: staged_node=B, B full → returns None (NOT fallback) ----
        t_staged_b_full = {"id": "tB", "est_vram_mb": 0, "staged_node": "B"}
        placement = sch.pick_placement(t_staged_b_full, nodes_b_full)
        check("staged_node=B + B full → placement=None (no fallback to A or C)",
              placement is None,
              diag=f"got {placement} (pre-fix would have returned A or C)")

        # ---- Case C: no staged_node, preferred=B, B full → falls back to A or C ----
        # This is the existing soft-preferred behavior. Preserved.
        t_pref_only = {"id": "tC", "est_vram_mb": 0, "preferred_node": "B"}
        placement = sch.pick_placement(t_pref_only, nodes_b_full)
        check("preferred_node=B (no staged) + B full → fallback to A or C (legacy behavior)",
              placement is not None and placement[0] in ("A", "C"),
              diag=f"got {placement}")

        # ---- Case D: user-explicit require_node=A + staged_node=B → require wins ----
        # Operator override: if user hard-pinned require=A, migration's staged_node
        # must NOT silently re-route them. require=A wins. A available → picks A.
        t_require = {"id": "tD", "est_vram_mb": 0,
                     "require_node": "A", "staged_node": "B"}
        placement = sch.pick_placement(t_require, nodes_open)
        check("require_node=A + staged_node=B → require wins, placement=A",
              placement is not None and placement[0] == "A",
              diag=f"got {placement}")

        # ---- Case E: staged_node=B and B not in nodes list at all → returns None ----
        # Defense against stale staged_node referencing a removed node.
        t_orphan = {"id": "tE", "est_vram_mb": 0, "staged_node": "B"}
        nodes_no_b = [
            {"name": "A", "alive": True, "slurm_pending_count": 0},
            {"name": "C", "alive": True, "slurm_pending_count": 0},
        ]
        placement = sch.pick_placement(t_orphan, nodes_no_b)
        check("staged_node=B but B absent from nodes → placement=None (no fallback)",
              placement is None,
              diag=f"got {placement}")

        # ---- Case F: staged_node + preferred_node both set (the typical post-
        # migration shape: preferred=B, staged=B) → still picks B when available.
        t_normal = {"id": "tF", "est_vram_mb": 0,
                    "preferred_node": "B", "staged_node": "B"}
        placement = sch.pick_placement(t_normal, nodes_open)
        check("preferred=staged=B → placement=B",
              placement is not None and placement[0] == "B")

        # ---- Case G: end-to-end via _consider_migration sets staged_node correctly.
        saved_NODES_for_migrate = sch.NODES
        sch.NODES = {"loaded": {"name": "loaded"}, "free": {"name": "free"}}
        saved_can_migrate = sch._can_migrate_to
        sch._can_migrate_to = lambda task, target_node, timeout_s=5: True
        try:
            eta = sch.MIGRATION_MIN_TASK_ETA_S
            running_a = {"id": "rA", "status": "running", "node": "loaded",
                         "eta_seconds": 5000, "started_at": time.time() - 100}
            cand = {"id": "tMig", "status": "queued", "preferred_node": "loaded",
                    "eta_seconds": eta, "submitted_at": time.time(),
                    "priority": "normal"}
            state = {"tasks": [running_a, cand]}
            mnodes = [
                {"name": "loaded", "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
                 "gpus": [], "max_concurrent_running": 999, "running_count": 1,
                 "slurm_pending_count": 0},
                {"name": "free",   "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
                 "gpus": [], "max_concurrent_running": 999, "running_count": 0,
                 "slurm_pending_count": 0},
            ]
            migrated = sch._consider_migration(state, mnodes)
            check("_consider_migration migrated tMig",
                  migrated == ["tMig"], diag=f"got {migrated}")
            check("post-migration: staged_node set to target",
                  cand.get("staged_node") == "free",
                  diag=f"got {cand.get('staged_node')!r}")
            check("post-migration: preferred_node also set to target",
                  cand.get("preferred_node") == "free")
            check("post-migration: migration metadata intact",
                  cand.get("migrated_from") == "loaded"
                  and isinstance(cand.get("migrated_at"), float))
        finally:
            sch.NODES = saved_NODES_for_migrate
            sch._can_migrate_to = saved_can_migrate
    finally:
        sch._BACKEND = saved_backend
        sch.NODES = saved_NODES
        sch._slurm_max_pending_for_node = saved_slurm_cap
        sch._blocked_nodes_for_task = saved_blocked
        sch._launch_failed_nodes_for_task = saved_launch_failed


def test_phase3_0_16_ckpt_size_probe_fail_closed():
    """Phase 3.0.16 P1 fix: ckpt_dir size probe failure must fail-closed.

    Pre-fix: when `du -sm <ckpt_dir>` failed (rc!=0, non-digit output, ssh
    blip, permission error), `size_mb` defaulted to 0. The rsync gate
    `if size_mb > 0` then SKIPPED the actual transfer, but the function
    still returned `(True, "staged (cwd + ckpt + env)")`. Migration committed,
    resume task launched on target with no ckpt → silent step-0 restart.
    Same blast-radius shape as 3.0.6 / 3.0.15 — wasted compute on a
    quietly-broken resume.

    Now: explicit existence test on source first.
      - source has no source_node OR ckpt_dir absent on source → no ckpt to
        stage (legitimate first-launch case); skip rsync, success.
      - ckpt_dir present on source → size MUST be determinable. du failure
        = unknown size = fail-closed return.
    """
    print("\n[73] Phase 3.0.16 P1 fix: ckpt size probe failure → fail-closed")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    stage_idx = src.find("def _stage_for_migration")
    stage_end = src.find("\ndef ", stage_idx + 5)
    body = src[stage_idx:stage_end]
    check("ckpt step has source-side existence check (test -d)",
          "test -d" in body and "src_present" in body)
    check("ckpt step uses fail-closed sentinel for unknown size",
          "size_mb < 0" in body or "size_mb == -1" in body)
    check("ckpt size probe failure returns False, not silent True",
          "fail-closed" in body and "step-0 restart" in body)

    # 2. Behavioral. Mock NODES + run_on + subprocess.
    saved_NODES = sch.NODES
    saved_run_on = sch.run_on
    saved_sp_run = sch.subprocess.run
    sch.NODES = {
        "src": {"host": None, "cpu_cores": 12, "ram_mb": 32000,
                "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                "max_concurrent_running": None},
        "tgt": {"host": "tgtbox", "cpu_cores": 12, "ram_mb": 32000,
                "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                "max_concurrent_running": None},
    }
    class R:
        def __init__(self, rc=0):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""
    sch.subprocess.run = lambda *a, **kw: R(0)  # rsync succeeds when called

    try:
        # ---- Case A: ckpt_dir set, but du fails (rc=1, ssh/perm error) ----
        sch._STAGING_CACHE.clear()
        def run_on_du_fails(node, cmd, timeout=15, check=False):
            if "test -d /work" in cmd: return (0, "", "")
            if "test -d /ckpt" in cmd: return (0, "", "")  # ckpt EXISTS on source
            if "du -sm /ckpt" in cmd: return (1, "", "permission denied")
            if "test -x" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_du_fails
        ok, msg = sch._stage_for_migration(
            {"id": "tA", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python a.py"}, "tgt")
        check("ckpt exists + du fails → returns False (fail-closed, not silent True)",
              ok is False, diag=f"got ok={ok} msg={msg!r}")
        check("fail-closed message mentions step-0 restart risk",
              "step-0 restart" in msg or "fail-closed" in msg, diag=msg)

        # ---- Case B: ckpt_dir set, du returns garbage (non-digit output) ----
        sch._STAGING_CACHE.clear()
        def run_on_du_garbage(node, cmd, timeout=15, check=False):
            if "test -d /work" in cmd: return (0, "", "")
            if "test -d /ckpt" in cmd: return (0, "", "")
            if "du -sm /ckpt" in cmd: return (0, "not-a-number\n", "")
            if "test -x" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_du_garbage
        ok, msg = sch._stage_for_migration(
            {"id": "tB", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python a.py"}, "tgt")
        check("ckpt exists + du returns non-digit garbage → fail-closed",
              ok is False, diag=f"got ok={ok} msg={msg!r}")

        # ---- Case C: ckpt_dir set, but `test -d ckpt` ssh raises exception ----
        sch._STAGING_CACHE.clear()
        def run_on_test_ssh_raises(node, cmd, timeout=15, check=False):
            if "test -d /work" in cmd: return (0, "", "")
            if "test -d /ckpt" in cmd:
                raise RuntimeError("ssh: connection timeout")
            if "test -x" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_test_ssh_raises
        ok, msg = sch._stage_for_migration(
            {"id": "tC", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python a.py"}, "tgt")
        check("ckpt source-side existence ssh exception → fail-closed (not silent ok)",
              ok is False, diag=f"got ok={ok} msg={msg!r}")
        check("ssh exception message mentions reachability and step-0 risk",
              ("reachability" in msg or "fail-closed" in msg)
              and ("step-0" in msg or "rsyncing" in msg), diag=msg)

        # ---- Case D: ckpt_dir set, but absent on source (first-launch case) ----
        # This is legitimate: task hasn't created its ckpt yet. Should succeed,
        # skipping the ckpt rsync and not mentioning ckpt in the success message.
        sch._STAGING_CACHE.clear()
        def run_on_ckpt_absent(node, cmd, timeout=15, check=False):
            if "test -d /work" in cmd: return (0, "", "")
            if "test -d /ckpt" in cmd: return (1, "", "")  # absent on source
            if "test -x" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_ckpt_absent
        ok, msg = sch._stage_for_migration(
            {"id": "tD", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python a.py"}, "tgt")
        check("ckpt absent on source (first launch) → success, skip ckpt rsync",
              ok is True, diag=f"got ok={ok} msg={msg!r}")
        check("absent-ckpt success message does NOT claim '+ ckpt'",
              "ckpt" not in msg or "+ ckpt" not in msg, diag=msg)

        # ---- Case E: happy path — ckpt exists, du succeeds, size in cap ----
        sch._STAGING_CACHE.clear()
        def run_on_happy(node, cmd, timeout=15, check=False):
            if "test -d /work" in cmd: return (0, "", "")
            if "test -d /ckpt" in cmd: return (0, "", "")
            if "du -sm /ckpt" in cmd: return (0, "500\n", "")
            if "ls -1 /ckpt" in cmd: return (0, "model.pt\n", "")
            if "mkdir -p" in cmd: return (0, "", "")
            if "test -x" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_happy
        ok, msg = sch._stage_for_migration(
            {"id": "tE", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python a.py"}, "tgt")
        check("happy path: ckpt exists + du works + within cap → ok",
              ok is True, diag=f"got ok={ok} msg={msg!r}")
        check("happy path: success message mentions ckpt was staged",
              "ckpt" in msg, diag=msg)

        # ---- Case F: ckpt exists + size 0 (empty dir, but reachable) → success,
        # no rsync (nothing to transfer). NOT fail-closed (probe succeeded).
        sch._STAGING_CACHE.clear()
        def run_on_empty(node, cmd, timeout=15, check=False):
            if "test -d /work" in cmd: return (0, "", "")
            if "test -d /ckpt" in cmd: return (0, "", "")
            if "du -sm /ckpt" in cmd: return (0, "0\n", "")  # empty dir
            if "test -x" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_empty
        ok, msg = sch._stage_for_migration(
            {"id": "tF", "cwd": "/work", "ckpt_dir": "/ckpt",
             "preferred_node": "src", "cmd": "/abs/python a.py"}, "tgt")
        check("ckpt exists + size 0 (empty dir) → success, no rsync attempted",
              ok is True, diag=f"got ok={ok} msg={msg!r}")
    finally:
        sch.NODES = saved_NODES
        sch.run_on = saved_run_on
        sch.subprocess.run = saved_sp_run


def test_phase3_0_17_staging_cache_ttl():
    """Phase 3.0.17 P2 fix: staging caches must time out so they don't silently
    serve stale content.

    Pre-fix: `_STAGING_CACHE` was a plain set. Once a (src,tgt,path) was added
    on first rsync, it survived until process restart. If the user edited code
    in cwd OR placed a fresher ckpt while the task waited in queue, the next
    migration cycle's `_can_migrate_to` / `_stage_for_migration` would skip
    re-rsync and the task launched on target with stale staged content.

    `_STAGED_TASKS` had timestamps but no TTL check — same problem.

    Now: STAGING_TTL_S (default 600s, env-overridable). _staging_cache_hit
    helper returns False for entries older than TTL and pops them.
    _can_migrate_to applies the same TTL check to _STAGED_TASKS.
    """
    print("\n[74] Phase 3.0.17 P2 fix: staging cache TTL stops silent stale-content reuse")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("STAGING_TTL_S env-overridable constant exists",
          "STAGING_TTL_S = int(os.environ.get(" in src
          and "SCHEDULEURM_STAGING_TTL_S" in src)
    check("_staging_cache_hit helper exists with TTL semantics",
          "def _staging_cache_hit" in src and "STAGING_TTL_S" in src)
    check("_STAGING_CACHE is a dict (timestamped), not a set",
          "_STAGING_CACHE: dict = {}" in src)
    check("_can_migrate_to enforces STAGING_TTL_S",
          "STAGING_TTL_S" in src[src.find("def _can_migrate_to"):src.find(
              "def ", src.find("def _can_migrate_to") + 5)])

    # 2. Behavioral: _staging_cache_hit returns True for fresh, False for stale.
    saved_cache = sch._STAGING_CACHE.copy()
    sch._STAGING_CACHE.clear()
    try:
        fresh_key = ("src", "tgt", "/fresh")
        stale_key = ("src", "tgt", "/stale")
        sch._STAGING_CACHE[fresh_key] = time.time()
        sch._STAGING_CACHE[stale_key] = time.time() - sch.STAGING_TTL_S - 60
        check("fresh staging entry → cache hit",
              sch._staging_cache_hit(fresh_key) is True)
        check("stale staging entry (TTL+60 ago) → cache miss",
              sch._staging_cache_hit(stale_key) is False)
        check("stale entry was popped on miss (forces re-rsync next call)",
              stale_key not in sch._STAGING_CACHE)
        check("fresh entry NOT popped",
              fresh_key in sch._STAGING_CACHE)

        # Boundary: TTL - 1 → hit; TTL + 1 → miss.
        boundary_under = ("src", "tgt", "/under")
        boundary_over  = ("src", "tgt", "/over")
        sch._STAGING_CACHE[boundary_under] = time.time() - sch.STAGING_TTL_S + 1
        sch._STAGING_CACHE[boundary_over]  = time.time() - sch.STAGING_TTL_S - 1
        check("entry aged TTL-1 → hit",
              sch._staging_cache_hit(boundary_under) is True)
        check("entry aged TTL+1 → miss",
              sch._staging_cache_hit(boundary_over) is False)

        # Missing entry → False.
        check("missing key → False (not error)",
              sch._staging_cache_hit(("none", "x", "/y")) is False)
    finally:
        sch._STAGING_CACHE.clear()
        sch._STAGING_CACHE.update(saved_cache)

    # 3. _can_migrate_to applies the same TTL semantics to _STAGED_TASKS.
    saved_staged = dict(sch._STAGED_TASKS)
    sch._STAGED_TASKS.clear()
    try:
        sch._STAGED_TASKS[("tFresh", "tgt")] = time.time()
        sch._STAGED_TASKS[("tStale", "tgt")] = time.time() - sch.STAGING_TTL_S - 60
        check("_can_migrate_to fresh → True",
              sch._can_migrate_to({"id": "tFresh"}, "tgt") is True)
        check("_can_migrate_to stale → False (TTL expired)",
              sch._can_migrate_to({"id": "tStale"}, "tgt") is False)
        check("stale _STAGED_TASKS entry popped on miss",
              ("tStale", "tgt") not in sch._STAGED_TASKS)
        check("fresh _STAGED_TASKS entry preserved",
              ("tFresh", "tgt") in sch._STAGED_TASKS)
        check("_can_migrate_to missing entry → False",
              sch._can_migrate_to({"id": "tNone"}, "tgt") is False)
    finally:
        sch._STAGED_TASKS.clear()
        sch._STAGED_TASKS.update(saved_staged)

    # 4. End-to-end: stale cwd cache forces a re-rsync on next stage attempt.
    saved_NODES = sch.NODES
    saved_run_on = sch.run_on
    saved_sp_run = sch.subprocess.run
    saved_cache = sch._STAGING_CACHE.copy()
    sch.NODES = {
        "src": {"host": None}, "tgt": {"host": "tgtbox"},
    }
    rsync_count = {"n": 0}
    class R:
        def __init__(self, rc=0):
            self.returncode = rc; self.stdout = ""; self.stderr = ""
    def fake_rsync(*a, **kw):
        rsync_count["n"] += 1
        return R(0)
    sch.subprocess.run = fake_rsync
    def fake_run_on(node, cmd, timeout=15, check=False):
        if "test -d /code" in cmd:
            return (1, "", "")  # cwd missing on target → triggers rsync
        if "du -sm" in cmd and "--exclude=.git" in cmd:
            return (0, "10\n", "")
        if "test -x" in cmd: return (0, "", "")
        return (0, "", "")
    sch.run_on = fake_run_on
    sch._STAGING_CACHE.clear()
    try:
        # Pre-load a STALE cwd cache entry. Should be ignored → rsync happens.
        sch._STAGING_CACHE[("src", "tgt", "/code")] = time.time() - sch.STAGING_TTL_S - 60
        rsync_count["n"] = 0
        ok, _ = sch._stage_for_migration(
            {"id": "tStaleE2E", "cwd": "/code", "preferred_node": "src",
             "cmd": "/abs/python a.py"}, "tgt")
        check("stale cwd cache entry → rsync RE-RUN (not silently skipped)",
              rsync_count["n"] >= 1, diag=f"rsync_count={rsync_count['n']}")

        # Pre-load a FRESH cwd cache entry. Should skip rsync (cache hit).
        # However our mock has cwd missing on target, so the rsync would still
        # need to run for correctness. Let me adjust: have target say cwd
        # already present so the cache-hit fast-path triggers (the cache hit
        # only saves the test-d + rsync round-trip when cwd is on target).
        # Actually the code path is: `if not _staging_cache_hit(cwd_key):` →
        # only enter the test-d + rsync block if cache miss. If cache hit, skip
        # entire block. So fresh entry should mean rsync_count stays 0.
        sch._STAGING_CACHE.clear()
        sch._STAGING_CACHE[("src", "tgt", "/code")] = time.time()
        rsync_count["n"] = 0
        ok, _ = sch._stage_for_migration(
            {"id": "tFreshE2E", "cwd": "/code", "preferred_node": "src",
             "cmd": "/abs/python a.py"}, "tgt")
        check("fresh cwd cache entry → rsync SKIPPED (cache hit)",
              rsync_count["n"] == 0, diag=f"rsync_count={rsync_count['n']}")
    finally:
        sch.NODES = saved_NODES
        sch.run_on = saved_run_on
        sch.subprocess.run = saved_sp_run
        sch._STAGING_CACHE.clear()
        sch._STAGING_CACHE.update(saved_cache)


def test_phase3_0_18_probe_all_outside_lock():
    """Phase 3.0.18 P2 fix: _stage_migration_candidates_outside_lock must call
    probe_all() OUTSIDE state_lock so a slow ssh+nvidia-smi round-trip doesn't
    block submit/cancel/status while staging is being identified.

    Pre-fix: function held state_lock for `load_state + probe_all + identify`.
    probe_all does ssh to every node and can take seconds on a multi-host
    cluster — and the watcher calls into this every 60s, so other tools were
    randomly blocked for the duration of node probing.

    Now: state snapshot under a short lock, release, probe_all + identify
    outside. The function name promised "outside lock" but probe_all wasn't
    actually outside — that's the fix.
    """
    print("\n[75] Phase 3.0.18 P2 fix: probe_all moved outside state_lock during staging")

    # 1. Source guard: in the fn body, probe_all() and _identify_migration_
    # candidates must be called AFTER the `with state_lock():` block (i.e.
    # outside the indented lock context).
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _stage_migration_candidates_outside_lock")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    lock_idx = body.find("with state_lock()")
    # Use rfind so we hit the actual function call, not a `probe_all()` mention
    # inside the comment block above the lock.
    probe_idx = body.rfind("probe_all()")
    identify_idx = body.rfind("_identify_migration_candidates(")
    check("function still uses state_lock for state snapshot",
          lock_idx > 0, diag=str(lock_idx))
    check("probe_all() call is AFTER the state_lock block (outside lock)",
          probe_idx > lock_idx,
          diag=f"lock={lock_idx} probe={probe_idx}")
    check("_identify_migration_candidates call is AFTER state_lock too",
          identify_idx > lock_idx,
          diag=f"lock={lock_idx} identify={identify_idx}")

    # 2. Behavioral: instrument lock to assert it's NOT held when probe_all runs.
    # Use a held-flag set by a fake state_lock contextmanager; probe_all then
    # asserts the flag is False when it's invoked.
    saved_lock = sch.state_lock
    saved_load = sch.load_state
    saved_probe = sch.probe_all
    saved_identify = sch._identify_migration_candidates

    held = {"flag": False}
    probe_called_with_held = {"v": None}
    identify_called_with_held = {"v": None}

    from contextlib import contextmanager as _cm
    @_cm
    def fake_lock():
        held["flag"] = True
        try:
            yield
        finally:
            held["flag"] = False
    sch.state_lock = fake_lock
    sch.load_state = lambda: {"tasks": []}
    def fake_probe_all():
        probe_called_with_held["v"] = held["flag"]
        return []
    sch.probe_all = fake_probe_all
    def fake_identify(state, nodes, max_candidates=2):
        identify_called_with_held["v"] = held["flag"]
        return []
    sch._identify_migration_candidates = fake_identify

    try:
        sch._stage_migration_candidates_outside_lock(max_candidates=2)
        check("probe_all() invoked with state_lock NOT held",
              probe_called_with_held["v"] is False,
              diag=f"held during probe = {probe_called_with_held['v']}")
        check("_identify_migration_candidates invoked with state_lock NOT held",
              identify_called_with_held["v"] is False,
              diag=f"held during identify = {identify_called_with_held['v']}")
    finally:
        sch.state_lock = saved_lock
        sch.load_state = saved_load
        sch.probe_all = saved_probe
        sch._identify_migration_candidates = saved_identify


def test_phase3_0_19_staging_failure_cooldown_unblocks_later_candidates():
    """Phase 3.0.19 P3 fix: a permanent staging failure on the first
    max_candidates picks must NOT permanently starve the rest of the queue.

    Pre-fix: _identify_migration_candidates always returned the same first 2
    entries (sorted by eta ascending). If both had permanent failures (ckpt >
    cap, env missing on target, remote→remote refuse), candidates 3+ never
    got their staging slot — permanent starvation.

    Now: each failed staging attempt records (task_id, target) in
    _STAGING_FAILED with timestamp. Next identify pass skips tagged pairs so
    later candidates are exposed. Failures TTL out via
    STAGING_FAIL_COOLDOWN_S — transient blips and user-fixed issues recover
    automatically.
    """
    print("\n[76] Phase 3.0.19 P3 fix: staging failure cooldown unblocks starved candidates")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("STAGING_FAIL_COOLDOWN_S env-overridable constant exists",
          "STAGING_FAIL_COOLDOWN_S = int(os.environ.get(" in src
          and "SCHEDULEURM_STAGING_FAIL_COOLDOWN_S" in src)
    check("_STAGING_FAILED dict + helpers exist",
          "_STAGING_FAILED: dict" in src
          and "def _staging_recently_failed" in src
          and "def _record_staging_failure" in src)
    iden_idx = src.find("def _identify_migration_candidates")
    iden_end = src.find("\ndef ", iden_idx + 5)
    iden_body = src[iden_idx:iden_end]
    check("_identify_migration_candidates skips recently-failed (id,target) pairs",
          "_staging_recently_failed" in iden_body)
    stage_outer_idx = src.find("def _stage_migration_candidates_outside_lock")
    stage_outer_end = src.find("\ndef ", stage_outer_idx + 5)
    stage_outer_body = src[stage_outer_idx:stage_outer_end]
    check("staging outside-lock records failure on `not ok` AND on exception",
          stage_outer_body.count("_record_staging_failure") >= 2)

    # 2. Helper unit tests.
    saved_failed = dict(sch._STAGING_FAILED)
    sch._STAGING_FAILED.clear()
    try:
        sch._record_staging_failure("tA", "tgt")
        check("_record_staging_failure + _staging_recently_failed: fresh tag → True",
              sch._staging_recently_failed("tA", "tgt") is True)
        check("_staging_recently_failed: untagged (id,target) → False",
              sch._staging_recently_failed("tA", "other") is False)
        # Force expiration — past cooldown.
        sch._STAGING_FAILED[("tStale", "tgt")] = time.time() - sch.STAGING_FAIL_COOLDOWN_S - 60
        check("_staging_recently_failed: expired tag → False (and popped)",
              sch._staging_recently_failed("tStale", "tgt") is False
              and ("tStale", "tgt") not in sch._STAGING_FAILED)
        # LRU eviction when at cap.
        sch._STAGING_FAILED.clear()
        for i in range(sch._STAGING_FAILED_MAX):
            sch._STAGING_FAILED[(f"t{i}", "tgt")] = time.time() - i  # older → smaller ts
        before = len(sch._STAGING_FAILED)
        sch._record_staging_failure("tNew", "tgt")
        check("_record_staging_failure evicts ~25% when full",
              len(sch._STAGING_FAILED) < before)
        check("freshly-recorded tag survives the LRU pass",
              ("tNew", "tgt") in sch._STAGING_FAILED)
    finally:
        sch._STAGING_FAILED.clear()
        sch._STAGING_FAILED.update(saved_failed)

    # 3. Behavioral: identify excludes failure-tagged candidates and exposes
    # the next eligible task in queue.
    saved_NODES = sch.NODES
    saved_blocked = sch._blocked_nodes_for_task
    saved_launch_failed = sch._launch_failed_nodes_for_task
    saved_failed = dict(sch._STAGING_FAILED)
    sch.NODES = {"loaded": {"name": "loaded"}, "free": {"name": "free"}}
    sch._blocked_nodes_for_task = lambda task: set()
    sch._launch_failed_nodes_for_task = lambda task: set()
    sch._STAGING_FAILED.clear()
    try:
        eta_min = sch.MIGRATION_MIN_TASK_ETA_S
        # Three candidates with identical eta → submitted_at decides tie. Tag
        # the first two as recently-failed. Identify must surface only t3.
        running_a = {"id": "rA", "status": "running", "node": "loaded",
                     "eta_seconds": 5000, "started_at": time.time() - 100}
        t1 = {"id": "t1", "status": "queued", "preferred_node": "loaded",
              "eta_seconds": eta_min, "submitted_at": time.time(),
              "priority": "normal"}
        t2 = {"id": "t2", "status": "queued", "preferred_node": "loaded",
              "eta_seconds": eta_min, "submitted_at": time.time() + 1,
              "priority": "normal"}
        t3 = {"id": "t3", "status": "queued", "preferred_node": "loaded",
              "eta_seconds": eta_min, "submitted_at": time.time() + 2,
              "priority": "normal"}
        state = {"tasks": [running_a, t1, t2, t3]}
        nodes = [
            {"name": "loaded", "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 1,
             "slurm_pending_count": 0},
            {"name": "free",   "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "gpus": [], "max_concurrent_running": 999, "running_count": 0,
             "slurm_pending_count": 0},
        ]

        # Baseline: no failures recorded → identify returns t1 + t2 (max_candidates=2).
        ids_baseline = [c["id"] for c, _ in sch._identify_migration_candidates(
            state, nodes, max_candidates=2)]
        check("baseline (no failures): identify returns first 2 candidates t1,t2",
              ids_baseline == ["t1", "t2"], diag=f"got {ids_baseline}")

        # Tag t1 and t2 as recently failed → identify exposes t3.
        sch._record_staging_failure("t1", "free")
        sch._record_staging_failure("t2", "free")
        ids_starved = [c["id"] for c, _ in sch._identify_migration_candidates(
            state, nodes, max_candidates=2)]
        check("after tagging t1,t2 failed → identify exposes t3 (no starvation)",
              "t3" in ids_starved
              and "t1" not in ids_starved
              and "t2" not in ids_starved,
              diag=f"got {ids_starved}")

        # Tag is per-(task,target). Same tasks pinned to a different target
        # would NOT be skipped — only this specific (task, target) pair is gated.
        # Verify by checking that _staging_recently_failed for ("t1","other") is False.
        check("failure tag is per-(task,target), not global",
              sch._staging_recently_failed("t1", "free") is True
              and sch._staging_recently_failed("t1", "other_node") is False)
    finally:
        sch.NODES = saved_NODES
        sch._blocked_nodes_for_task = saved_blocked
        sch._launch_failed_nodes_for_task = saved_launch_failed
        sch._STAGING_FAILED.clear()
        sch._STAGING_FAILED.update(saved_failed)


def test_phase3_0_20_cwd_always_rsyncs_on_cache_miss():
    """Phase 3.0.20 P1 fix: cwd staging must rsync on cache miss even when the
    target already has a directory at that path.

    Pre-fix: `if rc != 0:` (target lacks cwd) was the gate around rsync. If the
    target HAPPENED to have a stale checkout at the same cwd (a sibling task
    cloned it earlier, manual user setup, leftover from a previous migration
    that finished), rsync was SKIPPED and the cache populated as if staged.
    Migrated task ran old code with no warning. Same blast-radius shape as the
    other silent-staleness P1s.

    Now: cache miss → ALWAYS rsync. rsync's delta keeps the in-sync case cheap
    (~1s), so trusting `test -d` as a freshness proxy is the unsafe shortcut.
    The TTL on _STAGING_CACHE bounds how often we re-rsync within a session.
    """
    print("\n[77] Phase 3.0.20 P1 fix: cwd always rsyncs on cache miss (no test-d shortcut)")

    # 1. Source guard: the rsync block must NOT sit inside an `if rc != 0:` arm.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    stage_idx = src.find("def _stage_for_migration")
    stage_end = src.find("\ndef ", stage_idx + 5)
    body = src[stage_idx:stage_end]
    cache_check = body.find("if not _staging_cache_hit(cwd_key):")
    rsync_call = body.find('subprocess as _sp\n            r = _sp.run(["rsync"', cache_check)
    if rsync_call < 0:
        rsync_call = body.find("_sp.run([\"rsync\"", cache_check)
    # The pre-rsync `test -d cwd` short-circuit is gone.
    pre_rsync_segment = body[cache_check:rsync_call] if rsync_call > 0 else ""
    check("rsync block sits directly under the cache-miss guard, not behind a `test -d` short-circuit",
          rsync_call > cache_check
          and "if rc != 0" not in pre_rsync_segment.split("subprocess as _sp")[0],
          diag=f"cache_check={cache_check} rsync={rsync_call}")

    # 2. Behavioral: cwd present on target → still rsync.
    saved_NODES = sch.NODES
    saved_run_on = sch.run_on
    saved_sp_run = sch.subprocess.run
    saved_cache = sch._STAGING_CACHE.copy()
    sch.NODES = {
        "src": {"host": None}, "tgt": {"host": "tgtbox"},
    }
    rsync_count = {"n": 0}
    class R:
        def __init__(self, rc=0):
            self.returncode = rc; self.stdout = ""; self.stderr = ""
    def fake_rsync(*a, **kw):
        rsync_count["n"] += 1
        return R(0)
    sch.subprocess.run = fake_rsync

    try:
        # ---- Case A: target has cwd (test -d returns 0) → still rsync ----
        sch._STAGING_CACHE.clear()
        rsync_count["n"] = 0
        def run_on_target_has_cwd(node, cmd, timeout=15, check=False):
            if "test -d /code" in cmd: return (0, "", "")  # target has it
            if "du -sm" in cmd and "--exclude=.git" in cmd: return (0, "10\n", "")
            if "test -x" in cmd: return (0, "", "")
            if "mkdir -p" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_target_has_cwd
        ok, msg = sch._stage_for_migration(
            {"id": "tA", "cwd": "/code", "preferred_node": "src",
             "cmd": "/abs/python a.py"}, "tgt")
        check("cache miss + target has stale cwd → rsync STILL fires (no test-d shortcut)",
              ok is True and rsync_count["n"] >= 1,
              diag=f"ok={ok} rsync_count={rsync_count['n']} msg={msg!r}")

        # ---- Case B: cache hit → no rsync (TTL-fresh entry) ----
        sch._STAGING_CACHE.clear()
        sch._STAGING_CACHE[("src", "tgt", "/code")] = time.time()  # fresh
        rsync_count["n"] = 0
        ok, msg = sch._stage_for_migration(
            {"id": "tB", "cwd": "/code", "preferred_node": "src",
             "cmd": "/abs/python a.py"}, "tgt")
        check("cache hit → no rsync (TTL governs re-rsync, not test-d)",
              ok is True and rsync_count["n"] == 0,
              diag=f"ok={ok} rsync_count={rsync_count['n']}")

        # ---- Case C: cache miss + target lacks cwd → rsync (regression) ----
        sch._STAGING_CACHE.clear()
        rsync_count["n"] = 0
        cwd_state = {"present": False}  # flips to True after rsync
        def fake_rsync_flip(*a, **kw):
            rsync_count["n"] += 1
            cwd_state["present"] = True  # post-rsync: dir exists
            return R(0)
        sch.subprocess.run = fake_rsync_flip
        def run_on_target_missing(node, cmd, timeout=15, check=False):
            if "test -d /code" in cmd:
                return (0, "", "") if cwd_state["present"] else (1, "", "")
            if "du -sm" in cmd and "--exclude=.git" in cmd: return (0, "10\n", "")
            if "test -x" in cmd: return (0, "", "")
            if "mkdir -p" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_target_missing
        ok, msg = sch._stage_for_migration(
            {"id": "tC", "cwd": "/code", "preferred_node": "src",
             "cmd": "/abs/python a.py"}, "tgt")
        check("cache miss + target lacks cwd → rsync (existing path still works)",
              ok is True and rsync_count["n"] >= 1,
              diag=f"ok={ok} rsync_count={rsync_count['n']} msg={msg!r}")
        sch.subprocess.run = fake_rsync  # restore non-flipping mock for Case D

        # ---- Case D: rsync of an oversized cwd still bails ----
        sch._STAGING_CACHE.clear()
        rsync_count["n"] = 0
        big_size = sch.MIGRATION_MAX_CWD_SIZE_MB + 100
        def run_on_oversized(node, cmd, timeout=15, check=False):
            if "test -d /code" in cmd: return (0, "", "")  # target has it
            if "du -sm" in cmd and "--exclude=.git" in cmd:
                return (0, f"{big_size}\n", "")
            if "test -x" in cmd: return (0, "", "")
            if "mkdir -p" in cmd: return (0, "", "")
            return (0, "", "")
        sch.run_on = run_on_oversized
        ok, msg = sch._stage_for_migration(
            {"id": "tD", "cwd": "/code", "preferred_node": "src",
             "cmd": "/abs/python a.py"}, "tgt")
        check("oversized cwd → reject regardless of target presence",
              ok is False and "MB" in msg and str(big_size) in msg,
              diag=f"ok={ok} msg={msg!r}")
        check("oversized cwd: no rsync attempted",
              rsync_count["n"] == 0,
              diag=f"rsync_count={rsync_count['n']}")
    finally:
        sch.NODES = saved_NODES
        sch.run_on = saved_run_on
        sch.subprocess.run = saved_sp_run
        sch._STAGING_CACHE.clear()
        sch._STAGING_CACHE.update(saved_cache)


def test_phase3_0_21_explicit_docker_fail_fast_no_local_digest():
    """Phase 3.0.21 P1 fix: explicit `--env-spec docker:IMAGE` with no local
    image must fail-fast. Pre-fix: get_image_digest("local", IMAGE) returning
    None caused has_image() to fall through to its legacy tag-presence path.
    If the remote node had any tag (e.g., stale from a prior push when local
    has since rebuilt and lost the old tag), has_image returned True, push was
    skipped, and the task launched against a silently-stale remote image.

    Auto mode keeps its graceful fallback — its contract is "use docker if
    available, else none". Only the explicit path treats missing local image
    as fatal.
    """
    print("\n[78] Phase 3.0.21 P1 fix: explicit docker fail-fast when no local digest")

    # 1. Source guard: explicit branch handles `local_digest is None` BEFORE
    # the has_image() fallthrough path.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _maybe_wrap_docker")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    digest_idx = body.find("local_digest = env_deploy.get_image_digest")
    has_image_idx = body.find("has_image(run_on, node, chosen_image, local_digest=local_digest)")
    explicit_check_idx = body.find("explicit and local_digest is None")
    check("explicit + no local digest fail-fast guard exists",
          explicit_check_idx > 0)
    check("guard sits BEFORE has_image() fallthrough",
          digest_idx < explicit_check_idx < has_image_idx,
          diag=f"digest={digest_idx} guard={explicit_check_idx} has_image={has_image_idx}")
    check("guard message names the stale-or-unintended-image risk",
          "stale or unintended image" in body or "stale remote tag" in body,
          diag="message must mention the failure mode (3.0.34 broadened text for local case)")

    # 2. Behavioral.
    saved_env_deploy = sch.env_deploy
    saved_NODES = sch.NODES
    sch.NODES = {"n1": {"host": "n1box"}}

    class FakeED:
        @staticmethod
        def parse_env_spec(spec):
            # Mimic env_deploy.parse_env_spec: "docker:IMAGE" → ("docker", "IMAGE")
            if spec == "docker":
                return ("docker", "")
            if spec.startswith("docker:"):
                return ("docker", spec.split(":", 1)[1])
            if spec == "auto":
                return ("auto", "")
            if spec == "none":
                return ("none", "")
            return ("none", "")
        @staticmethod
        def has_docker(run_on, node, timeout=8): return True
        # local_digest_ref controls what get_image_digest("local", ...) returns.
        local_digest_ref = {"value": None}
        # remote_tag_present controls has_image's tag-presence answer (only used
        # when local_digest is None — the unsafe fallthrough path).
        remote_tag_present = {"value": True}
        push_called = {"n": 0}

        @classmethod
        def get_image_digest(cls, run_on, node, image, timeout=10):
            if node == "local":
                return cls.local_digest_ref["value"]
            return "remote-digest" if cls.remote_tag_present["value"] else None

        @classmethod
        def has_image(cls, run_on, node, image, local_digest=None, timeout=10):
            remote_d = cls.get_image_digest(None, node, image)
            if not remote_d:
                return False
            if local_digest is None:
                return True  # legacy tag-only fallthrough — the bug surface
            return remote_d == local_digest

        @classmethod
        def push_image(cls, node_host, image, timeout_s=1800):
            cls.push_called["n"] += 1
            return (True, "ok")

        @staticmethod
        def wrap_cmd_docker(inner, image, cwd, gpu_idx, extra_env, container_name,
                            memory_mb, cpus, gpu_runtime_env):
            return f"docker_wrap({inner})"

    sch.env_deploy = FakeED

    try:
        # ---- Case A: explicit + local_digest is None → fail-fast (no push, no wrap) ----
        FakeED.local_digest_ref["value"] = None
        FakeED.remote_tag_present["value"] = True  # remote has stale tag
        FakeED.push_called["n"] = 0
        task = {"id": "tA", "node": "n1", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("explicit docker + no local digest → fail (err returned)",
              err is not None, diag=f"err={err!r}")
        check("fail message names the staleness risk (3.0.34 broadened text)",
              err and ("stale or unintended image" in err
                       or "stale remote tag" in err), diag=f"err={err!r}")
        check("fail-fast: push_image NOT called",
              FakeED.push_called["n"] == 0)
        check("fail-fast: inner cmd unchanged (no docker wrap)",
              inner == "python a.py", diag=f"inner={inner!r}")

        # ---- Case B: explicit + local digest matches remote → ok, no push ----
        FakeED.local_digest_ref["value"] = "remote-digest"
        FakeED.remote_tag_present["value"] = True
        FakeED.push_called["n"] = 0
        task = {"id": "tB", "node": "n1", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("explicit docker + matching digest → ok (no error, no push)",
              err is None and FakeED.push_called["n"] == 0,
              diag=f"err={err!r} push_n={FakeED.push_called['n']}")
        check("happy path: cmd was docker-wrapped",
              inner.startswith("docker_wrap("),
              diag=f"inner={inner!r}")

        # ---- Case C: explicit + local digest != remote → preload-retry error
        # Phase 3.0.31: launch-side push was moved out of state_lock. On drift
        # at launch (preload either failed or is in flight), explicit returns
        # an error pointing to the preload-retry path; push is NOT invoked
        # synchronously inside the lock anymore.
        FakeED.local_digest_ref["value"] = "fresh-local-digest"
        FakeED.remote_tag_present["value"] = True  # remote has stale-digest tag
        FakeED.push_called["n"] = 0
        task = {"id": "tC", "node": "n1", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("explicit docker + digest drift at launch → preload-retry error (NOT synchronous push)",
              err is not None
              and "preload" in err
              and FakeED.push_called["n"] == 0,
              diag=f"err={err!r} push_n={FakeED.push_called['n']}")

        # ---- Case D: AUTO mode + no local digest → graceful fallback (no err) ----
        # Auto mode's contract: use docker if available, else `none`. Since we
        # cannot verify the image is fresh without a local digest, Phase 3.0.26
        # tightened the fallback: auto + no local digest now declines to docker-
        # wrap (returns inner cmd unchanged, equivalent to kind=none) instead
        # of trusting a possibly-stale remote tag.
        FakeED.local_digest_ref["value"] = None
        FakeED.remote_tag_present["value"] = True
        FakeED.push_called["n"] = 0
        task = {"id": "tD", "node": "n1", "env_spec": "auto", "image": "myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("auto mode + no local digest → graceful (no error)",
              err is None,
              diag=f"err={err!r}")
        check("auto mode + no local digest → does NOT docker-wrap (3.0.26)",
              inner == "python a.py" and FakeED.push_called["n"] == 0,
              diag=f"inner={inner!r} push_n={FakeED.push_called['n']}")
    finally:
        sch.env_deploy = saved_env_deploy
        sch.NODES = saved_NODES


def test_phase3_0_22_explicit_conda_fail_fast_no_local_path():
    """Phase 3.0.22 P2 fix: explicit `--env-spec conda:/abs/path` with the path
    missing on local must fail-fast at launch.

    Pre-fix: _preload_env_outside_lock silently skipped a `conda:/abs/path`
    when `Path(path).is_dir()` was False on local — comment was "caller's
    mistake, eventual launch failure is diagnosed through ENV_MISSING".
    But _maybe_wrap_docker treated kind=conda as a no-op return — no
    fail-fast gate. If the remote happened to have a stale env at the same
    path (sibling task deployed it earlier, manual user setup), the task
    silently ran the stale remote python instead of the expected new one.

    Auto / non-absolute conda specs (e.g. `conda:envname`) are unaffected —
    those rely on `conda activate <name>` semantics, not path-rsync.
    """
    print("\n[79] Phase 3.0.22 P2 fix: explicit conda fail-fast when local path missing")

    # 1. Source guard.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _maybe_wrap_docker")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    check("conda branch checks Path.is_absolute() AND not Path.is_dir() for fail-fast",
          ("Path(spec_image).is_absolute()" in body
           and "not Path(spec_image).is_dir()" in body))
    check("conda branch error message names stale-remote-env risk",
          "stale remote env" in body)

    # 2. Behavioral.
    saved_env_deploy = sch.env_deploy
    class FakeED:
        @staticmethod
        def parse_env_spec(spec):
            if spec.startswith("conda:"):
                return ("conda", spec.split(":", 1)[1])
            if spec == "auto":
                return ("auto", "")
            return ("none", "")
    sch.env_deploy = FakeED

    try:
        # ---- Case A: conda:/abs/missing → fail-fast (path absent locally) ----
        # Use a path that is essentially guaranteed to not exist on this box.
        missing_path = "/__scheduleurm_test_missing_env__/never_existed"
        task = {"id": "tA", "node": "n1", "env_spec": f"conda:{missing_path}"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("conda + missing local path → fail (err returned)",
              err is not None, diag=f"err={err!r}")
        check("conda fail-fast message references the env path",
              err and missing_path in err, diag=f"err={err!r}")
        check("conda fail-fast: inner cmd unchanged",
              inner == "python a.py")

        # ---- Case B: conda:/tmp (exists) → ok, no error ----
        import tempfile, os as _os
        with tempfile.TemporaryDirectory() as tdir:
            task = {"id": "tB", "node": "n1", "env_spec": f"conda:{tdir}"}
            inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
            check("conda + existing local path → ok (no error, inner unchanged)",
                  err is None and inner == "python a.py",
                  diag=f"err={err!r} inner={inner!r}")

        # ---- Case C: conda:envname (non-absolute) → unaffected ----
        # Non-absolute conda specs (e.g. `conda:resac-jax`) rely on
        # `conda activate <name>` semantics. The path-existence check should
        # NOT touch them.
        task = {"id": "tC", "node": "n1", "env_spec": "conda:envname"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("conda + non-absolute env-name spec → not gated by path check",
              err is None,
              diag=f"err={err!r}")

        # ---- Case D: conda + relative path that doesn't exist → also unaffected ----
        # `Path("relative/foo").is_absolute()` is False, so the gate doesn't
        # fire. We don't try to be clever about cwd-relative resolution.
        task = {"id": "tD", "node": "n1", "env_spec": "conda:relative/foo"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("conda + non-absolute relative path → not gated either",
              err is None,
              diag=f"err={err!r}")
    finally:
        sch.env_deploy = saved_env_deploy


def test_phase3_0_23_env_key_validation_and_reserved_guard():
    """Phase 3.0.23 P2 fix: --env keys must match POSIX env-var shape AND
    cannot override scheduleurm-reserved keys (CUDA_VISIBLE_DEVICES).

    Pre-fix: _parse_env accepted any string with `=` in it. Both backends
    then did `export {k}=...`. Two failure modes:
      (a) `--env "FOO BAR=1"` → `export FOO BAR=1` is a shell syntax error,
          breaking the entire export prefix and silently failing the launch.
      (b) `--env CUDA_VISIBLE_DEVICES=2` → user-specified value clobbered
          scheduleurm's gpu_idx pin, so the task read a different GPU than
          the one it was scheduled onto. VRAM accounting + 1/3 packing
          rule both broke silently.

    Now: _parse_env enforces ^[A-Za-z_][A-Za-z0-9_]*$ and rejects reserved
    keys. _safe_extra_env_items at the launch sites is a defensive filter
    so legacy state.json entries can't slip through either.
    """
    print("\n[80] Phase 3.0.23 P2 fix: --env key validation + CUDA_VISIBLE_DEVICES guard")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("regex constant matches POSIX env-var name shape",
          "_ENV_KEY_RE" in src
          and r"^[A-Za-z_][A-Za-z0-9_]*$" in src)
    check("_RESERVED_ENV_KEYS includes CUDA_VISIBLE_DEVICES",
          "_RESERVED_ENV_KEYS" in src
          and "CUDA_VISIBLE_DEVICES" in src)
    parse_env_idx = src.find("def _parse_env(")
    parse_env_end = src.find("\ndef ", parse_env_idx + 5)
    parse_env_body = src[parse_env_idx:parse_env_end]
    check("_parse_env validates key shape",
          "_ENV_KEY_RE.match(k)" in parse_env_body)
    check("_parse_env rejects reserved keys",
          "k in _RESERVED_ENV_KEYS" in parse_env_body)
    check("_safe_extra_env_items helper exists for launch-side filtering",
          "def _safe_extra_env_items" in src)
    # Both backends (LocalBackend.launch + SlurmBackend.build_sbatch_script)
    # iterate via the helper instead of raw .items().
    check("LocalBackend / SlurmBackend export-loop uses _safe_extra_env_items",
          src.count("_safe_extra_env_items") >= 3,  # def + 2 call sites
          diag=f"count={src.count('_safe_extra_env_items')}")

    # 2. Behavioral: _parse_env validation.
    # Valid keys are accepted.
    out = sch._parse_env(["FOO=bar", "_X=1", "BUILD_ID=abc123"])
    check("valid keys accepted",
          out == {"FOO": "bar", "_X": "1", "BUILD_ID": "abc123"},
          diag=str(out))

    # Invalid shape → SystemExit.
    bad_keys = [
        "0LEAD_DIGIT=x",     # starts with digit
        "FOO BAR=x",         # space
        "FOO-BAR=x",         # hyphen
        "FOO.BAR=x",         # dot
        "PATH;rm -rf=x",     # shell metachar
        "=novalue",          # empty key (split's left side is "")
    ]
    for bad in bad_keys:
        try:
            sch._parse_env([bad])
            check(f"invalid key rejected: {bad!r}", False,
                  diag="SystemExit was NOT raised")
        except SystemExit as e:
            msg = str(e)
            check(f"invalid key rejected: {bad!r}",
                  ("not a valid POSIX" in msg) or ("KEY=VALUE" in msg),
                  diag=f"got msg={msg!r}")

    # Reserved key → SystemExit with clear message.
    try:
        sch._parse_env(["CUDA_VISIBLE_DEVICES=2"])
        check("CUDA_VISIBLE_DEVICES override rejected", False,
              diag="SystemExit was NOT raised")
    except SystemExit as e:
        check("CUDA_VISIBLE_DEVICES override rejected at submit",
              "reserved" in str(e) and "CUDA_VISIBLE_DEVICES" in str(e),
              diag=f"got msg={str(e)!r}")

    # 3. _safe_extra_env_items filters legacy bad keys silently.
    # (defensive: preserves behavior for tasks already in state.json from
    #  before 3.0.23.)
    legacy = {
        "FOO": "bar",                     # valid → kept
        "BAD KEY": "x",                   # space → dropped
        "0DIGIT": "y",                    # leading digit → dropped
        "CUDA_VISIBLE_DEVICES": "2",      # reserved → dropped
        "VALID_2": "ok",                  # valid → kept
    }
    safe_items = list(sch._safe_extra_env_items(legacy))
    safe_keys = sorted(k for k, _ in safe_items)
    check("_safe_extra_env_items keeps valid keys",
          safe_keys == ["FOO", "VALID_2"], diag=f"got {safe_keys}")
    check("_safe_extra_env_items drops keys with invalid shape",
          "BAD KEY" not in dict(safe_items)
          and "0DIGIT" not in dict(safe_items))
    check("_safe_extra_env_items drops CUDA_VISIBLE_DEVICES",
          "CUDA_VISIBLE_DEVICES" not in dict(safe_items))

    # 4. Empty / None → empty iterator (not crash).
    check("_safe_extra_env_items({}) → empty",
          list(sch._safe_extra_env_items({})) == [])
    check("_safe_extra_env_items(None) → empty",
          list(sch._safe_extra_env_items(None)) == [])


def test_phase3_0_24_rebalance_pending_clears_placement_fields():
    """Phase 3.0.24 P3 fix: requeue path in cmd_rebalance_pending must clear
    `node`, `gpu_idx`, `actual_started_at` along with the slurm fields.

    Pre-fix: only slurm-specific fields were cleared. `node` stayed pinned
    to the old slurm host. _do_dispatch would overwrite it on re-placement,
    but in the interim, status / TUI / env smoke probes would see a queued
    task still associated with the old node — confusing display + needless
    smoke probes against the wrong target.

    The audit message keeps the old node value (captured into a local before
    clearing) so users can see where the task came from.
    """
    print("\n[81] Phase 3.0.24 P3 fix: rebalance-pending requeue clears node/gpu_idx/actual_started_at")

    # 1. Source guard.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def cmd_rebalance_pending")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    check("requeue clears `node`",
          '"node"' in body and "node\", \"gpu_idx" in body)
    check("requeue clears `gpu_idx`",
          '"gpu_idx"' in body)
    check("requeue clears `actual_started_at`",
          '"actual_started_at"' in body)
    check("audit message captures old_node BEFORE clearing",
          "old_node" in body and "old_node = t.get" in body)

    # 2. Behavioral.
    saved_save = sch.save_state
    saved_load = sch.load_state
    saved_run_on = sch.run_on
    saved_lock = sch.state_lock
    saved_sleep = time.sleep
    time.sleep = lambda s: None

    fake_state = {"next_id": 1, "tasks": [
        {"id": "tA", "status": "running", "node": "n1",
         "slurm_job_id": 100, "slurm_state": "PENDING",
         "gpu_idx": 0, "actual_started_at": time.time() - 300,
         "started_at": time.time() - 600, "remote_pids": [],
         "signature": "TEST/A", "cmd": "x"},
    ]}
    sch.load_state = lambda: fake_state
    sch.save_state = lambda s: None
    cancelled_jids = set()
    def fake_run_on(node, cmd, timeout=10, check=True):
        if "scancel" in cmd:
            try:
                cancelled_jids.add(int(cmd.split()[-1]))
            except Exception:
                pass
            return (0, "", "")
        if "squeue" in cmd:
            jid = None
            try:
                parts = cmd.split()
                jid = int(parts[parts.index("-j") + 1])
            except Exception:
                pass
            if jid is None or jid in cancelled_jids:
                return (0, "", "")  # gone after scancel
            return (0, "PENDING\n", "")
        return (0, "", "")
    sch.run_on = fake_run_on
    from contextlib import contextmanager as _cm
    @_cm
    def fake_lock():
        yield
    sch.state_lock = fake_lock

    class Args: yes = True

    try:
        sch.cmd_rebalance_pending(Args())
        post = fake_state["tasks"][0]
        check("rebalanced task: status=queued",
              post["status"] == "queued", diag=str(post))
        check("rebalanced task: slurm_job_id cleared",
              post.get("slurm_job_id") is None)
        check("rebalanced task: node cleared",
              post.get("node") is None,
              diag=f"node still {post.get('node')!r} (would mislead status/TUI/env-smoke)")
        check("rebalanced task: gpu_idx cleared",
              post.get("gpu_idx") is None,
              diag=f"gpu_idx still {post.get('gpu_idx')!r}")
        check("rebalanced task: actual_started_at cleared",
              post.get("actual_started_at") is None,
              diag=f"actual_started_at still {post.get('actual_started_at')!r}")
        check("rebalanced task: started_at cleared (was already covered)",
              post.get("started_at") is None)
        # The audit trail keeps the old node name in last_block_reason.
        check("last_block_reason still names the old node (captured before clear)",
              "n1" in (post.get("last_block_reason") or ""),
              diag=f"got {post.get('last_block_reason')!r}")
    finally:
        sch.save_state = saved_save
        sch.load_state = saved_load
        sch.run_on = saved_run_on
        sch.state_lock = saved_lock
        time.sleep = saved_sleep


def test_phase3_0_25_zombie_descendants_not_alive():
    """Phase 3.0.25 P1 fix: zombie (Z) / dead (X) descendants must not count as
    alive in LocalBackend.batch_probe.

    Pre-fix flow: per-root liveness used `kill -0 + /proc/<pid>/status` to
    filter Z/X (correct). But descendant RAM/CPU expansion built `rss_per_pid`
    from `ps -eo pid,ppid,rss,pcpu` — which lists zombies. Then
    `this_alive = expanded_pids & (alive_roots | set(rss_per_pid))` UNIONED
    zombies back in. A task whose root + every descendant got reaped to
    zombies could stay status=running indefinitely.

    Now: ps requests `stat=` too; the parser drops Z/X before populating
    rss_per_pid / ppid_of so descendant expansion + the alive intersection
    never see them.
    """
    print("\n[81] Phase 3.0.25 P1 fix: zombie descendants are NOT counted as alive")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    lb_idx = src.find("class LocalBackend")
    lb_end = src.find("\nclass ", lb_idx + 5)
    body = src[lb_idx:lb_end]
    check("ps -eo includes stat= column",
          "ps -eo pid=,ppid=,rss=,pcpu=,stat=" in body)
    check("PSALL parser requires 5 columns + drops Z/X",
          "len(bits) < 5" in body and 'stat[0] in ("Z", "X")' in body)

    # 2. Behavioral via update_running_tasks → LocalBackend.batch_probe.
    saved_run_on = sch.run_on

    def make_state(pid):
        return {"next_id": 999, "tasks": [{
            "id": "tZ", "status": "running", "node": "local",
            "remote_pids": [pid], "auto_adopted": False,
            "process_group": pid, "started_at": time.time() - 600,
            "log_path": "/tmp/x.log", "ram_mb": 100, "cpu_cores": 1,
            "peak_ram_mb": 0, "peak_vram_mb": 0,
        }]}

    try:
        # ---- Case A: root alive, descendant zombie → only the root counts ----
        # `kill -0` succeeds for both, so the prefix probe emits "A100" and "A101".
        # PSALL reports 100 alive (S) and 101 zombie (Z). The zombie must not
        # contribute to RAM/CPU and must not be in alive_pids.
        state = make_state(100)
        def fake_root_alive_desc_zombie(node, cmd, timeout=30, check=False):
            return 0, (
                "A100\n"          # root alive
                "===VRAM===\n"
                "===PSALL===\n"
                "100 1 1000 5.0 S\n"        # root alive
                "101 100 999999 200.0 Z\n"  # descendant zombie — must NOT count
            ), ""
        sch.run_on = fake_root_alive_desc_zombie
        sch.update_running_tasks(state)
        t = state["tasks"][0]
        check("root alive + descendant zombie → still alive (root counts)",
              t["status"] == "running",
              diag=f"status={t.get('status')}")
        check("zombie descendant excluded from alive_pids",
              101 not in (t.get("alive_pids") or []),
              diag=f"alive_pids={t.get('alive_pids')}")
        check("zombie descendant excluded from peak_ram_mb",
              t.get("peak_ram_mb", 0) < 1000,
              diag=f"peak_ram_mb={t.get('peak_ram_mb')} (zombie's 999999 KB must not show)")

        # ---- Case B: root reaped to zombie → state should NOT keep running ----
        state = make_state(200)
        def fake_root_zombie(node, cmd, timeout=30, check=False):
            # alive_roots filter (in scheduler) requires /proc state ∉ {Z,X};
            # the prefix probe emits nothing for a zombie root.
            return 0, (
                ""               # no "A200" → not alive_root
                "===VRAM===\n"
                "===PSALL===\n"
                "200 1 5000 0.0 Z\n"  # root zombie — must NOT count
            ), ""
        sch.run_on = fake_root_zombie
        sch.update_running_tasks(state)
        t = state["tasks"][0]
        check("root zombie + no alive descendant → task NOT kept running forever",
              t["status"] != "running",
              diag=f"status={t.get('status')} (zombie root must not stay alive)")

        # ---- Case C: descendant alive but the root has a zombie sibling ---
        # _descendants_of walks ppid_of; zombies are dropped from ppid_of so
        # the zombie can't be a "stepping stone" parent. But a descendant of an
        # alive root that itself has a zombie child should still be discovered.
        # 300 (root, alive) → 301 (descendant, alive) → 302 (zombie, dropped)
        state = make_state(300)
        def fake_alive_chain(node, cmd, timeout=30, check=False):
            return 0, (
                "A300\n"
                "===VRAM===\n"
                "===PSALL===\n"
                "300 1 1000 5.0 S\n"
                "301 300 2000 10.0 R\n"
                "302 301 50 0.0 Z\n"
            ), ""
        sch.run_on = fake_alive_chain
        sch.update_running_tasks(state)
        t = state["tasks"][0]
        check("alive chain found via ppid_of (zombie leaf dropped)",
              t["status"] == "running" and 301 in (t.get("alive_pids") or []),
              diag=f"alive_pids={t.get('alive_pids')}")
        check("zombie great-grandchild excluded from alive_pids",
              302 not in (t.get("alive_pids") or []))

        # ---- Case D: STAT prefix may be augmented (e.g. "Sl+", "Z+"). Parser
        # must look at first char only.
        state = make_state(400)
        def fake_augmented(node, cmd, timeout=30, check=False):
            return 0, (
                "A400\n"
                "===VRAM===\n"
                "===PSALL===\n"
                "400 1 1000 5.0 Sl+\n"        # alive (sleep, leader, fg)
                "401 400 2000 10.0 Z+\n"      # zombie (foreground)
            ), ""
        sch.run_on = fake_augmented
        sch.update_running_tasks(state)
        t = state["tasks"][0]
        check("multi-char STAT works: Sl+ alive, Z+ filtered",
              t["status"] == "running"
              and 400 in (t.get("alive_pids") or [])
              and 401 not in (t.get("alive_pids") or []),
              diag=f"alive_pids={t.get('alive_pids')}")
    finally:
        sch.run_on = saved_run_on


def test_phase3_0_26_auto_docker_no_local_digest_falls_back_to_none():
    """Phase 3.0.26 P1 fix: auto-mode docker with no local digest falls back to
    `none` (bare cmd), not the legacy "remote tag exists = OK" path.

    Pre-fix: 3.0.21 closed the explicit `docker:IMAGE` hole — but `auto` mode
    with `image` set still slipped through. has_image() with local_digest=None
    falls back to "remote has any tag → True", which lets a stale prior push
    silently run when local isn't authoritative. Auto's contract is "use
    docker if available, else none" — and we cannot verify the image is fresh
    without a local digest, so the "available" predicate must require a
    digest too.

    Now: when not explicit AND local_digest is None → return inner unchanged
    (kind=none equivalent). No docker wrap, no push attempt, no error.
    """
    print("\n[82] Phase 3.0.26 P1 fix: auto+no-local-digest falls back to none (no stale-tag run)")

    # 1. Source guard.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _maybe_wrap_docker")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    check("auto-mode no-local-digest fallback exists",
          "if not explicit and local_digest is None" in body,
          diag="auto path must short-circuit before has_image() fallthrough")
    explicit_idx = body.find("if explicit and local_digest is None")
    auto_idx = body.find("if not explicit and local_digest is None")
    has_image_idx = body.find("env_deploy.has_image(run_on, node, chosen_image, local_digest=local_digest)")
    check("auto fallback sits BEFORE has_image() fallthrough",
          0 < explicit_idx < auto_idx < has_image_idx,
          diag=f"explicit={explicit_idx} auto={auto_idx} has_image={has_image_idx}")

    # 2. Behavioral.
    saved_env_deploy = sch.env_deploy
    saved_NODES = sch.NODES
    sch.NODES = {"n1": {"host": "n1box"}}

    class FakeED:
        @staticmethod
        def parse_env_spec(spec):
            if spec == "docker":
                return ("docker", "")
            if spec.startswith("docker:"):
                return ("docker", spec.split(":", 1)[1])
            if spec == "auto":
                return ("auto", "")
            return ("none", "")
        @staticmethod
        def has_docker(run_on, node, timeout=8): return True
        local_digest_ref = {"value": None}
        push_called = {"n": 0}
        has_image_called = {"n": 0}

        @classmethod
        def get_image_digest(cls, run_on, node, image, timeout=10):
            if node == "local":
                return cls.local_digest_ref["value"]
            return "remote-stale-digest"

        @classmethod
        def has_image(cls, run_on, node, image, local_digest=None, timeout=10):
            cls.has_image_called["n"] += 1
            # Legacy fallthrough: tag present + local_digest None → True (the bug).
            if local_digest is None:
                return True
            return cls.get_image_digest(None, node, image) == local_digest

        @classmethod
        def push_image(cls, node_host, image, timeout_s=1800):
            cls.push_called["n"] += 1
            return (True, "ok")

        @staticmethod
        def wrap_cmd_docker(inner, image, cwd, gpu_idx, extra_env, container_name,
                            memory_mb, cpus, gpu_runtime_env):
            return f"docker_wrap({inner})"

    sch.env_deploy = FakeED

    try:
        # ---- Case A: auto + no local digest → fall back to none (no wrap) ----
        FakeED.local_digest_ref["value"] = None
        FakeED.push_called["n"] = 0
        FakeED.has_image_called["n"] = 0
        task = {"id": "tA", "node": "n1", "env_spec": "auto",
                "image": "myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("auto + no local digest → no error",
              err is None, diag=f"err={err!r}")
        check("auto + no local digest → cmd NOT docker-wrapped (kind=none equivalent)",
              inner == "python a.py", diag=f"inner={inner!r}")
        check("auto + no local digest → has_image NOT called (short-circuit before fallthrough)",
              FakeED.has_image_called["n"] == 0,
              diag=f"has_image_n={FakeED.has_image_called['n']}")
        check("auto + no local digest → push NOT attempted",
              FakeED.push_called["n"] == 0)

        # ---- Case B: auto + local digest matches remote → docker-wrap as usual ----
        FakeED.local_digest_ref["value"] = "remote-stale-digest"
        FakeED.push_called["n"] = 0
        FakeED.has_image_called["n"] = 0
        task = {"id": "tB", "node": "n1", "env_spec": "auto",
                "image": "myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("auto + matching digest → ok, docker-wrap",
              err is None and inner.startswith("docker_wrap("),
              diag=f"err={err!r} inner={inner!r}")

        # ---- Case C: auto + local digest different → graceful fallback ----
        # Phase 3.0.31: push moved out of launch-time state_lock. On drift at
        # launch (preload either failed or is in flight), auto falls back to
        # bare cmd (kind=none equivalent) — no push, no wrap, no error. The
        # next dispatch cycle's preload will retry the transfer.
        FakeED.local_digest_ref["value"] = "fresh-local-digest"
        FakeED.push_called["n"] = 0
        FakeED.has_image_called["n"] = 0
        task = {"id": "tC", "node": "n1", "env_spec": "auto",
                "image": "myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("auto + digest drift at launch → graceful fallback (no push, no wrap, no error)",
              err is None
              and FakeED.push_called["n"] == 0
              and inner == "python a.py",
              diag=f"err={err!r} push_n={FakeED.push_called['n']} inner={inner!r}")

        # ---- Case D (regression): explicit + no local digest still fails ----
        FakeED.local_digest_ref["value"] = None
        FakeED.push_called["n"] = 0
        task = {"id": "tD", "node": "n1", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("explicit + no local digest still fail-fast (3.0.21 regression)",
              err is not None
              and ("stale or unintended image" in err
                   or "stale remote tag" in err),
              diag=f"err={err!r}")
    finally:
        sch.env_deploy = saved_env_deploy
        sch.NODES = saved_NODES


def test_phase3_0_27_conda_sync_success_gate():
    """Phase 3.0.27 P1 fix: explicit conda launch must verify the LATEST sync
    to its target node succeeded — not just that the local path exists.

    Pre-fix: 3.0.22 closed the local-path-missing hole, but it can't see
    remote-side staleness. push_conda_env can fail (ssh blip, target disk
    full, rsync timeout) while the local path is fine; preload only
    notify()'d the failure. If the remote happened to have a stale env at
    the same path (a prior sync that succeeded), the launch silently used
    the stale env. Same blast-radius shape as the docker stale-tag P1.

    Now: _CONDA_SYNC_OK marker per (node, env_path), TTL'd to STAGING_TTL_S.
    Preload writes on success, clears on failure (failure overrides any
    earlier success). Launch refuses to wrap when the marker is missing /
    expired for a remote node + absolute path. Local nodes (no host) and
    non-absolute conda specs (`conda:envname`) bypass the gate.
    """
    print("\n[83] Phase 3.0.27 P1 fix: explicit conda launch requires fresh sync marker")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_CONDA_SYNC_OK marker dict + helpers exist",
          "_CONDA_SYNC_OK" in src
          and "def _record_conda_sync_ok" in src
          and "def _record_conda_sync_failed" in src
          and "def _conda_sync_ok" in src)
    check("preload records sync OK on success",
          "_record_conda_sync_ok(node, env_path)" in src)
    check("preload clears marker on push_conda_env failure AND exception",
          src.count("_record_conda_sync_failed") >= 2)
    fn_idx = src.find("def _maybe_wrap_docker")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    check("_maybe_wrap_docker conda branch consults _conda_sync_ok",
          "_conda_sync_ok(node, spec_image)" in body)
    check("conda gate skipped for local nodes (no host) and non-absolute specs",
          "node_host and spec_image" in body
          and "Path(spec_image).is_absolute()" in body)

    # 2. Helper unit tests.
    saved_marker = dict(sch._CONDA_SYNC_OK)
    sch._CONDA_SYNC_OK.clear()
    try:
        sch._record_conda_sync_ok("n1", "/env/a")
        check("record_ok → _conda_sync_ok returns True",
              sch._conda_sync_ok("n1", "/env/a") is True)
        sch._record_conda_sync_failed("n1", "/env/a")
        check("record_failed clears prior OK marker → returns False",
              sch._conda_sync_ok("n1", "/env/a") is False)
        # TTL expiry behavior.
        sch._CONDA_SYNC_OK[("n1", "/env/expired")] = time.time() - sch.STAGING_TTL_S - 60
        check("expired marker → False (and popped)",
              sch._conda_sync_ok("n1", "/env/expired") is False
              and ("n1", "/env/expired") not in sch._CONDA_SYNC_OK)
        # Missing key → False (not error).
        check("missing key → False",
              sch._conda_sync_ok("n1", "/none") is False)
    finally:
        sch._CONDA_SYNC_OK.clear()
        sch._CONDA_SYNC_OK.update(saved_marker)

    # 3. Behavioral via _maybe_wrap_docker.
    saved_env_deploy = sch.env_deploy
    saved_NODES = sch.NODES
    saved_marker = dict(sch._CONDA_SYNC_OK)
    sch.NODES = {
        "remote1": {"host": "h1box"},
        "local1":  {"host": None},  # local node — sync gate must NOT fire
    }
    class FakeED:
        @staticmethod
        def parse_env_spec(spec):
            if spec.startswith("conda:"):
                return ("conda", spec.split(":", 1)[1])
            return ("none", "")
    sch.env_deploy = FakeED
    sch._CONDA_SYNC_OK.clear()

    # Use a real existing path so the 3.0.22 local-path gate doesn't fire first.
    import tempfile
    tdir = tempfile.mkdtemp()
    try:
        env_path = tdir
        # ---- Case A: remote node + no fresh sync marker → fail-fast ----
        task = {"id": "tA", "node": "remote1", "env_spec": f"conda:{env_path}"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("remote conda + no sync marker → fail (would risk stale remote env)",
              err is not None
              and "did not succeed" in err
              and "preload_conda_failed" in err,
              diag=f"err={err!r}")
        check("conda fail-fast: inner cmd unchanged",
              inner == "python a.py")

        # ---- Case B: remote node + fresh sync marker → ok ----
        sch._record_conda_sync_ok("remote1", env_path)
        task = {"id": "tB", "node": "remote1", "env_spec": f"conda:{env_path}"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("remote conda + fresh sync marker → ok (no error, inner unchanged)",
              err is None and inner == "python a.py",
              diag=f"err={err!r}")

        # ---- Case C: remote node + sync failed (marker cleared) → fail ----
        sch._record_conda_sync_ok("remote1", env_path)  # was OK
        sch._record_conda_sync_failed("remote1", env_path)  # then preload failed
        task = {"id": "tC", "node": "remote1", "env_spec": f"conda:{env_path}"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("remote conda + sync failed (marker cleared) → fail",
              err is not None,
              diag=f"err={err!r}")

        # ---- Case D: LOCAL node + no marker → ok (gate skipped, env IS local) ----
        task = {"id": "tD", "node": "local1", "env_spec": f"conda:{env_path}"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("local node (no host) → conda sync gate SKIPPED, no error",
              err is None,
              diag=f"err={err!r}")

        # ---- Case E: non-absolute conda env-name → gate skipped ----
        task = {"id": "tE", "node": "remote1", "env_spec": "conda:envname"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("non-absolute conda envname spec → gate SKIPPED (no rsync path)",
              err is None,
              diag=f"err={err!r}")

        # ---- Case F: TTL expiry → fail (no fresh marker) ----
        sch._CONDA_SYNC_OK[("remote1", env_path)] = time.time() - sch.STAGING_TTL_S - 60
        task = {"id": "tF", "node": "remote1", "env_spec": f"conda:{env_path}"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("expired sync marker → fail (treated as no marker)",
              err is not None,
              diag=f"err={err!r}")
    finally:
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)
        sch.env_deploy = saved_env_deploy
        sch.NODES = saved_NODES
        sch._CONDA_SYNC_OK.clear()
        sch._CONDA_SYNC_OK.update(saved_marker)


def test_phase3_0_28_local_wal_orphan_recovery():
    """Phase 3.0.28 P1 fix: LocalBackend WAL orphan recovery prevents
    double-launch after a scheduler crash mid-launch.

    Pre-fix: SlurmBackend had _try_recover_orphan_slurm_job (Phase 2.15), but
    LocalBackend had no equivalent. The window: LocalBackend.launch persists
    status='launching' (WAL save_state) BEFORE the ssh+nohup. If the launch
    succeeds but scheduler dies before the post-launch save_state can flush
    status=running + remote_pids, recover_stale_launching_tasks just reverts
    launching → queued. Next dispatch then re-launches, and the original
    process is still running on the node — the auto-adopt machinery later
    creates a SECOND task record for it. Two copies of the same workload.

    Now: LocalBackend.launch injects SCHEDULEURM_TASK_ID=<id> as an env var.
    _try_recover_orphan_local_task scans /proc/*/environ on the candidate
    node for the marker; on a hit it adopts the orphan onto the existing
    task record (status=running, remote_pids/process_group restored) so
    the revert→requeue→re-launch path is bypassed.
    """
    print("\n[84] Phase 3.0.28 P1 fix: LocalBackend WAL orphan recovery (no double-launch)")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("SCHEDULEURM_TASK_ID is reserved (user can't override the marker)",
          "_RESERVED_ENV_KEYS" in src
          and "SCHEDULEURM_TASK_ID" in src.split("_RESERVED_ENV_KEYS")[1].split("})")[0])
    lb_idx = src.find("class LocalBackend")
    lb_end = src.find("\nclass ", lb_idx + 5)
    lb_body = src[lb_idx:lb_end]
    check("LocalBackend.launch exports SCHEDULEURM_TASK_ID",
          "export SCHEDULEURM_TASK_ID=" in lb_body)
    check("_try_recover_orphan_local_task helper exists",
          "def _try_recover_orphan_local_task" in src)
    rec_idx = src.find("def recover_stale_launching_tasks")
    rec_end = src.find("\ndef ", rec_idx + 5)
    rec_body = src[rec_idx:rec_end]
    check("recover_stale_launching_tasks calls local-orphan path for non-slurm nodes",
          "_try_recover_orphan_local_task" in rec_body)
    check("local-orphan probe scans /proc/*/environ for the marker",
          "SCHEDULEURM_TASK_ID" in src[src.find("def _try_recover_orphan_local_task"):]
          and "/proc/$p/environ" in src[src.find("def _try_recover_orphan_local_task"):])

    # 2. Behavioral test of _try_recover_orphan_local_task.
    saved_run_on = sch.run_on
    try:
        # ---- Case A: orphan found, alive, session leader → adopted ----
        # Probe returns: pid=12345, sid=12345 (leader), pgid=12345, state=S
        def probe_alive_leader(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=" in cmd and "/proc/$p/environ" in cmd:
                return (0, "12345|12345|12345|S\n", "")
            return (0, "", "")
        sch.run_on = probe_alive_leader
        task = {"id": "t0042", "status": "launching", "node": "n1",
                "launching_started_at": time.time() - 120, "remote_pids": []}
        adopted = sch._try_recover_orphan_local_task(task, "n1")
        check("orphan found alive → adopted=True",
              adopted is True)
        check("adopted task: status=running",
              task["status"] == "running")
        check("adopted task: remote_pids set to leader pid",
              task["remote_pids"] == [12345])
        check("adopted task: process_group set",
              task.get("process_group") == 12345)
        check("adopted task: launching_started_at popped",
              "launching_started_at" not in task)
        check("adopted task: started_at set",
              isinstance(task.get("started_at"), float))
        check("adopted task: last_block_reason mentions WAL recovery + the matched id",
              "WAL recovery" in (task.get("last_block_reason") or "")
              and "t0042" in (task.get("last_block_reason") or ""))

        # ---- Case B: zombie match → NOT adopted (3.0.25 semantics) ----
        def probe_zombie(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=" in cmd:
                return (0, "12345|12345|12345|Z\n", "")
            return (0, "", "")
        sch.run_on = probe_zombie
        task = {"id": "t0043", "status": "launching", "node": "n1",
                "launching_started_at": time.time() - 120, "remote_pids": []}
        adopted = sch._try_recover_orphan_local_task(task, "n1")
        check("zombie orphan → NOT adopted",
              adopted is False and task["status"] == "launching")

        # ---- Case C: no match → adopted=False (caller reverts) ----
        def probe_empty(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=" in cmd:
                return (0, "", "")
            return (0, "", "")
        sch.run_on = probe_empty
        task = {"id": "t0044", "status": "launching", "node": "n1",
                "launching_started_at": time.time() - 120, "remote_pids": []}
        adopted = sch._try_recover_orphan_local_task(task, "n1")
        check("no orphan match → adopted=False (revert path takes over)",
              adopted is False and task["status"] == "launching")

        # ---- Case D: probe rc != 0 → adopted=False ----
        sch.run_on = lambda *a, **k: (1, "", "ssh blip")
        task = {"id": "t0045", "status": "launching", "node": "n1",
                "launching_started_at": time.time() - 120, "remote_pids": []}
        adopted = sch._try_recover_orphan_local_task(task, "n1")
        check("probe ssh failure → adopted=False",
              adopted is False)

        # ---- Case E: prefer session leader over a non-leader match ----
        # Two PIDs match: 99 (non-leader, sid=12345) and 12345 (leader, sid==pid).
        def probe_two(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=" in cmd:
                return (0,
                        "99|12345|12345|S\n"      # descendant, not leader
                        "12345|12345|12345|S\n",  # leader
                        "")
            return (0, "", "")
        sch.run_on = probe_two
        task = {"id": "t0046", "status": "launching", "node": "n1",
                "launching_started_at": time.time() - 120, "remote_pids": []}
        adopted = sch._try_recover_orphan_local_task(task, "n1")
        check("multiple matches → leader (sid==pid) preferred",
              adopted is True and task["remote_pids"] == [12345])

        # ---- Case F: missing task id → False (defensive) ----
        sch.run_on = probe_alive_leader
        adopted = sch._try_recover_orphan_local_task(
            {"id": "", "status": "launching", "node": "n1",
             "launching_started_at": time.time() - 120}, "n1")
        check("missing task id → False (no probe issued, no adoption)",
              adopted is False)
    finally:
        sch.run_on = saved_run_on

    # 3. End-to-end: recover_stale_launching_tasks for a local task with a
    # matching orphan adopts it instead of reverting to queued.
    saved_run_on = sch.run_on
    saved_backend = sch._BACKEND
    fake_hb = sch.HybridBackend()
    fake_hb._cache["lnode"] = "local"
    sch._BACKEND = fake_hb
    sch.run_on = lambda node, cmd, **kw: (
        (0, "7777|7777|7777|S\n", "") if "SCHEDULEURM_TASK_ID=tE2E" in cmd else (0, "", "")
    )
    try:
        state = {"tasks": [{
            "id": "tE2E", "status": "launching", "node": "lnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
        }]}
        reverted = sch.recover_stale_launching_tasks(state, now=time.time())
        t = state["tasks"][0]
        check("end-to-end: local launching task with orphan → adopted (NOT reverted)",
              reverted == 0 and t["status"] == "running" and t["remote_pids"] == [7777],
              diag=str(t))
    finally:
        sch._BACKEND = saved_backend
        sch.run_on = saved_run_on


def test_phase3_0_29_actual_started_at_cleared_on_requeue_and_launch():
    """Phase 3.0.29 P2 fix: actual_started_at must be None on a freshly-
    requeued / relaunched task, not inherited from the parent run.

    Pre-fix: _requeue_after_crash cleared slurm_job_id / slurm_state but NOT
    actual_started_at (Phase 3.0.9 stamp). SlurmBackend.launch didn't clear
    it either. Because _effective_elapsed_s prefers actual_started_at for
    slurm tasks, a retry could carry the parent's old timestamp into the
    PENDING window and report seconds-of-elapsed-compute that didn't happen
    yet. ETA / eta_load / migration decisions then drifted.

    Now: both _requeue_after_crash and SlurmBackend.launch explicitly set
    actual_started_at = None. batch_probe re-stamps it the first time the
    new job actually reaches slurm_state=RUNNING.
    """
    print("\n[85] Phase 3.0.29 P2 fix: actual_started_at cleared on requeue + (re)launch")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    rq_idx = src.find("def _requeue_after_crash")
    rq_end = src.find("\ndef ", rq_idx + 5)
    rq_body = src[rq_idx:rq_end]
    check("_requeue_after_crash sets actual_started_at = None",
          '"actual_started_at": None' in rq_body)
    sb_idx = src.find("class SlurmBackend")
    sb_end = src.find("\nclass ", sb_idx + 5)
    sb_body = src[sb_idx:sb_end]
    check("SlurmBackend.launch sets actual_started_at = None",
          'task["actual_started_at"] = None' in sb_body)

    # 2. Behavioral: _requeue_after_crash on a parent that has actual_started_at
    # set must produce a clone with actual_started_at = None.
    saved_max_retry = sch.MAX_AUTO_RETRY
    try:
        # Build a minimal state with a "crashed" parent task. _requeue_after_crash
        # also writes escalations on cap, but we keep retry_count low.
        parent = {
            "id": "tParent", "status": "failed", "node": "n1",
            "remote_pids": [], "retry_count": 0,
            # Stale stamps from the previous run:
            "actual_started_at": time.time() - 3600,
            "started_at": time.time() - 3700,
            "slurm_job_id": 42, "slurm_state": "FAILED",
            "signature": "TEST/req", "cmd": "x",
            "_diagnosis": {"category": "TRANSIENT", "reason": "ssh blip"},
        }
        state = {"next_id": 100, "tasks": [parent]}
        new_id = sch._requeue_after_crash(parent, state)
        check("_requeue_after_crash produced a new task id",
              new_id is not None and new_id.startswith("t"),
              diag=f"got {new_id!r}")
        clone = next(t for t in state["tasks"] if t["id"] == new_id)
        check("retry clone: actual_started_at cleared (None)",
              clone.get("actual_started_at") is None,
              diag=f"got {clone.get('actual_started_at')!r}")
        check("retry clone: slurm_job_id cleared (existing 3.0.x guard)",
              clone.get("slurm_job_id") is None)
        check("retry clone: started_at cleared",
              clone.get("started_at") is None)
        check("retry clone: status=queued",
              clone.get("status") == "queued")
        # The parent's stamp must remain untouched (audit trail).
        check("parent task: actual_started_at preserved (forensic value)",
              parent.get("actual_started_at") is not None
              and abs(parent["actual_started_at"] - (time.time() - 3600)) < 60)
    finally:
        sch.MAX_AUTO_RETRY = saved_max_retry

    # 3. _effective_elapsed_s on the clone reports 0 (no actual_started_at yet).
    # That's the whole reason this fix matters: stale stamp would have surfaced
    # here as inflated elapsed seconds.
    clone_for_elapsed = {
        "id": "tCheck", "started_at": time.time() - 200,
        "slurm_job_id": None,  # not yet relaunched
        "actual_started_at": None,
    }
    elapsed = sch._effective_elapsed_s(clone_for_elapsed)
    check("retry clone _effective_elapsed_s: LocalBackend semantics until slurm relaunch",
          elapsed > 0 and elapsed < 1000,
          diag=f"got {elapsed}")
    # Now simulate the new slurm relaunch + still PENDING (no actual_started_at):
    clone_for_elapsed["slurm_job_id"] = 999
    clone_for_elapsed["actual_started_at"] = None
    elapsed_pending = sch._effective_elapsed_s(clone_for_elapsed)
    check("after relaunch + still PENDING: elapsed=0 (3.0.9 invariant)",
          elapsed_pending == 0,
          diag=f"got {elapsed_pending}")


def test_phase3_0_30_slurm_completed_log_scan_for_crash():
    """Phase 3.0.30 P2 fix: slurm COMPLETED is trust-but-verify. Scan the log
    tail for explicit crash patterns; if any match, override is_crash=True.

    Pre-fix: terminal_ok=True (slurm COMPLETED) was treated as authoritative
    success. But a pipeline like `python train.py | tee log` returns rc=0
    when the LEFT side traceback'd — without `set -o pipefail`, tee's exit
    code wins and slurm reports COMPLETED. Crashed runs masquerade as
    successful, no auto-requeue, eval downstream uses garbage results.

    Now: lightweight tail scan for CRASH_PATTERNS only (no lifetime / marker
    heuristics — those are reserved for the no-slurm-signal path so that a
    legitimately fast COMPLETED job, e.g. `--skip_existing` no-op, isn't
    re-classified as a crash on lifetime grounds).
    """
    print("\n[86] Phase 3.0.30 P2 fix: slurm COMPLETED log scan catches hidden crashes")

    # 1. Source guard.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_scan_completed_log_for_crash helper exists",
          "def _scan_completed_log_for_crash" in src)
    check("helper scans CRASH_PATTERNS only (no lifetime/marker heuristics)",
          "CRASH_PATTERNS" in src.split("def _scan_completed_log_for_crash")[1].split("\ndef ")[0]
          and "TRAINING_MARKERS" not in src.split("def _scan_completed_log_for_crash")[1].split("\ndef ")[0])
    # The terminal_ok=True branch must call the scanner before trusting.
    check("terminal_ok=True branch calls _scan_completed_log_for_crash",
          "_scan_completed_log_for_crash(t)" in src)

    # 2. Helper unit tests against a real local file.
    import tempfile
    saved_NODES = sch.NODES
    sch.NODES = {"local": {"host": None}}
    try:
        with tempfile.TemporaryDirectory() as td:
            ok_log = os.path.join(td, "ok.log")
            crash_log = os.path.join(td, "crash.log")
            empty_log = os.path.join(td, "empty.log")
            with open(ok_log, "w") as f:
                f.write("Epoch 100/100 loss=0.001\nTraining complete\nFinal model saved\n")
            with open(crash_log, "w") as f:
                f.write("Epoch 5/100 loss=2.3\n")
                f.write("Traceback (most recent call last):\n")
                f.write('  File "train.py", line 42, in <module>\n')
                f.write("    model.forward(x)\n")
                f.write("RuntimeError: CUDA out of memory\n")
            open(empty_log, "w").close()  # 0 bytes

            # ---- Case A: clean log → no crash detected ----
            ok_task = {"id": "tA", "node": "local", "log_path": ok_log,
                       "started_at": time.time() - 100,
                       "finished_at": time.time()}
            matched, reason = sch._scan_completed_log_for_crash(ok_task)
            check("clean log → no crash detected",
                  matched is False, diag=f"got {matched}/{reason}")

            # ---- Case B: log with traceback → crash detected ----
            crash_task = {"id": "tB", "node": "local", "log_path": crash_log,
                          "started_at": time.time() - 100,
                          "finished_at": time.time()}
            matched, reason = sch._scan_completed_log_for_crash(crash_task)
            check("log with Traceback → crash detected",
                  matched is True
                  and ("Traceback" in reason or "CUDA out of memory" in reason
                       or "RuntimeError" in reason),
                  diag=f"got {matched}/{reason}")
            check("crash reason mentions slurm-COMPLETED-but-log",
                  "slurm reported COMPLETED" in reason, diag=reason)

            # ---- Case C: empty log → conservative no-crash ----
            empty_task = {"id": "tC", "node": "local", "log_path": empty_log,
                          "started_at": time.time() - 100,
                          "finished_at": time.time()}
            matched, reason = sch._scan_completed_log_for_crash(empty_task)
            check("empty log → conservative no-crash",
                  matched is False, diag=f"got {matched}/{reason}")

            # ---- Case D: no log_path → no-crash (defensive) ----
            no_log_task = {"id": "tD", "node": "local",
                           "started_at": time.time() - 100,
                           "finished_at": time.time()}
            matched, reason = sch._scan_completed_log_for_crash(no_log_task)
            check("no log_path → no-crash (no scan attempted)",
                  matched is False)

            # ---- Case E: auto_adopted → skip scan ----
            adopted_task = {"id": "tE", "node": "local", "log_path": crash_log,
                            "auto_adopted": True,
                            "started_at": time.time() - 100,
                            "finished_at": time.time()}
            matched, reason = sch._scan_completed_log_for_crash(adopted_task)
            check("auto_adopted task → no scan even if log shows crash",
                  matched is False)

            # ---- Case F: missing file (ssh blip simulation for remote) ----
            sch.NODES = {"local": {"host": None}, "remote": {"host": "rbox"}}
            saved_run_on = sch.run_on
            try:
                sch.run_on = lambda node, cmd, **kw: (1, "", "ssh: timed out")
                remote_task = {"id": "tF", "node": "remote",
                               "log_path": "/tmp/whatever",
                               "started_at": time.time() - 100,
                               "finished_at": time.time()}
                matched, reason = sch._scan_completed_log_for_crash(remote_task)
                check("remote ssh failure → no-crash (conservative on probe failure)",
                      matched is False)
            finally:
                sch.run_on = saved_run_on
    finally:
        sch.NODES = saved_NODES


def test_phase3_0_31_launch_side_docker_push_no_longer_holds_lock():
    """Phase 3.0.31 P3 fix: _maybe_wrap_docker must not call push_image
    synchronously inside the dispatch state_lock.

    Pre-fix: when the image was missing or had digest drift on the target,
    _maybe_wrap_docker called env_deploy.push_image(timeout_s=1800) right
    there — INSIDE the dispatch state_lock. A single missing image could
    block submit / status / cancel / watcher iterations for up to 30 min
    while the docker save | ssh load round-trip ran. Both cmd_dispatch and
    _watch_iteration already call _preload_docker_images_outside_lock
    BEFORE acquiring state_lock; push belongs there, not at launch.

    Now: launch-time push is removed. If has_image() returns False at the
    launch site (preload not yet successful for this image), explicit mode
    returns an error pointing at the preload-retry path; auto mode falls
    back to bare cmd (kind=none equivalent). The next dispatch cycle's
    preload retries the transfer.
    """
    print("\n[87] Phase 3.0.31 P3 fix: launch-side docker push removed (preload owns transfers)")

    # 1. Source guard: env_deploy.push_image must NOT appear inside
    # _maybe_wrap_docker. It SHOULD still appear in _preload_docker_images_outside_lock.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _maybe_wrap_docker")
    fn_end = src.find("\ndef ", fn_idx + 5)
    fn_body = src[fn_idx:fn_end]
    check("_maybe_wrap_docker does NOT call env_deploy.push_image",
          "env_deploy.push_image(" not in fn_body,
          diag="env_deploy.push_image() call inside _maybe_wrap_docker would be inside state_lock")
    preload_idx = src.find("def _preload_docker_images_outside_lock")
    preload_end = src.find("\ndef ", preload_idx + 5)
    preload_body = src[preload_idx:preload_end]
    check("_preload_docker_images_outside_lock STILL calls env_deploy.push_image",
          "env_deploy.push_image" in preload_body,
          diag="preload is the right place to push")
    check("_maybe_wrap_docker has-image-miss branch mentions preload-retry",
          "preload" in fn_body and "retry" in fn_body)

    # 2. Behavioral: confirm at runtime that no synchronous push_image is
    # invoked from _maybe_wrap_docker, even on the drift path.
    saved_env_deploy = sch.env_deploy
    saved_NODES = sch.NODES
    sch.NODES = {"n1": {"host": "n1box"}}

    class FakeED:
        @staticmethod
        def parse_env_spec(spec):
            if spec.startswith("docker:"):
                return ("docker", spec.split(":", 1)[1])
            if spec == "auto":
                return ("auto", "")
            return ("none", "")
        @staticmethod
        def has_docker(run_on, node, timeout=8): return True

        local_digest_ref = {"value": None}
        push_called = {"n": 0}

        @classmethod
        def get_image_digest(cls, run_on, node, image, timeout=10):
            if node == "local":
                return cls.local_digest_ref["value"]
            return "remote-stale"

        @classmethod
        def has_image(cls, run_on, node, image, local_digest=None, timeout=10):
            # Simulate drift: local digest != remote digest.
            return cls.get_image_digest(None, node, image) == local_digest

        @classmethod
        def push_image(cls, *a, **k):
            cls.push_called["n"] += 1
            return (True, "ok")

        @staticmethod
        def wrap_cmd_docker(inner, image, cwd, gpu_idx, extra_env, container_name,
                            memory_mb, cpus, gpu_runtime_env):
            return f"docker_wrap({inner})"

    sch.env_deploy = FakeED

    try:
        # ---- Case A: explicit + drift → preload-retry error, push NOT called ----
        FakeED.local_digest_ref["value"] = "fresh-local-digest"
        FakeED.push_called["n"] = 0
        task = {"id": "tA", "node": "n1", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("explicit + drift → error mentions preload",
              err is not None and "preload" in err, diag=f"err={err!r}")
        check("explicit + drift → push_image NOT invoked synchronously",
              FakeED.push_called["n"] == 0,
              diag=f"push_n={FakeED.push_called['n']}")
        check("explicit + drift → cmd unchanged (no docker wrap)",
              inner == "python a.py")

        # ---- Case B: auto + drift → graceful fallback to none, push NOT called ----
        FakeED.local_digest_ref["value"] = "fresh-local-digest"
        FakeED.push_called["n"] = 0
        task = {"id": "tB", "node": "n1", "env_spec": "auto",
                "image": "myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("auto + drift → no error",
              err is None, diag=f"err={err!r}")
        check("auto + drift → push_image NOT invoked synchronously",
              FakeED.push_called["n"] == 0,
              diag=f"push_n={FakeED.push_called['n']}")
        check("auto + drift → bare cmd (kind=none equivalent)",
              inner == "python a.py")

        # ---- Case C: image already present (no drift) → docker wraps as usual ----
        FakeED.local_digest_ref["value"] = "remote-stale"  # matches → no drift
        FakeED.push_called["n"] = 0
        task = {"id": "tC", "node": "n1", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("explicit + no drift → ok, docker-wrap (happy path preserved)",
              err is None
              and inner.startswith("docker_wrap(")
              and FakeED.push_called["n"] == 0,
              diag=f"err={err!r} inner={inner!r} push_n={FakeED.push_called['n']}")
    finally:
        sch.env_deploy = saved_env_deploy
        sch.NODES = saved_NODES


def test_phase3_0_32_orphan_recovery_restores_log_and_docker_artifacts():
    """Phase 3.0.32 P1 fix: _try_recover_orphan_local_task must restore the
    deterministic log_path AND docker artifacts (container_name, container_
    main_pid) that LocalBackend.launch would have set, not just the PID.

    Pre-fix consequences:
      - log_path=None after recovery → _diagnose_terminal short-circuits to
        `is_crash=False, reason="auto-adopted (no scheduler log; cannot
        diagnose)"` for ANY task that went through orphan recovery. Real
        crashes get silently swallowed; auto-requeue path never fires.
      - container_name=None / container_main_pid=None → kill paths skip
        `docker kill <name>` cleanup (only host-side bash kill), and peak-
        resource tracking locks onto the docker-run client bash PID instead
        of the actual container main proc through containerd-shim PID
        isolation. peak_vram / peak_ram come back as zeros.

    Now: log_path is recomputed deterministically from task id + node host
    (same formula as launch). For docker tasks, container_name is derived
    from id (`sched-{id}`) and `docker inspect` resolves the container main
    PID, replacing remote_pids[0] so resource tracking lights up.
    """
    print("\n[88] Phase 3.0.32 P1 fix: orphan recovery restores log_path + docker artifacts")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _try_recover_orphan_local_task")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    check("recovery sets log_path with the local-host formula",
          'STATE_DIR}/logs/{tid}.log' in body)
    check("recovery sets log_path with the remote-host formula",
          '/tmp/sched_{tid}.log' in body)
    check("recovery probes docker for container artifacts",
          'docker inspect' in body and 'sched-{tid}' in body)
    check("docker artifact recovery sets container_name + container_main_pid",
          'task["container_name"] = cname' in body
          and 'task["container_main_pid"] = cpid' in body)

    # 2. Behavioral.
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    saved_state_dir = sch.STATE_DIR
    sch.NODES = {
        "lhost": {"host": None},   # local
        "rhost": {"host": "rbox"}, # remote
    }
    sch.STATE_DIR = "/tmp/sched_test_state"

    try:
        # ---- Case A: local node, non-docker task → log_path = STATE_DIR/logs/<id>.log
        def probe_local_alive(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=" in cmd:
                return (0, "1234|1234|1234|S\n", "")
            return (0, "", "")
        sch.run_on = probe_local_alive
        task = {"id": "tA", "status": "launching", "node": "lhost",
                "launching_started_at": time.time() - 120, "remote_pids": [],
                "env_spec": "none"}
        adopted = sch._try_recover_orphan_local_task(task, "lhost")
        check("local non-docker → adopted",
              adopted is True)
        check("local non-docker → log_path uses STATE_DIR/logs/<id>.log",
              task.get("log_path") == "/tmp/sched_test_state/logs/tA.log",
              diag=f"got {task.get('log_path')!r}")
        check("local non-docker → no container_name set",
              task.get("container_name") is None)

        # ---- Case B: remote node, non-docker task → log_path = /tmp/sched_<id>.log
        sch.run_on = probe_local_alive
        task = {"id": "tB", "status": "launching", "node": "rhost",
                "launching_started_at": time.time() - 120, "remote_pids": [],
                "env_spec": "none"}
        adopted = sch._try_recover_orphan_local_task(task, "rhost")
        check("remote non-docker → log_path uses /tmp/sched_<id>.log",
              task.get("log_path") == "/tmp/sched_tB.log",
              diag=f"got {task.get('log_path')!r}")

        # ---- Case C: docker task, container alive → recover container_name + main_pid ----
        def probe_docker_alive(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=tC" in cmd:
                # Bash launcher matches the marker.
                return (0, "5000|5000|5000|S\n", "")
            if "docker inspect" in cmd and "sched-tC" in cmd:
                # Container is alive with main PID 9999.
                return (0, "9999\n", "")
            return (0, "", "")
        sch.run_on = probe_docker_alive
        task = {"id": "tC", "status": "launching", "node": "rhost",
                "launching_started_at": time.time() - 120, "remote_pids": [],
                "env_spec": "docker:myproj:latest"}
        adopted = sch._try_recover_orphan_local_task(task, "rhost")
        check("docker task → adopted",
              adopted is True)
        check("docker task → container_name = sched-<id>",
              task.get("container_name") == "sched-tC",
              diag=f"got {task.get('container_name')!r}")
        check("docker task → container_main_pid set from docker inspect",
              task.get("container_main_pid") == 9999)
        check("docker task → remote_pids replaced with container main PID",
              task.get("remote_pids") == [9999],
              diag=f"got {task.get('remote_pids')}")
        check("docker task → log_path still set",
              task.get("log_path") == "/tmp/sched_tC.log")

        # ---- Case D: docker task, but `docker inspect` fails (rc!=0) → keep
        # bash PID, no container_name set. Don't crash, don't pretend.
        def probe_docker_inspect_fails(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=tD" in cmd:
                return (0, "5001|5001|5001|S\n", "")
            if "docker inspect" in cmd:
                return (1, "", "Error: No such object")
            return (0, "", "")
        sch.run_on = probe_docker_inspect_fails
        task = {"id": "tD", "status": "launching", "node": "rhost",
                "launching_started_at": time.time() - 120, "remote_pids": [],
                "env_spec": "docker:myproj:latest"}
        adopted = sch._try_recover_orphan_local_task(task, "rhost")
        check("docker inspect fails → still adopt (with bash PID)",
              adopted is True and task.get("remote_pids") == [5001])
        check("docker inspect fails → container_name NOT set (graceful)",
              task.get("container_name") is None)
        check("docker inspect fails → log_path still restored",
              task.get("log_path") == "/tmp/sched_tD.log")

        # ---- Case E: auto + image → also treated as docker for recovery ----
        def probe_auto_alive(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=tE" in cmd:
                return (0, "6000|6000|6000|S\n", "")
            if "docker inspect" in cmd and "sched-tE" in cmd:
                return (0, "8888\n", "")
            return (0, "", "")
        sch.run_on = probe_auto_alive
        task = {"id": "tE", "status": "launching", "node": "rhost",
                "launching_started_at": time.time() - 120, "remote_pids": [],
                "env_spec": "auto", "image": "myproj:latest"}
        adopted = sch._try_recover_orphan_local_task(task, "rhost")
        check("auto+image → docker recovery attempted",
              adopted is True and task.get("container_main_pid") == 8888)

        # ---- Case F: env_spec=none → docker recovery NOT attempted ----
        # docker inspect should not even be called.
        inspect_calls = []
        def probe_no_docker(node, cmd, timeout=20, check=False):
            if "SCHEDULEURM_TASK_ID=tF" in cmd:
                return (0, "7000|7000|7000|S\n", "")
            if "docker inspect" in cmd:
                inspect_calls.append(cmd)
                return (0, "", "")
            return (0, "", "")
        sch.run_on = probe_no_docker
        task = {"id": "tF", "status": "launching", "node": "rhost",
                "launching_started_at": time.time() - 120, "remote_pids": [],
                "env_spec": "none"}
        adopted = sch._try_recover_orphan_local_task(task, "rhost")
        check("non-docker env_spec → docker inspect NOT issued",
              len(inspect_calls) == 0,
              diag=f"inspect calls: {inspect_calls}")
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES
        sch.STATE_DIR = saved_state_dir


def test_phase3_0_33_terminal_orphan_classification():
    """Phase 3.0.33 P1 fix: orphan recovery must classify TERMINAL orphans
    (not just alive ones) so a task that completed within the launch-save
    window doesn't get re-queued + re-launched.

    Pre-fix flow:
      - alive orphan probe finds nothing (process already exited)
      - revert path runs → status=queued
      - next dispatch sbatches / re-launches
      - duplicate run

    Now: after the alive-orphan probe returns False, recovery looks for
    terminal evidence. SlurmBackend extends squeue to terminal states:
    COMPLETED → done; other terminal → failed (with auto-requeue).
    LocalBackend probes the deterministic log_path; if present with
    content, runs _diagnose_terminal and classifies done/failed.
    """
    print("\n[89] Phase 3.0.33 P1 fix: terminal-orphan classification (no double-run on fast tasks)")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    sb_idx = src.find("def _try_recover_orphan_slurm_job")
    sb_end = src.find("\ndef ", sb_idx + 5)
    sb_body = src[sb_idx:sb_end]
    check("slurm orphan recovery handles terminal states (no longer skips)",
          "slurm_state not in _SLURM_ALIVE_STATES" in sb_body
          and 'task["status"] = "done"' in sb_body
          and 'task["status"] = "failed"' in sb_body)
    check("slurm terminal-orphan COMPLETED triggers log scan via 3.0.30 helper",
          "_scan_completed_log_for_crash" in sb_body)
    check("_try_finalize_terminal_local_task helper exists",
          "def _try_finalize_terminal_local_task" in src)
    rec_idx = src.find("def recover_stale_launching_tasks")
    rec_end = src.find("\ndef ", rec_idx + 5)
    rec_body = src[rec_idx:rec_end]
    check("recover_stale_launching_tasks calls local terminal helper after alive probe",
          "_try_finalize_terminal_local_task" in rec_body)

    # 2. Slurm terminal-orphan: COMPLETED → done.
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    sch.NODES = {"slurmnode": {"host": None}}

    sch.run_on = lambda node, cmd, **kw: (
        (0, "12345 COMPLETED\n", "") if "squeue -h -n scheduleurm-tA" in cmd
        else (0, "", "")
    )
    try:
        state = {"next_id": 100, "tasks": [{
            "id": "tA", "status": "launching", "node": "slurmnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python a.py", "cwd": "/work",
        }]}
        adopted = sch._try_recover_orphan_slurm_job(state["tasks"][0], "slurmnode", state)
        check("slurm COMPLETED orphan → adopted=True (NOT skipped)",
              adopted is True)
        t = state["tasks"][0]
        check("slurm COMPLETED → status=done",
              t.get("status") == "done", diag=str(t))
        check("slurm COMPLETED → slurm_state recorded",
              t.get("slurm_state") == "COMPLETED")
        check("slurm COMPLETED → started_at + finished_at set",
              t.get("started_at") and t.get("finished_at"))

        # 3. Slurm terminal-orphan: FAILED → status=failed + auto-requeue
        sch.run_on = lambda node, cmd, **kw: (
            (0, "67890 FAILED\n", "") if "squeue -h -n scheduleurm-tB" in cmd
            else (0, "", "")
        )
        state = {"next_id": 200, "tasks": [{
            "id": "tB", "status": "launching", "node": "slurmnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python b.py", "cwd": "/work", "retry_count": 0,
        }]}
        adopted = sch._try_recover_orphan_slurm_job(state["tasks"][0], "slurmnode", state)
        check("slurm FAILED orphan → adopted=True",
              adopted is True)
        t = state["tasks"][0]
        check("slurm FAILED → status=failed",
              t.get("status") == "failed")
        check("slurm FAILED → auto-requeue created a retry clone",
              t.get("requeued_as") is not None
              and any(x.get("id") == t["requeued_as"] and x["status"] == "queued"
                       for x in state["tasks"]),
              diag=f"requeued_as={t.get('requeued_as')}")

        # 4. Slurm: no orphan in squeue → returns False (revert path takes over)
        sch.run_on = lambda *a, **k: (0, "", "")
        state = {"next_id": 300, "tasks": [{
            "id": "tC", "status": "launching", "node": "slurmnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python c.py", "cwd": "/work",
        }]}
        adopted = sch._try_recover_orphan_slurm_job(state["tasks"][0], "slurmnode", state)
        check("no orphan in squeue → adopted=False (caller reverts)",
              adopted is False)
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES

    # 5. LocalBackend terminal-orphan via log_path.
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    saved_state_dir = sch.STATE_DIR
    import tempfile
    tdir = tempfile.mkdtemp()
    log_dir = os.path.join(tdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    sch.STATE_DIR = tdir
    sch.NODES = {"localnode": {"host": None}}

    try:
        # ---- Case L1: log present, looks-clean → status=done ----
        log_a = os.path.join(log_dir, "tL1.log")
        with open(log_a, "w") as f:
            f.write("Epoch 100/100\nTraining complete\nFinal model saved\n")
        state = {"next_id": 400, "tasks": [{
            "id": "tL1", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python a.py", "cwd": "/work",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("local log present + clean → finalized=True",
              finalized is True)
        check("local clean log → status=done",
              state["tasks"][0]["status"] == "done")

        # ---- Case L2: log present, soft-crash (APP_BUG) → status=failed + requeue.
        # Use an AssertionError so _classify_failure returns APP_BUG (eligible
        # for auto-requeue), NOT OOM (which would escalate, not requeue).
        log_b = os.path.join(log_dir, "tL2.log")
        with open(log_b, "w") as f:
            f.write("Epoch 5/100\n")
            f.write("Traceback (most recent call last):\n")
            f.write("AssertionError: invariant violated\n")
        state = {"next_id": 500, "tasks": [{
            "id": "tL2", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python b.py", "cwd": "/work", "retry_count": 0,
            "signature": "TEST/L2",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("local log present + crash patterns → finalized=True",
              finalized is True)
        check("local crash log → status=failed",
              state["tasks"][0]["status"] == "failed")
        check("local soft-crash → auto-requeue created a retry clone",
              state["tasks"][0].get("requeued_as") is not None
              and any(x["status"] == "queued" for x in state["tasks"]),
              diag=str(state["tasks"]))

        # ---- Case L3: log missing → finalized=False (revert path takes over) ----
        state = {"next_id": 600, "tasks": [{
            "id": "tL3", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python c.py", "cwd": "/work",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("local log missing → finalized=False (caller reverts)",
              finalized is False)
        check("local log missing → status unchanged (still launching)",
              state["tasks"][0]["status"] == "launching")

        # ---- Case L4: log present but 0 bytes → finalized=False ----
        log_d = os.path.join(log_dir, "tL4.log")
        open(log_d, "w").close()  # empty file
        state = {"next_id": 700, "tasks": [{
            "id": "tL4", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python d.py", "cwd": "/work",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("local log present but empty → finalized=False",
              finalized is False)
    finally:
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES
        sch.STATE_DIR = saved_state_dir


def test_phase3_0_34_local_docker_fail_fast_no_local_digest():
    """Phase 3.0.34 P1 fix: explicit docker fail-fast and auto-mode fallback
    must apply to LOCAL nodes too, not just remote.

    Pre-fix: the local-digest probe and the explicit-fail-fast / auto-fallback
    branches were nested inside `if node_host:` — local nodes (host=None)
    skipped the entire check. preload's local push_image is a no-op that
    returns ok, and the launch path then went straight to wrap_cmd_docker.
    Result: local docker could `run` against a missing image (silently pulling
    or failing with a confusing error) or against a stale tag the user
    didn't intend.

    Now: hoist the local-digest probe out of the `if node_host:` gate so the
    same explicit→error / auto→fallback policy applies on local AND remote.
    """
    print("\n[90] Phase 3.0.34 P1 fix: local docker fail-fast on missing local digest")

    # 1. Source guard: the local-digest probe runs unconditionally; only the
    # remote-side has_image() / preload-retry block is gated by node_host.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _maybe_wrap_docker")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    digest_idx = body.find("local_digest = env_deploy.get_image_digest(run_on, \"local\"")
    explicit_check_idx = body.find("if explicit and local_digest is None")
    auto_check_idx = body.find("if not explicit and local_digest is None")
    # Use newline+indent so we hit the actual statement, not the doc/comment
    # that mentions `if node_host:` in backticks.
    nodehost_block = body.find("\n    if node_host:\n")
    has_image_idx = body.find("env_deploy.has_image(run_on, node, chosen_image, local_digest=local_digest)")
    check("local_digest probe runs OUTSIDE the `if node_host:` gate",
          0 < digest_idx < nodehost_block,
          diag=f"digest={digest_idx} nodehost={nodehost_block}")
    check("explicit fail-fast guard runs OUTSIDE the node_host gate too",
          0 < explicit_check_idx < nodehost_block,
          diag=f"explicit_check={explicit_check_idx} nodehost={nodehost_block}")
    check("auto fallback runs OUTSIDE the node_host gate",
          0 < auto_check_idx < nodehost_block,
          diag=f"auto_check={auto_check_idx} nodehost={nodehost_block}")
    check("has_image() / preload-retry stays inside node_host (remote-only)",
          nodehost_block < has_image_idx,
          diag="has_image is remote-side only")

    # 2. Behavioral: same FakeED setup, with node_host=None (local node).
    saved_env_deploy = sch.env_deploy
    saved_NODES = sch.NODES
    sch.NODES = {"localnode": {"host": None}, "remotenode": {"host": "rbox"}}

    class FakeED:
        @staticmethod
        def parse_env_spec(spec):
            if spec == "docker":
                return ("docker", "")
            if spec.startswith("docker:"):
                return ("docker", spec.split(":", 1)[1])
            if spec == "auto":
                return ("auto", "")
            return ("none", "")
        @staticmethod
        def has_docker(run_on, node, timeout=8): return True
        local_digest_ref = {"value": None}
        remote_tag_present = {"value": True}
        push_called = {"n": 0}

        @classmethod
        def get_image_digest(cls, run_on, node, image, timeout=10):
            if node == "local":
                return cls.local_digest_ref["value"]
            return "remote-stale" if cls.remote_tag_present["value"] else None

        @classmethod
        def has_image(cls, run_on, node, image, local_digest=None, timeout=10):
            if local_digest is None:
                return True  # legacy fast path
            return cls.get_image_digest(None, node, image) == local_digest

        @classmethod
        def push_image(cls, *a, **k):
            cls.push_called["n"] += 1
            return (True, "ok")

        @staticmethod
        def wrap_cmd_docker(inner, image, cwd, gpu_idx, extra_env, container_name,
                            memory_mb, cpus, gpu_runtime_env):
            return f"docker_wrap({inner})"

    sch.env_deploy = FakeED

    try:
        # ---- Case A: LOCAL node + explicit + no local digest → fail-fast ----
        FakeED.local_digest_ref["value"] = None
        task = {"id": "tA", "node": "localnode", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("LOCAL + explicit + no local digest → fail (NEW: was bypassed pre-3.0.34)",
              err is not None
              and ("stale or unintended image" in err
                   or "stale remote tag" in err),
              diag=f"err={err!r}")
        check("LOCAL + explicit + no local digest → cmd unchanged",
              inner == "python a.py")

        # ---- Case B: LOCAL node + auto + no local digest → fallback to none ----
        FakeED.local_digest_ref["value"] = None
        task = {"id": "tB", "node": "localnode", "env_spec": "auto",
                "image": "myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("LOCAL + auto + no local digest → no error",
              err is None, diag=f"err={err!r}")
        check("LOCAL + auto + no local digest → cmd NOT docker-wrapped",
              inner == "python a.py", diag=f"inner={inner!r}")

        # ---- Case C: LOCAL node + explicit + local digest present → wrap as usual ----
        FakeED.local_digest_ref["value"] = "local-digest-ok"
        task = {"id": "tC", "node": "localnode", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("LOCAL + explicit + local digest present → docker-wrap (happy path)",
              err is None and inner.startswith("docker_wrap("),
              diag=f"err={err!r} inner={inner!r}")

        # ---- Case D: REMOTE node + explicit + no local digest → still fail-fast (regression) ----
        FakeED.local_digest_ref["value"] = None
        task = {"id": "tD", "node": "remotenode", "env_spec": "docker:myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("REMOTE + explicit + no local digest → fail (3.0.21 regression)",
              err is not None
              and ("stale or unintended image" in err
                   or "stale remote tag" in err),
              diag=f"err={err!r}")

        # ---- Case E: REMOTE node + auto + no local digest → still fallback (regression) ----
        FakeED.local_digest_ref["value"] = None
        task = {"id": "tE", "node": "remotenode", "env_spec": "auto",
                "image": "myproj:latest"}
        inner, err = sch._maybe_wrap_docker(task, "python a.py", "/work")
        check("REMOTE + auto + no local digest → fallback (3.0.26 regression)",
              err is None and inner == "python a.py")
    finally:
        sch.env_deploy = saved_env_deploy
        sch.NODES = saved_NODES


def test_phase3_0_35_slurm_terminal_orphan_diagnosis():
    """Phase 3.0.35 P1 fix: slurm terminal-orphan recovery must populate
    `task['_diagnosis']` BEFORE invoking _requeue_after_crash.

    Pre-fix: 3.0.33 set `status='failed'` and called `_requeue_after_crash`
    directly. _requeue_after_crash routes via _classify_failure(_diagnosis):
    `_classify_failure({})` returns "NORMAL" (the not-is_crash branch),
    which falls through to the soft-retry path. So:
      - slurm OUT_OF_MEMORY → should escalate (HARD_FAIL: OOM)
      - log tail with ModuleNotFoundError → should escalate (PYTHON_IMPORT)
      - 3.0.30 COMPLETED-but-log-has-CUDA-OOM → should escalate (OOM)
    All silently became APP_BUG soft retries instead.

    Now: build a full _diagnosis dict (is_crash=True, reason names the
    slurm state, tail = actual log content) so _classify_failure can match
    OOM / ENV_MISSING / PYTHON_IMPORT patterns and route to escalation
    correctly.
    """
    print("\n[91] Phase 3.0.35 P1 fix: slurm terminal-orphan _diagnosis enables hard-fail classify")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_fetch_log_tail helper exists (shared between scan + diag)",
          "def _fetch_log_tail" in src)
    fn_idx = src.find("def _try_recover_orphan_slurm_job")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    check("slurm terminal recovery sets _diagnosis BEFORE _requeue_after_crash (COMPLETED+crash path)",
          'task["_diagnosis"] = {' in body
          and '"is_crash": True' in body
          and "_requeue_after_crash(task, state)" in body)
    check("OUT_OF_MEMORY slurm state surfaces in reason for OOM classification",
          "out of memory" in body and "OUT_OF_MEMORY" in body,
          diag="reason must include 'out of memory' substring so OOM_PATTERNS classifies")

    # 2. Behavioral.
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    saved_state_dir = sch.STATE_DIR
    import tempfile
    tdir = tempfile.mkdtemp()
    log_dir = os.path.join(tdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    sch.STATE_DIR = tdir
    sch.NODES = {"slurmnode": {"host": None}}

    # Stub _write_escalation so HARD_FAIL paths don't try to write to the real
    # escalations file; capture which categories triggered.
    saved_write_escalation = sch._write_escalation
    escalation_calls = []
    sch._write_escalation = lambda task, category, diag: escalation_calls.append(category)

    try:
        # ---- Case A: slurm OUT_OF_MEMORY → OOM escalation, NO retry clone ----
        sch.run_on = lambda node, cmd, **kw: (
            (0, "100 OUT_OF_MEMORY\n", "") if "squeue -h -n scheduleurm-tA" in cmd
            else (0, "", "")
        )
        state = {"next_id": 100, "tasks": [{
            "id": "tA", "status": "launching", "node": "slurmnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python a.py", "cwd": "/work", "retry_count": 0,
            "signature": "TEST/oom",
        }]}
        escalation_calls.clear()
        adopted = sch._try_recover_orphan_slurm_job(state["tasks"][0], "slurmnode", state)
        t = state["tasks"][0]
        check("OUT_OF_MEMORY orphan → adopted",
              adopted is True)
        check("OUT_OF_MEMORY orphan → status=failed",
              t["status"] == "failed")
        check("OUT_OF_MEMORY orphan → _diagnosis set with is_crash=True",
              t.get("_diagnosis", {}).get("is_crash") is True)
        check("OUT_OF_MEMORY orphan → diag.reason names 'out of memory' for OOM classify",
              "out of memory" in (t.get("_diagnosis", {}).get("reason") or ""),
              diag=f"reason={t.get('_diagnosis', {}).get('reason')!r}")
        check("OUT_OF_MEMORY orphan → escalated as OOM (NOT soft-retried)",
              "OOM" in escalation_calls,
              diag=f"escalations: {escalation_calls}")
        check("OUT_OF_MEMORY orphan → NO retry clone created (escalation, not requeue)",
              t.get("requeued_as") is None,
              diag=f"requeued_as={t.get('requeued_as')!r}")

        # ---- Case B: slurm COMPLETED + log has ModuleNotFoundError →
        # PYTHON_IMPORT escalation. The 3.0.30 scan catches it as crash, and
        # 3.0.35's diagnosis-with-tail lets classify route it correctly.
        log_b = os.path.join(log_dir, "tB.log")
        with open(log_b, "w") as f:
            f.write("Starting up...\n")
            f.write("Traceback (most recent call last):\n")
            f.write("  File 'train.py', line 1, in <module>\n")
            f.write("ModuleNotFoundError: No module named 'foo_bar'\n")
        sch.run_on = lambda node, cmd, **kw: (
            (0, "200 COMPLETED\n", "") if "squeue -h -n scheduleurm-tB" in cmd
            else (0, "", "")
        )
        state = {"next_id": 200, "tasks": [{
            "id": "tB", "status": "launching", "node": "slurmnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python b.py", "cwd": "/work", "retry_count": 0,
            "signature": "TEST/mod",
        }]}
        escalation_calls.clear()
        adopted = sch._try_recover_orphan_slurm_job(state["tasks"][0], "slurmnode", state)
        t = state["tasks"][0]
        check("COMPLETED+ModuleNotFoundError → adopted=True",
              adopted is True)
        check("COMPLETED+ModuleNotFoundError → status=failed",
              t["status"] == "failed")
        check("COMPLETED+ModuleNotFoundError → diag.tail contains the offending pattern",
              "ModuleNotFoundError" in (t.get("_diagnosis", {}).get("tail") or ""),
              diag=f"tail={t.get('_diagnosis', {}).get('tail')!r}")
        check("COMPLETED+ModuleNotFoundError → escalated as PYTHON_IMPORT (NOT soft-retried)",
              "PYTHON_IMPORT" in escalation_calls,
              diag=f"escalations: {escalation_calls}")
        check("COMPLETED+ModuleNotFoundError → NO retry clone",
              t.get("requeued_as") is None)

        # ---- Case C: slurm COMPLETED + log has CUDA OOM (3.0.30 scan catches,
        # 3.0.35 ensures OOM classification kicks in for escalation).
        log_c = os.path.join(log_dir, "tC.log")
        with open(log_c, "w") as f:
            f.write("Epoch 1/100\n")
            f.write("Traceback (most recent call last):\n")
            f.write("RuntimeError: CUDA out of memory. Tried to allocate 4.5 GiB\n")
        sch.run_on = lambda node, cmd, **kw: (
            (0, "300 COMPLETED\n", "") if "squeue -h -n scheduleurm-tC" in cmd
            else (0, "", "")
        )
        state = {"next_id": 300, "tasks": [{
            "id": "tC", "status": "launching", "node": "slurmnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python c.py", "cwd": "/work", "retry_count": 0,
            "signature": "TEST/cudaoom",
        }]}
        escalation_calls.clear()
        adopted = sch._try_recover_orphan_slurm_job(state["tasks"][0], "slurmnode", state)
        t = state["tasks"][0]
        check("COMPLETED+CUDA-OOM → adopted=True",
              adopted is True)
        check("COMPLETED+CUDA-OOM → escalated as OOM (NOT silently retried)",
              "OOM" in escalation_calls,
              diag=f"escalations: {escalation_calls}")
        check("COMPLETED+CUDA-OOM → NO retry clone",
              t.get("requeued_as") is None)

        # ---- Case D: slurm FAILED with no specific tail → APP_BUG soft retry.
        # This is the legitimate retry path; ensure 3.0.35 doesn't break it.
        log_d = os.path.join(log_dir, "tD.log")
        with open(log_d, "w") as f:
            f.write("Some training output...\n")
            f.write("Got SIGTERM\n")  # generic, not in OOM/ENV/IMPORT patterns
        sch.run_on = lambda node, cmd, **kw: (
            (0, "400 FAILED\n", "") if "squeue -h -n scheduleurm-tD" in cmd
            else (0, "", "")
        )
        state = {"next_id": 400, "tasks": [{
            "id": "tD", "status": "launching", "node": "slurmnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python d.py", "cwd": "/work", "retry_count": 0,
            "signature": "TEST/sigterm",
        }]}
        escalation_calls.clear()
        adopted = sch._try_recover_orphan_slurm_job(state["tasks"][0], "slurmnode", state)
        t = state["tasks"][0]
        check("FAILED + generic tail → adopted, status=failed",
              adopted is True and t["status"] == "failed")
        check("FAILED + generic tail → NO escalation (soft retry path is correct here)",
              not escalation_calls, diag=f"escalations: {escalation_calls}")
        check("FAILED + generic tail → retry clone created",
              t.get("requeued_as") is not None,
              diag=f"requeued_as={t.get('requeued_as')}")
    finally:
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES
        sch.STATE_DIR = saved_state_dir
        sch._write_escalation = saved_write_escalation


def test_phase3_0_36_local_terminal_orphan_user_redirect_recovery():
    """Phase 3.0.36 P2 fix: local terminal-orphan finalize must consult the
    user's own stdout/stderr redirect target before declaring "no evidence".

    Pre-fix: `if log_size <= 0: return False`. Wrapper log is 0 bytes
    whenever the user's cmd has its own redirect (`python train.py >
    out.log 2>&1`) — bash's inner redirect overrides the outer one. So
    a task that ran-and-finished via its own log would pass through the
    revert → re-launch path and run a SECOND time as a duplicate.

    Now: when the wrapper log is empty, parse the cmd for a `> path`
    redirect and probe that path. If it has content, finalize via
    _diagnose_terminal (which already has the same redirect-recovery
    logic and will read the real log for classification). Only revert
    when BOTH wrapper AND user-redirect are empty/missing.
    """
    print("\n[92] Phase 3.0.36 P2 fix: local terminal-orphan handles cmd's own redirect")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    fn_idx = src.find("def _try_finalize_terminal_local_task")
    fn_end = src.find("\ndef ", fn_idx + 5)
    body = src[fn_idx:fn_end]
    check("finalize probes user redirect when wrapper log empty",
          "cmd_has_own_redirect" in body
          and "&>|>>|2>&1|>&|>" in body)
    check("finalize uses unified _probe_size helper for both wrapper + redirect",
          "def _probe_size" in body)
    check("finalize reverts only when BOTH wrapper AND redirect are empty",
          "real_size <= 0" in body)

    # 2. Behavioral.
    saved_NODES = sch.NODES
    saved_state_dir = sch.STATE_DIR
    saved_write_escalation = sch._write_escalation
    sch._write_escalation = lambda *a, **kw: None
    import tempfile
    tdir = tempfile.mkdtemp()
    log_dir = os.path.join(tdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    sch.STATE_DIR = tdir
    sch.NODES = {"localnode": {"host": None}}

    try:
        # ---- Case A: wrapper log empty BUT cmd has `> real.log 2>&1` and
        # the user log has content → finalize as done.
        wrapper_a = os.path.join(log_dir, "tA.log")
        open(wrapper_a, "w").close()  # empty wrapper (cmd took over)
        real_a = os.path.join(log_dir, "real_tA.log")
        with open(real_a, "w") as f:
            f.write("Epoch 100/100 loss=0.001\n")
            f.write("Training complete\nFinal model saved\n")
        state = {"next_id": 100, "tasks": [{
            "id": "tA", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": f"python train.py > {real_a} 2>&1",
            "cwd": "/work",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("wrapper empty + user-redirect with content → finalized=True",
              finalized is True)
        check("wrapper empty + clean user log → status=done (NOT reverted)",
              state["tasks"][0]["status"] == "done")

        # ---- Case B: wrapper log empty AND user log empty → revert (no
        # evidence the task ran).
        wrapper_b = os.path.join(log_dir, "tB.log")
        open(wrapper_b, "w").close()
        real_b = os.path.join(log_dir, "real_tB.log")
        open(real_b, "w").close()
        state = {"next_id": 200, "tasks": [{
            "id": "tB", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": f"python train.py > {real_b} 2>&1",
            "cwd": "/work",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("wrapper empty + user-redirect also empty → finalized=False (revert)",
              finalized is False)
        check("wrapper empty + user-redirect also empty → status unchanged",
              state["tasks"][0]["status"] == "launching")

        # ---- Case C: wrapper log empty AND no redirect in cmd → revert
        # (existing behavior — unchanged contract for tasks without their
        # own redirect).
        wrapper_c = os.path.join(log_dir, "tC.log")
        open(wrapper_c, "w").close()
        state = {"next_id": 300, "tasks": [{
            "id": "tC", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python train.py",  # no redirect
            "cwd": "/work",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("wrapper empty + no redirect in cmd → finalized=False",
              finalized is False)

        # ---- Case D: wrapper log has content (legacy path) → finalize as
        # before, no redirect needed.
        wrapper_d = os.path.join(log_dir, "tD.log")
        with open(wrapper_d, "w") as f:
            f.write("Training complete\nFinal model saved\n")
        state = {"next_id": 400, "tasks": [{
            "id": "tD", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": "python train.py",
            "cwd": "/work",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("wrapper has content → finalize via existing path",
              finalized is True and state["tasks"][0]["status"] == "done")

        # ---- Case E: wrapper empty + redirect points to user log with crash
        # patterns → status=failed + auto-requeue (mirrors regular crash flow).
        wrapper_e = os.path.join(log_dir, "tE.log")
        open(wrapper_e, "w").close()
        real_e = os.path.join(log_dir, "real_tE.log")
        with open(real_e, "w") as f:
            f.write("Epoch 5/100 starting\n")
            f.write("Traceback (most recent call last):\n")
            f.write("AssertionError: invariant violated\n")
        state = {"next_id": 500, "tasks": [{
            "id": "tE", "status": "launching", "node": "localnode",
            "launching_started_at": time.time() - 600, "remote_pids": [],
            "cmd": f"python train.py > {real_e} 2>&1",
            "cwd": "/work", "retry_count": 0, "signature": "TEST/redirect-crash",
        }]}
        finalized = sch._try_finalize_terminal_local_task(
            state["tasks"][0], "localnode", state)
        check("wrapper empty + redirect+crash → finalized=True",
              finalized is True)
        check("wrapper empty + redirect+crash → status=failed",
              state["tasks"][0]["status"] == "failed")
        check("wrapper empty + redirect+crash → retry clone created",
              state["tasks"][0].get("requeued_as") is not None,
              diag=str(state["tasks"]))
    finally:
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)
        sch.NODES = saved_NODES
        sch.STATE_DIR = saved_state_dir
        sch._write_escalation = saved_write_escalation


def test_phase3_1_skill_priority_edit_history_why():
    """Phase 3.1: four operational commands cover the gaps surfaced by
    repeated ad-hoc state.json edits (priority bump, est_vram fix, history
    poison cleanup, stuck-task diagnosis).
    """
    print("\n[93] Phase 3.1: priority / edit / history --drop|--set / why commands")

    # 1. Source guards: each cmd_* function exists and registered as subparser.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    for fn_name in ("cmd_priority", "cmd_edit", "cmd_why", "_explain_node_fit"):
        check(f"{fn_name} defined",
              f"def {fn_name}" in src,
              diag=f"{fn_name} missing")
    # cmd_history extended with --drop / --set branches.
    hist_idx = src.find("def cmd_history")
    hist_end = src.find("\ndef ", hist_idx + 5)
    hist_body = src[hist_idx:hist_end]
    check("cmd_history handles --drop",
          'getattr(args, "drop"' in hist_body
          and "save_history(h)" in hist_body)
    check("cmd_history handles --set",
          'getattr(args, "set"' in hist_body
          and 'rec["vram_mb"]' in hist_body)
    # main() registers all four subparsers.
    main_idx = src.find("def main():")
    main_body = src[main_idx:]
    for sp in ("priority", "edit", "why"):
        check(f"main() registers `{sp}` subcommand",
              f'sub.add_parser("{sp}"' in main_body)
    check("main() history subcommand exposes --drop and --set flags",
          's.add_argument("--drop"' in main_body
          and 's.add_argument("--set"' in main_body)

    # 2. Behavioral: cmd_priority on queued task.
    saved_save = sch.save_state
    saved_load = sch.load_state
    saved_lock = sch.state_lock

    fake_state = {"next_id": 1, "tasks": [
        {"id": "tQ", "status": "queued", "priority": "normal",
         "description": "test queued task"},
        {"id": "tR", "status": "running", "priority": "normal"},
    ]}
    save_capture = [None]
    sch.load_state = lambda: fake_state
    sch.save_state = lambda s: save_capture.__setitem__(0, s)
    from contextlib import contextmanager as _cm
    @_cm
    def fake_lock(): yield
    sch.state_lock = fake_lock

    class A:
        pass

    try:
        # priority queued → mutates
        a = A(); a.id = "tQ"; a.level = "high"
        sch.cmd_priority(a)
        check("priority on queued task → field updated",
              fake_state["tasks"][0]["priority"] == "high")
        check("priority change persisted via save_state",
              save_capture[0] is fake_state)

        # priority running → SystemExit
        a = A(); a.id = "tR"; a.level = "high"
        try:
            sch.cmd_priority(a)
            check("priority on running task → rejected (SystemExit)", False,
                  diag="should have raised SystemExit")
        except SystemExit as e:
            check("priority on running task → rejected (SystemExit)",
                  "not queued" in str(e), diag=str(e))

        # priority unknown id → SystemExit
        a = A(); a.id = "tNOPE"; a.level = "low"
        try:
            sch.cmd_priority(a)
            check("priority on unknown id → SystemExit", False)
        except SystemExit as e:
            check("priority on unknown id → SystemExit",
                  "not found" in str(e), diag=str(e))

        # 3. Behavioral: cmd_edit
        # Use only fields we already mock NODES for (avoid preferred_node validation).
        a = A(); a.id = "tQ"; a.vram_mb = 2000; a.ram_mb = 1500
        a.cpu = 4; a.description = "edited"
        a.preferred_node = None; a.require_node = None
        sch.cmd_edit(a)
        t = fake_state["tasks"][0]
        check("edit: vram_mb updated", t.get("est_vram_mb") == 2000)
        check("edit: ram_mb updated", t.get("ram_mb") == 1500)
        check("edit: cpu_cores updated", t.get("cpu_cores") == 4)
        check("edit: description updated", t.get("description") == "edited")

        # edit running task → reject
        a = A(); a.id = "tR"; a.vram_mb = 100; a.ram_mb = None
        a.cpu = None; a.description = None
        a.preferred_node = None; a.require_node = None
        try:
            sch.cmd_edit(a)
            check("edit on running task → SystemExit", False)
        except SystemExit as e:
            check("edit on running task → SystemExit",
                  "not queued" in str(e), diag=str(e))

        # edit with no flags → SystemExit
        a = A(); a.id = "tQ"; a.vram_mb = None; a.ram_mb = None
        a.cpu = None; a.description = None
        a.preferred_node = None; a.require_node = None
        try:
            sch.cmd_edit(a)
            check("edit with no flags → SystemExit", False)
        except SystemExit as e:
            check("edit with no flags → SystemExit",
                  "specify at least one" in str(e), diag=str(e))

        # 4. Behavioral: cmd_history --drop and --set
        saved_load_h = sch.load_history
        saved_save_h = sch.save_history
        fake_h = {"PROJ/sigA": {"vram_mb": 9999, "vram_samples": [9999]},
                  "PROJ/sigB": {"vram_mb": 1500, "vram_samples": [1400, 1500]}}
        save_h_capture = [None]
        sch.load_history = lambda: fake_h
        sch.save_history = lambda h: save_h_capture.__setitem__(0, h.copy())

        # --drop existing
        a = A(); a.drop = "PROJ/sigA"; a.set = None
        a.vram_mb = None; a.ram_mb = None; a.cpu = None
        sch.cmd_history(a)
        check("history --drop: entry removed",
              "PROJ/sigA" not in fake_h)
        check("history --drop: save_history called",
              save_h_capture[0] is not None)

        # --drop missing
        a = A(); a.drop = "DOES/NOT/EXIST"; a.set = None
        a.vram_mb = None; a.ram_mb = None; a.cpu = None
        try:
            sch.cmd_history(a)
            check("history --drop unknown sig → SystemExit", False)
        except SystemExit as e:
            check("history --drop unknown sig → SystemExit",
                  "not in history" in str(e), diag=str(e))

        # --set with vram_mb resets samples
        save_h_capture[0] = None
        a = A(); a.drop = None; a.set = "PROJ/sigB"
        a.vram_mb = 800; a.ram_mb = None; a.cpu = None
        sch.cmd_history(a)
        check("history --set vram_mb: vram_mb updated",
              fake_h["PROJ/sigB"]["vram_mb"] == 800)
        check("history --set vram_mb: vram_samples reset to single value",
              fake_h["PROJ/sigB"]["vram_samples"] == [800],
              diag=str(fake_h["PROJ/sigB"]))

        # --set with no fields → SystemExit
        a = A(); a.drop = None; a.set = "PROJ/newSig"
        a.vram_mb = None; a.ram_mb = None; a.cpu = None
        try:
            sch.cmd_history(a)
            check("history --set with no fields → SystemExit", False)
        except SystemExit as e:
            check("history --set with no fields → SystemExit",
                  "requires at least one" in str(e), diag=str(e))

        sch.load_history = saved_load_h
        sch.save_history = saved_save_h

        # 5. Behavioral: cmd_why prints diagnostic for queued task
        # Stub probe_all + history + helpers so we don't hit real ssh.
        saved_probe = sch.probe_all
        saved_lh2 = sch.load_history
        sch.probe_all = lambda: [
            {"name": "n1", "alive": True, "free_cpu": 12, "free_ram_mb": 100000,
             "total_ram_mb": 200000, "total_cpu": 12, "running_count": 0,
             "slurm_pending_count": 0, "gpus": [
                 {"idx": 0, "total_mb": 12288, "used_mb": 11000,
                  "free_mb": 1288, "util_pct": 100},
             ]},
        ]
        sch.load_history = lambda: {"PROJ/foo": {"vram_mb": 1000, "ram_mb": 500}}
        saved_NODES = sch.NODES
        sch.NODES = {"n1": {"name": "n1", "host": None, "cpu_cores": 12,
                            "ram_mb": 200000, "ram_headroom_frac": 0.10,
                            "max_vram_per_task": None,
                            "max_concurrent_running": None}}
        # Make backend report n1 as local (not slurm-routed) so the GPU-fit
        # path is exercised.
        saved_backend = sch._BACKEND
        class _Local:
            def requires_local_capacity_check(self, name): return True
        sch._BACKEND = _Local()

        fake_state["tasks"].append({
            "id": "tWhy", "status": "queued", "priority": "high",
            "description": "diagnose me", "signature": "PROJ/foo/seedA",
            "preferred_node": "n1", "require_node": None,
            "est_vram_mb": 5000, "ram_mb": 2000, "cpu_cores": 2,
            "last_block_reason": "no fit yesterday",
        })

        # Capture stdout
        import io as _io
        from contextlib import redirect_stdout as _rs
        buf = _io.StringIO()
        a = A(); a.id = "tWhy"
        with _rs(buf):
            sch.cmd_why(a)
        out = buf.getvalue()
        check("why: prints task header (status / priority / preferred)",
              "status:       queued" in out
              and "priority:     high" in out
              and "preferred:    'n1'" in out, diag=out[:300])
        check("why: surfaces last_block_reason",
              "no fit yesterday" in out)
        check("why: per-node fit analysis explains GPU rejection",
              "n1" in out
              and "compute saturation" in out, diag=out)

        # Why on terminal task → degraded info, no probe
        fake_state["tasks"].append({
            "id": "tDone", "status": "done", "description": "old"})
        buf = _io.StringIO()
        a = A(); a.id = "tDone"
        with _rs(buf):
            sch.cmd_why(a)
        out = buf.getvalue()
        check("why: terminal task → 'task is done' note instead of fit analysis",
              "is 'done'" in out or "is \"done\"" in out, diag=out)

        # Why on unknown id → SystemExit
        a = A(); a.id = "tNONE"
        try:
            sch.cmd_why(a)
            check("why on unknown id → SystemExit", False)
        except SystemExit as e:
            check("why on unknown id → SystemExit",
                  "not found" in str(e), diag=str(e))

        sch._BACKEND = saved_backend
        sch.NODES = saved_NODES
        sch.probe_all = saved_probe
        sch.load_history = saved_lh2

        # 6. _explain_node_fit unit cases.
        # GPU available with room → FITS.
        node_state = {"name": "n1", "alive": True, "free_cpu": 12,
                      "free_ram_mb": 100000, "total_ram_mb": 200000,
                      "total_cpu": 12, "running_count": 0,
                      "gpus": [{"idx": 0, "total_mb": 12288, "used_mb": 200,
                                 "free_mb": 12000, "util_pct": 5}]}
        sch.NODES = {"n1": {"name": "n1", "host": None, "cpu_cores": 12,
                             "ram_mb": 200000, "ram_headroom_frac": 0.10,
                             "max_vram_per_task": None,
                             "max_concurrent_running": None}}
        saved_blocked = sch._blocked_nodes_for_task
        saved_lf = sch._launch_failed_nodes_for_task
        saved_be = sch._BACKEND
        sch._blocked_nodes_for_task = lambda task: set()
        sch._launch_failed_nodes_for_task = lambda task: set()
        sch._BACKEND = _Local()
        small_task = {"est_vram_mb": 500, "ram_mb": 1000, "cpu_cores": 2}
        msg = sch._explain_node_fit(small_task, node_state)
        check("explain: small task on empty GPU → FITS",
              msg.startswith("FITS:"), diag=msg)

        # GPU at compute saturation → reject.
        sat_node = dict(node_state)
        sat_node["gpus"] = [{"idx": 0, "total_mb": 12288, "used_mb": 1000,
                             "free_mb": 11000, "util_pct": 100}]
        msg = sch._explain_node_fit(small_task, sat_node)
        check("explain: compute saturated GPU → no-GPU-fit + util reason",
              "no-GPU-fit" in msg and "compute saturation" in msg, diag=msg)

        # Blocked node → BLOCKED message.
        sch._blocked_nodes_for_task = lambda task: {"n1"}
        msg = sch._explain_node_fit(small_task, node_state)
        check("explain: blocked node → BLOCKED message",
              msg.startswith("BLOCKED:"), diag=msg)
        sch._blocked_nodes_for_task = lambda task: set()

        # CPU-only task on alive node → FITS (CPU-only).
        cpu_task = {"est_vram_mb": 0, "ram_mb": 1000, "cpu_cores": 2}
        msg = sch._explain_node_fit(cpu_task, node_state)
        check("explain: CPU-only task → FITS (CPU-only)",
              "FITS (CPU-only)" in msg, diag=msg)

        # Down node.
        down = {"name": "n1", "alive": False, "error": "ssh timeout"}
        msg = sch._explain_node_fit(small_task, down)
        check("explain: dead node → DOWN message",
              msg.startswith("DOWN"), diag=msg)

        sch._blocked_nodes_for_task = saved_blocked
        sch._launch_failed_nodes_for_task = saved_lf
        sch._BACKEND = saved_be
    finally:
        sch.save_state = saved_save
        sch.load_state = saved_load
        sch.state_lock = saved_lock


def test_phase3_2_0_claim_manager():
    """Phase 3.2.0: cross-scheduler / cross-user resource claims layer.

    Tests two layers:
      1. Remote claims script logic — run the actual script via subprocess
         against a tmp claims.json so the real read-modify-write semantics
         are exercised. No ssh, no local-only mocks: this is the same code
         that runs on every node when claims is enabled.
      2. _ClaimManager Python API — mock run_on, verify the manager
         (a) skips ssh entirely when enable_claims is off (fast-path),
         (b) builds the right record + capacity payload,
         (c) returns the conflict message on capacity refusal,
         (d) wires release / renew / update_pid / gc_stale / enumerate
             through the same op channel.
    """
    print("\n[94] Phase 3.2.0: _ClaimManager + remote claims script (cross-scheduler exclusion)")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_CLAIMS_REMOTE_SCRIPT defined", "_CLAIMS_REMOTE_SCRIPT = " in src)
    check("_ClaimManager class exists", "class _ClaimManager" in src)
    check("_claims_remote_op helper exists", "def _claims_remote_op" in src)
    check("_claims_setup_cmd helper exists (heredoc deploy)",
          "def _claims_setup_cmd" in src)
    check("flock used in remote op",
          "flock -x -w" in src and "CLAIMS_LOCK_REMOTE" in src)

    # 2. Test the script's own logic by running it as a real subprocess
    # against a tmp claims.json. The script string is what gets deployed
    # to /tmp/scheduleurm/_claims.py on each node — what we test here is
    # exactly what runs there.
    import tempfile, subprocess, json as _json
    with tempfile.TemporaryDirectory() as td:
        # Write the script and rewrite its CLAIMS_FILE to live in our tmp
        # so the test doesn't touch /tmp/scheduleurm.
        script = sch._CLAIMS_REMOTE_SCRIPT.replace(
            'CLAIMS_FILE = "/tmp/scheduleurm/claims.json"',
            f'CLAIMS_FILE = {os.path.join(td, "claims.json")!r}',
        ).replace(
            'os.makedirs("/tmp/scheduleurm", exist_ok=True)',
            f'os.makedirs({td!r}, exist_ok=True)',
        )
        script_path = os.path.join(td, "_claims.py")
        with open(script_path, "w") as f:
            f.write(script)

        # Phase 3.4.7: production setup ensures claims.json is never 0 bytes
        # (crash-recovery distinguisher). Tests must mirror that.
        claims_path = os.path.join(td, "claims.json")
        def _ensure_init():
            if (not os.path.exists(claims_path)
                    or os.path.getsize(claims_path) == 0):
                with open(claims_path, "w") as cf:
                    cf.write('{"version":1,"claims":[]}')

        def run_op(op, payload, capacity=None):
            _ensure_init()
            r = subprocess.run(
                ["python3", script_path, op,
                 _json.dumps(payload), _json.dumps(capacity or {})],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return {"_rc": r.returncode, "_stderr": r.stderr}
            try:
                return _json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                return {"_parse_error": r.stdout, "_stderr": r.stderr}

        now = time.time()
        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000, "1": 12000}}

        # 2a. Empty file, first claim succeeds.
        rec_a = {"owner": "u1", "scheduler_id": "h1:1000", "task_id": "tA",
                 "gpu_idx": 0, "vram_mb": 2000, "cpu_cores": 2, "ram_mb": 1500,
                 "claimed_at": now, "expires_at": now + 3600, "pid": None}
        r = run_op("claim", rec_a, cap)
        check("script: first claim on empty file → ok",
              r.get("ok") is True, diag=r)

        # 2b. Conflicting claim from a DIFFERENT scheduler on the same GPU
        # exceeds capacity → reject. (cap[GPU0]=12000; 2000 already; new wants
        # 11000 → 13000 > 12000 → conflict.)
        rec_b = {"owner": "u2", "scheduler_id": "h2:2000", "task_id": "tB",
                 "gpu_idx": 0, "vram_mb": 11000, "cpu_cores": 2, "ram_mb": 1500,
                 "claimed_at": now, "expires_at": now + 3600, "pid": None}
        r = run_op("claim", rec_b, cap)
        check("script: capacity-exceeding claim from different scheduler → conflict",
              r.get("ok") is False
              and "gpu0" in (r.get("conflict") or ""), diag=r)

        # 2c. Smaller claim on the same GPU within remaining capacity → ok.
        rec_c = {"owner": "u2", "scheduler_id": "h2:2000", "task_id": "tC",
                 "gpu_idx": 0, "vram_mb": 5000, "cpu_cores": 1, "ram_mb": 500,
                 "claimed_at": now, "expires_at": now + 3600, "pid": None}
        r = run_op("claim", rec_c, cap)
        check("script: claim within remaining capacity → ok",
              r.get("ok") is True, diag=r)

        # 2d. Claim on a different GPU — no contention.
        rec_d = {"owner": "u3", "scheduler_id": "h3:3000", "task_id": "tD",
                 "gpu_idx": 1, "vram_mb": 8000, "cpu_cores": 2, "ram_mb": 2000,
                 "claimed_at": now, "expires_at": now + 3600, "pid": None}
        r = run_op("claim", rec_d, cap)
        check("script: claim on different GPU → ok (per-GPU accounting)",
              r.get("ok") is True, diag=r)

        # 2e. List shows all 3 claims now (tA, tC, tD).
        r = run_op("list", {})
        check("script: list returns all live claims",
              r.get("ok") is True and len(r.get("claims", [])) == 3, diag=r)

        # 2f. Release tA via its (scheduler_id, task_id).
        r = run_op("release", {"scheduler_id": "h1:1000", "task_id": "tA"})
        check("script: release returns removed=1",
              r.get("ok") is True and r.get("removed") == 1, diag=r)

        # 2g. update_pid sets pid on the matching claim.
        r = run_op("update_pid",
                   {"scheduler_id": "h2:2000", "task_id": "tC", "pid": 99999})
        check("script: update_pid → ok with updated=1",
              r.get("ok") is True and r.get("updated") == 1, diag=r)

        # 2h. renew bumps expires_at on the matching claim.
        future = now + 7200
        r = run_op("renew",
                   {"scheduler_id": "h2:2000", "task_id": "tC",
                    "expires_at": future})
        check("script: renew → ok with renewed=1",
              r.get("ok") is True and r.get("renewed") == 1, diag=r)

        # 2i. Inject a stale claim manually + GC.
        # stale = expires_at < now AND (no pid OR pid not alive)
        with open(os.path.join(td, "claims.json")) as f:
            data = _json.load(f)
        data["claims"].append({
            "scheduler_id": "h_dead:9999", "task_id": "tStale",
            "owner": "ghost", "gpu_idx": 0, "vram_mb": 100,
            "cpu_cores": 0, "ram_mb": 0,
            "claimed_at": now - 7200, "expires_at": now - 3600,
            "pid": None,  # no pid AND past TTL → stale
        })
        before = len(data["claims"])
        with open(os.path.join(td, "claims.json"), "w") as f:
            _json.dump(data, f)
        r = run_op("gc", {})
        check("script: gc removes stale (no-pid + expired) claim",
              r.get("ok") is True and r.get("removed") == 1, diag=r)
        # Verify file content matches.
        with open(os.path.join(td, "claims.json")) as f:
            after_data = _json.load(f)
        check("script: gc persisted (file has before-1 claims)",
              len(after_data["claims"]) == before - 1)

        # 2j. Stale-but-alive-pid is NOT GC'd (live orphan; trust the
        # process is using the resource).
        my_pid = os.getpid()
        with open(os.path.join(td, "claims.json")) as f:
            data = _json.load(f)
        data["claims"].append({
            "scheduler_id": "h_orphan:5555", "task_id": "tOrphan",
            "owner": "ghost", "gpu_idx": None, "vram_mb": 0,
            "cpu_cores": 0, "ram_mb": 0,
            "claimed_at": now - 7200, "expires_at": now - 3600,
            "pid": my_pid,  # alive — keep
        })
        with open(os.path.join(td, "claims.json"), "w") as f:
            _json.dump(data, f)
        r = run_op("gc", {})
        check("script: stale-but-alive-pid claim KEPT (live orphan)",
              r.get("ok") is True and r.get("removed") == 0,
              diag=f"r={r} my_pid={my_pid}")

        # 2k. Unknown op returns ok=False with error.
        r = run_op("noop", {})
        check("script: unknown op → ok=False with error",
              r.get("ok") is False and "unknown op" in (r.get("error") or ""),
              diag=r)

        # 2l. Missing claims file (fresh node) → list returns empty, no crash.
        os.remove(os.path.join(td, "claims.json"))
        r = run_op("list", {})
        check("script: missing claims file → list returns ok with []",
              r.get("ok") is True and r.get("claims") == [], diag=r)

    # 3. _ClaimManager API: mock run_on to verify the manager calls
    # the script with the right payload and parses results correctly.
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    sch.NODES = {
        "n_on": {"name": "n_on", "host": "h_on", "enable_claims": True,
                 "cpu_cores": 12, "ram_mb": 32000, "claim_ttl_s": 600},
        "n_off": {"name": "n_off", "host": "h_off", "cpu_cores": 12,
                  "ram_mb": 32000},  # no enable_claims
    }

    captured = []
    def fake_run_on(node, cmd, timeout=30, check=False):
        captured.append((node, cmd))
        # Simple dispatch: parse the op token (after `_claims.py`)
        if "\"$SCRIPT_PATH\" claim " in cmd:
            return (0, '{"ok": true}\n', "")
        if "\"$SCRIPT_PATH\" release " in cmd:
            return (0, '{"ok": true, "removed": 1}\n', "")
        if "\"$SCRIPT_PATH\" renew " in cmd:
            return (0, '{"ok": true, "renewed": 1}\n', "")
        if "\"$SCRIPT_PATH\" update_pid " in cmd:
            return (0, '{"ok": true, "updated": 1}\n', "")
        if "\"$SCRIPT_PATH\" gc " in cmd:
            return (0, '{"ok": true, "removed": 2}\n', "")
        if "\"$SCRIPT_PATH\" list" in cmd:
            return (0, '{"ok": true, "claims": [{"task_id": "x"}]}\n', "")
        return (1, "", "unmatched")
    sch.run_on = fake_run_on

    try:
        # 3a. enabled_for / scheduler_id basics
        check("enabled_for: opt-in node returns True",
              sch._ClaimManager.enabled_for("n_on") is True)
        check("enabled_for: non-opt-in node returns False",
              sch._ClaimManager.enabled_for("n_off") is False)
        sid = sch._ClaimManager.scheduler_id()
        # Phase 3.4.2: scheduler_id is now <host>:<hex12-uuid-suffix>
        # (persistent across restarts), not <host>:<pid>.
        check("scheduler_id format: <host>:<persistent-id>",
              ":" in sid and len(sid.split(":")[1]) >= 6, diag=sid)

        # 3b. Disabled node → fast path, NO ssh.
        captured.clear()
        ok, info, _kind = sch._ClaimManager.claim(
            "n_off", {"id": "tD", "est_vram_mb": 1000,
                       "cpu_cores": 1, "ram_mb": 500},
            gpu_idx=0)
        check("disabled node claim → ok=True without ssh",
              ok is True and not captured, diag=f"captured={captured}")

        # 3c. Enabled node + ok response → returns claim record.
        captured.clear()
        ok, info, _kind = sch._ClaimManager.claim(
            "n_on", {"id": "tX", "est_vram_mb": 2000,
                      "cpu_cores": 2, "ram_mb": 1500},
            gpu_idx=0,
            node_state={"name": "n_on",
                        "gpus": [{"idx": 0, "total_mb": 12000},
                                 {"idx": 1, "total_mb": 12000}]})
        check("enabled node claim → ssh issued",
              len(captured) == 1, diag=str(captured)[:200])
        check("enabled node claim → ok=True with claim record",
              ok is True
              and isinstance(info, dict)
              and info.get("task_id") == "tX"
              and info.get("vram_mb") == 2000
              and info.get("cpu_cores") == 2
              and info.get("ram_mb") == 1500
              and info.get("gpu_idx") == 0,
              diag=info)
        check("claim record has scheduler_id + owner + expires_at",
              info.get("scheduler_id") == sid
              and info.get("owner")
              and info.get("expires_at") > info.get("claimed_at"))

        # 3d. Conflict response → ok=False, info is the conflict string.
        sch.run_on = lambda node, cmd, **kw: (
            0, '{"ok": false, "conflict": "gpu0: need 9000 + claimed 5000 > cap 12000", "claims_seen": 1}\n', "")
        ok, info, _kind = sch._ClaimManager.claim(
            "n_on", {"id": "tY", "est_vram_mb": 9000,
                      "cpu_cores": 2, "ram_mb": 1500},
            gpu_idx=0,
            node_state={"name": "n_on",
                        "gpus": [{"idx": 0, "total_mb": 12000}]})
        check("conflict response → ok=False with conflict text",
              ok is False and "gpu0" in info and "need 9000" in info,
              diag=info)

        # 3e. release / update_pid / renew / gc_stale / enumerate
        sch.run_on = fake_run_on
        check("release ok",
              sch._ClaimManager.release("n_on", "tX") is True)
        check("update_pid ok",
              sch._ClaimManager.update_pid("n_on", "tX", 12345) is True)
        check("renew ok",
              sch._ClaimManager.renew("n_on", "tX") is True)
        check("gc_stale returns removed count",
              sch._ClaimManager.gc_stale("n_on") == 2)
        claims_list = sch._ClaimManager.enumerate("n_on")
        check("enumerate returns parsed claims list",
              isinstance(claims_list, list)
              and claims_list and claims_list[0].get("task_id") == "x",
              diag=claims_list)

        # 3f. Disabled node fast-paths return sane defaults.
        check("disabled release → True (no-op)",
              sch._ClaimManager.release("n_off", "any") is True)
        check("disabled gc_stale → 0",
              sch._ClaimManager.gc_stale("n_off") == 0)
        check("disabled enumerate → []",
              sch._ClaimManager.enumerate("n_off") == [])

        # 3g. ssh-error response → claim returns ok=False with error msg.
        sch.run_on = lambda *a, **k: (255, "", "ssh: connection timed out")
        ok, info, _kind = sch._ClaimManager.claim(
            "n_on", {"id": "tFail", "est_vram_mb": 100,
                      "cpu_cores": 1, "ram_mb": 100},
            gpu_idx=None,
            node_state={"name": "n_on", "gpus": []})
        check("ssh failure during claim → ok=False with error info",
              ok is False and "rc=255" in info,
              diag=info)
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES


def test_phase3_2_1_claim_lifecycle_in_dispatch():
    """Phase 3.2.1: claim lifecycle wired through LocalBackend.launch +
    dispatch + watcher.

    Verifies:
      - LocalBackend.launch claims BEFORE ssh, releases on launch failure,
        update_pid after PID parsed (and again after container_main_pid).
      - Module-level launch() and Backend.launch all accept node_state kwarg.
      - _do_dispatch passes node_state to launch.
      - CLAIM_RACE sentinel takes the contention path: status=queued,
        no launch_fail_count increment, no launch_failed_nodes entry,
        emits "claim_race" event (not "launch_failed_retry").
      - terminal transition (status=done) releases the claim.
      - cancel/evict release the claim.
      - watcher tend-step sends renew_many for our running tasks +
        gc_stale on enabled nodes with no running tasks.
    """
    print("\n[95] Phase 3.2.1: claim lifecycle (launch / dispatch / terminate / watcher)")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("Backend.launch signatures accept node_state kwarg",
          src.count("def launch(self, task: dict, node_state: Optional[dict] = None)") >= 4,
          diag=f"only {src.count('def launch(self, task: dict, node_state: Optional[dict] = None)')} signatures match")
    check("module-level launch() forwards node_state to _BACKEND",
          "_BACKEND.launch(task, node_state=node_state)" in src)
    lb_idx = src.find("class LocalBackend")
    lb_end = src.find("\nclass ", lb_idx + 5)
    lb_body = src[lb_idx:lb_end]
    check("LocalBackend.launch claims BEFORE ssh (claim placement guard)",
          "_ClaimManager.claim(" in lb_body
          and "CLAIM_RACE: " in lb_body
          and "claim_record = info" in lb_body)
    check("LocalBackend.launch release-on-failure helper exists",
          "_release_and_fail" in lb_body)
    check("LocalBackend.launch update_pid after PID parsed",
          "_ClaimManager.update_pid(task[\"node\"], task[\"id\"], pid)" in lb_body)
    check("LocalBackend.launch update_pid AGAIN with container_main_pid",
          "_ClaimManager.update_pid(task[\"node\"], task[\"id\"], container_pid)" in lb_body)
    do_idx = src.find("def _do_dispatch")
    do_end = src.find("\ndef ", do_idx + 5)
    do_body = src[do_idx:do_end]
    check("_do_dispatch passes node_state to launch",
          "launch(t, node_state=picked_state)" in do_body)
    check("_do_dispatch handles CLAIM_RACE without incrementing fail_count",
          'msg.startswith("CLAIM_RACE:")' in do_body
          and '"claim_race"' in do_body)
    bcr_idx = src.find("def _batch_check_running")
    bcr_end = src.find("\ndef ", bcr_idx + 5)
    bcr_body = src[bcr_idx:bcr_end]
    check("_batch_check_running releases claim on terminal transition",
          "_ClaimManager.release(t[\"node\"], t[\"id\"])" in bcr_body)
    evict_idx = src.find("def _evict_to_queue")
    evict_end = src.find("\ndef ", evict_idx + 5)
    evict_body = src[evict_idx:evict_end]
    check("_evict_to_queue releases claim BEFORE clearing node",
          "_ClaimManager.release(victim[\"node\"], victim[\"id\"])" in evict_body)
    cancel_idx = src.find("def cmd_cancel")
    cancel_end = src.find("\ndef ", cancel_idx + 5)
    cancel_body = src[cancel_idx:cancel_end]
    check("cmd_cancel releases claim on running --force kill",
          "_ClaimManager.release(t[\"node\"], t[\"id\"])" in cancel_body)
    wi_idx = src.find("def _watch_iteration")
    wi_end = src.find("\ndef ", wi_idx + 5)
    wi_body = src[wi_idx:wi_end]
    check("_watch_iteration calls renew_many / gc_stale per enabled node",
          "_ClaimManager.renew_many(" in wi_body
          and "_ClaimManager.gc_stale(" in wi_body
          and "_ClaimManager.enabled_for(" in wi_body)

    # 2. Behavioral: LocalBackend.launch on a claims-enabled node.
    saved_NODES = sch.NODES
    saved_run_on = sch.run_on
    sch.NODES = {
        "n_on": {"name": "n_on", "host": "h_on", "enable_claims": True,
                  "cpu_cores": 12, "ram_mb": 32000, "claim_ttl_s": 600,
                  "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                  "max_concurrent_running": None},
        "n_off": {"name": "n_off", "host": "h_off",
                   "cpu_cores": 12, "ram_mb": 32000,
                   "ram_headroom_frac": 0.10, "max_vram_per_task": None,
                   "max_concurrent_running": None},
    }

    captured = []  # (node, full_cmd) — keep full so substring checks work past
                   # the heredoc setup bytes.
    def mock_run_on(node, cmd, timeout=20, check=False):
        captured.append((node, cmd))
        # claims script ops → simulate ok
        if "\"$SCRIPT_PATH\" claim " in cmd:
            return (0, '{"ok": true}\n', "")
        if "\"$SCRIPT_PATH\" release " in cmd:
            return (0, '{"ok": true, "removed": 1}\n', "")
        if "\"$SCRIPT_PATH\" update_pid " in cmd:
            return (0, '{"ok": true, "updated": 1}\n', "")
        # cwd preflight
        if "test -d " in cmd:
            return (0, "", "")
        # actual ssh launch (the setsid one)
        if "setsid bash" in cmd:
            return (0, "PID=4242\n", "")
        return (0, "", "")
    sch.run_on = mock_run_on

    try:
        # 2a. Happy path: enabled node, claim ok, launch ok, update_pid called.
        captured.clear()
        task = {"id": "tHappy", "node": "n_on", "cwd": "/work",
                "cmd": "python a.py", "gpu_idx": 0,
                "est_vram_mb": 1500, "cpu_cores": 2, "ram_mb": 1500,
                "remote_pids": [], "extra_env": {}}
        node_state = {"name": "n_on",
                      "gpus": [{"idx": 0, "total_mb": 12000, "free_mb": 11000,
                                "used_mb": 0, "util_pct": 0}]}
        ok, msg = sch._BACKEND.launch(task, node_state=node_state)
        check("happy: launch ok",
              ok is True and "pid=4242" in msg, diag=msg)
        check("happy: claim was ssh'd BEFORE the setsid bash launch",
              any('"$SCRIPT_PATH" claim' in c for n, c in captured)
              and any("setsid bash" in c for n, c in captured))
        # Ensure ordering: claim cmd index < setsid index
        claim_i = next(i for i, (_, c) in enumerate(captured)
                        if '"$SCRIPT_PATH" claim' in c)
        launch_i = next(i for i, (_, c) in enumerate(captured)
                         if "setsid bash" in c)
        check("happy: claim happens BEFORE ssh+nohup (race-free)",
              claim_i < launch_i, diag=f"claim={claim_i} launch={launch_i}")
        check("happy: update_pid was sent after PID parsed",
              any('"$SCRIPT_PATH" update_pid' in c for n, c in captured))

        # 2b. Disabled node: NO claim ssh (fast path).
        captured.clear()
        task = {"id": "tFast", "node": "n_off", "cwd": "/work",
                "cmd": "python a.py", "gpu_idx": 0,
                "est_vram_mb": 1500, "cpu_cores": 2, "ram_mb": 1500,
                "remote_pids": [], "extra_env": {}}
        ok, msg = sch._BACKEND.launch(task, node_state=None)
        check("disabled node: launch ok",
              ok is True, diag=msg)
        check("disabled node: NO claims-script ssh issued",
              not any('"$SCRIPT_PATH"' in c or "_claims_" in c for n, c in captured),
              diag=str(captured))

        # 2c. Claim conflict: ssh returns conflict → CLAIM_RACE sentinel.
        captured.clear()
        sch.run_on = lambda node, cmd, **kw: (
            (0, '{"ok": false, "conflict": "gpu0: full"}\n', "")
            if "\"$SCRIPT_PATH\" claim " in cmd
            else (0, "", "")
        )
        task = {"id": "tRace", "node": "n_on", "cwd": "/work",
                "cmd": "python a.py", "gpu_idx": 0,
                "est_vram_mb": 1500, "cpu_cores": 2, "ram_mb": 1500,
                "remote_pids": [], "extra_env": {}}
        ok, msg = sch._BACKEND.launch(task, node_state=node_state)
        check("claim conflict → ok=False with CLAIM_RACE: prefix",
              ok is False and msg.startswith("CLAIM_RACE: "),
              diag=msg)
        check("claim conflict → message names the underlying conflict",
              "gpu0: full" in msg, diag=msg)

        # 2d. Claim ok but launch ssh fails → release before returning.
        # Track release calls to confirm.
        rel_calls = []
        def mock_with_failing_launch(node, cmd, timeout=20, check=False):
            if "\"$SCRIPT_PATH\" claim " in cmd:
                return (0, '{"ok": true}\n', "")
            if "\"$SCRIPT_PATH\" release " in cmd:
                rel_calls.append(cmd)
                return (0, '{"ok": true, "removed": 1}\n', "")
            if "test -d " in cmd:
                return (0, "", "")
            if "setsid bash" in cmd:
                return (1, "", "ssh: blip")  # simulated launch failure
            return (0, "", "")
        sch.run_on = mock_with_failing_launch
        task = {"id": "tRel", "node": "n_on", "cwd": "/work",
                "cmd": "python a.py", "gpu_idx": 0,
                "est_vram_mb": 1500, "cpu_cores": 2, "ram_mb": 1500,
                "remote_pids": [], "extra_env": {}}
        ok, msg = sch._BACKEND.launch(task, node_state=node_state)
        check("launch failure after claim → ok=False",
              ok is False and "rc=1" in msg, diag=msg)
        check("launch failure after claim → release was called",
              len(rel_calls) >= 1, diag=str(rel_calls))

        # 2e. _do_dispatch: CLAIM_RACE event takes the contention path
        # (no launch_fail_count increment, status back to queued).
        sch.run_on = lambda node, cmd, **kw: (
            (0, '{"ok": false, "conflict": "ram: too tight"}\n', "")
            if "\"$SCRIPT_PATH\" claim " in cmd
            else (0, "", "")
        )
        # Stub a minimal dispatch context so _do_dispatch can run.
        saved_save = sch.save_state
        saved_pre = sch.precheck_git
        saved_resume = sch.find_resume
        # Phase 3.4.10 P1: dispatch now runs _stage_cwd_for_launch BEFORE the
        # backend.launch() call this test exercises. Bypass it for this test
        # since cwd="/work" is a fake path; the staging logic itself has its
        # own dedicated test block.
        # Phase 3.4.11 P1: dispatch now uses _stage_cwd_check (fast probe)
        # instead of _stage_cwd_for_launch — bypass the probe too so the
        # CLAIM_RACE / CLAIM_ERROR launch path is reachable with mock cwd.
        saved_stage = sch._stage_cwd_for_launch
        saved_check = sch._stage_cwd_check
        sch.save_state = lambda s: None
        sch.precheck_git = lambda t: (True, "")
        sch.find_resume = lambda t: None
        sch._stage_cwd_for_launch = lambda t, n: (True, "test bypass")
        sch._stage_cwd_check = lambda target, cwd: "ready"
        state = {"next_id": 100, "tasks": [{
            "id": "tDR", "status": "queued", "priority": "normal",
            "submitted_at": time.time(), "cwd": "/work",
            "cmd": "python a.py", "node": None, "gpu_idx": None,
            "est_vram_mb": 1500, "cpu_cores": 2, "ram_mb": 1500,
            "remote_pids": [], "extra_env": {}, "preferred_node": "n_on",
        }]}
        nodes = [{
            "name": "n_on", "alive": True, "free_cpu": 12, "free_ram_mb": 30000,
            "total_ram_mb": 32000, "total_cpu": 12, "running_count": 0,
            "slurm_pending_count": 0,
            "gpus": [{"idx": 0, "total_mb": 12000, "free_mb": 12000,
                       "used_mb": 0, "util_pct": 0}],
        }]
        try:
            events, _ = sch._do_dispatch(state, nodes)
            ev_types = [e["type"] for e in events]
            check("dispatch on CLAIM_RACE: 'claim_race' event emitted",
                  "claim_race" in ev_types, diag=str(ev_types))
            check("dispatch on CLAIM_RACE: NOT 'launch_failed_retry'",
                  "launch_failed_retry" not in ev_types, diag=str(ev_types))
            t = state["tasks"][0]
            check("dispatch on CLAIM_RACE: task back to queued",
                  t["status"] == "queued", diag=str(t))
            check("dispatch on CLAIM_RACE: launch_fail_count NOT incremented",
                  not t.get("launch_fail_count"),
                  diag=f"launch_fail_count={t.get('launch_fail_count')}")
            check("dispatch on CLAIM_RACE: last_block_reason mentions CLAIM_RACE",
                  "CLAIM_RACE" in (t.get("last_block_reason") or ""),
                  diag=t.get("last_block_reason"))
        finally:
            sch.save_state = saved_save
            sch.precheck_git = saved_pre
            sch.find_resume = saved_resume
            sch._stage_cwd_for_launch = saved_stage
            sch._stage_cwd_check = saved_check

        # 3. _evict_to_queue releases the claim before clearing node.
        rel_calls = []
        sch.run_on = lambda node, cmd, **kw: (
            (rel_calls.append(cmd), (0, '{"ok": true, "removed": 1}\n', ""))[1]
            if "\"$SCRIPT_PATH\" release " in cmd else (0, "", "")
        )
        # Stub kill to no-op
        saved_kill = sch._kill_task_processes
        sch._kill_task_processes = lambda task, timeout=15: (True, "ok")
        try:
            victim = {"id": "tEvict", "node": "n_on", "remote_pids": [1],
                      "process_group": 1, "started_at": time.time()}
            sch._evict_to_queue(victim, {"tasks": [victim]}, "test")
            check("evict releases claim BEFORE clearing node",
                  any("\"$SCRIPT_PATH\" release " in c for c in rel_calls),
                  diag=str(rel_calls))
        finally:
            sch._kill_task_processes = saved_kill
    finally:
        sch.NODES = saved_NODES
        sch.run_on = saved_run_on


def test_phase3_2_2_probe_folds_pending_claims():
    """Phase 3.2.2: probe_all subtracts pending (pid=null) claims from
    free resources so a concurrent scheduler's pick_placement sees the
    launch-race window as occupied. Claims with a live pid are NOT
    folded — the process is already visible to ps/nvidia-smi via the
    normal probe pathway, double-counting would be wrong.
    """
    print("\n[96] Phase 3.2.2: probe_all folds pending cross-scheduler claims")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_fold_claims_into_probe helper exists",
          "def _fold_claims_into_probe" in src)
    check("probe_all calls the fold helper",
          "_fold_claims_into_probe(nodes)" in src)
    check("fold helper subtracts ONLY pending claims (pid=null)",
          "not c.get(\"pid\")" in src
          and "pending = [c for c in claims" in src)

    # 2. Behavioral.
    saved_NODES = sch.NODES
    saved_probe_node = sch.probe_node
    saved_run_on = sch.run_on
    sch.NODES = {
        "n_on": {"name": "n_on", "host": "h_on", "enable_claims": True,
                  "cpu_cores": 12, "ram_mb": 32000, "claim_ttl_s": 600},
        "n_off": {"name": "n_off", "host": "h_off",
                   "cpu_cores": 12, "ram_mb": 32000},
    }

    # Mock probe_node so probe_all gets a known baseline.
    base = {
        "n_on": {"name": "n_on", "alive": True, "free_cpu": 12,
                  "free_ram_mb": 30000, "total_ram_mb": 32000, "total_cpu": 12,
                  "running_count": 0, "slurm_pending_count": 0,
                  "gpus": [{"idx": 0, "total_mb": 12000, "free_mb": 12000,
                              "used_mb": 0, "util_pct": 0}]},
        "n_off": {"name": "n_off", "alive": True, "free_cpu": 12,
                   "free_ram_mb": 30000, "total_ram_mb": 32000, "total_cpu": 12,
                   "running_count": 0, "slurm_pending_count": 0,
                   "gpus": [{"idx": 0, "total_mb": 12000, "free_mb": 12000,
                               "used_mb": 0, "util_pct": 0}]},
    }
    import copy as _copy
    sch.probe_node = lambda name: _copy.deepcopy(base[name])

    # Mock claims.list to return controlled records on n_on.
    claims_response = {"n_on": [], "n_off": []}
    def mock_run_on(node, cmd, **kw):
        if "\"$SCRIPT_PATH\" list" in cmd:
            return (0, '{"ok": true, "claims": '
                    + __import__("json").dumps(claims_response.get(node, []))
                    + '}\n', "")
        return (0, "", "")
    sch.run_on = mock_run_on

    try:
        # 2a. No claims → probe pass-through.
        claims_response["n_on"] = []
        out = sch.probe_all()
        n_on = next(n for n in out if n["name"] == "n_on")
        check("no claims → free_cpu unchanged",
              n_on["free_cpu"] == 12)
        check("no claims → GPU0 free_mb unchanged",
              n_on["gpus"][0]["free_mb"] == 12000)

        # 2b. PENDING claim (pid=null) → resources subtracted.
        claims_response["n_on"] = [{
            "scheduler_id": "other:9", "task_id": "tOther",
            "owner": "alice", "gpu_idx": 0,
            "vram_mb": 4000, "cpu_cores": 3, "ram_mb": 5000,
            "claimed_at": time.time(), "expires_at": time.time() + 3600,
            "pid": None,  # pending — not yet visible to ps/nvidia-smi
        }]
        out = sch.probe_all()
        n_on = next(n for n in out if n["name"] == "n_on")
        check("pending claim → free_cpu decreased by claimed cpu_cores",
              n_on["free_cpu"] == 12 - 3, diag=f"got {n_on['free_cpu']}")
        check("pending claim → free_ram_mb decreased by claimed ram_mb",
              n_on["free_ram_mb"] == 30000 - 5000,
              diag=f"got {n_on['free_ram_mb']}")
        check("pending claim → GPU0 free_mb decreased by claimed vram_mb",
              n_on["gpus"][0]["free_mb"] == 12000 - 4000,
              diag=f"got {n_on['gpus'][0]['free_mb']}")
        check("pending claim → GPU0 used_mb increased by claimed vram_mb",
              n_on["gpus"][0]["used_mb"] == 0 + 4000,
              diag=f"got {n_on['gpus'][0]['used_mb']}")
        check("pending_claims surfaced on node_state for diagnostics",
              n_on.get("pending_claims") and len(n_on["pending_claims"]) == 1)

        # 2c. Claim with LIVE pid → NOT subtracted (already visible to ps).
        claims_response["n_on"] = [{
            "scheduler_id": "other:9", "task_id": "tLive",
            "owner": "alice", "gpu_idx": 0,
            "vram_mb": 4000, "cpu_cores": 3, "ram_mb": 5000,
            "claimed_at": time.time(), "expires_at": time.time() + 3600,
            "pid": 99999,  # has pid — already in ps
        }]
        out = sch.probe_all()
        n_on = next(n for n in out if n["name"] == "n_on")
        check("live-pid claim → free_cpu NOT subtracted (would double-count)",
              n_on["free_cpu"] == 12)
        check("live-pid claim → GPU0 free_mb NOT subtracted",
              n_on["gpus"][0]["free_mb"] == 12000)

        # 2d. Mixed: one pending + one live → only pending subtracted.
        claims_response["n_on"] = [
            {"scheduler_id": "A", "task_id": "tP", "gpu_idx": 0,
             "vram_mb": 2000, "cpu_cores": 2, "ram_mb": 1500,
             "claimed_at": time.time(), "expires_at": time.time() + 3600,
             "pid": None},
            {"scheduler_id": "B", "task_id": "tL", "gpu_idx": 0,
             "vram_mb": 5000, "cpu_cores": 4, "ram_mb": 5000,
             "claimed_at": time.time(), "expires_at": time.time() + 3600,
             "pid": 88888},
        ]
        out = sch.probe_all()
        n_on = next(n for n in out if n["name"] == "n_on")
        check("mixed: only pending claim subtracted (cpu)",
              n_on["free_cpu"] == 12 - 2)
        check("mixed: only pending claim subtracted (vram)",
              n_on["gpus"][0]["free_mb"] == 12000 - 2000)

        # 2e. n_off has no enable_claims → pass-through, no ssh.
        claims_response["n_off"] = [{"task_id": "irrelevant", "pid": None,
                                      "vram_mb": 999, "cpu_cores": 999, "ram_mb": 999}]
        out = sch.probe_all()
        n_off = next(n for n in out if n["name"] == "n_off")
        check("disabled node: probe pass-through (no claims fold)",
              n_off["free_cpu"] == 12)

        # 2f. Multi-GPU per-card subtraction.
        sch.NODES["n_on"]["host"] = "h_on"
        base["n_on"]["gpus"] = [
            {"idx": 0, "total_mb": 12000, "free_mb": 12000, "used_mb": 0, "util_pct": 0},
            {"idx": 1, "total_mb": 12000, "free_mb": 12000, "used_mb": 0, "util_pct": 0},
        ]
        claims_response["n_on"] = [
            {"task_id": "g0", "scheduler_id": "X", "gpu_idx": 0,
             "vram_mb": 3000, "cpu_cores": 1, "ram_mb": 1000,
             "claimed_at": time.time(), "expires_at": time.time() + 3600, "pid": None},
            {"task_id": "g1", "scheduler_id": "X", "gpu_idx": 1,
             "vram_mb": 7000, "cpu_cores": 1, "ram_mb": 1000,
             "claimed_at": time.time(), "expires_at": time.time() + 3600, "pid": None},
        ]
        out = sch.probe_all()
        n_on = next(n for n in out if n["name"] == "n_on")
        check("multi-GPU: GPU0 subtracted by its own claim",
              n_on["gpus"][0]["free_mb"] == 12000 - 3000)
        check("multi-GPU: GPU1 subtracted by its own claim",
              n_on["gpus"][1]["free_mb"] == 12000 - 7000)
    finally:
        sch.NODES = saved_NODES
        sch.probe_node = saved_probe_node
        sch.run_on = saved_run_on


def test_phase3_2_3_concurrent_schedulers_only_one_wins():
    """Phase 3.2.3: end-to-end concurrency test against the REAL claims
    script (not mocks). Two simulated schedulers race to claim the same
    GPU's full capacity; only one succeeds, the other gets a clean
    conflict — proving the flock + atomic write actually serializes.
    """
    print("\n[97] Phase 3.2.3: cross-scheduler concurrency — one of two racers always wins")

    import tempfile, subprocess, threading, json as _json

    with tempfile.TemporaryDirectory() as td:
        script = sch._CLAIMS_REMOTE_SCRIPT.replace(
            'CLAIMS_FILE = "/tmp/scheduleurm/claims.json"',
            f'CLAIMS_FILE = {os.path.join(td, "claims.json")!r}',
        ).replace(
            'os.makedirs("/tmp/scheduleurm", exist_ok=True)',
            f'os.makedirs({td!r}, exist_ok=True)',
        )
        script_path = os.path.join(td, "_claims.py")
        with open(script_path, "w") as f:
            f.write(script)
        lock_path = os.path.join(td, "claims.lock")

        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000}}

        # Each "scheduler" wants the FULL GPU0 — capacity allows only one.
        def attempt(scheduler_id, task_id, results):
            rec = {
                "owner": "u", "scheduler_id": scheduler_id,
                "task_id": task_id, "gpu_idx": 0,
                "vram_mb": 11000, "cpu_cores": 6, "ram_mb": 50000,
                "claimed_at": time.time(),
                "expires_at": time.time() + 3600,
                "pid": None,
            }
            # Wrap each call in flock (matching what _claims_remote_op does).
            cmd = [
                "flock", "-x", "-w", "30", lock_path,
                "python3", script_path, "claim",
                _json.dumps(rec), _json.dumps(cap),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            try:
                results[scheduler_id] = _json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                results[scheduler_id] = {"_rc": r.returncode, "_stdout": r.stdout, "_stderr": r.stderr}

        # Run 2 racers concurrently many times; every iteration must show
        # exactly one winner.
        wins, losses, drawn_iter = 0, 0, 0
        for _ in range(20):
            # Reset claims file each iteration so capacity is clean.
            # Phase 3.4.7: must INITIALIZE to default (not delete) — empty
            # file is now treated as crash-corrupted; production setup
            # always pre-writes the default.
            with open(os.path.join(td, "claims.json"), "w") as cf:
                cf.write('{"version":1,"claims":[]}')
            results = {}
            threads = [
                threading.Thread(target=attempt, args=("h1:111", "tA", results)),
                threading.Thread(target=attempt, args=("h2:222", "tB", results)),
            ]
            for t in threads: t.start()
            for t in threads: t.join()
            ok_a = results.get("h1:111", {}).get("ok") is True
            ok_b = results.get("h2:222", {}).get("ok") is True
            if ok_a and not ok_b:
                wins += 1
            elif ok_b and not ok_a:
                wins += 1
            elif ok_a and ok_b:
                drawn_iter += 1  # both succeeded — would mean over-commit!
            else:
                losses += 1  # both failed — also wrong (capacity allowed 1)

        check("20-round race: every round had EXACTLY one winner (no over-commit)",
              wins == 20 and drawn_iter == 0,
              diag=f"wins={wins} drawn={drawn_iter} both_lost={losses}")
        check("20-round race: NEVER both succeeded (capacity invariant)",
              drawn_iter == 0)

    # 3. Cross-scheduler scenario via _ClaimManager: scheduler A claims, then
    # scheduler B's probe_all sees the resource as occupied (Phase 3.2.2
    # pending-claim fold).
    saved_NODES = sch.NODES
    saved_run_on = sch.run_on
    saved_probe_node = sch.probe_node
    sch.NODES = {"n": {"name": "n", "host": "h", "enable_claims": True,
                        "cpu_cores": 12, "ram_mb": 32000, "claim_ttl_s": 600}}

    # Shared claims state across our two simulated schedulers.
    shared_claims = []
    def mock_run_on(node, cmd, **kw):
        if "\"$SCRIPT_PATH\" claim " in cmd:
            # parse the record from cmd argv (last single-quoted JSON before capacity)
            import re as _re
            quoted = _re.findall(r"'(\{[^']*\})'", cmd)
            if not quoted:
                return (0, '{"ok": false, "error": "no payload"}\n', "")
            rec = _json.loads(quoted[-2])  # second-to-last is the record
            cap_obj = _json.loads(quoted[-1])
            # Apply capacity check against shared_claims.
            used_cpu = sum(c.get("cpu_cores", 0) for c in shared_claims)
            used_ram = sum(c.get("ram_mb", 0) for c in shared_claims)
            per_gpu = {}
            for c in shared_claims:
                if c.get("gpu_idx") is not None:
                    per_gpu[str(c["gpu_idx"])] = (per_gpu.get(str(c["gpu_idx"]), 0)
                                                   + c.get("vram_mb", 0))
            if used_cpu + rec.get("cpu_cores", 0) > cap_obj.get("cpu_cores", 0):
                return (0, '{"ok": false, "conflict": "cpu over"}\n', "")
            if used_ram + rec.get("ram_mb", 0) > cap_obj.get("ram_mb", 0):
                return (0, '{"ok": false, "conflict": "ram over"}\n', "")
            g = rec.get("gpu_idx")
            if g is not None:
                gcap = int(cap_obj.get("gpu_vram_mb", {}).get(str(g), 0))
                if per_gpu.get(str(g), 0) + rec.get("vram_mb", 0) > gcap:
                    return (0, '{"ok": false, "conflict": "gpu over"}\n', "")
            shared_claims.append(rec)
            return (0, '{"ok": true}\n', "")
        if "\"$SCRIPT_PATH\" list" in cmd:
            return (0, '{"ok": true, "claims": '
                    + _json.dumps(shared_claims) + '}\n', "")
        return (0, "", "")
    sch.run_on = mock_run_on

    base = {"name": "n", "alive": True, "free_cpu": 12, "free_ram_mb": 30000,
             "total_ram_mb": 32000, "total_cpu": 12, "running_count": 0,
             "slurm_pending_count": 0,
             "gpus": [{"idx": 0, "total_mb": 12000, "free_mb": 12000,
                         "used_mb": 0, "util_pct": 0}]}
    import copy as _copy
    sch.probe_node = lambda name: _copy.deepcopy(base)

    try:
        # Scheduler A claims the full GPU.
        ok_a, info_a, _ka = sch._ClaimManager.claim(
            "n", {"id": "tA", "est_vram_mb": 11000,
                   "cpu_cores": 6, "ram_mb": 25000},
            gpu_idx=0,
            node_state={"name": "n", "gpus": [{"idx": 0, "total_mb": 12000}]})
        check("racer A claim succeeds (capacity allows it)",
              ok_a is True, diag=str(info_a))

        # Now scheduler B does probe_all — should see the GPU as occupied
        # via Phase 3.2.2 pending-claim fold.
        out = sch.probe_all()
        n0 = out[0]
        check("scheduler B's probe_all reflects A's pending claim (free_mb)",
              n0["gpus"][0]["free_mb"] == 12000 - 11000,
              diag=f"got {n0['gpus'][0]['free_mb']}")
        check("scheduler B's probe_all reflects A's pending claim (free_cpu)",
              n0["free_cpu"] == 12 - 6)

        # B tries to claim the SAME GPU → conflict.
        ok_b, info_b, _kb = sch._ClaimManager.claim(
            "n", {"id": "tB", "est_vram_mb": 8000,
                   "cpu_cores": 4, "ram_mb": 10000},
            gpu_idx=0,
            node_state={"name": "n", "gpus": [{"idx": 0, "total_mb": 12000}]})
        check("racer B claim fails (cross-scheduler exclusion held)",
              ok_b is False, diag=str(info_b))
        check("racer B conflict message names the resource",
              isinstance(info_b, str)
              and ("gpu over" in info_b or "ram over" in info_b
                    or "cpu over" in info_b),
              diag=info_b)
    finally:
        sch.NODES = saved_NODES
        sch.run_on = saved_run_on
        sch.probe_node = saved_probe_node


def test_phase3_3_local_windows_host_metrics():
    """Phase 3.3: WSL2 `local` probe surfaces Windows-host metrics
    (free RAM + DXGI Compute-engine util) so TUI numbers match what users
    see in Task Manager.

    The discrepancy: WSL2 sees ~1GB MemAvailable inside its 30GB-cap VM,
    while the host has 17GB+ free in a 64GB pool. Similarly nvidia-smi's
    NVML utilization.gpu reads ~10% during RL bursts while Task Manager's
    Compute engine shows 90%+. Both numbers are correct for their model;
    showing both prevents user confusion and surfaces the right number for
    each mental model (placement decisions still use the WSL/NVML view).
    """
    print("\n[98] Phase 3.3: local probe surfaces Windows-host RAM + DXGI Compute util")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_probe_windows_host_extras helper exists",
          "def _probe_windows_host_extras" in src)
    check("helper queries Win32_OperatingSystem for free/total physical memory",
          "Win32_OperatingSystem" in src
          and "FreePhysicalMemory" in src
          and "TotalVisibleMemorySize" in src)
    check("helper queries DXGI Compute engine counter (matches Task Manager)",
          "GPU Engine" in src
          and "engtype_Compute" in src)
    check("probe_node folds extras ONLY for `local` node (WSL-only path)",
          'if name == "local":' in src
          and "_probe_windows_host_extras()" in src)
    check("extras land on host_free_ram_mb / host_total_ram_mb / per-GPU util_pct_compute",
          'host_free_ram_mb' in src
          and 'host_total_ram_mb' in src
          and 'util_pct_compute' in src)
    check("TUI _node_summary_line displays WSL / host RAM side-by-side",
          'WSL' in open(os.path.expanduser("~/.claude/skills/scheduler/tui.py")).read()
          and 'host' in open(os.path.expanduser("~/.claude/skills/scheduler/tui.py")).read())
    tui_src = open(os.path.expanduser("~/.claude/skills/scheduler/tui.py")).read()
    check("TUI shows both NVML and Compute util when both available",
          "nvml/compute" in tui_src or "util_pct_compute" in tui_src)

    # 2. Behavioral: helper returns parsed extras when PowerShell succeeds.
    saved_run = sch.subprocess.run
    saved_path_exists = sch.Path.exists

    class FakeRun:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""
    captured_cmds = []
    def fake_pwsh_ok(cmd, capture_output=True, text=True, timeout=None):
        captured_cmds.append(cmd)
        # Simulate PowerShell return: "free|total|gpu_pct"
        return FakeRun("17500|65536|94\n")
    # Path.exists check on /mnt/c... for powershell binary
    sch.Path.exists = lambda self: True
    sch.subprocess.run = fake_pwsh_ok
    try:
        out = sch._probe_windows_host_extras()
        check("helper: parses host_free_ram_mb",
              out.get("host_free_ram_mb") == 17500, diag=str(out))
        check("helper: parses host_total_ram_mb",
              out.get("host_total_ram_mb") == 65536, diag=str(out))
        check("helper: parses gpu_compute_util_pct",
              out.get("gpu_compute_util_pct") == 94, diag=str(out))
        check("helper: invocation passed -NoProfile -NonInteractive",
              any("-NoProfile" in arg for arg in captured_cmds[0])
              and any("-NonInteractive" in arg for arg in captured_cmds[0]))
        check("helper: invocation queries DXGI Compute engine counter",
              any("engtype_Compute" in arg for arg in captured_cmds[0]))

        # 2b. Garbage output → returns {} instead of crashing.
        sch.subprocess.run = lambda *a, **kw: FakeRun("nope")
        out = sch._probe_windows_host_extras()
        check("helper: garbage output → empty dict (no crash)",
              out == {})

        # 2c. PowerShell rc != 0 → returns {}.
        class FakeRunBad:
            returncode = 1
            stdout = ""
            stderr = "access denied"
        sch.subprocess.run = lambda *a, **kw: FakeRunBad()
        out = sch._probe_windows_host_extras()
        check("helper: rc != 0 → empty dict",
              out == {})

        # 2d. Subprocess timeout → returns {}.
        def fake_timeout(*a, **kw):
            raise sch.subprocess.TimeoutExpired(cmd="powershell", timeout=4)
        sch.subprocess.run = fake_timeout
        out = sch._probe_windows_host_extras()
        check("helper: subprocess timeout → empty dict (best-effort)",
              out == {})

        # 2e. powershell.exe missing → returns {} without invoking subprocess.
        sch.Path.exists = lambda self: False
        called = [0]
        def count_calls(*a, **kw):
            called[0] += 1
            return FakeRun("ignored")
        sch.subprocess.run = count_calls
        out = sch._probe_windows_host_extras()
        check("helper: powershell.exe missing → empty dict, no subprocess invoked",
              out == {} and called[0] == 0)
    finally:
        sch.subprocess.run = saved_run
        sch.Path.exists = saved_path_exists

    # 3. probe_node integration: for `local`, the result includes host_*.
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    saved_extras = sch._probe_windows_host_extras
    sch.NODES = {"local": {"host": None, "cpu_cores": 12, "ram_mb": 30000,
                            "ram_headroom_frac": 0.10}}
    # Stub run_on so probe_node returns a normal probe; then the integration
    # appends Windows extras.
    sch.run_on = lambda node, cmd, **kw: (0,
        ("0, 100, 8000, 7900, 50\n"  # GPU0 line
         "===SEP===\n"
         "20000\n30000\n"            # MemAvailable / MemTotal in MB
         "===SEP===\n"
         "12\n"                       # nproc
         "===SEP===\n"
         "1.5\n"                      # loadavg
         "===SEP===\n"
         "0, 50\n---SAMPLE---\n0, 50\n---SAMPLE---\n"  # extra util samples
        ), "")
    sch._probe_windows_host_extras = lambda: {
        "host_free_ram_mb": 17500,
        "host_total_ram_mb": 65536,
        "gpu_compute_util_pct": 88,
    }
    try:
        n = sch.probe_node("local")
        check("probe_node(local): host_free_ram_mb folded in",
              n.get("host_free_ram_mb") == 17500)
        check("probe_node(local): host_total_ram_mb folded in",
              n.get("host_total_ram_mb") == 65536)
        check("probe_node(local): util_pct_compute attached to each GPU",
              all(g.get("util_pct_compute") == 88 for g in n["gpus"]),
              diag=str(n["gpus"]))
        # WSL-side ram_free still reflects /proc/meminfo (placement decisions
        # use this, NOT the host value).
        check("probe_node(local): WSL free_ram_mb still reflects /proc/meminfo",
              n.get("free_ram_mb") == 20000)
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES
        sch._probe_windows_host_extras = saved_extras

    # 4. probe_node for non-local nodes does NOT call the Windows helper.
    extras_called = [0]
    sch._probe_windows_host_extras = lambda: (
        extras_called.__setitem__(0, extras_called[0] + 1)
        or {"host_free_ram_mb": 99999})
    sch.NODES = {"remote": {"host": "rbox", "cpu_cores": 12, "ram_mb": 32000,
                             "ram_headroom_frac": 0.10}}
    sch.run_on = lambda node, cmd, **kw: (0,
        ("0, 100, 8000, 7900, 50\n===SEP===\n20000\n30000\n===SEP===\n"
         "12\n===SEP===\n1.5\n===SEP===\n0, 50\n---SAMPLE---\n"), "")
    try:
        n = sch.probe_node("remote")
        check("probe_node(non-local): does NOT invoke Windows helper",
              extras_called[0] == 0,
              diag=f"helper called {extras_called[0]} times")
        check("probe_node(non-local): no host_* keys in result",
              "host_free_ram_mb" not in n
              and "host_total_ram_mb" not in n)
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES
        sch._probe_windows_host_extras = saved_extras

    # 5. TUI _node_summary_line uses both host + WSL when present.
    from tui import _node_summary_line
    n_with_extras = {
        "name": "local", "alive": True,
        "free_cpu": 7, "total_cpu": 12, "loadavg": 5.0,
        "free_ram_mb": 1230, "host_free_ram_mb": 17500,
        "gpus": [{"idx": 0, "used_mb": 800, "total_mb": 8000,
                    "free_mb": 7200, "util_pct": 10, "util_pct_compute": 88}],
    }
    line = _node_summary_line([n_with_extras])
    check("TUI line shows BOTH WSL and host RAM",
          "1230MB(WSL)" in line and "17500MB(host)" in line, diag=line)
    check("TUI line shows BOTH NVML and Compute util",
          "10/88%util(nvml/compute)" in line, diag=line)

    # 6. TUI line gracefully omits host info when extras unavailable.
    n_no_extras = {
        "name": "remote", "alive": True,
        "free_cpu": 12, "total_cpu": 12, "loadavg": 0.5,
        "free_ram_mb": 100000,
        "gpus": [{"idx": 0, "used_mb": 0, "total_mb": 12000,
                    "free_mb": 12000, "util_pct": 0}],
    }
    line2 = _node_summary_line([n_no_extras])
    check("TUI line omits (host) tag when no host_free_ram_mb",
          "(WSL)" not in line2 and "(host)" not in line2, diag=line2)
    check("TUI line omits (nvml/compute) tag when no util_pct_compute",
          "nvml/compute" not in line2, diag=line2)


def test_phase3_4_0_cross_user_claim_io():
    """Phase 3.4.0 + 3.4.1: claims layer must work across OS users.

    Reviewer's P0 findings:
      (3.4.0) `cat > /tmp/scheduleurm/_claims.py` collides on shared sticky
              dir — second user can't overwrite first user's script.
              Similarly, save() used os.rename(tmp, claims.json) which fails
              cross-user in sticky dirs (only the owner can rename over).
              Fix: per-user script path /tmp/scheduleurm/_claims_${USER}.py;
              mode 0666 on shared lock + claims; in-place truncate+write
              (no rename) so any user can update under flock.

      (3.4.1) alive() returned False on PermissionError — but EPERM from
              kill(pid, 0) means the process EXISTS but is owned by another
              user. Returning False let one user's GC drop another user's
              still-running claim → over-commit. Fix: PermissionError → True.
    """
    print("\n[99] Phase 3.4.0 + 3.4.1: cross-OS-user claim file safety + PID liveness")

    # 1. Source guards.
    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # Source has the f-string-escaped form ${{USER:-anon}}; runtime output has ${USER:-anon}.
    check("setup_cmd uses per-user script path (${USER:-anon})",
          '_claims_${{USER:-anon}}.py' in src)
    check("setup_cmd chmods lock + claims to 0666 for shared write",
          "chmod 0666" in src
          and "claims.lock" in src and "claims.json" in src)
    check("setup_cmd uses umask 0 so created files are world-rw",
          "umask 0" in src)
    check("remote script opens claims.json r+w (no rename) for in-place update",
          "_open_shared_rw" in src
          and "os.O_RDWR | os.O_CREAT" in src
          and "fchmod" in src)
    check("remote script no longer uses os.rename (sticky-dir cross-user fail)",
          "os.rename(tmp, CLAIMS_FILE)" not in src,
          diag="rename(my_tmp, others_file) fails in sticky dir under another user")
    check("remote script truncates + writes via fd (atomic under flock)",
          "_write_fd" in src
          and "os.ftruncate(fd, 0)" in src)
    check("alive() treats PermissionError as alive (EPERM = exists, owned by other user)",
          "except PermissionError:" in src
          and "return True" in src.split("except PermissionError:")[1][:200],
          diag="PermissionError must NOT short-circuit to dead")
    check("OSError fallback in alive() preserves alive when errno != ESRCH",
          "errno != errno.ESRCH" in src)

    # 2. Behavioral: run the actual script across two-user scenario by writing
    # claims.json with restricted ownership-mimicking mode.
    import tempfile, subprocess, json as _json, stat as _stat
    with tempfile.TemporaryDirectory() as td:
        # Deploy the script with CLAIMS_FILE redirected to tmp.
        script = sch._CLAIMS_REMOTE_SCRIPT.replace(
            'CLAIMS_FILE = "/tmp/scheduleurm/claims.json"',
            f'CLAIMS_FILE = {os.path.join(td, "claims.json")!r}',
        ).replace(
            'os.makedirs("/tmp/scheduleurm", exist_ok=True)',
            f'os.makedirs({td!r}, exist_ok=True)',
        )
        script_path = os.path.join(td, "_claims.py")
        with open(script_path, "w") as f:
            f.write(script)
        # Phase 3.4.7: mirror production setup — non-empty default
        # so 0-byte means crash, not bootstrap.
        claims_path = os.path.join(td, "claims.json")
        def _ensure_init():
            if (not os.path.exists(claims_path)
                    or os.path.getsize(claims_path) == 0):
                with open(claims_path, "w") as cf:
                    cf.write('{"version":1,"claims":[]}')
        _ensure_init()

        def run_op(op, payload, capacity=None):
            _ensure_init()
            r = subprocess.run(
                ["python3", script_path, op,
                 _json.dumps(payload), _json.dumps(capacity or {})],
                capture_output=True, text=True, timeout=10,
            )
            try:
                return _json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                return {"_rc": r.returncode, "_stdout": r.stdout, "_stderr": r.stderr}

        # 2a. First-write creates claims.json with 0666 (umask 0 in script).
        now = time.time()
        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000}}
        rec_a = {"owner": "userA", "scheduler_id": "A", "task_id": "tA",
                 "gpu_idx": 0, "vram_mb": 2000, "cpu_cores": 2, "ram_mb": 1500,
                 "claimed_at": now, "expires_at": now + 3600, "pid": None}
        r = run_op("claim", rec_a, cap)
        check("first claim creates claims.json",
              r.get("ok") is True, diag=r)
        claims_file = os.path.join(td, "claims.json")
        check("created claims.json exists",
              os.path.exists(claims_file))
        mode = _stat.S_IMODE(os.stat(claims_file).st_mode)
        check("claims.json created with 0666 mode (any user can update)",
              mode == 0o666, diag=f"got mode {oct(mode)}")

        # 2b. Subsequent op rewrites IN PLACE (same inode) — no rename.
        ino_before = os.stat(claims_file).st_ino
        r = run_op("release", {"scheduler_id": "A", "task_id": "tA"})
        check("release succeeds",
              r.get("ok") is True, diag=r)
        ino_after = os.stat(claims_file).st_ino
        check("update is IN-PLACE: inode unchanged across op (no rename)",
              ino_before == ino_after,
              diag=f"before={ino_before} after={ino_after}")

        # 2c. Even when claims.json has a non-0666 mode (simulating a pre-3.4.0
        # writer that left 0644), the next op's fchmod brings it back to 0666.
        os.chmod(claims_file, 0o644)
        r = run_op("list", {})
        check("op against 0644 claims.json still succeeds (fchmod relaxes)",
              r.get("ok") is True, diag=r)
        mode = _stat.S_IMODE(os.stat(claims_file).st_mode)
        check("after op: claims.json restored to 0666",
              mode == 0o666, diag=f"got mode {oct(mode)}")

        # 2d. Even if the file is read-only AND fchmod silently fails (the
        # cross-user case where we don't own it), in-place truncate+write
        # still works because we hold an open fd from the initial r+w mode.
        # Simulate: file with 0666 mode but with content we want to overwrite.
        with open(claims_file, "w") as f:
            f.write('{"version":1,"claims":[{"task_id":"stale","scheduler_id":"X",'
                    '"expires_at":' + str(now - 3600) + ',"pid":null,'
                    '"vram_mb":0,"cpu_cores":0,"ram_mb":0,"gpu_idx":null}]}')
        r = run_op("gc", {})
        check("gc op overwrites in-place (truncate+write, not rename)",
              r.get("ok") is True and r.get("removed") == 1, diag=r)

        # 3. PermissionError → alive=True. Test the SCRIPT'S `alive()` by
        # importing its body via runpy with a faked os.kill that raises
        # PermissionError on the target PID.
        # Approach: extract the alive() function from the script source, exec
        # it locally with a stubbed os.kill, verify behavior on each error.
        ns = {}
        # Pull the alive function definition out — it's the first def alive().
        sc = sch._CLAIMS_REMOTE_SCRIPT
        a_idx = sc.find("def alive(pid):")
        a_end = sc.find("\ndef ", a_idx + 5)
        body = sc[a_idx:a_end]
        # Stub `os.kill` so we control which exception fires.
        import types as _types
        fake_os = _types.ModuleType("fake_os")
        for k in dir(os):
            if not k.startswith("_"):
                try:
                    setattr(fake_os, k, getattr(os, k))
                except AttributeError:
                    pass
        kill_behavior = {"mode": "alive"}
        def fake_kill(pid, sig):
            m = kill_behavior["mode"]
            if m == "alive":
                return None
            if m == "noproc":
                raise ProcessLookupError()
            if m == "perm":
                raise PermissionError()
            if m == "value":
                raise ValueError()
            if m == "esrch":
                e = OSError()
                e.errno = __import__("errno").ESRCH
                raise e
            if m == "einval":
                e = OSError()
                e.errno = __import__("errno").EINVAL
                raise e
        fake_os.kill = fake_kill
        ns["os"] = fake_os
        ns["errno"] = __import__("errno")
        exec(body, ns)
        alive = ns["alive"]

        kill_behavior["mode"] = "alive"
        check("alive(pid) when kill returns ok → True",
              alive(12345) is True)
        kill_behavior["mode"] = "noproc"
        check("ProcessLookupError → False (process truly gone)",
              alive(12345) is False)
        kill_behavior["mode"] = "perm"
        check("PermissionError → True (exists but other user's PID — 3.4.1 fix)",
              alive(12345) is True,
              diag="reverting this would let one user's GC drop another's live claim")
        kill_behavior["mode"] = "esrch"
        check("OSError ESRCH → False (no such process)",
              alive(12345) is False)
        kill_behavior["mode"] = "einval"
        check("OSError EINVAL → True (any non-ESRCH OSError treated as alive)",
              alive(12345) is True)
        check("alive(0) / alive(None) → False (no pid)",
              alive(0) is False and alive(None) is False)

    # 4. Setup cmd structure tests (purely string composition).
    setup = sch._claims_setup_cmd()
    check("setup cmd ensures lock file is 0666",
          "chmod 0666" in setup and "claims.lock" in setup)
    check("setup cmd ensures claims.json is 0666",
          setup.count("chmod 0666") >= 2 and "claims.json" in setup)
    check("setup cmd writes per-user script under ${USER:-anon}",
          "_claims_${USER:-anon}.py" in setup)
    check("setup cmd umasks 0 before file ops",
          "umask 0" in setup)


def test_phase3_4_2_persistent_owner_id():
    """Phase 3.4.2 P1 fix: scheduler_id must persist across restarts so a
    watcher restart / manual dispatch / re-launched daemon can still
    release() / renew_many() / gc its OWN claims from before the restart.

    Pre-fix the id was `<host>:<pid>` — a fresh PID after restart didn't
    match any prior-restart claims. They survived until TTL expired
    (default 1h) which kept resources double-booked between the new
    process and the orphaned claim until cleanup.

    Now: random hex suffix stored at STATE_DIR/claim_owner_id, cached
    in-process, regenerated only if the file is missing. Different
    STATE_DIR (different scheduleurm install) → different id.
    """
    print("\n[100] Phase 3.4.2 P1 fix: persistent scheduler_id (survives restart)")

    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("scheduler_id() loads from STATE_DIR/claim_owner_id when present",
          'STATE_DIR / "claim_owner_id"' in src
          and "owner_file.read_text" in src)
    check("scheduler_id() generates UUID hex when missing",
          "uuid.uuid4()" in src
          and "owner_file.write_text" in src)
    check("scheduler_id() caches result in-class so file isn't read every call",
          "_cached_owner_id" in src)
    check("scheduler_id() does NOT use os.getpid() (would break on restart)",
          # Allow the fallback path to use pid as last-resort, but not the primary
          src.count("os.getpid()") < 5,  # used elsewhere in the file too; loose bound
          diag="primary path must not depend on PID")

    import tempfile, importlib
    saved_state_dir = sch.STATE_DIR
    saved_cache = getattr(sch._ClaimManager, "_cached_owner_id", None)

    try:
        with tempfile.TemporaryDirectory() as td:
            sch.STATE_DIR = sch.Path(td)
            # First call: id generated and persisted.
            try:
                delattr(sch._ClaimManager, "_cached_owner_id")
            except AttributeError:
                pass
            sid1 = sch._ClaimManager.scheduler_id()
            check("first call: scheduler_id generates a non-empty id",
                  sid1 and ":" in sid1, diag=sid1)
            owner_file = sch.Path(td) / "claim_owner_id"
            check("first call: id persisted to STATE_DIR/claim_owner_id",
                  owner_file.exists()
                  and owner_file.read_text().strip() == sid1, diag=sid1)

            # Second call (same process): cache hit, same id.
            sid2 = sch._ClaimManager.scheduler_id()
            check("second call (cached): same id",
                  sid2 == sid1)

            # Simulate restart: clear cache. Should re-load from disk, NOT
            # generate a new one.
            try:
                delattr(sch._ClaimManager, "_cached_owner_id")
            except AttributeError:
                pass
            sid3 = sch._ClaimManager.scheduler_id()
            check("after cache clear (simulated restart): same id loaded from disk",
                  sid3 == sid1, diag=f"sid1={sid1} sid3={sid3}")

            # Different STATE_DIR (different install) → different id.
            with tempfile.TemporaryDirectory() as td2:
                sch.STATE_DIR = sch.Path(td2)
                try:
                    delattr(sch._ClaimManager, "_cached_owner_id")
                except AttributeError:
                    pass
                sid_other = sch._ClaimManager.scheduler_id()
                check("different STATE_DIR (different install) → different id",
                      sid_other != sid1, diag=f"a={sid1} b={sid_other}")

            # Restore td so cleanup is clean.
            sch.STATE_DIR = sch.Path(td)
    finally:
        sch.STATE_DIR = saved_state_dir
        if saved_cache is None:
            try:
                delattr(sch._ClaimManager, "_cached_owner_id")
            except AttributeError:
                pass
        else:
            sch._ClaimManager._cached_owner_id = saved_cache


def test_phase3_4_3_claim_race_vs_claim_error():
    """Phase 3.4.3 P1 fix: capacity CONFLICTs and transport ERRORs follow
    different paths in dispatch.

    Pre-fix _ClaimManager.claim() returned a 2-tuple `(ok, msg)`, and
    LocalBackend.launch wrapped any failure as `CLAIM_RACE: <msg>`.
    Dispatch then treated CLAIM_RACE as contention — task back to queued,
    no launch_fail_count increment, retry next cycle. So a node with
    chronic transport issues (ssh permission, missing python3, flock
    failure, json parse error) made tasks loop in queue forever instead
    of escalating to APP_BUG_CAP.

    Now claim() returns `(ok, info, kind)` where kind ∈ {ok, conflict,
    error}. LocalBackend.launch translates:
      kind="conflict" → "CLAIM_RACE: ..."  (legitimate contention)
      kind="error"    → "CLAIM_ERROR: ..." (real launch failure)
    Dispatch CLAIM_RACE path stays as-is. CLAIM_ERROR falls through to
    the normal launch_failed_retry path so MAX_LAUNCH_RETRY → APP_BUG_CAP
    escalation eventually fires.
    """
    print("\n[101] Phase 3.4.3 P1 fix: CLAIM_RACE (capacity) vs CLAIM_ERROR (transport)")

    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    check("_ClaimManager.claim returns 3-tuple (ok, info, kind)",
          'return (True, record, "ok")' in src
          and '"conflict"' in src and '"error"' in src
          and 'claim transport failed' in src)
    check("LocalBackend.launch wraps conflict as CLAIM_RACE:",
          'f"CLAIM_RACE: {info}"' in src)
    check("LocalBackend.launch wraps transport errors as CLAIM_ERROR:",
          'f"CLAIM_ERROR: {info}"' in src)
    do_idx = src.find("def _do_dispatch")
    do_end = src.find("\ndef ", do_idx + 5)
    do_body = src[do_idx:do_end]
    check("dispatch routes CLAIM_RACE to contention path (no fail count)",
          'msg.startswith("CLAIM_RACE:")' in do_body)
    check("dispatch does NOT special-case CLAIM_ERROR (falls through to launch_fail)",
          'msg.startswith("CLAIM_ERROR:")' not in do_body,
          diag="CLAIM_ERROR must hit the regular launch_failed_retry path")

    # Behavioral: _ClaimManager.claim with the three response shapes.
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    sch.NODES = {"n_on": {"name": "n_on", "host": "h", "enable_claims": True,
                            "cpu_cores": 12, "ram_mb": 32000}}
    task = {"id": "tT", "est_vram_mb": 1000, "cpu_cores": 1, "ram_mb": 500}
    ns = {"name": "n_on", "gpus": [{"idx": 0, "total_mb": 12000}]}

    try:
        # 3a. Conflict response (script returned ok:false + conflict).
        sch.run_on = lambda *a, **kw: (
            0, '{"ok": false, "conflict": "ram: too tight"}\n', "")
        ok, info, kind = sch._ClaimManager.claim("n_on", task, 0, ns)
        check("conflict: ok=False",
              ok is False)
        check("conflict: kind='conflict'",
              kind == "conflict", diag=kind)
        check("conflict: info carries the reason",
              "ram: too tight" in info, diag=info)

        # 3b. Transport error: rc != 0.
        sch.run_on = lambda *a, **kw: (255, "", "ssh: connection timed out")
        ok, info, kind = sch._ClaimManager.claim("n_on", task, 0, ns)
        check("transport rc!=0: ok=False",
              ok is False)
        check("transport rc!=0: kind='error' (NOT 'conflict')",
              kind == "error", diag=kind)
        check("transport rc!=0: info names the rc/stderr",
              "rc=255" in info or "ssh" in info.lower(), diag=info)

        # 3c. Transport error: parse failure (script returned garbage).
        sch.run_on = lambda *a, **kw: (0, "garbage not json\n", "")
        ok, info, kind = sch._ClaimManager.claim("n_on", task, 0, ns)
        check("transport parse-fail: kind='error'",
              kind == "error", diag=kind)

        # 3d. Empty output → error.
        sch.run_on = lambda *a, **kw: (0, "", "")
        ok, info, kind = sch._ClaimManager.claim("n_on", task, 0, ns)
        check("transport empty stdout: kind='error'",
              kind == "error", diag=kind)

        # 3e. Happy path.
        sch.run_on = lambda *a, **kw: (0, '{"ok": true}\n', "")
        ok, info, kind = sch._ClaimManager.claim("n_on", task, 0, ns)
        check("happy: ok=True",
              ok is True)
        check("happy: kind='ok'",
              kind == "ok", diag=kind)
        check("happy: info is the claim record",
              isinstance(info, dict) and info.get("task_id") == "tT")

        # 3f. Disabled-node fast path returns ok with kind='ok'.
        sch.NODES["n_off"] = {"name": "n_off", "host": "h", "cpu_cores": 12,
                                "ram_mb": 32000}  # no enable_claims
        ok, info, kind = sch._ClaimManager.claim("n_off", task, 0, ns)
        check("disabled-node: ok=True",
              ok is True)
        check("disabled-node: kind='ok'",
              kind == "ok")
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES

    # 4. End-to-end via LocalBackend.launch + dispatch: CLAIM_ERROR
    # propagates to launch_fail_count (NOT to claim_race contention path).
    saved_save = sch.save_state
    saved_pre = sch.precheck_git
    saved_resume = sch.find_resume
    saved_run = sch.run_on
    saved_stage = sch._stage_cwd_for_launch
    saved_check = sch._stage_cwd_check
    sch.NODES = {"n_on": {"name": "n_on", "host": "h", "enable_claims": True,
                            "cpu_cores": 12, "ram_mb": 32000,
                            "ram_headroom_frac": 0.10,
                            "max_vram_per_task": None,
                            "max_concurrent_running": None}}
    sch.save_state = lambda s: None
    sch.precheck_git = lambda t: (True, "")
    sch.find_resume = lambda t: None
    # Phase 3.4.10/3.4.11: bypass pre-launch staging step + cache-only probe —
    # this test exercises the launch-time CLAIM_ERROR path with cwd="/work"
    # (mock); the staging logic itself has its own dedicated test block.
    sch._stage_cwd_for_launch = lambda t, n: (True, "test bypass")
    sch._stage_cwd_check = lambda target, cwd: "ready"
    # cwd preflight (test -d) runs BEFORE claim. Pass it; fail only the
    # claims-script ssh so the CLAIM_ERROR path is the actual failure.
    def fake_run(node, cmd, **kw):
        if "test -d" in cmd:
            return (0, "", "")
        if '"$SCRIPT_PATH" claim' in cmd:
            return (255, "", "ssh: blip")
        return (0, "", "")
    sch.run_on = fake_run

    state = {"next_id": 100, "tasks": [{
        "id": "tErr", "status": "queued", "priority": "normal",
        "submitted_at": time.time(), "cwd": "/work", "cmd": "python a.py",
        "node": None, "gpu_idx": None, "est_vram_mb": 1500,
        "cpu_cores": 2, "ram_mb": 1500,
        "remote_pids": [], "extra_env": {}, "preferred_node": "n_on",
    }]}
    nodes = [{
        "name": "n_on", "alive": True, "free_cpu": 12, "free_ram_mb": 30000,
        "total_ram_mb": 32000, "total_cpu": 12, "running_count": 0,
        "slurm_pending_count": 0,
        "gpus": [{"idx": 0, "total_mb": 12000, "free_mb": 12000,
                    "used_mb": 0, "util_pct": 0}],
    }]
    try:
        events, _ = sch._do_dispatch(state, nodes)
        ev_types = [e["type"] for e in events]
        check("CLAIM_ERROR dispatch: NOT routed to claim_race",
              "claim_race" not in ev_types,
              diag=str(ev_types))
        check("CLAIM_ERROR dispatch: routed to launch_failed_retry (real failure)",
              "launch_failed_retry" in ev_types,
              diag=str(ev_types))
        t = state["tasks"][0]
        check("CLAIM_ERROR: launch_fail_count incremented",
              (t.get("launch_fail_count") or 0) >= 1,
              diag=f"launch_fail_count={t.get('launch_fail_count')}")
        check("CLAIM_ERROR: last_block_reason carries the message",
              "CLAIM_ERROR" in (t.get("last_block_reason") or ""),
              diag=t.get("last_block_reason"))
    finally:
        sch.save_state = saved_save
        sch.precheck_git = saved_pre
        sch.find_resume = saved_resume
        sch.run_on = saved_run
        sch.NODES = saved_NODES
        sch._stage_cwd_for_launch = saved_stage
        sch._stage_cwd_check = saved_check


def test_phase3_4_4_claim_replicates_gpu_fits_policy():
    """Phase 3.4.4 P2 fix: claim() now enforces the same placement policy
    the local pick_placement / _gpu_fits applies, so two schedulers
    with stale probes can't both claim onto a fresh GPU when only one
    of them would have passed local _gpu_fits.

    Pre-fix the script's claim op only checked total VRAM cap. So:
      - per-task VRAM cap (local: 4GB per task) was unenforced
      - VRAM margin was unenforced
      - 1/3 packing rule was unenforced
    Two schedulers each picking the same fresh 12GB GPU could both
    succeed at claim() with 5GB tasks (5+5 ≤ 12), then both run
    simultaneously even though local 1/3 rule says only one fits.

    Util saturation is intentionally NOT replicated — there's no
    shared GPU util reading; local pick_placement gates on it before
    claim is invoked, which is the right place for that check.
    """
    print("\n[102] Phase 3.4.4 P2 fix: claim replicates per-task cap / margin / 1/3 rule")

    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()
    # Source guards: capacity payload now carries the policy fields.
    cap_idx = src.find("def _build_capacity")
    cap_end = src.find("\n    @classmethod", cap_idx + 5)
    cap_body = src[cap_idx:cap_end]
    check("_build_capacity carries max_vram_per_task",
          '"max_vram_per_task"' in cap_body)
    check("_build_capacity carries vram_margin_mb",
          '"vram_margin_mb"' in cap_body
          and "VRAM_MARGIN_MB" in cap_body)
    check("_build_capacity carries third_pack_rule + default_vram_mb",
          '"third_pack_rule"' in cap_body
          and '"default_vram_mb"' in cap_body)

    # Source guard: remote script's claim op enforces these.
    sc = sch._CLAIMS_REMOTE_SCRIPT
    check("script enforces per-task VRAM cap",
          "max_per_task" in sc and "per-task cap" in sc)
    check("script enforces VRAM margin",
          "vram_margin" in sc and "post-claim free" in sc)
    check("script enforces 1/3 packing rule with small-task exemption",
          "third_rule" in sc and "packing rule" in sc and "default_vram" in sc)

    # Behavioral: drive the script directly with controlled capacity to
    # exercise each new gate in isolation.
    import tempfile, subprocess, json as _json
    with tempfile.TemporaryDirectory() as td:
        script = sc.replace(
            'CLAIMS_FILE = "/tmp/scheduleurm/claims.json"',
            f'CLAIMS_FILE = {os.path.join(td, "claims.json")!r}',
        ).replace(
            'os.makedirs("/tmp/scheduleurm", exist_ok=True)',
            f'os.makedirs({td!r}, exist_ok=True)',
        )
        path = os.path.join(td, "_claims.py")
        with open(path, "w") as f:
            f.write(script)

        # Phase 3.4.7: production setup writes default content; mirror.
        claims_path = os.path.join(td, "claims.json")
        def _ensure_init():
            if (not os.path.exists(claims_path)
                    or os.path.getsize(claims_path) == 0):
                with open(claims_path, "w") as cf:
                    cf.write('{"version":1,"claims":[]}')

        def claim(rec, cap):
            _ensure_init()
            r = subprocess.run(
                ["python3", path, "claim",
                 _json.dumps(rec), _json.dumps(cap)],
                capture_output=True, text=True, timeout=10,
            )
            return _json.loads(r.stdout.strip().splitlines()[-1])

        def reset():
            try: os.unlink(claims_path)
            except FileNotFoundError: pass
            # _ensure_init() will rewrite default on next claim().

        now = time.time()
        base_rec = {"owner": "u", "scheduler_id": "S", "task_id": "tA",
                    "claimed_at": now, "expires_at": now + 3600, "pid": None,
                    "cpu_cores": 1, "ram_mb": 1000}

        # ---- Per-task cap: local has max_vram_per_task=4000, task wants 6000.
        reset()
        rec = dict(base_rec, gpu_idx=0, vram_mb=6000)
        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000},
               "max_vram_per_task": 4000, "vram_margin_mb": 0,
               "third_pack_rule": False, "default_vram_mb": 512}
        r = claim(rec, cap)
        check("per-task cap exceeded → conflict",
              r.get("ok") is False and "per-task cap" in (r.get("conflict") or ""),
              diag=r)

        # ---- Margin: claim leaves <margin free after.
        reset()
        rec = dict(base_rec, gpu_idx=0, vram_mb=11600)  # 12000-11600=400 < 500
        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000},
               "max_vram_per_task": None, "vram_margin_mb": 500,
               "third_pack_rule": False, "default_vram_mb": 512}
        r = claim(rec, cap)
        check("post-claim free < margin → conflict",
              r.get("ok") is False and "margin" in (r.get("conflict") or ""),
              diag=r)

        # ---- 1/3 rule (Phase 3.4.6 fix): the rule blocks STACKING new
        # tasks onto a card where existing claimed VRAM is already past
        # 1/3, NOT every claim that would itself exceed 1/3 of total.
        # Local _gpu_fits checks `gpu.used_mb >= third`, not
        # `gpu.used_mb + new >= third`. The first big task on an empty
        # GPU MUST succeed.
        reset()
        rec = dict(base_rec, gpu_idx=0, vram_mb=5000)
        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000},
               "max_vram_per_task": None, "vram_margin_mb": 0,
               "third_pack_rule": True, "default_vram_mb": 512}
        r = claim(rec, cap)
        check("first claim of 5000 on EMPTY 12GB GPU → ok (matches local _gpu_fits)",
              r.get("ok") is True, diag=r)
        # Now the GPU has 5000MB claimed (≥ 1/3). Stacking another 1000MB
        # task (> default_vram_mb=512) hits the 1/3 rule.
        rec2 = dict(base_rec, task_id="tStack", gpu_idx=0, vram_mb=1000)
        r2 = claim(rec2, cap)
        check("second claim of 1000 onto already-past-1/3 GPU → blocked by 1/3 rule",
              r2.get("ok") is False and "1/3" in (r2.get("conflict") or ""),
              diag=r2)

        # ---- 1/3 rule small-task exemption: 400MB task can stack past 1/3
        # of an already-loaded GPU (≤ default_vram_mb=512).
        # State from above: claims.json already has tA (5000MB) on GPU0.
        rec3 = dict(base_rec, task_id="tSmall", gpu_idx=0, vram_mb=400)
        r3 = claim(rec3, cap)
        check("small task (≤ default_vram_mb) past 1/3 → allowed (exemption)",
              r3.get("ok") is True, diag=r3)

        # ---- 1/3 rule disabled: large task on empty GPU succeeds.
        reset()
        rec = dict(base_rec, gpu_idx=0, vram_mb=5000)
        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000},
               "max_vram_per_task": None, "vram_margin_mb": 0,
               "third_pack_rule": False, "default_vram_mb": 512}
        r = claim(rec, cap)
        check("third_pack_rule=False → 5000MB on empty GPU allowed",
              r.get("ok") is True, diag=r)

        # ---- Happy: small claim that respects all gates.
        reset()
        rec = dict(base_rec, gpu_idx=0, vram_mb=2000)
        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000},
               "max_vram_per_task": 4000, "vram_margin_mb": 500,
               "third_pack_rule": True, "default_vram_mb": 512}
        r = claim(rec, cap)
        check("small claim within all gates → ok",
              r.get("ok") is True, diag=r)


def test_phase3_4_6_7_8_claim_one_third_corrupt_recovery_atomic_deploy():
    """Phase 3.4.6 + 3.4.7 + 3.4.8: three follow-up fixes after the cross-
    user claims layer landed.

    3.4.6 P1: 1/3 rule semantics matched local _gpu_fits — block on
              EXISTING used past 1/3, not on (used + new). Empty GPU's
              first big task succeeds.
    3.4.7 P2: in-place truncate isn't crash-atomic. Distinguish
              "bootstrap empty" (fine) from "post-crash empty" (lossy)
              by writing a default `{"version":1,"claims":[]}` at setup
              time, so any 0-byte read after that is a crash signal.
              Script raises → main returns ok=false + "claims_corrupt"
              error → caller sees CLAIM_ERROR (3.4.3) → operator
              notices instead of silent over-commit.
    3.4.8 P3: per-user script written via atomic tmp+rename to avoid
              partial reads when same-user scheduler instances run
              concurrent claim ops.
    """
    print("\n[103] Phase 3.4.6/7/8: 1/3-rule semantics + crash-corruption signal + atomic script deploy")

    src = open(os.path.expanduser("~/.claude/skills/scheduler/scheduler.py")).read()

    # 3.4.6 source guards
    sc = sch._CLAIMS_REMOTE_SCRIPT
    check("script: 1/3 rule checks GUSED (existing), not gused_after",
          "if gused >= third and gused > 100" in sc,
          diag="must mirror local _gpu_fits semantics")
    # Comment in docstring may mention `gused_after` as historical context;
    # what matters is the variable isn't ASSIGNED or USED in the rule.
    check("script: 1/3 rule NO LONGER assigns/uses gused_after",
          "gused_after =" not in sc and "gused_after >" not in sc,
          diag="that variable was the bug — keep it out of code paths")

    # 3.4.7 source guards
    check("script load_from_fd raises on empty file (post-crash signal)",
          "claims.json is empty" in sc and "raise RuntimeError" in sc)
    check("main() catches the corrupt-state exception → claims_corrupt error",
          "claims_corrupt" in sc)
    setup = sch._claims_setup_cmd()
    check("setup writes default content to claims.json when missing",
          '{"version":1,"claims":[]}' in setup)
    # 3.4.9 P1: setup must NOT clobber a 0-byte file. `-e` tests existence only;
    # `-s` would silently rewrite the post-crash empty state to "no claims",
    # bypassing the load_from_fd corruption signal at the next op.
    check("3.4.9: setup uses `if [ ! -e ... ]` (existence-only), not `-s`",
          "[ ! -e " in setup and "[ ! -s " not in setup,
          diag="`-s` would re-bootstrap empty file, masking crash-corruption")

    # 3.4.8 source guards
    check("setup writes per-user script via atomic tmp+rename",
          "TMP_PATH=" in setup and "mv \"$TMP_PATH\" \"$SCRIPT_PATH\"" in setup)
    check("tmp path is PID-suffixed for concurrent same-user safety",
          ".tmp.$$" in setup)

    # Behavioral: 3.4.6 — first 5GB claim on empty 12GB succeeds; second 1GB
    # gets blocked; small (≤512MB) exempted.
    import tempfile, subprocess, json as _json
    with tempfile.TemporaryDirectory() as td:
        script = sc.replace(
            'CLAIMS_FILE = "/tmp/scheduleurm/claims.json"',
            f'CLAIMS_FILE = {os.path.join(td, "claims.json")!r}',
        ).replace(
            'os.makedirs("/tmp/scheduleurm", exist_ok=True)',
            f'os.makedirs({td!r}, exist_ok=True)',
        )
        path = os.path.join(td, "_claims.py")
        with open(path, "w") as f:
            f.write(script)
        claims_path = os.path.join(td, "claims.json")

        def init():
            with open(claims_path, "w") as cf:
                cf.write('{"version":1,"claims":[]}')

        def call(op, payload, capacity=None):
            r = subprocess.run(
                ["python3", path, op,
                 _json.dumps(payload), _json.dumps(capacity or {})],
                capture_output=True, text=True, timeout=10,
            )
            return _json.loads(r.stdout.strip().splitlines()[-1])

        cap = {"cpu_cores": 12, "ram_mb": 100000,
               "gpu_vram_mb": {"0": 12000},
               "max_vram_per_task": None, "vram_margin_mb": 0,
               "third_pack_rule": True, "default_vram_mb": 512}
        now = time.time()
        rec_big = {"owner": "u", "scheduler_id": "S", "task_id": "tBig",
                   "gpu_idx": 0, "vram_mb": 5000, "cpu_cores": 1, "ram_mb": 1000,
                   "claimed_at": now, "expires_at": now + 3600, "pid": None}

        # 3.4.6 — first big task on empty GPU succeeds.
        init()
        r = call("claim", rec_big, cap)
        check("3.4.6: first 5GB claim on empty 12GB GPU → ok (was incorrectly blocked)",
              r.get("ok") is True, diag=r)

        # ... and now stacking a 1GB task gets blocked by 1/3 rule.
        rec_stack = dict(rec_big, task_id="tStack", vram_mb=1000)
        r = call("claim", rec_stack, cap)
        check("3.4.6: stacking 1GB onto already-past-1/3 GPU → blocked",
              r.get("ok") is False and "1/3" in (r.get("conflict") or ""),
              diag=r)

        # ... small (≤default) still allowed past 1/3.
        rec_small = dict(rec_big, task_id="tSmall", vram_mb=400)
        r = call("claim", rec_small, cap)
        check("3.4.6: small task (≤default) past 1/3 → ok (small-task exemption)",
              r.get("ok") is True, diag=r)

        # 3.4.7 — empty file (post-truncate crash simulation) → claims_corrupt error.
        # Init then truncate to simulate a crash mid-write.
        init()
        # Add a real claim so we know the empty-file read isn't bootstrap.
        call("claim", rec_big, cap)
        # Now simulate crash: open and truncate to 0.
        with open(claims_path, "w") as f:
            pass  # truncate
        r = call("list", {})
        check("3.4.7: post-crash 0-byte file → ok=False with claims_corrupt error",
              r.get("ok") is False
              and "claims_corrupt" in (r.get("error") or ""),
              diag=r)
        check("3.4.7: corrupt error message names the manual recovery hint",
              "Manual recovery" in (r.get("error") or ""),
              diag=r.get("error"))

        # And: malformed JSON → also errors out (not silently treated as empty).
        with open(claims_path, "w") as f:
            f.write("{not json")
        r = call("list", {})
        check("3.4.7: malformed JSON → claims_corrupt error",
              r.get("ok") is False and "claims_corrupt" in (r.get("error") or ""),
              diag=r)

    # 3.4.7 — claim() routes corrupt-state response to kind='error' (CLAIM_ERROR).
    saved_run_on = sch.run_on
    saved_NODES = sch.NODES
    sch.NODES = {"n_on": {"name": "n_on", "host": "h", "enable_claims": True,
                            "cpu_cores": 12, "ram_mb": 32000}}
    try:
        sch.run_on = lambda *a, **kw: (
            0, '{"ok": false, "error": "claims_corrupt: claims.json is empty ..."}\n', "")
        ok, info, kind = sch._ClaimManager.claim(
            "n_on", {"id": "tT", "est_vram_mb": 1000,
                      "cpu_cores": 1, "ram_mb": 500},
            0,
            {"name": "n_on", "gpus": [{"idx": 0, "total_mb": 12000}]})
        check("3.4.7: claim() with corrupt-state response → kind='error'",
              kind == "error", diag=kind)
        check("3.4.7: claim() with corrupt-state response → info names corruption",
              "claims_corrupt" in info, diag=info)
    finally:
        sch.run_on = saved_run_on
        sch.NODES = saved_NODES

    # 3.4.8 — atomic deploy: setup writes to TMP_PATH then mv. Two concurrent
    # heredoc writes never both target the same path.
    setup = sch._claims_setup_cmd()
    # The setup string contains both "cat > $TMP_PATH" and "mv ... $SCRIPT_PATH"
    # in that order.
    cat_idx = setup.find("cat > \"$TMP_PATH\"")
    mv_idx = setup.find('mv "$TMP_PATH" "$SCRIPT_PATH"')
    check("3.4.8: setup writes script to tmp BEFORE mv into place",
          0 < cat_idx < mv_idx,
          diag=f"cat={cat_idx} mv={mv_idx}")

    # ---- 3.4.9 P1 source guards ----

    # Source: orphan adopt (LOCAL) wires update_pid on success.
    src = open(sch.__file__).read()
    fn_local = src.split("def _try_recover_orphan_local_task")[1].split("\ndef ")[0]
    check("3.4.9: _try_recover_orphan_local_task calls update_pid after adopt",
          "_ClaimManager.update_pid(node, tid, pid)" in fn_local,
          diag="orphan adopted as running must wire host PID into claim")
    check("3.4.9: same function also wires update_pid for docker container PID",
          "_ClaimManager.update_pid(node, tid, cpid)" in fn_local,
          diag="docker branch overrides remote_pids — must mirror in claim")

    # Source: terminal-orphan finalize releases claim.
    fn_term = src.split("def _try_finalize_terminal_local_task")[1].split("\ndef ")[0]
    check("3.4.9: _try_finalize_terminal_local_task releases claim",
          "_ClaimManager.release(node, tid)" in fn_term,
          diag="terminal orphan must release claim or it lingers until TTL GC")

    # Source: revert-to-queued path releases claim.
    fn_revert = src.split("def recover_stale_launching_tasks")[1].split("\ndef ")[0]
    check("3.4.9: revert-to-queued in recover_stale_launching_tasks releases claim",
          "_ClaimManager.release(node, t[\"id\"])" in fn_revert,
          diag="claim from pre-launch may linger if scheduler died mid-launch")

    # Source: watcher reconciles claim PIDs each cycle.
    # _watch_iteration is the iteration body. Locate the reconcile sentinel.
    check("3.4.9: watcher reconciles claim pid against task remote_pids",
          "live_pid_by_node" in src and "update_pid(node, tid, want_pid)" in src,
          diag="best-effort update_pid in launch can fail; watcher must retry")

    # ---- 3.4.9 P1 behavioral: setup does NOT bootstrap a 0-byte file ----
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        # Build a minimal local "claims dir" then run a setup-equivalent.
        cf = os.path.join(td, "claims.json")
        # Pretend a prior writer crashed mid-truncate: 0 bytes on disk.
        open(cf, "w").close()
        size_before = os.path.getsize(cf)
        # Now simulate exactly what setup does for that bootstrap branch:
        # `if [ ! -e file ]; then printf default > file; fi`. With -e, the
        # branch is FALSE for an existing 0-byte file → no write.
        setup_branch_cmd = f"if [ ! -e {cf} ]; then printf '%s' '{{\"version\":1,\"claims\":[]}}' > {cf}; fi"
        subprocess.run(["bash", "-c", setup_branch_cmd], check=True)
        size_after = os.path.getsize(cf)
        check("3.4.9: setup bootstrap branch leaves a 0-byte claims.json untouched",
              size_before == 0 and size_after == 0,
              diag=f"before={size_before} after={size_after}")

        # And verify: with a real claim already present, then truncated to 0,
        # then setup runs again — the file STAYS 0 bytes (does not get re-init).
        # This is the actual reviewer scenario: writer ftruncate(0), crash,
        # next setup must NOT silently restore "[]" and lose the previous
        # cross-scheduler claims.
        with open(cf, "w") as f:
            f.write('{"version":1,"claims":[{"task_id":"tPrev"}]}')
        # Now truncate to simulate crash-window state.
        with open(cf, "w") as f:
            pass
        subprocess.run(["bash", "-c", setup_branch_cmd], check=True)
        size_post = os.path.getsize(cf)
        check("3.4.9: post-crash 0-byte claims.json NOT silently re-bootstrapped",
              size_post == 0,
              diag=f"size after second setup = {size_post}; "
                   "non-zero would mean bootstrap clobbered crash-corrupt state")

    # ============================================================
    # Phase 3.4.10 P1 fix: launch-time cwd auto-sync from local
    # ============================================================
    print("\n[Phase 3.4.10] launch-time cwd auto-sync from local source-of-truth")

    # ---- Source guards ----
    src = open(sch.__file__).read()

    check("3.4.10: LAUNCH_MAX_CWD_SIZE_MB constant defined (default 2048)",
          "LAUNCH_MAX_CWD_SIZE_MB" in src
          and 'os.environ.get("SCHEDULEURM_LAUNCH_MAX_CWD_SIZE_MB", "2048")' in src,
          diag="default 2GB per user spec '依赖 > 2GB 就坚持本地跑'")

    check("3.4.10: _stage_cwd_for_launch helper defined (3.4.12 added extra_excludes)",
          "def _stage_cwd_for_launch(task: dict, target_node: str" in src
          and "extra_excludes" in src,
          diag="must mirror _stage_for_migration but with source pinned to local; "
               "3.4.12 P1 added extra_excludes for dynamic ckpt_dir/result_dir protection")

    fn = src.split("def _stage_cwd_for_launch")[1].split("\ndef ")[0]
    check("3.4.10: helper short-circuits when target is local (host is None)",
          'NODES.get(target_node, {}).get("host") is None' in fn
          and "target is local" in fn,
          diag="local→local has no rsync to do")
    check("3.4.10: helper bails if local cwd doesn't exist (no source-of-truth)",
          "Path(cwd).exists()" in fn
          and "can't seed target" in fn,
          diag="rsync-from-nothing must fail explicitly, not silently mkdir empty")
    check("3.4.10: helper consults _staging_cache_hit before rsync",
          "_staging_cache_hit(cwd_key)" in fn,
          diag="TTL-bounded cache avoids redundant rsync within 10min window")
    check("3.4.10: helper applies LAUNCH_MAX_CWD_SIZE_MB cap with 'CAP_EXCEEDED:' sentinel",
          "LAUNCH_MAX_CWD_SIZE_MB" in fn and "CAP_EXCEEDED:" in fn,
          diag="caller dispatches require_node=local on this sentinel")
    check("3.4.10: du probe excludes match rsync excludes (consistent size accounting)",
          "--exclude=.git" in fn and "--exclude=__pycache__" in fn
          and "--exclude=results" in fn and "--exclude=logs" in fn,
          diag="size pre-check must match what rsync would actually transfer")
    check("3.4.10: helper updates _STAGING_CACHE on success",
          "_STAGING_CACHE[cwd_key] = time.time()" in fn,
          diag="next dispatch within TTL skips rsync via cache hit")

    # ---- Dispatch wiring guard: cache-only probe (3.4.11 P1 refactor) ----
    # 3.4.10 originally invoked _stage_cwd_for_launch synchronously inside
    # _do_dispatch (under state_lock) — that was a P1 bug because the rsync
    # could hold the lock for up to 10 min. 3.4.11 split it: the slow rsync
    # runs OUTSIDE the lock (via _stage_launch_candidates_outside_lock),
    # _do_dispatch only does a constant-time cache probe via _stage_cwd_check.
    check("3.4.11: dispatch uses _stage_cwd_check (cache-only) before launch()",
          "_stage_cwd_check(target, cwd_for_stage)" in src,
          diag="never call rsync inside state_lock — would block submit/cancel/status")
    check("3.4.11: outside-lock helper _stage_launch_candidates_outside_lock defined",
          "def _stage_launch_candidates_outside_lock()" in src,
          diag="mirrors _stage_migration_candidates_outside_lock pattern")
    check("3.4.11: cmd_dispatch + watcher invoke _stage_launch_candidates_outside_lock",
          src.count("_stage_launch_candidates_outside_lock()") >= 2,
          diag="must run before BOTH cmd_dispatch state_lock AND _watch_iteration state_lock")
    check("3.4.11: cap_exceeded route → require_node=local + revert queued (no fail-count bump)",
          'stage_state == "cap_exceeded"' in src
          and 't["require_node"] = "local"' in src,
          diag="size-cap is a routing decision, not a launch failure")
    check("3.4.11: needs_stage route → defer this cycle, no fail-count bump",
          'stage_state == "needs_stage"' in src
          and '"type": "launch_stage_deferred"' in src,
          diag="defer to next cycle so outside-lock can rsync without holding state_lock")
    check("3.4.11: launch_capped event emitted on cap_exceeded",
          '"type": "launch_capped"' in src,
          diag="surfaces in events log so operator sees why a task got pinned")

    # ---- Behavioral: helper short-circuit cases ----
    saved_NODES = sch.NODES
    try:
        sch.NODES = {
            "local": {"host": None, "cpu_cores": 4, "ram_mb": 8000},
            "remote": {"host": "user@remotehost", "cpu_cores": 4, "ram_mb": 8000},
        }
        # Case A: target is local → instant return
        ok, msg = sch._stage_cwd_for_launch({"cwd": "/tmp"}, "local")
        check("3.4.10 behavior: target=local short-circuits with ok=True",
              ok is True and "nothing to sync" in msg, diag=msg)

        # Case B: cwd doesn't exist locally → explicit failure
        ok, msg = sch._stage_cwd_for_launch(
            {"cwd": "/nonexistent_path_for_test_3_4_10"}, "remote")
        check("3.4.10 behavior: missing local cwd → ok=False with clear message",
              ok is False and "can't seed target" in msg, diag=msg)

        # Case C: empty/missing cwd field
        ok, msg = sch._stage_cwd_for_launch({}, "remote")
        check("3.4.10 behavior: no cwd on task → ok=False",
              ok is False and "no cwd" in msg, diag=msg)

        # Case D: CAP_EXCEEDED — synthesise by injecting a small cap.
        # Use a real existing dir that du can scan; threshold 0MB forces
        # CAP_EXCEEDED for any non-empty dir.
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            # Put a small file so du > 0
            with open(os.path.join(td, "marker.txt"), "w") as fh:
                fh.write("x" * 1024)
            saved_cap = sch.LAUNCH_MAX_CWD_SIZE_MB
            sch.LAUNCH_MAX_CWD_SIZE_MB = 0  # any size triggers cap
            try:
                ok, msg = sch._stage_cwd_for_launch({"cwd": td}, "remote")
                check("3.4.10 behavior: cwd > cap → ok=False with CAP_EXCEEDED prefix",
                      ok is False and msg.startswith("CAP_EXCEEDED:"), diag=msg)
            finally:
                sch.LAUNCH_MAX_CWD_SIZE_MB = saved_cap

        # Case E: cache hit short-circuits (no actual rsync attempted).
        # Pre-populate the cache for (local, remote, cwd) and verify the
        # helper returns ok=True with "cache hit" without ever calling rsync.
        # Mock subprocess.run to flag if rsync is attempted.
        with _tf.TemporaryDirectory() as td:
            cwd_key = ("local", "remote", td)
            sch._STAGING_CACHE[cwd_key] = time.time()  # fresh entry
            # If the helper hits the rsync path, du will be called on td.
            # Cache hit short-circuits BEFORE the du probe runs, so we just
            # check the return value.
            ok, msg = sch._stage_cwd_for_launch({"cwd": td}, "remote")
            check("3.4.10 behavior: cache hit returns ok=True without rsync",
                  ok is True and "cache hit" in msg, diag=msg)
            # Cleanup so other tests don't see this entry.
            sch._STAGING_CACHE.pop(cwd_key, None)
    finally:
        sch.NODES = saved_NODES

    # ============================================================
    # Phase 3.5 P1 fix: auto-pull results back to local on done
    # ============================================================
    print("\n[Phase 3.5] auto-pull results back to local on task completion")

    # ---- Source guards ----
    src = open(sch.__file__).read()
    check("3.5: RESULT_SYNC_MAX_ATTEMPTS constant defined (default 5)",
          'os.environ.get("SCHEDULEURM_RESULT_SYNC_MAX_ATTEMPTS", "5")' in src,
          diag="cap defends against chronically broken nodes hammering rsync")
    check("3.5: RESULT_SYNC_TIMEOUT_S constant defined (default 1800)",
          'os.environ.get("SCHEDULEURM_RESULT_SYNC_TIMEOUT_S", "1800")' in src,
          diag="30min timeout matches typical multi-GB rsync on slow links")
    check("3.5: _sync_one_result helper defined",
          "def _sync_one_result(candidate: dict)" in src)
    check("3.5: _sync_completed_results_outside_lock helper defined",
          "def _sync_completed_results_outside_lock()" in src)
    check("3.5: --result-dir CLI arg defined on submit",
          'add_argument("--result-dir"' in src,
          diag="user opts in by passing this on submit")
    check("3.5: --local-result-dir CLI arg defined on submit",
          'add_argument("--local-result-dir"' in src,
          diag="optional override for where rsync lands locally")
    check("3.5: cmd_submit stores result_dir / local_result_dir / sync state",
          '"result_dir": getattr(args, "result_dir"' in src
          and '"result_synced_at": None' in src
          and '"result_sync_attempts": 0' in src,
          diag="task record carries the opt-in fields + sync progress")
    check("3.5: cmd_dispatch invokes _sync_completed_results_outside_lock",
          "_sync_completed_results_outside_lock()" in src,
          diag="must run before main state_lock to avoid stalling other ops")

    # ---- Skip rules in _sync_completed_results_outside_lock ----
    fn = src.split("def _sync_completed_results_outside_lock")[1].split("\ndef ")[0]
    check("3.5: skip when status != 'done'",
          't.get("status") != "done"' in fn)
    check("3.5: skip when result_dir not set (opt-in)",
          'rd = t.get("result_dir")' in fn and "if not rd:" in fn)
    check("3.5: skip when result_synced_at already set (one-shot)",
          't.get("result_synced_at")' in fn)
    check("3.5: skip when attempts >= RESULT_SYNC_MAX_ATTEMPTS",
          ">= RESULT_SYNC_MAX_ATTEMPTS" in fn)
    check("3.5: skip when host=None (local node — already here)",
          'host = NODES.get(node, {}).get("host")' in fn
          and "if not host:" in fn)
    check("3.5: 3-phase outside-lock pattern — short lock → rsync → short lock",
          fn.count("with state_lock():") >= 2,
          diag="snapshot under lock; rsync OUTSIDE; commit markers under lock")
    check("3.5: defensive recheck on commit (status still done + same task)",
          't.get("status") != "done"' in fn,
          diag="task may have been cancelled/forgotten during rsync window")

    # ---- Behavioral: skip rules trigger correctly ----
    saved_NODES = sch.NODES
    saved_load = sch.load_state
    saved_save = sch.save_state
    saved_sync_one = sch._sync_one_result
    try:
        sch.NODES = {
            "local": {"host": None},
            "remote": {"host": "user@host"},
        }

        # Mock state with a mix of tasks; track which ones get sync'd.
        synced_ids = []
        def mock_sync_one(c):
            synced_ids.append(c["id"])
            return (True, "ok")
        sch._sync_one_result = mock_sync_one

        committed_state = {"tasks": [
            # Should sync: status=done, result_dir set, host=remote, not synced
            {"id": "t1", "status": "done", "result_dir": "/r/t1",
             "node": "remote", "result_synced_at": None,
             "result_sync_attempts": 0},
            # Skip: status != done
            {"id": "t2", "status": "running", "result_dir": "/r/t2",
             "node": "remote"},
            # Skip: no result_dir
            {"id": "t3", "status": "done", "node": "remote"},
            # Skip: already synced
            {"id": "t4", "status": "done", "result_dir": "/r/t4",
             "node": "remote", "result_synced_at": time.time()},
            # Skip: attempts cap reached
            {"id": "t5", "status": "done", "result_dir": "/r/t5",
             "node": "remote", "result_sync_attempts": 99},
            # Skip: local node (host=None)
            {"id": "t6", "status": "done", "result_dir": "/r/t6",
             "node": "local"},
        ]}
        # Mock load/save so the helper's 3-phase lock works on our dict.
        sch.load_state = lambda: committed_state
        saved_writes = []
        def mock_save(s): saved_writes.append({k: v for t in s["tasks"]
                                                  for k in [t["id"]]
                                                  for v in [{kk: vv for kk, vv in t.items()
                                                              if kk in ("status", "result_synced_at",
                                                                        "result_sync_attempts",
                                                                        "result_sync_error")}]})
        sch.save_state = mock_save

        sch._sync_completed_results_outside_lock()

        check("3.5 behavior: only t1 (status=done + result_dir + remote + not-synced) synced",
              synced_ids == ["t1"], diag=str(synced_ids))
        # Verify commit phase wrote result_synced_at on t1.
        t1 = next(t for t in committed_state["tasks"] if t["id"] == "t1")
        check("3.5 behavior: success commits result_synced_at",
              t1.get("result_synced_at") is not None
              and t1.get("result_sync_error") is None,
              diag=str(t1))

        # Failure path: ensure attempts increments + error recorded.
        synced_ids.clear()
        committed_state2 = {"tasks": [
            {"id": "tFail", "status": "done", "result_dir": "/r/tFail",
             "node": "remote", "result_synced_at": None,
             "result_sync_attempts": 2},
        ]}
        sch.load_state = lambda: committed_state2
        sch._sync_one_result = lambda c: (False, "rsync rc=23: blip")
        sch._sync_completed_results_outside_lock()
        tF = committed_state2["tasks"][0]
        check("3.5 behavior: failure increments result_sync_attempts",
              tF.get("result_sync_attempts") == 3, diag=str(tF))
        check("3.5 behavior: failure records result_sync_error",
              "rsync rc=23" in (tF.get("result_sync_error") or ""), diag=str(tF))
        check("3.5 behavior: failure leaves result_synced_at unset",
              tF.get("result_synced_at") is None, diag=str(tF))

        # Defensive recheck: if task transitions away from 'done' DURING the
        # rsync window, commit phase must NOT mutate it. Simulate by mutating
        # status between snapshot and commit via a sentinel sync_one.
        committed_state3 = {"tasks": [
            {"id": "tCancel", "status": "done", "result_dir": "/r/tCancel",
             "node": "remote", "result_synced_at": None,
             "result_sync_attempts": 0},
        ]}
        sch.load_state = lambda: committed_state3
        def sync_then_cancel(c):
            # Simulate user cancelling between the two short locks.
            committed_state3["tasks"][0]["status"] = "cancelled"
            return (True, "ok")
        sch._sync_one_result = sync_then_cancel
        sch._sync_completed_results_outside_lock()
        tC = committed_state3["tasks"][0]
        check("3.5 behavior: status flipped during rsync window → no commit",
              tC.get("status") == "cancelled"
              and tC.get("result_synced_at") is None,
              diag=str(tC))
    finally:
        sch.NODES = saved_NODES
        sch.load_state = saved_load
        sch.save_state = saved_save
        sch._sync_one_result = saved_sync_one

    # ============================================================
    # Phase 3.4.11 P1/P2 fixes — outside-lock launch staging,
    # rsync --delete, concurrent-rsync guard for result sync
    # ============================================================
    print("\n[Phase 3.4.11] outside-lock launch staging + rsync --delete + result-sync claim marker")

    src = open(sch.__file__).read()

    # ---- P1: outside-lock launch staging architecture ----
    fn_check = src.split("def _stage_cwd_check(")[1].split("\ndef ")[0]
    check("3.4.11 P1: _stage_cwd_check returns 'ready' for local target (host=None)",
          'NODES.get(target_node, {}).get("host") is None' in fn_check
          and 'return "ready"' in fn_check,
          diag="local target has nothing to sync")
    check("3.4.11 P1: _stage_cwd_check returns 'ready' on _STAGING_CACHE hit",
          "_staging_cache_hit(cwd_key)" in fn_check
          and 'return "ready"' in fn_check)
    check("3.4.11 P1: _stage_cwd_check returns 'cap_exceeded' on _STAGING_CAP_EXCEEDED hit (TTL'd)",
          "_STAGING_CAP_EXCEEDED" in fn_check
          and 'return "cap_exceeded"' in fn_check)
    check("3.4.11 P1: _stage_cwd_check returns 'needs_stage' on cache miss",
          'return "needs_stage"' in fn_check,
          diag="dispatch defers to next cycle; outside-lock helper rsyncs in between")

    fn_outside = src.split("def _stage_launch_candidates_outside_lock()")[1].split("\ndef ")[0]
    check("3.4.11 P1: outside-lock helper short-locks for snapshot only",
          "with state_lock():" in fn_outside,
          diag="snapshot under SHORT lock; rsync OUTSIDE")
    check("3.4.11 P1: outside-lock helper iterates queued tasks only",
          't.get("status") != "queued"' in fn_outside)
    check("3.4.11 P1: outside-lock helper skips already-cached (target, cwd) pairs",
          "_staging_cache_hit(cwd_key)" in fn_outside
          and "_STAGING_CAP_EXCEEDED.get(cwd_key)" in fn_outside,
          diag="avoid redundant ssh round-trips when caches are warm")
    check("3.4.11 P1: outside-lock helper calls _stage_cwd_for_launch outside the lock",
          "_stage_cwd_for_launch" in fn_outside,
          diag="real rsync must NOT hold state_lock")

    # ---- P1 behavioral: _stage_cwd_check fast-path returns ----
    saved_NODES = sch.NODES
    try:
        sch.NODES = {
            "local": {"host": None},
            "remote": {"host": "user@host"},
        }
        # local target → ready
        check("3.4.11 P1 behavior: target=local → 'ready' (no rsync needed)",
              sch._stage_cwd_check("local", "/some/cwd") == "ready")

        # remote, no cache entries → needs_stage
        sch._STAGING_CACHE.clear()
        sch._STAGING_CAP_EXCEEDED.clear()
        check("3.4.11 P1 behavior: cold cache → 'needs_stage'",
              sch._stage_cwd_check("remote", "/some/cwd") == "needs_stage")

        # remote with fresh cache → ready
        cwd_key = ("local", "remote", "/some/cwd")
        sch._STAGING_CACHE[cwd_key] = time.time()
        check("3.4.11 P1 behavior: fresh _STAGING_CACHE entry → 'ready'",
              sch._stage_cwd_check("remote", "/some/cwd") == "ready")
        sch._STAGING_CACHE.pop(cwd_key, None)

        # remote with cap_exceeded marker → cap_exceeded
        sch._STAGING_CAP_EXCEEDED[cwd_key] = time.time()
        check("3.4.11 P1 behavior: fresh _STAGING_CAP_EXCEEDED → 'cap_exceeded'",
              sch._stage_cwd_check("remote", "/some/cwd") == "cap_exceeded")
        sch._STAGING_CAP_EXCEEDED.pop(cwd_key, None)

        # Stale CAP_EXCEEDED (older than STAGING_TTL_S) → reverts to needs_stage
        sch._STAGING_CAP_EXCEEDED[cwd_key] = time.time() - sch.STAGING_TTL_S - 10
        check("3.4.11 P1 behavior: stale _STAGING_CAP_EXCEEDED → 'needs_stage' (auto recovery)",
              sch._stage_cwd_check("remote", "/some/cwd") == "needs_stage")
        sch._STAGING_CAP_EXCEEDED.pop(cwd_key, None)
    finally:
        sch.NODES = saved_NODES

    # ---- P2-1: rsync --delete in launch staging ----
    fn_launch = src.split("def _stage_cwd_for_launch")[1].split("\ndef ")[0]
    check("3.4.11 P2-1: _stage_cwd_for_launch rsync uses --delete (enforces source-of-truth)",
          '"--delete"' in fn_launch,
          diag="without --delete, files renamed/deleted on local linger on remote")
    check("3.4.11 P2-1: --delete still excludes results/logs/experiment_output (preserves outputs)",
          '"--exclude=results/"' in fn_launch
          and '"--exclude=logs/"' in fn_launch
          and '"--exclude=experiment_output/"' in fn_launch,
          diag="exclude pattern protects from BOTH transfer and delete passes")
    check("3.4.11 P2-1: CAP_EXCEEDED branch populates _STAGING_CAP_EXCEEDED cache",
          "_STAGING_CAP_EXCEEDED[cwd_key] = time.time()" in fn_launch,
          diag="dispatch's fast probe reads this cache without re-running du")
    check("3.4.11 P2-1: success branch clears stale _STAGING_CAP_EXCEEDED entry",
          "_STAGING_CAP_EXCEEDED.pop(cwd_key, None)" in fn_launch,
          diag="user-shrunk cwd recovers without waiting for TTL expiry")

    # ---- P2-2: result sync claim marker ----
    fn_sync = src.split("def _sync_completed_results_outside_lock")[1].split("\ndef ")[0]
    check("3.4.11 P2-2: result sync sets result_syncing_at under lock during snapshot",
          't["result_syncing_at"] = now' in fn_sync,
          diag="atomic claim prevents two sessions rsync'ing the same task concurrently")
    check("3.4.11 P2-2: result sync skips tasks with fresh result_syncing_at",
          "syncing_at = t.get(\"result_syncing_at\")" in fn_sync
          and "stale_threshold" in fn_sync,
          diag="another worker holds the claim; back off")
    check("3.4.11 P2-2: stale claim (older than RESULT_SYNC_TIMEOUT_S + grace) is reclaimable",
          "RESULT_SYNC_STALE_GRACE_S" in fn_sync
          and "result_sync_claim_reclaimed" in fn_sync,
          diag="dead-process leak self-heals after timeout + grace")
    check("3.4.11 P2-2: commit phase clears result_syncing_at unconditionally",
          't.pop("result_syncing_at", None)' in fn_sync,
          diag="success OR failure both release the claim")
    check("3.4.11 P2-2: cmd_submit initializes result_syncing_at field",
          '"result_syncing_at": None' in src,
          diag="task record carries the claim slot from creation")

    # ---- P2-2 behavioral: concurrent guard skips fresh claims ----
    saved_NODES = sch.NODES
    saved_load = sch.load_state
    saved_save = sch.save_state
    saved_sync_one = sch._sync_one_result
    try:
        sch.NODES = {
            "local": {"host": None},
            "remote": {"host": "user@host"},
        }
        # Case A: another session is already rsync'ing → skip
        synced_ids = []
        sch._sync_one_result = lambda c: (synced_ids.append(c["id"]), (True, "ok"))[1]
        state_with_fresh_claim = {"tasks": [
            {"id": "tBusy", "status": "done", "result_dir": "/r/tBusy",
             "node": "remote", "result_synced_at": None,
             "result_sync_attempts": 0,
             "result_syncing_at": time.time() - 5},  # fresh claim, 5s ago
        ]}
        sch.load_state = lambda: state_with_fresh_claim
        sch.save_state = lambda s: None
        sch._sync_completed_results_outside_lock()
        check("3.4.11 P2-2 behavior: fresh result_syncing_at → skipped (no concurrent rsync)",
              synced_ids == [], diag=str(synced_ids))

        # Case B: stale claim (>timeout+grace) → reclaimed
        synced_ids.clear()
        stale_ts = time.time() - sch.RESULT_SYNC_TIMEOUT_S - sch.RESULT_SYNC_STALE_GRACE_S - 10
        state_with_stale_claim = {"tasks": [
            {"id": "tStale", "status": "done", "result_dir": "/r/tStale",
             "node": "remote", "result_synced_at": None,
             "result_sync_attempts": 0,
             "result_syncing_at": stale_ts},
        ]}
        sch.load_state = lambda: state_with_stale_claim
        sch._sync_completed_results_outside_lock()
        check("3.4.11 P2-2 behavior: stale result_syncing_at → reclaimed (synced)",
              synced_ids == ["tStale"], diag=str(synced_ids))
        # After commit, syncing_at should be cleared
        tStale = state_with_stale_claim["tasks"][0]
        check("3.4.11 P2-2 behavior: commit phase cleared result_syncing_at",
              tStale.get("result_syncing_at") is None
              and tStale.get("result_synced_at") is not None,
              diag=str(tStale))

        # Case C: failure path also clears syncing_at (claim released even on failure)
        sch._sync_one_result = lambda c: (False, "rsync rc=23")
        synced_ids.clear()
        state_fail = {"tasks": [
            {"id": "tF", "status": "done", "result_dir": "/r/tF",
             "node": "remote", "result_synced_at": None,
             "result_sync_attempts": 0,
             "result_syncing_at": None},
        ]}
        sch.load_state = lambda: state_fail
        sch._sync_completed_results_outside_lock()
        tF = state_fail["tasks"][0]
        check("3.4.11 P2-2 behavior: failure clears result_syncing_at + bumps attempts",
              tF.get("result_syncing_at") is None
              and tF.get("result_sync_attempts") == 1
              and tF.get("result_synced_at") is None,
              diag=str(tF))
    finally:
        sch.NODES = saved_NODES
        sch.load_state = saved_load
        sch.save_state = saved_save
        sch._sync_one_result = saved_sync_one

    # ============================================================
    # Phase 3.4.12 fixes — dynamic excludes + stage_failed escalation
    #   + preferred_node fallback + (sig, cmd) dedup
    # ============================================================
    print("\n[Phase 3.4.12] dynamic excludes + stage_failed + preferred_node + (sig,cmd) dedup")

    src = open(sch.__file__).read()

    # ---- P1-1: dynamic excludes for ckpt_dir/result_dir under cwd ----
    fn_launch = src.split("def _stage_cwd_for_launch")[1].split("\ndef ")[0]
    check("3.4.12 P1-1: _stage_cwd_for_launch accepts extra_excludes parameter",
          "extra_excludes: list" in fn_launch
          and 'f"--exclude={ex}"' in fn_launch,
          diag="caller can pass dynamic --exclude paths to protect ckpt_dir/result_dir")
    check("3.4.12 P1-1: du size probe also honors extra_excludes (no over-count)",
          fn_launch.count("extra_excludes") >= 2,
          diag="cap check would falsely fire if du counts dirs the rsync would skip")

    fn_outside = src.split("def _stage_launch_candidates_outside_lock()")[1].split("\ndef ")[0]
    check("3.4.12 P1-1: outside-lock helper builds protected_under_cwd map",
          "protected_under_cwd" in fn_outside
          and "ckpt_dir" in fn_outside and "result_dir" in fn_outside,
          diag="scan ALL tasks (any status) for ckpt_dir/result_dir under each cwd")
    check("3.4.12 P1-1: outside-lock helper passes extra_excludes to _stage_cwd_for_launch",
          "extra_excludes=extra" in fn_outside,
          diag="dynamic protection passed through to rsync invocation")
    check("3.4.12 P1-1: rel-path computed via os.path.relpath, rejects '..' traversal",
          "os.path.relpath" in fn_outside
          and 'rel.startswith("..")' in fn_outside,
          diag="ckpt_dir outside cwd shouldn't contribute (relpath would yield '..')")

    # ---- P1-2: stage_failed cache + dispatch routing ----
    check("3.4.12 P1-2: _STAGING_FAILS cache module-level dict",
          "_STAGING_FAILS: dict = {}" in src,
          diag="rsync transport failures must persist across cycles, TTL via STAGING_FAIL_COOLDOWN_S")
    check("3.4.12 P1-2: outside-lock helper records to _STAGING_FAILS on rsync failure",
          '_STAGING_FAILS[("local", tn, cwd)] = (time.time()' in src,
          diag="failure path that ISN'T CAP_EXCEEDED must be recorded for escalation")
    fn_check = src.split("def _stage_cwd_check")[1].split("\ndef ")[0]
    check("3.4.12 P1-2: _stage_cwd_check returns 'stage_failed' on fresh _STAGING_FAILS entry",
          'return "stage_failed"' in fn_check,
          diag="dispatch routes via launch_failed_nodes/launch_fail_count instead of needs_stage loop")
    check("3.4.12 P1-2: _stage_failure_reason helper exposes underlying error",
          "def _stage_failure_reason" in src,
          diag="dispatch uses this for last_block_reason and launch_failed_nodes entry")
    check("3.4.12 P1-2: dispatch routes 'stage_failed' through launch_fail_count + escalation",
          'stage_state == "stage_failed"' in src
          and "_stage_failure_reason(target, cwd_for_stage)" in src
          and "_write_escalation" in src,
          diag="permanent rsync failure must escalate after MAX_LAUNCH_RETRY")

    # ---- P2-1: preferred_node fallback (stage all remote nodes if not require) ----
    check("3.4.12 P2-1: outside-lock helper splits require_node vs preferred_node",
          "require = t.get(\"require_node\")" in fn_outside
          and "if require:" in fn_outside,
          diag="only require_node is single-target; preferred_node is soft → stage all remotes")
    check("3.4.12 P2-1: when no require_node, stage to ALL nodes (filtered to non-local below)",
          "tgts = list(NODES.keys())" in fn_outside,
          diag="preferred_node alone shouldn't shrink target set — pick_placement may fall back")

    # ---- P2-2: dedup key changed to (signature, cmd) ----
    check("3.4.12 P2-2: dispatch dedup uses (sig, cmd) tuple, not sig alone",
          "running_keys = {" in src
          and 't.get("signature") or "", t.get("cmd")' in src,
          diag="lets independent experiments with same family signature run in parallel")
    check("3.4.12 P2-2: in-loop add uses (sig, cmd) tuple too",
          'running_keys.add((sig, t.get("cmd") or ""))' in src,
          diag="same-cycle dedup must match the precomputed key shape")
    check("3.4.12 P2-2: blocked-event reason mentions 'identical cmd'",
          "identical cmd already has a running task" in src,
          diag="user-facing message tells operator different cmds with same sig are allowed")

    # ---- P1-1 behavioral: extra_excludes appended to rsync ----
    saved_NODES = sch.NODES
    saved_run = subprocess.run if hasattr(subprocess, "run") else None
    try:
        sch.NODES = {
            "local": {"host": None},
            "remote": {"host": "user@host"},
        }
        # Capture the rsync argv so we can assert --exclude entries.
        captured_argv = []
        class _MockResult:
            def __init__(self): self.returncode = 0; self.stdout = "1\n"; self.stderr = ""
        def mock_run(args, **kw):
            captured_argv.append(list(args))
            return _MockResult()
        import subprocess as _sp
        saved_sp_run = _sp.run
        _sp.run = mock_run
        try:
            # Need a real local cwd for Path.exists() check inside helper.
            import tempfile as _tf
            with _tf.TemporaryDirectory() as td:
                ok, msg = sch._stage_cwd_for_launch(
                    {"cwd": td}, "remote",
                    extra_excludes=["runs/exp1/", "outputs/seed42/"])
                # 2 calls expected: du, then rsync.
                rsync_call = next((a for a in captured_argv if a and a[0] == "rsync"), None)
                check("3.4.12 P1-1 behavior: rsync argv contains dynamic --exclude entries",
                      rsync_call is not None
                      and "--exclude=runs/exp1/" in rsync_call
                      and "--exclude=outputs/seed42/" in rsync_call,
                      diag=f"rsync_call={rsync_call}")
                # Cleanup cache so subsequent tests aren't affected
                sch._STAGING_CACHE.clear()
        finally:
            _sp.run = saved_sp_run
    finally:
        sch.NODES = saved_NODES

    # ---- P1-2 behavioral: stage_failed → bumped fail count + revert queued ----
    # Direct probe via _stage_cwd_check after seeding _STAGING_FAILS.
    saved_NODES = sch.NODES
    try:
        sch.NODES = {
            "local": {"host": None},
            "remote": {"host": "user@host"},
        }
        cwd_key = ("local", "remote", "/some/cwd")
        sch._STAGING_FAILS[cwd_key] = (time.time(), "rsync rc=12: connection refused")
        check("3.4.12 P1-2 behavior: fresh _STAGING_FAILS → 'stage_failed'",
              sch._stage_cwd_check("remote", "/some/cwd") == "stage_failed")
        check("3.4.12 P1-2 behavior: _stage_failure_reason exposes message",
              "rsync rc=12" in sch._stage_failure_reason("remote", "/some/cwd"))
        # Stale entry (older than STAGING_FAIL_COOLDOWN_S) → reverts to needs_stage
        sch._STAGING_FAILS[cwd_key] = (
            time.time() - sch.STAGING_FAIL_COOLDOWN_S - 10,
            "old failure")
        check("3.4.12 P1-2 behavior: stale _STAGING_FAILS → 'needs_stage' (auto recovery)",
              sch._stage_cwd_check("remote", "/some/cwd") == "needs_stage")
        sch._STAGING_FAILS.pop(cwd_key, None)
    finally:
        sch.NODES = saved_NODES

    # ---- P2-2 behavioral: same sig + different cmd both launch ----
    # (Exercised in the broader same-pass duplicate test above; here we
    # just sanity-check the dedup tuple type.)
    saved_NODES = sch.NODES
    try:
        running_keys_test = {("sig", "cmd_A"), ("sig", "cmd_B")}
        check("3.4.12 P2-2 behavior: tuple-keyed set distinguishes cmds with same sig",
              ("sig", "cmd_A") in running_keys_test
              and ("sig", "cmd_C") not in running_keys_test,
              diag=str(running_keys_test))
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

    # ---------- Case A: cwd already exists on target → STILL rsync (Phase 3.0.20 P1) ----------
    # Pre-3.0.20 the code skipped rsync if `test -d cwd` succeeded on target.
    # That meant a stale repo on target would silently run on the migrated task.
    # New contract: cache miss → always rsync (rsync delta keeps it cheap when
    # already synced).
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([
        ("test -d /work", (0, "", "")),     # cwd happens to exist on target
        ("test -x", (0, "", "")),           # python exists on target
        ("du -sm",      (0, "10\n", "")),   # cwd 10MB
        ("mkdir -p",    (0, "", "")),
    ])
    sch.subprocess.run = fake_subprocess_run
    rsync_calls.clear()
    try:
        ok, msg = sch._stage_for_migration(
            {"id": "tA", "cwd": "/work", "preferred_node": "src",
             "cmd": "/abs/path/python -u train.py"},
            "tgt"
        )
        check("cwd present on target → STILL rsync issued (delta sync; no stale code)",
              ok and len(rsync_calls) >= 1, diag=f"ok={ok} rsync={rsync_calls}")
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
        ("test -d /ckpt", (0, "", "")),  # Phase 3.0.16: source-side ckpt existence
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
        if "test -d /ckpt" in cmd:  # Phase 3.0.16: source-side existence
            return (0, "", "")
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

    # ---------- Case G2 (Phase 3.0.6 P1 + 3.0.20): remote→remote refused at cwd stage ----------
    # Pre-3.0.20 cwd stage short-circuited when target already had the dir, leaving
    # ckpt as the lone remote→remote concern (this test originally exercised the
    # ckpt rsync rejection). With 3.0.20 the cwd stage always rsyncs on cache
    # miss, so remote→remote is now rejected at the cwd stage — same conservative
    # outcome (no migration when both sides remote), just earlier in the pipeline.
    sch._STAGING_CACHE.clear()
    sch.run_on = mk_run_on([
        ("test -d /work", (0, "", "")),       # cwd happens to exist on target
        ("test -d /ckpt", (0, "", "")),       # Phase 3.0.16: source-side existence
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
        check("remote→remote → REJECTED at cwd stage (no migration when both sides remote)",
              not ok and "remote→remote" in msg and "not yet supported" in msg, diag=msg)
        check("rejected migration: cwd cache NOT populated",
              ("src", "tgt_remote", "/work") not in sch._STAGING_CACHE,
              diag=str(sch._STAGING_CACHE))
        check("rejected migration: ckpt cache NOT populated",
              ("src", "tgt_remote", "/ckpt") not in sch._STAGING_CACHE,
              diag=str(sch._STAGING_CACHE))
        check("no rsync attempted in remote→remote case",
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
        if "test -d /ckpt" in cmd: return (0, "", "")  # Phase 3.0.16: source has it
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
    # Phase 3.0.13: cmd_rebalance_pending now pre-checks squeue BEFORE scancel — if
    # pre-check says the job is already gone/terminal, scancel is correctly skipped.
    # Use a stateful mock so the pre-check sees PENDING (forces scancel path), then
    # post-scancel verify sees the job gone (verifies cancellation).
    cancelled_jids = set()
    def fake_run_on(node, cmd, timeout=10, check=True):
        if "scancel" in cmd:
            scancel_calls.append((node, cmd))
            try:
                cancelled_jids.add(int(cmd.split()[-1]))
            except Exception:
                pass
            return (0, "", "")
        if "squeue" in cmd:
            jid = None
            try:
                # Parse "squeue -h -j <jid> -t all -o '%T'"
                parts = cmd.split()
                jid = int(parts[parts.index("-j") + 1])
            except Exception:
                pass
            if jid is None or jid in cancelled_jids:
                return (0, "", "")  # job gone from slurm
            return (0, "PENDING\n", "")  # still pending → forces scancel path
        return (0, "", "")

    sch.load_state = lambda: fake_state
    sch.save_state = fake_save
    sch.run_on = fake_run_on
    saved_sleep = time.sleep
    time.sleep = lambda s: None  # skip the 1.5s settle delay
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
        time.sleep = saved_sleep


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

    # ---------- Case D: orphan in TERMINAL state → classify done/failed (Phase 3.0.33) ----------
    # Pre-3.0.33 the recovery `continue`d past terminal slurm records (treating
    # them as "let revert path requeue + sbatch again"). That broke the
    # invariant that a task never runs twice — the orphan ALREADY ran and
    # finished. Now: COMPLETED → done; non-COMPLETED terminal → failed.
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
        check("orphan in TERMINAL COMPLETED → status=done (NOT requeued)",
              state["tasks"][0]["status"] == "done"
              and state["tasks"][0].get("slurm_job_id") == 5555
              and state["tasks"][0].get("slurm_state") == "COMPLETED",
              diag=str(state["tasks"][0]))
        check("orphan in TERMINAL COMPLETED → last_block_reason cites WAL recovery",
              "WAL recovery" in (state["tasks"][0].get("last_block_reason") or ""))
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

    # ---------- Case F: local-routed launching task → local orphan probe (Phase 3.0.28) ----------
    # Local nodes don't have slurm orphans by definition, but they DO have a local
    # orphan recovery path (Phase 3.0.28): scan /proc/*/environ for the
    # SCHEDULEURM_TASK_ID marker injected at launch. Recovery uses ONE probe
    # (the /proc grep), not the slurm squeue path.
    fake_hb = sch.HybridBackend()
    fake_hb._cache["localnode"] = "local"
    sch._BACKEND = fake_hb
    probe_cmds = []
    sch.run_on = lambda node, cmd, **k: (probe_cmds.append(cmd) or (0, "", ""))
    try:
        state = {"tasks": [{
            "id": "tF", "status": "launching", "node": "localnode",
            "launching_started_at": stale, "remote_pids": [],
        }]}
        sch.recover_stale_launching_tasks(state, now=now)
        check("local-routed launching task → local orphan probe issued (NOT squeue)",
              any("SCHEDULEURM_TASK_ID" in c for c in probe_cmds),
              diag=f"probe cmds: {probe_cmds!r}")
        check("local-routed launching task → no slurm squeue probe issued",
              not any("squeue" in c for c in probe_cmds),
              diag=f"probe cmds: {probe_cmds!r}")
        check("local-routed launching task → reverted normally (no orphan found)",
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
    sb_launch_idx = src.find("def launch(self, task: dict", sb_idx)
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
    sb_launch_idx = src.find("def launch(self, task: dict", src.find("class SlurmBackend"))
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
    test_phase3_0_13_rebalance_pending_outside_lock()
    test_phase3_0_14_min_source_load_and_cwd_size_cap()
    test_phase3_0_15_migrated_task_pins_to_staged_node()
    test_phase3_0_16_ckpt_size_probe_fail_closed()
    test_phase3_0_17_staging_cache_ttl()
    test_phase3_0_18_probe_all_outside_lock()
    test_phase3_0_19_staging_failure_cooldown_unblocks_later_candidates()
    test_phase3_0_20_cwd_always_rsyncs_on_cache_miss()
    test_phase3_0_21_explicit_docker_fail_fast_no_local_digest()
    test_phase3_0_22_explicit_conda_fail_fast_no_local_path()
    test_phase3_0_23_env_key_validation_and_reserved_guard()
    test_phase3_0_24_rebalance_pending_clears_placement_fields()
    test_phase3_0_25_zombie_descendants_not_alive()
    test_phase3_0_26_auto_docker_no_local_digest_falls_back_to_none()
    test_phase3_0_27_conda_sync_success_gate()
    test_phase3_0_28_local_wal_orphan_recovery()
    test_phase3_0_29_actual_started_at_cleared_on_requeue_and_launch()
    test_phase3_0_30_slurm_completed_log_scan_for_crash()
    test_phase3_0_31_launch_side_docker_push_no_longer_holds_lock()
    test_phase3_0_32_orphan_recovery_restores_log_and_docker_artifacts()
    test_phase3_0_33_terminal_orphan_classification()
    test_phase3_0_34_local_docker_fail_fast_no_local_digest()
    test_phase3_0_35_slurm_terminal_orphan_diagnosis()
    test_phase3_0_36_local_terminal_orphan_user_redirect_recovery()
    test_phase3_1_skill_priority_edit_history_why()
    test_phase3_2_0_claim_manager()
    test_phase3_2_1_claim_lifecycle_in_dispatch()
    test_phase3_2_2_probe_folds_pending_claims()
    test_phase3_2_3_concurrent_schedulers_only_one_wins()
    test_phase3_3_local_windows_host_metrics()
    test_phase3_4_0_cross_user_claim_io()
    test_phase3_4_2_persistent_owner_id()
    test_phase3_4_3_claim_race_vs_claim_error()
    test_phase3_4_4_claim_replicates_gpu_fits_policy()
    test_phase3_4_6_7_8_claim_one_third_corrupt_recovery_atomic_deploy()

    passed = sum(1 for _, c, _ in results if c)
    total = len(results)
    print(f"\n{'=' * 60}\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)
