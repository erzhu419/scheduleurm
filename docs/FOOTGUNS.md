# Footguns

A war-stories changelog. Every entry corresponds to a real bug that hit production, a fix, and a regression test in `skill/test_regression.py` that prevents it from happening again. The user's principle: **每个 footgun 必须落地为 regression test + scheduler guard + SKILL.md note**. Pay once, never twice.

## OOM_PATTERNS bare "Killed" → 50h compute lost

**Bug.** `OOM_PATTERNS = (..., "Killed", ...)` matched substrings in our own diag text like "task killed mid-training". Tasks that were manually `kill -9`'d during normal operation got classified as OOM → routed to escalation (no retry) instead of normal requeue → 4 wsrl/s1024 tasks lost ~14h each = 50h+ compute gone before the user noticed.

**Fix.** Tighten to actual kernel OOM-killer message format only:
```python
OOM_PATTERNS = ("CUDA out of memory", "out of memory", "MemoryError",
                "Killed process",  # kernel format ("Killed process N (cmd) total-vm:...")
                "oom-kill", "oom_reaper")
```

**Regression test.** `test_oom_classify_no_false_positive` — feeds `_classify_failure` with strings like "task killed mid-training" and asserts the result is NOT OOM.

## Test-state leak into live queue.json (twice)

**Bug.** `test_dispatch_skips_duplicate_signature` called `sch._do_dispatch(fake_state)` to verify dedup logic. Fine in principle — but `_do_dispatch` calls `save_state(state)` on the WAL transition, which writes to the **real** `queue.json` because `sch.QUEUE_FILE` was the real path. Result: 1600 live tasks → 3-20 fake test tasks in queue.json. Twice. (Recovered both times from `queue.json.corrupt-<ts>` backup.)

**Fix.** Stub `sch.save_state = lambda s: None` in any test that calls `_do_dispatch` with fake state.

**Regression test.** `test_no_test_writes_live_queue_with_fake_state` — uses `ast.walk` on the test file source to find any test function calling `sch._do_dispatch(...)` without a sibling `sch.save_state = lambda` line. Fails the suite if found. This catches the regression at the source-code level, not at runtime.

## stdout 0-byte log false-positive

**Bug.** Some tasks completed successfully but their stdout log was 0 bytes. Watcher's "log empty + no success marker" rule false-classified them as crashes → re-launched the same task on top of itself. Root cause: tasks invoked python without `-u` (unbuffered) and finished too fast to flush stdout buffer to disk.

**Fix.** `_inject_python_u()` at launch — rewrites `python -u` (or inserts `-u`) into the cmd before launching, so stdout is unbuffered from the start.

**Regression test.** `test_inject_python_u` — verifies the rewrite handles `python script.py`, `/abs/path/python script.py`, `python -u script.py` (already has it), and `conda run python script.py` correctly.

## Docker `--gpus all` GPU leak

**Bug.** Earlier versions used `docker run --gpus all` so the container saw every GPU on the host. A task pinned to GPU 1 by the scheduler could happily use GPU 0 too — silent placement violation, contention with other tasks. Codex review caught this.

**Fix.**
```python
args += ["--gpus", f"device={gpu_idx}"]      # hard-pin at container level
args += ["-e", "CUDA_VISIBLE_DEVICES=0"]     # inside container, pinned device enumerates as 0
```
Plus `--memory ${ram_mb}m --cpus ${cpu_cores}` so the container also honors scheduler's RAM/CPU budgets at the cgroup level.

**Regression test.** `test_env_deploy_wrap_docker` and `test_env_deploy_doc_matches_code` — assert no `--gpus all` in source or docstring; assert `device=N` form is present.

## Container PID tracking via containerd-shim

**Bug.** `docker run` returns immediately with the **client** PID. The actual container processes are children of `containerd-shim`, NOT children of the docker-run client. So scheduler's process-tree probing (`pgrep -P`) found nothing → task immediately marked dead → re-launched on top of running container.

**Fix.** After `docker run`, query `docker inspect --format '{{.State.Pid}}' <container_name>` (with retry loop, since the container takes a moment to be inspectable) → use that PID as the root for tracking.

**Regression test.** `test_kill_includes_docker_for_named_container` — verifies the kill path uses `docker kill <name>` for tasks with a container name set, not just host-side process kills.

## Image push inside state_lock blocked watcher

**Bug.** First-time docker dispatch needed to push the image to the remote node (`docker save | ssh node docker load`, ~5-30 min for ML images). This was happening inside the state lock. Result: watcher held the lock for 30 min, no other dispatch / status / cancel could proceed; user thought scheduler was hung.

**Fix.** `_preload_docker_images_outside_lock` — called BEFORE acquiring the state lock in both `cmd_dispatch` and `_watch_iteration`. Enumerates queued tasks, identifies needed images, pushes outside the lock.

**Regression test.** `test_preload_uses_spec_image_or_image_field` — verifies preload doesn't skip on `not image` before parsing `--env-spec` (some tasks encode the image inline in `--env-spec docker:IMAGE` rather than separate `--image` field).

## Image digest drift bypassing the launch path

**Bug.** Preload at dispatch correctly checked image digest and re-pushed on drift. But the **launch** path also called `has_image()` as a safety net — without passing `local_digest`. So if preload ran without a digest check (older code path), and the launch path's `has_image()` returned True on tag presence alone, a stale image would run silently.

**Fix.** Launch path now fetches local digest before calling `has_image()`:
```python
local_digest = get_image_digest(run_on, "local", image)
if not has_image(run_on, node, image, local_digest=local_digest):
    push_image(node_host, image)
```

**Regression test.** `test_launch_path_uses_digest_check` — greps source for `has_image(...local_digest=...)` in launch path.

## Worst-fit placement → fragmentation

**Bug.** Original placement scored GPUs by `-free_mb` (largest free first) — classic "worst-fit". Over time, every GPU ended up half-full. New tasks couldn't fit anywhere even though plenty of total free VRAM existed. Codex flagged this.

**Fix.** Best-fit warm-first scoring:
```python
score = (warm_first_bonus, fits_remaining_after_placement)
# warm_first_bonus = -1 if GPU already has tasks, else 0
# fits_remaining = free - task_vram - VRAM_MARGIN_MB
```
Places on the warmest fitting GPU with the smallest leftover. Keeps GPUs either full or empty, not all half-full.

**Regression test.** `test_pick_placement_best_fit_warm_first` — synthetic 3-GPU layout, asserts placement order matches best-fit warm-first.

## Inflated history pinning estimates at 5GB

**Bug.** `history_record` used `cur["ram_mb"] = max(cur["ram_mb"], peak_ram_mb)`. A single anomalous run (e.g. one bad seed with full replay buffer + 10× ensemble = 5GB) pinned all future estimates at 5GB, even though typical runs only used 1-2GB. Result: queued tasks blocked on RAM headroom even though their **actual** need was way lower than the estimate.

**Fix.** p80 of last 10 samples:
```python
HISTORY_SAMPLES_PER_SIG = 10
HISTORY_PERCENTILE = 80

def _fold(field, samples_field, new_sample):
    samples = cur.get(samples_field) or []
    if not samples and cur.get(field):
        samples = [int(cur[field])]    # legacy migration
    samples.append(int(new_sample))
    samples = samples[-HISTORY_SAMPLES_PER_SIG:]
    cur[samples_field] = samples
    cur[field] = _percentile(samples, HISTORY_PERCENTILE)
```

**Regression test.** `test_history_record_p80_outlier_resistance` — 9 typical runs + 1 outlier; asserts estimate stays near typical. Plus legacy single-value migration test, plus sliding-window eviction test.

## peak_vram > 0 + no success marker → "ambiguous; assumed normal"

**Bug.** `diagnose_terminal` had a 4-rule decision tree. For a task with `peak_vram_mb > 0` (had been allocating GPU memory = was definitely past startup) but no success marker in tail (didn't finish cleanly) — original logic fell into "ambiguous, assume normal". This silently marked tasks as `done` even though they had clearly crashed mid-training. Re-runs were not triggered.

**Fix.** New diagnose branch:
```python
if peak_vram_mb > 0 and not success_marker_in_tail:
    return {"is_crash": True, "reason": "task had GPU activity but no success marker — crashed during training"}
```

**Regression test.** `test_diagnose_peak_vram_implies_crash_without_success`.

## Stale `launching` WAL state after watcher restart

**Bug.** Watcher set status="launching" before ssh, then ssh hung, then watcher process was killed. Restart would see a record with status="launching" forever — task neither re-dispatched nor cleaned up.

**Fix.** Watcher startup runs WAL recovery: any `launching` record older than `LAUNCHING_RESET_S` (60s) reverts to `queued`. Logged as `wal_revert` event so operator can see what happened.

**Regression test.** `test_launching_state_field_persistence` — verifies dispatch sets launching+save before launch, and source contains the recovery branch.

## `ram_headroom_frac` denominator overestimating WSL RAM

**Bug.** WSL2 advertises a higher MemTotal than what's usable (host has 64GB, WSL2 sees 32GB). `NODES["local"]["ram_mb"] = 56*1024` (host's view) was being used as the denominator for headroom. Headroom check passed even though probed free RAM was already low. WSL OOM still triggered.

**Fix.** Headroom denominator uses `min(declared_ram_mb, probed_MemTotal)`. Caps the over-declaration.

**Regression test.** `test_probe_ram_budget_cap`.

## CPU-saturated remote not auto-skipped

**Bug.** Remote node at 100% CPU was still being targeted by dispatch. New task got dispatched, sat blocked behind 12 other CPU-bound procs, made no progress for hours.

**Fix.** Probe rejects placement if `loadavg / cpu_cores >= 1.0`. Dispatch routes around CPU-saturated nodes instead of piling on.

**Regression test.** `test_ram_placement_check` (covers RAM; CPU is similar).

## Conda env not deployed to remote

**Bug.** When user submits `conda run -n myenv python ...`, scheduler launches via `bash -lc` — but `bash -lc` doesn't source `~/.bashrc` (where conda init lives). Task fails with "command not found: conda". User had to manually `conda env create` on every remote.

**Fix.**
1. SKILL.md rule: don't put `conda` in `--cmd`; use absolute python path instead (`/home/u/anaconda3/envs/X/bin/python`).
2. `--env-spec conda:/abs/path` triggers rsync of the local conda env to the same absolute path on remote at submit time. Idempotent — re-syncs are fast.

**Regression test.** `test_env_spec_conda_parsing`, `test_conda_preload_helpers`, `test_preload_handles_conda_spec`.

## Resumable task killed → no resume

**Bug.** User cancels a long-running task because they want to redirect it to another node. Re-submits it. New task starts from step 0, even though the previous task wrote a checkpoint at step 30000. User loses 8 hours of training because they thought "cancel + resubmit = continue".

**Fix.** Resumable-task safety memory rule: **never kill a task with a usable checkpoint**. Patch the queue.json field in place, or use `--require-node` flip to force redirection on next dispatch. Cancel is the last resort, not the routing tool.

**No regression test** — this is a process rule, not a code path. The task record always has `ckpt_dir` so manual auditing is possible.

## ckpt_dir cross-signature conflict

**Bug.** Two tasks with different signatures but the same `--ckpt-dir`. They'd both happily write to the same directory and clobber each other.

**Fix.** Dispatch refuses placement if a queued task's `ckpt_dir` is currently used by a running task with a different signature. Adds the cross-sig check alongside the same-sig race-guard.

**Regression test.** `test_ckpt_dir_cross_sig_conflict`.

---

The pattern is consistent: **a footgun fires once → the fix lands as code + regression test + memory note + (sometimes) SKILL.md guidance**. The regression suite is now ~290 checks. Most of those checks would never have been written if the bug hadn't actually happened. That's by design — speculative tests rot, regression tests don't.
