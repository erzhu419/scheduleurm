"""Embedded remote claims script payload deployed to scheduler nodes."""

from __future__ import annotations

CLAIMS_REMOTE_SCRIPT_TEXT = '''#!/usr/bin/env python3
"""scheduleurm remote claims daemon — deployed by _ClaimManager.

Caller wraps invocations in flock so the file mutates atomically. Designed
to work across OS users sharing the same node:
  - claims.json is opened r+w with mode 0666 (set + fchmod); writes happen
    in-place via truncate+write under flock — never via tmp+rename, because
    rename(my_tmp, other_users_file) fails in a sticky directory like /tmp.
  - alive() returns True on PermissionError (kill(pid, 0) → EPERM means
    the process exists but is owned by another user); returning False would
    let one user's GC drop another user's still-running claim and let the
    GPU get over-committed.
"""
import errno, json, math, os, subprocess, sys, time

CLAIMS_FILE = "/tmp/scheduleurm/claims.json"

# Phase 3.4.0: ensure new files are 0666 so any OS user sharing the node
# can update the same claims.json under flock.
os.umask(0)

def alive(pid):
    """True iff the PID exists on this node (regardless of owner).

    Phase 3.4.1 P0 fix: PermissionError from os.kill(pid, 0) means EPERM —
    the process exists but is owned by a different user. Returning False
    here treated other-user claims' PIDs as dead → one user's GC pass would
    drop another user's still-running claim → over-commit. Cross-OS-user
    correctness needs PermissionError → alive=True.
    """
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, owned by another user
    except ValueError:
        return False
    except OSError as e:
        # ESRCH = no such process; anything else (EPERM / EINVAL / ...) means
        # the process exists or the kernel rejected the probe — treat as alive
        # so we never drop a live claim on a benign error.
        return e.errno != errno.ESRCH

def _open_shared_rw():
    """Open claims.json for in-place read+write under flock. Creates with
    0666 if missing; chmods to 0666 even when it existed (so a pre-3.4.0
    file with 0644 / 0600 from a prior writer becomes shareable). Returns
    fd (caller closes)."""
    fd = os.open(CLAIMS_FILE, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except (PermissionError, OSError):
        # Not the owner; mode stays as-is. Sticky dir lets us still truncate
        # and write through the fd we already hold open.
        pass
    return fd

def _read_fd(fd):
    """Read entire content of fd (already at offset 0 or rewind here)."""
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        b = os.read(fd, 65536)
        if not b:
            break
        chunks.append(b)
    return b"".join(chunks).decode("utf-8", errors="replace")

def _write_fd(fd, text):
    """Truncate fd to 0 and write text. In-place rewrite — no rename needed."""
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, text.encode("utf-8"))
    try:
        os.fsync(fd)
    except OSError:
        pass

def load_from_fd(fd):
    """Phase 3.4.7 P2: distinguish 'bootstrap empty' from 'corrupt empty'.
    Setup writes a default `{"version":1,"claims":[]}` so claims.json is
    NEVER 0 bytes during normal operation. A 0-byte file therefore means
    a writer crashed between ftruncate(0) and the subsequent write — we
    must not silently treat that as 'no claims', or the next op would
    over-commit by re-issuing every running task's resources. Same for
    JSON parse failures: surface as ParseError so the caller can return
    a transport-error result; CLAIM_ERROR routes to the regular launch-
    fail path (3.4.3) so the operator sees an actionable failure rather
    than silent data loss.
    """
    text = _read_fd(fd)
    if not text.strip():
        raise RuntimeError(
            "claims.json is empty — likely a partial write (post-crash). "
            "Manual recovery: inspect /tmp/scheduleurm/claims.json on the "
            "node, or `rm` it ONLY if no scheduleurm tasks are running."
        )
    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(
            "claims.json failed to parse (%s) — likely a partial write. "
            "Manual recovery as above." % str(e)[:120])
    if not isinstance(data, dict) or "claims" not in data:
        raise RuntimeError("claims.json missing required schema")
    return data

def save_to_fd(fd, data):
    _write_fd(fd, json.dumps(data))

def is_stale(c, now):
    """Stale = TTL expired AND (no pid OR pid dead). A live pid past TTL is
    treated as orphan-but-still-using-resources; live scheduler should renew."""
    if c.get("expires_at", 0) >= now:
        return False
    pid = c.get("pid")
    if pid and alive(pid):
        return False
    return True

def gc_claims(claims, now):
    return [c for c in claims if not is_stale(c, now)]

def dedup_claims(claims):
    """Phase 3.4.16 P1 fix: prior `claim` op blindly appended a new record
    without removing existing (scheduler_id, task_id) entries. So re-launch /
    migration / heal-resubmit produced multiple claim records for the same
    task — inflating apparent VRAM usage on whichever GPUs the previous
    attempts targeted (real-world incident: jtl110gpu2:GPU1 showed 2140MB
    used while nvidia-smi reported 10MB; 2 stale claims for tasks that had
    since moved to GPU0 had never been cleared).

    `claim` op is now upsert (see remote-script main()). This helper does
    the equivalent on every load so that pre-fix duplicate records still
    sitting in claims.json self-heal on the next op. Keep the LATEST
    record per (scheduler_id, task_id) by claimed_at; older duplicates
    are dropped.
    """
    by_key = {}
    for c in claims:
        key = (c.get("scheduler_id"), c.get("task_id"))
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = c
            continue
        try:
            cur_ts = float(c.get("claimed_at") or 0)
            prev_ts = float(prev.get("claimed_at") or 0)
        except (TypeError, ValueError):
            cur_ts = prev_ts = 0
        if cur_ts > prev_ts:
            by_key[key] = c
    return list(by_key.values())

def record_key(c):
    return (c.get("scheduler_id"), c.get("task_id"))

def gc_intents(intents, now):
    """Drop abandoned queue intents. Intents are pre-launch tickets, so a
    dead scheduler has no PID to probe; expiry alone is the recovery signal."""
    return [i for i in intents if float(i.get("expires_at", 0) or 0) >= now]

def upsert_intent(intents, payload, now):
    """Insert/refresh this task's shared queue intent while preserving its
    original intent_at timestamp. That timestamp is the cross-scheduler FIFO
    order; refreshing retries must not jump the task to the back."""
    key = record_key(payload)
    out = []
    found = None
    for i in intents:
        if record_key(i) == key:
            found = i
        else:
            out.append(i)
    intent = dict(payload)
    intent["intent_at"] = float((found or {}).get("intent_at") or now)
    intent["expires_at"] = float(payload.get("intent_expires_at") or payload.get("expires_at") or now + 180)
    intent["pid"] = None
    out.append(intent)
    out.sort(key=lambda x: (float(x.get("intent_at", 0) or 0), str(x.get("scheduler_id")), str(x.get("task_id"))))
    return out

def capacity_conflicts(payload, active_claims, capacity):
    """Return conflicts if payload cannot fit next to active_claims.

    Shared by capacity rejection and FIFO/backfill checks so the remote
    arbiter uses exactly one resource model.
    """
    used_cpu = sum(
        0 if c.get("ignore_cpu_capacity") else c.get("cpu_cores", 0)
        for c in active_claims
    )
    used_ram = sum(c.get("ram_mb", 0) for c in active_claims)
    per_gpu = {}
    for c in active_claims:
        g = c.get("gpu_idx")
        if g is not None:
            per_gpu[str(g)] = per_gpu.get(str(g), 0) + c.get("vram_mb", 0)
    cpu_need = payload.get("cpu_cores", 0)
    ram_need = payload.get("ram_mb", 0)
    gpu_idx = payload.get("gpu_idx")
    vram_need = payload.get("vram_mb", 0)
    cpu_cap = capacity.get("cpu_cores", 0)
    ram_cap = capacity.get("ram_mb", 0)
    gpu_caps = capacity.get("gpu_vram_mb", {}) or {}
    # Phase 3.4.4: cross-scheduler enforcement of local placement policy.
    max_per_task = capacity.get("max_vram_per_task")
    vram_margin = int(capacity.get("vram_margin_mb", 0) or 0)
    third_rule = bool(capacity.get("third_pack_rule"))
    conflicts = []
    ignore_cpu = bool(payload.get("ignore_cpu_capacity"))
    if not ignore_cpu and used_cpu + cpu_need > cpu_cap:
        conflicts.append("cpu: need %d + claimed %d > cap %d" % (cpu_need, used_cpu, cpu_cap))
    if used_ram + ram_need > ram_cap:
        conflicts.append("ram: need %dMB + claimed %dMB > cap %dMB" % (ram_need, used_ram, ram_cap))
    if gpu_idx is not None:
        gkey = str(gpu_idx)
        gcap = int(gpu_caps.get(gkey, 0))
        gused = per_gpu.get(gkey, 0)
        # Per-task VRAM cap (e.g. local enforces ≤ 4GB per task).
        if max_per_task is not None and vram_need > int(max_per_task):
            conflicts.append("gpu%s: need %dMB > per-task cap %dMB" % (
                gkey, vram_need, int(max_per_task)))
        # Total cap (raw VRAM math).
        if gused + vram_need > gcap:
            conflicts.append("gpu%s: need %dMB + claimed %dMB > cap %dMB" % (
                gkey, vram_need, gused, gcap))
        # VRAM margin: leave at least margin MB free after this claim.
        if gcap > 0 and (gcap - gused - vram_need) < vram_margin:
            conflicts.append("gpu%s: post-claim free %dMB < margin %dMB" % (
                gkey, gcap - gused - vram_need, vram_margin))
        # 1/3 packing rule. Match local _gpu_fits semantics:
        #   - empty/noise-only GPU: allow the first task even if it is large
        #   - occupied GPU: don't add work if it is already at the freeze line
        #     OR this claim would cross it. The line includes a small grace
        #     window because 1/3 is a heuristic, not a physical OOM boundary.
        # There is deliberately no small-task exemption.
        if third_rule and not bool(payload.get("ignore_one_third_pack_rule")) and gcap > 0:
            third = gcap // 3
            grace = max(0, int(capacity.get("one_third_grace_mb", 0) or 0))
            freeze = min(gcap, third + grace)
            empty_used = max(1, int(capacity.get("gpu_empty_used_mb", 200) or 200))
            if gused >= empty_used and (gused >= freeze or gused + vram_need >= freeze):
                conflicts.append(
                    "gpu%s: occupied claim would cross 1/3+grace line "
                    "(claimed %dMB + need %dMB ≥ %dMB of %dMB, packing rule)" % (
                        gkey, gused, vram_need, freeze, gcap))
    return conflicts

def _read_live_snapshot(capacity):
    """Best-effort live resource view while the claims lock is held.

    Tests may pass capacity["live_snapshot"]; production reads /proc and
    nvidia-smi on the target node. Failure means "no extra external usage"
    rather than a claim-script crash.
    """
    snap = capacity.get("live_snapshot")
    if isinstance(snap, dict):
        return snap
    out = {"loadavg": None, "mem_available_mb": None, "gpu_used_mb": {}}
    try:
        with open("/proc/loadavg") as f:
            out["loadavg"] = float((f.read().split() or ["0"])[0])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    bits = line.split()
                    if len(bits) >= 2:
                        out["mem_available_mb"] = int(int(bits[1]) / 1024)
                    break
    except Exception:
        pass
    try:
        raw = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null",
            shell=True, timeout=float(capacity.get("live_check_timeout_s", 3) or 3),
        ).decode("utf-8", "ignore")
        used = {}
        for line in raw.splitlines():
            bits = [b.strip() for b in line.split(",")]
            if len(bits) < 2:
                continue
            try:
                used[str(int(bits[0]))] = int(bits[1])
            except Exception:
                continue
        out["gpu_used_mb"] = used
    except Exception:
        pass
    return out

def live_external_claims(active_claims, capacity):
    """Represent non-scheduleurm/manual live usage as synthetic claims."""
    if not capacity.get("live_check"):
        return []
    snap = _read_live_snapshot(capacity)
    externals = []
    running_claims = [c for c in active_claims if c.get("pid")]
    claimed_cpu = sum(
        0 if c.get("ignore_cpu_capacity") else int(c.get("cpu_cores") or 0)
        for c in running_claims
    )
    claimed_ram = sum(int(c.get("ram_mb") or 0) for c in running_claims)
    claimed_gpu = {}
    for c in running_claims:
        g = c.get("gpu_idx")
        if g is not None:
            claimed_gpu[str(g)] = claimed_gpu.get(str(g), 0) + int(c.get("vram_mb") or 0)

    ext_cpu = 0
    try:
        if snap.get("loadavg") is not None:
            # Match the node probes' whole-core estimate.  ceil() turns any
            # sub-core daemon/SSH noise into a full external claim, making a
            # legitimate full-capacity CPU batch impossible to launch.
            ext_cpu = max(0, int(round(float(snap.get("loadavg")))) - claimed_cpu)
    except Exception:
        pass
    ext_ram = 0
    try:
        avail = snap.get("mem_available_mb")
        cap_ram = int(capacity.get("ram_mb", 0) or 0)
        if avail is not None and cap_ram > 0:
            live_used = max(0, cap_ram - int(avail))
            ext_ram = max(0, live_used - claimed_ram)
    except Exception:
        pass
    if ext_cpu or ext_ram:
        externals.append({
            "scheduler_id": "__external__", "task_id": "__cpu_ram_live__",
            "gpu_idx": None, "vram_mb": 0,
            "cpu_cores": ext_cpu, "ram_mb": ext_ram,
            "pid": -1,
        })

    gpu_used = snap.get("gpu_used_mb") or {}
    if isinstance(gpu_used, dict):
        for g, used in gpu_used.items():
            try:
                ext_vram = max(0, int(used) - int(claimed_gpu.get(str(g), 0)))
            except Exception:
                continue
            if ext_vram <= 0:
                continue
            externals.append({
                "scheduler_id": "__external__", "task_id": "__gpu%s_live__" % str(g),
                "gpu_idx": int(g), "vram_mb": ext_vram,
                "cpu_cores": 0, "ram_mb": 0,
                "pid": -1,
            })
    return externals

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "missing op"}))
        return
    op = sys.argv[1]
    payload = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    capacity = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    os.makedirs("/tmp/scheduleurm", exist_ok=True)
    fd = _open_shared_rw()
    try:
        data = load_from_fd(fd)
    except RuntimeError as e:
        # Phase 3.4.7 P2: parse / empty failure means a writer crashed
        # mid-truncate. Don't silently treat as no claims — return an
        # error so caller's claim() routes to CLAIM_ERROR (3.4.3) and
        # the operator sees the corrupted-state message.
        try:
            os.close(fd)
        except OSError:
            pass
        print(json.dumps({"ok": False,
                            "error": "claims_corrupt: " + str(e)}))
        return
    claims = data.get("claims", [])
    intents = data.get("intents", [])
    now = time.time()
    fresh = gc_claims(claims, now)
    # Phase 3.4.16 P1 fix: self-heal pre-fix duplicate (scheduler_id, task_id)
    # records left over from earlier claim() that didn't upsert. See
    # dedup_claims docstring for the bug + repro.
    fresh = dedup_claims(fresh)
    fresh_intents = gc_intents(intents, now)
    if op == "claim":
        fresh_intents = upsert_intent(fresh_intents, payload, now)
        key = record_key(payload)
        external = live_external_claims(fresh, capacity)
        active = fresh + external
        strict_after = float(capacity.get("fifo_strict_after_s", 0) or 0)
        # FIFO-with-backfill: older intents get priority only when the
        # current claim would delay them. If an older task cannot fit now,
        # or if both older and current fit together, this task may backfill.
        for older in fresh_intents:
            if record_key(older) == key:
                break
            if strict_after > 0:
                try:
                    older_wait = now - float(older.get("intent_at", now) or now)
                except Exception:
                    older_wait = 0
                older_can_ever_fit = not capacity_conflicts(older, [], capacity)
                older_fits_empty_after_current = not capacity_conflicts(older, [payload], capacity)
                if older_wait >= strict_after and older_can_ever_fit and not older_fits_empty_after_current:
                    data["claims"] = fresh
                    data["intents"] = fresh_intents
                    save_to_fd(fd, data)
                    print(json.dumps({
                        "ok": False,
                        "conflict": "fifo-strict: older intent %s/%s waited %.0fs" % (
                            older.get("scheduler_id"), older.get("task_id"), older_wait),
                        "claims_seen": len(fresh),
                        "intents_seen": len(fresh_intents),
                        "external_claims_seen": len(external),
                    }))
                    return
            older_fits_now = not capacity_conflicts(older, active, capacity)
            if not older_fits_now:
                continue
            older_fits_after_current = not capacity_conflicts(older, active + [payload], capacity)
            if not older_fits_after_current:
                data["claims"] = fresh
                data["intents"] = fresh_intents
                save_to_fd(fd, data)
                print(json.dumps({
                    "ok": False,
                    "conflict": "fifo: older intent %s/%s has priority" % (
                        older.get("scheduler_id"), older.get("task_id")),
                    "claims_seen": len(fresh),
                    "intents_seen": len(fresh_intents),
                    "external_claims_seen": len(external),
                }))
                return
        conflicts = capacity_conflicts(payload, active, capacity)
        if conflicts:
            data["claims"] = fresh
            data["intents"] = fresh_intents
            save_to_fd(fd, data)
            print(json.dumps({"ok": False, "conflict": "; ".join(conflicts),
                              "claims_seen": len(fresh),
                              "external_claims_seen": len(external)}))
            return
        # Phase 3.4.16 P1 fix: claim is now UPSERT by (scheduler_id, task_id).
        # Pre-fix this was a blind append, which produced duplicate records
        # across re-launch / migration / heal-resubmit (each adding a new
        # entry, none clearing the prior one). Result: phantom VRAM usage
        # on stale GPU indices that the task had since moved away from.
        # Removing all prior records for the same (sid, tid) before the
        # append makes claim() idempotent and migration-safe.
        fresh = [c for c in fresh if record_key(c) != key]
        fresh.append(payload)
        data["claims"] = fresh
        data["intents"] = [i for i in fresh_intents if record_key(i) != key]
        save_to_fd(fd, data)
        print(json.dumps({"ok": True}))
    elif op == "release":
        sid = payload.get("scheduler_id")
        tid = payload.get("task_id")
        kept = [c for c in fresh if not (c.get("scheduler_id") == sid and c.get("task_id") == tid)]
        kept_intents = [i for i in fresh_intents if not (i.get("scheduler_id") == sid and i.get("task_id") == tid)]
        data["claims"] = kept
        data["intents"] = kept_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "removed": len(fresh) - len(kept),
                          "removed_intents": len(fresh_intents) - len(kept_intents)}))
    elif op == "update_pid":
        sid = payload.get("scheduler_id")
        tid = payload.get("task_id")
        pid = payload.get("pid")
        updated = 0
        for c in fresh:
            if c.get("scheduler_id") == sid and c.get("task_id") == tid:
                c["pid"] = int(pid) if pid else None
                updated += 1
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "updated": updated}))
    elif op == "renew":
        sid = payload.get("scheduler_id")
        tid = payload.get("task_id")
        new_exp = payload.get("expires_at")
        renewed = 0
        for c in fresh:
            if c.get("scheduler_id") == sid and c.get("task_id") == tid:
                if new_exp:
                    c["expires_at"] = float(new_exp)
                renewed += 1
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "renewed": renewed}))
    elif op == "renew_many":
        # Bulk renew for one scheduler. Watcher sends our running task ids
        # every cycle; the same call also implicitly GC's stale entries
        # because gc_claims already ran above.
        sid = payload.get("scheduler_id")
        tids = set(payload.get("task_ids") or [])
        new_exp = payload.get("expires_at")
        records = payload.get("records") or {}
        if not isinstance(records, dict):
            records = {}
        renewed = 0
        updated = 0
        for c in fresh:
            if c.get("scheduler_id") == sid and c.get("task_id") in tids:
                if new_exp:
                    c["expires_at"] = float(new_exp)
                rec = records.get(str(c.get("task_id")))
                if isinstance(rec, dict):
                    for key in ("gpu_idx", "vram_mb", "cpu_cores", "ram_mb", "pid",
                                "ignore_cpu_capacity", "ignore_one_third_pack_rule"):
                        if key not in rec:
                            continue
                        val = rec.get(key)
                        try:
                            if key in ("ignore_cpu_capacity", "ignore_one_third_pack_rule"):
                                c[key] = bool(val)
                            elif key in ("gpu_idx", "pid"):
                                c[key] = int(val) if val is not None else None
                            else:
                                c[key] = max(0, int(val))
                        except Exception:
                            pass
                    updated += 1
                renewed += 1
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "renewed": renewed,
                          "updated": updated,
                          "removed_stale": len(claims) - len(fresh),
                          "removed_stale_intents": len(intents) - len(fresh_intents)}))
    elif op == "gc":
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "removed": len(claims) - len(fresh),
                          "removed_intents": len(intents) - len(fresh_intents)}))
    elif op == "list":
        if len(fresh) != len(claims) or len(fresh_intents) != len(intents):
            data["claims"] = fresh
            data["intents"] = fresh_intents
            save_to_fd(fd, data)
        print(json.dumps({"ok": True, "claims": fresh, "intents": fresh_intents,
                          "removed_stale": len(claims) - len(fresh),
                          "removed_stale_intents": len(intents) - len(fresh_intents)}))
    else:
        print(json.dumps({"ok": False, "error": "unknown op " + repr(op)}))
    try:
        os.close(fd)
    except OSError:
        pass

main()
'''
