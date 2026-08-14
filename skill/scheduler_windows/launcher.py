"""Embedded Windows launcher scripts used by the Windows backend."""

from __future__ import annotations


WINDOWS_LAUNCHER = r'''
import base64, ctypes, json, os, subprocess, sys, time
from ctypes import wintypes

CREATE_NEW_PROCESS_GROUP = 0x00000200
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPTHREAD = 0x00000004
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
THREAD_SET_INFORMATION = 0x0020

k32 = ctypes.windll.kernel32

class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]

class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
    ]

class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [
        ("Mask", ctypes.c_size_t),
        ("Group", wintypes.WORD),
        ("Reserved", wintypes.WORD * 3),
    ]

k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32FirstW.restype = wintypes.BOOL
k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.restype = wintypes.BOOL
k32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32First.restype = wintypes.BOOL
k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32Next.restype = wintypes.BOOL
k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.OpenThread.restype = wintypes.HANDLE
k32.SetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(GROUP_AFFINITY), ctypes.POINTER(GROUP_AFFINITY)]
k32.SetThreadGroupAffinity.restype = wintypes.BOOL
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.CloseHandle.restype = wintypes.BOOL
k32.GetActiveProcessorGroupCount.argtypes = []
k32.GetActiveProcessorGroupCount.restype = wintypes.WORD
k32.GetActiveProcessorCount.argtypes = [wintypes.WORD]
k32.GetActiveProcessorCount.restype = wintypes.DWORD

def process_parent_map():
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return {}
    out = {}
    pe = PROCESSENTRY32W()
    pe.dwSize = ctypes.sizeof(pe)
    try:
        ok = k32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            out[int(pe.th32ProcessID)] = int(pe.th32ParentProcessID)
            ok = k32.Process32NextW(snap, ctypes.byref(pe))
    finally:
        k32.CloseHandle(snap)
    return out

def descendants(root):
    parents = process_parent_map()
    todo = [int(root)]
    seen = set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for pid, ppid in list(parents.items()):
            if ppid == cur and pid not in seen:
                todo.append(pid)
    return seen

def threads_for_pid(pid):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snap == INVALID_HANDLE_VALUE:
        return []
    out = []
    te = THREADENTRY32()
    te.dwSize = ctypes.sizeof(te)
    try:
        ok = k32.Thread32First(snap, ctypes.byref(te))
        while ok:
            if int(te.th32OwnerProcessID) == int(pid):
                out.append(int(te.th32ThreadID))
            ok = k32.Thread32Next(snap, ctypes.byref(te))
    finally:
        k32.CloseHandle(snap)
    return out

def pin_pid(pid, slot, skip_ht_pair=True):
    try:
        n_groups = int(k32.GetActiveProcessorGroupCount())
        cpus_per = int(k32.GetActiveProcessorCount(0)) if n_groups else 64
        slots_per_group = max(1, cpus_per // (2 if skip_ht_pair else 1))
        group = int(slot // slots_per_group) % max(1, n_groups)
        cpu = int(slot % slots_per_group) * (2 if skip_ht_pair else 1)
        aff = GROUP_AFFINITY()
        aff.Mask = 1 << cpu
        aff.Group = group
        ok_any = False
        for tid in threads_for_pid(pid):
            h = k32.OpenThread(THREAD_SET_INFORMATION, False, tid)
            if not h:
                continue
            try:
                ok_any = bool(k32.SetThreadGroupAffinity(h, ctypes.byref(aff), None)) or ok_any
            finally:
                k32.CloseHandle(h)
        return ok_any, group, cpu
    except Exception:
        return False, -1, -1

def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--payload-file":
        with open(sys.argv[2], "rb") as f:
            payload_b64 = f.read()
    else:
        payload_b64 = sys.argv[1].encode("ascii")
    payload = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (payload.get("env") or {}).items()})
    cwd = payload["cwd"]
    log_path = payload["log_path"]
    pid_path = payload.get("wrapper_pid_path")
    if pid_path:
        try:
            os.makedirs(os.path.dirname(pid_path), exist_ok=True)
            with open(pid_path, "w", encoding="ascii") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab", buffering=0) as log:
        log.write(("scheduleurm windows wrapper start task=%s cwd=%s\n" % (payload.get("task_id"), cwd)).encode())
        cpu_plan = payload.get("cpu_plan") or {}
        if cpu_plan:
            log.write(("scheduleurm cpu-plan %s\n" % json.dumps(cpu_plan, sort_keys=True)).encode())
        if payload.get("argv"):
            proc = subprocess.Popen(payload["argv"], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen(payload["cmdline"], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, shell=True, creationflags=CREATE_NEW_PROCESS_GROUP)
        log.write(("scheduleurm child pid=%s auto_pin=%s\n" % (proc.pid, payload.get("auto_pin"))).encode())
        assigned = {}
        next_slot = 0
        pin_base = int(payload.get("pin_base") or 0)
        pin_cores = max(1, int(payload.get("pin_cores") or 1))
        started_at = time.time()
        last_resource_log = 0.0
        loop_count = 0
        resource_interval = float(payload.get("resource_log_interval_s") or 60.0)
        while proc.poll() is None:
            loop_count += 1
            active = sorted(descendants(proc.pid))
            pin_ok_count = 0
            if payload.get("auto_pin"):
                assigned = {pid: slot for pid, slot in assigned.items() if pid in active}
                used_slots = set(assigned.values())
                for pid in active:
                    if pid not in assigned:
                        while next_slot in used_slots:
                            next_slot += 1
                        assigned[pid] = next_slot % pin_cores
                        used_slots.add(next_slot)
                        next_slot += 1
                    pin_slot = pin_base + (assigned[pid] % pin_cores)
                    ok, group, cpu = pin_pid(pid, pin_slot, bool(payload.get("skip_ht_pair", True)))
                    if ok:
                        pin_ok_count += 1
                    if ok and not os.environ.get("SCHEDULEURM_PIN_QUIET"):
                        log.write(("[pin] pid=%s slot=%s group=%s cpu=%s\n" % (pid, pin_slot, group, cpu)).encode())
            now = time.time()
            if now - last_resource_log >= resource_interval:
                progress = {
                    "task_id": payload.get("task_id"),
                    "elapsed_s": int(now - started_at),
                    "wrapper_loop": loop_count,
                    "root_pid": proc.pid,
                    "child_process_count": len(active),
                    "pinned_process_count": len(assigned),
                    "pin_ok_this_loop": pin_ok_count,
                    "next_pin_slot": next_slot,
                    "pin_base": pin_base,
                    "pin_cores": pin_cores,
                    "auto_pin": bool(payload.get("auto_pin")),
                    "cpu_plan": cpu_plan,
                }
                log.write(("scheduleurm resource-progress %s\n" % json.dumps(progress, sort_keys=True)).encode())
                last_resource_log = now
            time.sleep(float(payload.get("pin_interval_s", 1.0)))
        rc = proc.wait()
        log.write(("scheduleurm child exit rc=%s duration_s=%s assigned_processes=%s\n" % (
            rc, int(time.time() - started_at), len(assigned)
        )).encode())
    sys.exit(rc)

if __name__ == "__main__":
    main()
'''


WINDOWS_DETACHER = r'''
import os, subprocess, sys

py, launcher, payload, outp, errp, pidp = sys.argv[1:7]
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
flags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB
os.makedirs(os.path.dirname(outp), exist_ok=True)
os.makedirs(os.path.dirname(pidp), exist_ok=True)
out = open(outp, "ab", buffering=0)
err = open(errp, "ab", buffering=0)
p = subprocess.Popen(
    [py, launcher, "--payload-file", payload],
    stdin=subprocess.DEVNULL,
    stdout=out,
    stderr=err,
    close_fds=True,
    creationflags=flags,
)
with open(pidp, "w", encoding="ascii") as f:
    f.write(str(p.pid))
print("PID=" + str(p.pid))
'''
