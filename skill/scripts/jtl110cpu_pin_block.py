"""Cross-group physical-core pinning for Windows machines with >64 logical CPUs.

GENERAL: Windows uses Processor Groups (≤64 logical CPUs each) for any machine with >64
logical CPUs. A process is locked to ONE group at startup, restricting it to ≤64 logical
CPUs visible at once. mp.Pool spawn keeps all workers in master's group → 125 workers
fight over 64 logical → ~50% per-worker efficiency. SetThreadGroupAffinity is the only
documented way to move a thread to a different group from inside Python.

This block queries the runtime layout, distributes workers evenly across groups, and pins
each worker to a unique physical core (every-other logical CPU to skip HT pair sibling).

Paste this INSIDE your mp.Pool worker function, AFTER `import libsumo` AND `import torch`
AND `torch.set_num_threads(1)`. Earlier (e.g. in mp.Pool initializer) doesn't stick —
torch/libsumo init resets thread affinity to the full group mask.

Verified 2026-05-13 on jtl110cpu (256 logical / 128 physical / 4 groups): 125 workers ×
564 ckpts × 10 ep SUMO eval ran at ~120-125 sumCPU-s per wallclock-s (vs ~60 without
pinning) — 2x throughput.

Falsified attempts (do NOT use):
  - hash(p.pid)%N  → Windows PIDs are multiples of 4, only 8 unique slots → 16x oversub
  - mp.Manager().Value() counter → increments not atomic, all workers read 0
  - PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY in CreateProcessW → constant = 0x30003 not 0x30007
  - mp.Pool initializer pin → reset by subsequent torch/libsumo imports
  - ctypes without argtypes → 64-bit HANDLE truncated to int32 → SetThreadGroupAffinity
    returns 0 silently
"""

def _windows_query_groups():
    """Return (n_groups, list_of_cpus_per_group) at runtime via Win32 kernel32.
    Works on any Windows version supporting Processor Groups (Win7+)."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    k32.GetActiveProcessorGroupCount.argtypes = []
    k32.GetActiveProcessorGroupCount.restype  = wintypes.WORD
    k32.GetActiveProcessorCount.argtypes      = [wintypes.WORD]
    k32.GetActiveProcessorCount.restype       = wintypes.DWORD
    n_groups = k32.GetActiveProcessorGroupCount()
    return n_groups, [k32.GetActiveProcessorCount(g) for g in range(n_groups)]


def windows_pin_to_unique_physical_core(skip_ht_pair=True):
    """Call from inside worker function AFTER torch.set_num_threads(1) and import libsumo.

    Args:
        skip_ht_pair: If True, pin to every-OTHER logical CPU in group (= unique physical
            core, no HT-sibling contention). Halves available slots per group but doubles
            per-worker throughput. Set False if you intentionally want to use both HT
            threads of each physical core (e.g. compute is memory-bound, HT helps).

    Returns:
        (ok: bool, target_group: int, cpu_in_group: int) — ok=True iff Win32 call succeeded.
    """
    try:
        import ctypes, multiprocessing as _mp
        from ctypes import wintypes, c_size_t, byref, POINTER

        class _GA(ctypes.Structure):
            _fields_ = [
                ('Mask',     c_size_t),
                ('Group',    wintypes.WORD),
                ('Reserved', wintypes.WORD * 3),
            ]

        _k32 = ctypes.windll.kernel32
        # CRITICAL: declare argtypes — without these, 64-bit HANDLE truncates to int32,
        # SetThreadGroupAffinity returns 0 silently and pin doesn't take effect.
        _k32.GetCurrentThread.argtypes      = []
        _k32.GetCurrentThread.restype       = wintypes.HANDLE
        _k32.SetThreadGroupAffinity.argtypes = [wintypes.HANDLE, POINTER(_GA), POINTER(_GA)]
        _k32.SetThreadGroupAffinity.restype  = wintypes.BOOL

        # Query layout at runtime — portable across all Windows multi-group hosts
        n_groups, cpus_per_group = _windows_query_groups()
        # Assume groups are uniform (true for all current SKUs); use group 0's size.
        cpus_per = cpus_per_group[0] if cpus_per_group else 64
        slots_per_group = cpus_per // 2 if skip_ht_pair else cpus_per

        name = _mp.current_process().name
        try:
            n = int(name.split('-')[-1])
        except Exception:
            n = 1
        n0 = max(0, n - 1)

        # Distribute round-robin across groups: each group gets ~equal share.
        # Worker N maps to (group = N // slots_per_group, slot = N % slots_per_group)
        target_group = (n0 // slots_per_group) % n_groups
        slot         = n0 % slots_per_group
        cpu_in_group = slot * (2 if skip_ht_pair else 1)

        aff = _GA()
        aff.Mask  = 1 << cpu_in_group
        aff.Group = target_group
        ok = _k32.SetThreadGroupAffinity(_k32.GetCurrentThread(), byref(aff), None)
        return ok, target_group, cpu_in_group
    except Exception:
        return False, -1, -1


def windows_pin_to_unique_physical_core_with_log(log_dir):
    """As above but write a verification log line. Use for debugging only."""
    ok, tg, ci = False, -1, -1
    try:
        import ctypes, multiprocessing as _mp, os as _os
        from ctypes import wintypes, c_size_t, byref, POINTER

        class _GA(ctypes.Structure):
            _fields_ = [('Mask', c_size_t), ('Group', wintypes.WORD), ('Reserved', wintypes.WORD*3)]

        _k32 = ctypes.windll.kernel32
        _k32.GetCurrentThread.argtypes      = []
        _k32.GetCurrentThread.restype       = wintypes.HANDLE
        _k32.SetThreadGroupAffinity.argtypes = [wintypes.HANDLE, POINTER(_GA), POINTER(_GA)]
        _k32.SetThreadGroupAffinity.restype  = wintypes.BOOL
        _k32.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE, POINTER(_GA)]
        _k32.GetThreadGroupAffinity.restype  = wintypes.BOOL

        name = _mp.current_process().name
        try:    n = int(name.split('-')[-1])
        except: n = 1
        n0 = max(0, n - 1)
        tg = n0 // 32
        ci = (n0 % 32) * 2

        aff = _GA(); aff.Mask = 1 << ci; aff.Group = tg
        ok = _k32.SetThreadGroupAffinity(_k32.GetCurrentThread(), byref(aff), None)

        rb = _GA()
        _k32.GetThreadGroupAffinity(_k32.GetCurrentThread(), byref(rb))
        bits = bin(rb.Mask).count('1')

        try: _os.makedirs(log_dir, exist_ok=True)
        except Exception: pass
        with open(_os.path.join(log_dir, f'w{n:04d}.log'), 'w') as f:
            f.write(f'name={name} setOK={ok} target=(g{tg},cpu{ci}) actual=(g{rb.Group},mask_bits={bits})\n')
    except Exception:
        pass
    return ok, tg, ci
