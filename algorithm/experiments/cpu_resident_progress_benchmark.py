"""Controlled CPU resident load with explicit readiness and tqdm progress."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time


def _progress(iterable, *, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit="s",
            mininterval=0.5,
            dynamic_ncols=False,
            ascii=True,
            file=sys.stdout,
        )
    except Exception:
        return iterable


def _resident_worker(stop_event, ready_queue, worker_index: int) -> None:
    ready_queue.put(int(worker_index))
    salt = int(worker_index) * 1_000_003
    accumulator = 0.0
    while not stop_event.is_set():
        for index in range(20_000):
            value = ((index + salt) % 10_007) / 10_007.0
            accumulator += math.sin(value) * math.cos(value + 0.37)
        salt += 97
    if not math.isfinite(accumulator):
        raise RuntimeError("nonfinite resident checksum")


def _atomic_ready(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--duration-s", type=int, default=7200)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--label", default="scheduleurm-cpu-resident")
    args = parser.parse_args()

    workers = max(1, int(args.workers))
    duration_s = max(1, int(args.duration_s))
    args.ready_file.unlink(missing_ok=True)
    args.stop_file.unlink(missing_ok=True)
    context = mp.get_context("fork")
    stop_event = context.Event()
    ready_queue = context.Queue()
    children = [
        context.Process(
            target=_resident_worker,
            args=(stop_event, ready_queue, index),
            daemon=False,
        )
        for index in range(workers)
    ]
    print(
        f"CPU_RESIDENT_START label={args.label} workers={workers} "
        f"duration_s={duration_s}",
        flush=True,
    )
    for child in children:
        child.start()

    ready_workers = set()
    deadline = time.monotonic() + 120.0
    while len(ready_workers) < workers:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            stop_event.set()
            for child in children:
                child.join(timeout=2.0)
            raise TimeoutError(
                f"only {len(ready_workers)}/{workers} resident workers became ready"
            )
        try:
            ready_workers.add(
                int(ready_queue.get(timeout=min(2.0, remaining)))
            )
        except queue.Empty:
            continue

    ready_payload = {
        "schema_version": 1,
        "label": str(args.label),
        "workers": workers,
        "pid": os.getpid(),
        "ready_monotonic_s": time.monotonic(),
    }
    _atomic_ready(args.ready_file, ready_payload)
    print(
        "CPU_RESIDENT_READY "
        + json.dumps(ready_payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )

    elapsed = 0
    try:
        for elapsed in _progress(
            range(1, duration_s + 1),
            total=duration_s,
            desc=str(args.label),
        ):
            if args.stop_file.exists():
                break
            time.sleep(1.0)
            if elapsed == 1 or elapsed % 10 == 0:
                print(
                    f"CPU_RESIDENT_PROGRESS Second {elapsed}/{duration_s} "
                    f"workers={workers}",
                    flush=True,
                )
    finally:
        stop_event.set()
        for child in children:
            child.join(timeout=15.0)
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5.0)
    failed = [
        child.pid
        for child in children
        if child.exitcode not in (0, None)
    ]
    print(
        f"CPU_RESIDENT_DONE label={args.label} workers={workers} "
        f"elapsed_s={elapsed} failed_children={len(failed)}",
        flush=True,
    )
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
