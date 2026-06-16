"""Parallel CPU progress benchmark for large-node marginal-efficiency probes."""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor


def _cpu_chunk(items: int, salt: int) -> float:
    acc = 0.0
    n = max(1, int(items))
    for i in range(n):
        x = ((i + salt) % 10007) / 10007.0
        acc += math.sin(x) * math.cos(x + 0.37) + math.sqrt(x + 0.001)
    return acc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--work-items", type=int, default=120000)
    parser.add_argument("--label", default="scheduleurm-cpu-parallel")
    parser.add_argument("--log-interval", type=int, default=2)
    args = parser.parse_args()

    workers = max(1, int(args.workers))
    steps = max(1, int(args.steps))
    print(
        "CPU_PARALLEL_START "
        f"label={args.label} workers={workers} steps={steps} "
        f"work_items={int(args.work_items)} host_cpu={os.cpu_count()}",
        flush=True,
    )
    checksum = 0.0
    start = time.perf_counter()
    window_start = start
    window_steps = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for step in range(1, steps + 1):
            salts = [step * 1000003 + idx for idx in range(workers)]
            for value in pool.map(_cpu_chunk, [int(args.work_items)] * workers, salts):
                checksum += value
            window_steps += 1
            if step == steps or step % max(1, int(args.log_interval)) == 0:
                now = time.perf_counter()
                window_elapsed = max(1e-9, now - window_start)
                total_elapsed = max(1e-9, now - start)
                rate = window_steps / window_elapsed
                total_rate = step / total_elapsed
                eta = max(0, steps - step) / max(1e-9, total_rate)
                print(
                    f"CPU_PARALLEL_PROGRESS Step {step}/{steps} "
                    f"rate={rate:.9g} step/s total_rate={total_rate:.9g} step/s ETA {eta:.1f}s",
                    flush=True,
                )
                window_start = now
                window_steps = 0
    elapsed = time.perf_counter() - start
    print(
        f"CPU_PARALLEL_DONE label={args.label} steps={steps} elapsed={elapsed:.6f}s "
        f"rate={steps / max(elapsed, 1e-9):.9g} step/s checksum={checksum:.6g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
