"""Python 2/3 compatible parallel CPU progress benchmark."""
from __future__ import print_function

import argparse
import math
import multiprocessing
import os
import sys
import time


def _emit(text):
    print(text)
    sys.stdout.flush()


def _cpu_chunk(args):
    items, salt = args
    acc = 0.0
    n = max(1, int(items))
    for i in range(n):
        x = ((i + salt) % 10007) / 10007.0
        acc += math.sin(x) * math.cos(x + 0.37) + math.sqrt(x + 0.001)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count()))
    parser.add_argument("--work-items", type=int, default=120000)
    parser.add_argument("--label", default="scheduleurm-cpu-parallel-compat")
    parser.add_argument("--log-interval", type=int, default=2)
    args = parser.parse_args()
    workers = max(1, int(args.workers))
    steps = max(1, int(args.steps))
    host_cpu = multiprocessing.cpu_count()
    _emit(
        "CPU_PARALLEL_START label=%s workers=%d steps=%d work_items=%d host_cpu=%s"
        % (args.label, workers, steps, int(args.work_items), host_cpu)
    )
    checksum = 0.0
    start = time.time()
    window_start = start
    window_steps = 0
    pool = multiprocessing.Pool(processes=workers)
    try:
        for step in range(1, steps + 1):
            salts = [step * 1000003 + idx for idx in range(workers)]
            for value in pool.map(_cpu_chunk, [(int(args.work_items), salt) for salt in salts]):
                checksum += value
            window_steps += 1
            if step == steps or step % max(1, int(args.log_interval)) == 0:
                now = time.time()
                window_elapsed = max(1e-9, now - window_start)
                total_elapsed = max(1e-9, now - start)
                rate = window_steps / window_elapsed
                total_rate = step / total_elapsed
                eta = max(0, steps - step) / max(1e-9, total_rate)
                _emit(
                    "CPU_PARALLEL_PROGRESS Step %d/%d rate=%.9g step/s total_rate=%.9g step/s ETA %.1fs"
                    % (step, steps, rate, total_rate, eta)
                )
                window_start = now
                window_steps = 0
    finally:
        pool.close()
        pool.join()
    elapsed = time.time() - start
    _emit(
        "CPU_PARALLEL_DONE label=%s steps=%d elapsed=%.6fs rate=%.9g step/s checksum=%.6g"
        % (args.label, steps, elapsed, steps / max(elapsed, 1e-9), checksum)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
