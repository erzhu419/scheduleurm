"""Python 2/3 compatible CPU progress benchmark."""
from __future__ import print_function

import argparse
import math
import sys
import time


def _emit(text):
    print(text)
    sys.stdout.flush()


def _light_work(items):
    acc = 0.0
    for i in range(max(1, int(items))):
        acc += (i % 17) * 0.000001
    return acc


def _cpu_work(items):
    acc = 0.0
    for i in range(max(1, int(items))):
        x = (i % 1000) / 1000.0
        acc += math.sin(x) * math.cos(x + 0.1)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--mode", choices=("light", "cpu"), default="light")
    parser.add_argument("--work-items", type=int, default=10000)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--label", default="scheduleurm-cpu-compat")
    args = parser.parse_args()
    steps = max(1, int(args.steps))
    work = _cpu_work if args.mode == "cpu" else _light_work
    _emit(
        "CPU_BENCH_START label=%s mode=%s steps=%d work_items=%d sleep_s=%.6f"
        % (args.label, args.mode, steps, int(args.work_items), float(args.sleep_s))
    )
    start = time.time()
    checksum = 0.0
    for i in range(1, steps + 1):
        t0 = time.time()
        checksum += work(args.work_items)
        if args.sleep_s > 0:
            time.sleep(float(args.sleep_s))
        dt = time.time() - t0
        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-9)
        remaining = max(0, steps - i)
        eta = remaining / max(rate, 1e-9)
        _emit(
            "Step %d/%d dt=%.6fs elapsed=%.3fs rate=%.6f step/s ETA %.1fs"
            % (i, steps, dt, elapsed, rate, eta)
        )
    elapsed = time.time() - start
    _emit(
        "CPU_BENCH_DONE label=%s mode=%s steps=%d elapsed=%.6fs rate=%.6f step/s checksum=%.6f"
        % (args.label, args.mode, steps, elapsed, steps / max(elapsed, 1e-9), checksum)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
