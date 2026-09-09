"""CPU/light progress benchmark for Scheduleurm service-curve validation."""
from __future__ import annotations

import argparse
import math
import sys
import time


def _progress(iterable, *, total: int, desc: str, unit: str):
    try:
        from tqdm import tqdm
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            mininterval=0.5,
            dynamic_ncols=False,
            ascii=True,
            file=sys.stdout,
        )
    except Exception:
        return iterable


def _phase(name: str, event: str) -> None:
    print(f"ScheduleurmPhase name={name} event={event}", flush=True)


def _light_work(items: int) -> float:
    acc = 0.0
    for i in range(max(1, int(items))):
        acc += (i % 17) * 0.000001
    return acc


def _cpu_work(items: int) -> float:
    acc = 0.0
    for i in range(max(1, int(items))):
        x = (i % 1000) / 1000.0
        acc += math.sin(x) * math.cos(x + 0.1)
    return acc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--mode", choices=("light", "cpu"), default="light")
    parser.add_argument("--work-items", type=int, default=10000)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--label", default="scheduleurm-cpu-bench")
    args = parser.parse_args()

    _phase("initialization", "start")
    steps = max(1, int(args.steps))
    work = _cpu_work if args.mode == "cpu" else _light_work
    _phase("initialization", "end")
    print(
        f"CPU_BENCH_START label={args.label} mode={args.mode} "
        f"steps={steps} work_items={int(args.work_items)} sleep_s={float(args.sleep_s):.6f}",
        flush=True,
    )
    start = time.time()
    checksum = 0.0
    _phase("outer_loop", "start")
    for i in _progress(range(1, steps + 1), total=steps, desc=str(args.label), unit="step"):
        t0 = time.time()
        checksum += work(args.work_items)
        if args.sleep_s > 0:
            time.sleep(float(args.sleep_s))
        dt = time.time() - t0
        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-9)
        remaining = max(0, steps - i)
        eta = remaining / max(rate, 1e-9)
        print(
            f"Step {i}/{steps} dt={dt:.6f}s elapsed={elapsed:.3f}s "
            f"rate={rate:.6f} step/s ETA {eta:.1f}s",
            flush=True,
        )
    _phase("outer_loop", "end")
    elapsed = time.time() - start
    print(
        f"CPU_BENCH_DONE label={args.label} mode={args.mode} steps={steps} "
        f"elapsed={elapsed:.6f}s rate={steps / max(elapsed, 1e-9):.6f} "
        f"checksum={checksum:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
