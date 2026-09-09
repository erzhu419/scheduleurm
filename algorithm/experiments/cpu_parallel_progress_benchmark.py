"""Parallel CPU progress benchmark for large-node marginal-efficiency probes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor


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


def _phase(name: str, event: str, *, index: int | None = None) -> None:
    suffix = f" index={int(index)}" if index is not None else ""
    print(f"ScheduleurmPhase name={name} event={event}{suffix}", flush=True)


def _save_checkpoint(
    *,
    checkpoint_dir: str,
    label: str,
    phase_name: str,
    step: int,
    checksum: float,
    payload_bytes: int,
) -> None:
    root = Path(checkpoint_dir or "/tmp/scheduleurm_cpu_checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{label}_{phase_name}_{int(step)}.ckpt"
    _phase(phase_name, "start", index=step)
    header = json.dumps(
        {"label": label, "phase": phase_name, "step": int(step), "checksum": checksum},
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    remaining = max(0, int(payload_bytes))
    block = bytes(min(1024 * 1024, max(1, remaining)))
    with path.open("wb") as handle:
        handle.write(header)
        while remaining > 0:
            chunk = block[: min(len(block), remaining)]
            handle.write(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    _phase(phase_name, "end", index=step)


def _cpu_chunk(items: int, salt: int) -> float:
    acc = 0.0
    n = max(1, int(items))
    for i in range(n):
        x = ((i + salt) % 10007) / 10007.0
        acc += math.sin(x) * math.cos(x + 0.37) + math.sqrt(x + 0.001)
    return acc


def _cpu_counters() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    if len(values) < 4:
        raise RuntimeError("/proc/stat cpu row is incomplete")
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total - idle, total


def _process_cpu_seconds(value: os.times_result) -> float:
    return float(value.user + value.system + value.children_user + value.children_system)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--work-items", type=int, default=120000)
    parser.add_argument("--label", default="scheduleurm-cpu-parallel")
    parser.add_argument("--log-interval", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-bytes", type=int, default=0)
    parser.add_argument("--final-save", action="store_true")
    args = parser.parse_args()

    _phase("initialization", "start")
    workers = max(1, int(args.workers))
    steps = max(1, int(args.steps))
    _phase("initialization", "end")
    print(
        "CPU_PARALLEL_START "
        f"label={args.label} workers={workers} steps={steps} "
        f"work_items={int(args.work_items)} host_cpu={os.cpu_count()}",
        flush=True,
    )
    checksum = 0.0
    start = time.perf_counter()
    load_start = os.getloadavg()
    busy_start, total_start = _cpu_counters()
    process_start = os.times()
    window_start = start
    window_steps = 0
    _phase("outer_loop", "start")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for step in _progress(range(1, steps + 1), total=steps, desc=str(args.label), unit="step"):
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
            if int(args.checkpoint_interval) > 0 and step % int(args.checkpoint_interval) == 0:
                _save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    label=args.label,
                    phase_name="checkpoint",
                    step=step,
                    checksum=checksum,
                    payload_bytes=args.checkpoint_bytes,
                )
    _phase("outer_loop", "end")
    process_end = os.times()
    busy_end, total_end = _cpu_counters()
    load_end = os.getloadavg()
    compute_elapsed = max(1e-9, time.perf_counter() - start)
    logical_cpus = max(1, int(os.cpu_count() or 1))
    delta_total = max(1, int(total_end - total_start))
    system_busy_fraction = max(0.0, min(1.0, float(busy_end - busy_start) / float(delta_total)))
    system_busy_core_equiv = system_busy_fraction * float(logical_cpus)
    own_cpu_s = max(0.0, _process_cpu_seconds(process_end) - _process_cpu_seconds(process_start))
    own_cpu_core_equiv = own_cpu_s / compute_elapsed
    resource_state = {
        "schema_version": 1,
        "logical_cpus": logical_cpus,
        "workers": workers,
        "measurement_wall_s": compute_elapsed,
        "system_busy_fraction": system_busy_fraction,
        "system_busy_core_equiv": system_busy_core_equiv,
        "own_cpu_s": own_cpu_s,
        "own_cpu_core_equiv": own_cpu_core_equiv,
        "external_cpu_core_equiv": max(0.0, system_busy_core_equiv - own_cpu_core_equiv),
        "load1_start": float(load_start[0]),
        "load1_end": float(load_end[0]),
        "load5_start": float(load_start[1]),
        "load5_end": float(load_end[1]),
    }
    print(
        "ScheduleurmResourceState "
        + json.dumps(resource_state, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    if args.final_save:
        _save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            label=args.label,
            phase_name="final_save",
            step=steps,
            checksum=checksum,
            payload_bytes=args.checkpoint_bytes,
        )
    elapsed = time.perf_counter() - start
    print(
        f"CPU_PARALLEL_DONE label={args.label} steps={steps} elapsed={elapsed:.6f}s "
        f"rate={steps / max(elapsed, 1e-9):.9g} step/s checksum={checksum:.6g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
