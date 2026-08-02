"""Task-native CPU benchmark with durable outer progress and terminal writes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time


def _progress(iterable, *, total: int, label: str):
    try:
        from tqdm import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=label,
            unit="step",
            mininterval=0.5,
            dynamic_ncols=False,
            ascii=True,
            file=sys.stdout,
        )
    except Exception:
        return iterable


def _kernel(work_items: int, seed: int) -> float:
    """One indivisible CPU service unit; the outer loop is ETA progress."""

    acc = 0.0
    offset = int(seed) % 997
    for index in range(max(1, int(work_items))):
        value = ((index + offset) % 1000) / 1000.0
        acc += math.sin(value) * math.cos(value + 0.1)
    return acc


def _write_checkpoint(directory: Path, step: int, size: int) -> float:
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{int(step):06d}.bin"
    remaining = max(0, int(size))
    block = b"\0" * min(1024 * 1024, max(1, remaining))
    with path.open("wb") as handle:
        while remaining > 0:
            chunk = block if remaining >= len(block) else block[:remaining]
            handle.write(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    return time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--mode", choices=("light", "cpu"), required=True)
    parser.add_argument("--work-items", type=int, default=1000)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-bytes", type=int, default=0)
    parser.add_argument("--label", default="scheduleurm-file-progress-cpu")
    args = parser.parse_args()

    steps = max(1, int(args.steps))
    output_dir = args.output_dir.resolve()
    progress_path = args.progress_file.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    print("ScheduleurmPhase name=initialization event=start", flush=True)
    initialization_started = time.monotonic()
    checksum = _kernel(max(1, int(args.work_items) // 10), int(args.seed))
    initialization_s = time.monotonic() - initialization_started
    print("ScheduleurmPhase name=initialization event=end", flush=True)

    started = time.monotonic()
    checkpoint_total_s = 0.0
    print("ScheduleurmPhase name=outer_loop event=start", flush=True)
    with progress_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "elapsed_s", "checkpoint_s", "checksum"),
        )
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        for step in _progress(range(1, steps + 1), total=steps, label=args.label):
            if args.mode == "cpu":
                checksum += _kernel(int(args.work_items), int(args.seed) + step)
            else:
                checksum += _kernel(int(args.work_items), int(args.seed) + step)
                if float(args.sleep_s) > 0.0:
                    time.sleep(float(args.sleep_s))
            checkpoint_s = 0.0
            if (
                int(args.checkpoint_interval) > 0
                and step % int(args.checkpoint_interval) == 0
            ):
                checkpoint_s = _write_checkpoint(
                    checkpoint_dir,
                    step,
                    int(args.checkpoint_bytes),
                )
                checkpoint_total_s += checkpoint_s
            writer.writerow(
                {
                    "step": step,
                    "elapsed_s": f"{time.monotonic() - started:.9f}",
                    "checkpoint_s": f"{checkpoint_s:.9f}",
                    "checksum": f"{checksum:.12f}",
                }
            )
            handle.flush()
            os.fsync(handle.fileno())
    print("ScheduleurmPhase name=outer_loop event=end", flush=True)

    print("ScheduleurmPhase name=finalization event=start", flush=True)
    finalization_started = time.monotonic()
    final_path = output_dir / "final.json"
    with final_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "steps": steps,
                "mode": args.mode,
                "seed": int(args.seed),
                "checksum": checksum,
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    finalization_s = time.monotonic() - finalization_started
    print("ScheduleurmPhase name=finalization event=end", flush=True)
    total_wall_s = time.monotonic() - started + initialization_s
    print(
        "ScheduleurmCpuCompletion "
        f"mode={args.mode} steps={steps} initialization_s={initialization_s:.9f} "
        f"checkpoint_s={checkpoint_total_s:.9f} "
        f"finalization_s={finalization_s:.9f} total_wall_s={total_wall_s:.9f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
