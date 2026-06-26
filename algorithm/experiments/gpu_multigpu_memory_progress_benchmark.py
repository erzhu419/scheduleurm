"""Multi-GPU memory/compute progress benchmark with explicit ETA lines.

This probe is intentionally synthetic.  It exercises the scheduling surface
needed by Scheduleurm: a large multi-GPU memory resident job can be kept alive
while small marginal jobs are probed on the remaining or colocated capacity.
"""
from __future__ import annotations

import argparse
import os
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


def _parse_devices(raw: str, count: int) -> list[int]:
    text = str(raw or "").strip()
    if text:
        out = [int(x) for x in text.split(",") if x.strip()]
    else:
        out = list(range(max(0, int(count))))
    return sorted(dict.fromkeys(out))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--devices", default="")
    parser.add_argument("--reserve-gb-per-gpu", type=float, default=0.0)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--label", default="scheduleurm-multigpu-memory")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--commit-reserve", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for gpu_multigpu_memory_progress_benchmark")
    devices = _parse_devices(args.devices, torch.cuda.device_count())
    if not devices:
        raise SystemExit("at least one CUDA device is required")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    reserves = []
    mats = []
    reserve_gb = max(0.0, float(args.reserve_gb_per_gpu))
    reserve_elems = int(reserve_gb * (1024**3) / max(1, bytes_per_elem))
    for dev_idx in devices:
        dev = torch.device(f"cuda:{dev_idx}")
        chunks = []
        if reserve_elems > 0:
            chunk_elems = max(1, int(256 * 1024 * 1024 / max(1, bytes_per_elem)))
            remaining = reserve_elems
            while remaining > 0:
                n = min(chunk_elems, remaining)
                tensor = torch.empty((n,), dtype=dtype, device=dev)
                if args.commit_reserve:
                    tensor.fill_(0)
                chunks.append(tensor)
                remaining -= n
        reserves.append(chunks)
        size = max(16, int(args.matrix_size))
        a = torch.randn((size, size), dtype=dtype, device=dev)
        b = torch.randn((size, size), dtype=dtype, device=dev)
        mats.append((dev, a, b))

    for dev_idx in devices:
        torch.cuda.synchronize(dev_idx)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allocated = {
        str(dev_idx): int(torch.cuda.memory_allocated(dev_idx) // (1024 * 1024))
        for dev_idx in devices
    }
    reserved = {
        str(dev_idx): int(torch.cuda.memory_reserved(dev_idx) // (1024 * 1024))
        for dev_idx in devices
    }
    print(
        "BENCH_START "
        f"backend=torch_multigpu_memory label={args.label} devices={devices} "
        f"cuda_visible={visible!r} reserve_gb_per_gpu={reserve_gb:.3f} "
        f"matrix_size={int(args.matrix_size)} dtype={args.dtype} "
        f"allocated_mb={allocated} reserved_mb={reserved}",
        flush=True,
    )

    for _ in range(max(0, int(args.warmup))):
        _step(mats)
    for dev_idx in devices:
        torch.cuda.synchronize(dev_idx)

    steps = max(1, int(args.steps))
    start = time.perf_counter()
    window_start = start
    window_steps = 0
    checksum = 0.0
    for step in _progress(range(1, steps + 1), total=steps, desc=str(args.label), unit="step"):
        checksum += _step(mats)
        window_steps += 1
        if step == steps or step % max(1, int(args.log_interval)) == 0:
            for dev_idx in devices:
                torch.cuda.synchronize(dev_idx)
            now = time.perf_counter()
            window_elapsed = max(1e-9, now - window_start)
            total_elapsed = max(1e-9, now - start)
            rate = window_steps / window_elapsed
            total_rate = step / total_elapsed
            eta = max(0, steps - step) / max(1e-9, total_rate)
            print(
                f"BENCH_PROGRESS Step {step}/{steps} "
                f"rate={rate:.9g} step/s total_rate={total_rate:.9g} step/s ETA {eta:.1f}s",
                flush=True,
            )
            window_start = now
            window_steps = 0
    print(f"BENCH_DONE checksum={checksum:.6g}", flush=True)
    return 0


def _step(mats) -> float:
    checksum = 0.0
    for _dev, a, b in mats:
        c = a @ b
        checksum += float(c[0, 0].detach().float().cpu())
    return checksum


if __name__ == "__main__":
    raise SystemExit(main())
