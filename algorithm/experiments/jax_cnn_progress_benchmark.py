"""JAX CNN-style synthetic GPU benchmark.

This provides a pure GPU convolutional workload when PyTorch/torchvision is not
yet available on a measurement node.  It uses synthetic images and a JIT-compiled
conv stack, so the measured rate is compute/activation service rather than
input-pipeline service.
"""
from __future__ import annotations

import argparse
import time


def _progress(iterable, *, total: int, desc: str, unit: str):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit=unit, mininterval=0.1, dynamic_ncols=False)
    except Exception:
        return iterable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--mode", default="train", choices=["forward", "train"])
    parser.add_argument("--label", default="jax_cnn")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(123)
    x = jax.random.normal(
        key,
        (args.batch_size, args.image_size, args.image_size, 3),
        dtype=jnp.float32,
    )
    params = _init_params(jax.random.PRNGKey(456), depth=args.depth, channels=args.channels)

    def forward(local_params, image):
        h = image
        for kernel, bias in local_params:
            h = jax.lax.conv_general_dilated(
                h,
                kernel,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            h = jax.nn.relu(h + bias)
        return jnp.mean(h * h)

    if args.mode == "train":
        step_fn = jax.jit(jax.value_and_grad(forward))
    else:
        step_fn = jax.jit(forward)

    devices = jax.devices()
    print(
        "BENCH_START "
        f"backend=jax_cnn device={devices[0]} mode={args.mode} batch={args.batch_size} "
        f"image={args.image_size} channels={args.channels} depth={args.depth} label={args.label}",
        flush=True,
    )
    for _ in range(max(0, args.warmup)):
        out = step_fn(params, x)
        _block(out)

    start = time.perf_counter()
    window_start = start
    window_steps = 0
    total_steps = max(1, args.steps)
    for step in _progress(range(1, total_steps + 1), total=total_steps, desc=args.label, unit="step"):
        out = step_fn(params, x)
        _block(out)
        window_steps += 1
        if step == total_steps or step % max(1, args.log_interval) == 0:
            now = time.perf_counter()
            elapsed = max(1e-9, now - window_start)
            rate = window_steps / elapsed
            total_elapsed = max(1e-9, now - start)
            total_rate = step / total_elapsed
            remaining = max(0, total_steps - step)
            eta = remaining / max(1e-9, total_rate)
            print(
                f"BENCH_PROGRESS Step {step}/{args.steps} "
                f"rate={rate:.9g} step/s total_rate={total_rate:.9g} step/s ETA {eta:.1f}s",
                flush=True,
            )
            window_start = now
            window_steps = 0
    print("BENCH_DONE", flush=True)
    return 0


def _init_params(key, *, depth: int, channels: int):
    import jax
    import jax.numpy as jnp

    params = []
    in_channels = 3
    for idx in range(max(1, int(depth))):
        key, sub = jax.random.split(key)
        out_channels = max(8, int(channels))
        kernel = jax.random.normal(
            sub,
            (3, 3, in_channels, out_channels),
            dtype=jnp.float32,
        ) * jnp.float32((2.0 / max(1, 3 * 3 * in_channels)) ** 0.5)
        bias = jnp.zeros((1, 1, 1, out_channels), dtype=jnp.float32)
        params.append((kernel, bias))
        in_channels = out_channels
    return tuple(params)


def _block(value) -> None:
    if isinstance(value, tuple):
        for item in value:
            _block(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block(item)
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


if __name__ == "__main__":
    raise SystemExit(main())
