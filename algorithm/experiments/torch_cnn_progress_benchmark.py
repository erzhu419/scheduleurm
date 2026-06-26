"""Torch CNN synthetic progress benchmark for pure-GPU service curves.

The benchmark uses standard torchvision CNN architectures with synthetic
images, so it measures GPU compute/activation pressure without a data-loader
or storage bottleneck.  It prints Scheduleurm-compatible rate lines.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet18", "resnet50", "convnext_tiny"])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--mode", default="train", choices=["forward", "train"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--label", default="torch_cnn")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    import torch
    try:
        import torchvision.models as models
    except Exception:
        models = None

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for torch_cnn_progress_benchmark")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = _build_model(args.model, models, torch).to(device)
    model.train(args.mode == "train")
    image = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    target = torch.randint(0, args.classes, (args.batch_size,), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) if args.mode == "train" else None
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and args.mode == "train"))

    print(
        "BENCH_START "
        f"backend=torch device={torch.cuda.get_device_name(0)!r} "
        f"model={args.model} mode={args.mode} batch={args.batch_size} "
        f"image={args.image_size} amp={str(bool(args.amp)).lower()} label={args.label}",
        flush=True,
    )
    for _ in range(max(0, args.warmup)):
        _step(model, image, target, criterion, optimizer, scaler, args.mode, args.amp)
    torch.cuda.synchronize()

    start = time.perf_counter()
    window_start = start
    window_steps = 0
    total_steps = max(1, args.steps)
    for step in _progress(range(1, total_steps + 1), total=total_steps, desc=args.label, unit="step"):
        _step(model, image, target, criterion, optimizer, scaler, args.mode, args.amp)
        window_steps += 1
        if step == total_steps or step % max(1, args.log_interval) == 0:
            torch.cuda.synchronize()
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


def _build_model(name: str, models, torch):
    if models is None:
        return _FallbackCNN(torch)
    if name == "resnet18":
        return models.resnet18(weights=None)
    if name == "resnet50":
        return models.resnet50(weights=None)
    if name == "convnext_tiny":
        return models.convnext_tiny(weights=None)
    raise ValueError(name)


class _FallbackCNN:
    """Torch-only CNN stack used when torchvision is not installed on a node."""

    def __new__(cls, torch):
        layers = []
        in_channels = 3
        channels = (32, 64, 96, 128, 128)
        for out_channels in channels:
            layers.extend([
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels
        layers.extend([
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_channels, 1000),
        ])
        return torch.nn.Sequential(*layers)


def _step(model, image, target, criterion, optimizer, scaler, mode: str, amp: bool) -> None:
    import torch

    if mode == "forward":
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=bool(amp)):
            _ = model(image)
        return
    assert optimizer is not None
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=bool(amp)):
        output = model(image)
        loss = criterion(output, target)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    raise SystemExit(main())
