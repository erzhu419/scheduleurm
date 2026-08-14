"""CPU/GPU intent policy for submitted training commands."""

from __future__ import annotations

import re


def cmd_looks_like_training(cmd: str) -> bool:
    lower = (cmd or "").lower()
    if re.search(r"\bpython(?:\d(?:\.\d+)?)?\b[^\n;]*\s-m\s+[\w.]*train\b", lower):
        return True
    if re.search(r"\bpython(?:\d(?:\.\d+)?)?\b[^\n;]*\b[\w./-]*train[\w./-]*\.py\b", lower):
        return True
    return any(p in lower for p in (
        "train_", "/train.py", " train.py", "trainer.py",
        "/main_train", "run_train", "do_train",
        "h2o+_bus_main.py",
    ))


def cmd_explicitly_cpu(cmd: str) -> bool:
    lower = (cmd or "").lower()
    norm = lower.replace("=", " ")
    norm = re.sub(r"(?:^|\s)--replay[-_]?device\s+cpu(?:\s|$)", " ", norm)
    cuda_empty = bool(re.search(
        r"(?:^|\s)(?:export\s+)?cuda_visible_devices\s*=\s*(?:\"\"|''|-1|;|$|\s)",
        lower,
    ))
    return any(p in norm for p in (
        "--device cpu", "--cpu-only", "--no-cuda", "--no-gpu", "--use-cpu",
        "device cpu",
    )) or cuda_empty


def task_looks_like_training(cmd: str, description: str = "") -> bool:
    if cmd_looks_like_training(cmd):
        return True
    lower_desc = (description or "").lower()
    # Training words must be tokens. A raw ``"train" in description`` check
    # also matches ordinary algorithm names such as ``constrained_ei`` and can
    # permanently strand intentional CPU workloads behind the GPU-intent gate.
    training_word = re.search(
        r"(?:^|[^a-z0-9])(?:pre)?train(?:ing|ed|er)?(?:$|[^a-z0-9])",
        lower_desc,
    )
    return bool(training_word) or any(p in lower_desc for p in (
        "baseline:",
        " iql", "iql ", "awac", "td3", "rlpd", "wsrl", "sac", "bc on",
    ))


def cpu_training_policy_reason(
    cmd: str,
    description: str = "",
    allow_cpu_training: bool = False,
    est_vram_mb: int = 0,
):
    if not task_looks_like_training(cmd, description):
        return None
    try:
        vram = int(est_vram_mb or 0)
    except Exception:
        vram = 0
    explicit_cpu = cmd_explicitly_cpu(cmd)
    if allow_cpu_training:
        if explicit_cpu and vram > 0:
            return ("training-looking task has an explicit CPU device flag but scheduler vram>0 "
                    "would reserve a GPU while the program runs on CPU. Use --vram 0 with "
                    "--allow-cpu-training, or remove the CPU flag for GPU training.")
        return None
    if explicit_cpu and vram > 0:
        return ("training-looking task explicitly requests CPU in the inner cmd, but scheduler "
                "would reserve a GPU because vram>0. Remove/replace the CPU flag for GPU "
                "training, or use --vram 0 with --allow-cpu-training if CPU training is intentional.")
    if explicit_cpu:
        return ("training-looking task has vram=0 and an explicit CPU device flag; "
                "scheduler would run it CPU-only. Resubmit with --vram <MB> and a GPU device flag, "
                "or pass --allow-cpu-training if CPU training is intentional.")
    if vram > 0:
        return None
    return ("training-looking task has vram=0, so scheduler would run it CPU-only. "
            "Resubmit with --vram <MB>, or pass --allow-cpu-training if CPU training is intentional.")
