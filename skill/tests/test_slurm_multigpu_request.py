from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_scheduler():
    path = Path(__file__).resolve().parents[1] / "scheduler.py"
    spec = importlib.util.spec_from_file_location("scheduleurm_scheduler_test_slurm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_slurm_sbatch_preserves_single_gpu_default():
    sch = _load_scheduler()
    task = {
        "id": "tSingleGpu",
        "cwd": "/tmp",
        "cmd": "python train.py",
        "cpu_cores": 2,
        "ram_mb": 4096,
        "est_vram_mb": 4096,
    }
    script = sch.SlurmBackend()._build_sbatch_script(task, "python train.py", "/tmp/out.log")
    assert "#SBATCH --gres=gpu:1" in script


def test_slurm_sbatch_requests_declared_multigpu_count():
    sch = _load_scheduler()
    task = {
        "id": "tMultiGpu",
        "cwd": "/tmp",
        "cmd": "torchrun --nproc_per_node 4 train_llm.py",
        "cpu_cores": 16,
        "ram_mb": 32768,
        "est_vram_mb": 30720,
        "gpu_count": 4,
    }
    script = sch.SlurmBackend()._build_sbatch_script(task, "torchrun train_llm.py", "/tmp/out.log")
    assert "#SBATCH --gres=gpu:4" in script
