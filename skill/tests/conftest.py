from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "skill" / "scheduler.py"


@pytest.fixture
def check():
    failures: list[str] = []

    def _check(name: str, cond, diag: str = "") -> None:
        ok = bool(cond)
        if not ok:
            detail = f"{name}: {diag}" if diag else str(name)
            failures.append(detail)

    yield _check

    if failures:
        pytest.fail("\n".join(failures), pytrace=False)


@pytest.fixture
def sch(tmp_path, monkeypatch):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    module_name = f"scheduleurm_scheduler_pytest_{id(tmp_path)}"
    spec = importlib.util.spec_from_file_location(module_name, SCHEDULER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scheduler from {SCHEDULER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    state_dir = tmp_path / "scheduler_state"
    log_dir = state_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "STATE_DIR", state_dir, raising=False)
    monkeypatch.setattr(module, "QUEUE_FILE", state_dir / "queue.json", raising=False)
    monkeypatch.setattr(module, "VRAM_FILE", state_dir / "vram_history.json", raising=False)
    monkeypatch.setattr(module, "RUNTIME_FILE", state_dir / "runtime_history.json", raising=False)
    monkeypatch.setattr(module, "LOCK_FILE", state_dir / ".lock", raising=False)
    monkeypatch.setattr(module, "LOG_DIR", log_dir, raising=False)
    monkeypatch.setattr(module, "ARCHIVE_FILE", state_dir / "queue_archive.jsonl", raising=False)
    monkeypatch.setattr(module, "ESCALATIONS_FILE", state_dir / "escalations.jsonl", raising=False)
    monkeypatch.setattr(module, "notify", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(module, "tmp", tmp_path, raising=False)
    if hasattr(module, "_configure_algorithm"):
        module._configure_algorithm("legacy")

    yield module

    sys.modules.pop(module_name, None)
