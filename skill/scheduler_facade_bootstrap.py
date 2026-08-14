"""Import-path and optional helper setup for the scheduler.py facade."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_env_deploy(skill_dir: Path) -> Any:
    try:
        spec = importlib.util.spec_from_file_location("env_deploy", str(skill_dir / "env_deploy.py"))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception:
        return None
    return None


def prepare_scheduler_facade(file_path: str) -> tuple[Any, Path, Path]:
    skill_dir = Path(file_path).resolve().parent
    repo_root = skill_dir.parent
    for path in (str(skill_dir), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return _load_env_deploy(skill_dir), skill_dir, repo_root
