"""Runtime-bound log helper wrappers for scheduler.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_log_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def maybe_rotate_log(path: Path, max_mb: int, generations: int):
        try:
            if not path.exists() or path.stat().st_size < max_mb * 1024 * 1024:
                return
        except Exception:
            return
        base = str(path)
        for i in range(generations - 1, 0, -1):
            src, dst = Path(f"{base}.{i}"), Path(f"{base}.{i+1}")
            if src.exists():
                try:
                    src.rename(dst)
                except Exception:
                    pass
        try:
            path.rename(Path(f"{base}.1"))
        except Exception:
            pass

    def append_heal_inbox(filepath, entry):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        _ns(namespace, "maybe_rotate_log")(path, max_mb=1, generations=3)
        with open(path, "a") as handle:
            handle.write(entry)

    return {
        "maybe_rotate_log": maybe_rotate_log,
        "append_heal_inbox": append_heal_inbox,
    }
