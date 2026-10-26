"""Read active display records without blocking the TUI's Python interpreter."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_active_state(path: str | Path) -> dict:
    try:
        with Path(path).open() as stream:
            state = json.load(stream)
    except FileNotFoundError:
        return {"tasks": []}
    state["tasks"] = [
        task for task in state.get("tasks", [])
        if task.get("status") in ("queued", "launching", "running")
    ]
    return state


if __name__ == "__main__":
    json.dump(read_active_state(sys.argv[1]), sys.stdout, separators=(",", ":"))
