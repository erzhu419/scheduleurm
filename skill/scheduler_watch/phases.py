"""Watch-loop phase timing helpers."""

from __future__ import annotations

import time
from typing import Callable, Optional


class WatchPhaseRecorder:
    """Collect slow watch-loop phases using scheduler's JSON payload format."""

    def __init__(self, threshold_s: float, *, clock: Optional[Callable[[], float]] = None):
        self.threshold_s = float(threshold_s)
        self._clock = clock or time.time
        self.phases: list[dict] = []

    def record(self, name: str, started_at: float, **extra) -> Optional[dict]:
        seconds = self._clock() - started_at
        if seconds < self.threshold_s:
            return None
        item = {"phase": name, "seconds": round(seconds, 3)}
        item.update({k: v for k, v in extra.items() if v is not None})
        self.phases.append(item)
        return item

    def payload(self) -> dict:
        return {
            "threshold_s": self.threshold_s,
            "phases": list(self.phases),
        }

    def __bool__(self) -> bool:
        return bool(self.phases)
