"""Trace-driven fast-forward simulation for Scheduleurm experiments.

This package is intentionally separate from ``skill/scheduler.py``.  It uses
real progress/service observations as a cache, then replays whole batches without
running every training task to completion.
"""

