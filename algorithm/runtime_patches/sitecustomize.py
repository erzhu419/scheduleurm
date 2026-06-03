"""Experiment-only Python startup patches.

This module is loaded automatically when its directory is prepended to
``PYTHONPATH``.  It is intentionally scoped to Scheduleurm experiments so we do
not mutate shared conda environments while still keeping real workload commands
portable across the local host and remote GPU nodes.
"""
from __future__ import annotations

import platform
import re


def _patch_conda_forge_sys_version_parser() -> None:
    """Accept CPython sys.version strings emitted by recent conda-forge builds.

    Some remote nodes expose Python 3.11.15 as
    ``3.11.15 | packaged by conda-forge | (...) [GCC ...]`` while their
    stdlib ``platform`` regex only accepts the vanilla CPython form.  JAX imports
    ``cloudpickle``, which calls ``platform.python_implementation()`` during
    startup, so the parse failure kills real RL workloads before they reach
    CUDA/JAX initialization.
    """

    parser = getattr(platform, "_sys_version_parser", None)
    if parser is None or "packaged\\ by\\ conda" in getattr(parser, "pattern", ""):
        return
    platform._sys_version_cache.clear()
    platform._sys_version_parser = re.compile(
        r"([\w.+]+)\s*"
        r"(?:\ \|\ packaged\ by\ conda(?:\-forge)?\ \|)?\s*"
        r"\(#?([^,]+)(?:,\s*([\w ]*)(?:,\s*([\w :]*))?)?\)\s*"
        r"\[([^\]]+)\]?"
    )


_patch_conda_forge_sys_version_parser()
