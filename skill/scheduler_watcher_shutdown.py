"""Compatibility alias for the canonical watcher shutdown module.

This module owns mutable shutdown state. Re-exporting with ``import *`` creates
a facade whose attributes can diverge from the globals used by the imported
functions. Alias the module object itself so every import path observes and
patches one state registry.
"""

from __future__ import annotations

import sys

if __package__:
    from .scheduler_watch import shutdown as _implementation
else:
    from scheduler_watch import shutdown as _implementation

sys.modules[__name__] = _implementation
