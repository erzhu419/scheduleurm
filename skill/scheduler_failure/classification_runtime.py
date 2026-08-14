"""Runtime-bound failure category and local OOM detection wrappers."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_failure_classification_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _detect_oom_kills_local(state):
        return _ns(namespace, "_detect_oom_kills_local_impl")(
            state,
            deps=_ns(namespace, "_LocalOomKillDetectionDeps")(
                syslog_path="/var/log/syslog",
                path_exists=lambda path: Path(path).exists(),
                check_output=subprocess.check_output,
                now=time.time,
            ),
        )

    def _classify_failure(diag):
        return _ns(namespace, "_classify_failure_impl")(
            diag,
            deps=_ns(namespace, "_FailureClassificationDeps")(
                env_missing_patterns=_ns(namespace, "ENV_MISSING_PATTERNS"),
                python_import_patterns=_ns(namespace, "PYTHON_IMPORT_PATTERNS"),
                cuda_runtime_patterns=_ns(namespace, "CUDA_RUNTIME_PATTERNS"),
                invalid_flag_patterns=_ns(namespace, "INVALID_FLAG_PATTERNS"),
                disk_full_patterns=_ns(namespace, "DISK_FULL_PATTERNS"),
                oom_patterns=_ns(namespace, "OOM_PATTERNS"),
            ),
        )

    return {
        "_detect_oom_kills_local": _detect_oom_kills_local,
        "_classify_failure": _classify_failure,
    }
