"""Shared helpers for read-only external-SOTA workload adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from simulation.defaults import build_default_cache
from simulation.tasksets import taskset_by_name


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts" / "sota_adapters"


def add_common_args(parser: argparse.ArgumentParser, *, default_name: str) -> None:
    parser.add_argument("--taskset", default="hybrid_research_portfolio")
    parser.add_argument("--output", default=str(ARTIFACT_ROOT / default_name))


def load_taskset_rows(taskset_name: str) -> dict[str, Any]:
    cache = build_default_cache()
    taskset = taskset_by_name(taskset_name)
    members = []
    for member in taskset.members:
        profiles = []
        for record in cache.profiles(member.workload_key, include_boundaries=True):
            profiles.append(record.snapshot())
        members.append({
            "member": member.snapshot(cache),
            "profiles": profiles,
        })
    return {
        "taskset": taskset.snapshot(cache),
        "members": members,
    }


def write_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def sanitize_name(value: str) -> str:
    out = []
    for ch in str(value).lower():
        if ch.isalnum() or ch == "-":
            out.append(ch)
        elif ch in {"_", ".", "/"}:
            out.append("-")
    text = "".join(out).strip("-")
    return text or "scheduleurm-workload"
