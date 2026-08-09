"""Fail-closed launcher for one controlled loaded-state completion wave.

The measurement campaign intentionally records task-native natural completion,
but it does not own the cluster and therefore cannot reserve GPUs against other
users.  This runner refuses to start a full wave unless every registered GPU on
the selected hardware node is idle at the preflight snapshot.  The downstream
stochastic gate still verifies each scenario's hash-bound pre-launch snapshot,
so a later race fails closed rather than entering the theorem-facing cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.experiments.critical_gpu_completion_campaign import (
    NODE_SPECS,
    _gpu_idle_snapshot,
)
from algorithm.experiments.critical_gpu_loaded_completion_campaign import (
    ALL_WAVES,
    ARTIFACT_ROOT,
    CAMPAIGN_DATE,
    build_loaded_completion_campaign,
)


def run_safe_loaded_wave(
    *,
    node: str,
    wave: int,
    allow_launch: bool = False,
    keep_remote_output: bool = False,
) -> dict[str, Any]:
    if node not in NODE_SPECS:
        raise ValueError(f"unsupported node {node!r}")
    if int(wave) not in ALL_WAVES:
        raise ValueError(f"wave must be one of {ALL_WAVES!r}")
    common = {
        "gate": "critical_gpu_loaded_wave_runner",
        "schema_version": 1,
        "node": node,
        "wave": int(wave),
        "allow_launch": bool(allow_launch),
        "ordinary_running_tasks_touched": False,
        "legacy_scheduler_limits_bypassed": True,
        "requires_all_registered_gpus_idle": True,
        "downstream_row_level_prelaunch_audit_required": True,
    }
    if not allow_launch:
        campaign = build_loaded_completion_campaign(
            node=node,
            wave=int(wave),
            allow_launch=False,
            keep_remote_output=keep_remote_output,
        )
        return {
            **common,
            "status": "MANIFEST_ONLY",
            "pass": False,
            "launched": False,
            "idle_snapshot": None,
            "campaign": campaign,
        }

    idle = _gpu_idle_snapshot(NODE_SPECS[node])
    if idle.get("ready") is not True:
        return {
            **common,
            "status": "WAIT_GPU_IDLE",
            "pass": False,
            "launched": False,
            "idle_snapshot": idle,
            "campaign": None,
            "claim_boundary": (
                "No benchmark process was launched because at least one registered "
                "GPU failed the idle preflight. Existing user processes were not "
                "modified."
            ),
        }

    campaign = build_loaded_completion_campaign(
        node=node,
        wave=int(wave),
        allow_launch=True,
        keep_remote_output=keep_remote_output,
    )
    return {
        **common,
        "status": "PASS" if campaign.get("pass") is True else "CAMPAIGN_INCOMPLETE",
        "pass": campaign.get("pass") is True,
        "launched": True,
        "idle_snapshot": idle,
        "campaign": campaign,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    idle = report.get("idle_snapshot") or {}
    lines = [
        "# Critical GPU Loaded-Wave Safe Runner",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Node: `{report.get('node')}`",
        f"- Wave: `{report.get('wave')}`",
        f"- Launched: `{str(bool(report.get('launched'))).lower()}`",
        f"- All registered GPUs idle: `{str(bool(idle.get('ready'))).lower()}`",
        "",
        "| GPU | Used MiB | Utilization |",
        "|---:|---:|---:|",
    ]
    for row in idle.get("selected_gpus") or []:
        lines.append(
            f"| {row.get('index')} | {row.get('used_mb')} | {row.get('util_pct')}% |"
        )
    boundary = str(report.get("claim_boundary") or "").strip()
    if boundary:
        lines.extend(["", boundary])
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], *, output: Path, markdown: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(markdown_report(report), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, choices=tuple(NODE_SPECS))
    parser.add_argument("--wave", required=True, type=int, choices=ALL_WAVES)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--keep-remote-output", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = run_safe_loaded_wave(
        node=args.node,
        wave=args.wave,
        allow_launch=args.allow_launch,
        keep_remote_output=args.keep_remote_output,
    )
    output = args.output or (
        ARTIFACT_ROOT
        / f"critical_gpu_loaded_wave_runner_{args.node}_r{args.wave:02d}_{CAMPAIGN_DATE}.json"
    )
    markdown = args.markdown_output or output.with_suffix(".md")
    write_outputs(report, output=output, markdown=markdown)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass": report["pass"],
                "launched": report["launched"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
