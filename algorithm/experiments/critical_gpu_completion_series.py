"""Resume-safe driver for the pre-registered critical GPU campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .critical_gpu_completion_campaign import (
    ARTIFACT_ROOT,
    CAMPAIGN_PREFIX,
    NODE_SPECS,
    PROTOCOL,
    build_critical_gpu_completion_campaign,
    campaign_cells,
)


def wave_artifact(node: str, wave: int) -> Path:
    return ARTIFACT_ROOT / f"{CAMPAIGN_PREFIX}_{node}_r{int(wave):02d}_20260803.json"


def completed_wave(node: str, wave: int) -> dict[str, Any] | None:
    path = wave_artifact(node, wave)
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {cell["cell_id"] for cell in campaign_cells(node=node, wave=wave)}
    observed = {str(row.get("cell_id")) for row in report.get("rows") or []}
    if not (
        report.get("measurement_protocol") == PROTOCOL
        and report.get("status") == "PASS"
        and report.get("pass") is True
        and expected == observed
        and all(bool(row.get("ready")) or bool(row.get("capacity_boundary")) for row in report["rows"])
    ):
        return None
    return report


def run_series(
    *,
    node: str,
    first_wave: int = 1,
    last_wave: int = 13,
    allow_launch: bool = False,
) -> dict[str, Any]:
    if node not in NODE_SPECS:
        raise ValueError(f"unsupported node {node!r}")
    if not (1 <= int(first_wave) <= int(last_wave) <= 13):
        raise ValueError("registered series must satisfy 1 <= first_wave <= last_wave <= 13")

    rows: list[dict[str, Any]] = []
    for wave in range(int(first_wave), int(last_wave) + 1):
        prior = completed_wave(node, wave)
        if prior is not None:
            rows.append({"wave": wave, "status": "SKIPPED_ALREADY_PASS", "campaign_id": prior["campaign_id"]})
            continue
        if not allow_launch:
            rows.append({"wave": wave, "status": "PENDING_LAUNCH"})
            continue
        report = build_critical_gpu_completion_campaign(node=node, wave=wave, allow_launch=True)
        rows.append(
            {
                "wave": wave,
                "status": report.get("status"),
                "campaign_id": report.get("campaign_id"),
                "ready_row_count": report.get("ready_row_count", 0),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
        if report.get("status") != "PASS":
            break

    expected_count = int(last_wave) - int(first_wave) + 1
    finished = sum(row["status"] in {"PASS", "SKIPPED_ALREADY_PASS"} for row in rows)
    return {
        "gate": "critical_gpu_completion_series",
        "node": node,
        "first_wave": int(first_wave),
        "last_wave": int(last_wave),
        "allow_launch": bool(allow_launch),
        "waves": rows,
        "pass": finished == expected_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=sorted(NODE_SPECS), required=True)
    parser.add_argument("--first-wave", type=int, default=1)
    parser.add_argument("--last-wave", type=int, default=13)
    parser.add_argument("--allow-launch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_series(
        node=args.node,
        first_wave=args.first_wave,
        last_wave=args.last_wave,
        allow_launch=args.allow_launch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] or not args.allow_launch else 2


if __name__ == "__main__":
    raise SystemExit(main())
