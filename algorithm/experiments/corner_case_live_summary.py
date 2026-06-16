"""Summarize live corner-case probe artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"


BASELINES = {
    "node007_28p8gb_add_small": "node007_empty_small",
    "node007_30gb_add_small": "node007_empty_small",
    "node007_30gb_add_small_stable": "node007_empty_small",
    "cpu_half_node001_add_single_v2": "cpu_empty_node001_single",
    "cpu_full_node001_add_single_v2": "cpu_empty_node001_single",
    "cpu_half_node001_add_single": "cpu_empty_node002_single",
    "cpu_full_node003_add_single": "cpu_empty_node002_single",
}


def build_summary(run_id: str) -> dict[str, Any]:
    report_dir = RUN_ROOT / run_id / "reports"
    rows = []
    by_id = {}
    for path in sorted(report_dir.glob("*_remote_corner_case_probe.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_path"] = str(path)
        by_id[str(row.get("scenario_id"))] = row
    for scenario_id, row in sorted(by_id.items()):
        baseline_id = BASELINES.get(scenario_id)
        baseline = by_id.get(baseline_id or "")
        loaded_rate = float(row.get("probe_last_rate_step_s") or 0.0)
        base_rate = float((baseline or {}).get("probe_last_rate_step_s") or 0.0)
        ratio = loaded_rate / base_rate if base_rate > 0 else None
        rows.append({
            "scenario_id": scenario_id,
            "node": row.get("node"),
            "pass": bool(row.get("pass")),
            "stable_eta_ready": bool((row.get("stable_eta") or {}).get("ready")),
            "probe_last_rate_step_s": loaded_rate,
            "probe_mean_rate_step_s": float(row.get("probe_mean_rate_step_s") or 0.0),
            "background_last_rate_step_s": float(row.get("background_last_rate_step_s") or 0.0),
            "baseline_scenario_id": baseline_id or "",
            "loaded_to_empty_rate_ratio": ratio,
            "artifact": row.get("_path"),
        })
    required = {
        "node007_empty_small",
        "node007_30gb_add_small_stable",
        "cpu_empty_node001_single",
        "cpu_half_node001_add_single_v2",
        "cpu_full_node001_add_single_v2",
    }
    present = {row["scenario_id"] for row in rows}
    stable_required = [
        row for row in rows
        if row["scenario_id"] in required
    ]
    return {
        "gate": "corner_case_live_summary",
        "run_id": run_id,
        "required_scenarios": sorted(required),
        "missing_required_scenarios": sorted(required - present),
        "scenario_count": len(rows),
        "all_required_present": required <= present,
        "all_required_pass": all(row["pass"] for row in stable_required) and required <= present,
        "all_required_stable_eta": all(row["stable_eta_ready"] for row in stable_required) and required <= present,
        "rows": rows,
        "policy_admission_matrix": _policy_admission_matrix(rows),
    }


def write_summary(summary: Mapping[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(_markdown(summary, out_json), encoding="utf-8")


def _markdown(summary: Mapping[str, Any], out_json: Path) -> str:
    lines = [
        "# Corner-Case Live Probe Summary",
        "",
        f"JSON artifact: `{out_json}`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| scenario_count | {summary.get('scenario_count')} |",
        f"| all_required_present | {str(bool(summary.get('all_required_present'))).lower()} |",
        f"| all_required_pass | {str(bool(summary.get('all_required_pass'))).lower()} |",
        f"| all_required_stable_eta | {str(bool(summary.get('all_required_stable_eta'))).lower()} |",
        "",
        "| scenario | node | stable ETA | last rate | background rate | ratio to empty |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows") or []:
        ratio = row.get("loaded_to_empty_rate_ratio")
        ratio_text = "" if ratio is None else f"{float(ratio):.4f}"
        lines.append(
            f"| `{row.get('scenario_id')}` | `{row.get('node')}` | "
            f"{str(bool(row.get('stable_eta_ready'))).lower()} | "
            f"{float(row.get('probe_last_rate_step_s') or 0.0):.6g} | "
            f"{float(row.get('background_last_rate_step_s') or 0.0):.6g} | "
            f"{ratio_text} |"
        )
    lines.extend([
        "",
        "## Policy Admission Matrix",
        "",
        "| scenario | Scheduleurm theorem | Salus/Gandiva-style packing | Pollux/Sia-style goodput | Gavel-style finish time | legacy safety |",
        "|---|---|---|---|---|---|",
    ])
    for row in summary.get("policy_admission_matrix") or []:
        lines.append(
            f"| `{row.get('scenario_id')}` | {row.get('scheduleurm_theorem')} | "
            f"{row.get('memory_pack_sota')} | {row.get('goodput_sota')} | "
            f"{row.get('gavel_finish_time')} | {row.get('legacy_safety')} |"
        )
    lines.extend([
        "",
        "Interpretation boundary: these are controlled synthetic marginal-efficiency "
        "probes.  They validate ETA logging, co-location feasibility, and rate "
        "ratios for this measured slice; they are not a global theorem population "
        "unless admitted into the service-cache/replay gate.",
        "",
    ])
    return "\n".join(lines)


def _policy_admission_matrix(rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        ratio = row.get("loaded_to_empty_rate_ratio")
        stable = bool(row.get("stable_eta_ready"))
        has_background = float(row.get("background_last_rate_step_s") or 0.0) > 0.0
        is_gpu = scenario_id.startswith("node007")
        is_cpu = scenario_id.startswith("cpu")
        if not stable:
            theorem = "pending stable ETA"
            goodput = "pending stable ETA"
        else:
            theorem = "admit: positive lower service"
            if ratio is None:
                goodput = "baseline row"
            elif float(ratio) >= 0.90:
                goodput = f"admit: ratio {float(ratio):.3f}"
            else:
                goodput = f"defer: ratio {float(ratio):.3f}"
        if is_gpu and has_background:
            memory_pack = "admit if memory fit"
            legacy = "node007 override admits; generic one-third would block"
        elif is_gpu:
            memory_pack = "baseline row"
            legacy = "baseline row"
        elif is_cpu:
            memory_pack = "not applicable"
            legacy = "capacity gate admits if CPU/RAM fit"
        else:
            memory_pack = "not classified"
            legacy = "not classified"
        gavel = "baseline row" if not has_background else "needs resident-delay/JCT holdout"
        out.append({
            "scenario_id": scenario_id,
            "scheduleurm_theorem": theorem,
            "memory_pack_sota": memory_pack,
            "goodput_sota": goodput,
            "gavel_finish_time": gavel,
            "legacy_safety": legacy,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)
    summary = build_summary(args.run_id)
    write_summary(summary, out_json=Path(args.out_json), out_md=Path(args.out_md))
    print(json.dumps({
        "pass": bool(summary.get("all_required_pass")) and bool(summary.get("all_required_stable_eta")),
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, indent=2, sort_keys=True))
    return 0 if summary.get("all_required_pass") and summary.get("all_required_stable_eta") else 2


if __name__ == "__main__":
    raise SystemExit(main())
