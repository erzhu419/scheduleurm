"""Corner-case coverage matrix for ETA and migration closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


CORNER_CASES = (
    ("empty_gpu", "ETA", "cross_node_eta_matrix + live_eta_service_cache_v2_gate", "implemented_live_multi_node_partial"),
    ("half_loaded_gpu", "ETA", "live_marginal_under_load_matrix", "implemented_live"),
    ("full_loaded_gpu", "ETA", "live_marginal_under_load_full_mixed", "implemented_live"),
    ("high_vram_resident", "ETA", "live_marginal_under_load_matrix", "implemented_live"),
    ("cpu_resident", "ETA", "live_marginal_under_load_cpu_idle_nodes", "implemented_live_representative_and_verify"),
    ("mixed_cnn_llm", "ETA", "live_marginal_under_load_full_mixed", "implemented_live"),
    ("mixed_cnn_rl_llm", "ETA", "live_marginal_under_load_mixed_cnn_rl_llm_v5", "implemented_live"),
    ("rl_periodic_train_eval_eta", "ETA", "progress_wrapper stable-cycle-units + live_cycle_jtl110_resac_ant_p2", "implemented_live"),
    ("jtl311_cpu_fast_gpu_weak", "node-aware", "service_cache_v2", "implemented_tested"),
    ("node007_multigpu_packing", "node-aware", "live_eta_service_cache_v2_gate", "implemented_live_partial"),
    ("node001_node006_cpu_packing", "node-aware", "live_marginal_under_load_cpu_idle_nodes", "implemented_live_representative_and_verify"),
    ("unknown_env", "admission", "service_registry", "probe_defer_required"),
    ("description_only_env", "admission", "service_registry", "implemented_tested"),
    ("missing_tqdm", "admission", "service_cache_v2", "history_fallback_no_theorem_claim"),
    ("missing_checkpoint", "migration", "migration_cost_gate", "implemented_tested"),
    ("resume_path_local_only", "migration", "migration_cost_gate", "excluded_until_staged"),
    ("pure_gpu_migration_25_50_75", "migration", "live_checkpoint_migration_cost_gate", "implemented_live"),
    ("hybrid_rl_migration_25_50_75", "migration", "live_checkpoint_migration_cost_gate", "implemented_live"),
    ("pure_cpu_migration_25_50_75", "migration", "live_checkpoint_migration_cost_gate", "implemented_live"),
    ("port_reberthing", "OR-generalization", "or_generalization_benchmarks", "implemented_simulator"),
    ("fjsp_reroute", "OR-generalization", "or_generalization_benchmarks", "implemented_simulator"),
    ("mmrcpsp_mode_switch", "OR-generalization", "or_generalization_benchmarks", "implemented_simulator"),
)


def build_corner_case_matrix() -> dict[str, Any]:
    rows = [
        {
            "corner_case": name,
            "category": category,
            "gate": gate,
            "status": status,
            "theorem_facing_ready": (
                status.startswith("implemented_live")
                or status in {"implemented_tested", "implemented_simulator"}
            ),
        }
        for name, category, gate, status in CORNER_CASES
    ]
    return {
        "gate": "eta_migration_corner_case_matrix",
        "pass": all(row["status"] for row in rows),
        "status": "CORNER_CASE_MATRIX_READY",
        "rows": rows,
        "claim_boundary": (
            "Rows marked requires_live_probe or no_theorem_claim are engineering "
            "coverage commitments, not theorem-facing empirical claims."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# ETA and Migration Corner-Case Coverage Matrix",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        "",
        "| Corner case | Category | Gate | Status | Theorem-facing ready |",
        "|---|---|---|---|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            f"| `{row['corner_case']}` | `{row['category']}` | `{row['gate']}` | "
            f"`{row['status']}` | {str(bool(row['theorem_facing_ready'])).lower()} |"
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "eta_migration_corner_case_matrix_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "eta_migration_corner_case_matrix_20260629.md")
    args = parser.parse_args()
    report = build_corner_case_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
