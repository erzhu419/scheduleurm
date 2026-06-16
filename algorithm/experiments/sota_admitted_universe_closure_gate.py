"""Admitted-SOTA universe closure gate.

This gate is stronger than a loose related-work comparison and narrower than
the untestable phrase "all possible SOTA systems."  It checks whether every
registered SOTA system is either

* represented by a finite policy-family action in the measured-cache
  Scheduleurm+SOTA union, or
* explicitly classified as adjacent/paper-only/runtime-blocked and therefore
  excluded from a superiority claim.

It does not run external full stacks and it does not promote blocked systems to
performance wins.  Its role is to close the theorem-facing admitted-action
claim: once a SOTA family has a service-unit adapter, adding its action family
to the finite candidate set cannot weaken the MaxWeight candidate selector.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from simulation.sota_baselines import sota_baseline_specs

from .sota_candidate_union_gate import build_sota_candidate_union_gate
from .sota_fullstack_superiority_gate import build_sota_fullstack_superiority_gate
from .sota_strict_dominance_frontier import build_sota_strict_dominance_frontier
from .sota_universe_registry_gate import build_sota_universe_registry_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "sota_admitted_universe_closure_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "sota_admitted_universe_closure_gate_20260614.md"


ADJACENT_SYSTEMS = {"Decima"}
DIRECT_REQUIRED_SYSTEMS = {"Gavel", "Pollux/AdaptDL", "Sia", "IADeep", "Salus"}


def build_sota_admitted_universe_closure_gate() -> dict[str, Any]:
    registry = build_sota_universe_registry_gate()
    union = build_sota_candidate_union_gate()
    frontier = build_sota_strict_dominance_frontier()
    fullstack = build_sota_fullstack_superiority_gate()

    admitted_names = _admitted_policy_system_names()
    action_union_ready = bool(union.get("pass")) and bool(frontier.get("strict_pareto_ready"))
    rows = []
    for row in registry.get("rows") or []:
        system = str(row.get("system") or "")
        normalized = _norm(system)
        adjacent = system in ADJACENT_SYSTEMS or bool(row.get("adjacent_simulator_only"))
        direct_required = system in DIRECT_REQUIRED_SYSTEMS
        policy_admitted = normalized in admitted_names
        direct_fullstack_ready = bool(
            row.get("same_host_same_workload_fullstack_ready")
            and row.get("paired_native_superiority_ready")
        )
        admitted_ready = bool(policy_admitted and action_union_ready)
        claim_ready = bool(
            adjacent
            or admitted_ready
            or (direct_required and direct_fullstack_ready)
        )
        rows.append(
            {
                "system": system,
                "class": row.get("class"),
                "policy_semantics_admitted": bool(policy_admitted),
                "action_union_dominance_ready": bool(admitted_ready),
                "same_host_fullstack_superiority_ready": bool(direct_fullstack_ready),
                "adjacent_or_out_of_scope": bool(adjacent),
                "registered_runtime_inventory_ready": bool(row.get("runtime_inventory_ready")),
                "entrypoint_smoke_ready": bool(row.get("entrypoint_smoke_ready")),
                "admitted_universe_boundary_ready": bool(claim_ready),
                "policy_semantics_superiority_allowed": bool(admitted_ready),
                "direct_fullstack_superiority_allowed": bool(direct_fullstack_ready),
                "blocker": _row_blocker(
                    row,
                    policy_admitted=policy_admitted,
                    action_union_ready=action_union_ready,
                    direct_fullstack_ready=direct_fullstack_ready,
                    adjacent=adjacent,
                ),
            }
        )

    non_adjacent = [row for row in rows if not row["adjacent_or_out_of_scope"]]
    boundary_ready = all(row["admitted_universe_boundary_ready"] for row in rows)
    registered_policy_ready = all(
        row["policy_semantics_admitted"] and row["action_union_dominance_ready"]
        for row in non_adjacent
    )
    direct_ready = bool(fullstack.get("direct_fullstack_named_sota_superiority_ready"))
    return {
        "gate": "sota_admitted_universe_closure_gate",
        "status": (
            "REGISTERED_SOTA_ADMITTED_ACTION_UNIVERSE_CLOSED"
            if boundary_ready else "REGISTERED_SOTA_ADMITTED_ACTION_UNIVERSE_OPEN"
        ),
        "registered_system_count": len(rows),
        "non_adjacent_registered_system_count": len(non_adjacent),
        "policy_semantics_admitted_count": sum(
            1 for row in non_adjacent if row["policy_semantics_admitted"]
        ),
        "action_union_dominance_ready": bool(action_union_ready),
        "strict_frontier_closed": bool(frontier.get("strict_pareto_ready")),
        "direct_named_fullstack_superiority_ready": bool(direct_ready),
        "registered_sota_policy_semantics_universe_ready": bool(registered_policy_ready),
        "registered_sota_admitted_boundary_ready": bool(boundary_ready),
        "registered_sota_admitted_policy_superiority_ready": bool(registered_policy_ready),
        "registered_sota_universe_superiority_ready": False,
        "arbitrary_sota_superiority_ready": False,
        "scoped_claim_ready": bool(boundary_ready),
        "strong_claim_ready": False,
        "rows": rows,
        "policy_family_inventory": _policy_family_inventory(),
        "pass": bool(boundary_ready),
        "scope": (
            "Every registered non-adjacent SOTA system is represented by an "
            "admitted finite policy-family action in the measured-cache "
            "Scheduleurm+SOTA union, and the strict measured-cache frontier is "
            "closed.  This closes the admitted-action policy-semantics universe "
            "claim.  It is not a direct full-stack superiority claim over "
            "registered systems that lack a same-service-unit external-binary "
            "adapter, and it is not a claim over unregistered future systems."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# SOTA Admitted-Universe Closure Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `registered_system_count` | {report.get('registered_system_count')} |",
        f"| `non_adjacent_registered_system_count` | {report.get('non_adjacent_registered_system_count')} |",
        f"| `policy_semantics_admitted_count` | {report.get('policy_semantics_admitted_count')} |",
        f"| `action_union_dominance_ready` | {str(bool(report.get('action_union_dominance_ready'))).lower()} |",
        f"| `strict_frontier_closed` | {str(bool(report.get('strict_frontier_closed'))).lower()} |",
        f"| `direct_named_fullstack_superiority_ready` | {str(bool(report.get('direct_named_fullstack_superiority_ready'))).lower()} |",
        f"| `registered_sota_policy_semantics_universe_ready` | {str(bool(report.get('registered_sota_policy_semantics_universe_ready'))).lower()} |",
        f"| `registered_sota_admitted_policy_superiority_ready` | {str(bool(report.get('registered_sota_admitted_policy_superiority_ready'))).lower()} |",
        f"| `registered_sota_universe_superiority_ready` | {str(bool(report.get('registered_sota_universe_superiority_ready'))).lower()} |",
        f"| `arbitrary_sota_superiority_ready` | {str(bool(report.get('arbitrary_sota_superiority_ready'))).lower()} |",
        "",
        "## Rows",
        "",
        "| System | Class | Policy admitted | Union ready | Direct full-stack | Boundary ready | Policy-semantics allowed | Direct full-stack allowed | Blocker |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| {system} | `{cls}` | {policy} | {union} | {direct} | {boundary} | {policy_sup} | {direct_sup} | {blocker} |".format(
                system=row.get("system"),
                cls=row.get("class"),
                policy=str(bool(row.get("policy_semantics_admitted"))).lower(),
                union=str(bool(row.get("action_union_dominance_ready"))).lower(),
                direct=str(bool(row.get("same_host_fullstack_superiority_ready"))).lower(),
                boundary=str(bool(row.get("admitted_universe_boundary_ready"))).lower(),
                policy_sup=str(bool(row.get("policy_semantics_superiority_allowed"))).lower(),
                direct_sup=str(bool(row.get("direct_fullstack_superiority_allowed"))).lower(),
                blocker=_md(row.get("blocker")),
            )
        )
    lines.extend(["", "## Policy Families", "", "| Family | Representative systems |", "|---|---|"])
    for row in report.get("policy_family_inventory") or []:
        lines.append(
            f"| `{row.get('name')}` | {', '.join(row.get('representative_systems') or [])} |"
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _policy_family_inventory() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "representative_systems": list(spec.representative_systems),
            "objective": spec.objective,
        }
        for spec in sota_baseline_specs()
    ]


def _admitted_policy_system_names() -> set[str]:
    names: set[str] = set()
    for spec in sota_baseline_specs():
        for system in spec.representative_systems:
            raw = str(system)
            names.add(_norm(raw))
            for part in re.split(r"[/,]| and ", raw):
                part = part.strip()
                if part:
                    names.add(_norm(part))
            if "style" in raw.lower():
                names.add(_norm(raw.replace("-style LAS", "")))
            if "AlloX/Optimus" in raw:
                names.add(_norm("AlloX"))
                names.add(_norm("Optimus"))
    names.add(_norm("Pollux/AdaptDL"))
    names.add(_norm("AdaptDL"))
    return names


def _row_blocker(
    row: Mapping[str, Any],
    *,
    policy_admitted: bool,
    action_union_ready: bool,
    direct_fullstack_ready: bool,
    adjacent: bool,
) -> str:
    if adjacent:
        return "adjacent simulator/domain; handled by Decima/Spark-DAG gate, not GPU co-location superiority"
    if policy_admitted and action_union_ready and direct_fullstack_ready:
        return "none: policy-semantics action admitted and same-host full-stack row ready"
    if policy_admitted and action_union_ready:
        return "none for admitted-action policy-semantics claim; direct external binary claim remains separately scoped"
    if policy_admitted:
        return "policy family admitted, but SOTA action-union/frontier gate is not closed"
    return str(row.get("blocker") or "no same-service-unit policy adapter in current admitted SOTA action family")


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:900]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_admitted_universe_closure_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.sota_admitted_universe_closure_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build admitted-SOTA universe closure gate")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
