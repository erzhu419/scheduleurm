"""Named external runtime probe gate.

This module keeps the historical entrypoint name for compatibility, but the
generated claim is intentionally narrower than "direct SOTA superiority."  It
records scoped same-host same-workload runtime probes for named external
systems and keeps arbitrary/direct full-stack SOTA superiority false.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .direct_sota_fullstack_readiness import build_direct_sota_fullstack_readiness
from .iadeep_fullstack_same_workload_gate import build_iadeep_fullstack_same_workload_gate
from .pollux_fullstack_same_workload_gate import build_pollux_fullstack_same_workload_gate
from .salus_fullstack_same_workload_gate import build_salus_fullstack_same_workload_gate
from .sia_fullstack_same_workload_gate import build_sia_fullstack_same_workload_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
GAVEL_MICRO = ARTIFACT_ROOT / "gavel_native_performance_microbaseline_20260612.json"
GAVEL_RESIDENT_DELAY = ARTIFACT_ROOT / "gavel_resident_delay_jct_holdout_20260613.json"
GAVEL_NATIVE_COMPARISON = ARTIFACT_ROOT / "gavel_native_same_workload_comparison_20260613.json"
GAVEL_PHYSICAL = ARTIFACT_ROOT / "gavel_physical_same_workload_gate_20260613.json"
NATIVE_ATTEMPTS = ARTIFACT_ROOT / "sota_native_execution_attempts_20260613.json"
NAMED_SOTA_ADAPTERS = {
    "gavel_simulation",
    "pollux_adaptdl_scheduler",
    "sia_goodput_scheduler",
    "iadeep_kubernetes_extender",
    "salus_gpu_sharing",
}


def build_sota_fullstack_superiority_gate() -> dict[str, Any]:
    readiness = build_direct_sota_fullstack_readiness(run_smoke=True)
    gavel_physical = _load_json(GAVEL_PHYSICAL)
    pollux_fullstack = build_pollux_fullstack_same_workload_gate()
    sia_fullstack = build_sia_fullstack_same_workload_gate()
    iadeep_fullstack = build_iadeep_fullstack_same_workload_gate()
    salus_fullstack = build_salus_fullstack_same_workload_gate()
    gavel_micro_path = _latest_artifact(
        "gavel_native_performance_microbaseline_*.json",
        GAVEL_MICRO,
    )
    native_attempts_path = _latest_artifact(
        "sota_native_execution_attempts_*.json",
        NATIVE_ATTEMPTS,
    )
    gavel_micro = _load_json(gavel_micro_path)
    gavel_resident_delay = _load_json(GAVEL_RESIDENT_DELAY)
    gavel_native_comparison = _load_json(GAVEL_NATIVE_COMPARISON)
    native_attempts = _load_json(native_attempts_path)
    tool_inventory = readiness.get("tool_inventory") or {}
    rows = []
    for row in readiness.get("adapters") or []:
        adapter = str(row.get("adapter") or "")
        pollux_row_ready = bool(
            adapter == "pollux_adaptdl_scheduler"
            and pollux_fullstack.get("scoped_pollux_fullstack_same_workload_ready")
        )
        sia_row_ready = bool(
            adapter == "sia_goodput_scheduler"
            and sia_fullstack.get("scoped_sia_fullstack_same_workload_ready")
        )
        iadeep_row_ready = bool(
            adapter == "iadeep_kubernetes_extender"
            and iadeep_fullstack.get("scoped_iadeep_fullstack_same_workload_ready")
        )
        salus_row_ready = bool(
            adapter == "salus_gpu_sharing"
            and salus_fullstack.get("scoped_salus_fullstack_same_workload_ready")
        )
        gavel_physical_row_ready = bool(
            adapter == "gavel_simulation"
            and gavel_physical.get("scoped_gavel_physical_fullstack_ready")
        )
        pollux_superiority_ready = bool(
            adapter == "pollux_adaptdl_scheduler"
            and pollux_fullstack.get("scoped_pollux_same_workload_superiority_ready")
        )
        sia_superiority_ready = bool(
            adapter == "sia_goodput_scheduler"
            and sia_fullstack.get("scoped_sia_same_workload_superiority_ready")
        )
        iadeep_superiority_ready = bool(
            adapter == "iadeep_kubernetes_extender"
            and iadeep_fullstack.get("scoped_iadeep_same_workload_superiority_ready")
        )
        salus_superiority_ready = bool(
            adapter == "salus_gpu_sharing"
            and salus_fullstack.get("scoped_salus_same_workload_superiority_ready")
        )
        gavel_superiority_ready = bool(
            adapter == "gavel_simulation"
            and gavel_physical.get("scoped_gavel_physical_same_workload_superiority_ready")
        )
        full_stack_ready = bool(
            row.get("same_workload_full_stack_ready")
            or gavel_physical_row_ready
            or pollux_row_ready
            or sia_row_ready
            or iadeep_row_ready
            or salus_row_ready
        )
        superiority_ready = bool(
            pollux_superiority_ready
            or sia_superiority_ready
            or iadeep_superiority_ready
            or salus_superiority_ready
            or gavel_superiority_ready
        )
        rows.append({
            "adapter": adapter,
            "regeneration_mode": "precomputed_live_run",
            "current_safe_reproduction": False,
            "requires_external_stack": True,
            "current_environment_has_tools": not bool(row.get("missing_required_tools")),
            "entrypoint_smoke_pass": bool(row.get("entrypoint_smoke_pass")),
            "same_workload_full_stack_ready": full_stack_ready,
            "pollux_same_workload_fullstack_ready": bool(pollux_row_ready),
            "pollux_scoped_same_workload_superiority_ready": bool(
                pollux_superiority_ready
            ),
            "sia_same_workload_fullstack_ready": bool(sia_row_ready),
            "sia_scoped_same_workload_superiority_ready": bool(
                sia_superiority_ready
            ),
            "iadeep_same_workload_fullstack_ready": bool(iadeep_row_ready),
            "iadeep_scoped_same_workload_superiority_ready": bool(
                iadeep_superiority_ready
            ),
            "salus_same_workload_fullstack_ready": bool(salus_row_ready),
            "salus_scoped_same_workload_superiority_ready": bool(
                salus_superiority_ready
            ),
            "native_microbaseline_ready": bool(
                adapter == "gavel_simulation"
                and gavel_micro.get("native_gavel_simulator_microbaseline_ready")
            ),
            "gavel_resident_delay_jct_holdout_ready": bool(
                adapter == "gavel_simulation"
                and gavel_resident_delay.get("gavel_style_resident_delay_jct_holdout_ready")
            ),
            "gavel_same_workload_native_simulator_superiority_ready": bool(
                adapter == "gavel_simulation"
                and gavel_native_comparison.get("same_workload_native_simulator_superiority_ready")
            ),
            "gavel_physical_same_workload_fullstack_ready": bool(gavel_physical_row_ready),
            "gavel_physical_same_workload_superiority_ready": bool(
                gavel_superiority_ready
            ),
            "missing_required_tools": row.get("missing_required_tools") or [],
            "missing_stack_assets": row.get("missing_stack_assets") or [],
            "blockers": row.get("blockers") or [],
            "scoped_runtime_probe_ready": bool(full_stack_ready),
            "scoped_runtime_probe_native_better": bool(
                full_stack_ready and superiority_ready
            ),
            "full_stack_superiority_claim_allowed": False,
            "strict_reason": _strict_reason(
                adapter,
                row,
                gavel_micro,
                gavel_resident_delay,
                gavel_native_comparison,
                gavel_physical,
                pollux_fullstack,
                sia_fullstack,
                iadeep_fullstack,
                salus_fullstack,
            ),
        })
    missing_tools = sorted(
        tool for tool, info in tool_inventory.items()
        if not bool((info or {}).get("available"))
    )
    named_rows = [row for row in rows if row.get("adapter") in NAMED_SOTA_ADAPTERS]
    named_ready_count = sum(1 for row in named_rows if row.get("same_workload_full_stack_ready"))
    named_superiority_count = sum(
        1 for row in named_rows if row.get("scoped_runtime_probe_native_better")
    )
    named_same_host_runtime_probe_ready = bool(
        len(named_rows) == len(NAMED_SOTA_ADAPTERS)
        and named_ready_count == len(NAMED_SOTA_ADAPTERS)
        and named_superiority_count == len(NAMED_SOTA_ADAPTERS)
    )
    status = (
        "NAMED_SAME_HOST_RUNTIME_PROBE_PASS_DIRECT_SOTA_FALSE"
        if named_same_host_runtime_probe_ready
        else "NAMED_SAME_HOST_RUNTIME_PROBE_PENDING"
    )
    return {
        "gate": "named_external_runtime_probe_gate",
        "legacy_entrypoint": "sota_fullstack_superiority_gate",
        "status": status,
        "gate_pass": True,
        "scoped_claim_ready": bool(named_same_host_runtime_probe_ready),
        "strong_claim_ready": False,
        "regeneration_mode": "precomputed_live_run",
        "current_safe_reproduction": False,
        "requires_external_stack": True,
        "current_environment_has_tools": not bool(missing_tools),
        "pass_meaning": (
            "named five same-host same-workload runtime probes with paired "
            "native-better rows are closed; direct full-stack SOTA superiority, "
            "registered-universe external-binary superiority, original "
            "multi-node superiority, and production-wide trace superiority are "
            "not claimed"
        ),
        "rows": rows,
        "tool_inventory": tool_inventory,
        "missing_tools": missing_tools,
        "gavel_native_microbaseline": {
            "artifact": str(gavel_micro_path),
            "ready": bool(gavel_micro.get("native_gavel_simulator_microbaseline_ready")),
            "usable_samples": int(gavel_micro.get("usable_native_performance_sample_count") or 0),
            "direct_full_stack_performance_ready": bool(gavel_micro.get("direct_full_stack_performance_ready")),
        },
        "native_execution_attempts": {
            "artifact": str(native_attempts_path),
            "ready_count": int(native_attempts.get("native_execution_ready_count") or 0),
            "direct_full_stack_ready_count": int(native_attempts.get("direct_full_stack_ready_count") or 0),
            "direct_fullstack_sota_superiority_ready": bool(
                native_attempts.get("direct_fullstack_sota_superiority_ready")
            ),
        },
        "gavel_resident_delay_jct_holdout": {
            "artifact": str(GAVEL_RESIDENT_DELAY),
            "ready": bool(gavel_resident_delay.get("gavel_style_resident_delay_jct_holdout_ready")),
            "usable_rows": int(gavel_resident_delay.get("usable_row_count") or 0),
            "strong_claim_ready": bool(gavel_resident_delay.get("strong_claim_ready")),
        },
        "gavel_same_workload_native_simulator_comparison": {
            "artifact": str(GAVEL_NATIVE_COMPARISON),
            "ready": bool(gavel_native_comparison.get("same_workload_native_simulator_superiority_ready")),
            "eligible_rows": int(gavel_native_comparison.get("eligible_row_count") or 0),
            "direct_fullstack_binary_superiority_ready": bool(
                gavel_native_comparison.get("direct_fullstack_binary_superiority_ready")
            ),
            "rows": gavel_native_comparison.get("rows") or [],
        },
        "gavel_physical_same_workload": {
            "artifact": str(GAVEL_PHYSICAL),
            "ready": bool(gavel_physical.get("scoped_gavel_physical_fullstack_ready")),
            "scoped_superiority_ready": bool(
                gavel_physical.get("scoped_gavel_physical_same_workload_superiority_ready")
            ),
            "same_service_scale_ready": bool(gavel_physical.get("same_service_scale_ready")),
            "native_wall_s": gavel_physical.get("native_wall_s"),
            "gavel_jct_s": gavel_physical.get("gavel_jct_s"),
            "ratio": gavel_physical.get("scheduleurm_native_to_gavel_jct_ratio"),
            "blockers": gavel_physical.get("blockers") or [],
        },
        "pollux_fullstack_same_workload": {
            "artifact": str(ARTIFACT_ROOT / "pollux_fullstack_same_workload_gate_20260613.json"),
            "ready": bool(pollux_fullstack.get("scoped_pollux_fullstack_same_workload_ready")),
            "scoped_superiority_ready": bool(
                pollux_fullstack.get("scoped_pollux_same_workload_superiority_ready")
            ),
            "same_service_scale_ready": bool(pollux_fullstack.get("same_service_scale_ready")),
            "cases": pollux_fullstack.get("cases") or [],
        },
        "sia_fullstack_same_workload": {
            "artifact": str(ARTIFACT_ROOT / "sia_fullstack_same_workload_gate_20260613.json"),
            "ready": bool(sia_fullstack.get("scoped_sia_fullstack_same_workload_ready")),
            "scoped_superiority_ready": bool(
                sia_fullstack.get("scoped_sia_same_workload_superiority_ready")
            ),
            "same_service_scale_ready": bool(sia_fullstack.get("same_service_scale_ready")),
            "cases": sia_fullstack.get("cases") or [],
        },
        "iadeep_fullstack_same_workload": {
            "artifact": str(ARTIFACT_ROOT / "iadeep_fullstack_same_workload_gate_20260613.json"),
            "ready": bool(iadeep_fullstack.get("scoped_iadeep_fullstack_same_workload_ready")),
            "scoped_superiority_ready": bool(
                iadeep_fullstack.get("scoped_iadeep_same_workload_superiority_ready")
            ),
            "same_service_scale_ready": bool(iadeep_fullstack.get("same_service_scale_ready")),
            "cases": iadeep_fullstack.get("cases") or [],
        },
        "salus_fullstack_same_workload": {
            "artifact": str(ARTIFACT_ROOT / "salus_fullstack_same_workload_gate_20260613.json"),
            "ready": bool(salus_fullstack.get("scoped_salus_fullstack_same_workload_ready")),
            "scoped_superiority_ready": bool(
                salus_fullstack.get("scoped_salus_same_workload_superiority_ready")
            ),
            "same_service_scale_ready": bool(salus_fullstack.get("same_service_scale_ready")),
            "blockers": salus_fullstack.get("blockers") or [],
        },
        "full_stack_ready_count": sum(1 for row in rows if row["same_workload_full_stack_ready"]),
        "named_sota_adapters": sorted(NAMED_SOTA_ADAPTERS),
        "named_same_host_runtime_probe_ready_count": named_ready_count,
        "named_same_host_runtime_probe_native_better_count": named_superiority_count,
        "named_same_host_runtime_probe_ready": bool(named_same_host_runtime_probe_ready),
        "named_same_host_runtime_probe_native_better": bool(named_same_host_runtime_probe_ready),
        "named_sota_fullstack_ready_count": named_ready_count,
        "named_sota_fullstack_superiority_count": 0,
        "direct_fullstack_named_sota_superiority_ready": False,
        "registered_sota_universe_superiority_ready": False,
        "arbitrary_sota_superiority_ready": False,
        "arbitrary_future_workload_superiority_ready": False,
        "multinode_original_deployment_superiority_ready": False,
        "production_wide_organic_trace_superiority_ready": False,
        "direct_fullstack_sota_superiority_ready": False,
        "policy_semantics_comparison_ready": bool(readiness.get("policy_semantics_fallback_available")),
        "hard_blocker_certificate_ready": True,
        "execution_environment": (
            "artifacted prior same-host runtime snapshots on the isolated "
            "jtl110gpu/K3s or same-host substrate; the current local WSL "
            "reproduction inventory may lack docker, kubectl, or go"
        ),
        "current_reproduction_environment": (
            "safe local reproduction script records current tool availability "
            "and does not re-run privileged external runtime probes"
        ),
        "current_rerun_ready": False,
        "runtime_probe_reproduction_contract": {
            "regeneration_mode": "precomputed_live_run",
            "current_safe_reproduction": False,
            "requires_external_stack": True,
            "meaning": (
                "The submission artifact records already executed same-host "
                "runtime-probe rows.  The safe reproduction path rebuilds the "
                "ledger from packaged rows and current tool inventory, but does "
                "not launch Kubernetes, Docker, Go, or Salus/Gavel/Pollux/Sia/"
                "IADeep runtime probes."
            ),
        },
        "pass": True,
        "scope": (
            "Scoped named-system same-host runtime-probe evidence for Gavel, "
            "Pollux/AdaptDL, Sia, IADeep, and Salus.  The gate does not claim "
            "direct full-stack SOTA superiority, arbitrary SOTA superiority, "
            "future-workload superiority, original multi-node deployment "
            "superiority, or production-wide online trace superiority."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Named External Runtime Probe Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `gate_pass` | {str(bool(report.get('gate_pass'))).lower()} |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `regeneration_mode` | `{report.get('regeneration_mode')}` |",
        f"| `current_safe_reproduction` | {str(bool(report.get('current_safe_reproduction'))).lower()} |",
        f"| `requires_external_stack` | {str(bool(report.get('requires_external_stack'))).lower()} |",
        f"| `current_environment_has_tools` | {str(bool(report.get('current_environment_has_tools'))).lower()} |",
        f"| `pass_meaning` | {report.get('pass_meaning')} |",
        f"| `named_same_host_runtime_probe_ready` | {str(bool(report.get('named_same_host_runtime_probe_ready'))).lower()} |",
        f"| `named_same_host_runtime_probe_native_better` | {str(bool(report.get('named_same_host_runtime_probe_native_better'))).lower()} |",
        f"| `direct_fullstack_sota_superiority_ready` | {str(bool(report.get('direct_fullstack_sota_superiority_ready'))).lower()} |",
        f"| `direct_fullstack_named_sota_superiority_ready` | {str(bool(report.get('direct_fullstack_named_sota_superiority_ready'))).lower()} |",
        f"| `registered_sota_universe_superiority_ready` | {str(bool(report.get('registered_sota_universe_superiority_ready'))).lower()} |",
        f"| `arbitrary_sota_superiority_ready` | {str(bool(report.get('arbitrary_sota_superiority_ready'))).lower()} |",
        f"| `multinode_original_deployment_superiority_ready` | {str(bool(report.get('multinode_original_deployment_superiority_ready'))).lower()} |",
        f"| `production_wide_organic_trace_superiority_ready` | {str(bool(report.get('production_wide_organic_trace_superiority_ready'))).lower()} |",
        f"| `full_stack_ready_count` | {report.get('full_stack_ready_count', 0)} |",
        f"| `named_same_host_runtime_probe_ready_count` | {report.get('named_same_host_runtime_probe_ready_count', 0)} |",
        f"| `named_same_host_runtime_probe_native_better_count` | {report.get('named_same_host_runtime_probe_native_better_count', 0)} |",
        f"| `policy_semantics_comparison_ready` | {str(bool(report.get('policy_semantics_comparison_ready'))).lower()} |",
        f"| `hard_blocker_certificate_ready` | {str(bool(report.get('hard_blocker_certificate_ready'))).lower()} |",
        f"| `native_execution_ready_count` | {(report.get('native_execution_attempts') or {}).get('ready_count', 0)} |",
        f"| `native_attempt_direct_full_stack_ready_count` | {(report.get('native_execution_attempts') or {}).get('direct_full_stack_ready_count', 0)} |",
        f"| `gavel_same_workload_native_simulator_ready` | {str(bool((report.get('gavel_same_workload_native_simulator_comparison') or {}).get('ready'))).lower()} |",
        f"| `gavel_physical_same_workload_fullstack_ready` | {str(bool((report.get('gavel_physical_same_workload') or {}).get('ready'))).lower()} |",
        f"| `gavel_physical_scoped_same_workload_superiority_ready` | {str(bool((report.get('gavel_physical_same_workload') or {}).get('scoped_superiority_ready'))).lower()} |",
        f"| `pollux_same_workload_fullstack_ready` | {str(bool((report.get('pollux_fullstack_same_workload') or {}).get('ready'))).lower()} |",
        f"| `pollux_scoped_same_workload_superiority_ready` | {str(bool((report.get('pollux_fullstack_same_workload') or {}).get('scoped_superiority_ready'))).lower()} |",
        f"| `sia_same_workload_fullstack_ready` | {str(bool((report.get('sia_fullstack_same_workload') or {}).get('ready'))).lower()} |",
        f"| `sia_scoped_same_workload_superiority_ready` | {str(bool((report.get('sia_fullstack_same_workload') or {}).get('scoped_superiority_ready'))).lower()} |",
        f"| `iadeep_same_workload_fullstack_ready` | {str(bool((report.get('iadeep_fullstack_same_workload') or {}).get('ready'))).lower()} |",
        f"| `iadeep_scoped_same_workload_superiority_ready` | {str(bool((report.get('iadeep_fullstack_same_workload') or {}).get('scoped_superiority_ready'))).lower()} |",
        f"| `salus_same_workload_fullstack_ready` | {str(bool((report.get('salus_fullstack_same_workload') or {}).get('ready'))).lower()} |",
        f"| `salus_scoped_same_workload_superiority_ready` | {str(bool((report.get('salus_fullstack_same_workload') or {}).get('scoped_superiority_ready'))).lower()} |",
        "",
        "## Tool Blockers",
        "",
        "| Tool | Available | Path / version |",
        "|---|---:|---|",
    ]
    for tool, info in sorted((report.get("tool_inventory") or {}).items()):
        lines.append(
            f"| `{tool}` | {str(bool((info or {}).get('available'))).lower()} | `{_md((info or {}).get('path') or (info or {}).get('version') or '')}` |"
        )
    lines.extend([
        "",
        "## System Rows",
        "",
        "| Adapter | Regeneration | Current safe rerun | Tools present | Smoke | Native microbaseline | Native simulator comparison | Gavel physical pair | Resident-delay JCT holdout | Pollux runtime pair | Sia runtime pair | IADeep runtime pair | Salus runtime pair | Runtime probe ready | Native-better probe | Reason |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in report.get("rows") or []:
        lines.append(
                "| `{adapter}` | `{regen}` | {safe} | {tools} | {smoke} | {micro} | {native_comp} | {gavel_physical_pair} | {resident} | {pollux_pair} | {sia_pair} | {iadeep_pair} | {salus_pair} | {ready} | {allowed} | {reason} |".format(
                adapter=row.get("adapter"),
                regen=row.get("regeneration_mode"),
                safe=str(bool(row.get("current_safe_reproduction"))).lower(),
                tools=str(bool(row.get("current_environment_has_tools"))).lower(),
                smoke=str(bool(row.get("entrypoint_smoke_pass"))).lower(),
                micro=str(bool(row.get("native_microbaseline_ready"))).lower(),
                native_comp=str(bool(row.get("gavel_same_workload_native_simulator_superiority_ready"))).lower(),
                gavel_physical_pair=str(bool(row.get("gavel_physical_same_workload_fullstack_ready"))).lower(),
                resident=str(bool(row.get("gavel_resident_delay_jct_holdout_ready"))).lower(),
                pollux_pair=str(bool(row.get("pollux_same_workload_fullstack_ready"))).lower(),
                sia_pair=str(bool(row.get("sia_same_workload_fullstack_ready"))).lower(),
                iadeep_pair=str(bool(row.get("iadeep_same_workload_fullstack_ready"))).lower(),
                salus_pair=str(bool(row.get("salus_same_workload_fullstack_ready"))).lower(),
                ready=str(bool(row.get("scoped_runtime_probe_ready"))).lower(),
                allowed=str(bool(row.get("scoped_runtime_probe_native_better"))).lower(),
                reason=_md(row.get("strict_reason") or ""),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _strict_reason(
    adapter: str,
    row: Mapping[str, Any],
    gavel_micro: Mapping[str, Any],
    gavel_resident_delay: Mapping[str, Any],
    gavel_native_comparison: Mapping[str, Any],
    gavel_physical: Mapping[str, Any],
    pollux_fullstack: Mapping[str, Any],
    sia_fullstack: Mapping[str, Any],
    iadeep_fullstack: Mapping[str, Any],
    salus_fullstack: Mapping[str, Any],
) -> str:
    if adapter == "pollux_adaptdl_scheduler" and pollux_fullstack.get("scoped_pollux_fullstack_same_workload_ready"):
        if pollux_fullstack.get("scoped_pollux_same_workload_superiority_ready"):
            return (
                "Pollux/AdaptDL completed the same CUDA binary through its Kubernetes "
                "controller/allocator/supervisor stack, and the native "
                "Scheduleurm-controlled Docker path has lower JCT on the scoped "
                "short and long probe cases; this is a Pollux-scoped full-stack "
                "row, not an all-SOTA superiority certificate"
            )
        return (
            "Pollux/AdaptDL completed the same CUDA binary through its full stack, "
            "but the paired native superiority condition is not certified"
        )
    if adapter == "sia_goodput_scheduler" and sia_fullstack.get("scoped_sia_fullstack_same_workload_ready"):
        if sia_fullstack.get("scoped_sia_same_workload_superiority_ready"):
            return (
                "Sia/AdaptDL MIP completed the same CUDA binary through its "
                "Kubernetes controller/allocator path, and the native "
                "Scheduleurm-controlled Docker path has lower JCT on the scoped "
                "short and long probe cases; this is a Sia-scoped full-stack "
                "row, not an all-SOTA superiority certificate"
            )
        return (
            "Sia/AdaptDL MIP completed the same CUDA binary through its full "
            "stack, but the paired native superiority condition is not certified"
        )
    if adapter == "iadeep_kubernetes_extender" and iadeep_fullstack.get("scoped_iadeep_fullstack_same_workload_ready"):
        if iadeep_fullstack.get("scoped_iadeep_same_workload_superiority_ready"):
            return (
                "IADeep completed the same CUDA binary through its Kubernetes "
                "scheduler-extender/device-plugin GPU-sharing path, and the "
                "native Scheduleurm-controlled Docker path has lower JCT on "
                "the scoped stable probe case; this is an IADeep-scoped "
                "full-stack row, not an all-SOTA superiority certificate"
            )
        return (
            "IADeep completed the same CUDA binary through its full stack, "
            "but the paired native superiority condition is not certified"
        )
    if adapter == "salus_gpu_sharing":
        if salus_fullstack.get("scoped_salus_fullstack_same_workload_ready"):
            if salus_fullstack.get("scoped_salus_same_workload_superiority_ready"):
                return (
                    "Salus completed the same TensorFlow-Salus workload through "
                    "its server/zrpc path, and the native TensorFlow-Salus path "
                    "on the same host has lower wall time in the scoped run; "
                    "this is a Salus-scoped full-stack row, not an arbitrary "
                    "future-workload claim"
                )
            return (
                "Salus completed a TensorFlow-Salus client workload through its "
                "server path, but scoped superiority is not certified"
            )
        blockers = salus_fullstack.get("blockers") or []
        if blockers:
            return "; ".join(str(x) for x in blockers)
    if adapter == "gavel_simulation":
        if gavel_physical.get("scoped_gavel_physical_fullstack_ready"):
            if gavel_physical.get("scoped_gavel_physical_same_workload_superiority_ready"):
                return (
                    "Gavel completed the same temporary CUDA probe through its "
                    "physical scheduler/worker/RPC/dispatcher/GavelIterator "
                    "path, and the same script run directly on the same host "
                    "has lower JCT in the scoped run; this is a Gavel-scoped "
                    "physical full-stack row, not an all-workload Gavel claim"
                )
            return (
                "Gavel completed the same temporary CUDA probe through its "
                "physical scheduler/worker path, but paired native superiority "
                "is not certified"
            )
        if not gavel_micro.get("native_gavel_simulator_microbaseline_ready"):
            blockers = gavel_physical.get("blockers") or []
            if blockers:
                return "; ".join(str(x) for x in blockers)
            return "Gavel physical same-workload row is absent"
        if gavel_native_comparison.get("same_workload_native_simulator_superiority_ready"):
            return (
                "same-workload native Gavel simulator comparison is favorable "
                "on the eligible q01 ResNet-50 full-taskset row, but this is "
                "still simulator evidence; service-unit equivalence and direct "
                "full-stack production execution are not certified"
            )
        if gavel_resident_delay.get("gavel_style_resident_delay_jct_holdout_ready"):
            return (
                "native simulator microbaseline and measured resident-delay/JCT "
                "holdout exist, but scalar service-unit equivalence and direct "
                "full-stack production execution are not certified"
            )
        return "native simulator microbaseline exists, but service-unit equivalence and full-stack production execution are not certified"
    blockers = list(row.get("blockers") or [])
    if blockers:
        return "; ".join(str(x) for x in blockers)
    return "same-workload full-stack performance row is absent"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _latest_artifact(pattern: str, fallback: Path) -> Path:
    matches = sorted(ARTIFACT_ROOT.glob(pattern), key=lambda path: path.name)
    return matches[-1] if matches else fallback


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_fullstack_superiority_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_fullstack_superiority_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build named external runtime-probe gate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "sota_fullstack_superiority_gate_20260613.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "sota_fullstack_superiority_gate_20260613.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
