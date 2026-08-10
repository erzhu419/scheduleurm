"""Audit the exact boundary between port trajectories and the frame theorem.

The registered port trajectory objective uses bounded virtual terminal-cost
service, not physical vessel throughput.  This certificate therefore closes
the finite fixed-horizon interface while deliberately leaving stochastic
physical-port recurrence open.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_ROOT = REPO_ROOT.parent / "proof"
LEAN_SOURCE = PROOF_ROOT / "Scheduleurm" / "FrameBasedStability.lean"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_FULL = ARTIFACT_ROOT / "port_trajectory_registered_refresh_v1_20260810.json.gz"
DEFAULT_REFRESH = ARTIFACT_ROOT / "port_trajectory_registered_refresh_v1_20260810.json"
DEFAULT_JSON = ARTIFACT_ROOT / "port_frame_bridge_certificate_v1_20260810.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "port_frame_bridge_certificate_v1_20260810.md"
REQUIRED_THEOREMS = (
    "secondOrderTerm_frame_le",
    "frameApproximateOracle_iff_durationNormalized",
    "frame_approximate_maxWeight_lyapunov_drift",
    "frame_approximate_maxWeight_lyapunov_drift_uniform",
    "frame_lyapunov_drift_duration_one",
    "elapsedFrameTime_le",
    "frame_nat_model_positive_recurrent_via_finite_set",
)


class PortFrameBridgeError(ValueError):
    """Raised when a deterministic frame-interface obligation fails."""


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _load_json(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    raw = gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise PortFrameBridgeError(f"expected JSON object: {source}")
    return payload, raw


def inspect_lean_source(path: str | Path = LEAN_SOURCE) -> dict[str, Any]:
    source = Path(path)
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    missing = [name for name in REQUIRED_THEOREMS if not re.search(rf"\b{name}\b", text)]
    forbidden = sorted(set(re.findall(r"\b(?:sorry|admit|axiom)\b", text)))
    return {
        "relative_path": "Scheduleurm/FrameBasedStability.lean",
        "sha256": _sha256(raw),
        "required_theorems": list(REQUIRED_THEOREMS),
        "missing_theorems": missing,
        "forbidden_tokens": forbidden,
        "source_contract_ready": not missing and not forbidden,
    }


def run_lean_check(proof_root: str | Path = PROOF_ROOT) -> dict[str, Any]:
    root = Path(proof_root)
    version = subprocess.run(
        ["lake", "env", "lean", "--version"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    compile_result = subprocess.run(
        ["lake", "env", "lean", "Scheduleurm/FrameBasedStability.lean"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    return {
        "command": "lake env lean Scheduleurm/FrameBasedStability.lean",
        "lean_version_command_exit_code": version.returncode,
        "lean_version": version.stdout.strip() or version.stderr.strip(),
        "compile_exit_code": compile_result.returncode,
        "stdout": compile_result.stdout,
        "stderr": compile_result.stderr,
        "compile_ready": compile_result.returncode == 0,
    }


def _frame_rows(full: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = full.get("candidate_independent_frame_specs") or {}
    family = full.get("trajectory_family") or {}
    if not isinstance(frames, Mapping) or not isinstance(family, Mapping):
        raise PortFrameBridgeError("trajectory frame/family schema is invalid")
    if len(frames) != 4 or len(family) != 31:
        raise PortFrameBridgeError("registered frame/family cardinality drifted")
    rows: list[dict[str, Any]] = []
    for context, frame in sorted(frames.items()):
        horizon = float(frame["horizon"])
        metrics = tuple(str(row) for row in frame["cost_metrics"])
        duration_metric = metrics[0]
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise PortFrameBridgeError(f"{context}: frame horizon is not positive")
        if (frame.get("derivation") or {}).get("candidate_independent") is not True:
            raise PortFrameBridgeError(f"{context}: frame depends on candidate outcome")
        durations = []
        for candidate_id, candidate in sorted(family.items()):
            objectives = candidate.get("context_objectives") or {}
            if context not in objectives:
                raise PortFrameBridgeError(f"{candidate_id}: missing context {context}")
            costs = objectives[context].get("terminal_costs") or {}
            duration = float(costs[duration_metric])
            if not math.isfinite(duration) or duration <= 0.0 or duration > horizon + 1e-8:
                raise PortFrameBridgeError(
                    f"{candidate_id}/{context}: duration lies outside (0,H]"
                )
            durations.append(duration)
        rows.append(
            {
                "context": context,
                "candidate_count": len(durations),
                "duration_metric": duration_metric,
                "observed_duration_min": min(durations),
                "observed_duration_max": max(durations),
                "common_frame_horizon_H": horizon,
                "maximum_idle_padding": horizon - min(durations),
                "all_trajectories_fit_common_horizon": True,
                "positive_common_denominator": True,
                "fixed_H_division_preserves_scalar_order": True,
            }
        )
    return rows


def build_certificate(
    *,
    full_path: str | Path = DEFAULT_FULL,
    refresh_path: str | Path = DEFAULT_REFRESH,
    lean_source: str | Path = LEAN_SOURCE,
    lean_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    full, full_raw = _load_json(full_path)
    refresh, refresh_raw = _load_json(refresh_path)
    if full.get("schema_version") != "scheduleurm.port_trajectory_upgrade.v2":
        raise PortFrameBridgeError("unexpected full trajectory schema")
    if refresh.get("schema_version") != (
        "scheduleurm.port_trajectory_registered_refresh.compact.v1"
    ):
        raise PortFrameBridgeError("unexpected refresh schema")
    if (refresh.get("gate") or {}).get("pass") is not True:
        raise PortFrameBridgeError("port refresh gate is open")
    ledger = refresh.get("ledger") or {}
    if full.get("artifact_hash") != ledger.get("full_artifact_hash"):
        raise PortFrameBridgeError("full and compact semantic hashes differ")
    if refresh.get("full_decompressed_sha256") != _sha256(full_raw):
        raise PortFrameBridgeError("decompressed full ledger hash differs")
    if full.get("trajectory_oracle_gap_alpha0") != 0.0:
        raise PortFrameBridgeError("generated trajectory oracle is not exact")
    if full.get("all_candidate_results_feasible") is not True:
        raise PortFrameBridgeError("infeasible registered trajectory")
    if full.get("terminal_coordinates_are_physical_service") is not False:
        raise PortFrameBridgeError("terminal virtual service boundary drifted")
    penalty = full.get("bounded_penalty_certificate") or {}
    if penalty.get("ready") is not True or penalty.get("queue_independent") is not True:
        raise PortFrameBridgeError("finite-family penalty bound is open")

    frames = _frame_rows(full)
    lean_source_report = inspect_lean_source(lean_source)
    check = dict(lean_check or run_lean_check(Path(lean_source).parents[1]))
    lean_ready = bool(
        lean_source_report["source_contract_ready"]
        and check.get("compile_ready") is True
        and check.get("compile_exit_code") == 0
    )
    if not lean_ready:
        raise PortFrameBridgeError("Lean frame theorem compilation is open")
    deterministic_interface_ready = bool(
        all(row["all_trajectories_fit_common_horizon"] for row in frames)
        and full.get("trajectory_family_size") == 31
        and full.get("trajectory_oracle_gap_alpha0") == 0.0
        and penalty.get("ready") is True
    )
    report: dict[str, Any] = {
        "schema_version": "scheduleurm.port_frame_bridge_certificate.v1",
        "gate": {
            "pass": deterministic_interface_ready and lean_ready,
            "status": "PORT_DETERMINISTIC_VIRTUAL_FRAME_BRIDGE_PASS",
            "deterministic_virtual_frame_interface_ready": deterministic_interface_ready,
            "lean_variable_duration_frame_theorem_ready": lean_ready,
            "physical_service_frame_interface_ready": False,
            "stochastic_port_arrival_service_model_ready": False,
            "port_slack_certificate_ready": False,
            "local_return_certificate_ready": False,
            "physical_port_positive_recurrence_claim_ready": False,
        },
        "inputs": {
            "full_gzip_sha256": _sha256(Path(full_path).read_bytes()),
            "full_decompressed_sha256": _sha256(full_raw),
            "refresh_compact_sha256": _sha256(refresh_raw),
            "full_semantic_artifact_hash": full["artifact_hash"],
        },
        "frame_rows": frames,
        "finite_family": {
            "candidate_count": full["trajectory_family_size"],
            "context_count": len(frames),
            "candidate_context_pair_count": full[
                "evaluated_trajectory_instance_pairs"
            ],
            "exact_oracle_gap_alpha0": full["trajectory_oracle_gap_alpha0"],
            "oracle_slope_alpha1": full["trajectory_oracle_gap_alpha1"],
            "finite_family_penalty_bound_P0": penalty["finite_family_P0_bound"],
            "queue_scaled_penalty_slope_beta": penalty[
                "queue_scaled_penalty_slope_beta"
            ],
        },
        "lean": {
            "source": lean_source_report,
            "build": check,
        },
        "mathematical_mapping": {
            "embedding": (
                "for each registered context, idle-pad every completed trajectory to its "
                "candidate-independent horizon H; all candidates then have the same positive frame duration"
            ),
            "ordering": (
                "division of a context's scalar virtual-service-minus-penalty score by the common H "
                "preserves its finite-family ordering"
            ),
            "lean_connection": (
                "FrameBasedStability proves the duration-normalized oracle equivalence, frame drift, "
                "uniform minimum-duration drift, elapsed-time bound, and conditional finite-set recurrence"
            ),
            "terminal_coordinate_semantics": (
                "bounded virtual terminal-cost service nu_j=(U_j-C_j)/U_j; not physical vessel departures"
            ),
        },
        "claim_boundary": {
            "supports": [
                "a deterministic fixed-frame embedding of every registered port trajectory",
                "positive finite frame horizons and exact finite-family virtual-objective selection",
                "a compiled Lean theorem interface for variable-duration drift under separately supplied assumptions",
            ],
            "does_not_support": [
                "identifying terminal virtual-cost coordinates with physical port service",
                "a calibrated stochastic vessel-arrival and service model",
                "positive slack, local return, or physical-port positive recurrence",
                "inferring a Markov-chain theorem from finite deterministic replay",
            ],
        },
    }
    report["artifact_sha256_excluding_self"] = _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    finite = report["finite_family"]
    lines = [
        "# Port deterministic frame bridge certificate v1",
        "",
        f"- Status: `{gate['status']}`",
        f"- Lean compile exit code: `{report['lean']['build']['compile_exit_code']}`.",
        f"- Finite family: `{finite['candidate_count']}` candidates over `{finite['context_count']}` contexts.",
        f"- Exact generated-family oracle gap: `{finite['exact_oracle_gap_alpha0']}`.",
        f"- Finite-family penalty bound: `P0={finite['finite_family_penalty_bound_P0']}`; `beta={finite['queue_scaled_penalty_slope_beta']}`.",
        "",
        "| Context | Candidates | Observed duration range | Common H | Fits H |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["frame_rows"]:
        lines.append(
            f"| `{row['context']}` | {row['candidate_count']} | "
            f"[{row['observed_duration_min']:.6f}, {row['observed_duration_max']:.6f}] | "
            f"{row['common_frame_horizon_H']:.6f} | true |"
        )
    lines.extend(
        [
            "",
            "## Closed interface",
            "",
            "Every registered trajectory fits a candidate-independent positive horizon and can be idle-padded to that horizon. Dividing the registered scalar virtual objective by the common horizon preserves ordering, and the referenced variable-duration Lean file compiles without sorry/admit/axiom.",
            "",
            "## Open operational assumptions",
            "",
            "The terminal coordinates are bounded virtual cost service, not physical departures. A physical-port recurrence claim still requires a stochastic vessel-arrival/service model, positive slack, and local return. This certificate does not infer those assumptions from finite replay.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    *,
    json_output: str | Path = DEFAULT_JSON,
    markdown_output: str | Path = DEFAULT_MARKDOWN,
    force: bool = False,
) -> dict[str, Any]:
    outputs = [Path(json_output), Path(markdown_output)]
    if not force:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise PortFrameBridgeError(f"refusing to overwrite artifacts: {existing}")
    report = build_certificate()
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].write_text(markdown_report(report), encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = write_artifacts(
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        force=args.force,
    )
    print(json.dumps(report["gate"], sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PortFrameBridgeError",
    "build_certificate",
    "inspect_lean_source",
    "markdown_report",
    "run_lean_check",
    "write_artifacts",
]
