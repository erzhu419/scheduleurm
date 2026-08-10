"""Execute and serialize the committed port statewise trajectory v3 protocol."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from algorithm.experiments.port_statewise_trajectory_v3 import (
    build_port_statewise_trajectory_v3,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = REPO_ROOT / "algorithm" / "experiments" / "port_statewise_trajectory_v3.py"
IMPLEMENTATION_SHA256 = "3805825e5a81cb6b81bde2329db79fb792bf00b130db5673dcdb39ae65ecd09f"
DEFAULT_JSON = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_statewise_trajectory_v3_20260810.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "port_statewise_trajectory_v3_20260810.md"


class PortStatewiseArtifactError(ValueError):
    """Raised when the committed protocol or output contract is violated."""


def build_artifact() -> dict[str, Any]:
    implementation_commit = _committed_file_contract(
        IMPLEMENTATION, IMPLEMENTATION_SHA256
    )
    wrapper_path = Path(__file__).resolve()
    wrapper_hash = _file_sha256(wrapper_path)
    wrapper_commit = _committed_file_contract(wrapper_path, wrapper_hash)
    report = build_port_statewise_trajectory_v3()
    artifact = {
        "schema_version": "scheduleurm.port.statewise_trajectory.v3.artifact.v1",
        "execution_contract": {
            "implementation_path": IMPLEMENTATION.relative_to(REPO_ROOT).as_posix(),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "implementation_commit": implementation_commit,
            "wrapper_path": wrapper_path.relative_to(REPO_ROOT).as_posix(),
            "wrapper_sha256": wrapper_hash,
            "wrapper_commit": wrapper_commit,
            "results_observed_before_protocol_commit": False,
        },
        "report": report,
    }
    artifact["artifact_sha256_excluding_self"] = _digest(artifact)
    return artifact


def write_outputs(
    artifact: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> dict[str, str]:
    output = Path(json_path)
    markdown = Path(markdown_path)
    for path in (output, markdown):
        if path.exists():
            raise PortStatewiseArtifactError(f"refusing to overwrite result: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(markdown_report(artifact), encoding="utf-8")
    return {
        "json": str(output),
        "json_sha256": _file_sha256(output),
        "markdown": str(markdown),
        "markdown_sha256": _file_sha256(markdown),
    }


def markdown_report(artifact: Mapping[str, Any]) -> str:
    report = artifact["report"]
    source = report["certificates"]["bacasp_source_core"]
    migration = report["certificates"]["synthetic_migration"]
    selection = migration["selection"]
    lines = [
        "# Port statewise trajectory v3",
        "",
        f"- Status: `{report['status']}`",
        f"- Artifact SHA-256: `{artifact['artifact_sha256_excluding_self']}`",
        f"- BACASP source-core ready: `{str(source['ready']).lower()}`",
        f"- Synthetic statewise migration ready: `{str(migration['ready']).lower()}`",
        f"- Exact duration-normalized oracle gap: `{selection['oracle_gap_alpha0']}`",
        f"- Finite-family penalty bound P0: `{selection['finite_family_penalty_bound_P0']}`",
        f"- Selected action: `{selection['selected_candidate']['candidate_id']}`",
        "- Terminal-cost Pareto status is diagnostic and never overrides the robust oracle.",
        "- BACASP source-core evidence and synthetic migration evidence are non-substitutable.",
        "",
        "## Claim Boundary",
        "",
        "Supported:",
    ]
    lines.extend(f"- {claim}" for claim in report["claim_boundary"]["supports"])
    lines.extend(["", "Not supported:"])
    lines.extend(f"- {claim}" for claim in report["claim_boundary"]["does_not_support"])
    lines.append("")
    return "\n".join(lines)


def _committed_file_contract(path: Path, expected_sha256: str) -> str:
    source = path.resolve()
    relative = source.relative_to(REPO_ROOT)
    if _file_sha256(source) != expected_sha256:
        raise PortStatewiseArtifactError(f"disk hash mismatch: {relative}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise PortStatewiseArtifactError(f"file is not committed: {relative}")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if sha256(committed).hexdigest() != expected_sha256:
        raise PortStatewiseArtifactError(f"committed hash mismatch: {relative}")
    return commit


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    artifact = build_artifact()
    outputs = write_outputs(
        artifact,
        json_path=args.json,
        markdown_path=args.markdown,
    )
    report = artifact["report"]
    print(json.dumps({"status": report["status"], "outputs": outputs}, indent=2))
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
