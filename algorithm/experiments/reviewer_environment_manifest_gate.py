"""Reviewer environment manifest gate.

This is the clean-room reproducibility contract for the OR artifact.  It does
not claim a Docker/Nix container exists; it checks the concrete commands used by
the one-command reproduction script and points reviewers to the pinned
environment files supplied with the repository.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


TOOLS = ("python3", "pytest", "lake", "pdflatex", "bibtex", "make", "rg")
MAKE_TARGETS = (
    "verify-gates",
    "verify-lean",
    "paper-tables",
    "artifact-manifest",
    "reproduce-or-submission",
)


def build_reviewer_environment_manifest_gate() -> dict[str, Any]:
    tools = {tool: _tool_row(tool) for tool in TOOLS}
    required_ready = all(tools[tool]["available"] for tool in ("python3", "pytest", "pdflatex", "bibtex", "make", "rg"))
    lean_ready = bool(tools["lake"]["available"]) and (REPO_ROOT.parent / "proof" / "lakefile.toml").exists()
    makefile = REPO_ROOT / "Makefile"
    reproduction_script = REPO_ROOT / "scripts" / "reproduce_or_submission.sh"
    environment_file = REPO_ROOT / "environment.yml"
    requirements_file = REPO_ROOT / "requirements-reviewer.txt"
    return {
        "gate": "reviewer_environment_manifest_gate",
        "status": "REVIEWER_ENVIRONMENT_MANIFEST_READY_CONTAINER_FALSE",
        "gate_pass": required_ready and reproduction_script.exists() and makefile.exists(),
        "scoped_claim_ready": required_ready and reproduction_script.exists() and makefile.exists(),
        "strong_claim_ready": False,
        "pass_meaning": (
            "current-host command manifest and reviewer environment contract, not a "
            "clean Docker/Nix container proof"
        ),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "tools": tools,
        "lean_build_ready": lean_ready,
        "makefile_exists": makefile.exists(),
        "reproduction_script_exists": reproduction_script.exists(),
        "reproduction_script_executable": reproduction_script.exists() and bool(reproduction_script.stat().st_mode & 0o111),
        "environment_yml_exists": environment_file.exists(),
        "requirements_reviewer_exists": requirements_file.exists(),
        "make_targets": list(MAKE_TARGETS),
        "make_targets_ready": makefile.exists() and all(_make_target_present(makefile, target) for target in MAKE_TARGETS),
        "clean_container_ready": False,
        "blocker": (
            "A one-command host reproduction path exists, but no Docker/Nix clean-room "
            "container is supplied.  Reviewers should use environment.yml or "
            "requirements-reviewer.txt plus the Lean toolchain in ../proof."
        ),
        "next_threshold": "Add Dockerfile.reviewer or a Nix flake and run the full reproduction script inside it.",
        "pass": required_ready and reproduction_script.exists() and makefile.exists(),
        "scope": (
            "Audits the current host and supplied environment contract for the OR "
            "artifact.  It is a reproducibility manifest, not a containerized "
            "clean-room execution certificate."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Reviewer Environment Manifest Gate",
        "",
        "This gate certifies the current-host reproduction contract. It does not claim a clean Docker/Nix container.",
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
        f"| `lean_build_ready` | {str(bool(report.get('lean_build_ready'))).lower()} |",
        f"| `make_targets_ready` | {str(bool(report.get('make_targets_ready'))).lower()} |",
        f"| `environment_yml_exists` | {str(bool(report.get('environment_yml_exists'))).lower()} |",
        f"| `requirements_reviewer_exists` | {str(bool(report.get('requirements_reviewer_exists'))).lower()} |",
        f"| `clean_container_ready` | {str(bool(report.get('clean_container_ready'))).lower()} |",
        "",
        "## Tools",
        "",
        "| Tool | Available | Path | Version |",
        "|---|---:|---|---|",
    ]
    for tool, row in sorted((report.get("tools") or {}).items()):
        lines.append(
            "| `{tool}` | {available} | `{path}` | `{version}` |".format(
                tool=tool,
                available=str(bool(row.get("available"))).lower(),
                path=_md(row.get("path") or ""),
                version=_md(row.get("version") or ""),
            )
        )
    lines.extend([
        "",
        "## Blocker",
        "",
        str(report.get("blocker") or "none"),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _tool_row(tool: str) -> dict[str, Any]:
    path = shutil.which(tool)
    row = {"available": bool(path), "path": path or "", "version": ""}
    if not path:
        return row
    cmd = [path, "--version"]
    if tool == "bibtex":
        cmd = [path, "--version"]
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=8,
            check=False,
        )
        row["version"] = (proc.stdout or "").splitlines()[0][:240] if proc.stdout else ""
    except Exception as exc:
        row["version"] = str(exc)
    return row


def _make_target_present(makefile: Path, target: str) -> bool:
    if not makefile.exists():
        return False
    needle = f"{target}:"
    return any(line.startswith(needle) for line in makefile.read_text(encoding="utf-8").splitlines())


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_reviewer_environment_manifest_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.reviewer_environment_manifest_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build reviewer environment manifest gate")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "reviewer_environment_manifest_gate_20260612.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "reviewer_environment_manifest_gate_20260612.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
