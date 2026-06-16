"""Salus source-build feasibility gate.

This gate records whether the current artifact environment can replace the
official Salus Docker image with a clean source build.  It is intentionally
strict: a source build is considered a valid path only if the TensorFlow-Salus
build tree and Salus' pinned native dependencies are available.  Building Salus
without TensorFlow-Salus would not exercise the Salus full-stack runtime used by
its TensorFlow clients.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SALUS_REPO = REPO_ROOT / "reference" / "repos" / "salus"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "salus_source_build_feasibility_gate_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "salus_source_build_feasibility_gate_20260613.md"


def build_salus_source_build_feasibility_gate(
    *,
    salus_repo: str | Path = SALUS_REPO,
) -> dict[str, Any]:
    salus_repo = Path(salus_repo).expanduser()
    cmake_lists = _read_text(salus_repo / "CMakeLists.txt")
    dockerfile = _read_text(salus_repo / "Dockerfile")
    find_tf = _read_text(salus_repo / "cmake" / "FindTensorFlow.cmake")
    readme = _read_text(salus_repo / "README.md")
    tools = {tool: _tool_info(tool) for tool in ("cmake", "gcc", "g++", "protoc", "pkg-config", "docker", "kubectl", "nvidia-smi")}
    tensorflow_dir = os.environ.get("TensorFlow_DIR") or ""
    tensorflow_tree = _tensorflow_tree_status(Path(tensorflow_dir).expanduser()) if tensorflow_dir else {
        "path": "",
        "exists": False,
        "has_bazel_tensorflow": False,
        "has_libtensorflow_framework": False,
        "has_libtensorflow_kernels": False,
    }
    local_tf_salus_repo = REPO_ROOT / "reference" / "repos" / "tensorflow-salus"
    required_contracts = {
        "requires_custom_tensorflow_salus": "tensorflow-salus" in readme and "tensorflow-salus" in dockerfile,
        "requires_tensorflow_bazel_tree": "bazel-tensorflow/tensorflow/core" in find_tf,
        "requires_libtensorflow_kernels": "TensorFlow_Kernel_LIBRARY" in find_tf,
        "requires_protobuf_3_4_exact": "find_package(Protobuf 3.4.0 EXACT REQUIRED)" in cmake_lists,
        "requires_boost_1_66_exact": "find_package(Boost 1.66 EXACT REQUIRED" in cmake_lists,
        "requires_zeromq": "find_package(ZeroMQ REQUIRED)" in cmake_lists,
        "requires_gperftools_when_tcmalloc": "find_package(Gperftools REQUIRED)" in cmake_lists,
        "official_prod_image_pinned_cuda91": "nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04" in dockerfile,
    }
    source_build_ready = bool(
        salus_repo.exists()
        and tools["cmake"]["available"]
        and tools["gcc"]["available"]
        and tools["g++"]["available"]
        and tools["protoc"]["available"]
        and tensorflow_tree["has_bazel_tensorflow"]
        and tensorflow_tree["has_libtensorflow_framework"]
        and tensorflow_tree["has_libtensorflow_kernels"]
    )
    blockers = _blockers(
        tools=tools,
        tensorflow_tree=tensorflow_tree,
        local_tf_salus_repo=local_tf_salus_repo,
        required_contracts=required_contracts,
        source_build_ready=source_build_ready,
    )
    return {
        "gate": "salus_source_build_feasibility_gate",
        "salus_repo": str(salus_repo),
        "source_build_ready": bool(source_build_ready),
        "valid_fullstack_substitute_ready": bool(source_build_ready),
        "direct_fullstack_salus_superiority_ready": False,
        "tools": tools,
        "tensorflow_dir": tensorflow_dir,
        "tensorflow_tree": tensorflow_tree,
        "local_tensorflow_salus_repo": {
            "path": str(local_tf_salus_repo),
            "exists": local_tf_salus_repo.exists(),
        },
        "required_contracts": required_contracts,
        "blockers": blockers,
        "pass": True,
        "scope": (
            "Strict feasibility certificate for replacing the official Salus "
            "image with a local source build.  A source build is not a valid "
            "full-stack Salus baseline unless it includes TensorFlow-Salus and "
            "the pinned native dependency contract required by upstream Salus."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Salus Source-Build Feasibility Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `source_build_ready` | {str(bool(report.get('source_build_ready'))).lower()} |",
        f"| `valid_fullstack_substitute_ready` | {str(bool(report.get('valid_fullstack_substitute_ready'))).lower()} |",
        f"| `direct_fullstack_salus_superiority_ready` | {str(bool(report.get('direct_fullstack_salus_superiority_ready'))).lower()} |",
        "",
        "## Tools",
        "",
        "| Tool | Available | Version | Path |",
        "|---|---:|---|---|",
    ]
    for tool, info in sorted((report.get("tools") or {}).items()):
        lines.append(
            f"| `{tool}` | {str(bool(info.get('available'))).lower()} | `{_md(info.get('version'))}` | `{_md(info.get('path'))}` |"
        )
    lines.extend([
        "",
        "## TensorFlow-Salus Contract",
        "",
        "| Field | Value |",
        "|---|---|",
    ])
    tf_tree = report.get("tensorflow_tree") or {}
    lines.extend([
        f"| `TensorFlow_DIR` | `{_md(report.get('tensorflow_dir'))}` |",
        f"| `has_bazel_tensorflow` | `{str(bool(tf_tree.get('has_bazel_tensorflow'))).lower()}` |",
        f"| `has_libtensorflow_framework` | `{str(bool(tf_tree.get('has_libtensorflow_framework'))).lower()}` |",
        f"| `has_libtensorflow_kernels` | `{str(bool(tf_tree.get('has_libtensorflow_kernels'))).lower()}` |",
        f"| `local_tensorflow_salus_repo.exists` | `{str(bool((report.get('local_tensorflow_salus_repo') or {}).get('exists'))).lower()}` |",
        "",
        "## Blockers",
        "",
        "| Blocker |",
        "|---|",
    ])
    blockers = report.get("blockers") or []
    if not blockers:
        lines.append("| none |")
    for blocker in blockers:
        lines.append(f"| {_md(blocker)} |")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _tool_info(tool: str) -> dict[str, Any]:
    path = shutil.which(tool)
    if not path:
        return {"available": False, "path": "", "version": ""}
    try:
        proc = subprocess.run(
            [tool, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        version = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
    except Exception as exc:
        version = f"{type(exc).__name__}: {exc}"
    return {"available": True, "path": path, "version": version}


def _tensorflow_tree_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "has_bazel_tensorflow": (path / "bazel-tensorflow" / "tensorflow" / "core").exists(),
        "has_libtensorflow_framework": any(path.glob("bazel-bin/tensorflow/libtensorflow_framework.so*")) if path.exists() else False,
        "has_libtensorflow_kernels": any(path.glob("bazel-bin/tensorflow/libtensorflow_kernels.so*")) if path.exists() else False,
    }


def _blockers(
    *,
    tools: Mapping[str, Mapping[str, Any]],
    tensorflow_tree: Mapping[str, Any],
    local_tf_salus_repo: Path,
    required_contracts: Mapping[str, bool],
    source_build_ready: bool,
) -> list[str]:
    if source_build_ready:
        return []
    blockers: list[str] = []
    for tool in ("protoc", "docker", "kubectl"):
        if not tools.get(tool, {}).get("available"):
            blockers.append(f"{tool} is not available in the current local environment")
    if required_contracts.get("requires_custom_tensorflow_salus") and not local_tf_salus_repo.exists():
        blockers.append("TensorFlow-Salus source repository is not present under reference/repos/tensorflow-salus")
    if required_contracts.get("requires_tensorflow_bazel_tree") and not tensorflow_tree.get("has_bazel_tensorflow"):
        blockers.append("TensorFlow_DIR does not point to a TensorFlow-Salus bazel source tree")
    if required_contracts.get("requires_libtensorflow_kernels") and not tensorflow_tree.get("has_libtensorflow_kernels"):
        blockers.append("TensorFlow-Salus libtensorflow_kernels.so artifact is absent")
    if required_contracts.get("requires_protobuf_3_4_exact"):
        blockers.append("Salus requires Protobuf 3.4.0 exactly; the current environment has no verified matching protoc")
    if required_contracts.get("requires_boost_1_66_exact"):
        blockers.append("Salus requires Boost 1.66 exactly; no verified matching local Boost tree is present")
    if required_contracts.get("official_prod_image_pinned_cuda91"):
        blockers.append("upstream production image is pinned to CUDA 9.1/cuDNN7, so source-build substitution still needs old-runtime compatibility evidence")
    return blockers


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1200]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_salus_source_build_feasibility_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.salus_source_build_feasibility_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Salus source-build feasibility gate")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
