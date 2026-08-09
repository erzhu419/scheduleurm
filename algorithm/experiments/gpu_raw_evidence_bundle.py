"""Build a deterministic archive for theorem-facing GPU measurement evidence.

The stochastic gates retain hashes and summaries, but several task-native
progress logs live below the scheduler run directory.  This module follows
only explicit local JSON/path references from caller-supplied root artifacts,
copies the referenced repository artifacts and scheduler-run files into one
deterministic gzip-compressed tar archive, and emits a hash manifest.

It never launches work and refuses missing, changing, or out-of-scope local
evidence.  Remote ``/tmp`` paths are metadata, not archive inputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
SCHEDULER_RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "critical_gpu_raw_evidence_bundle_20260809.tar.gz"
DEFAULT_MANIFEST = ARTIFACT_ROOT / "critical_gpu_raw_evidence_bundle_20260809.json"
DEFAULT_MARKDOWN = DEFAULT_MANIFEST.with_suffix(".md")
SCHEMA_VERSION = 1


class EvidenceBundleError(ValueError):
    """Raised when a root artifact cannot be closed over local evidence."""


def build_gpu_raw_evidence_bundle(
    *,
    root_artifacts: Sequence[Path],
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    markdown_path: Path = DEFAULT_MARKDOWN,
    repo_root: Path = REPO_ROOT,
    scheduler_run_root: Path = SCHEDULER_RUN_ROOT,
) -> dict[str, Any]:
    """Validate and atomically write one deterministic evidence bundle."""

    roots = tuple(Path(path).resolve() for path in root_artifacts)
    if not roots:
        raise EvidenceBundleError("at least one root artifact is required")
    repo = Path(repo_root).resolve()
    run_root = Path(scheduler_run_root).resolve()
    outputs = {
        Path(output_path).resolve(),
        Path(manifest_path).resolve(),
        Path(markdown_path).resolve(),
    }
    entries, root_rows = _collect_evidence(
        roots=roots,
        repo_root=repo,
        scheduler_run_root=run_root,
        excluded_paths=outputs,
    )
    core = {
        "gate": "gpu_raw_evidence_bundle",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "pass": True,
        "root_artifact_count": len(root_rows),
        "root_artifacts": root_rows,
        "entry_count": len(entries),
        "entries": [entry.snapshot() for entry in entries],
        "ordinary_running_tasks_touched": False,
        "launches_work": False,
        "claim_boundary": (
            "The archive contains only local files explicitly reachable from "
            "the supplied theorem-facing GPU artifacts. It preserves raw "
            "task-native progress and sidecar evidence for those finite "
            "measurements; it is not evidence for unmeasured hardware, load "
            "states, workloads, or production tasks."
        ),
    }
    core_bytes = _canonical_json(core)
    bundle_bytes = _deterministic_tar_gz(entries=entries, manifest_bytes=core_bytes)
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    manifest = {
        **core,
        "bundle_path": str(Path(output_path).resolve()),
        "bundle_size_bytes": len(bundle_bytes),
        "bundle_sha256": bundle_sha256,
        "embedded_manifest_sha256": hashlib.sha256(core_bytes).hexdigest(),
    }
    _atomic_write_bytes(Path(output_path), bundle_bytes)
    _atomic_write_bytes(Path(manifest_path), _canonical_json(manifest))
    _atomic_write_bytes(
        Path(markdown_path),
        _markdown(manifest, manifest_path=Path(manifest_path)).encode("utf-8"),
    )
    return manifest


class _Entry:
    __slots__ = ("source", "archive_path", "kind", "size", "sha256", "bytes")

    def __init__(self, *, source: Path, archive_path: str, kind: str, data: bytes):
        self.source = source
        self.archive_path = archive_path
        self.kind = kind
        self.size = len(data)
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.bytes = data

    def snapshot(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source),
            "archive_path": self.archive_path,
            "kind": self.kind,
            "size_bytes": self.size,
            "sha256": self.sha256,
        }


def _collect_evidence(
    *,
    roots: Sequence[Path],
    repo_root: Path,
    scheduler_run_root: Path,
    excluded_paths: set[Path],
) -> tuple[list[_Entry], list[dict[str, Any]]]:
    queue: list[tuple[Path, str]] = [(path, "artifact") for path in roots]
    seen_json: set[Path] = set()
    candidates: dict[Path, str] = {}
    root_rows: list[dict[str, Any]] = []
    for path in roots:
        if not _is_within(path, repo_root):
            raise EvidenceBundleError(f"root artifact is outside repository: {path}")
        if not path.is_file() or path.suffix.lower() != ".json":
            raise EvidenceBundleError(f"root artifact must be a JSON file: {path}")
        raw = _stable_read(path)
        payload = _decode_json(raw, path)
        _validate_root(payload, path)
        root_rows.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "gate": payload.get("gate"),
                "status": payload.get("status"),
            }
        )

    while queue:
        queued_source, queued_kind = queue.pop(0)
        source = queued_source.resolve()
        if source in seen_json:
            continue
        seen_json.add(source)
        if source in excluded_paths:
            raise EvidenceBundleError(f"artifact graph references an output path: {source}")
        raw = _stable_read(source)
        payload = _decode_json(raw, source)
        candidates[source] = queued_kind
        for value in _path_values(payload):
            path = _normalize_candidate(value)
            if path is None or path in excluded_paths:
                continue
            if _is_within(path, scheduler_run_root):
                if not path.is_file():
                    raise EvidenceBundleError(f"referenced scheduler evidence is missing: {path}")
                candidates[path] = "raw_scheduler_evidence"
                if path.suffix.lower() == ".json":
                    queue.append((path, "raw_scheduler_evidence"))
            elif _is_within(path, repo_root) and path.suffix.lower() == ".json":
                if not path.is_file():
                    raise EvidenceBundleError(f"referenced repository artifact is missing: {path}")
                queue.append((path, "artifact"))

    entries: list[_Entry] = []
    archive_paths: set[str] = set()
    for source, kind in sorted(candidates.items(), key=lambda item: str(item[0])):
        if kind == "artifact":
            relative = source.relative_to(repo_root)
            archive_path = f"repository/{relative.as_posix()}"
        else:
            relative = source.relative_to(scheduler_run_root)
            archive_path = f"scheduler_runs/{relative.as_posix()}"
        if archive_path in archive_paths:
            raise EvidenceBundleError(f"duplicate archive path: {archive_path}")
        archive_paths.add(archive_path)
        entries.append(
            _Entry(
                source=source,
                archive_path=archive_path,
                kind=kind,
                data=_stable_read(source),
            )
        )
    if not any(entry.kind == "raw_scheduler_evidence" for entry in entries):
        raise EvidenceBundleError("artifact graph contains no scheduler-run evidence")
    return entries, root_rows


def _validate_root(payload: Mapping[str, Any], path: Path) -> None:
    if payload.get("status") != "PASS" or payload.get("pass") is not True:
        raise EvidenceBundleError(f"root artifact is not PASS: {path}")
    if payload.get("ordinary_running_tasks_touched") is True:
        raise EvidenceBundleError(f"root artifact touched an ordinary running task: {path}")


def _path_values(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(child, str) and (
                str(key) == "path"
                or str(key).endswith("_path")
                or str(key).endswith("_log_path")
            ):
                yield child
            yield from _path_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _path_values(child)


def _normalize_candidate(value: str) -> Path | None:
    text = str(value).strip()
    if not text or not text.startswith("/"):
        return None
    return Path(text).resolve()


def _stable_read(path: Path) -> bytes:
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != after.st_size
    ):
        raise EvidenceBundleError(f"evidence changed while being read: {path}")
    return data


def _decode_json(raw: bytes, path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBundleError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceBundleError(f"JSON artifact root must be an object: {path}")
    return value


def _deterministic_tar_gz(*, entries: Sequence[_Entry], manifest_bytes: bytes) -> bytes:
    target = io.BytesIO()
    with gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
            _add_tar_bytes(archive, "manifest.json", manifest_bytes)
            for entry in entries:
                _add_tar_bytes(archive, entry.archive_path, entry.bytes)
    return target.getvalue()


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _markdown(report: Mapping[str, Any], *, manifest_path: Path) -> str:
    return "\n".join(
        (
            "# GPU raw evidence bundle",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Status: `{report.get('status')}`",
            f"- Root artifacts: `{report.get('root_artifact_count')}`",
            f"- Archived files: `{report.get('entry_count')}`",
            f"- Bundle bytes: `{report.get('bundle_size_bytes')}`",
            f"- Bundle SHA-256: `{report.get('bundle_sha256')}`",
            "",
            str(report.get("claim_boundary") or ""),
            "",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-artifact", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_gpu_raw_evidence_bundle(
            root_artifacts=args.root_artifact,
            output_path=args.output,
            manifest_path=args.manifest,
            markdown_path=args.markdown,
        )
    except EvidenceBundleError as exc:
        print(json.dumps({"status": "FAIL_EVIDENCE", "pass": False, "error": str(exc)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass": report["pass"],
                "entry_count": report["entry_count"],
                "bundle_sha256": report["bundle_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
