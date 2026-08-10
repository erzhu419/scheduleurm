"""Audit MMRCPSP reruns without confusing gzip headers with result drift."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_RENEWAL_PRIMARY_GZIP = (
    ARTIFACT_ROOT / "mmrcpsp_renewal_stream_v1_20260810.json.gz"
)
DEFAULT_RENEWAL_PRIMARY_COMPACT = (
    ARTIFACT_ROOT / "mmrcpsp_renewal_stream_v1_20260810.json"
)
DEFAULT_RENEWAL_PRIMARY_MARKDOWN = (
    REPO_ROOT / "md" / "mmrcpsp_renewal_stream_v1_20260810.md"
)
DEFAULT_RENEWAL_REPRO_GZIP = Path(
    "/tmp/mmrcpsp_renewal_stream_v1_repro_20260810.json.gz"
)
DEFAULT_RENEWAL_REPRO_COMPACT = Path(
    "/tmp/mmrcpsp_renewal_stream_v1_repro_20260810.json"
)
DEFAULT_RENEWAL_REPRO_MARKDOWN = Path(
    "/tmp/mmrcpsp_renewal_stream_v1_repro_20260810.md"
)
DEFAULT_BALANCE_PRIMARY_GZIP = (
    ARTIFACT_ROOT / "mmrcpsp_class_balance_holdout_v1_20260810.json.gz"
)
DEFAULT_BALANCE_PRIMARY_COMPACT = (
    ARTIFACT_ROOT / "mmrcpsp_class_balance_holdout_v1_20260810.json"
)
DEFAULT_BALANCE_PRIMARY_MARKDOWN = (
    REPO_ROOT / "md" / "mmrcpsp_class_balance_holdout_v1_20260810.md"
)
DEFAULT_BALANCE_REPRO_GZIP = Path(
    "/tmp/mmrcpsp_class_balance_holdout_v1_repro_20260810.json.gz"
)
DEFAULT_BALANCE_REPRO_COMPACT = Path(
    "/tmp/mmrcpsp_class_balance_holdout_v1_repro_20260810.json"
)
DEFAULT_BALANCE_REPRO_MARKDOWN = Path(
    "/tmp/mmrcpsp_class_balance_holdout_v1_repro_20260810.md"
)
DEFAULT_JSON = ARTIFACT_ROOT / "mmrcpsp_reproducibility_audit_20260810.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "mmrcpsp_reproducibility_audit_20260810.md"

CONTAINER_DEPENDENT_COMPACT_FIELDS = frozenset(
    {"full_artifact_gzip_sha256", "artifact_sha256_excluding_self"}
)


class MMRCPSPReproducibilityAuditError(ValueError):
    """Raised when an input pair is invalid or semantic reproducibility fails."""


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _read_gzip_original_filename(path: str | Path) -> str | None:
    """Read the optional FNAME member from an RFC 1952 header."""
    raw = Path(path).read_bytes()
    if len(raw) < 10 or raw[:2] != b"\x1f\x8b" or raw[2] != 8:
        raise MMRCPSPReproducibilityAuditError(f"not a gzip stream: {path}")
    flags = raw[3]
    offset = 10
    if flags & 0x04:
        if len(raw) < offset + 2:
            raise MMRCPSPReproducibilityAuditError(f"truncated gzip XLEN: {path}")
        xlen = int.from_bytes(raw[offset : offset + 2], "little")
        offset += 2 + xlen
    if not flags & 0x08:
        return None
    end = raw.find(b"\x00", offset)
    if end < 0:
        raise MMRCPSPReproducibilityAuditError(f"truncated gzip FNAME: {path}")
    return raw[offset:end].decode("latin-1")


def _canonical_semantic_compact(payload: Mapping[str, Any]) -> bytes:
    normalized = deepcopy(dict(payload))
    for field in CONTAINER_DEPENDENT_COMPACT_FIELDS:
        normalized.pop(field, None)
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def audit_pair(
    *,
    name: str,
    expected_schema: str,
    primary_gzip: str | Path,
    primary_compact: str | Path,
    primary_markdown: str | Path,
    reproduction_gzip: str | Path,
    reproduction_compact: str | Path,
    reproduction_markdown: str | Path,
) -> dict[str, Any]:
    paths = {
        "primary_gzip": Path(primary_gzip),
        "primary_compact": Path(primary_compact),
        "primary_markdown": Path(primary_markdown),
        "reproduction_gzip": Path(reproduction_gzip),
        "reproduction_compact": Path(reproduction_compact),
        "reproduction_markdown": Path(reproduction_markdown),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise MMRCPSPReproducibilityAuditError(
            f"missing reproduction inputs for {name}: {missing}"
        )

    primary_gzip_raw = paths["primary_gzip"].read_bytes()
    reproduction_gzip_raw = paths["reproduction_gzip"].read_bytes()
    primary_full_raw = gzip.decompress(primary_gzip_raw)
    reproduction_full_raw = gzip.decompress(reproduction_gzip_raw)
    primary_compact_raw = paths["primary_compact"].read_bytes()
    reproduction_compact_raw = paths["reproduction_compact"].read_bytes()
    primary_compact_payload = json.loads(primary_compact_raw)
    reproduction_compact_payload = json.loads(reproduction_compact_raw)
    if primary_compact_payload.get("schema_version") != expected_schema:
        raise MMRCPSPReproducibilityAuditError(
            f"unexpected primary schema for {name}"
        )
    if reproduction_compact_payload.get("schema_version") != expected_schema:
        raise MMRCPSPReproducibilityAuditError(
            f"unexpected reproduction schema for {name}"
        )

    primary_semantic = _canonical_semantic_compact(primary_compact_payload)
    reproduction_semantic = _canonical_semantic_compact(
        reproduction_compact_payload
    )
    primary_markdown_raw = paths["primary_markdown"].read_bytes()
    reproduction_markdown_raw = paths["reproduction_markdown"].read_bytes()
    full_equal = primary_full_raw == reproduction_full_raw
    semantic_compact_equal = primary_semantic == reproduction_semantic
    markdown_equal = primary_markdown_raw == reproduction_markdown_raw
    raw_gzip_equal = primary_gzip_raw == reproduction_gzip_raw
    semantic_ready = full_equal and semantic_compact_equal and markdown_equal
    if not semantic_ready:
        raise MMRCPSPReproducibilityAuditError(
            f"semantic reproduction failed for {name}"
        )

    return {
        "name": name,
        "expected_schema": expected_schema,
        "semantic_reproducibility_ready": semantic_ready,
        "decompressed_full_artifact_equal": full_equal,
        "semantic_compact_artifact_equal": semantic_compact_equal,
        "markdown_equal": markdown_equal,
        "raw_gzip_container_equal": raw_gzip_equal,
        "path_independent_gzip_container_reproducibility_ready": raw_gzip_equal,
        "gzip_header_original_filename": {
            "primary": _read_gzip_original_filename(paths["primary_gzip"]),
            "reproduction": _read_gzip_original_filename(
                paths["reproduction_gzip"]
            ),
        },
        "hashes": {
            "primary_gzip": _sha256(primary_gzip_raw),
            "reproduction_gzip": _sha256(reproduction_gzip_raw),
            "primary_decompressed_full": _sha256(primary_full_raw),
            "reproduction_decompressed_full": _sha256(reproduction_full_raw),
            "primary_compact_raw": _sha256(primary_compact_raw),
            "reproduction_compact_raw": _sha256(reproduction_compact_raw),
            "primary_compact_semantic": _sha256(primary_semantic),
            "reproduction_compact_semantic": _sha256(reproduction_semantic),
            "primary_markdown": _sha256(primary_markdown_raw),
            "reproduction_markdown": _sha256(reproduction_markdown_raw),
        },
        "container_difference_explanation": (
            "Python gzip embeds the output basename in the optional FNAME "
            "header; a different output path changes container bytes and the "
            "compact file's recorded gzip hash without changing decompressed "
            "results."
        ),
        "excluded_compact_fields": sorted(CONTAINER_DEPENDENT_COMPACT_FIELDS),
    }


def build_audit(**overrides: str | Path) -> dict[str, Any]:
    renewal = audit_pair(
        name="mmrcpsp_renewal_stream_v1",
        expected_schema="scheduleurm.mmrcpsp_renewal_stream.compact.v1",
        primary_gzip=overrides.get(
            "renewal_primary_gzip", DEFAULT_RENEWAL_PRIMARY_GZIP
        ),
        primary_compact=overrides.get(
            "renewal_primary_compact", DEFAULT_RENEWAL_PRIMARY_COMPACT
        ),
        primary_markdown=overrides.get(
            "renewal_primary_markdown", DEFAULT_RENEWAL_PRIMARY_MARKDOWN
        ),
        reproduction_gzip=overrides.get(
            "renewal_repro_gzip", DEFAULT_RENEWAL_REPRO_GZIP
        ),
        reproduction_compact=overrides.get(
            "renewal_repro_compact", DEFAULT_RENEWAL_REPRO_COMPACT
        ),
        reproduction_markdown=overrides.get(
            "renewal_repro_markdown", DEFAULT_RENEWAL_REPRO_MARKDOWN
        ),
    )
    balance = audit_pair(
        name="mmrcpsp_class_balance_holdout_v1",
        expected_schema="scheduleurm.mmrcpsp_class_balance_holdout.compact.v1",
        primary_gzip=overrides.get(
            "balance_primary_gzip", DEFAULT_BALANCE_PRIMARY_GZIP
        ),
        primary_compact=overrides.get(
            "balance_primary_compact", DEFAULT_BALANCE_PRIMARY_COMPACT
        ),
        primary_markdown=overrides.get(
            "balance_primary_markdown", DEFAULT_BALANCE_PRIMARY_MARKDOWN
        ),
        reproduction_gzip=overrides.get(
            "balance_repro_gzip", DEFAULT_BALANCE_REPRO_GZIP
        ),
        reproduction_compact=overrides.get(
            "balance_repro_compact", DEFAULT_BALANCE_REPRO_COMPACT
        ),
        reproduction_markdown=overrides.get(
            "balance_repro_markdown", DEFAULT_BALANCE_REPRO_MARKDOWN
        ),
    )
    pairs = [renewal, balance]
    report: dict[str, Any] = {
        "schema_version": "scheduleurm.mmrcpsp.reproducibility_audit.v1",
        "gate": {
            "pass": all(row["semantic_reproducibility_ready"] for row in pairs),
            "status": "MMRCPSP_SEMANTIC_REPRODUCIBILITY_PASS",
            "pair_count": len(pairs),
            "semantic_reproducibility_ready_count": sum(
                bool(row["semantic_reproducibility_ready"]) for row in pairs
            ),
            "raw_gzip_container_match_count": sum(
                bool(row["raw_gzip_container_equal"]) for row in pairs
            ),
            "bitwise_path_independent_gzip_claim_ready": all(
                row["raw_gzip_container_equal"] for row in pairs
            ),
        },
        "pairs": pairs,
        "claim_boundary": {
            "supports": (
                "exact semantic reproduction of both deterministic MMRCPSP "
                "experiment matrices under the frozen code and input ledgers"
            ),
            "does_not_support": (
                "path-independent bitwise identity of gzip containers whose "
                "headers retain different output basenames"
            ),
        },
    }
    report["artifact_sha256_excluding_self"] = _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# MMRCPSP Reproducibility Audit",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- Semantic equality is evaluated after decompression and after removing only container-dependent self-hash fields.",
        "- Raw gzip equality is reported separately and is not used to excuse any result drift.",
        "",
        "| Experiment | Full JSON after decompression | Compact semantics | Markdown | Raw gzip | Gzip FNAME |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["pairs"]:
        names = row["gzip_header_original_filename"]
        lines.append(
            f"| {row['name']} | {row['decompressed_full_artifact_equal']} | "
            f"{row['semantic_compact_artifact_equal']} | {row['markdown_equal']} | "
            f"{row['raw_gzip_container_equal']} | `{names['primary']}` vs `{names['reproduction']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Supports: {report['claim_boundary']['supports']}.",
            f"- Does not support: {report['claim_boundary']['does_not_support']}.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    report: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    output = Path(json_path)
    markdown = Path(markdown_path)
    for path in (output, markdown):
        if path.exists():
            raise MMRCPSPReproducibilityAuditError(
                f"refusing to overwrite result: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(markdown_report(report), encoding="utf-8")
    return output, markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_audit()
    output, markdown = write_outputs(
        report, json_path=args.json, markdown_path=args.markdown
    )
    print(
        json.dumps(
            {"gate": report["gate"], "json": str(output), "markdown": str(markdown)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
