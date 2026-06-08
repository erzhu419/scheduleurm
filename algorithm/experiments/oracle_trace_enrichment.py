"""Enrich scheduler candidate traces with theorem lower-service semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.oracle_trace import load_trace_slots

from .theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


MATCH_KEYS = ("action_id", "candidate_bucket", "class_key", "regime_key")


def build_service_lookup(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        lower = row.get("lower_service") or row.get("lower_service_vector") or row.get("service_vector")
        if not isinstance(lower, Mapping) or not lower:
            continue
        payload = {
            "lower_service": dict(lower),
            "penalty_units": _as_float(row.get("penalty_units"), 0.0),
            "source": row.get("source"),
        }
        for key in MATCH_KEYS:
            value = str(row.get(key) or "").strip()
            if value:
                lookup[(key, value)] = payload
    return lookup


def enrich_trace_slots(
    slots: Iterable[Mapping[str, Any]],
    *,
    service_rows: Iterable[Mapping[str, Any]],
    queue_vector: Mapping[str, Any],
) -> dict[str, Any]:
    slot_list = [dict(slot) for slot in slots]
    if not isinstance(queue_vector, Mapping) or not queue_vector:
        return {
            "status": "ENRICHMENT_BLOCKED",
            "input_slot_count": len(slot_list),
            "enriched_slot_count": 0,
            "blocker_count": 1,
            "blockers": [{"reason": "missing_queue_vector"}],
            "usable_for_theorem": False,
            "enriched_slots": [],
        }
    lookup = build_service_lookup(service_rows)
    enriched = []
    blockers = []
    for slot in slot_list:
        converted, slot_blockers = enrich_one_slot(slot, lookup=lookup, queue_vector=queue_vector)
        if slot_blockers:
            blockers.extend(slot_blockers)
        else:
            enriched.append(converted)
    if blockers or not enriched:
        return {
            "status": "ENRICHMENT_BLOCKED" if blockers else "NO_TRACE_SLOTS",
            "input_slot_count": len(slot_list),
            "enriched_slot_count": len(enriched),
            "blocker_count": len(blockers),
            "blockers": blockers[:100],
            "usable_for_theorem": False,
            "enriched_slots": enriched,
        }
    audit = build_theorem_oracle_audit_from_trace(enriched)
    return {
        "status": "ENRICHED_THEOREM_PASS" if audit.get("usable_for_theorem") else "ENRICHED_THEOREM_FAIL",
        "input_slot_count": len(enriched),
        "enriched_slot_count": len(enriched),
        "blocker_count": 0,
        "blockers": [],
        "usable_for_theorem": bool(audit.get("usable_for_theorem")),
        "oracle_audit": audit,
        "enriched_slots": enriched,
    }


def enrich_one_slot(
    slot: Mapping[str, Any],
    *,
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    queue_vector: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    slot_id = slot.get("slot_id")
    out = dict(slot)
    out["score_semantics"] = "robust_maxweight_lower_service"
    out["queue_vector"] = dict(queue_vector)
    out["best_score_exact_over_candidate_set"] = bool(
        slot.get("best_score_exact_over_candidate_set", True)
    )
    blockers = []
    candidates = []
    for row in slot.get("candidates") or []:
        candidate = dict(row)
        match = _lookup_candidate(candidate, lookup)
        if match is None:
            blockers.append({
                "slot_id": slot_id,
                "action_id": candidate.get("action_id"),
                "reason": "candidate_lower_service_not_found",
                "candidate_bucket": candidate.get("candidate_bucket"),
                "class_key": candidate.get("class_key"),
                "regime_key": candidate.get("regime_key"),
            })
        else:
            candidate["lower_service"] = dict(match.get("lower_service") or {})
            candidate["penalty_units"] = _as_float(match.get("penalty_units"), 0.0)
            candidate["lower_service_source"] = match.get("source")
        candidates.append(candidate)
    out["candidates"] = candidates
    return out, blockers


def audit_enriched_trace_file(
    *,
    trace_path: str | Path,
    service_lookup_path: str | Path,
    queue_vector_path: str | Path,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    if not trace.exists():
        return {
            "status": "NO_TRACE",
            "input": str(trace),
            "input_slot_count": 0,
            "enriched_slot_count": 0,
            "blocker_count": 1,
            "usable_for_theorem": False,
            "blockers": [{"reason": "trace_file_does_not_exist", "path": str(trace)}],
        }
    service_rows = _json_rows(Path(service_lookup_path).expanduser())
    queue_vector = _json_mapping(Path(queue_vector_path).expanduser())
    report = enrich_trace_slots(
        load_trace_slots(trace),
        service_rows=service_rows,
        queue_vector=queue_vector,
    )
    report["input"] = str(trace)
    report["service_lookup"] = str(Path(service_lookup_path).expanduser())
    report["queue_vector"] = str(Path(queue_vector_path).expanduser())
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    audit = report.get("oracle_audit") or {}
    lines = [
        "# Module55 Oracle Trace Lower-Service Enrichment",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `input_slot_count` | {report.get('input_slot_count', 0)} |",
        f"| `enriched_slot_count` | {report.get('enriched_slot_count', 0)} |",
        f"| `blocker_count` | {report.get('blocker_count', 0)} |",
        f"| `alpha0` | {_fmt(audit.get('alpha0'))} |",
        f"| `alpha1` | {_fmt(audit.get('alpha1'))} |",
        f"| `usable_for_theorem` | {str(bool(report.get('usable_for_theorem'))).lower()} |",
        "",
    ]
    blockers = list(report.get("blockers") or [])
    if blockers:
        lines.extend(["## Blockers", "", "| Reason | Slot | Action | Bucket |", "|---|---|---|---|"])
        for row in blockers[:50]:
            lines.append(
                "| `{reason}` | `{slot}` | `{action}` | `{bucket}` |".format(
                    reason=row.get("reason"),
                    slot=row.get("slot_id"),
                    action=row.get("action_id"),
                    bucket=row.get("candidate_bucket"),
                )
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "This module is the bridge from live candidate-family traces to theorem",
        "`alpha0, alpha1`.  It only passes when every candidate in every slot can",
        "be matched to a lower-service vector and a queue vector is supplied.",
        "",
    ])
    return "\n".join(lines)


def _lookup_candidate(
    candidate: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for key in MATCH_KEYS:
        value = str(candidate.get(key) or "").strip()
        if value and (key, value) in lookup:
            return lookup[(key, value)]
    return None


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
    else:
        payload = None
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "records", "actions", "candidate_services", "service_rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, Mapping)]
        if any(key in payload for key in ("lower_service", "lower_service_vector", "service_vector")):
            return [dict(payload)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return dict(payload.get("queue_vector") or payload)
    return {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return "NA"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_enrich(args: argparse.Namespace) -> int:
    report = audit_enriched_trace_file(
        trace_path=args.input,
        service_lookup_path=args.service_lookup,
        queue_vector_path=args.queue_vector,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        path = Path(args.markdown_output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("usable_for_theorem") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.oracle_trace_enrichment")
    sub = p.add_subparsers(dest="cmd", required=True)
    enrich = sub.add_parser("enrich", help="Attach lower-service vectors to candidate trace slots")
    enrich.add_argument("--input", required=True)
    enrich.add_argument("--service-lookup", required=True)
    enrich.add_argument("--queue-vector", required=True)
    enrich.add_argument("--output", required=True)
    enrich.add_argument("--markdown-output", default="")
    enrich.set_defaults(func=_cmd_enrich)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
