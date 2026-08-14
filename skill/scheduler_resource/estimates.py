"""Queued task RAM/VRAM estimate refresh logic."""

from __future__ import annotations

import time as _time
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ResourceEstimateDeps:
    maybe_lower_explicit_resource_estimate: Callable[..., bool]
    effective_est_vram: Callable[[dict, dict, dict], int]
    effective_est_ram: Callable[[dict, dict, dict], int]
    live_sibling_ram_floor: Callable[[dict, dict], int]
    live_sibling_resource_estimate_index: Callable[[dict], dict]
    description_sibling_resource_estimate_index: Callable[[dict], dict]
    now: Callable[[], float] = _time.time


@dataclass(frozen=True)
class EffectiveResourceEstimateDeps:
    default_vram_mb: int
    default_ram_mb: int
    untrusted_startup_oom_sample: Callable[[dict], bool]


def _desc_key(task: dict) -> str:
    return (task.get("description") or "").split(":")[0].strip().lower()


def _history_vram(record) -> int:
    if isinstance(record, dict):
        if _history_has_host_aggregate_vram(record):
            return 0
        return int(record.get("vram_mb") or 0)
    if isinstance(record, int):
        return int(record)
    return 0


def _history_ram(record) -> int:
    if isinstance(record, dict):
        return int(record.get("ram_mb") or 0)
    return 0


def _median_or_zero(values: list[int]) -> int:
    if not values:
        return 0
    values = sorted(values)
    return int(values[len(values) // 2])


_SHARD_INDEX_RE = re.compile(
    r"(?i)(?P<prefix>shard(?:[_-]?|=))(?P<index>\d+)"
    r"(?=(?:[_-]?of[_-]?|/)\d+\b)"
)
_SEED_RE = re.compile(r"(?i)(?P<prefix>\bseed[_=-]?)(?P<index>\d+)\b")
_DESCRIPTION_SEED_RE = re.compile(r"(?i)\bseed(?:\s*[_=-]\s*|\s+)?\d+\b")


def _is_host_worker_family_pair(host_family: object, worker_family: object) -> bool:
    host = str(host_family or "").strip().lower()
    worker = str(worker_family or "").strip().lower()
    return bool(
        host
        and worker
        and host.endswith("-host")
        and worker.endswith("-worker")
        and host[:-len("-host")] == worker[:-len("-worker")]
    )


def task_has_host_aggregate_vram(task: dict) -> bool:
    """Return whether task-level VRAM aggregates a host's worker processes.

    Host dispatch tasks reserve one worker-sized VRAM budget per GPU, while
    their process probe reports the sum across every GPU child.  The explicit
    host/worker family pair is the persisted contract that distinguishes this
    aggregate telemetry from ordinary single-task VRAM.
    """
    if not isinstance(task, dict):
        return False
    host_family = task.get("ram_resource_family") or task.get("resource_family")
    return _is_host_worker_family_pair(
        host_family,
        task.get("vram_resource_family"),
    )


def _history_has_host_aggregate_vram(record: dict) -> bool:
    if str(record.get("vram_observation_scope") or "").strip().lower() == "host_aggregate":
        return True
    return _is_host_worker_family_pair(
        record.get("ram_resource_family_key"),
        record.get("vram_resource_family_key"),
    )


def description_resource_family_key(task: dict) -> str:
    """Return a conservative project+description resource family.

    Full signatures remain the primary family because they preserve exact
    configuration.  This key is a fallback for repeated experiments whose run
    directory changes while the submitted description stays stable.  Only
    shard and seed indices are normalized; other description parameters remain
    distinct.
    """
    project = str(task.get("project") or "").strip().lower()
    description = str(task.get("description") or "").strip().lower()
    if not project or not description:
        return ""
    description = _SHARD_INDEX_RE.sub(r"\g<prefix>*", description)
    description = _DESCRIPTION_SEED_RE.sub("seed=*", description)
    description = " ".join(description.split())
    return f"{project}|{description}"


def resource_family_key(task: dict, kind: str | None = None) -> str:
    """Return the stable resource class for sharded/seeded sibling tasks.

    A kind-specific family lets a short, formal-shaped smoke calibrate GPU
    memory without treating its short-lived host-memory footprint as evidence
    for a long training run.
    """
    if kind not in (None, "ram", "vram"):
        raise ValueError(f"unsupported resource kind: {kind}")
    explicit = (
        task.get(f"{kind}_resource_family") if kind else None
    ) or task.get("resource_family") or task.get("resource_class")
    source = explicit or task.get("signature") or task.get("description") or ""
    source = str(source).strip().lower()
    if not source:
        return ""
    # Explicit family names are caller-defined isolation boundaries. Only
    # inferred signature/description families normalize shard and seed ids.
    if not explicit:
        source = _SHARD_INDEX_RE.sub(r"\g<prefix>*", source)
        source = _SEED_RE.sub(r"\g<prefix>*", source)
    project = str(task.get("project") or "").strip().lower()
    return f"{project}|{source}"


def resource_history_metadata(task: dict) -> dict[str, str]:
    """Return compact resource-family metadata persisted with peak history."""
    if not isinstance(task, dict):
        return {}
    metadata = {
        "project": str(task.get("project") or "").strip().lower(),
        "description_resource_family_key": description_resource_family_key(task),
        "ram_resource_family_key": resource_family_key(task, "ram"),
        "vram_resource_family_key": resource_family_key(task, "vram"),
        "resource_mode": _task_resource_mode(task),
    }
    if task_has_host_aggregate_vram(task):
        metadata["vram_observation_scope"] = "host_aggregate"
    return {key: value for key, value in metadata.items() if value}


def _explicit_resource_family(task: dict, kind: str) -> str:
    return str(
        task.get(f"{kind}_resource_family")
        or task.get("resource_family")
        or task.get("resource_class")
        or ""
    ).strip()


def _exact_history_matches_resource_family(
    task: dict,
    history_entry: dict | None,
    kind: str,
) -> bool:
    """Keep an explicit family isolated from stale exact-signature history."""
    if not _explicit_resource_family(task, kind):
        return True
    if not isinstance(history_entry, dict):
        return False
    expected = resource_family_key(task, kind)
    observed = str(
        history_entry.get(f"{kind}_resource_family_key") or ""
    ).strip()
    return bool(expected and observed == expected)


def live_sibling_resource_estimate_index(state: dict) -> dict[tuple[str, str], int]:
    """Build exact live-family RAM/VRAM estimates in one state pass."""
    samples: dict[tuple[str, str], list[int]] = {}
    for other in state.get("tasks", []):
        if other.get("status") not in ("running", "launching", "done"):
            continue
        for kind, minimum_observed in (("ram", 16), ("vram", 100)):
            if kind == "vram" and task_has_host_aggregate_vram(other):
                continue
            family = resource_family_key(other, kind)
            if not family:
                continue
            observed = max(
                int(other.get(f"current_{kind}_mb") or 0),
                int(other.get(f"peak_{kind}_mb") or 0),
            )
            # Startup/compile samples are incomplete. They may raise an
            # estimate immediately, but cannot lower queued descendants until
            # the task has emitted progress or completed.
            has_progress = bool(
                other.get("last_progress_line")
                or int(other.get("runtime_current_unit") or 0) > 0
            )
            submitted = int(other.get(
                "est_vram_mb" if kind == "vram" else "ram_mb") or 0)
            mature = other.get("status") == "done" or has_progress
            if not mature and observed <= submitted:
                continue
            if observed > minimum_observed:
                samples.setdefault((family, kind), []).append(observed)
    return {key: _median_or_zero(values) for key, values in samples.items()}


def live_sibling_resource_estimate(task: dict, state: dict, kind: str) -> int:
    """Return median observed RAM/VRAM for an exact live resource family."""
    if kind not in ("ram", "vram"):
        raise ValueError(f"unsupported resource kind: {kind}")
    family = resource_family_key(task, kind)
    if not family:
        return 0
    return int(live_sibling_resource_estimate_index(state).get((family, kind), 0))


def _upper_observed_or_zero(values: list[int]) -> int:
    """Return a conservative nearest-rank p80 observation."""
    if not values:
        return 0
    values = sorted(values)
    rank = max(1, (len(values) * 80 + 99) // 100)
    return int(values[min(len(values), rank) - 1])


def _high_confidence_observed_or_zero(values: list[int]) -> int:
    """Return nearest-rank p90 for a deliberately broad project fallback."""
    if not values:
        return 0
    values = sorted(values)
    rank = max(1, (len(values) * 90 + 99) // 100)
    return int(values[min(len(values), rank) - 1])


def _task_resource_mode(task: dict) -> str:
    if task.get("est_vram_mb_explicit") and int(task.get("est_vram_mb") or 0) == 0:
        return "cpu"
    observed_vram = max(
        int(task.get("est_vram_mb") or 0),
        int(task.get("current_vram_mb") or 0),
        int(task.get("peak_vram_mb") or 0),
    )
    return "gpu" if observed_vram > 100 else "cpu"


def description_sibling_resource_estimate_index(state: dict) -> dict[tuple[str, str, str], int]:
    """Build strong cross-signature description evidence in one state pass.

    This fallback deliberately requires at least two successful/live samples.
    A single running process may not have reached its peak yet and must not
    silently lower a user's explicit resource request.
    """
    samples: dict[tuple[str, str, str], list[int]] = {}
    for other in state.get("tasks", []):
        if other.get("status") not in ("running", "done"):
            continue
        family = description_resource_family_key(other)
        if not family:
            continue
        mode = _task_resource_mode(other)
        for kind, minimum_observed in (("ram", 16), ("vram", 100)):
            if kind == "vram" and task_has_host_aggregate_vram(other):
                continue
            observed = max(
                int(other.get(f"current_{kind}_mb") or 0),
                int(other.get(f"peak_{kind}_mb") or 0),
            )
            if observed > minimum_observed:
                key_mode = mode if kind == "ram" else "vram"
                samples.setdefault((family, key_mode, kind), []).append(observed)
    return {
        key: _upper_observed_or_zero(values)
        for key, values in samples.items()
        if len(values) >= 2
    }


def description_sibling_resource_estimate(task: dict, state: dict, kind: str) -> int:
    """Return indexed cross-signature evidence for one queued task."""
    if kind not in ("ram", "vram"):
        raise ValueError(f"unsupported resource kind: {kind}")
    family = description_resource_family_key(task)
    if not family:
        return 0
    mode = _task_resource_mode(task) if kind == "ram" else "vram"
    return int(description_sibling_resource_estimate_index(state).get((family, mode, kind), 0))



def _signature_prefix(signature: str) -> str:
    parts = [part.strip().lower() for part in str(signature or "").split("/")]
    return "/".join(parts[:2]) if len(parts) >= 2 else ""


def _mature_resource_observation(task: dict, kind: str) -> int:
    if task.get("status") not in ("running", "done"):
        return 0
    if task.get("status") == "running" and not (
        task.get("last_progress_line")
        or int(task.get("runtime_current_unit") or 0) > 0
    ):
        return 0
    if kind == "vram" and task_has_host_aggregate_vram(task):
        return 0
    observed = max(
        int(task.get(f"current_{kind}_mb") or 0),
        int(task.get(f"peak_{kind}_mb") or 0),
    )
    minimum = 100 if kind == "vram" else 16
    return observed if observed > minimum else 0


def learned_resource_estimate_index(
    state: dict,
    history: dict,
) -> dict[tuple[str, str, str, str], dict[str, int | str]]:
    """Index actual resource evidence once for all queued tasks.

    Explicit submit values are cold-start hints. Once a sibling, campaign, or
    sufficiently sampled project has real telemetry, placement should use that
    telemetry instead. The index keeps refresh O(tasks + history) rather than
    rescanning all tasks for every queued item.
    """
    buckets: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    root_projects: dict[str, set[str]] = defaultdict(set)

    for task in state.get("tasks", []):
        project = str(task.get("project") or "").strip().lower()
        signature = str(task.get("signature") or "")
        root = signature.split("/", 1)[0].strip().lower()
        if project and root:
            root_projects[root].add(project)

    def add(scope: str, key: str, mode: str, kind: str, observed: int) -> None:
        if key and observed > 0:
            buckets[(scope, key, mode, kind)].append(int(observed))

    for task in state.get("tasks", []):
        project = str(task.get("project") or "").strip().lower()
        prefix = _signature_prefix(task.get("signature") or "")
        mode = _task_resource_mode(task)
        for kind in ("ram", "vram"):
            observed = _mature_resource_observation(task, kind)
            if not observed:
                continue
            kind_mode = mode if kind == "vram" else ""
            add("family", resource_family_key(task, kind), kind_mode, kind, observed)
            add("prefix", prefix, kind_mode, kind, observed)
            add("project", project, kind_mode, kind, observed)

    for signature, raw_record in history.items():
        record = raw_record if isinstance(raw_record, dict) else {
            "vram_mb": raw_record if isinstance(raw_record, int) else 0,
        }
        prefix = _signature_prefix(signature)
        root = str(signature or "").split("/", 1)[0].strip().lower()
        project = str(record.get("project") or "").strip().lower()
        if not project and len(root_projects.get(root, set())) == 1:
            project = next(iter(root_projects[root]))
        if not project:
            project = root
        recorded_mode = str(record.get("resource_mode") or "").strip().lower()
        for kind in ("ram", "vram"):
            observed = _history_ram(record) if kind == "ram" else _history_vram(record)
            if not observed:
                continue
            # Legacy history has no CPU/GPU metadata. It remains useful for
            # host RAM, but must not turn a queued CPU task into a GPU task.
            if kind == "vram" and recorded_mode not in ("cpu", "gpu"):
                continue
            kind_mode = recorded_mode if kind == "vram" else ""
            family = str(record.get(f"{kind}_resource_family_key") or "").strip().lower()
            add("family", family, kind_mode, kind, observed)
            add("prefix", prefix, kind_mode, kind, observed)
            add("project", project, kind_mode, kind, observed)

    result: dict[tuple[str, str, str, str], dict[str, int | str]] = {}
    for index_key, values in buckets.items():
        scope = index_key[0]
        minimum_samples = 1 if scope == "family" else 2 if scope == "prefix" else 5
        if len(values) < minimum_samples:
            continue
        observed = (
            _high_confidence_observed_or_zero(values)
            if scope == "project"
            else _upper_observed_or_zero(values)
        )
        result[index_key] = {
            "observed_mb": observed,
            "sample_count": len(values),
            "scope": scope,
        }
    return result

def effective_est_vram(task: dict, state: dict, history: dict, *, deps: EffectiveResourceEstimateDeps) -> int:
    """Best-effort VRAM estimate for a task when exact history is unavailable."""
    sig = task.get("signature") or ""
    own_vram = _history_vram(history.get(sig))
    if own_vram:
        return own_vram

    project = task.get("project")
    desc_key = _desc_key(task)
    candidates = []
    self_id = task.get("id")

    for other in state.get("tasks", []):
        if other.get("id") == self_id:
            continue
        if other.get("project") != project:
            continue
        if task_has_host_aggregate_vram(other):
            continue
        if desc_key and _desc_key(other) == desc_key:
            peak = int(other.get("peak_vram_mb") or 0)
            if peak > 100 and not deps.untrusted_startup_oom_sample(other):
                candidates.append(peak)

    if not candidates and sig:
        prefix = "/".join(sig.split("/")[:2])
        for sig2, record in history.items():
            if sig2 == sig:
                continue
            if "/".join(sig2.split("/")[:2]) != prefix:
                continue
            vram = _history_vram(record)
            if vram > 0:
                candidates.append(vram)

    if not candidates and project:
        for other in state.get("tasks", []):
            if other.get("id") == self_id:
                continue
            if other.get("project") != project:
                continue
            if task_has_host_aggregate_vram(other):
                continue
            peak = int(other.get("peak_vram_mb") or 0)
            if peak > 100 and not deps.untrusted_startup_oom_sample(other):
                candidates.append(peak)

    if not candidates and project:
        for sig2, record in history.items():
            if not sig2.startswith(project + "/"):
                continue
            vram = _history_vram(record)
            if vram > 0:
                candidates.append(vram)

    estimate = _median_or_zero(candidates)
    if estimate:
        return estimate
    stored = task.get("est_vram_mb")
    if stored:
        return int(stored)
    return deps.default_vram_mb


def effective_est_ram(task: dict, state: dict, history: dict, *, deps: EffectiveResourceEstimateDeps) -> int:
    """Best-effort RAM estimate mirroring effective_est_vram."""
    sig = task.get("signature") or ""
    own_ram = _history_ram(history.get(sig))
    if own_ram:
        return own_ram

    project = task.get("project")
    desc_key = _desc_key(task)
    candidates = []
    self_id = task.get("id")

    for other in state.get("tasks", []):
        if other.get("id") == self_id:
            continue
        if other.get("project") != project:
            continue
        if desc_key and _desc_key(other) == desc_key:
            peak = int(other.get("peak_ram_mb") or 0)
            if peak > 100:
                candidates.append(peak)

    if not candidates and sig:
        prefix = "/".join(sig.split("/")[:2])
        for sig2, record in history.items():
            if sig2 == sig:
                continue
            if "/".join(sig2.split("/")[:2]) != prefix:
                continue
            ram = _history_ram(record)
            if ram > 0:
                candidates.append(ram)

    if not candidates and project:
        for other in state.get("tasks", []):
            if other.get("id") == self_id:
                continue
            if other.get("project") != project:
                continue
            peak = int(other.get("peak_ram_mb") or 0)
            if peak > 100:
                candidates.append(peak)

    if not candidates and project:
        for sig2, record in history.items():
            if not sig2.startswith(project + "/"):
                continue
            ram = _history_ram(record)
            if ram > 0:
                candidates.append(ram)

    estimate = _median_or_zero(candidates)
    if estimate:
        return estimate
    stored = task.get("ram_mb")
    if stored:
        return int(stored)
    return deps.default_ram_mb


def live_sibling_ram_floor(task: dict, state: dict) -> int:
    """Return a RAM floor from currently-running sibling tasks, or 0."""
    project = task.get("project")
    if not project:
        return 0
    sig = task.get("signature") or ""
    prefix = "/".join(sig.split("/")[:2]) if sig else ""
    desc_key = _desc_key(task)
    self_id = task.get("id")
    desc_candidates = []
    prefix_candidates = []
    project_candidates = []
    for other in state.get("tasks", []):
        if other.get("id") == self_id:
            continue
        if other.get("project") != project:
            continue
        if other.get("status") not in ("running", "launching"):
            continue
        ram = max(int(other.get("current_ram_mb") or 0), int(other.get("peak_ram_mb") or 0))
        if ram <= 100:
            continue
        project_candidates.append(ram)
        if desc_key and _desc_key(other) == desc_key:
            desc_candidates.append(ram)
        other_sig = other.get("signature") or ""
        if prefix and "/".join(other_sig.split("/")[:2]) == prefix:
            prefix_candidates.append(ram)

    for candidates, min_samples in (
        (desc_candidates, 1),
        (prefix_candidates, 1),
        (project_candidates, 2),
    ):
        if len(candidates) >= min_samples:
            return _median_or_zero(candidates)
    return 0


def with_resource_slack(observed_mb: int, *, min_mb: int) -> int:
    observed = int(observed_mb or 0)
    if observed <= 0:
        return 0
    return max(int(min_mb), (observed * 6 + 4) // 5)


def maybe_lower_explicit_resource_estimate(
    task: dict,
    key: str,
    observed_mb: int,
    *,
    min_mb: int,
    kind: str,
    now: Callable[[], float] = _time.time,
) -> bool:
    cur = int(task.get(key) or 0)
    proposed = with_resource_slack(observed_mb, min_mb=min_mb)
    if cur <= 0 or proposed <= 0:
        return False
    if proposed >= cur:
        return False
    if proposed * 100 > cur * 85:
        return False
    task[key] = proposed
    update = {"ts": now(), "kind": kind, f"old_{key}": cur, f"new_{key}": proposed}
    if int(observed_mb or 0) > 0:
        update["observed_mb"] = int(observed_mb)
    task["last_resource_estimate_update"] = update
    return True


def _record_observed_estimate_update(
    task: dict,
    key: str,
    old_value: int,
    new_value: int,
    observed_mb: int,
    kind: str,
    *,
    now: Callable[[], float],
) -> None:
    task["last_resource_estimate_update"] = {
        "ts": now(),
        "kind": kind,
        f"old_{key}": old_value,
        f"new_{key}": new_value,
        "observed_mb": int(observed_mb),
    }


def _apply_live_family_estimate(
    task: dict,
    key: str,
    observed_mb: int,
    *,
    min_mb: int,
    explicit: bool,
    kind: str,
    deps: ResourceEstimateDeps,
) -> bool:
    """Track a live family with slack while avoiding small estimate churn."""
    observed_mb = int(observed_mb or 0)
    cur = int(task.get(key) or 0)
    if key == "est_vram_mb" and explicit and cur == 0:
        return False
    if observed_mb <= 0:
        return False
    if explicit and deps.maybe_lower_explicit_resource_estimate(
        task,
        key,
        observed_mb,
        min_mb=min_mb,
        kind=f"{kind}_lower",
    ):
        return True

    proposed = with_resource_slack(observed_mb, min_mb=min_mb)
    if proposed <= 0:
        return False
    if cur > 0:
        if proposed < cur:
            # A live sibling is stronger evidence than a submit-time/default
            # estimate. Avoid rewriting only when the difference is too small
            # to affect placement.
            if proposed * 100 > cur * 85:
                return False
        elif proposed <= cur * 1.10:
            return False
    task[key] = proposed
    _record_observed_estimate_update(
        task,
        key,
        cur,
        proposed,
        observed_mb,
        (
            f"{kind}_initialize"
            if not cur
            else f"{kind}_lower"
            if proposed < cur
            else f"{kind}_raise"
        ),
        now=deps.now,
    )
    return True


def _apply_learned_fallback_estimate(
    task: dict,
    key: str,
    evidence: dict | None,
    *,
    min_mb: int,
    explicit: bool,
    kind: str,
    deps: ResourceEstimateDeps,
) -> bool:
    """Lower a cold-start estimate only when indexed actual evidence is lower."""
    if not evidence:
        return False
    observed_mb = int(evidence.get("observed_mb") or 0)
    cur = int(task.get(key) or 0)
    proposed = with_resource_slack(observed_mb, min_mb=min_mb)
    if cur <= 0 or proposed <= 0 or proposed >= cur:
        return False
    changed = _apply_live_family_estimate(
        task,
        key,
        observed_mb,
        min_mb=min_mb,
        explicit=explicit,
        kind=kind,
        deps=deps,
    )
    if changed and isinstance(task.get("last_resource_estimate_update"), dict):
        task["last_resource_estimate_update"].update({
            "evidence_scope": evidence.get("scope") or "",
            "sample_count": int(evidence.get("sample_count") or 0),
        })
    return changed

def refresh_queued_resource_estimates(
    state: dict,
    history_cache: dict,
    *,
    deps: ResourceEstimateDeps,
) -> None:
    """Refresh queued resource budgets from exact history and live sibling evidence."""
    live_estimate_index = deps.live_sibling_resource_estimate_index(state)
    description_estimate_index = deps.description_sibling_resource_estimate_index(state)
    learned_estimate_index = learned_resource_estimate_index(state, history_cache)

    def indexed_live_estimate(task: dict, kind: str) -> int:
        return int(live_estimate_index.get((resource_family_key(task, kind), kind), 0))

    def indexed_description_estimate(task: dict, kind: str) -> int:
        mode = _task_resource_mode(task) if kind == "ram" else "vram"
        key = (description_resource_family_key(task), mode, kind)
        return int(description_estimate_index.get(key, 0))

    def indexed_learned_estimate(task: dict, kind: str) -> dict | None:
        mode = _task_resource_mode(task) if kind == "vram" else ""
        candidates = [("family", resource_family_key(task, kind))]
        # A caller-supplied family names a specific implementation/runtime
        # contract. Falling back to signature-prefix or project evidence would
        # leak measurements from older code into that new contract.
        if not _explicit_resource_family(task, kind):
            candidates.extend([
                ("prefix", _signature_prefix(task.get("signature") or "")),
                ("project", str(task.get("project") or "").strip().lower()),
            ])
        for scope, key in candidates:
            evidence = learned_estimate_index.get((scope, key, mode, kind))
            if evidence:
                return evidence
        return None

    for t in state["tasks"]:
        if t.get("status") != "queued":
            continue
        sig = t.get("signature") or ""
        h = history_cache.get(sig)
        if isinstance(h, int):
            h = {"vram_mb": h}
        exact_vram = (
            _history_vram(h)
            if _exact_history_matches_resource_family(t, h, "vram")
            else 0
        )
        live_vram = indexed_live_estimate(t, "vram")
        description_vram = indexed_description_estimate(t, "vram")
        learned_vram = indexed_learned_estimate(t, "vram")
        # A live task from this exact resource family reflects the current
        # code/configuration and therefore supersedes older history.
        if live_vram:
            _apply_live_family_estimate(
                t,
                "est_vram_mb",
                live_vram,
                min_mb=256,
                explicit=bool(t.get("est_vram_mb_explicit")),
                kind="vram_live_family",
                deps=deps,
            )
        elif exact_vram:
            new_est = exact_vram
            if t.get("est_vram_mb_explicit"):
                _apply_live_family_estimate(
                    t,
                    "est_vram_mb",
                    new_est,
                    min_mb=256,
                    explicit=True,
                    kind="vram_exact_history",
                    deps=deps,
                )
            elif new_est != t.get("est_vram_mb"):
                t["est_vram_mb"] = new_est
        elif learned_vram and learned_vram.get("scope") == "family":
            _apply_learned_fallback_estimate(
                t,
                "est_vram_mb",
                learned_vram,
                min_mb=256,
                explicit=bool(t.get("est_vram_mb_explicit")),
                kind="vram_learned_family",
                deps=deps,
            )
        elif description_vram and not _explicit_resource_family(t, "vram"):
            _apply_live_family_estimate(
                t,
                "est_vram_mb",
                description_vram,
                min_mb=256,
                explicit=bool(t.get("est_vram_mb_explicit")),
                kind="vram_description_sibling",
                deps=deps,
            )
        elif learned_vram:
            _apply_learned_fallback_estimate(
                t,
                "est_vram_mb",
                learned_vram,
                min_mb=256,
                explicit=bool(t.get("est_vram_mb_explicit")),
                kind=f"vram_learned_{learned_vram.get('scope') or 'fallback'}",
                deps=deps,
            )
        else:
            cur = int(t.get("est_vram_mb") or 0)
            # This fallback only lowers an existing non-explicit estimate.  A
            # zero estimate cannot be lowered, and calling the broad sibling
            # scan for thousands of CPU tasks was pure O(queue^2) work.
            if not t.get("est_vram_mb_explicit") and cur > 0:
                new_est = deps.effective_est_vram(t, state, history_cache)
                if new_est and 0 < new_est < cur:
                    t["est_vram_mb"] = new_est

        exact_ram = (
            _history_ram(h)
            if _exact_history_matches_resource_family(t, h, "ram")
            else 0
        )
        live_ram_family = indexed_live_estimate(t, "ram")
        description_ram = indexed_description_estimate(t, "ram")
        learned_ram = indexed_learned_estimate(t, "ram")
        if live_ram_family:
            _apply_live_family_estimate(
                t,
                "ram_mb",
                live_ram_family,
                min_mb=512,
                explicit=bool(t.get("ram_mb_explicit")),
                kind="ram_live_family",
                deps=deps,
            )
        elif exact_ram:
            if t.get("ram_mb_explicit"):
                _apply_live_family_estimate(
                    t,
                    "ram_mb",
                    exact_ram,
                    min_mb=512,
                    explicit=True,
                    kind="ram_exact_history",
                    deps=deps,
                )
            elif exact_ram != t.get("ram_mb"):
                t["ram_mb"] = exact_ram
        elif learned_ram and learned_ram.get("scope") == "family":
            _apply_learned_fallback_estimate(
                t,
                "ram_mb",
                learned_ram,
                min_mb=512,
                explicit=bool(t.get("ram_mb_explicit")),
                kind="ram_learned_family",
                deps=deps,
            )
        elif description_ram and not _explicit_resource_family(t, "ram"):
            _apply_live_family_estimate(
                t,
                "ram_mb",
                description_ram,
                min_mb=512,
                explicit=bool(t.get("ram_mb_explicit")),
                kind="ram_description_sibling",
                deps=deps,
            )
        elif learned_ram:
            _apply_learned_fallback_estimate(
                t,
                "ram_mb",
                learned_ram,
                min_mb=512,
                explicit=bool(t.get("ram_mb_explicit")),
                kind=f"ram_learned_{learned_ram.get('scope') or 'fallback'}",
                deps=deps,
            )
        elif not t.get("ram_mb_explicit"):
            live_ram = deps.live_sibling_ram_floor(t, state)
            cur_ram = int(t.get("ram_mb") or 0)
            if live_ram and live_ram > cur_ram * 1.1:
                t["ram_mb"] = live_ram
                t["last_resource_estimate_update"] = {
                    "ts": deps.now(),
                    "kind": "ram_live_sibling_floor",
                    "old_ram_mb": cur_ram,
                    "new_ram_mb": live_ram,
                }
                continue
            if cur_ram > 0:
                new_ram = deps.effective_est_ram(t, state, history_cache)
                if new_ram and 0 < new_ram < cur_ram:
                    t["ram_mb"] = new_ram
