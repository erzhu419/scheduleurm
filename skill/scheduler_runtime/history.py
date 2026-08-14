"""Resource history and runtime-history lookup helpers."""

from __future__ import annotations

import os
import re
import shlex
import time as _time
from dataclasses import dataclass
from typing import Any, Callable, Optional


def raise_system_exit(message: str):
    raise SystemExit(message)


@dataclass(frozen=True)
class ResourceHistoryDeps:
    load_history: Callable[[], dict]
    save_history: Callable[[dict], Any]
    max_entries: int
    samples_per_sig: int
    percentile_value: int
    now: Callable[[], float] = _time.time


@dataclass(frozen=True)
class HistoryCommandDeps:
    load_history: Callable[[], dict]
    save_history: Callable[[dict], Any]
    now: Callable[[], float] = _time.time
    print_fn: Callable[..., Any] = print
    exit_fn: Callable[[str], Any] = raise_system_exit


@dataclass(frozen=True)
class RuntimeHistoryLookupDeps:
    load_runtime_history: Callable[[], dict]
    history_get: Callable[[str], dict | None]
    task_runtime_keys: Callable[[dict], list]
    runtime_total_units_from_cmd: Callable[[str], int]
    queued_has_stale_live_eta: Callable[[dict], bool]
    clear_live_eta_fields: Callable[..., bool]
    eta_confidence_for_source: Callable[[str], str]
    min_walltime_s: int
    walltime_mult: float
    closest_min_score: float
    now: Callable[[], float] = _time.time


def history_get(sig: str, *, load_history: Callable[[], dict]) -> Optional[dict]:
    """Look up resource history for a signature, migrating legacy int records."""
    if not sig:
        return None
    raw = load_history().get(sig)
    if raw is None:
        return None
    if isinstance(raw, int):
        return {"vram_mb": raw}
    return raw


def task_duration_s(task: dict) -> int:
    if task.get("started_at") and task.get("finished_at"):
        return max(0, int(task["finished_at"] - task["started_at"]))
    return 0


def task_has_progress_marker(task: dict) -> bool:
    """True once a task has made train/eval progress, not just initialized."""
    if task.get("runtime_current_unit") or task.get("progress_ratio"):
        return True
    line = task.get("last_progress_line") or ""
    return bool(re.search(r"\bIter\s+\d+\b|\[\s*\d+\s*/\s*\d+\]|Epoch\s+\d+|%\|", line))


def is_oom_like_task(task: dict) -> bool:
    if task.get("failure_category") == "OOM":
        return True
    text = (
        f"{task.get('last_block_reason') or ''}\n"
        f"{(task.get('_diagnosis') or {}).get('reason') or ''}\n"
        f"{(task.get('_diagnosis') or {}).get('tail') or ''}"
    ).lower()
    return "out of memory" in text or "resource_exhausted" in text


def untrusted_startup_oom_sample(task: dict, duration_s: int | None = None) -> bool:
    """Short OOM before progress is usually startup/preallocation, not steady-state need."""
    if task.get("status") != "failed":
        return False
    if not is_oom_like_task(task):
        return False
    if task_has_progress_marker(task):
        return False
    duration_s = task_duration_s(task) if duration_s is None else duration_s
    return 0 < duration_s <= 5 * 60


def percentile(samples: list, p: int) -> int:
    """p-th percentile (0..100) of samples. Returns 0 for an empty list."""
    if not samples:
        return 0
    sorted_samples = sorted(samples)
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = (p / 100.0) * (len(sorted_samples) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    return int(sorted_samples[lo] + (rank - lo) * (sorted_samples[hi] - sorted_samples[lo]))


def history_record(
    sig: str,
    peak_vram_mb: int = 0,
    peak_ram_mb: int = 0,
    cpu_cores: int = 0,
    duration_s: int = 0,
    metadata: dict | None = None,
    *,
    deps: ResourceHistoryDeps,
) -> None:
    """Fold new peak samples and EWMA duration into resource history for a signature."""
    if not sig:
        return
    history = deps.load_history()
    cur = history.get(sig)
    if isinstance(cur, int):
        cur = {"vram_mb": cur}
    elif cur is None or not isinstance(cur, dict):
        cur = {}

    if isinstance(metadata, dict):
        for key in (
            "project",
            "description_resource_family_key",
            "ram_resource_family_key",
            "vram_resource_family_key",
            "resource_mode",
            "vram_observation_scope",
        ):
            value = metadata.get(key)
            if value:
                cur[key] = str(value)


    def fold(field_name: str, samples_field: str, new_sample: int) -> None:
        samples = cur.get(samples_field) or []
        if not samples and cur.get(field_name):
            samples = [int(cur[field_name])]
        samples.append(int(new_sample))
        samples = samples[-deps.samples_per_sig:]
        cur[samples_field] = samples
        cur[field_name] = percentile(samples, deps.percentile_value)

    if peak_vram_mb > 0:
        fold("vram_mb", "vram_samples", peak_vram_mb)
    if peak_ram_mb > 0:
        fold("ram_mb", "ram_samples", peak_ram_mb)
    if cpu_cores > 0:
        cur["cpu_cores"] = max(cur.get("cpu_cores", 0), cpu_cores)
    if duration_s > 0:
        prev = cur.get("dur_s_ewma", 0)
        cur["dur_s_ewma"] = int(0.7 * prev + 0.3 * duration_s) if prev else int(duration_s)
        cur["dur_s_runs"] = cur.get("dur_s_runs", 0) + 1
    cur["last_seen"] = int(deps.now())
    history[sig] = cur
    if len(history) > deps.max_entries:
        kept = sorted(
            history.items(),
            key=lambda kv: -(kv[1].get("last_seen", 0) if isinstance(kv[1], dict) else 0),
        )
        history = dict(kept[:deps.max_entries])
    deps.save_history(history)


def _runtime_cmd_argv(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd or "")
    except Exception:
        return (cmd or "").split()


def _runtime_cmd_tokens_from_argv(toks: list[str]) -> set:
    out = set()
    skip_next_for = {
        "--seed", "--current_time", "--current-time", "--run_name", "--run-name",
        "--output", "--out", "--log_dir", "--log-dir",
    }
    skip_next = False
    for tok in toks:
        if skip_next:
            skip_next = False
            continue
        if tok in skip_next_for:
            out.add(tok)
            skip_next = True
            continue
        if any(tok.startswith(flag + "=") for flag in skip_next_for):
            out.add(tok.split("=", 1)[0])
            continue
        base = os.path.basename(tok)
        if base in ("python", "python3", "python3.10", "python3.11", "python3.12", "bash", "sh"):
            continue
        norm = re.sub(r"\d+", "<N>", tok.lower())
        if len(norm) >= 2:
            out.add(norm)
    return out


def runtime_cmd_tokens(cmd: str) -> set:
    return _runtime_cmd_tokens_from_argv(_runtime_cmd_argv(cmd))


def _runtime_script_name_from_argv(toks: list[str]) -> str:
    for index, tok in enumerate(toks):
        if tok == "-m" and index + 1 < len(toks):
            return f"-m:{str(toks[index + 1]).strip().lower()}"
        if tok.endswith((".py", ".sh")) or ".py:" in tok:
            return os.path.basename(tok)
    return ""


def runtime_script_name(cmd: str) -> str:
    return _runtime_script_name_from_argv(_runtime_cmd_argv(cmd))


_RUNTIME_IDENTITY_FLAGS = frozenset({
    "--method", "--algorithm", "--algo", "--baseline", "--baselines",
    "--implementation", "--protocol", "--heldout", "--problem",
    "--family", "--env", "--profile", "--phase", "--mode", "--variant",
    "--experiment-variant", "--policy-variant", "--line",
})

_RUNTIME_IDENTITY_SWITCHES = frozenset({
    "--audit-only",
    "--dry-run",
    "--finalize-existing",
    "--no-train",
    "--smoke",
    "--validate-only",
})

_KG_LODO_SCRIPT = "run_lodo_manifest_shard.py"
_BAPR_INDEPENDENT_SPECIALIST_MODULE = (
    "-m:jax_experiments.analysis."
    "launch_bapr_v3_stochastic_independent_specialist"
)

# These fields name a run within a workload family; they do not change the
# runtime shape for the listed entrypoint. All other identity fields remain
# hard constraints, so history never crosses heldout problem, family, or env.
_RUNTIME_SOFT_IDENTITY_FLAGS_BY_SCRIPT = {
    _KG_LODO_SCRIPT: frozenset({"--experiment-variant"}),
    _BAPR_INDEPENDENT_SPECIALIST_MODULE: frozenset({"--mode"}),
}

_RUNTIME_BUDGET_FLAGS_BY_SCRIPT = {
    _KG_LODO_SCRIPT: ("--N", "--n0", "--d", "--meta-source-d"),
    _BAPR_INDEPENDENT_SPECIALIST_MODULE: ("--target-next-iteration",),
}


def _runtime_cmd_identity_from_argv(tokens: list[str]) -> dict[str, str]:
    identity = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-m" and index + 1 < len(tokens):
            identity["__entrypoint__"] = f"module:{str(tokens[index + 1]).strip().lower()}"
            index += 2
            continue
        if token.endswith((".py", ".sh")) or ".py:" in token:
            identity.setdefault("__entrypoint__", f"script:{os.path.basename(token).lower()}")
        if token in _RUNTIME_IDENTITY_SWITCHES:
            identity[f"__switch__:{token}"] = "1"
        if token in _RUNTIME_IDENTITY_FLAGS and index + 1 < len(tokens):
            identity[token] = str(tokens[index + 1]).strip().lower()
            index += 2
            continue
        flag, separator, value = token.partition("=")
        if separator and flag in _RUNTIME_IDENTITY_FLAGS:
            identity[flag] = value.strip().lower()
        index += 1
    return identity


def runtime_cmd_identity(cmd: str) -> dict[str, str]:
    """Extract semantic workload fields that must not cross-match in ETA."""
    return _runtime_cmd_identity_from_argv(_runtime_cmd_argv(cmd))


def _runtime_identities_compatible(left: dict, right: dict) -> bool:
    if ("__entrypoint__" in left) != ("__entrypoint__" in right):
        return False
    for flag in (*_RUNTIME_IDENTITY_FLAGS, "__entrypoint__"):
        if flag in left and flag in right and left[flag] != right[flag]:
            return False
    for switch in _RUNTIME_IDENTITY_SWITCHES:
        key = f"__switch__:{switch}"
        if (key in left) != (key in right):
            return False
    return True


def _runtime_identities_compatible_for_scripts(
    left: dict,
    right: dict,
    left_script: str,
    right_script: str,
) -> bool:
    if left_script != right_script:
        return _runtime_identities_compatible(left, right)
    ignored = _RUNTIME_SOFT_IDENTITY_FLAGS_BY_SCRIPT.get(left_script)
    if not ignored:
        return _runtime_identities_compatible(left, right)
    return _runtime_identities_compatible(
        {key: value for key, value in left.items() if key not in ignored},
        {key: value for key, value in right.items() if key not in ignored},
    )


def _runtime_cmd_budget_identity_from_argv(
    tokens: list[str], script: str
) -> tuple:
    flags = _RUNTIME_BUDGET_FLAGS_BY_SCRIPT.get(script)
    if not flags:
        return ()
    flag_set = frozenset(flags)
    values = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in flag_set and index + 1 < len(tokens):
            values[token] = str(tokens[index + 1]).strip().lower()
            index += 2
            continue
        flag, separator, value = token.partition("=")
        if separator and flag in flag_set:
            values[flag] = value.strip().lower()
        index += 1
    return tuple((flag, values[flag]) for flag in flags if flag in values)


def runtime_cmd_budget_identity(cmd: str, script: str = "") -> tuple:
    """Return exact workload-size fields used to rank otherwise-near history."""
    tokens = _runtime_cmd_argv(cmd)
    script = script or _runtime_script_name_from_argv(tokens)
    return _runtime_cmd_budget_identity_from_argv(tokens, script)


def runtime_history_closest_index(history: dict) -> list[dict]:
    rows = []
    command_features = {}
    for key, rec in (history or {}).items():
        if not isinstance(rec, dict) or int(rec.get("total_s") or 0) <= 0:
            continue
        rec_cmd = rec.get("cmd") or ""
        features = command_features.get(rec_cmd)
        if features is None:
            rec_argv = _runtime_cmd_argv(rec_cmd)
            rec_tokens = _runtime_cmd_tokens_from_argv(rec_argv)
            rec_script = _runtime_script_name_from_argv(rec_argv)
            features = (
                rec_tokens,
                rec_script,
                _runtime_cmd_identity_from_argv(rec_argv),
                _runtime_cmd_budget_identity_from_argv(rec_argv, rec_script),
            )
            command_features[rec_cmd] = features
        rec_tokens, rec_script, rec_identity, rec_budget_identity = features
        if not rec_tokens:
            continue
        rows.append({
            "key": key,
            "rec": rec,
            "tokens": rec_tokens,
            "script": rec_script,
            "identity": rec_identity,
            "budget_identity": rec_budget_identity,
            "project": rec.get("project") or "",
            "cwd_base": os.path.basename(str(rec.get("cwd") or "")),
        })
    return rows


def _runtime_record_units(rec: dict, deps: RuntimeHistoryLookupDeps) -> int:
    try:
        cmd_units = int(deps.runtime_total_units_from_cmd(rec.get("cmd") or "") or 0)
    except Exception:
        cmd_units = 0
    if cmd_units > 0:
        return cmd_units
    try:
        units = int(rec.get("total_units") or 0)
    except Exception:
        units = 0
    if units > 0:
        return units
    return 0


def _runtime_record_unit_s(rec: dict, rec_units: int) -> float:
    try:
        declared_units = int(rec.get("total_units") or 0)
    except Exception:
        declared_units = 0
    try:
        unit_s = float(rec.get("unit_s") or 0)
    except Exception:
        unit_s = 0.0
    if unit_s > 0 and (declared_units <= 0 or declared_units == rec_units):
        return unit_s
    try:
        total_s = float(rec.get("total_s") or 0)
    except Exception:
        total_s = 0.0
    if total_s > 0 and rec_units > 0:
        return total_s / float(rec_units)
    return 0.0


def runtime_history_closest(
    task: dict,
    history: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
    deps: RuntimeHistoryLookupDeps,
    closest_index: Optional[list[dict]] = None,
):
    """Return a conservative closest runtime-history record for novel signatures."""
    payload = task_runtime_payload(task)
    task_cmd = payload.get("cmd") or ""
    task_argv = _runtime_cmd_argv(task_cmd)
    task_tokens = _runtime_cmd_tokens_from_argv(task_argv)
    if not task_tokens:
        return None, None
    task_script = _runtime_script_name_from_argv(task_argv)
    task_identity = _runtime_cmd_identity_from_argv(task_argv)
    task_budget_identity = _runtime_cmd_budget_identity_from_argv(
        task_argv, task_script)
    task_project = payload.get("project") or ""
    task_cwd_base = os.path.basename(payload.get("cwd") or "")
    task_units = deps.runtime_total_units_from_cmd(payload.get("cmd") or "")
    best = None
    records = closest_index if closest_index is not None else runtime_history_closest_index(history)
    for item in records:
        key = item.get("key")
        rec = item.get("rec") or {}
        rec_identity = item.get("identity") or {}
        rec_script = item.get("script") or ""
        if not _runtime_identities_compatible_for_scripts(
            task_identity, rec_identity, task_script, rec_script
        ):
            continue
        rec_tokens = item.get("tokens") or set()
        inter = len(task_tokens & rec_tokens)
        union = len(task_tokens | rec_tokens) or 1
        score = inter / union
        if task_script and rec_script and task_script == rec_script:
            score += 0.25
        shared_identity = {
            key for key in set(task_identity) & set(rec_identity)
            if task_identity[key] == rec_identity[key]
        }
        score += 0.05 * len(shared_identity)
        rec_budget_identity = item.get("budget_identity") or ()
        if task_budget_identity and rec_budget_identity:
            if task_budget_identity == rec_budget_identity:
                score += 0.20
            else:
                score -= 0.20
        if task_project and item.get("project") == task_project:
            score += 0.15
        rec_cwd_base = item.get("cwd_base") or ""
        if task_cwd_base and rec_cwd_base and task_cwd_base == rec_cwd_base:
            score += 0.10
        if score < deps.closest_min_score:
            continue
        try:
            last_seen = int(rec.get("last_seen") or 0)
        except Exception:
            last_seen = 0
        if best is None or (score, last_seen) > (best[0], best[1]):
            best = (score, last_seen, key, rec)
    if not best:
        return None, None
    score, _last_seen, key, rec = best
    out = dict(rec)
    out["source"] = f"closest:{key}:score={score:.2f}"
    rec_units = _runtime_record_units(rec, deps)
    unit_s = _runtime_record_unit_s(rec, rec_units)
    if task_units > 0 and unit_s > 0:
        out["total_s"] = int(unit_s * float(task_units))
        out["walltime_s"] = int(max(deps.min_walltime_s, out["total_s"] * deps.walltime_mult))
        out["total_units"] = task_units
        out["unit_s"] = unit_s
        if rec_units > 0:
            out["scaled_from_total_units"] = rec_units
    return out, f"closest:{key}"


def runtime_history_best(
    task: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
    deps: RuntimeHistoryLookupDeps,
    runtime_history: Optional[dict] = None,
    closest_index: Optional[list[dict]] = None,
):
    history = runtime_history if runtime_history is not None else deps.load_runtime_history()
    for key, kind, _payload in deps.task_runtime_keys(task):
        rec = history.get(key)
        if isinstance(rec, dict) and int(rec.get("total_s") or 0) > 0:
            return rec, key, kind
    rec, key = runtime_history_closest(
        task,
        history,
        task_runtime_payload=task_runtime_payload,
        deps=deps,
        closest_index=closest_index,
    )
    if rec:
        return rec, key, "closest"
    return None, None, None


def runtime_total_history_s(
    task: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
    deps: RuntimeHistoryLookupDeps,
    runtime_history: Optional[dict] = None,
    closest_index: Optional[list[dict]] = None,
) -> int:
    rec, _key, _kind = runtime_history_best(
        task,
        task_runtime_payload=task_runtime_payload,
        deps=deps,
        runtime_history=runtime_history,
        closest_index=closest_index,
    )
    return int(rec.get("total_s") or 0) if rec else 0


def history_eta_for_task(
    task: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
    deps: RuntimeHistoryLookupDeps,
    runtime_history: Optional[dict] = None,
    closest_index: Optional[list[dict]] = None,
    resource_history: Optional[dict] = None,
) -> tuple[int, str]:
    """Return a full-run ETA estimate for a queued or launching task."""
    total_s = int(task.get("runtime_total_s_est") or 0)
    if total_s > 0:
        return total_s, task.get("runtime_est_source") or "runtime_profile"
    rec, _key, _kind = runtime_history_best(
        task,
        task_runtime_payload=task_runtime_payload,
        deps=deps,
        runtime_history=runtime_history,
        closest_index=closest_index,
    )
    total_s = int(rec.get("total_s") or 0) if rec else 0
    if total_s > 0:
        source = "peer_progress" if rec.get("live_peer_task_id") else "runtime_history"
        return total_s, source
    sig = task.get("signature") or ""
    if resource_history is not None:
        hist = resource_history.get(sig) or {}
        if isinstance(hist, int):
            hist = {"vram_mb": hist}
    else:
        hist = deps.history_get(sig) or {}
    total_s = int(hist.get("dur_s_ewma") or 0)
    if total_s > 0:
        return total_s, "duration_ewma"
    return 0, ""


_ACTIVE_PEER_ETA_MAX_AGE_S = 5 * 60


def _active_runtime_projection_history(
    state: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
    now: float,
) -> dict:
    """Build ephemeral runtime records from fresh, progressing sibling tasks."""
    records = {}
    for peer in state.get("tasks", []):
        if peer.get("status") != "running":
            continue
        try:
            total_s = int(peer.get("runtime_total_s_est") or 0)
            current = int(peer.get("runtime_current_unit") or 0)
            total_units = int(peer.get("runtime_total_units") or 0)
            unit_s = float(peer.get("runtime_unit_s_est") or 0.0)
            updated_at = float(
                peer.get("runtime_progress_at")
                or peer.get("eta_updated_at")
                or 0.0
            )
        except (TypeError, ValueError):
            continue
        if total_s <= 0 or current <= 0 or total_units <= 0 or unit_s <= 0:
            continue
        if updated_at <= 0 or now - updated_at > _ACTIVE_PEER_ETA_MAX_AGE_S:
            continue
        payload = task_runtime_payload(peer)
        records[f"peer:{peer.get('id') or len(records)}"] = {
            **payload,
            "total_s": total_s,
            "total_units": total_units,
            "unit_s": unit_s,
            "source": peer.get("runtime_est_source") or "progress",
            # Completed history wins score ties; this is only a no-history fallback.
            "last_seen": 0,
            "live_peer_task_id": str(peer.get("id") or ""),
        }
    return records


def seed_pending_eta_from_history(
    state: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
    deps: RuntimeHistoryLookupDeps,
    runtime_history_cache: Optional[dict] = None,
    runtime_closest_index: Optional[list[dict]] = None,
    resource_history_cache: Optional[dict] = None,
) -> int:
    """Fill queued/launching ETA from local-test profiles, runtime history, or duration EWMA."""
    changed = 0
    candidates = []
    for task in state.get("tasks", []):
        if task.get("status") not in ("queued", "launching"):
            continue
        if deps.queued_has_stale_live_eta(task):
            if deps.clear_live_eta_fields(task, clear_runtime_projection=True):
                task["eta_detail"] = "cleared stale live ETA after task returned to queue"
                changed += 1
        if int(task.get("eta_seconds") or 0) <= 0:
            candidates.append(task)

    # The common watcher path already has ETA on every pending task. Avoid
    # loading and tokenizing the entire runtime-history index while holding the
    # state lock when there is no lookup work to perform.
    if not candidates:
        return changed
    if runtime_history_cache is None:
        runtime_history_cache = deps.load_runtime_history()
    peer_history = _active_runtime_projection_history(
        state,
        task_runtime_payload=task_runtime_payload,
        now=float(deps.now()),
    )
    if peer_history:
        runtime_history_cache = {**peer_history, **runtime_history_cache}
        runtime_closest_index = None
    if runtime_closest_index is None:
        runtime_closest_index = runtime_history_closest_index(runtime_history_cache)
    for task in candidates:
        eta, source = history_eta_for_task(
            task,
            task_runtime_payload=task_runtime_payload,
            deps=deps,
            runtime_history=runtime_history_cache,
            closest_index=runtime_closest_index,
            resource_history=resource_history_cache,
        )
        if eta <= 0:
            continue
        task["eta_seconds"] = int(eta)
        task["eta_source"] = source
        task["eta_confidence"] = deps.eta_confidence_for_source(source)
        task["eta_updated_at"] = int(deps.now())
        task["eta_detail"] = f"queued ETA seeded from {source}"
        changed += 1
    return changed


def runtime_walltime_for_task(
    task: dict,
    *,
    task_runtime_payload: Callable[[dict], dict],
    deps: RuntimeHistoryLookupDeps,
) -> int:
    total_s = runtime_total_history_s(task, task_runtime_payload=task_runtime_payload, deps=deps)
    if total_s <= 0:
        return 0
    return int(max(deps.min_walltime_s, total_s * deps.walltime_mult))


def cmd_history(args, *, deps: HistoryCommandDeps) -> None:
    """Show or edit resource history per signature."""
    if getattr(args, "drop", None):
        history = deps.load_history()
        if args.drop not in history:
            return deps.exit_fn(f"signature {args.drop!r} not in history")
        old = history.pop(args.drop)
        deps.save_history(history)
        deps.print_fn(f"dropped {args.drop!r}:")
        deps.print_fn(f"  was: {old}")
        deps.print_fn("  next runs of this signature will accumulate fresh peaks.")
        return
    if getattr(args, "set", None):
        if all(getattr(args, key, None) is None for key in ("vram_mb", "ram_mb", "cpu")):
            return deps.exit_fn("--set requires at least one of --vram-mb / --ram-mb / --cpu")
        history = deps.load_history()
        rec = history.get(args.set, {})
        if isinstance(rec, int):
            rec = {"vram_mb": rec}
        elif not isinstance(rec, dict):
            rec = {}
        if args.vram_mb is not None:
            rec["vram_mb"] = int(args.vram_mb)
            rec["vram_samples"] = [int(args.vram_mb)]
        if args.ram_mb is not None:
            rec["ram_mb"] = int(args.ram_mb)
            rec["ram_samples"] = [int(args.ram_mb)]
        if args.cpu is not None:
            rec["cpu_cores"] = int(args.cpu)
        rec["last_seen"] = int(deps.now())
        history[args.set] = rec
        deps.save_history(history)
        deps.print_fn(f"set {args.set!r}:")
        deps.print_fn(f"  {rec}")
        return
    history = deps.load_history()
    if not history:
        deps.print_fn("(no resource history yet - runs will record automatically as they finish)")
        return
    rows = []
    for sig, raw in history.items():
        if isinstance(raw, int):
            raw = {"vram_mb": raw}
        rows.append((sig, raw.get("vram_mb", 0), raw.get("ram_mb", 0), raw.get("cpu_cores", 0)))
    rows.sort(key=lambda row: -row[1])
    deps.print_fn(f"  {'signature':<40s} {'vram':>8s} {'ram':>10s} {'cpu':>5s}")
    for sig, vram, ram, cpu in rows:
        deps.print_fn(f"  {sig:<40s} {vram:>6}MB {ram:>8}MB {cpu:>5}")
