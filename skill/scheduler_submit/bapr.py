"""BAPR-specific seed batch and result projection helpers."""

from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from .shell import (
        expand_simple_seed_loop_inner,
        seed_value_substitute,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .shell import (
        expand_simple_seed_loop_inner,
        seed_value_substitute,
    )


def submit_context_is_bapr(cmd: str = "", cwd: str = "", signature: str = "",
                           project: str = "") -> bool:
    hay = " ".join(str(x or "") for x in (cmd, cwd, signature, project))
    return (
        str(signature or "").startswith("BAPR/")
        or str(project or "") == "BAPR"
        or "/BAPR" in hay
        or "/bapr_v15" in hay
    )


def split_bapr_seed_batch_submit_args(args):
    if getattr(args, "allow_seed_batch", False):
        return []
    if not submit_context_is_bapr(
        getattr(args, "cmd", ""),
        getattr(args, "cwd", ""),
        getattr(args, "signature", ""),
        getattr(args, "project", ""),
    ):
        return []
    try:
        toks = shlex.split(getattr(args, "cmd", "") or "")
    except Exception:
        return []
    if len(toks) < 3 or os.path.basename(toks[0]) not in ("bash", "sh", "zsh", "dash"):
        return []
    c_i = None
    for i, tok in enumerate(toks[1:], 1):
        if tok in ("-c", "-lc"):
            c_i = i
            break
    if c_i is None or c_i + 1 >= len(toks):
        return []
    expanded_inners = expand_simple_seed_loop_inner(toks[c_i + 1])
    if not expanded_inners:
        return []
    result = []
    for seed, new_inner in expanded_inners:
        vals = vars(args).copy()
        new_toks = list(toks)
        new_toks[c_i + 1] = new_inner
        vals["cmd"] = " ".join(shlex.quote(t) for t in new_toks)
        sig = str(getattr(args, "signature", "") or "").rstrip("/")
        vals["signature"] = f"{sig}/s{seed}" if sig else f"s{seed}"
        desc = str(getattr(args, "description", "") or "").strip()
        vals["description"] = f"{desc} seed {seed}".strip()
        for key in ("ckpt_dir", "result_dir", "local_result_dir"):
            if key in vals and vals[key]:
                vals[key] = seed_value_substitute(vals[key], seed)
        if vals.get("wait_for_files"):
            vals["wait_for_files"] = seed_value_substitute(vals["wait_for_files"], seed)
        result.append(argparse.Namespace(**vals))
    return result


def infer_bapr_run_seed_ckpt(cmd: str, cwd: str = ""):
    candidates = []
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = []
    if toks:
        candidates.append(toks)
        for i, tok in enumerate(toks[:-1]):
            if tok in ("-c", "-lc"):
                try:
                    candidates.append(shlex.split(toks[i + 1]))
                except Exception:
                    pass
    for ctoks in candidates:
        for i, tok in enumerate(ctoks):
            if not tok.endswith("run_seed.sh"):
                continue
            args = ctoks[i + 1:]
            if args and os.path.basename(tok) == "run_seed.sh":
                script_path = tok
            else:
                continue
            if len(args) < 3:
                continue
            algo, env_name, seed = args[0], args[1], args[2]
            dwell = args[4] if len(args) >= 5 and args[4] else "60"
            tag = args[6] if len(args) >= 7 and args[6] else "paper"
            if any(("$" in str(x) or ";" in str(x)) for x in (algo, env_name, seed, dwell, tag)):
                continue
            env_short = env_name[:-3] if env_name.endswith("-v2") else env_name
            run_name = f"{tag}_{algo}_{env_short}_dw{dwell}_s{seed}"
            base = os.path.dirname(script_path) if os.path.isabs(script_path) else (cwd or os.getcwd())
            result_dir = os.path.join(base, "jax_experiments", "results_paper", run_name)
            return {
                "result_dir": result_dir,
                "ckpt_dir": os.path.join(result_dir, "checkpoints"),
                "resume_managed_by_cmd": True,
                "source": "wrapper:run_seed.sh",
            }
    return None


def default_conflict_path_key(path):
    if not path:
        return ""
    return os.path.normpath(os.path.expanduser(str(path).rstrip("/")))


def infer_bapr_result_dirs_from_cmd(
    cmd: str,
    cwd: str = "",
    *,
    conflict_path_key: Callable[[Any], str] = default_conflict_path_key,
) -> list:
    out = []
    inferred = infer_bapr_run_seed_ckpt(cmd, cwd)
    if inferred and inferred.get("result_dir"):
        out.append(inferred["result_dir"])

    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = []
    for i, tok in enumerate(toks[:-1]):
        if tok not in ("-c", "-lc"):
            continue
        for _, inner in expand_simple_seed_loop_inner(toks[i + 1]):
            shell = " ".join([shlex.quote(toks[0]), shlex.quote(tok), shlex.quote(inner)])
            inferred = infer_bapr_run_seed_ckpt(shell, cwd)
            if inferred and inferred.get("result_dir"):
                out.append(inferred["result_dir"])

    dedup = []
    seen = set()
    for path in out:
        key = conflict_path_key(path)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup


def bapr_run_seed_metas_from_cmd(
    cmd: str,
    cwd: str = "",
    *,
    conflict_path_key: Callable[[Any], str] = default_conflict_path_key,
) -> list[dict]:
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        toks = []
    candidate_token_lists = []
    if toks:
        candidate_token_lists.append(toks)
        for i, tok in enumerate(toks[:-1]):
            if tok in ("-c", "-lc"):
                inner = toks[i + 1]
                try:
                    candidate_token_lists.append(shlex.split(inner))
                except Exception:
                    pass
                for _seed, expanded_inner in expand_simple_seed_loop_inner(inner):
                    try:
                        candidate_token_lists.append(shlex.split(expanded_inner))
                    except Exception:
                        pass

    metas = []
    seen = set()
    for ctoks in candidate_token_lists:
        for i, tok in enumerate(ctoks):
            if not tok.endswith("run_seed.sh"):
                continue
            args = ctoks[i + 1:]
            if os.path.basename(tok) != "run_seed.sh" or len(args) < 3:
                continue
            algo, env_name, seed = args[0], args[1], args[2]
            max_iters = args[3] if len(args) >= 4 and args[3] else "1500"
            dwell = args[4] if len(args) >= 5 and args[4] else "60"
            target_mode = args[5] if len(args) >= 6 and args[5] else "min"
            tag = args[6] if len(args) >= 7 and args[6] else "paper"
            if any(("$" in str(x) or ";" in str(x)) for x in (
                algo, env_name, seed, max_iters, dwell, target_mode, tag,
            )):
                continue
            try:
                max_iters_i = int(max_iters)
            except Exception:
                continue
            env_short = env_name[:-3] if env_name.endswith("-v2") else env_name
            run_name = f"{tag}_{algo}_{env_short}_dw{dwell}_s{seed}"
            base = os.path.dirname(tok) if os.path.isabs(tok) else (cwd or os.getcwd())
            result_dir = os.path.join(base, "jax_experiments", "results_paper", run_name)
            key = conflict_path_key(result_dir)
            if not key or key in seen:
                continue
            seen.add(key)
            metas.append({
                "algo": algo,
                "env": env_name,
                "seed": str(seed),
                "max_iters": max_iters_i,
                "dwell": str(dwell),
                "target_mode": str(target_mode),
                "tag": str(tag),
                "run_name": run_name,
                "result_dir": result_dir,
            })
    return metas


def bapr_batch_projection(
    task: dict,
    tail_text: str,
    elapsed_s: float,
    *,
    load_eta_tracker_module: Callable[[], Any],
    clean_result_path: Callable[[str], str],
    conflict_path_key: Callable[[Any], str] = default_conflict_path_key,
) -> Optional[dict]:
    metas = bapr_run_seed_metas_from_cmd(
        task.get("cmd", ""),
        task.get("cwd", ""),
        conflict_path_key=conflict_path_key,
    )
    if len(metas) <= 1:
        return None
    et = load_eta_tracker_module()
    if not et:
        return None
    try:
        current = et._extract_current_only_from_tail(tail_text)
    except Exception:
        current = None
    if current is None:
        try:
            progress = et.parse_progress(tail_text, cmd=task.get("cmd"))
            current = progress[0] if progress else None
        except Exception:
            current = None

    run_to_idx = {m["run_name"]: i for i, m in enumerate(metas)}
    completed = set()
    for m in re.finditer(r"Results saved to:\s*(\S+)", tail_text or ""):
        raw = clean_result_path(m.group(1))
        parts = Path(raw).parts
        run_name = ""
        if parts:
            run_name = parts[-2] if parts[-1] == "logs" and len(parts) >= 2 else parts[-1]
        idx = run_to_idx.get(run_name)
        if idx is not None:
            completed.add(idx)
    for i, meta in enumerate(metas):
        try:
            if (Path(meta["result_dir"]) / "done.ok").exists():
                completed.add(i)
        except Exception:
            pass

    completed_count = 0
    while completed_count < len(metas) and completed_count in completed:
        completed_count += 1
    total_units = sum(int(m.get("max_iters") or 0) for m in metas)
    completed_units = sum(int(metas[i].get("max_iters") or 0) for i in range(completed_count))
    if completed_count >= len(metas):
        return {
            "source": "bapr_seed_batch",
            "eta_s": 0,
            "total_s": int(max(0, elapsed_s)),
            "current": total_units,
            "total_units": total_units,
            "unit_s": (float(elapsed_s) / float(total_units)) if total_units > 0 else None,
        }
    if current is None:
        return None
    current_total = int(metas[completed_count].get("max_iters") or 0)
    current = max(0, min(int(current), current_total))
    current_units = completed_units + current
    if total_units <= 0 or current_units <= 0 or elapsed_s <= 0:
        return None
    unit_s = float(elapsed_s) / float(current_units)
    eta_s = int(max(0, (total_units - current_units) * unit_s))
    return {
        "source": "bapr_seed_batch",
        "eta_s": eta_s,
        "total_s": int(max(elapsed_s, total_units * unit_s)),
        "current": int(current_units),
        "total_units": int(total_units),
        "unit_s": unit_s,
        "completed_seeds": completed_count,
        "seed_count": len(metas),
    }
