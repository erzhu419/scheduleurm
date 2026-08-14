from __future__ import annotations

import fnmatch
import ast
import base64
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


RESULT_FILE_EXTS = frozenset({
    ".csv", ".json", ".jsonl", ".pkl", ".pickle", ".pt", ".pth", ".ckpt",
    ".npz", ".npy", ".parquet", ".feather", ".txt", ".log", ".yaml", ".yml",
})
RESULT_FILE_FLAGS = frozenset({
    "--output", "--out", "--outfile", "--out-file", "--out_file",
    "--out-csv", "--out_csv", "--output-csv", "--output_csv", "--csv",
    "--out-json", "--out_json", "--output-json", "--output_json", "--json",
    "--metrics", "--metrics-file", "--metrics_file",
    "--log-file", "--log_file",
})
RESULT_DIR_FLAGS = frozenset({
    "--result-dir", "--result_dir", "--results-dir", "--results_dir",
    "--out-dir", "--out_dir",
    "--save-dir", "--save_dir", "--save-root", "--save_root",
    "--log-dir", "--log_dir", "--logging-dir", "--logging_dir",
    "--tb-dir", "--tb_dir", "--tensorboard-dir", "--tensorboard_dir",
})
FINAL_MODEL_SUCCESS_FILES = ("final_policy", "final_q", "final_norm")
COMPLETION_RESULT_FILES = (
    "summary.csv", "summary.json", "metrics.csv", "metrics.json",
    "results.csv", "results.json", "result.csv", "result.json",
    "summary.md", "group.json",
    "DONE", "done", "_SUCCESS", "_SUCCESS.json",
)
FAILURE_RESULT_FILES = ("_FAILED.exitcode",)

PROGRESS_COMPLETION_FILES = (
    "logs/iteration.npy",
    "iteration.npy",
)
LOG_RESULT_RE = re.compile(
    r"(?i)(?:"
    r"results?\s+saved\s+to|logging\s+to|"
    r"output(?:\s+written)?(?:\s+to)?|"
    r"output\s+already\s+present|"
    r"out_(?:json|csv|path)|"
    r"saved(?:\s+\w+){0,4}\s+to|saved"
    r")\s*:?\s+(?P<path>\S+)"
)


@dataclass(frozen=True)
class ResultArtifactDeps:
    node_configs: dict
    fetch_log_tail: Callable[[dict], tuple[str, int]]
    node_is_windows: Callable[[str], bool]
    windows_path_for_task: Callable[[dict, str], str]
    ps_quote: Callable[[str], str]
    run_windows_ps: Callable[..., tuple]
    run_on: Callable[..., tuple]
    run_subprocess: Callable[..., subprocess.CompletedProcess] = subprocess.run
    now: Callable[[], float] = time.time


@dataclass(frozen=True)
class ResultsCommandDeps:
    state_lock: Callable[..., object]
    load_state: Callable[[], dict]
    load_archive_tasks: Callable[[], list]
    task_result_artifacts_for_display: Callable[[dict, bool], list]


def clean_result_path(raw: str) -> str:
    if raw is None:
        return ""
    path = str(raw).strip().strip("'\"`<>")
    path = re.sub(r"\x1b\[[0-9;]*m", "", path)
    path = path.rstrip(".,;)]}")
    if not path or path.startswith("-"):
        return ""
    suffix = Path(path).suffix.lower()
    if (
        path.startswith(("/", "./", "../", "~/", "$HOME/"))
        or "/" in path
        or suffix in RESULT_FILE_EXTS
    ):
        return path
    return ""


def resolve_task_path(task: dict, raw_path: str) -> str:
    path = clean_result_path(raw_path)
    if not path:
        return ""
    if path.startswith("$HOME/"):
        path = "~/" + path[len("$HOME/"):]
    if path.startswith("~/"):
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    cwd = task.get("cwd") or ""
    if cwd and cwd not in ("(unknown)", "."):
        return os.path.normpath(os.path.join(cwd, path))
    return os.path.normpath(path)


def result_kind_for_path(path: str, preferred: str = "") -> str:
    if preferred in ("file", "dir"):
        return preferred
    return "file" if Path(path).suffix.lower() in RESULT_FILE_EXTS else "dir"


def add_result_artifact(artifacts: list, task: dict, raw_path: str,
                        kind: str = "", source: str = "") -> None:
    path = resolve_task_path(task, raw_path)
    if not path:
        return
    node = task.get("node") or task.get("last_node") or "unknown"
    rec = {
        "path": path,
        "node": node,
        "kind": result_kind_for_path(path, kind),
        "source": source or "inferred",
    }
    key = (rec["node"], rec["path"])
    for old in artifacts:
        if (old.get("node"), old.get("path")) == key:
            if rec["source"] not in (old.get("source") or ""):
                old["source"] = ",".join(x for x in [old.get("source"), rec["source"]] if x)
            if old.get("kind") == "dir" and rec["kind"] == "file":
                old["kind"] = "file"
            return
    artifacts.append(rec)


def cmd_flag_values(tokens: list, flags: set) -> dict:
    vals = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        flag, eq, val = tok.partition("=")
        if eq and flag in flags:
            vals.setdefault(flag, []).append(val)
            i += 1
            continue
        if tok in flags and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if not nxt.startswith("-"):
                vals.setdefault(tok, []).append(nxt)
                i += 2
                continue
        i += 1
    return vals


def cmd_token_variants(cmd: str) -> list:
    if not cmd:
        return []
    variants = []

    def add(tokens):
        if tokens and tokens not in variants:
            variants.append(tokens)

    try:
        add(shlex.split(cmd))
    except Exception:
        add(str(cmd).split())

    shells = {"bash", "sh", "dash", "zsh", "ksh"}
    for tokens in list(variants):
        shell_seen = False
        for i, tok in enumerate(tokens[:-1]):
            if os.path.basename(str(tok)) in shells:
                shell_seen = True
                continue
            if shell_seen and tok in ("-c", "-lc"):
                try:
                    add(shlex.split(tokens[i + 1]))
                except Exception:
                    add(str(tokens[i + 1]).split())
                break
    return variants


def cmd_flag_values_from_cmd(cmd: str, flags: set) -> dict:
    vals = {}
    for tokens in cmd_token_variants(cmd):
        for flag, items in cmd_flag_values(tokens, flags).items():
            bucket = vals.setdefault(flag, [])
            for item in items:
                if item not in bucket:
                    bucket.append(item)
    return vals


def result_artifacts_from_cmd(task: dict) -> list:
    cmd = task.get("cmd") or ""
    artifacts = []
    if not cmd or (cmd.startswith("(auto-adopted") and " --" not in cmd):
        return artifacts
    file_vals = cmd_flag_values_from_cmd(cmd, RESULT_FILE_FLAGS)
    dir_vals = cmd_flag_values_from_cmd(cmd, RESULT_DIR_FLAGS)
    for flag, vals in file_vals.items():
        for value in vals:
            add_result_artifact(artifacts, task, value, "file", f"cmd:{flag}")
    for flag, vals in dir_vals.items():
        for value in vals:
            add_result_artifact(artifacts, task, value, "dir", f"cmd:{flag}")

    roots = []
    for key in ("--save-root", "--save_root"):
        roots.extend(dir_vals.get(key, []))
    run_names = []
    for key in ("--run-name", "--run_name", "--name"):
        run_names.extend(cmd_flag_values_from_cmd(cmd, {key}).get(key, []))
    if roots and run_names:
        for root in roots:
            for run_name in run_names:
                if run_name and not run_name.startswith("-"):
                    add_result_artifact(
                        artifacts,
                        task,
                        os.path.join(root, run_name),
                        "dir",
                        "cmd:save_root+run_name",
                    )
                    for subdir in ("model", "logs", "pic"):
                        add_result_artifact(
                            artifacts,
                            task,
                            os.path.join(root, subdir, run_name),
                            "dir",
                            f"cmd:save_root+{subdir}+run_name",
                        )
    return artifacts


def save_root_run_name_pairs_from_cmd(task: dict) -> list:
    cmd = task.get("cmd") or ""
    if not cmd or (cmd.startswith("(auto-adopted") and " --" not in cmd):
        return []
    vals = cmd_flag_values_from_cmd(
        cmd,
        {"--save-root", "--save_root", "--run-name", "--run_name", "--name"},
    )
    roots = []
    for key in ("--save-root", "--save_root"):
        roots.extend(vals.get(key, []))
    run_names = []
    for key in ("--run-name", "--run_name", "--name"):
        run_names.extend(vals.get(key, []))
    pairs = []
    for root in roots:
        if not root or str(root).startswith("-"):
            continue
        for run_name in run_names:
            if run_name and not str(run_name).startswith("-"):
                pairs.append((str(root), str(run_name)))
    return pairs


def bash_path_arg(path: str) -> str:
    path = str(path)
    if path.startswith("~/"):
        rest = path[2:]
        if not rest:
            return '"$HOME"'
        return '"$HOME"/' + shlex.quote(rest)
    return shlex.quote(path)


def final_model_file_groups_from_cmd(task: dict) -> list:
    groups = []
    for root, run_name in save_root_run_name_pairs_from_cmd(task):
        model_dir = resolve_task_path(task, os.path.join(root, "model", run_name))
        if not model_dir:
            continue
        groups.append((
            model_dir,
            [os.path.join(model_dir, name) for name in FINAL_MODEL_SUCCESS_FILES],
        ))
    return groups


def mtimes_match_task_window(task: dict, mtimes: list) -> bool:
    if len(mtimes) < len(FINAL_MODEL_SUCCESS_FILES):
        return False
    started = float(task.get("started_at") or 0)
    if started > 0 and min(float(value) for value in mtimes) < started - 300:
        return False
    return True


def _mtime_matches_task_window(task: dict, mtime: float) -> bool:
    started = float(task.get("started_at") or 0)
    return started <= 0 or float(mtime) >= started - 300


def _completion_marker_for_name(name: str) -> str:
    name = Path(str(name)).name
    return f"result_artifact:{name}" if name else "result_artifact"


def _cmd_int_value(task: dict, flags: set[str]) -> int:
    vals = cmd_flag_values_from_cmd(task.get("cmd") or "", flags)
    for flag in flags:
        for raw in vals.get(flag, []):
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return 0


def _expected_progress_units_from_cmd(task: dict) -> int:
    return _cmd_int_value(
        task,
        {"--max_iters", "--max-iters", "--iterations", "--num_iters", "--num-iters"},
    )


def _read_npy_count_last(path: Path) -> tuple[int, float | int | None]:
    with path.open("rb") as fh:
        if fh.read(6) != b"\x93NUMPY":
            return 0, None
        major = fh.read(1)[0]
        fh.read(1)
        if major == 1:
            header_len = struct.unpack("<H", fh.read(2))[0]
        else:
            header_len = struct.unpack("<I", fh.read(4))[0]
        header = ast.literal_eval(fh.read(header_len).decode("latin1").strip())
        shape = header.get("shape") or ()
        if isinstance(shape, int):
            shape = (shape,)
        count = 1
        for value in shape:
            count *= int(value)
        if count <= 0:
            return 0, None
        descr = str(header.get("descr") or "")
        m = re.search(r"([iuf])(\d+)$", descr)
        if not m:
            return count, None
        kind, size_text = m.groups()
        item_size = int(size_text)
        data_start = fh.tell()
        fh.seek(data_start + (count - 1) * item_size)
        raw = fh.read(item_size)
        if len(raw) != item_size:
            return count, None
        byteorder = "big" if descr.startswith(">") else "little"
        if kind == "i":
            return count, int.from_bytes(raw, byteorder=byteorder, signed=True)
        if kind == "u":
            return count, int.from_bytes(raw, byteorder=byteorder, signed=False)
        if kind == "f":
            fmt = {4: "f", 8: "d"}.get(item_size)
            if fmt:
                prefix = ">" if byteorder == "big" else "<"
                return count, float(struct.unpack(prefix + fmt, raw)[0])
        return count, None


_REMOTE_NPY_COMPLETION_PROBE = r'''
import ast, os, re, struct, sys
path = sys.argv[1]
expected = int(sys.argv[2])
def read(path):
    with open(path, "rb") as fh:
        if fh.read(6) != b"\x93NUMPY":
            return 0, None
        major = fh.read(1)[0]
        fh.read(1)
        header_len = struct.unpack("<H", fh.read(2))[0] if major == 1 else struct.unpack("<I", fh.read(4))[0]
        header = ast.literal_eval(fh.read(header_len).decode("latin1").strip())
        shape = header.get("shape") or ()
        if isinstance(shape, int):
            shape = (shape,)
        count = 1
        for value in shape:
            count *= int(value)
        if count <= 0:
            return 0, None
        descr = str(header.get("descr") or "")
        m = re.search(r"([iuf])(\d+)$", descr)
        if not m:
            return count, None
        kind, size_text = m.groups()
        item_size = int(size_text)
        data_start = fh.tell()
        fh.seek(data_start + (count - 1) * item_size)
        raw = fh.read(item_size)
        if len(raw) != item_size:
            return count, None
        byteorder = "big" if descr.startswith(">") else "little"
        if kind == "i":
            return count, int.from_bytes(raw, byteorder=byteorder, signed=True)
        if kind == "u":
            return count, int.from_bytes(raw, byteorder=byteorder, signed=False)
        if kind == "f":
            fmt = {4: "f", 8: "d"}.get(item_size)
            if fmt:
                prefix = ">" if byteorder == "big" else "<"
                return count, float(struct.unpack(prefix + fmt, raw)[0])
        return count, None
count, last = read(path)
mtime = int(os.path.getmtime(path))
print(f"NPY {mtime} {count} {'' if last is None else last}")
'''


def terminal_result_artifact_success(task: dict, *, deps: ResultArtifactDeps) -> str:
    """Treat fresh declared result artifacts as a terminal success marker.

    This is intentionally stricter for directories than generic result discovery:
    a directory only counts when a known completion/summary file is present and
    fresh for this task run. That avoids turning partial output directories into
    successful terminal evidence.
    """
    artifacts = discover_result_artifacts(task, include_log=False, deps=deps)
    if not artifacts:
        return ""
    node = task.get("node") or task.get("last_node") or "local"
    node_info = deps.node_configs.get(node) or {}
    expected_progress_units = _expected_progress_units_from_cmd(task)

    def _progress_marker(candidate_name: str, count: int, last_value) -> str:
        if expected_progress_units <= 0 or count < expected_progress_units:
            return ""
        if last_value is not None:
            try:
                if float(last_value) < float(expected_progress_units - 1):
                    return ""
            except (TypeError, ValueError):
                return ""
        return _completion_marker_for_name(candidate_name)

    def _local_success(path: str, kind: str) -> str:
        p = Path(os.path.expanduser(path))
        if kind == "file":
            if p.is_file() and _mtime_matches_task_window(task, p.stat().st_mtime):
                return _completion_marker_for_name(p.name)
            return ""
        if not p.is_dir():
            return ""
        for name in COMPLETION_RESULT_FILES:
            candidate = p / name
            if candidate.is_file() and _mtime_matches_task_window(task, candidate.stat().st_mtime):
                return _completion_marker_for_name(name)
        if expected_progress_units > 0:
            for name in PROGRESS_COMPLETION_FILES:
                candidate = p / name
                if not candidate.is_file() or not _mtime_matches_task_window(task, candidate.stat().st_mtime):
                    continue
                try:
                    count, last_value = _read_npy_count_last(candidate)
                except Exception:
                    continue
                marker = _progress_marker(name, count, last_value)
                if marker:
                    return marker
        return ""

    def _remote_success(path: str, kind: str) -> str:
        quoted = bash_path_arg(path)
        names = " ".join(shlex.quote(name) for name in COMPLETION_RESULT_FILES)
        if kind == "file":
            cmd = (
                f"p={quoted}; "
                "if [ -f \"$p\" ]; then "
                "printf 'FILE %s %s\\n' \"$(stat -c %Y \"$p\")\" \"$(basename \"$p\")\"; "
                "fi"
            )
        else:
            cmd = (
                f"p={quoted}; "
                "if [ -d \"$p\" ]; then "
                f"for n in {names}; do "
                "if [ -f \"$p/$n\" ]; then "
                "printf 'DIR %s %s\\n' \"$(stat -c %Y \"$p/$n\")\" \"$n\"; exit 0; "
                "fi; "
                "done; "
                "fi"
            )
        rc, out, _ = deps.run_on(node, cmd, timeout=10, check=False)
        if rc != 0:
            return ""
        if (out or "").strip():
            for line in out.splitlines():
                parts = line.strip().split(maxsplit=2)
                if len(parts) < 3:
                    continue
                try:
                    mtime = float(parts[1])
                except ValueError:
                    continue
                if _mtime_matches_task_window(task, mtime):
                    return _completion_marker_for_name(parts[2])
        if kind != "file" and expected_progress_units > 0:
            probe = shlex.quote(_REMOTE_NPY_COMPLETION_PROBE)
            for name in PROGRESS_COMPLETION_FILES:
                candidate = path.rstrip("/") + "/" + name
                cmd = (
                    f"p={bash_path_arg(candidate)}; "
                    "if [ -f \"$p\" ]; then "
                    f"python3 -c {probe} \"$p\" {int(expected_progress_units)}; "
                    "fi"
                )
                rc, out, _ = deps.run_on(node, cmd, timeout=10, check=False)
                if rc != 0 or not (out or "").strip():
                    continue
                for line in out.splitlines():
                    parts = line.strip().split(maxsplit=3)
                    if len(parts) < 3 or parts[0] != "NPY":
                        continue
                    try:
                        mtime = float(parts[1])
                        count = int(parts[2])
                        last_value = parts[3] if len(parts) > 3 and parts[3] != "" else None
                    except ValueError:
                        continue
                    if not _mtime_matches_task_window(task, mtime):
                        continue
                    marker = _progress_marker(name, count, last_value)
                    if marker:
                        return marker
        return ""

    def _windows_success(path: str, kind: str) -> str:
        win_path = deps.windows_path_for_task(task, path)
        names = "@(" + ",".join(deps.ps_quote(name) for name in COMPLETION_RESULT_FILES) + ")"
        ps = rf'''
$path = {deps.ps_quote(win_path)}
$kind = {deps.ps_quote(kind)}
if ($kind -eq "file") {{
  if (Test-Path -LiteralPath $path -PathType Leaf) {{
    $item = Get-Item -LiteralPath $path
    [Console]::WriteLine("FILE {0} {1}", [int64]([DateTimeOffset]::new($item.LastWriteTimeUtc).ToUnixTimeSeconds()), $item.Name)
  }}
  exit 0
}}
if (Test-Path -LiteralPath $path -PathType Container) {{
  foreach ($name in {names}) {{
    $candidate = Join-Path $path $name
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {{
      $item = Get-Item -LiteralPath $candidate
      [Console]::WriteLine("DIR {0} {1}", [int64]([DateTimeOffset]::new($item.LastWriteTimeUtc).ToUnixTimeSeconds()), $item.Name)
      exit 0
    }}
  }}
}}
'''
        rc, out, _ = deps.run_windows_ps(node, ps, timeout=10, check=False)
        if rc != 0 or not (out or "").strip():
            return ""
        for line in out.splitlines():
            parts = line.strip().split(maxsplit=2)
            if len(parts) < 3:
                continue
            try:
                mtime = float(parts[1])
            except ValueError:
                continue
            if _mtime_matches_task_window(task, mtime):
                return _completion_marker_for_name(parts[2])
        return ""

    artifacts = sorted(
        artifacts,
        key=lambda rec: 0 if str(rec.get("source") or "").startswith("cmd:") else 1,
    )
    for rec in artifacts:
        path = rec.get("path") or ""
        kind = rec.get("kind") or result_kind_for_path(path)
        try:
            if node_info.get("host") is None:
                marker = _local_success(path, kind)
            elif deps.node_is_windows(node):
                marker = _windows_success(path, kind)
            else:
                marker = _remote_success(path, kind)
            if marker:
                return marker
        except Exception:
            continue
    return ""


def _terminal_evidence_item(task: dict, *, deps: ResultArtifactDeps) -> dict:
    artifacts = sorted(
        discover_result_artifacts(task, include_log=False, deps=deps),
        key=lambda rec: 0 if str(rec.get("source") or "").startswith("cmd:") else 1,
    )
    return {
        "id": str(task.get("id") or ""),
        "log_path": str(task.get("log_path") or ""),
        "artifacts": [
            {
                "path": str(rec.get("path") or ""),
                "kind": str(rec.get("kind") or result_kind_for_path(rec.get("path") or "")),
            }
            for rec in artifacts
            if rec.get("path")
        ],
    }


def _local_terminal_evidence(item: dict) -> dict:
    rec = {
        "id": item.get("id"),
        "log_path": item.get("log_path") or "",
        "tail": "",
        "log_size": 0,
    }
    log_path = Path(os.path.expandvars(os.path.expanduser(item.get("log_path") or "")))
    try:
        if log_path.is_file():
            rec["log_size"] = log_path.stat().st_size
            with log_path.open("rb") as fh:
                fh.seek(max(0, rec["log_size"] - 4096))
                rec["tail"] = fh.read().decode("utf-8", errors="replace")
    except OSError:
        pass
    for artifact in item.get("artifacts") or []:
        path = Path(os.path.expandvars(os.path.expanduser(artifact.get("path") or "")))
        try:
            if artifact.get("kind") == "file" and path.is_file():
                rec["result_name"] = path.name
                rec["result_mtime"] = path.stat().st_mtime
                break
            if artifact.get("kind") != "file" and path.is_dir():
                for name in COMPLETION_RESULT_FILES:
                    candidate = path / name
                    if candidate.is_file():
                        rec["result_name"] = name
                        rec["result_mtime"] = candidate.stat().st_mtime
                        break
                if rec.get("result_name"):
                    break
        except OSError:
            continue
    for artifact in item.get("artifacts") or []:
        if artifact.get("kind") == "file":
            continue
        path = Path(os.path.expandvars(os.path.expanduser(artifact.get("path") or "")))
        try:
            for name in FAILURE_RESULT_FILES:
                candidate = path / name
                if not candidate.is_file():
                    continue
                rec["failure_name"] = name
                rec["failure_mtime"] = candidate.stat().st_mtime
                rec["failure_value"] = candidate.read_text(
                    encoding="utf-8", errors="replace"
                )[:64].strip()
                break
            if rec.get("failure_name"):
                break
        except OSError:
            continue
    return rec


def _terminal_evidence_chunks(items: list[dict], batch_size: int) -> list[list[dict]]:
    size = max(1, int(batch_size or 1))
    return [items[index:index + size] for index in range(0, len(items), size)]


def _remote_terminal_evidence_command(items: list[dict]) -> str:
    completion_names = " ".join(shlex.quote(name) for name in COMPLETION_RESULT_FILES)
    failure_names = " ".join(shlex.quote(name) for name in FAILURE_RESULT_FILES)
    blocks = ["command -v base64 >/dev/null 2>&1 || exit 69"]
    for item in items:
        tid = shlex.quote(str(item.get("id") or ""))
        log_arg = bash_path_arg(item.get("log_path") or "/nonexistent")
        block = [
            f"tid={tid}",
            f"lp={log_arg}",
            "sz=0",
            "tail64=",
            "if [ -f \"$lp\" ]; then "
            "sz=$(wc -c < \"$lp\" 2>/dev/null || printf 0); "
            "tail64=$(tail -c 4096 \"$lp\" 2>/dev/null | base64 | tr -d '\\n'); fi",
            "rn=",
            "rm=0",
            "fn=",
            "fm=0",
            "fv64=",
        ]
        for artifact in item.get("artifacts") or []:
            path_arg = bash_path_arg(artifact.get("path") or "/nonexistent")
            block.append(f"p={path_arg}")
            if artifact.get("kind") == "file":
                block.append(
                    "if [ -z \"$rn\" ] && [ -f \"$p\" ]; then "
                    "rn=$(basename -- \"$p\"); "
                    "rm=$(stat -c %Y -- \"$p\" 2>/dev/null || printf 0); fi"
                )
            else:
                block.append(
                    "if [ -z \"$rn\" ] && [ -d \"$p\" ]; then "
                    f"for n in {completion_names}; do "
                    "if [ -f \"$p/$n\" ]; then rn=$n; "
                    "rm=$(stat -c %Y -- \"$p/$n\" 2>/dev/null || printf 0); "
                    "break; fi; done; fi"
                )
                block.append(
                    "if [ -z \"$fn\" ] && [ -d \"$p\" ]; then "
                    f"for n in {failure_names}; do "
                    "if [ -f \"$p/$n\" ]; then fn=$n; "
                    "fm=$(stat -c %Y -- \"$p/$n\" 2>/dev/null || printf 0); "
                    "fv64=$(head -c 64 \"$p/$n\" 2>/dev/null | base64 | tr -d '\\n'); "
                    "break; fi; done; fi"
                )
        block.extend([
            "rn64=$(printf %s \"$rn\" | base64 | tr -d '\\n')",
            "fn64=$(printf %s \"$fn\" | base64 | tr -d '\\n')",
            "printf 'E\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
            "\"$tid\" \"$sz\" \"$tail64\" \"$rn64\" \"$rm\" "
            "\"$fn64\" \"$fm\" \"$fv64\"",
        ])
        blocks.append("; ".join(block))
    return "\n".join(blocks)


def collect_terminal_evidence(
    tasks: list[dict],
    *,
    deps: ResultArtifactDeps,
    max_workers: int = 4,
    batch_size: int = 32,
    timeout_s: float = 0.0,
) -> dict:
    """Fetch terminal log tails and declared completion markers once per node batch."""
    deadline = (
        time.monotonic() + float(timeout_s)
        if float(timeout_s or 0.0) > 0.0
        else None
    )

    def _remaining_timeout(default_s: float) -> float:
        if deadline is None:
            return default_s
        return max(0.0, min(default_s, deadline - time.monotonic()))

    task_by_id = {
        str(task.get("id") or ""): task
        for task in tasks
        if task.get("id")
    }
    item_by_id = {
        tid: _terminal_evidence_item(task, deps=deps)
        for tid, task in task_by_id.items()
    }
    raw_records: dict[str, dict] = {}
    remote_groups: dict[str, list[dict]] = {}
    for tid, task in task_by_id.items():
        node = str(task.get("node") or task.get("last_node") or "local")
        node_info = deps.node_configs.get(node) or {}
        item = item_by_id[tid]
        if node_info.get("host") is None:
            raw_records[tid] = _local_terminal_evidence(item)
        elif deps.node_is_windows(node):
            # Windows remains on the bounded deep-diagnosis path until it has
            # an equivalent multi-file PowerShell probe.
            continue
        else:
            remote_groups.setdefault(node, []).append(item)

    jobs = [
        (node, _terminal_evidence_chunks(items, batch_size))
        for node, items in remote_groups.items()
    ]

    def _probe_remote_chunk(node: str, chunk: list[dict]) -> list[dict]:
        cmd = _remote_terminal_evidence_command(chunk)
        out = ""
        for attempt in range(2):
            call_timeout = _remaining_timeout(15.0)
            if call_timeout <= 0.0:
                break
            try:
                rc, out, _ = deps.run_on(
                    node,
                    cmd,
                    timeout=max(0.05, call_timeout),
                    check=False,
                )
            except Exception:
                rc, out = 1, ""
            if rc == 0 and out:
                break
            if attempt == 0 and _remaining_timeout(0.2) > 0.0:
                time.sleep(_remaining_timeout(0.2))
        if not out or rc != 0:
            return []
        records = []
        for line in out.splitlines():
            parts = line.split("\t", 8)
            if len(parts) not in (6, 9) or parts[0] != "E":
                continue
            try:
                tail = base64.b64decode(parts[3]).decode("utf-8", errors="replace")
                result_name = base64.b64decode(parts[4]).decode("utf-8", errors="replace")
                log_size = int(parts[2] or 0)
                result_mtime = float(parts[5] or 0)
                failure_name = (
                    base64.b64decode(parts[6]).decode("utf-8", errors="replace")
                    if len(parts) == 9 else ""
                )
                failure_mtime = float(parts[7] or 0) if len(parts) == 9 else 0
                failure_value = (
                    base64.b64decode(parts[8]).decode("utf-8", errors="replace")
                    if len(parts) == 9 else ""
                )
            except (TypeError, ValueError):
                continue
            records.append({
                "id": parts[1],
                "log_path": item_by_id.get(parts[1], {}).get("log_path") or "",
                "tail": tail,
                "log_size": log_size,
                "result_name": result_name,
                "result_mtime": result_mtime,
                "failure_name": failure_name,
                "failure_mtime": failure_mtime,
                "failure_value": failure_value,
            })
        return records

    def _probe_remote_node(job: tuple[str, list[list[dict]]]) -> list[dict]:
        node, chunks = job
        records = []
        for chunk in chunks:
            if deadline is not None and _remaining_timeout(1.0) <= 0.0:
                break
            records.extend(_probe_remote_chunk(node, chunk))
        return records

    if jobs:
        workers = max(1, min(len(jobs), int(max_workers or 1)))
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(_probe_remote_node, job) for job in jobs]
        pending = set()
        try:
            if deadline is None:
                done = set(futures)
            else:
                done, pending = wait(
                    futures,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            for future in done:
                try:
                    records = future.result()
                except Exception:
                    records = []
                for rec in records:
                    raw_records[str(rec.get("id") or "")] = rec
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=not pending, cancel_futures=True)

    evidence = {}
    for tid, rec in raw_records.items():
        task = task_by_id.get(tid)
        if task is None:
            continue
        result_name = str(rec.get("result_name") or "")
        result_mtime = 0.0
        marker = ""
        if result_name:
            try:
                result_mtime = float(rec.get("result_mtime") or 0)
            except (TypeError, ValueError):
                result_mtime = 0
            if result_mtime > 0 and _mtime_matches_task_window(task, result_mtime):
                marker = _completion_marker_for_name(result_name)
        failure_reason = ""
        failure_mtime = 0.0
        failure_name = str(rec.get("failure_name") or "")
        if not marker and failure_name:
            try:
                failure_mtime = float(rec.get("failure_mtime") or 0)
            except (TypeError, ValueError):
                failure_mtime = 0
            if failure_mtime > 0 and _mtime_matches_task_window(task, failure_mtime):
                failure_value = str(rec.get("failure_value") or "").strip()
                failure_reason = f"result_failure:{failure_name}"
                if failure_value:
                    failure_reason += f"={failure_value[:32]}"
        if marker and failure_reason:
            if failure_mtime > result_mtime:
                marker = ""
            else:
                failure_reason = ""
        evidence[tid] = {
            "log_path": str(rec.get("log_path") or task.get("log_path") or ""),
            "tail": str(rec.get("tail") or ""),
            "log_size": int(rec.get("log_size") or 0),
            "success_marker": marker,
            "failure_reason": failure_reason,
            "artifact_checked": _expected_progress_units_from_cmd(task) <= 0,
            "result_artifacts": item_by_id[tid].get("artifacts") or [],
        }
    return evidence


def fetch_log_result_lines(task: dict, max_lines: int = 50, *, deps: ResultArtifactDeps) -> list:
    log_path = task.get("log_path")
    if not log_path or task.get("auto_adopted"):
        return []
    node = task.get("node")
    grep_re = (
        r"Results saved to|Logging to|Output|output already present|"
        r"out_json|out_csv|Saved"
    )
    try:
        if node and deps.node_configs.get(node, {}).get("host") is None:
            log_file = Path(log_path)
            if not log_file.exists():
                return []
            result = deps.run_subprocess(
                ["grep", "-E", "-i", "-m", str(int(max_lines)), grep_re, str(log_file)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode not in (0, 1):
                return []
            return [line for line in (result.stdout or "").splitlines() if LOG_RESULT_RE.search(line)]
        if node and deps.node_is_windows(node):
            win_path = deps.windows_path_for_task(task, log_path)
            ps = rf'''
$path = {deps.ps_quote(win_path)}
if (-not (Test-Path -LiteralPath $path)) {{ exit 0 }}
Get-Content -LiteralPath $path -ErrorAction SilentlyContinue |
  Select-String -Pattern {deps.ps_quote(grep_re)} |
  Select-Object -First {int(max_lines)} |
  ForEach-Object {{ $_.Line }}
'''
            rc, out, _ = deps.run_windows_ps(node, ps, timeout=15, check=False)
            if rc != 0 or not out:
                return []
            return [line for line in out.splitlines() if LOG_RESULT_RE.search(line)]
        if node:
            rc, out, _ = deps.run_on(
                node,
                f"grep -E -i -m {int(max_lines)} {shlex.quote(grep_re)} "
                f"{shlex.quote(log_path)} 2>/dev/null || true",
                timeout=15,
                check=False,
            )
            if rc != 0 or not out:
                return []
            return [line for line in out.splitlines() if LOG_RESULT_RE.search(line)]
    except Exception:
        return []
    return []


def result_artifacts_from_log(task: dict, *, deps: ResultArtifactDeps) -> list:
    artifacts = []
    tail_text, _ = deps.fetch_log_tail(task)
    if not tail_text:
        diag = task.get("_diagnosis") or {}
        tail_text = diag.get("tail") or ""
    lines = []
    if tail_text and tail_text != "(no log)":
        lines.extend(tail_text.splitlines())
    lines.extend(fetch_log_result_lines(task, deps=deps))
    if not lines:
        return artifacts
    for line in lines:
        match = LOG_RESULT_RE.search(line)
        if not match:
            continue
        raw = match.group("path")
        kind = "file" if Path(clean_result_path(raw)).suffix.lower() in RESULT_FILE_EXTS else ""
        add_result_artifact(artifacts, task, raw, kind, "log")
    return artifacts


def discover_result_artifacts(task: dict, include_log: bool = True,
                              *, deps: ResultArtifactDeps) -> list:
    artifacts = []
    if task.get("result_dir"):
        add_result_artifact(artifacts, task, task.get("result_dir"), "dir", "declared:result_dir")
    for rec in result_artifacts_from_cmd(task):
        add_result_artifact(artifacts, task, rec["path"], rec.get("kind"), rec.get("source"))
    if include_log:
        for rec in result_artifacts_from_log(task, deps=deps):
            add_result_artifact(artifacts, task, rec["path"], rec.get("kind"), rec.get("source"))
    return artifacts


def apply_discovered_result_artifacts(task: dict, discovered: list,
                                      *, deps: ResultArtifactDeps) -> list:
    if not discovered:
        return []
    combined = []
    for rec in (task.get("result_artifacts") or []) + discovered:
        add_result_artifact(combined, task, rec.get("path"), rec.get("kind"), rec.get("source"))
    task["result_artifacts"] = combined
    task["result_artifacts_discovered_at"] = deps.now()
    return combined


def record_result_artifacts(task: dict, *, deps: ResultArtifactDeps) -> list:
    include_log = not bool(task.get("result_dir") or task.get("result_dirs"))
    discovered = discover_result_artifacts(task, include_log=include_log, deps=deps)
    return apply_discovered_result_artifacts(task, discovered, deps=deps)


def task_result_artifacts_for_display(task: dict, include_log: bool = True,
                                      *, deps: ResultArtifactDeps) -> list:
    stored = task.get("result_artifacts") or []
    if stored:
        return stored
    return discover_result_artifacts(task, include_log=include_log, deps=deps)


def print_result_artifacts(task: dict, include_log: bool = True, *,
                           deps: ResultArtifactDeps,
                           remote_path_for_node: Callable[[str, str], str],
                           write: Callable[[str], object] = print) -> None:
    artifacts = task_result_artifacts_for_display(task, include_log=include_log, deps=deps)
    if not artifacts:
        return
    write("\n# result artifacts:")
    for rec in artifacts:
        node = rec.get("node") or task.get("node") or "unknown"
        kind = rec.get("kind") or "path"
        source = rec.get("source") or "inferred"
        path = remote_path_for_node(node, rec.get("path")) if node in deps.node_configs else rec.get("path")
        write(f"#   - [{kind}] {node}:{path}  ({source})")


def cmd_results(args, *, deps: ResultsCommandDeps) -> None:
    task_ids = set(args.task_ids or [])
    include_archive = not args.no_archive
    scan_logs = (bool(task_ids) or bool(args.scan_logs)) and not args.no_log_scan
    with deps.state_lock():
        state = deps.load_state()
        records = [("queue", task) for task in state.get("tasks", [])]
        if include_archive:
            records.extend(("archive", task) for task in deps.load_archive_tasks())

    statuses = set(args.status or [])
    candidates = []
    for source, task in records:
        if task_ids and task.get("id") not in task_ids:
            continue
        if args.project and not fnmatch.fnmatch(task.get("project") or "", args.project):
            continue
        if args.signature and not fnmatch.fnmatch(task.get("signature") or "", args.signature):
            continue
        if statuses and task.get("status") not in statuses:
            continue
        candidates.append((source, task))

    candidates.sort(key=lambda item: float(
        item[1].get("finished_at") or item[1].get("started_at") or item[1].get("submitted_at") or 0
    ), reverse=True)

    rows = []
    scanned = 0
    for source, task in candidates:
        if args.limit and len(rows) >= args.limit:
            break
        scanned += 1
        artifacts = deps.task_result_artifacts_for_display(task, include_log=scan_logs)
        if not artifacts and not args.include_empty and not task_ids:
            continue
        rows.append({
            "source": source,
            "id": task.get("id"),
            "status": task.get("status"),
            "project": task.get("project"),
            "signature": task.get("signature"),
            "node": task.get("node"),
            "description": task.get("description"),
            "log_path": task.get("log_path"),
            "artifacts": artifacts,
        })

    if args.json:
        print(json.dumps({"results": rows, "matched": len(candidates), "scanned": scanned}, indent=2))
        return
    if not rows:
        where = "queue+archive" if include_archive else "queue"
        print(f"(no result artifacts found in {where}; matched {len(candidates)} task records)")
        if not scan_logs:
            print("  note: logs were not scanned for this batch query; add --scan-logs or pass task IDs")
        return
    for row in rows:
        desc = (row.get("description") or "")[:80]
        print(f"[{row['id']}] {row['status']} {row.get('project') or '?'} "
              f"{row.get('node') or '-'} ({row['source']})  {desc}")
        if row.get("signature"):
            print(f"  sig: {row['signature']}")
        if row.get("log_path"):
            print(f"  log: {row['log_path']}")
        artifacts = row.get("artifacts") or []
        if not artifacts:
            print("  results: (none inferred)")
        for rec in artifacts:
            print(f"  - [{rec.get('kind') or 'path'}] {rec.get('node') or row.get('node') or 'unknown'}:"
                  f"{rec.get('path')}  ({rec.get('source') or 'inferred'})")
