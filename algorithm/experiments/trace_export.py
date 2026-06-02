"""Experiment trace export and manifest bootstrap.

Phase 0 only writes a manifest.  Later phases will add passive trace export and
normalization, but all of them should share this reproducibility identity.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from .schema import DEFAULT_ALGORITHM_MODULES, DEFAULT_THEOREM_TARGET, SCHEMA_VERSION


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_commit(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], repo)


def _git_dirty(repo: Path) -> bool:
    out = _run(["git", "status", "--porcelain"], repo)
    return bool(out)


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def default_runs_dir() -> Path:
    return Path.home() / ".claude" / "scheduler" / "experiments" / "runs"


def build_manifest(
    run_id: str,
    scheduleurm_repo: Path | None = None,
    proof_repo: Path | None = None,
    theorem_target: str = DEFAULT_THEOREM_TARGET,
) -> Dict[str, Any]:
    scheduleurm_repo = (scheduleurm_repo or _repo_root()).resolve()
    proof_repo = (proof_repo or (scheduleurm_repo.parent / "proof")).resolve()
    proof_upload = proof_repo / "ScheduleurmUpload.lean"
    created_at = _dt.datetime.now(_dt.timezone.utc).timestamp()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_ts": created_at,
        "created_at_iso": _now_iso(),
        "scheduleurm_repo": str(scheduleurm_repo),
        "scheduleurm_commit": _git_commit(scheduleurm_repo),
        "scheduleurm_dirty": _git_dirty(scheduleurm_repo),
        "proof_repo": str(proof_repo),
        "proof_commit": _git_commit(proof_repo),
        "proof_upload_path": str(proof_upload),
        "proof_upload_sha256": _sha256(proof_upload),
        "theorem_target": theorem_target,
        "slot_policy": {
            "source": "watcher_cycle",
            "delta_ref_s": 60,
            "min_slot_s": 10,
            "max_slot_s": 300,
        },
        "class_key_version": "v1_project_sigprefix_kind_resource",
        "signature_prefix_depth": 3,
        "regime_key_version": "v1_node_gpu_external_load",
        "algorithm_modules": dict(DEFAULT_ALGORITHM_MODULES),
        "finite_feature_metric": "algorithm.features:finite_feature_metric",
        "action_space_mode": "statewise_uniform|all_feasible|full_sampled|historical_full",
        "candidate_generator_version": "v1_scheduleurm_greedy_neighborhood_active_bucket",
        "active_bucket_representative_rule": "min robust score per finite bucket",
        "progress_tiers_allowed_for_theorem": ["A", "B-count"],
        "time_work_proxy_allowed_for_theorem": False,
        "notes": "",
    }


def init_run(
    run_id: str,
    runs_dir: Path | None = None,
    scheduleurm_repo: Path | None = None,
    proof_repo: Path | None = None,
    theorem_target: str = DEFAULT_THEOREM_TARGET,
) -> Path:
    if not run_id or "/" in run_id or "\x00" in run_id:
        raise ValueError("run_id must be a non-empty path segment")
    runs_dir = (runs_dir or default_runs_dir()).resolve()
    run_dir = runs_dir / run_id
    manifest = build_manifest(run_id, scheduleurm_repo, proof_repo, theorem_target)
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("raw", "normalized", "calibration", "reports"):
        (run_dir / child).mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _cmd_init_run(args: argparse.Namespace) -> int:
    path = init_run(
        args.run_id,
        runs_dir=Path(args.runs_dir).expanduser() if args.runs_dir else None,
        scheduleurm_repo=Path(args.scheduleurm_repo).expanduser() if args.scheduleurm_repo else None,
        proof_repo=Path(args.proof_repo).expanduser() if args.proof_repo else None,
        theorem_target=args.theorem_target,
    )
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.trace_export")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init-run", help="Create an experiment run manifest and directory skeleton")
    s.add_argument("--run-id", required=True)
    s.add_argument("--runs-dir", default="")
    s.add_argument("--scheduleurm-repo", default="")
    s.add_argument("--proof-repo", default="")
    s.add_argument("--theorem-target", default=DEFAULT_THEOREM_TARGET)
    s.set_defaults(func=_cmd_init_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
