"""Checkpoint/resume inference and validation helpers for submit policy."""

from __future__ import annotations

import os
import re
import shlex


COMMAND_MANAGED_RESUME_RUNNERS = frozenset({
    "run_training_dispatch_v6.py",
    "run_training_dispatch_v7.py",
    "run_training_matrix_v4.py",
})


def _python_script_basename(cmd: str) -> str:
    try:
        tokens = shlex.split(cmd or "")
    except ValueError:
        return ""
    for token in tokens:
        if token.endswith(".py"):
            return os.path.basename(token)
    return ""


def _python_script_path(cmd: str, cwd: str) -> str:
    try:
        tokens = shlex.split(cmd or "")
    except ValueError:
        return ""
    for token in tokens:
        if not token.endswith(".py"):
            continue
        path = token if os.path.isabs(token) else os.path.join(cwd or os.getcwd(), token)
        path = os.path.normpath(path)
        return path if os.path.isfile(path) else ""
    return ""

try:
    from .bapr import infer_bapr_run_seed_ckpt
    from .shell import (
        abs_under,
        arg_value,
        extract_script_cd_dir,
        safe_read_text,
        script_invocation_from_cmd,
        shell_expand_simple,
        simple_shell_env_from_script,
        submit_policy_text,
    )
    from .training_policy import cmd_looks_like_training
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .bapr import infer_bapr_run_seed_ckpt
    from .shell import (
        abs_under,
        arg_value,
        extract_script_cd_dir,
        safe_read_text,
        script_invocation_from_cmd,
        shell_expand_simple,
        simple_shell_env_from_script,
        submit_policy_text,
    )
    from .training_policy import cmd_looks_like_training


def cmd_has_resume_flag(cmd: str) -> bool:
    if not cmd:
        return False
    norm = cmd.replace("=", " ")
    return _python_script_basename(cmd) in COMMAND_MANAGED_RESUME_RUNNERS or any(p in norm for p in (
        "--resume ", "--resume_from ", "--resume-from ",
        "--load_ckpt ", "--load-ckpt ", "--load_from ", "--load-from ",
        "--init_from ", "--init-from ",
        "--ckpt_path ", "--ckpt-path ",
        "--restore ", "--restore_from ", "--restore-from ",
    )) or norm.rstrip().endswith("--resume")


def infer_direct_jax_train_ckpt(cmd: str, cwd: str = ""):
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        return None
    save_root = arg_value(toks, "--save_root") or arg_value(toks, "--save-root")
    run_name = arg_value(toks, "--run_name") or arg_value(toks, "--run-name")
    has_resume = cmd_has_resume_flag(cmd)
    if save_root and run_name:
        base = cwd or os.getcwd()
        return {
            "ckpt_dir": os.path.join(abs_under(base, save_root), run_name, "checkpoints"),
            "resume_managed_by_cmd": bool(has_resume),
            "source": "python-args",
        }
    return None


def infer_command_managed_resume(cmd: str, cwd: str = ""):
    runner = _python_script_basename(cmd)
    if runner not in COMMAND_MANAGED_RESUME_RUNNERS:
        return None
    script = _python_script_path(cmd, cwd)
    if not script or os.path.basename(script) != runner or not safe_read_text(script):
        return None
    return {
        "resume_managed_by_cmd": True,
        "source": f"command-managed:{runner}",
    }


def infer_wrapper_ckpt(cmd: str, cwd: str = ""):
    script, script_args = script_invocation_from_cmd(cmd, cwd)
    if not script:
        return None
    text = safe_read_text(script)
    if not text:
        return None
    env = simple_shell_env_from_script(text, script_args)
    script_cwd = extract_script_cd_dir(text, script, cwd)

    save_root = None
    m = re.search(r"--save[_-]root\s+([^\s\\]+)", text)
    if m:
        save_root = shell_expand_simple(m.group(1), env)
    run_name = env.get("RUN_NAME")
    if not run_name:
        m = re.search(r"--run[_-]name\s+([^\s\\]+)", text)
        if m:
            run_name = shell_expand_simple(m.group(1), env)
    if not (save_root and run_name):
        return None
    if "$" in save_root or "$" in run_name or not run_name.strip():
        return None
    return {
        "result_dir": os.path.join(abs_under(script_cwd, save_root), run_name),
        "ckpt_dir": os.path.join(abs_under(script_cwd, save_root), run_name, "checkpoints"),
        "resume_managed_by_cmd": cmd_has_resume_flag(text),
        "source": f"wrapper:{os.path.basename(script)}",
    }


def infer_checkpoint_from_submit(cmd: str, cwd: str = ""):
    return (
        infer_command_managed_resume(cmd, cwd)
        or infer_direct_jax_train_ckpt(cmd, cwd)
        or infer_wrapper_ckpt(cmd, cwd)
        or infer_bapr_run_seed_ckpt(cmd, cwd)
    )


def candidate_training_source_paths(cmd: str, cwd: str = ""):
    paths = []
    script, _ = script_invocation_from_cmd(cmd, cwd)
    script_text = safe_read_text(script) if script else ""
    policy = submit_policy_text(cmd, cwd)
    bases = []
    for base in (
        cwd,
        extract_script_cd_dir(script_text, script, cwd) if script_text else "",
        os.path.dirname(script) if script else "",
    ):
        if base and base not in bases:
            bases.append(base)
    if not bases:
        bases = [os.getcwd()]

    for mod in re.findall(r"-m\s+([A-Za-z_][\w.]*train[A-Za-z0-9_.]*)", policy):
        rel = mod.replace(".", os.sep) + ".py"
        for base in bases:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                paths.append(os.path.normpath(p))
    for py in re.findall(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_./-]*train[A-Za-z0-9_./-]*\.py)", policy):
        for base in bases:
            p = py if os.path.isabs(py) else os.path.join(base, py)
            if os.path.exists(p):
                paths.append(os.path.normpath(p))

    # Pipelines frequently invoke concrete trainers through subprocess rather
    # than exposing every trainer in the submitted shell command.  Inspect
    # those local, literal ``*train*.py`` references as well, so a wrapper is
    # assessed against the checkpoint contract it actually delegates to.  The
    # traversal is intentionally local and bounded by the discovered paths;
    # it never follows imports or arbitrary shell expansion.
    scan_index = 0
    while scan_index < len(paths):
        parent = paths[scan_index]
        scan_index += 1
        parent_src = safe_read_text(parent)
        if not parent_src:
            continue
        parent_bases = [os.path.dirname(parent), *bases]
        for module in re.findall(
            r"(?:from|import)\s+([A-Za-z_][\w.]*train[A-Za-z0-9_.]*)",
            parent_src,
        ):
            relative = module.replace(".", os.sep) + ".py"
            for base in bases:
                child = os.path.normpath(os.path.join(base, relative))
                if os.path.exists(child) and child not in paths:
                    paths.append(child)
        for py in re.findall(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_./-]*train[A-Za-z0-9_./-]*\.py)", parent_src):
            for base in parent_bases:
                child = py if os.path.isabs(py) else os.path.join(base, py)
                child = os.path.normpath(child)
                if os.path.exists(child) and child not in paths:
                    paths.append(child)

    expanded = []
    for p in paths:
        if p not in expanded:
            expanded.append(p)
        src = safe_read_text(p)
        if "jax_experiments.common.checkpoint" in src:
            for base in bases:
                cp = os.path.join(base, "jax_experiments", "common", "checkpoint.py")
                if os.path.exists(cp) and os.path.normpath(cp) not in expanded:
                    expanded.append(os.path.normpath(cp))
    return expanded


def checkpoint_contract_reason(cmd: str, cwd: str, ckpt_dir, resume_flag, allow_no_resume: bool):
    policy_cmd = submit_policy_text(cmd, cwd)
    if allow_no_resume or not cmd_looks_like_training(policy_cmd) or not ckpt_dir:
        return None
    if not (cmd_has_resume_flag(policy_cmd) or resume_flag):
        return None
    paths = candidate_training_source_paths(cmd, cwd)
    if not paths:
        return ("could not inspect local training source for checkpoint contract. "
                "Submit with a local cwd/wrapper the scheduler can read, or pass "
                "--allow-no-resume to explicitly accept non-resumable relaunches.")
    src = "\n".join(safe_read_text(p) for p in paths)
    lower = src.lower()
    checks = [
        ("save_checkpoint call", "save_checkpoint(" in src or "torch.save(" in src or ".save_checkpoint(" in src),
        ("load_checkpoint call", "load_checkpoint(" in src or "torch.load(" in src or ".load_checkpoint(" in src),
        ("checkpoint existence gate", "has_checkpoint(" in src or "os.path.exists" in src or "path.exists" in lower),
        ("iteration/start_iter state", bool(re.search(r"start_(?:iteration|iter)|['\"]iteration['\"]|global_step|epoch", src))),
        (
            "resume advances past saved iter",
            bool(
                re.search(
                    r"iteration['\"]\]\s*\+\s*1"
                    r"|start_(?:iteration|iter|epoch).*range\("
                    r"|range\(\s*start_(?:iteration|iter|epoch)",
                    src,
                    re.S,
                )
            ),
        ),
        ("model parameters", "params.pkl" in src or "state_dict" in lower or "nnx.state" in src or "flax" in lower),
        ("optimizer state", "opt_state" in lower or "optimizer" in lower),
        ("replay/buffer state", "replay_buffer" in lower and ("to_numpy" in src or "from_numpy" in src or ".npz" in src or "pickle" in lower)),
        (
            "total step/count state",
            "total_steps" in lower
            or "total_updates" in lower
            or "global_step" in lower,
        ),
    ]
    missing = [name for name, ok in checks if not ok]
    if missing:
        return ("checkpoint contract is incomplete or unverifiable in inspected source "
                f"({', '.join(os.path.basename(p) for p in paths)}); missing: "
                f"{', '.join(missing)}. Add full-state save/load or pass "
                "--allow-no-resume to explicitly accept restart-from-zero risk.")
    return None


def resume_capability_reason(cmd: str, ckpt_dir, resume_flag, allow_no_resume: bool):
    if allow_no_resume:
        return None
    if not cmd_looks_like_training(cmd):
        return None
    if not ckpt_dir:
        return None
    if cmd_has_resume_flag(cmd):
        return None
    if resume_flag:
        return None
    return ("training-looking task has --ckpt-dir but no resume flag in cmd nor --resume-flag at submit. "
            "On crash/eviction/reboot, relaunch starts from step 0 even though ckpts exist. "
            "Either: (a) add --resume / --resume_from <path> to the cmd, "
            "(b) pass --resume-flag '--resume_from' at submit (scheduler appends '<flag> <ckpt>' on relaunch), "
            "or (c) pass --allow-no-resume to override (e.g. script genuinely cannot resume — must accept replay loss).")
