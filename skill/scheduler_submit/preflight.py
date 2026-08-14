"""Submit command policy preflight."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SubmitPreflightDeps:
    submit_policy_text: Callable[[str, str], str]
    infer_checkpoint_from_submit: Callable[[str, str], dict | None]
    canonical_node_name: Callable[[Any], str]
    canonicalize_node_list: Callable[[Any], list]
    simple_sac_large_data_reason: Callable[[str, str], str]
    cpu_training_policy_reason: Callable[[str, str, bool, int], str | None]
    task_looks_like_training: Callable[[str, str], bool]
    cmd_looks_like_training: Callable[[str], bool]
    resume_capability_reason: Callable[[str, str, str, bool], str | None]
    checkpoint_contract_reason: Callable[[str, str, str, str, bool], str | None]
    same_declared_path: Callable[[Any, Any], bool]


@dataclass(frozen=True)
class SubmitPreflightResult:
    raw_submit_cmd: str
    policy_cmd: str
    inferred_ckpt_dir: str = ""
    inferred_result_dir: str = ""
    inferred_resume_managed: bool = False
    inferred_ckpt_source: str = ""
    ckpt_dir_was_inferred: bool = False
    notes: list[str] = field(default_factory=list)


class SubmitPreflightRefusal(Exception):
    def __init__(self, lines: list[str], code: int = 2):
        super().__init__("\n".join(lines))
        self.lines = lines
        self.code = code


def _refuse(*lines: str, code: int = 2) -> None:
    raise SubmitPreflightRefusal(list(lines), code=code)


def run_submit_preflight(
    args: Any,
    *,
    deps: SubmitPreflightDeps,
    default_vram_mb: int,
    min_cpu_training_justification_len: int = 30,
) -> SubmitPreflightResult:
    """Validate and normalize submit args before state-lock work begins."""
    raw_submit_cmd = args.cmd
    policy_cmd = deps.submit_policy_text(args.cmd, args.cwd)
    inferred_checkpoint = deps.infer_checkpoint_from_submit(args.cmd, args.cwd)
    inferred_ckpt_dir = ""
    inferred_result_dir = ""
    inferred_resume_managed = False
    inferred_ckpt_source = ""
    if inferred_checkpoint:
        inferred_ckpt_dir = inferred_checkpoint.get("ckpt_dir") or ""
        inferred_result_dir = inferred_checkpoint.get("result_dir") or ""
        inferred_resume_managed = bool(inferred_checkpoint.get("resume_managed_by_cmd"))
        inferred_ckpt_source = inferred_checkpoint.get("source") or ""

    ckpt_dir_was_inferred = False
    if not args.ckpt_dir and inferred_ckpt_dir:
        args.ckpt_dir = inferred_ckpt_dir
        ckpt_dir_was_inferred = True
    if not getattr(args, "result_dir", None) and inferred_result_dir:
        args.result_dir = inferred_result_dir
    if getattr(args, "require_node", None):
        args.require_node = deps.canonical_node_name(args.require_node)
    if getattr(args, "preferred_node", None):
        args.preferred_node = deps.canonical_node_name(args.preferred_node)
    if getattr(args, "allowed_nodes", None):
        args.allowed_nodes = deps.canonicalize_node_list(args.allowed_nodes)

    notes: list[str] = []
    large_data_reason = deps.simple_sac_large_data_reason(raw_submit_cmd, args.cwd)
    if large_data_reason and not getattr(args, "allow_remote_large_data", False):
        if args.require_node and args.require_node != "local":
            _refuse(
                "REFUSED: task appears to require large local-only SimpleSAC data.",
                f"  reason: {large_data_reason}",
                "  Either keep --require-node local, disable snapshot/per-file data, "
                "or pass --allow-remote-large-data after manually staging the data.",
            )
        if args.preferred_node and args.preferred_node != "local":
            _refuse(
                "REFUSED: task prefers a remote node but appears to require large "
                "local-only SimpleSAC data.",
                f"  reason: {large_data_reason}",
                "  Either keep it local, disable snapshot/per-file data, or pass "
                "--allow-remote-large-data after manually staging the data.",
            )
        if args.require_node != "local":
            args.require_node = "local"
            notes.append(f"NOTE: forcing --require-node local: {large_data_reason}")

    submit_vram_for_policy = args.vram if args.vram is not None else default_vram_mb
    cpu_training_reason = deps.cpu_training_policy_reason(
        policy_cmd,
        args.description,
        bool(getattr(args, "allow_cpu_training", False)),
        submit_vram_for_policy,
    )
    if cpu_training_reason:
        _refuse(
            "REFUSED: cmd looks like training but scheduler CPU/GPU policy is inconsistent.",
            f"  cmd: {args.cmd[:120]}",
            f"  reason: {cpu_training_reason}",
            "  Either:",
            "    (a) submit with --vram <N> and remove/replace CPU device flags  (use GPU)",
            "    (b) pass --allow-cpu-training AND --cpu-training-justification '<reason>'",
        )

    if (
        bool(getattr(args, "allow_cpu_training", False))
        and deps.task_looks_like_training(policy_cmd, args.description)
    ):
        just = (getattr(args, "cpu_training_justification", "") or "").strip()
        if len(just) < min_cpu_training_justification_len:
            _refuse(
                "REFUSED: --allow-cpu-training requires --cpu-training-justification "
                f"with >={min_cpu_training_justification_len} chars of explanation.",
                f"  cmd: {args.cmd[:120]}",
                "  Reason: training tasks should default to GPU. CPU training is the exception, "
                "not the rule. Document why GPU isn't right HERE so it's auditable later.",
                "  Example: --cpu-training-justification "
                "'tiny MLP, GPU saturated by other priority work, completion in <30 min'",
                f"  Currently passed: {just[:60]!r} ({len(just)} chars)",
            )

    if (
        deps.cmd_looks_like_training(policy_cmd)
        and not args.ckpt_dir
        and not getattr(args, "allow_no_ckpt", False)
    ):
        _refuse(
            "REFUSED: cmd looks like training but --ckpt-dir is not set.",
            f"  cmd: {args.cmd[:120]}",
            "  Without ckpt-dir, eviction/crash loses ALL progress (relaunches start at step 0).",
            "  Either:",
            "    (a) submit with --ckpt-dir <abs-path-on-target> and --resume-flag '--resume_from'  (recommended)",
            "    (b) pass --allow-no-ckpt  (override; OK for short debug runs / one-shot evals)",
        )

    resume_reason = deps.resume_capability_reason(
        policy_cmd,
        args.ckpt_dir,
        args.resume_flag,
        bool(getattr(args, "allow_no_resume", False)),
    )
    if resume_reason:
        _refuse(
            "REFUSED: cmd looks like training but resume is not wired up.",
            f"  cmd: {args.cmd[:120]}",
            f"  reason: {resume_reason}",
        )

    contract_reason = deps.checkpoint_contract_reason(
        raw_submit_cmd,
        args.cwd,
        args.ckpt_dir,
        args.resume_flag,
        bool(getattr(args, "allow_no_resume", False)),
    )
    if contract_reason:
        _refuse(
            "REFUSED: training checkpoint/resume contract is not verifiable.",
            f"  cmd: {raw_submit_cmd[:120]}",
            f"  reason: {contract_reason}",
            "  If this is a short/debug run, pass --allow-no-resume. Otherwise fix the "
            "training code so checkpoint save/load includes iteration, model params, "
            "optimizer state, replay/buffer state, and total step counters.",
        )

    if args.ckpt_dir and deps.same_declared_path(args.ckpt_dir, args.cwd):
        _refuse(
            "REFUSED: --ckpt-dir must not equal --cwd.",
            f"  cwd:      {args.cwd}",
            f"  ckpt-dir: {args.ckpt_dir}",
            "  reason: launch staging uses rsync --delete for cwd; if ckpts live at cwd root, "
            "remote checkpoint/output files can be deleted during code sync.",
            "  Put checkpoints in a dedicated subdirectory, e.g. --ckpt-dir <cwd>/checkpoints/<run>.",
        )

    result_dir = getattr(args, "result_dir", None)
    if result_dir and deps.same_declared_path(result_dir, args.cwd):
        _refuse(
            "REFUSED: --result-dir must not equal --cwd.",
            f"  cwd:        {args.cwd}",
            f"  result-dir: {result_dir}",
            "  reason: launch staging uses rsync --delete for cwd; result files written at cwd root "
            "are indistinguishable from stale code files.",
            "  Put results in a dedicated subdirectory, e.g. --result-dir <cwd>/results/<run>.",
        )

    return SubmitPreflightResult(
        raw_submit_cmd=raw_submit_cmd,
        policy_cmd=policy_cmd,
        inferred_ckpt_dir=inferred_ckpt_dir,
        inferred_result_dir=inferred_result_dir,
        inferred_resume_managed=inferred_resume_managed,
        inferred_ckpt_source=inferred_ckpt_source,
        ckpt_dir_was_inferred=ckpt_dir_was_inferred,
        notes=notes,
    )
