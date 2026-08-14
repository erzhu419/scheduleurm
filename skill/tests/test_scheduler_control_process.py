from pathlib import Path

from skill import scheduler_control_process as control


REPO = Path("/home/erzhu419/mine_code/scheduleurm")
SCHEDULER_FILE = REPO / "skill" / "scheduler.py"
SUBCOMMANDS = {"watch", "status", "task-log"}
MARKERS = ("/.claude/skills/scheduler/", "/scheduleurm/skill/")


def test_is_repo_path_accepts_root_and_children_only():
    assert control.is_repo_path(str(REPO), repo_root=REPO)
    assert control.is_repo_path(str(REPO / "skill" / "scheduler.py"), repo_root=REPO)
    assert not control.is_repo_path("/home/erzhu419/mine_code/scheduleurm2", repo_root=REPO)
    assert not control.is_repo_path("/home/erzhu419/mine_code/TransitDuet", repo_root=REPO)


def test_is_scheduler_control_cmdline_matches_known_scheduler_invocations():
    assert control.is_scheduler_control_cmdline(
        f"python3 {SCHEDULER_FILE} watch --interval 60",
        scheduler_file=SCHEDULER_FILE,
        control_subcommands=SUBCOMMANDS,
        path_markers=MARKERS,
    )
    assert control.is_scheduler_control_cmdline(
        "python3 /home/erzhu419/.claude/skills/scheduler/scheduler.py status",
        scheduler_file=SCHEDULER_FILE,
        control_subcommands=SUBCOMMANDS,
        path_markers=MARKERS,
    )
    assert not control.is_scheduler_control_cmdline(
        f"python3 {SCHEDULER_FILE} unknown-subcommand",
        scheduler_file=SCHEDULER_FILE,
        control_subcommands=SUBCOMMANDS,
        path_markers=MARKERS,
    )
    assert not control.is_scheduler_control_cmdline(
        "python3 /tmp/not-scheduler.py watch",
        scheduler_file=SCHEDULER_FILE,
        control_subcommands=SUBCOMMANDS,
        path_markers=MARKERS,
    )


def test_is_scheduler_control_process_filters_repo_tests_and_helpers_only():
    common = {
        "repo_root": REPO,
        "scheduler_file": SCHEDULER_FILE,
        "control_subcommands": SUBCOMMANDS,
        "path_markers": MARKERS,
    }
    assert control.is_scheduler_control_process(str(REPO), "python3 -m pytest -q", **common)
    assert control.is_scheduler_control_process(str(REPO), "python3 -", **common)
    assert control.is_scheduler_control_process(str(REPO), "python3 -u -", **common)
    assert control.is_scheduler_control_process(str(REPO), "coverage run -m pytest", **common)
    assert control.is_scheduler_control_process(
        "/home/zhengliang01/scheduleurm_work/TransitDuet",
        f"python3 {SCHEDULER_FILE} task-log t1",
        **common,
    )
    assert not control.is_scheduler_control_process(
        "/home/zhengliang01/scheduleurm_work/TransitDuet",
        "python3 train_transit.py --seed 1",
        **common,
    )
    assert not control.is_scheduler_control_process(
        "/home/zhengliang01/scheduleurm_work/TransitDuet",
        "python3 -m pytest -q",
        **common,
    )
