from __future__ import annotations

from contextlib import contextmanager

from skill.scheduler_eta.tail_snapshot import (
    EtaTailSnapshotDeps,
    eta_tail_snapshot_outside_lock,
)


def test_eta_tail_snapshot_tails_after_releasing_state_lock():
    events = []
    lock_depth = {"value": 0}
    state = {"tasks": [{"id": "t1", "status": "running"}]}

    @contextmanager
    def state_lock(**kwargs):
        events.append(("lock_enter", kwargs))
        lock_depth["value"] += 1
        try:
            yield
        finally:
            lock_depth["value"] -= 1
            events.append(("lock_exit", None))

    def tail_outputs(by_node):
        events.append(("tail", lock_depth["value"], by_node))
        return {"node001": "tail"}

    out, ids = eta_tail_snapshot_outside_lock(
        "unit",
        deps=EtaTailSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: state,
            eta_refresh_due=lambda snapshot: True,
            eta_tail_targets=lambda snapshot: ({"node001": [({"id": "t1"}, "/tmp/t.log")]}, []),
            eta_tail_outputs_by_node=tail_outputs,
            notify=lambda *args, **kwargs: None,
        ),
    )

    assert out == {"node001": "tail"}
    assert ids == {"t1"}
    assert events[0] == ("lock_enter", {"shared": True, "purpose": "unit:snapshot"})
    assert events[1] == ("lock_exit", None)
    assert events[2][0:2] == ("tail", 0)
