def test_task_id_allocation_crosses_t9999_without_wrap(check, sch):
    state = {"next_id": 9999, "tasks": [{"id": "t9998"}]}

    first = sch._allocate_task_id(state)
    state["tasks"].append({"id": first})
    second = sch._allocate_task_id(state)

    check("task id allocation emits t9999",
          first == "t9999",
          diag=f"first={first}")
    check("task id allocation continues to t10000 without wrap",
          second == "t10000",
          diag=f"second={second}")
    check("task id allocation advances next_id past t10000",
          state["next_id"] == 10001,
          diag=f"next_id={state.get('next_id')}")


def test_task_id_allocation_repairs_stale_next_id(check, sch):
    state = {
        "next_id": 1,
        "tasks": [{"id": "t0001"}, {"id": "t10000"}, {"id": "manual"}],
    }

    tid = sch._allocate_task_id(state)

    check("stale next_id skips existing numeric task ids",
          tid == "t10001",
          diag=f"tid={tid}")
    check("stale next_id repair stores following counter",
          state["next_id"] == 10002,
          diag=f"next_id={state.get('next_id')}")


def test_task_id_log_paths_keep_full_post_9999_id(check, sch):
    tid = "t10000"
    local_log = f"{sch.STATE_DIR}/logs/{tid}.log"
    remote_log = f"/tmp/sched_{tid}.log"
    windows_log = sch.WindowsBackend()._log_path({"id": tid, "node": "jtl110cpu"})

    check("local log path keeps full t10000 id",
          local_log.endswith("/logs/t10000.log"),
          diag=local_log)
    check("remote log path keeps full t10000 id",
          remote_log == "/tmp/sched_t10000.log",
          diag=remote_log)
    check("windows log path keeps full t10000 id",
          windows_log.endswith(r"\logs\t10000.log"),
          diag=windows_log)
