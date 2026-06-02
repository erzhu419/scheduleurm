def test_tui_slurm_status_mapping(check, sch):
    import tui

    check("TUI maps running Slurm states to running",
          tui._slurm_status_for_tui("RUNNING") == "running"
          and tui._slurm_status_for_tui("COMPLETING") == "running")
    check("TUI maps pending Slurm states to queued",
          tui._slurm_status_for_tui("PENDING") == "queued"
          and tui._slurm_status_for_tui("CONFIGURING") == "queued")
    check("TUI preserves terminal Slurm states",
          tui._slurm_status_for_tui("FAILED") == "failed"
          and tui._slurm_status_for_tui("OUT_OF_MEMORY") == "out_of_memory")
    check("TUI maps unknown Slurm states to launching",
          tui._slurm_status_for_tui("BOOT_FAIL") == "launching")


def test_tui_virtual_slurm_tasks_from_nodes(check, sch):
    import tui

    nodes = [
        {
            "name": "node001",
            "alive": True,
            "slurm_cluster": True,
            "slurm_jobs": [
                {
                    "job_id": "101",
                    "state": "RUNNING",
                    "bucket": "gpu",
                    "nodes": "node001",
                    "name": "train-a",
                    "user": "alice",
                    "cpus": "4",
                    "mem": "16G",
                    "gres": "gpu:1",
                },
                {
                    "job_id": "102",
                    "state": "PENDING",
                    "bucket": "cpu",
                    "nodes": "node001",
                    "name": "prep",
                    "user": "bob",
                    "cpus": "2",
                },
            ],
        },
        {
            "name": "node002",
            "alive": False,
            "slurm_cluster": True,
            "slurm_jobs": [{"job_id": "201", "state": "RUNNING"}],
        },
        {
            "name": "node003",
            "alive": True,
            "slurm_cluster": False,
            "slurm_jobs": [{"job_id": "301", "state": "RUNNING"}],
        },
    ]
    existing = [
        {"id": "tracked", "slurm_job_id": "102"},
        {"id": "slurm:999"},
    ]
    rows = tui._virtual_slurm_tasks_from_nodes(nodes, existing)
    check("TUI creates virtual rows only for live Slurm-cluster jobs",
          [row["id"] for row in rows] == ["slurm:101"],
          diag=str(rows))
    row = rows[0] if rows else {}
    check("TUI virtual Slurm GPU job exposes task-table fields",
          row.get("status") == "running"
          and row.get("project") == "train-a"
          and row.get("est_vram_mb") == 1
          and row.get("auto_adopted") is True
          and row.get("origin") == "external"
          and row.get("slurm_job_id") == "101",
          diag=str(row))

    duplicate_by_id = tui._virtual_slurm_tasks_from_nodes(nodes, [{"id": "slurm:101"}])
    check("TUI suppresses virtual Slurm rows that collide with existing task id",
          all(row["id"] != "slurm:101" for row in duplicate_by_id),
          diag=str(duplicate_by_id))
