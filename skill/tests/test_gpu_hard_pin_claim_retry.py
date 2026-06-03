def test_require_gpu_pin_disables_local_claim_gpu_retry(check, sch):
    saved_nodes = dict(sch.NODES)
    saved_run_on = sch.run_on
    saved_claim = sch._ClaimManager.claim
    saved_update_pid = sch._ClaimManager.update_pid
    try:
        sch.NODES.clear()
        sch.NODES.update({
            "n": {"host": "host", "enable_claims": True, "cpu_cores": 8, "ram_mb": 32000}
        })
        calls = []

        def fake_claim(cls, node, task, gpu_idx, node_state=None):
            calls.append(gpu_idx)
            return (False, f"gpu{gpu_idx} over budget", "conflict")

        def fake_update_pid(cls, node, task_id, pid):
            return True

        def fake_run_on(node, cmd, **kw):
            if "test -d" in cmd:
                return (0, "", "")
            return (0, "PID=123\n", "")

        sch._ClaimManager.claim = classmethod(fake_claim)
        sch._ClaimManager.update_pid = classmethod(fake_update_pid)
        sch.run_on = fake_run_on
        task = {
            "id": "tPinned",
            "status": "queued",
            "signature": "ScheduleurmBench/pinned",
            "cmd": "python train.py",
            "cwd": "/repo",
            "env_spec": "none",
            "image": "",
            "extra_env": {},
            "ckpt_glob": "*",
            "resume_flag": "",
            "node": "n",
            "gpu_idx": 1,
            "require_gpu_idx": 1,
            "est_vram_mb": 1000,
            "ram_mb": 1000,
            "cpu_cores": 1,
            "remote_pids": [],
        }
        node_state = {
            "name": "n",
            "alive": True,
            "free_cpu": 8,
            "free_ram_mb": 30000,
            "total_ram_mb": 32000,
            "total_cpu": 8,
            "gpus": [
                {"idx": 0, "total_mb": 12000, "used_mb": 0, "free_mb": 12000, "util_pct": 0},
                {"idx": 1, "total_mb": 12000, "used_mb": 0, "free_mb": 12000, "util_pct": 0},
            ],
        }

        ok, msg = sch.LocalBackend().launch(task, node_state=node_state)
        check(
            "require_gpu_idx prevents alternate-GPU claim retry",
            ok is False and calls == [1] and task.get("gpu_idx") == 1
            and "CLAIM_RACE" in msg,
            diag=f"ok={ok} msg={msg} calls={calls} task_gpu={task.get('gpu_idx')}",
        )
    finally:
        sch.NODES.clear()
        sch.NODES.update(saved_nodes)
        sch.run_on = saved_run_on
        sch._ClaimManager.claim = saved_claim
        sch._ClaimManager.update_pid = saved_update_pid
