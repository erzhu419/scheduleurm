import os
from pathlib import Path


def test_algorithm_directory_is_scheduler_sibling(check, sch):
    skill_dir = Path(sch.__file__).resolve().parent
    check("algorithm directory is not nested under skill",
          not (skill_dir / "algorithm").exists())
    check("algorithm directory exists beside skill",
          (skill_dir.parent / "algorithm" / "placement.py").exists())


def test_algorithm_policy_loads(check, sch):
    policy = sch._configure_algorithm("legacy")
    check("algorithm legacy loads", getattr(policy, "name", None) == "legacy")

    policy = sch._configure_algorithm("sweetspot_v1")
    check("algorithm sweetspot_v1 loads", getattr(policy, "name", None) in ("sweetspot_v1", "sweetspot"))

    policy = sch._configure_algorithm("legacy")
    check("algorithm reset to legacy", getattr(policy, "name", None) == "legacy")


def test_algorithm_gpu_score_is_optional(check, sch):
    sch._configure_algorithm("legacy")
    legacy = (1, 0, 10, 0.5, 1000)
    task = {"id": "t-score", "est_vram_mb": 500}
    node = {"name": "n0"}
    gpu = {"idx": 0, "running_task_count": 2, "used_mb": 1000, "total_mb": 12000, "util_pct": 50}
    check("legacy algorithm score unchanged",
          sch._algorithm_gpu_score(task, node, gpu, legacy) == legacy)

    sch._configure_algorithm("sweetspot_v1")
    scored = sch._algorithm_gpu_score(task, node, gpu, legacy)
    check("sweetspot algorithm score extends legacy",
          isinstance(scored, tuple) and scored[-len(legacy):] == legacy,
          diag=str(scored))
    audit = sch._algorithm_selected_gpu_audit(task, node, gpu)
    check("sweetspot algorithm audit has finite bucket",
          bool(audit.get("candidate_bucket") and audit.get("score", {}).get("components")),
          diag=str(audit))
    snap = sch._algorithm_config_snapshot()
    check("sweetspot algorithm snapshot records score weights",
          "score_weights" in snap,
          diag=str(snap))
    sch._configure_algorithm("legacy")


def test_algorithm_gpu_admission_params(check, sch):
    old = os.environ.get("SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU")
    os.environ["SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU"] = "3"
    try:
        sch._configure_algorithm("sweetspot_v1")
        task = {"est_vram_mb": 500}
        gpu = {
            "idx": 0,
            "running_task_count": 3,
            "used_mb": 1000,
            "free_mb": 9000,
            "total_mb": 12000,
            "util_pct": 20,
        }
        reason = sch._algorithm_gpu_fit_block_reason(task, gpu, {})
        check("sweetspot max task parameter blocks full GPU",
              "task_count 3/3" in reason,
              diag=reason)
    finally:
        if old is None:
            os.environ.pop("SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU", None)
        else:
            os.environ["SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU"] = old
        sch._configure_algorithm("legacy")
