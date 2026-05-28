from algorithm.action_model import (
    action_candidate_jsonl_row,
    assignment_from_gpu_candidate,
    single_assignment_action,
)
from algorithm.candidates import CandidateRecord, active_bucket_representatives, approximation_report
from algorithm.features import (
    class_key,
    finite_feature_metric,
    gpu_candidate_features,
    resource_bucket,
    task_kind,
)
from algorithm.scoring import RobustScoreWeights, score_gpu_candidate


def _task(**overrides):
    task = {
        "id": "t-feat",
        "project": "projA",
        "signature": "projA/run/train/seed1",
        "cmd": "python train.py --seed 1",
        "priority": "high",
        "submitted_at": 100.0,
        "est_vram_mb": 4096,
        "ram_mb": 8192,
        "cpu_cores": 2,
    }
    task.update(overrides)
    return task


def _node_gpu():
    node = {"name": "nodeA"}
    gpu = {
        "idx": 0,
        "running_task_count": 2,
        "used_mb": 6000,
        "free_mb": 18000,
        "total_mb": 24000,
        "util_pct": 40,
    }
    return node, gpu


def test_algorithm_feature_surface_is_finite_and_stable(check, sch):
    task = _task()
    node, gpu = _node_gpu()
    features = gpu_candidate_features(
        task,
        node,
        gpu,
        {
            "now_ts": 220.0,
            "gpu_empty_used_mb": 200,
            "sweet_spot_tasks_per_gpu": 3,
            "legacy_score": (1, 0, 30.0, 0.4),
        },
    )
    check("algorithm task kind distinguishes gpu train",
          task_kind(task) == "gpu_train",
          diag=task_kind(task))
    check("algorithm class key includes finite resource bucket",
          resource_bucket(task) in class_key(task),
          diag=class_key(task))
    check("algorithm candidate bucket is finite-feature key",
          str(features.get("candidate_bucket", "")).startswith("bucket:v1|"),
          diag=str(features))
    check("algorithm feature map records proof-facing class/regime",
          bool(features.get("class_key") and features.get("regime_key")),
          diag=str(features))


def test_algorithm_metric_and_score_are_auditable(check, sch):
    task = _task()
    node, gpu = _node_gpu()
    f1 = gpu_candidate_features(task, node, gpu, {"now_ts": 220.0, "legacy_score": (1, 0, 30.0)})
    gpu2 = dict(gpu)
    gpu2.update({"used_mb": 12000, "util_pct": 80, "running_task_count": 4})
    f2 = gpu_candidate_features(task, node, gpu2, {"now_ts": 220.0, "legacy_score": (1, 0, 30.0)})
    dist = finite_feature_metric(f1, f2)
    check("algorithm finite-feature metric notices interference change",
          dist > 0.0,
          diag=str(dist))
    weights = RobustScoreWeights(sweet_spot_tasks_per_gpu=3)
    score1 = score_gpu_candidate(f1, weights)
    score2 = score_gpu_candidate(f2, weights)
    check("algorithm robust score exposes bounded components",
          "vram_pressure" in score1.components and "colocation_penalty" in score1.components,
          diag=str(score1.snapshot()))
    check("algorithm robust score penalizes heavier candidate",
          score2.score > score1.score,
          diag=f"{score1.snapshot()} -> {score2.snapshot()}")


def test_algorithm_active_bucket_representatives(check, sch):
    task = _task()
    node, gpu = _node_gpu()
    f1 = gpu_candidate_features(task, node, gpu, {"now_ts": 220.0})
    f2 = dict(f1)
    f2["post_vram_frac"] = f1["post_vram_frac"] + 0.01
    f2["candidate_bucket"] = f1["candidate_bucket"]
    gpu3 = dict(gpu)
    gpu3.update({"idx": 1, "running_task_count": 0, "used_mb": 0, "util_pct": 0})
    f3 = gpu_candidate_features(task, {"name": "nodeA"}, gpu3, {"now_ts": 220.0})
    records = [
        CandidateRecord("nodeA", 0, (2.0,), f1, {"name": "worse"}),
        CandidateRecord("nodeA", 0, (1.0,), f2, {"name": "better"}),
        CandidateRecord("nodeA", 1, (3.0,), f3, {"name": "other_bucket"}),
    ]
    selected = active_bucket_representatives(records, max_per_bucket=1)
    report = approximation_report(records, selected)
    check("algorithm active-bucket keeps one representative per bucket",
          report["selected_bucket_count"] == report["full_bucket_count"],
          diag=str(report))
    check("algorithm active-bucket picks best score inside bucket",
          any(r.audit.get("name") == "better" for r in selected),
          diag=str([r.snapshot() for r in selected]))


def test_algorithm_global_action_schema(check, sch):
    task = _task(id="t-action")
    node, gpu = _node_gpu()
    features = gpu_candidate_features(task, node, gpu, {"now_ts": 220.0})
    assignment = assignment_from_gpu_candidate(
        task, node["name"], gpu["idx"], features)
    action = single_assignment_action(
        assignment, score=(1.0, 2.0), metadata={"policy": "unit"})
    row = action_candidate_jsonl_row("slot-001", action, chosen=True)
    check("algorithm global action has stable id",
          row["action_id"] == action.action_id and row["action_id"].startswith("a_"),
          diag=str(row))
    check("algorithm global action records complete assignment",
          row["assignments"][0]["task_id"] == "t-action"
          and row["assignments"][0]["node"] == "nodeA"
          and row["assignments"][0]["gpu_idx"] == 0,
          diag=str(row))
    check("algorithm global action exposes bucket/class signatures",
          bool(row["bucket_signature"] and row["class_signature"]),
          diag=str(row))
