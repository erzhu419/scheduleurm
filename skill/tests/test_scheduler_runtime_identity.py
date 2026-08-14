from __future__ import annotations

from skill.scheduler_runtime_identity import (
    candidate_runtime_seconds,
    runtime_device_kind_for_task,
    runtime_node_bucket_key,
    runtime_node_bucket_keys,
    task_runtime_keys,
    task_runtime_payload,
)


def _payload(task):
    return task_runtime_payload(
        task,
        conflict_path_key=lambda value: str(value or "").rstrip("/"),
    )


def test_task_runtime_payload_normalizes_runtime_fields():
    payload = task_runtime_payload(
        {
            "signature": "sig",
            "project": "proj",
            "description": "desc",
            "cmd": "python train.py",
            "cwd": "/work/project/",
            "env_spec": "",
            "image": "img",
            "extra_env": ["not", "a", "dict"],
        },
        conflict_path_key=lambda value: f"key:{value}",
    )

    assert payload == {
        "signature": "sig",
        "project": "proj",
        "description": "desc",
        "cmd": "python train.py",
        "cwd": "key:/work/project/",
        "env_spec": "none",
        "image": "img",
        "extra_env": {},
    }


def test_task_runtime_keys_ignore_metadata_for_exact_key_but_keep_signature_key():
    base = {
        "signature": "sig-a",
        "project": "proj-a",
        "description": "first",
        "cmd": "python train.py --seed 1",
        "cwd": "/work/project",
        "env_spec": "jax",
        "image": "image-a",
        "extra_env": {"B": "2", "A": "1"},
    }
    changed_metadata = dict(
        base,
        signature="sig-b",
        project="proj-b",
        description="second",
        priority="high",
        cpu_cores=64,
    )

    first = task_runtime_keys(base, task_runtime_payload=_payload)
    second = task_runtime_keys(changed_metadata, task_runtime_payload=_payload)

    assert first[0][1] == "exact"
    assert first[0][0] == second[0][0]
    assert first[1][:2] == ("sig:sig-a", "signature")
    assert second[1][:2] == ("sig:sig-b", "signature")

    changed_runtime = dict(base, cmd="python eval.py --seed 1")
    assert task_runtime_keys(changed_runtime, task_runtime_payload=_payload)[0][0] != first[0][0]


def test_task_runtime_keys_without_cmd_only_use_signature_key():
    keys = task_runtime_keys({"signature": "sig-only", "cwd": "/work"}, task_runtime_payload=_payload)

    assert keys == [("sig:sig-only", "signature", _payload({"signature": "sig-only", "cwd": "/work"}))]


def test_runtime_device_kind_prefers_cpu_for_cpu_launches_or_no_gpu():
    assert runtime_device_kind_for_task(
        {"launch_cpu_mode": False},
        None,
        task_launch_cpu_mode=lambda task: False,
    ) == "cpu"
    assert runtime_device_kind_for_task(
        {"launch_cpu_mode": False},
        0,
        task_launch_cpu_mode=lambda task: False,
    ) == "gpu"
    assert runtime_device_kind_for_task(
        {"launch_cpu_mode": True},
        0,
        task_launch_cpu_mode=lambda task: True,
    ) == "cpu"


def test_runtime_node_bucket_helpers_canonicalize_aliases_to_current_bucket():
    aliases = {"node007-direct": "node007", "node-one": "node001"}

    assert runtime_node_bucket_key(
        "",
        "",
        canonical_node_name=lambda name: aliases.get(name, name),
    ) == "?:?"

    buckets = runtime_node_bucket_keys(
        "node007",
        "gpu",
        canonical_node_name=lambda name: aliases.get(name, name),
        node_name_aliases=aliases,
    )

    assert buckets == ["node007:gpu", "node007:gpu"]


def test_candidate_runtime_seconds_reads_node_device_bucket_and_fallback_buckets():
    payload = _payload(
        {
            "signature": "sig-a",
            "cmd": "python train.py",
            "cwd": "/work/project",
            "env_spec": "jax",
            "extra_env": {"A": "1"},
        }
    )
    exact_key = task_runtime_keys(payload, task_runtime_payload=lambda task: task)[0][0]
    history = {
        exact_key: {
            "node_runtime": {
                "node001:gpu": {"total_s": 77},
                "node001:cpu": {"total_s": 123},
            }
        },
        "sig:sig-a": {
            "node_runtime": {
                "node002:gpu": {"total_s": 44},
            }
        },
    }

    assert candidate_runtime_seconds(
        payload,
        "node001",
        0,
        load_runtime_history=lambda: history,
        task_runtime_keys=lambda task: task_runtime_keys(task, task_runtime_payload=lambda item: item),
        runtime_device_kind_for_task=lambda task, gpu_idx: "gpu" if gpu_idx is not None else "cpu",
        runtime_node_bucket_keys=lambda node, device: [f"{node}:{device}"],
    ) == 77
    assert candidate_runtime_seconds(
        payload,
        "node002",
        0,
        load_runtime_history=lambda: history,
        task_runtime_keys=lambda task: [("sig:sig-a", "signature", task)],
        runtime_device_kind_for_task=lambda task, gpu_idx: "gpu",
        runtime_node_bucket_keys=lambda node, device: [f"{node}:{device}"],
    ) == 44
    assert candidate_runtime_seconds(
        payload,
        "node004",
        0,
        load_runtime_history=lambda: {
            exact_key: {"node_runtime": {"legacy-node004:gpu": {"total_s": 66}}}
        },
        task_runtime_keys=lambda task: task_runtime_keys(task, task_runtime_payload=lambda item: item),
        runtime_device_kind_for_task=lambda task, gpu_idx: "gpu",
        runtime_node_bucket_keys=lambda node, device: [f"{node}:{device}", f"legacy-{node}:{device}"],
    ) == 66
    assert candidate_runtime_seconds(
        payload,
        "node003",
        0,
        load_runtime_history=lambda: history,
        task_runtime_keys=lambda task: [("missing", "exact", task)],
        runtime_device_kind_for_task=lambda task, gpu_idx: "gpu",
        runtime_node_bucket_keys=lambda node, device: [f"{node}:{device}"],
    ) == 0
