from __future__ import annotations

from skill.scheduler_environment.docker_preload import DockerPreloadDeps, preload_docker_images_outside_lock


class _EnvDeploy:
    def __init__(self, calls, *, has_image=False, conda_ok=True, docker_ok=True):
        self.calls = calls
        self.has_image_value = has_image
        self.conda_ok = conda_ok
        self.docker_ok = docker_ok

    def parse_env_spec(self, spec):
        self.calls.append(("parse_env_spec", spec))
        if spec == "bad":
            raise ValueError("bad spec")
        if spec.startswith("docker:"):
            return "docker", spec.split(":", 1)[1]
        if spec.startswith("conda:"):
            return "conda", spec.split(":", 1)[1]
        if spec == "auto":
            return "auto", ""
        return spec, ""

    def get_image_digest(self, run_on, node, image):
        self.calls.append(("get_image_digest", node, image))
        return f"digest:{image}"

    def has_docker(self, run_on, node, timeout=8):
        self.calls.append(("has_docker", node, timeout))
        return self.docker_ok

    def has_image(self, run_on, node, image, local_digest=None):
        self.calls.append(("has_image", node, image, local_digest))
        return self.has_image_value

    def push_image(self, node_host, image, timeout_s=1800):
        self.calls.append(("push_image", node_host, image, timeout_s))
        return True, "ok"

    def push_conda_env(self, node_host, src, dst, timeout_s=3600):
        self.calls.append(("push_conda_env", node_host, src, dst, timeout_s))
        return self.conda_ok, "conda-msg"


def _task(task_id, **overrides):
    task = {
        "id": task_id,
        "status": "queued",
        "env_spec": "none",
    }
    task.update(overrides)
    return task


def _deps(*, calls, state, env_deploy=None):
    def notify(event, payload, **kwargs):
        calls.append(("notify", event, payload, kwargs))

    return DockerPreloadDeps(
        env_deploy=env_deploy,
        load_state=lambda: state,
        node_configs={
            "local": {"host": None},
            "node1": {"host": "node1-host"},
            "node2": {"host": "node2-host"},
            "win1": {"host": "win1-host"},
        },
        node_is_windows=lambda node: node == "win1",
        run_on=lambda *args, **kwargs: (0, "", ""),
        notify=notify,
        record_conda_sync_ok=lambda node, path: calls.append(("conda_ok", node, path)),
        record_conda_sync_failed=lambda node, path: calls.append(("conda_failed", node, path)),
    )


def test_no_env_deploy_returns_without_loading_state():
    calls = []

    preload_docker_images_outside_lock(
        deps=DockerPreloadDeps(
            env_deploy=None,
            load_state=lambda: calls.append(("load_state",)) or {"tasks": []},
            node_configs={},
            node_is_windows=lambda node: False,
            run_on=lambda *args, **kwargs: (0, "", ""),
            notify=lambda *args, **kwargs: calls.append(("notify", args, kwargs)),
            record_conda_sync_ok=lambda node, path: calls.append(("conda_ok", node, path)),
            record_conda_sync_failed=lambda node, path: calls.append(("conda_failed", node, path)),
        )
    )

    assert calls == []


def test_docker_preload_uses_target_filter_and_skips_windows():
    calls = []
    env = _EnvDeploy(calls)
    state = {
        "tasks": [
            _task("skip", env_spec="docker:unused"),
            _task("target", env_spec="docker:img:v1"),
            _task("win", env_spec="docker:img:v1", require_node="win1"),
        ]
    }

    preload_docker_images_outside_lock(
        task_ids={"target", "win"},
        deps=_deps(calls=calls, state=state, env_deploy=env),
    )

    assert ("push_image", "node1-host", "img:v1", 1800) in calls
    assert ("push_image", "node2-host", "img:v1", 1800) in calls
    assert not any(call[0] == "push_image" and call[1] == "win1-host" for call in calls)
    assert not any(call == ("parse_env_spec", "docker:unused") for call in calls)
    assert any(call[0] == "notify" and call[1] == "preload_image_ok" for call in calls)


def test_auto_env_uses_task_image_when_spec_payload_missing():
    calls = []
    env = _EnvDeploy(calls)
    state = {"tasks": [_task("auto", env_spec="auto", image="auto-img", require_node="node1")]}

    preload_docker_images_outside_lock(deps=_deps(calls=calls, state=state, env_deploy=env))

    assert ("get_image_digest", "local", "auto-img") in calls
    assert ("push_image", "node1-host", "auto-img", 1800) in calls


def test_docker_preload_skips_push_when_remote_image_matches_digest():
    calls = []
    env = _EnvDeploy(calls, has_image=True)
    state = {"tasks": [_task("t1", env_spec="docker:img:v1", require_node="node1")]}

    preload_docker_images_outside_lock(deps=_deps(calls=calls, state=state, env_deploy=env))

    assert ("has_image", "node1", "img:v1", "digest:img:v1") in calls
    assert not any(call[0] == "push_image" for call in calls)


def test_conda_preload_records_success_for_existing_remote_target(tmp_path):
    calls = []
    env_path = tmp_path / "env"
    env_path.mkdir()
    env = _EnvDeploy(calls, conda_ok=True)
    state = {
        "tasks": [
            _task("conda", env_spec=f"conda:{env_path}", require_node="node1"),
            _task("local", env_spec=f"conda:{env_path}", require_node="local"),
        ]
    }

    preload_docker_images_outside_lock(deps=_deps(calls=calls, state=state, env_deploy=env))

    assert ("push_conda_env", "node1-host", str(env_path), str(env_path), 3600) in calls
    assert ("conda_ok", "node1", str(env_path)) in calls
    assert not any(call[0] == "push_conda_env" and call[1] is None for call in calls)
    assert any(call[0] == "notify" and call[1] == "preload_conda_ok" for call in calls)


def test_conda_preload_records_failed_sync(tmp_path):
    calls = []
    env_path = tmp_path / "env"
    env_path.mkdir()
    env = _EnvDeploy(calls, conda_ok=False)
    state = {"tasks": [_task("conda", env_spec=f"conda:{env_path}", require_node="node1")]}

    preload_docker_images_outside_lock(deps=_deps(calls=calls, state=state, env_deploy=env))

    assert ("conda_failed", "node1", str(env_path)) in calls
    assert any(call[0] == "notify" and call[1] == "preload_conda_failed" for call in calls)
