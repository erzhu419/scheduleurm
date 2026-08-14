from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "install.sh"
SKILL_SOURCE = REPO_ROOT / "skill"


def _run_install(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    home = tmp_path / "home"
    install_root = tmp_path / "runtime"
    skill_destination = install_root / "scheduler"
    clean_cwd = tmp_path / "clean-cwd"
    home.mkdir(parents=True, exist_ok=True)
    clean_cwd.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("SCHEDULEURM_CACHE_ARTIFACTS", None)
    env.update(
        {
            "HOME": str(home),
            "SCHEDULEURM_SKILL_DIR": str(skill_destination),
        }
    )
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=clean_cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    return result, install_root, skill_destination


def _seed_existing_runtime(install_root: Path, skill_destination: Path) -> None:
    for path, marker in (
        (skill_destination, "old-skill"),
        (install_root / "algorithm", "old-algorithm"),
        (install_root / "simulation", "old-simulation"),
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / marker).write_text(marker + "\n", encoding="utf-8")


def test_copy_install_publishes_complete_runtime_and_selected_cache(tmp_path):
    install_root = tmp_path / "runtime"
    skill_destination = install_root / "scheduler"
    _seed_existing_runtime(install_root, skill_destination)

    cache_source = tmp_path / "selected service cache.json"
    cache_payload = {"version": 2, "records": []}
    cache_source.write_text(json.dumps(cache_payload), encoding="utf-8")

    result, install_root, skill_destination = _run_install(
        tmp_path,
        "--copy",
        "--no-systemd",
        extra_env={"SCHEDULEURM_CACHE_ARTIFACTS": str(cache_source)},
    )

    assert result.returncode == 0, result.stderr
    assert skill_destination.is_dir()
    assert not skill_destination.is_symlink()
    assert not (skill_destination / "old-skill").exists()
    assert not (install_root / "algorithm" / "old-algorithm").exists()
    assert not (install_root / "simulation" / "old-simulation").exists()

    source_modules = {
        path.relative_to(SKILL_SOURCE)
        for path in SKILL_SOURCE.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing_modules = sorted(
        str(relative)
        for relative in source_modules
        if not (skill_destination / relative).is_file()
    )
    assert missing_modules == []
    assert (install_root / "algorithm" / "theorem_dispatch" / "policy.py").is_file()
    assert (install_root / "simulation" / "fast_forward.py").is_file()
    assert (install_root / "simulation" / "service_cache.py").is_file()

    installed_cache = (
        install_root / "md" / "experiment_artifacts" / cache_source.name
    )
    assert json.loads(installed_cache.read_text(encoding="utf-8")) == cache_payload
    assert list(install_root.glob(".scheduler.stage.*")) == []

    probe_cwd = tmp_path / "installed-probe"
    probe_cwd.mkdir()
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-c",
            """
import os
import pathlib
import sys

skill = pathlib.Path(sys.argv[1]).resolve()
root = skill.parent
sys.path.insert(0, str(skill))
import scheduler
import algorithm
import simulation

expected = {
    scheduler: skill,
    algorithm: root / "algorithm",
    simulation: root / "simulation",
}
for module, expected_root in expected.items():
    module_path = pathlib.Path(module.__file__).resolve()
    expected_root = expected_root.resolve()
    assert os.path.commonpath((str(module_path), str(expected_root))) == str(expected_root)
assert algorithm.load_placement_policy("legacy").name == "legacy"
""",
            str(skill_destination),
        ],
        cwd=probe_cwd,
        env={"HOME": str(tmp_path / "probe-home"), "PATH": os.environ["PATH"]},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stderr


def test_copy_install_succeeds_without_cache_configuration(tmp_path):
    result, install_root, skill_destination = _run_install(
        tmp_path,
        "--copy",
        "--no-systemd",
    )

    assert result.returncode == 0, result.stderr
    assert (skill_destination / "scheduler.py").is_file()
    assert (install_root / "algorithm" / "__init__.py").is_file()
    assert (install_root / "simulation" / "__init__.py").is_file()
    assert not (install_root / "md").exists()
    assert list(install_root.glob(".scheduler.stage.*")) == []


def test_copy_install_rejects_invalid_cache_without_touching_existing_runtime(tmp_path):
    install_root = tmp_path / "runtime"
    skill_destination = install_root / "scheduler"
    _seed_existing_runtime(install_root, skill_destination)
    invalid_cache = tmp_path / "invalid-cache.json"
    invalid_cache.write_text("{not-json", encoding="utf-8")

    result, install_root, skill_destination = _run_install(
        tmp_path,
        "--copy",
        "--no-systemd",
        "--cache-artifact",
        str(invalid_cache),
    )

    assert result.returncode != 0
    assert "invalid JSON cache artifact" in result.stderr
    assert (skill_destination / "old-skill").read_text(encoding="utf-8") == "old-skill\n"
    assert (
        install_root / "algorithm" / "old-algorithm"
    ).read_text(encoding="utf-8") == "old-algorithm\n"
    assert (
        install_root / "simulation" / "old-simulation"
    ).read_text(encoding="utf-8") == "old-simulation\n"
    assert not (skill_destination / "scheduler.py").exists()
    assert list(install_root.glob(".scheduler.stage.*")) == []


def test_copy_install_validation_failure_keeps_existing_runtime(tmp_path):
    install_root = tmp_path / "runtime"
    skill_destination = install_root / "scheduler"
    _seed_existing_runtime(install_root, skill_destination)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/bin/sh\nexit 23\n", encoding="ascii")
    fake_python.chmod(0o755)

    result, install_root, skill_destination = _run_install(
        tmp_path,
        "--copy",
        "--no-systemd",
        extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode == 23
    assert "verifying staged runtime in a clean process" in result.stdout
    assert (skill_destination / "old-skill").is_file()
    assert (install_root / "algorithm" / "old-algorithm").is_file()
    assert (install_root / "simulation" / "old-simulation").is_file()
    assert not (skill_destination / "scheduler.py").exists()
    assert list(install_root.glob(".scheduler.stage.*")) == []


@pytest.mark.skipif(
    not Path("/run/systemd/system").is_dir(),
    reason="installer only enters its systemd branch when systemd is detected",
)
def test_copy_install_rolls_back_published_runtime_on_later_failure(tmp_path):
    install_root = tmp_path / "runtime"
    skill_destination = install_root / "scheduler"
    _seed_existing_runtime(install_root, skill_destination)

    cache_name = "rollback-cache.json"
    installed_cache = install_root / "md" / "experiment_artifacts" / cache_name
    installed_cache.parent.mkdir(parents=True)
    installed_cache.write_text(json.dumps({"generation": "old"}), encoding="utf-8")
    cache_source = tmp_path / cache_name
    cache_source.write_text(json.dumps({"generation": "new"}), encoding="utf-8")

    fake_bin = tmp_path / "fake-systemd-bin"
    fake_bin.mkdir()
    fake_systemctl = fake_bin / "systemctl"
    fake_systemctl.write_text(
        """#!/bin/sh
if [ "$*" = "--user is-active scheduler.service" ]; then
    exit 1
fi
if [ "$*" = "--user start scheduler.service" ]; then
    exit 41
fi
exit 0
""",
        encoding="ascii",
    )
    fake_systemctl.chmod(0o755)

    result, install_root, skill_destination = _run_install(
        tmp_path,
        "--copy",
        extra_env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SCHEDULEURM_CACHE_ARTIFACTS": str(cache_source),
        },
    )

    assert result.returncode == 41
    assert "restoring the previous COPY-mode runtime" in result.stderr
    assert (skill_destination / "old-skill").is_file()
    assert (install_root / "algorithm" / "old-algorithm").is_file()
    assert (install_root / "simulation" / "old-simulation").is_file()
    assert not (skill_destination / "scheduler.py").exists()
    assert json.loads(installed_cache.read_text(encoding="utf-8")) == {
        "generation": "old"
    }
    assert list(install_root.glob(".scheduler.stage.*")) == []


def test_default_link_mode_behavior_is_preserved(tmp_path):
    result, install_root, skill_destination = _run_install(
        tmp_path,
        "--no-systemd",
    )

    assert result.returncode == 0, result.stderr
    assert skill_destination.is_symlink()
    assert skill_destination.resolve() == (REPO_ROOT / "skill").resolve()
    algorithm_destination = install_root / "algorithm"
    assert algorithm_destination.is_symlink()
    assert algorithm_destination.resolve() == (REPO_ROOT / "algorithm").resolve()
    assert not (install_root / "simulation").exists()
