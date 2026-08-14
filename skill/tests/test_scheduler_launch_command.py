from __future__ import annotations

from skill.scheduler_launch_command import inject_python_u
from skill.scheduler_windows_launcher import WINDOWS_DETACHER, WINDOWS_LAUNCHER


def test_inject_python_u_handles_common_shell_forms():
    cases = [
        ("python train.py --seed 42", "python -u train.py --seed 42"),
        ("python3 train.py", "python3 -u train.py"),
        ("/conda/envs/x/bin/python train.py", "/conda/envs/x/bin/python -u train.py"),
        ("conda run -n env python -m mod", "conda run -n env python -u -m mod"),
        ("PYTHONPATH=. python script.py && echo done", "PYTHONPATH=. python -u script.py && echo done"),
        ("python -u train.py", "python -u train.py"),
        ("python -uX foo", "python -uX foo"),
        ("python --user --version", "python -u --user --version"),
        ("bash run.sh", "bash run.sh"),
        ("", ""),
    ]

    for cmd, expected in cases:
        assert inject_python_u(cmd) == expected


def test_windows_launcher_text_exposes_payload_and_pid_contracts():
    assert "def main()" in WINDOWS_LAUNCHER
    assert "scheduleurm windows wrapper start" in WINDOWS_LAUNCHER
    assert "scheduleurm resource-progress" in WINDOWS_LAUNCHER
    assert "--payload-file" in WINDOWS_LAUNCHER
    assert "print(\"PID=\" + str(p.pid))" in WINDOWS_DETACHER
