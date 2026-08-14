from __future__ import annotations

from types import SimpleNamespace

from skill import scheduler_windows_host_extras as extras


def test_probe_windows_host_extras_parses_ram_cpu_and_compute_gpu():
    calls = []

    def run_subprocess(args, **kwargs):
        calls.append((args, kwargs))
        script = args[-1]
        if "ComputerInfo" in script:
            return SimpleNamespace(returncode=0, stdout="noise\n20000|64000|25\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="88\n", stderr="")

    got = extras.probe_windows_host_extras(
        deps=extras.WindowsHostExtrasDeps(
            path_exists=lambda path: path == extras.WINDOWS_POWERSHELL_PATHS[0],
            run_subprocess=run_subprocess,
        )
    )

    assert got == {
        "host_free_ram_mb": 20000,
        "host_total_ram_mb": 64000,
        "host_cpu_load_pct": 25,
        "gpu_compute_util_pct": 88,
    }
    assert len(calls) == 2


def test_probe_windows_host_extras_returns_empty_when_powershell_missing():
    got = extras.probe_windows_host_extras(
        deps=extras.WindowsHostExtrasDeps(
            path_exists=lambda path: False,
            run_subprocess=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        )
    )

    assert got == {}
