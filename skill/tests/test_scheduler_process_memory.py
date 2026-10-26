import io

from skill.scheduler_probe.memory import PROCESS_MEMORY_SCRIPT, TASK_MEMORY_SCRIPT


def _reader(monkeypatch, files):
    def read(path, *args, **kwargs):
        value = files.get(path)
        if value is None:
            raise PermissionError(path)
        return io.StringIO(value)

    monkeypatch.setattr("builtins.open", read)
    namespace = {}
    exec(PROCESS_MEMORY_SCRIPT, namespace)
    return namespace["process_pss_kb"]


def test_pss_uses_rollup_instead_of_rss_or_private_memory(monkeypatch):
    read = _reader(monkeypatch, {
        "/proc/100/smaps_rollup": "Rss: 6400000 kB\nPss: 1300000 kB\nPrivate_Dirty: 800000 kB\n",
    })
    assert read(100) == 1300000


def test_old_hpc_kernel_smaps_sums_only_pss_lines(monkeypatch):
    read = _reader(monkeypatch, {
        "/proc/100/smaps": "Rss: 200000 kB\nPss: 1024 kB\nSwapPss: 900 kB\nPss: 512 kB\n",
        "/proc/101/smaps_rollup": "Pss: 0 kB\n",
    })
    assert read(100) == 1536
    assert read(101) == 0
    assert read(102) is None


def test_backend_memory_script_reads_only_task_roots_and_live_descendants(monkeypatch, capsys):
    import sys

    _reader(monkeypatch, {
        f"/proc/{pid}/smaps_rollup": "Pss: 1024 kB\n" for pid in range(100, 105)
    })
    monkeypatch.setattr(sys, "argv", ["probe", "100"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(
        "100 1 2048 1 S\n101 100 2048 2 R\n102 101 2048 0 S\n"
        "103 1 2048 1 S\n104 100 2048 0 Z\n"
    ))
    exec(TASK_MEMORY_SCRIPT, {})
    assert capsys.readouterr().out.splitlines() == ["100 1024", "101 1024", "102 1024"]
