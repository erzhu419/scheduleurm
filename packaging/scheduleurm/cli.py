"""Launch the same runtime used by the repository installation."""

from importlib.metadata import version
from pathlib import Path
import runpy
import sys


def main():
    if sys.argv[1:] == ["--version"]:
        print(version("scheduleurm"))
        return
    runpy.run_path(str(Path(__file__).parent / "skill" / "scheduler.py"), run_name="__main__")


def mcp_main():
    runpy.run_path(
        str(Path(__file__).parent / "skill" / "integrations" / "scheduler_mcp.py"),
        run_name="__main__",
    )
