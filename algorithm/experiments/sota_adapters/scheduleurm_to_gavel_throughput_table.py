"""Export Scheduleurm measured service cache as a Gavel-style throughput seed."""
from __future__ import annotations

import argparse
from typing import Any

from .common import add_common_args, load_taskset_rows, write_json


def build_table(taskset_name: str) -> dict[str, Any]:
    bundle = load_taskset_rows(taskset_name)
    throughputs: dict[str, dict[str, float]] = {}
    for row in bundle["members"]:
        key = row["member"]["workload_key"]
        throughputs[key] = {}
        for record in row["profiles"]:
            if record.get("capacity_boundary"):
                continue
            profile = int(record["profile"])
            throughputs[key][f"scheduleurm_profile_{profile}"] = float(record["aggregate_rate"])
    return {
        "format": "scheduleurm_gavel_throughput_seed_v1",
        "scope": (
            "Measured service-cache export for direct Gavel adapter development; "
            "schema validation against Gavel's native throughput JSON remains required."
        ),
        "taskset": bundle["taskset"],
        "throughputs": throughputs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_name="gavel_throughputs.json")
    args = parser.parse_args(argv)
    path = write_json(args.output, build_table(args.taskset))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
