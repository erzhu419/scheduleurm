"""Export Scheduleurm tasksets as Salus benchmark-driver seed metadata."""
from __future__ import annotations

import argparse
from typing import Any

from .common import add_common_args, load_taskset_rows, write_json


def build_manifest(taskset_name: str) -> dict[str, Any]:
    bundle = load_taskset_rows(taskset_name)
    sessions = []
    for row in bundle["members"]:
        member = row["member"]
        sessions.append({
            "name": member["workload_key"],
            "resource_kind": member["resource_kind"],
            "count": int(member["task_count"]),
            "total_units": float(member["total_units"]),
            "resource_count": int(member["resource_count"]),
            "profiles": [
                {
                    "profile": int(record["profile"]),
                    "aggregate_rate": float(record["aggregate_rate"]),
                    "capacity_boundary": bool(record.get("capacity_boundary")),
                }
                for record in row["profiles"]
            ],
        })
    return {
        "format": "scheduleurm_salus_benchmark_seed_v1",
        "scope": (
            "Benchmark-driver seed metadata for Salus adapter development; "
            "Salus server/runtime integration remains required before direct comparison."
        ),
        "taskset": bundle["taskset"],
        "sessions": sessions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_name="salus_benchmark_seed.json")
    args = parser.parse_args(argv)
    path = write_json(args.output, build_manifest(args.taskset))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
