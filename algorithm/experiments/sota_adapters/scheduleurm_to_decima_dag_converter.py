"""Export Scheduleurm tasksets as Decima Spark-DAG seed metadata."""
from __future__ import annotations

import argparse
from typing import Any

from .common import add_common_args, load_taskset_rows, write_json


def build_dags(taskset_name: str) -> dict[str, Any]:
    bundle = load_taskset_rows(taskset_name)
    dags = []
    for row in bundle["members"]:
        member = row["member"]
        for idx in range(int(member["task_count"])):
            dags.append({
                "dag_id": f"{member['workload_key']}-{idx:05d}",
                "arrival_time": 0.0,
                "nodes": [
                    {
                        "node_id": 0,
                        "duration_units": float(member["total_units"]),
                        "resource_kind": member["resource_kind"],
                    }
                ],
                "edges": [],
            })
    return {
        "format": "scheduleurm_decima_dag_seed_v1",
        "scope": (
            "Single-stage DAG seed for Decima adapter development; Decima remains "
            "a Spark-DAG simulator, not a direct GPU co-location baseline."
        ),
        "taskset": bundle["taskset"],
        "dags": dags,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_name="decima_dag_seed.json")
    args = parser.parse_args(argv)
    path = write_json(args.output, build_dags(args.taskset))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
