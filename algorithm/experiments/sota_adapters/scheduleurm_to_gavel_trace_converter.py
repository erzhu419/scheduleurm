"""Export Scheduleurm tasksets as Gavel-style trace seeds.

The JSON export is the canonical Scheduleurm-to-Gavel audit seed.  The optional
native ``.trace`` export uses Gavel's tab-separated trace schema so that the
external runner can be smoke-tested.  That native file is a compatibility seed;
it is not, by itself, a validated same-workload full-stack comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import add_common_args, load_taskset_rows, write_json


GAVEL_TEMPLATE_BY_RESOURCE_KIND = {
    "gpu_heavy": {
        "job_type": "ResNet-18 (batch size 32)",
        "command": "python3 main.py --data_dir=%s/cifar10 --batch_size 32",
        "working_directory": "image_classification/cifar10",
        "num_steps_arg": "--num_steps",
        "needs_data_dir": 1,
    },
    "gpu_cnn": {
        "job_type": "ResNet-50 (batch size 16)",
        "command": "python3 main.py -j 8 -a resnet50 -b 16 %s/imagenet/",
        "working_directory": "image_classification/imagenet",
        "num_steps_arg": "--num_minibatches",
        "needs_data_dir": 1,
    },
    "hybrid_rl": {
        "job_type": "A3C",
        "command": "python3 main.py --env PongDeterministic-v4 --workers 4 --amsgrad True",
        "working_directory": "rl",
        "num_steps_arg": "--max-steps",
        "needs_data_dir": 0,
    },
    "cpu_heavy": {
        "job_type": "Recommendation (batch size 8192)",
        "command": "python3 train.py --data_dir %s/ml-20m/pro_sg/ --batch_size 8192",
        "working_directory": "recommendation",
        "num_steps_arg": "-n",
        "needs_data_dir": 1,
    },
    "light_control": {
        "job_type": "Recommendation (batch size 512)",
        "command": "python3 train.py --data_dir %s/ml-20m/pro_sg/ --batch_size 512",
        "working_directory": "recommendation",
        "num_steps_arg": "-n",
        "needs_data_dir": 1,
    },
}


def build_trace(taskset_name: str) -> dict[str, Any]:
    bundle = load_taskset_rows(taskset_name)
    jobs = []
    for row in bundle["members"]:
        member = row["member"]
        for idx in range(int(member["task_count"])):
            jobs.append({
                "job_id": f"{member['workload_key']}-{idx:05d}",
                "arrival_time": 0.0,
                "workload_key": member["workload_key"],
                "resource_kind": member["resource_kind"],
                "resource_count": int(member["resource_count"]),
                "total_units": float(member["total_units"]),
            })
    return {
        "format": "scheduleurm_gavel_static_trace_seed_v1",
        "scope": (
            "Static Scheduleurm taskset export for Gavel adapter development; "
            "native Gavel trace-schema validation remains required before direct comparison."
        ),
        "taskset": bundle["taskset"],
        "jobs": jobs,
    }


def build_native_trace_lines(taskset_name: str) -> list[str]:
    bundle = load_taskset_rows(taskset_name)
    lines: list[str] = []
    for row in bundle["members"]:
        member = row["member"]
        template = _template_for(member["resource_kind"])
        scale_factor = max(1, int(member.get("resource_count") or 1))
        total_steps = max(1, int(round(float(member["total_units"]))))
        for _ in range(int(member["task_count"])):
            fields = [
                template["job_type"],
                template["command"],
                template["working_directory"],
                template["num_steps_arg"],
                str(template["needs_data_dir"]),
                str(total_steps),
                str(scale_factor),
                "1",
                "-1.000000",
                "0.000000",
            ]
            lines.append("\t".join(fields))
    return lines


def _template_for(resource_kind: str) -> dict[str, Any]:
    template = GAVEL_TEMPLATE_BY_RESOURCE_KIND.get(resource_kind)
    if template is None:
        return GAVEL_TEMPLATE_BY_RESOURCE_KIND["gpu_heavy"]
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_name="gavel_static_trace.json")
    parser.add_argument(
        "--native-trace-output",
        default=None,
        help="Optional Gavel tab-separated .trace compatibility seed output.",
    )
    args = parser.parse_args(argv)
    path = write_json(args.output, build_trace(args.taskset))
    if args.native_trace_output:
        native_path = Path(args.native_trace_output).expanduser()
        native_path.parent.mkdir(parents=True, exist_ok=True)
        native_path.write_text(
            "\n".join(build_native_trace_lines(args.taskset)) + "\n",
            encoding="utf-8",
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
