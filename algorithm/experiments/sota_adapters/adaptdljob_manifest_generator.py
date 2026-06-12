"""Generate AdaptDLJob manifest seeds for Scheduleurm tasksets."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import ARTIFACT_ROOT, load_taskset_rows, sanitize_name


def build_manifest(taskset_name: str) -> str:
    bundle = load_taskset_rows(taskset_name)
    docs = []
    for row in bundle["members"]:
        member = row["member"]
        name = sanitize_name(member["workload_key"])
        docs.append(f"""apiVersion: adaptdl.petuum.com/v1
kind: AdaptDLJob
metadata:
  name: {name}
  labels:
    scheduleurm-taskset: {sanitize_name(taskset_name)}
spec:
  placement: any
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: scheduleurm/adaptdl-placeholder:latest
        command: ["python", "-m", "scheduleurm_workload_adapter"]
        args: ["--workload-key", "{member['workload_key']}", "--total-units", "{member['total_units']}"]
        resources:
          limits:
            cpu: "{max(1, int(member['resource_count']))}"
""")
    return "---\n".join(docs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", default="hybrid_research_portfolio")
    parser.add_argument("--output", default=str(ARTIFACT_ROOT / "adaptdljobs.yaml"))
    args = parser.parse_args(argv)
    path = Path(args.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_manifest(args.taskset), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
