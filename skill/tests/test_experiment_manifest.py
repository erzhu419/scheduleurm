import json
import tempfile
from pathlib import Path


def test_experiment_manifest_init_run(check, sch):
    from algorithm.experiments.trace_export import init_run

    with tempfile.TemporaryDirectory() as d:
        path = init_run("unit_manifest", runs_dir=Path(d))
        data = json.loads(path.read_text())
        check("experiment manifest written", path.name == "manifest.json" and path.exists())
        check("experiment manifest schema", data.get("schema_version") == "scheduleurm-exp-v1")
        check("experiment manifest theorem target recorded",
              "approx_oracle" in data.get("theorem_target", ""))
        check("experiment manifest records algorithm modules",
              data.get("algorithm_modules", {}).get("candidate_set", "").startswith("algorithm.candidates"),
              diag=str(data.get("algorithm_modules")))
        check("experiment run directories created",
              all((path.parent / x).is_dir() for x in ("raw", "normalized", "calibration", "reports")))
