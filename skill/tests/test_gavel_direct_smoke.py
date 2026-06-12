from algorithm.experiments.gavel_direct_native_smoke import _classify_trace_result
from algorithm.experiments.gavel_native_performance_microbaseline import (
    _parse_gavel_log,
    _status,
)
from algorithm.experiments.sota_adapters.scheduleurm_to_gavel_trace_converter import (
    build_native_trace_lines,
)


def test_gavel_trace_classifier_detects_non_progress_timeout():
    result = {
        "timeout": True,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": (
            "scheduler:WARNING [0.0] 1 GPUs of type v100 left unused. "
            "Number of active jobs: 20\n"
            "scheduler:INFO [0.0] Number of completed jobs: 0\n"
        ),
    }

    assert _classify_trace_result(result) == "TIMEOUT_NON_PROGRESS"


def test_gavel_native_trace_seed_uses_tab_separated_schema():
    lines = build_native_trace_lines("q01_gpu_bound_compute")

    assert lines
    first = lines[0].split("\t")
    assert len(first) == 10
    assert first[0] == "ResNet-18 (batch size 32)"
    assert first[3] == "--num_steps"
    assert int(first[5]) > 0
    assert int(first[6]) >= 1


def test_gavel_native_microbaseline_parser_extracts_completion_rows(tmp_path):
    log = tmp_path / "gavel.log"
    log.write_text(
        "\n".join(
            [
                "scheduler:INFO [22.29] Number of completed jobs: 8",
                "Total duration: 22.297 seconds (0.01 hours)",
                "Job completion times:",
                "Job 0: 11.149",
                "Job 1: 22.296",
                "Average job completion time: 20.900 seconds (0.01 hours)",
                "Cluster utilization: 0.500",
            ]
        ),
        encoding="utf-8",
    )

    parsed = _parse_gavel_log(log)

    assert parsed["completed_count"] == 8
    assert parsed["jct_count"] == 2
    assert parsed["average_jct_s"] == 20.9
    assert parsed["cluster_utilization"] == 0.5
    assert _status({"returncode": 0, "timeout": False}, parsed, expected=8) == "PASS"
