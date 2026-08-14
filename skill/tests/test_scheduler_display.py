from skill import scheduler_display as display


def test_format_mem_gb_uses_legacy_precision_and_approx_marker():
    assert display.format_mem_gb(512) == "0.50GB"
    assert display.format_mem_gb(9 * 1024) == "9.00GB"
    assert display.format_mem_gb(10 * 1024) == "10.0GB"
    assert display.format_mem_gb("bad") == "0.00GB"
    assert display.format_mem_gb(1536, approx=True) == "~1.50GB"


def test_format_node_ram_summary_marks_effective_host_free_ram():
    assert display.format_node_ram_summary({}) == ""
    assert display.format_node_ram_summary({"free_ram_mb": 2048}) == "ram_free=2.00GB"
    assert display.format_node_ram_summary({
        "free_ram_mb": 2048,
        "host_free_ram_mb": 4096,
    }) == "ram_free=2.00GB(eff)"


def test_format_task_location_for_cpu_gpu_and_legacy_records():
    assert display.format_task_location({}) == "-"
    assert display.format_task_location({"node": "node001"}) == "node001:CPU"
    assert display.format_task_location({"node": "node007", "gpu_idx": 2}) == "node007:GPU2"
    assert display.format_task_location(
        {"node": "zhengliang-hpc", "slurm_job_id": "123", "slurm_state": "PENDING"},
        legacy_bucket="cpu",
    ) == "zhengliang-hpc:LEGACY-CPU#123:PENDING"
    assert display.format_task_location(
        {"node": "zhengliang-hpc", "slurm_job_id": "124"},
        legacy_bucket="gpu",
    ) == "zhengliang-hpc:LEGACY-GPU#124"


def test_format_task_vram_usage_prefers_current_then_peak_then_estimate():
    assert display.format_task_vram_usage({
        "status": "running",
        "current_vram_mb": 2048,
        "peak_vram_mb": 4096,
        "est_vram_mb": 8192,
    }) == "cur=2.00GB"
    assert display.format_task_vram_usage({
        "status": "running",
        "current_vram_mb": 0,
        "vram_estimation_source": "aggregate_observed_zero",
    }) == "cur=0.00GB"
    assert display.format_task_vram_usage({"peak_vram_mb": 4096, "est_vram_mb": 8192}) == "peak=4.00GB"
    assert display.format_task_vram_usage({"est_vram_mb": 8192}) == "~8.00GB"


def test_format_task_ram_usage_prefers_current_then_peak_then_declared():
    assert display.format_task_ram_usage({
        "status": "running",
        "current_ram_mb": 2048,
        "peak_ram_mb": 4096,
        "ram_mb": 8192,
    }) == "Rcur=2.00GB"
    assert display.format_task_ram_usage({"peak_ram_mb": 4096, "ram_mb": 8192}) == "Rpeak=4.00GB"
    assert display.format_task_ram_usage({"ram_mb": 8192}) == "~8.00GB"
