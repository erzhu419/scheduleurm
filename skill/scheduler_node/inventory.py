"""Pure node-name canonicalization helpers for scheduler task records."""

from __future__ import annotations

import os
from pathlib import Path


NODE_NAME_ALIASES = {
    # Older queue/history records used this internal name. Keep accepting it,
    # but the live scheduler has exactly one visible/dispatchable node007.
    "node007-direct": "node007",
}

JTL110GPU_RE_SAC_JAX_ENV = "/home/erzhu419/.venvs/resac-jax-gpu1-0438"
JTL110GPU_RE_SAC_JAX_SITE = f"{JTL110GPU_RE_SAC_JAX_ENV}/lib/python3.11/site-packages"
JTL110GPU_RE_SAC_JAX_NVIDIA_LIBS = ":".join([
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cublas/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_nvcc/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_nvrtc/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_runtime/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cudnn/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cufft/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/curand/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cusolver/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cusparse/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/nccl/lib",
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/nvjitlink/lib",
])
JTL110GPU_RE_SAC_JAX_PATH = (
    f"{JTL110GPU_RE_SAC_JAX_SITE}/nvidia/cuda_nvcc/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
NODE007_RESAC_JAX_ENV = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310"
NODE007_SCOMP_ENV = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310"
OFFLINE_SUMO_ENV = "/home/erzhu419/.conda/envs/offline-sumo"
OFFLINE_SUMO_PYTHON = f"{OFFLINE_SUMO_ENV}/bin/python"
BUS_TORCH_CMD_REWRITES = [
    ("/home/erzhu419/anaconda3/bin/python -u sac_ensemble",
     f"{OFFLINE_SUMO_PYTHON} -u sac_ensemble"),
    ("/home/erzhu419/anaconda3/bin/python sac_ensemble",
     f"{OFFLINE_SUMO_PYTHON} sac_ensemble"),
]
BAPR_JAX_LAUNCH_INCLUDE_PATHS = [
    "SC-OLH-KG",
    "jax_experiments/__init__.py",
    "jax_experiments/train.py",
    "jax_experiments/analysis",
    "jax_experiments/algos",
    "jax_experiments/common",
    "jax_experiments/configs",
    "jax_experiments/envs",
    "jax_experiments/networks",
    "mode_profiles.py",
    "normalization.py",
    "bapr_components.py",
]

# These paths are intentionally excluded from launch staging by KG-SYNTH
# submissions. Result sync writes into them continuously, so they must not
# invalidate the source-code staging fingerprint either.
LAUNCH_STAGE_FINGERPRINT_EXCLUDE_PATHS = [
    "SC-OLH-KG/manuscript/review",
    "SC-OLH-KG/results",
    "SC-OLH-KG/profiles",
    "SC-OLH-KG/checkpoints",
    "SC-OLH-KG/data/external",
]

JTL110GPU_CAPACITY_POLICY = {
    "max_vram_per_task": None,
    "max_concurrent_running": None,
    "max_tasks_per_gpu": 4,
    "allow_gpu_over_one_third": True,
    "enable_claims": True,
    "gpu_util_saturation_pct": None,
    "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
}


NODES = {
    "local": {
        "host": None,
        "cpu_cores": 16,
        "reserved_cpu_cores": int(os.environ.get("SCHEDULEURM_LOCAL_RESERVED_CPU_CORES", "0")),
        "ram_mb": 56 * 1024,
        "ram_headroom_mb": 2048,
        "ram_headroom_frac": 0.20,
        "max_vram_per_task": None,
        "max_concurrent_running": 10,
        "max_tasks_per_gpu": 4,
        "gpu_util_saturation_pct": None,
        "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
        "cmd_rewrites": BUS_TORCH_CMD_REWRITES,
    },
    "jtl110gpu": {
        **JTL110GPU_CAPACITY_POLICY,
        "host": "jtl110gpu",
        "cpu_cores": 12,
        "ram_mb": 0,
        "ram_headroom_frac": 0.10,
        "launch_stage_include_paths": BAPR_JAX_LAUNCH_INCLUDE_PATHS,
        "launch_stage_fingerprint_exclude_paths": LAUNCH_STAGE_FINGERPRINT_EXCLUDE_PATHS,
        "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
            ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
             f"{JTL110GPU_RE_SAC_JAX_ENV}/bin/python"),
        ],
        "launch_extra_env": {
            "PATH": JTL110GPU_RE_SAC_JAX_PATH,
            "LD_LIBRARY_PATH": JTL110GPU_RE_SAC_JAX_NVIDIA_LIBS,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
        },
    },
    "jtl110gpu2": {
        **JTL110GPU_CAPACITY_POLICY,
        "host": "jtl110gpu2",
        "cpu_cores": 12,
        "ram_mb": 0,
        "ram_headroom_frac": 0.10,
        "launch_stage_include_paths": BAPR_JAX_LAUNCH_INCLUDE_PATHS,
        "launch_stage_fingerprint_exclude_paths": LAUNCH_STAGE_FINGERPRINT_EXCLUDE_PATHS,
        "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
            ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
             f"{JTL110GPU_RE_SAC_JAX_ENV}/bin/python"),
        ],
        "launch_extra_env": {
            "PATH": JTL110GPU_RE_SAC_JAX_PATH,
            "LD_LIBRARY_PATH": JTL110GPU_RE_SAC_JAX_NVIDIA_LIBS,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
        },
    },
    "jtl311linux": {
        "host": "jtl311linux",
        "cpu_cores": 8,
        "ram_mb": 0,
        "ram_headroom_frac": 0.10,
        "max_vram_per_task": None,
        "max_concurrent_running": None,
        "max_tasks_per_gpu": 3,
        "enable_claims": True,
        "gpu_util_saturation_pct": None,
        "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
        "launch_stage_include_paths": BAPR_JAX_LAUNCH_INCLUDE_PATHS,
        "launch_stage_fingerprint_exclude_paths": LAUNCH_STAGE_FINGERPRINT_EXCLUDE_PATHS,
        "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
            ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
             f"{JTL110GPU_RE_SAC_JAX_ENV}/bin/python"),
        ],
        "launch_extra_env": {
            "PATH": JTL110GPU_RE_SAC_JAX_PATH,
            "LD_LIBRARY_PATH": JTL110GPU_RE_SAC_JAX_NVIDIA_LIBS,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
        },
    },
    "zhengliang-hpc": {
        "host": "202.197.46.16",
        "ssh_user": "zhengliang01",
        "ssh_proxy_jump": "jtl110gpu",
        "ssh_proxy_jumps": ["jtl110gpu", "jtl110gpu2"],
        "cpu_cores": 64,
        "ram_mb": 0,
        "ram_headroom_frac": 0.10,
        "max_vram_per_task": None,
        "max_concurrent_running": 0,
        "monitor_only": True,
        "skip_launch_staging": True,
        "disable_auto_adopt": True,
        "only_when_targeted": True,
        "stage_only_when_targeted": True,
        "relay_node": None,
        "relay_nodes": [],
        "relay_root": "/tmp/scheduleurm-hpc-relay/zhengliang-hpc",
        "remote_workspace_root": "/home/zhengliang01/scheduleurm_work",
        "remote_path_prefixes": [
            str(Path.home() / "mine_code"),
            "/home/erzhu419/mine_code",
        ],
        "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
    },
    "node007": {
        "host": "202.197.46.16",
        "ssh_user": "zhengliang01",
        "ssh_proxy_jump": "jtl110gpu",
        "ssh_proxy_jumps": ["jtl110gpu", "jtl110gpu2"],
        "sudo_ssh_host": "node007",
        "sudo_ssh_run_as": "zhengliang01",
        "cpu_cores": 64,
        "ram_mb": 0,
        "ram_headroom_frac": 0.10,
        "max_vram_per_task": None,
        "max_concurrent_running": 24,
        "max_tasks_per_gpu": 4,
        "small_vram_task_threshold_mb": 1536,
        "small_vram_max_concurrent_running": 24,
        "small_vram_max_tasks_per_gpu": 6,
        "allow_gpu_over_one_third": True,
        "gpu_util_saturation_pct": None,
        "ignore_cpu_for_gpu_tasks": True,
        "cpu_slot_accounting": True,
        "min_free_user_threads_for_gpu_task": 256,
        "max_user_thread_fraction_for_gpu_task": 0.92,
        "probe_attempts": 8,
        "enable_claims": False,
        "skip_launch_staging": False,
        "only_when_targeted": False,
        "stage_only_when_targeted": False,
        "relay_node": None,
        "relay_nodes": [],
        "relay_root": "/tmp/scheduleurm-hpc-relay/node007",
        "remote_workspace_root": "/home/zhengliang01/scheduleurm_work",
        "remote_path_prefixes": [
            str(Path.home() / "mine_code"),
            "/home/erzhu419/mine_code",
        ],
        # The shared node007 workspace is reachable through the head-node
        # proxy. Prefer generic rsync so projects other than the legacy BAPR
        # tree can stage their complete source directory as well.
        "launch_staging_method": "rsync",
        "launch_stage_include_paths": BAPR_JAX_LAUNCH_INCLUDE_PATHS,
        "launch_stage_fingerprint_exclude_paths": LAUNCH_STAGE_FINGERPRINT_EXCLUDE_PATHS,
        "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
            ("/home/erzhu419/miniconda3/envs/csbapr/bin/python",
             "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"),
            ("/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python",
             f"{NODE007_SCOMP_ENV}/bin/python"),
            ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
             f"{NODE007_RESAC_JAX_ENV}/bin/python"),
        ],
        "launch_extra_env": {
            "PATH": (
                f"/cm/local/apps/cuda-driver/libs/535.261.03/bin:"
                f"{NODE007_RESAC_JAX_ENV}/bin:/usr/local/bin:/usr/bin:/bin"
            ),
            "LD_LIBRARY_PATH": "/cm/local/apps/cuda-driver/libs/535.261.03/lib64",
            "BAPR_PYTHON": f"{NODE007_RESAC_JAX_ENV}/bin/python",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
        },
        "resume_scan_python": f"{NODE007_RESAC_JAX_ENV}/bin/python",
        "nvidia_smi_path": "/cm/local/apps/cuda-driver/libs/535.261.03/bin/nvidia-smi",
        "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
    },
    "jtl110cpu": {
        "host": "tf290q6n.zjz-service.cn",
        "ssh_user": "erzhu419",
        "ssh_port": 22945,
        "ssh_identity": str(Path.home() / ".ssh" / "id_ed25519"),
        "os": "windows",
        "cpu_cores": 128,
        "ram_mb": 512 * 1024,
        "ram_headroom_frac": 0.10,
        "max_vram_per_task": 0,
        "windows_python": r"F:\v\Scripts\python.exe",
        "windows_workspace_root": r"F:\erzhu419_smoke",
        "windows_scheduleurm_dir": r"F:\erzhu419_smoke\.scheduleurm",
        "windows_auto_pin": True,
        "windows_skip_ht_pair": True,
        "capabilities": ["cpu", "jax_cpu"],
        "cpu_labor_node": True,
    },
    "jtl110cpu2": {
        "host": "tf290q6n.zjz-service.cn",
        "ssh_user": "erzhu419",
        "ssh_port": 23565,
        "ssh_identity": str(Path.home() / ".ssh" / "id_ed25519"),
        "os": "windows",
        "cpu_cores": 128,
        "ram_mb": 512 * 1024,
        "ram_headroom_frac": 0.10,
        "max_vram_per_task": 0,
        "windows_python": r"F:\v\Scripts\python.exe",
        "windows_workspace_root": r"F:\erzhu419_smoke",
        "windows_scheduleurm_dir": r"F:\erzhu419_smoke\.scheduleurm",
        "windows_auto_pin": True,
        "windows_skip_ht_pair": False,
        "capabilities": ["cpu", "jax_cpu"],
        "cpu_labor_node": True,
    },
}

for _hpc_cpu_idx in range(1, 7):
    _hpc_cpu_node = f"node{_hpc_cpu_idx:03d}"
    NODES[_hpc_cpu_node] = {
        "host": "202.197.46.16",
        "ssh_user": "zhengliang01",
        "ssh_proxy_jump": "jtl110gpu",
        "ssh_proxy_jumps": ["jtl110gpu", "jtl110gpu2"],
        "sudo_ssh_host": _hpc_cpu_node,
        "sudo_ssh_run_as": "zhengliang01",
        "cpu_cores": 192,
        # Keep only a small OS/service margin. Placement uses live CPU
        # headroom, so queued CPU work may fill the remaining 172 cores.
        "reserved_cpu_cores": max(
            0, int(os.environ.get("SCHEDULEURM_HPC_RESERVED_CPU_CORES", "20"))
        ),
        "ram_mb": 0,
        "ram_headroom_frac": 0.10,
        "max_vram_per_task": 0,
        "max_concurrent_running": None,
        "cpu_slot_accounting": True,
        # XLA's CPU backend creates a large per-process thread pool. Six such
        # jobs fit below the HPC account's 4096-thread limit; a seventh does
        # not reliably compile.
        "min_free_user_threads_for_jax_cpu_task": 1024,
        "jax_cpu_launch_user_thread_reserve": 512,
        "live_cpu_backfill": os.environ.get(
            "SCHEDULEURM_HPC_LIVE_CPU_BACKFILL", "1"
        ).lower() not in ("0", "false", "no", "off"),
        # Mature jobs may backfill genuinely idle cores. Fresh launches retain
        # their declared reservation during the startup grace period, and the
        # dispatch wave debits each launch immediately from the probe snapshot.
        "live_cpu_backfill_max_task_cores": max(
            1, int(os.environ.get("SCHEDULEURM_HPC_LIVE_CPU_BACKFILL_MAX_TASK_CORES", "64"))
        ),
        # Bursty CPU SAAS jobs are admitted from live headroom in staggered
        # waves. This is only the final overcommit fuse; live free CPU and the
        # startup reservation normally stop dispatch first.
        "persistent_cpu_live_backfill_max_declared_cores": max(
            0,
            int(
                os.environ.get(
                    "SCHEDULEURM_HPC_PERSISTENT_CPU_MAX_DECLARED_CORES",
                    "276",
                )
            ),
        ),
        "live_cpu_backfill_startup_grace_s": max(
            0, int(os.environ.get("SCHEDULEURM_HPC_LIVE_CPU_BACKFILL_STARTUP_GRACE_S", "90"))
        ),
        "live_cpu_backfill_single_thread_startup_grace_s": max(
            0, int(os.environ.get("SCHEDULEURM_HPC_SINGLE_THREAD_STARTUP_GRACE_S", "10"))
        ),
        "skip_launch_staging": False,
        # These nodes use the same path convention but not one mounted
        # workspace.  Each node must receive its own staging success marker.
        "only_when_targeted": True,
        "stage_only_when_targeted": True,
        "relay_node": None,
        "relay_nodes": [],
        "relay_root": f"/tmp/scheduleurm-hpc-relay/{_hpc_cpu_node}",
        "remote_workspace_root": "/home/zhengliang01/scheduleurm_work",
        # The six compute nodes are reached through the same login account and
        # see the same NFS-backed home workspace. Serialize/alias launch
        # staging so concurrent rsyncs cannot update one source tree at once.
        "shared_workspace_group": "zhengliang-hpc-home",
        "disable_auto_adopt": True,
        "auto_adopt_owners": ["zhengliang01", "root"],
        "remote_path_prefixes": [
            str(Path.home() / "mine_code"),
            "/home/erzhu419/mine_code",
        ],
        # Keep the shared HPC workspace in sync with the actual BAPR/JAX
        # source set.  Without this, a staged CPU task can keep executing an
        # old analysis module after the control workspace has been fixed.
        "launch_stage_include_paths": BAPR_JAX_LAUNCH_INCLUDE_PATHS,
        "launch_stage_fingerprint_exclude_paths": LAUNCH_STAGE_FINGERPRINT_EXCLUDE_PATHS,
        "cmd_rewrites": BUS_TORCH_CMD_REWRITES + [
            ("/home/erzhu419/.conda/envs/resac-jax/bin/python",
             f"{NODE007_RESAC_JAX_ENV}/bin/python"),
        ],
        "launch_extra_env": {
            "PATH": (
                f"{NODE007_RESAC_JAX_ENV}/bin:"
                "/usr/local/bin:/usr/bin:/bin"
            ),
            "LD_LIBRARY_PATH": "/cm/local/apps/cuda-driver/libs/535.261.03/lib64",
            "BAPR_PYTHON": f"{NODE007_RESAC_JAX_ENV}/bin/python",
        },
        "resume_scan_python": f"{NODE007_RESAC_JAX_ENV}/bin/python",
        "capabilities": ["cpu", "jax_cpu"],
    }

TASK_NODE_FIELDS = {
    "node", "last_node", "assigned_node", "required_node", "require_node",
    "preferred_node", "resume_checkpoint_node", "checkpoint_node",
}
TASK_NODE_LIST_FIELDS = {
    "allowed_nodes", "resume_preferred_nodes", "blocked_nodes",
}
TASK_NODE_LOCATION_LIST_FIELDS = (
    "resume_locations", "checkpoint_locations", "last_resume_locations",
    "resume_candidates",
)


def canonical_node_name(name, aliases: dict[str, str] | None = None) -> str:
    text = str(name or "")
    return (aliases or NODE_NAME_ALIASES).get(text, text)


def canonicalize_node_list(values, aliases: dict[str, str] | None = None) -> list:
    """Canonicalize node aliases while preserving the user's node set."""
    out = []
    for value in values or []:
        canon = canonical_node_name(value, aliases)
        if canon and canon not in out:
            out.append(canon)
    return out


def canonicalize_state_node_names(
    state: dict,
    aliases: dict[str, str] | None = None,
) -> bool:
    changed = False
    for task in (state or {}).get("tasks") or []:
        if not isinstance(task, dict):
            continue
        for key in TASK_NODE_FIELDS:
            value = task.get(key)
            canon = canonical_node_name(value, aliases)
            if value and canon != value:
                task[key] = canon
                changed = True
        for key in TASK_NODE_LIST_FIELDS:
            values = task.get(key)
            if not isinstance(values, list):
                continue
            canonical_values = canonicalize_node_list(values, aliases)
            if canonical_values != values:
                task[key] = canonical_values
                changed = True
        if (
            task.get("allowed_nodes_user_explicit")
            and isinstance(task.get("allowed_nodes_submitted"), list)
        ):
            locked_allowed = canonicalize_node_list(task.get("allowed_nodes_submitted"), aliases)
            current_allowed = (
                task.get("allowed_nodes")
                if isinstance(task.get("allowed_nodes"), list)
                else []
            )
            if current_allowed != locked_allowed:
                if locked_allowed:
                    task["allowed_nodes"] = locked_allowed
                else:
                    task.pop("allowed_nodes", None)
                task.pop("allowed_nodes_expanded_reason", None)
                changed = True
            if task.get("allowed_nodes_submitted") != locked_allowed:
                task["allowed_nodes_submitted"] = locked_allowed
                changed = True
        for list_key in TASK_NODE_LOCATION_LIST_FIELDS:
            locs = task.get(list_key)
            if not isinstance(locs, list):
                continue
            for loc in locs:
                if not isinstance(loc, dict):
                    continue
                value = loc.get("node")
                canon = canonical_node_name(value, aliases)
                if value and canon != value:
                    loc["node"] = canon
                    changed = True
    return changed
