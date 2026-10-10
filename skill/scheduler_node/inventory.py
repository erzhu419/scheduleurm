"""Pure node-name canonicalization helpers for scheduler task records."""

from __future__ import annotations

import json
import os
from pathlib import Path


NODE_NAME_ALIASES = {
    # Older queue/history records used this internal name. Keep accepting it,
    # but the live scheduler has exactly one visible/dispatchable node007.
    "node007-direct": "node007",
}

WINDOWS_TASK_DISPATCH_BLOCK_REASON = (
    "windows-node-disabled: task dispatch to Windows nodes is prohibited"
)

GPU_A_RE_SAC_JAX_ENV = "/home/researcher/.venvs/resac-jax-gpu1-0438"
GPU_A_RE_SAC_JAX_SITE = f"{GPU_A_RE_SAC_JAX_ENV}/lib/python3.11/site-packages"
GPU_A_RE_SAC_JAX_NVIDIA_LIBS = ":".join([
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cublas/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cuda_nvcc/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cuda_nvrtc/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cuda_runtime/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cudnn/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cufft/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/curand/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cusolver/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cusparse/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/nccl/lib",
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/nvjitlink/lib",
])
GPU_A_RE_SAC_JAX_PATH = (
    f"{GPU_A_RE_SAC_JAX_SITE}/nvidia/cuda_nvcc/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
NODE007_RESAC_JAX_ENV = "/home/hpcuser/scheduleurm_work/conda_envs/csbapr-gpu-py310"
NODE007_RESAC_JAX_0438_ENV = "/home/hpcuser/scheduleurm_work/conda_envs/bapr-v39-gpu-0438"
NODE007_SCOMP_ENV = "/home/hpcuser/scheduleurm_work/conda_envs/scomp-py310"
OFFLINE_SUMO_ENV = "/home/researcher/.conda/envs/offline-sumo"
OFFLINE_SUMO_PYTHON = f"{OFFLINE_SUMO_ENV}/bin/python"
BUS_TORCH_CMD_REWRITES = [
    ("/home/researcher/anaconda3/bin/python -u -m bus_experiments",
     f"{OFFLINE_SUMO_PYTHON} -u -m bus_experiments"),
    ("/home/researcher/anaconda3/bin/python -m bus_experiments",
     f"{OFFLINE_SUMO_PYTHON} -m bus_experiments"),
    ("/home/researcher/anaconda3/bin/python -u sac_ensemble",
     f"{OFFLINE_SUMO_PYTHON} -u sac_ensemble"),
    ("/home/researcher/anaconda3/bin/python sac_ensemble",
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

GPU_A_CAPACITY_POLICY = {
    "max_vram_per_task": None,
    "max_concurrent_running": None,
    "max_tasks_per_gpu": 4,
    "allow_gpu_over_one_third": True,
    "enable_claims": True,
    "gpu_util_saturation_pct": None,
    "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
}


def load_nodes(*, environ=None, home=None) -> dict:
    """Read operator-owned configuration without shipping a private cluster."""
    env = os.environ if environ is None else environ
    home_path = Path.home() if home is None else Path(home)
    explicit = env.get("SCHEDULEURM_NODES_FILE")
    path = Path(explicit).expanduser() if explicit else home_path / ".claude" / "scheduler" / "nodes.json"
    if explicit or path.exists():
        nodes = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(nodes, dict) or not nodes or any(
            not name or not isinstance(config, dict) for name, config in nodes.items()
        ):
            raise ValueError(f"{path}: expected a non-empty object mapping node names to configurations")
        return nodes
    return {
        "local": {
            "host": None,
            "cpu_cores": 1,
            "reserved_cpu_cores": 0,
            "ram_mb": 0,
            "ram_headroom_frac": 0.1,
            "max_concurrent_running": 1,
            "max_tasks_per_gpu": 1,
            "capabilities": ["cpu", "cuda", "torch_cuda", "jax_cuda"],
        }
    }


NODES = load_nodes()

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
