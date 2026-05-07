"""env_deploy: pick + apply environment delivery strategy per task.

Strategies, chosen at submit time via `--env-spec`:

  * `none` (default, back-compat): cmd already references an absolute conda python path,
    env must exist on target. Failure → ENV_MISSING escalation, watcher routes to heal.
  * `docker:IMAGE[:TAG]`: launch wraps cmd in
        `docker rm -f sched-<id> 2>/dev/null || true; \\
         docker run --rm --name sched-<id> \\
            --gpus device=<N> -e CUDA_VISIBLE_DEVICES=0 \\
            --memory <ram_mb>m --cpus <cpu_cores> \\
            -v $cwd:$cwd -w $cwd $image bash -c '<inner>'`
    GPU is HARD-pinned to one device via `--gpus device=<N>` (older versions used the
    "expose every GPU" flag which leaked GPUs across pinned tasks; Codex flagged it).
    First time on a target, scheduler does `docker save | ssh node docker load` to push
    image (~minutes for ML images, one-time). On subsequent dispatches, image digest is
    compared against local; drift triggers re-push (Codex P1b).
  * `conda:/abs/path/to/env`: pre-dispatch, scheduler `rsync -az --partial --delete`'s
    local conda env to target node at the SAME absolute path. Cmd's absolute python path
    then resolves on the synced env. Idempotent: re-sync of unchanged env is fast (~s).
  * `auto`: probe target for docker access at dispatch time:
      - target has `docker info` working AND `--image` is set → docker path
      - else → fall back to `none` (caller's cmd must use absolute python path)

Auto path needs `--image` to know what to wrap with — without it, falls back to none.

The wrapper is applied in scheduler.launch() after `_inject_python_u` and before
`--resume_from` injection (resume flag goes inside the docker command, not outside).
Push for missing-or-drifted images is triggered both at dispatch preload (outside the
state_lock — see scheduler._preload_docker_images_outside_lock) and at launch as a
safety net if preload didn't run or failed.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Optional


def parse_env_spec(spec: Optional[str]) -> tuple[str, Optional[str]]:
    """Parse '--env-spec' value. Returns (kind, payload_or_none).

    Examples:
        None / "" / "none"            → ("none", None)
        "docker:myproject:latest"     → ("docker", "myproject:latest")
        "docker"                      → ("docker", None)  # caller must provide --image separately
        "conda:/home/u/.conda/envs/x" → ("conda", "/home/u/.conda/envs/x")
        "auto"                        → ("auto", None)    # caller must provide --image to enable docker
    """
    if not spec or spec == "none":
        return ("none", None)
    if spec == "auto":
        return ("auto", None)
    if spec.startswith("docker:"):
        return ("docker", spec[len("docker:"):])
    if spec == "docker":
        return ("docker", None)
    if spec.startswith("conda:"):
        env_path = spec[len("conda:"):]
        if not env_path:
            raise ValueError("--env-spec conda: requires an absolute env path")
        if not os.path.isabs(env_path):
            raise ValueError(f"conda env path must be absolute: {env_path!r}")
        return ("conda", env_path)
    raise ValueError(
        f"unrecognized --env-spec {spec!r}. Use 'none', 'docker[:IMAGE]', "
        f"'conda:/abs/path/to/env', or 'auto'."
    )


# ---------- conda env auto-sync ----------

def has_conda_env(run_on, node: str, env_path: str, timeout: int = 8) -> bool:
    """True iff `<env_path>/bin/python --version` works on the target node."""
    try:
        rc, _, _ = run_on(
            node,
            f"{shlex.quote(env_path + '/bin/python')} --version >/dev/null 2>&1 && echo OK",
            timeout=timeout, check=False,
        )
        return rc == 0
    except Exception:
        return False


def push_conda_env(node_host: Optional[str], local_path: str, remote_path: str,
                   timeout_s: int = 3600) -> tuple[bool, str]:
    """rsync local conda env to remote at the same absolute path. Idempotent: rsync's
    delta algorithm makes re-sync of unchanged env trivial. First sync is the slow one
    (multi-GB env can take minutes).

    Returns (ok, msg). For local node (node_host=None), this is a no-op since the env
    is presumably ALREADY at local_path (that's where we're rsyncing FROM).

    Path matching policy: remote_path == local_path. Caller must align paths upstream
    (e.g., always use ~/<conda>/envs/X identically across nodes; users with different
    conda installation paths must symlink one of them).
    """
    if node_host is None:
        return (True, "local: conda env is the source of truth")
    # -a: archive (perms, symlinks, times). -z: compress over network. -q: quiet (we
    # capture stderr separately). --delete: drop files on remote that local doesn't
    # have (so removing a pkg locally also drops it remotely). --partial: keep partial
    # on interrupt for resume.
    import subprocess
    src = local_path.rstrip("/") + "/"  # trailing slash → contents of dir, not the dir itself
    dst_parent = os.path.dirname(remote_path)
    cmd = [
        "rsync", "-az", "--partial", "--delete",
        "--rsync-path", f"mkdir -p {shlex.quote(dst_parent)} && rsync",
        src, f"{node_host}:{remote_path.rstrip('/')}/",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode == 0:
            return (True, f"rsync ok → {node_host}:{remote_path}")
        return (False, f"rsync failed (rc={r.returncode}): {(r.stderr or r.stdout)[:300]}")
    except subprocess.TimeoutExpired:
        return (False, f"rsync timeout after {timeout_s}s pushing {local_path} to {node_host}")
    except Exception as e:
        return (False, f"push_conda_env error: {e}")

def has_docker(run_on, node: str, timeout: int = 8) -> bool:
    """True iff `docker info` exits 0 on the target. Probes daemon access, not just CLI."""
    try:
        rc, _, _ = run_on(node, "docker info >/dev/null 2>&1 && echo OK",
                          timeout=timeout, check=False)
        return rc == 0
    except Exception:
        return False


def get_image_digest(run_on, node: str, image: str, timeout: int = 8) -> Optional[str]:
    """Return `docker inspect IMAGE --format {{.Id}}` (sha256:...) or None on failure."""
    try:
        rc, out, _ = run_on(
            node,
            f"docker inspect --format '{{{{.Id}}}}' {shlex.quote(image)} 2>/dev/null",
            timeout=timeout, check=False,
        )
        s = (out or "").strip() if rc == 0 else ""
        return s if s.startswith("sha256:") else None
    except Exception:
        return None


def has_image(run_on, node: str, image: str, timeout: int = 8,
              local_digest: Optional[str] = None) -> bool:
    """True iff target has the image AND (when local_digest provided) its digest matches.

    Without the digest check, a `tag` like `myproj:latest` could exist on remote with an
    OLD content while local has been rebuilt — push would skip and a stale image runs.
    Codex P1. Pass local_digest=get_image_digest(run_on, 'local', image) to enable the
    drift detection.
    """
    remote_digest = get_image_digest(run_on, node, image, timeout=timeout)
    if not remote_digest:
        return False  # not present at all
    if local_digest is None:
        return True  # legacy fast path: just tag-presence
    return remote_digest == local_digest


def push_image(node_host: Optional[str], image: str, timeout_s: int = 1800) -> tuple[bool, str]:
    """`docker save IMAGE | ssh HOST docker load` (or `docker load` if local).

    Returns (ok, msg). ~1-30 min depending on image size — caller should treat as a
    blocking step (don't run during dispatch loop; do it once at submit or on first
    dispatch to that node).

    node_host=None means local; the function is then a no-op (image must already be
    on local for the scheduler to invoke it).
    """
    if node_host is None:
        # Local: assume image is present (it lives in local docker daemon)
        return (True, "local: docker image expected to be present in local daemon")
    cmd = (
        f"docker save {shlex.quote(image)} | "
        f"ssh {shlex.quote(node_host)} 'docker load'"
    )
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout_s)
        ok = r.returncode == 0
        msg = (r.stdout + r.stderr).strip()[-400:]
        return (ok, msg if msg else f"docker load on {node_host} returned {r.returncode}")
    except subprocess.TimeoutExpired:
        return (False, f"timeout after {timeout_s}s pushing {image} to {node_host}")
    except Exception as e:
        return (False, f"push_image failed: {e}")


class _ShellLiteral(str):
    """Marker subclass: this arg is already shell-safe; the docker_run join must NOT
    shlex.quote it. Used for GPU specs whose value is a runtime-expanded env var
    (e.g. `device=$CUDA_VISIBLE_DEVICES` for slurm pin-passthrough). shlex.quote
    would single-quote the `$` and break the bash expansion."""
    pass


def wrap_cmd_docker(inner: str, image: str, cwd: str, gpu_idx: Optional[int],
                    extra_env: Optional[dict] = None,
                    container_name: Optional[str] = None,
                    memory_mb: Optional[int] = None,
                    cpus: Optional[float] = None,
                    gpu_runtime_env: Optional[str] = None) -> str:
    """Wrap an already-built shell cmd in `docker run`.

    GPU pinning has two modes:

    1. **Static pin (LocalBackend)**: gpu_idx=N (integer). Emits `--gpus device=N`
       to HARD-PIN the container to that one GPU. Container sees only that device,
       enumerated as 0 inside. Also sets `CUDA_VISIBLE_DEVICES=0` so frameworks
       that read the env var still pick the pinned device. Without this combo,
       `--gpus all` would let a task assigned to GPU1 see/use GPU0 too — silent
       placement violation.

    2. **Runtime pin (SlurmBackend, Phase 2.6)**: gpu_runtime_env="CUDA_VISIBLE_DEVICES".
       Emits `--gpus "device=$CUDA_VISIBLE_DEVICES"` so the docker pin matches
       whatever GPU slurm's gres allocator chose at job runtime. Without this,
       gpu_idx would either be None (Phase 2.3 behavior — task gets NO GPU
       inside the container even though slurm allocated one) or a stale
       scheduleurm-picked value that doesn't match slurm's actual allocation.

    CPU-only: gpu_idx=None AND gpu_runtime_env=None. No `--gpus` flag; explicitly
    null CUDA_VISIBLE_DEVICES inside the container.

    Container name: `--name <container_name>` so cancel paths can `docker kill
    <name>` instead of relying on `docker run` PID descent (containerd-shim makes
    the actual container procs NOT children of the docker client). Caller passes
    task id as name.

    - `-v $cwd:$cwd -w $cwd`: absolute host paths resolve identically inside.
    - `--rm`: container is ephemeral; state lives in mounted volumes.
    - extra_env entries become `-e K=V`.
    """
    args: list = ["docker", "run", "--rm"]
    if container_name:
        args += ["--name", container_name]
    # Cgroup-level resource limits so the container honors scheduler budgets (Codex P1).
    # Without these, scheduler RAM/CPU caps are advisory at scheduler level only — container
    # could consume all host RAM/CPU and OOM neighbor tasks.
    if memory_mb and memory_mb > 0:
        args += ["--memory", f"{memory_mb}m"]
    if cpus and cpus > 0:
        args += ["--cpus", str(cpus)]
    if gpu_runtime_env:
        # Slurm path: docker pins to whatever GPU slurm has set in this env var at runtime.
        # The arg must be unquoted at shell-join time so `$CUDA_VISIBLE_DEVICES` expands.
        # We wrap in double quotes so a value like "0,1" stays as one arg even if any of
        # those bytes were spaces (paranoia — slurm uses comma, not space).
        args += ["--gpus", _ShellLiteral(f'"device=${gpu_runtime_env}"')]
        args += ["-e", "CUDA_VISIBLE_DEVICES=0"]
    elif gpu_idx is not None:
        # Static pin (LocalBackend): scheduleurm-decided GPU.
        args += ["--gpus", f"device={gpu_idx}"]
        args += ["-e", "CUDA_VISIBLE_DEVICES=0"]
    else:
        # CPU-only: don't request GPU at all; explicitly null CUDA_VISIBLE_DEVICES so any
        # CUDA-init in the cmd doesn't pick up host-leaked GPU exposure.
        args += ["-e", "CUDA_VISIBLE_DEVICES="]
    args += ["-v", f"{cwd}:{cwd}", "-w", cwd]
    if extra_env:
        for k, v in extra_env.items():
            # Skip CUDA_VISIBLE_DEVICES from extra_env: we set it explicitly above based on
            # gpu_idx / gpu_runtime_env. User-supplied value would override our pinning.
            if k == "CUDA_VISIBLE_DEVICES":
                continue
            args += ["-e", f"{k}={v}"]
    args += [image, "bash", "-c", inner]
    # Quote everything EXCEPT _ShellLiteral fragments. The literal class marks pre-quoted
    # args (currently only the runtime-env GPU spec) where we need bash to expand $VAR.
    docker_run = " ".join(
        a if isinstance(a, _ShellLiteral) else shlex.quote(a)
        for a in args
    )
    # Pre-cleanup: remove any stale container with same name (left over from prior crashed
    # launch where --rm didn't fire because container exited dirtily). Without this, the
    # second launch fails with "container name already in use". Codex P1.
    if container_name:
        cleanup = f"docker rm -f {shlex.quote(container_name)} 2>/dev/null || true"
        return f"{cleanup}; {docker_run}"
    return docker_run
