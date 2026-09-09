"""Natural-completion CPU ETA under controlled half/full resident load."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .native_cpu_allocation_worker_state_campaign import (
    _deploy_progress_nodes,
)
from .native_cpu_colocation_state_targeted_campaign import (
    _deploy_nodes,
    _survey_nodes,
)
from .phase_aware_cpu_completion_matrix import (
    CpuCompletionCell,
    _run_cell,
)
from .remote_workload_selected_profile_probe import (
    _remote_popen,
    _run_remote_capture,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
REMOTE_BENCHMARK = (
    "/tmp/scheduleurm_progress_wrapper_pkg/algorithm/experiments/"
    "cpu_resident_progress_benchmark.py"
)
HALF_LOAD = "controlled_half_96"
FULL_LOAD = "controlled_full_180"
LOAD_SPECS = {
    HALF_LOAD: {
        "resident_workers": 96,
        "profiles": (1, 2, 4, 8, 16, 32, 64, 96),
        "expected_regime": "cpu_external_heavy",
        "resource_state": "half_loaded",
    },
    FULL_LOAD: {
        "resident_workers": 180,
        "profiles": (1, 2, 4, 8),
        "expected_regime": "cpu_external_saturated",
        "resource_state": "full_loaded",
    },
}


def build_native_cpu_controlled_resident_worker_campaign(
    *,
    allow_launch: bool = False,
    tag: str = "native_cpu_controlled_resident_worker_20260727",
    half_nodes: Sequence[str] = ("node001", "node002"),
    full_nodes: Sequence[str] = ("node003", "node005"),
    replicates: int = 3,
    replicate_offset: int = 0,
    max_parallel: int = 4,
    half_profiles: Sequence[int] | None = None,
    full_profiles: Sequence[int] | None = None,
) -> dict[str, Any]:
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    if int(replicate_offset) < 0:
        raise ValueError("replicate_offset must be nonnegative")
    lane_specs = {
        **{str(node): HALF_LOAD for node in half_nodes},
        **{str(node): FULL_LOAD for node in full_nodes},
    }
    if len(lane_specs) != len(tuple(half_nodes)) + len(tuple(full_nodes)):
        raise ValueError("half/full node sets must be disjoint")
    nodes = tuple(sorted(lane_specs))
    load_specs = _resolved_load_specs(
        half_profiles=half_profiles,
        full_profiles=full_profiles,
    )
    deploy_rows = _deploy_nodes(
        nodes,
        tag=f"{tag}_resident_campaign",
        max_parallel=max_parallel,
    )
    progress_deploy_rows = _deploy_progress_nodes(
        nodes,
        tag=f"{tag}_resident_campaign",
        max_parallel=max_parallel,
    )
    initial_surveys = _survey_nodes(
        nodes,
        tag=f"{tag}_resident_initial",
        max_parallel=max_parallel,
    )
    initial_by_node = {
        str(row["node"]): row for row in initial_surveys
    }
    native_deploy_ready = {
        str(row["node"])
        for row in deploy_rows
        if row.get("ready")
    }
    progress_deploy_ready = {
        str(row["node"])
        for row in progress_deploy_rows
        if row.get("ready")
    }
    launchable_nodes = {
        node
        for node in nodes
        if node in native_deploy_ready
        and node in progress_deploy_ready
        if initial_by_node.get(node, {}).get("ready")
        and initial_by_node[node].get("dispatch_external_cpu_regime")
        == "cpu_external_idle"
    }
    plan = _build_plan(
        lane_specs=lane_specs,
        replicates=replicates,
        replicate_offset=replicate_offset,
        load_specs=load_specs,
    )

    if not allow_launch:
        rows = [
            {**row, "launched": False, "status": "MANIFEST"}
            for row in plan
        ]
        lane_results = []
    else:
        rows, lane_results = _run_lanes(
            plan=plan,
            lane_specs=lane_specs,
            launchable_nodes=launchable_nodes,
            tag=tag,
            max_parallel=max_parallel,
            load_specs=load_specs,
        )

    accepted = [row for row in rows if row.get("measurement_valid")]
    expected_cells = {
        (load_class, profile, replicate)
        for load_class, spec in LOAD_SPECS.items()
        if any(value == load_class for value in lane_specs.values())
        for profile in spec["profiles"]
        if profile in load_specs[load_class]["profiles"]
        for replicate in range(
            int(replicate_offset),
            int(replicate_offset) + int(replicates),
        )
    }
    accepted_cells = {
        (
            str(row.get("load_class")),
            int(row.get("workers") or 0),
            int(row.get("replicate") or 0),
        )
        for row in accepted
    }
    coverage_ready = expected_cells <= accepted_cells
    failed = [
        row
        for row in rows
        if row.get("launched") and not row.get("measurement_valid")
    ]
    return {
        "gate": "native_cpu_controlled_resident_worker_campaign",
        "schema_version": 1,
        "allow_launch": bool(allow_launch),
        "tag": tag,
        "load_specs": load_specs,
        "lane_specs": lane_specs,
        "replicates": int(replicates),
        "replicate_offset": int(replicate_offset),
        "deploy_rows": deploy_rows,
        "progress_deploy_rows": progress_deploy_rows,
        "initial_surveys": initial_surveys,
        "launchable_nodes": sorted(launchable_nodes),
        "planned_count": len(plan),
        "launched_count": sum(bool(row.get("launched")) for row in rows),
        "accepted_count": len(accepted),
        "failed_count": len(failed),
        "coverage_ready": coverage_ready,
        "status": (
            "PASS"
            if allow_launch and coverage_ready and not failed
            else ("MANIFEST" if not allow_launch else "PARTIAL")
        ),
        "lane_results": lane_results,
        "rows": rows,
        "measurement_contract": {
            "target_eta": (
                "startup + outer loop + checkpoint + final save of the target task"
            ),
            "resident_setup_excluded_from_target_eta": True,
            "resident_progress_source": "tqdm",
            "target_progress_source": "tqdm/progress_wrapper",
            "legacy_scheduler_limits_bypassed": True,
            "capacity_rule": "resident_workers + target_workers <= 192",
            "no_touch_user_tasks": True,
        },
    }


def _build_plan(
    *,
    lane_specs: Mapping[str, str],
    replicates: int,
    replicate_offset: int = 0,
    load_specs: Mapping[str, Mapping[str, Any]] = LOAD_SPECS,
) -> list[dict[str, Any]]:
    nodes_by_load: dict[str, list[str]] = {}
    for node, load_class in sorted(lane_specs.items()):
        nodes_by_load.setdefault(load_class, []).append(node)
    plan = []
    for load_class, nodes in sorted(nodes_by_load.items()):
        spec = load_specs[load_class]
        cursor = 0
        for local_replicate in range(int(replicates)):
            replicate = int(replicate_offset) + local_replicate
            profiles = tuple(spec["profiles"])
            rotated = profiles[local_replicate % len(profiles) :] + profiles[
                : local_replicate % len(profiles)
            ]
            for profile in rotated:
                node = nodes[cursor % len(nodes)]
                cursor += 1
                plan.append(
                    {
                        "cell_id": (
                            f"{load_class}_w{profile}_r{replicate}"
                        ),
                        "node": node,
                        "load_class": load_class,
                        "resident_workers": int(spec["resident_workers"]),
                        "workers": int(profile),
                        "replicate": int(replicate),
                        "expected_regime": str(spec["expected_regime"]),
                        "resource_state": str(spec["resource_state"]),
                    }
                )
    return plan


def _run_lanes(
    *,
    plan: Sequence[Mapping[str, Any]],
    lane_specs: Mapping[str, str],
    launchable_nodes: set[str],
    tag: str,
    max_parallel: int,
    load_specs: Mapping[str, Mapping[str, Any]] = LOAD_SPECS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lanes: dict[str, list[dict[str, Any]]] = {}
    for row in plan:
        lanes.setdefault(str(row["node"]), []).append(dict(row))

    def run_lane(node: str, cells: Sequence[Mapping[str, Any]]):
        if node not in launchable_nodes:
            return (
                [
                    {
                        **dict(row),
                        "launched": False,
                        "measurement_valid": False,
                        "status": "DEFERRED_INITIAL_STATE_NOT_IDLE",
                    }
                    for row in cells
                ],
                {
                    "node": node,
                    "resident_started": False,
                    "resident_stopped": True,
                    "status": "DEFERRED_INITIAL_STATE_NOT_IDLE",
                },
            )
        load_class = str(lane_specs[node])
        spec = load_specs[load_class]
        lane_root = (
            f"/tmp/scheduleurm_cpu_resident/{tag}/{node}/{load_class}"
        )
        ready_file = f"{lane_root}/resident.ready.json"
        stop_file = f"{lane_root}/resident.stop"
        resident_log = f"{lane_root}/resident.log"
        setup = (
            f"mkdir -p {shlex.quote(lane_root)}; "
            f"rm -f {shlex.quote(ready_file)} {shlex.quote(stop_file)}; "
            f"exec python3 -u {shlex.quote(REMOTE_BENCHMARK)} "
            f"--workers {int(spec['resident_workers'])} --duration-s 7200 "
            f"--ready-file {shlex.quote(ready_file)} "
            f"--stop-file {shlex.quote(stop_file)} "
            f"--label {shlex.quote(f'{tag}_{node}_{load_class}')} "
            f"> {shlex.quote(resident_log)} 2>&1"
        )
        resident_proc = _remote_popen(node, setup, timeout_s=7300)
        resident_ready = _wait_for_ready(
            node=node,
            ready_file=ready_file,
            resident_proc=resident_proc,
            tag=tag,
        )
        output = []
        try:
            if not resident_ready.get("ready"):
                return (
                    [
                        {
                            **dict(row),
                            "launched": False,
                            "measurement_valid": False,
                            "status": "DEFERRED_RESIDENT_NOT_READY",
                        }
                        for row in cells
                    ],
                    {
                        "node": node,
                        "resident_started": True,
                        "resident_stopped": False,
                        "status": "RESIDENT_NOT_READY",
                        "resident_ready": resident_ready,
                    },
                )
            for row in cells:
                preflight = _survey_nodes(
                    (node,),
                    tag=f"{tag}_{row['cell_id']}_loaded_preflight",
                    max_parallel=1,
                )[0]
                if (
                    not preflight.get("ready")
                    or preflight.get("dispatch_external_cpu_regime")
                    != row["expected_regime"]
                ):
                    output.append(
                        {
                            **dict(row),
                            "launched": False,
                            "measurement_valid": False,
                            "status": "DEFERRED_RESIDENT_REGIME_MISMATCH",
                            "dispatch_preflight": preflight,
                        }
                    )
                    continue
                cell = CpuCompletionCell(
                    cell_id=str(row["cell_id"]),
                    workload_key="cpu_heavy_local_bench",
                    workload_label="cpu",
                    node=node,
                    workers=int(row["workers"]),
                    steps=160,
                    work_items=90_000,
                    checkpoint_interval=40,
                    checkpoint_bytes=8 * 1024 * 1024,
                    resource_state=str(row["resource_state"]),
                    observed_external_load=float(
                        preflight.get("dispatch_external_cpu_fraction")
                        or 0.0
                    )
                    * 192.0,
                )
                measured = _run_cell(
                    cell,
                    repeat_index=int(row["replicate"]),
                    run_prefix=tag,
                    deploy_bundle=False,
                )
                model = measured.get("completion_model") or {}
                ready = bool(
                    measured.get("measurement_valid")
                    and measured.get("completion_model_ready")
                    and model.get("natural_exit")
                    and float(model.get("checkpoint_observed_s") or 0.0)
                    > 0.0
                    and float(model.get("save_observed_s") or 0.0) > 0.0
                )
                output.append(
                    {
                        **measured,
                        **{
                            key: row[key]
                            for key in (
                                "load_class",
                                "resident_workers",
                                "workers",
                                "replicate",
                                "expected_regime",
                            )
                        },
                        "dispatch_preflight": preflight,
                        "measurement_valid": ready,
                        "status": "READY" if ready else "NOT_READY",
                    }
                )
        finally:
            _run_remote_capture(
                node,
                f"touch {shlex.quote(stop_file)}",
                Path("/tmp") / f"{tag}_{node}_resident_stop",
                timeout_s=60,
            )
            try:
                resident_proc.wait(timeout=90)
            except Exception:
                resident_proc.terminate()
                resident_proc.wait(timeout=15)
        return (
            output,
            {
                "node": node,
                "resident_started": True,
                "resident_stopped": resident_proc.poll() is not None,
                "resident_returncode": resident_proc.returncode,
                "resident_ready": resident_ready,
                "status": "COMPLETE",
            },
        )

    rows = []
    lane_results = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_parallel), len(lanes) or 1))
    ) as pool:
        futures = {
            pool.submit(run_lane, node, cells): node
            for node, cells in sorted(lanes.items())
        }
        for future in as_completed(futures):
            lane_rows, lane_result = future.result()
            rows.extend(lane_rows)
            lane_results.append(lane_result)
    rows.sort(
        key=lambda row: (
            str(row.get("load_class") or ""),
            int(row.get("replicate") or 0),
            int(row.get("workers") or 0),
            str(row.get("node") or ""),
        )
    )
    lane_results.sort(key=lambda row: str(row.get("node") or ""))
    return rows, lane_results


def _wait_for_ready(
    *,
    node: str,
    ready_file: str,
    resident_proc,
    tag: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + 180.0
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if resident_proc.poll() is not None:
            return {
                "ready": False,
                "reason": "resident_exited_before_ready",
                "returncode": resident_proc.returncode,
            }
        rc, out, err = _run_remote_capture(
            node,
            (
                f"if [ -s {shlex.quote(ready_file)} ]; then "
                f"cat {shlex.quote(ready_file)}; else exit 3; fi"
            ),
            Path("/tmp") / f"{tag}_{node}_resident_ready_{attempt}",
            timeout_s=30,
        )
        if int(rc) == 0:
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                return {"ready": True, "payload": payload}
        time.sleep(1.0)
    return {"ready": False, "reason": "resident_ready_timeout"}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _parse_nodes(raw: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in str(raw or "").split(",")
            if item.strip()
        )
    )


def _parse_profiles(raw: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(item.strip())
                for item in str(raw or "").split(",")
                if item.strip()
            }
        )
    )


def _resolved_load_specs(
    *,
    half_profiles: Sequence[int] | None,
    full_profiles: Sequence[int] | None,
) -> dict[str, dict[str, Any]]:
    requested = {
        HALF_LOAD: (
            tuple(int(value) for value in half_profiles)
            if half_profiles is not None
            else tuple(LOAD_SPECS[HALF_LOAD]["profiles"])
        ),
        FULL_LOAD: (
            tuple(int(value) for value in full_profiles)
            if full_profiles is not None
            else tuple(LOAD_SPECS[FULL_LOAD]["profiles"])
        ),
    }
    resolved: dict[str, dict[str, Any]] = {}
    for load_class, profiles in requested.items():
        allowed = {
            int(value) for value in LOAD_SPECS[load_class]["profiles"]
        }
        selected = tuple(sorted({int(value) for value in profiles}))
        if not selected or any(value not in allowed for value in selected):
            raise ValueError(f"invalid profiles for {load_class}: {selected}")
        resident = int(LOAD_SPECS[load_class]["resident_workers"])
        if any(resident + value > 192 for value in selected):
            raise ValueError(f"capacity-unsafe profiles for {load_class}")
        resolved[load_class] = {
            **LOAD_SPECS[load_class],
            "profiles": selected,
        }
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument(
        "--tag",
        default="native_cpu_controlled_resident_worker_20260727",
    )
    parser.add_argument("--half-nodes", default="node001,node002")
    parser.add_argument("--full-nodes", default="node003,node005")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--replicate-offset", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument(
        "--half-profiles",
        default=",".join(str(value) for value in LOAD_SPECS[HALF_LOAD]["profiles"]),
    )
    parser.add_argument(
        "--full-profiles",
        default=",".join(str(value) for value in LOAD_SPECS[FULL_LOAD]["profiles"]),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ARTIFACT_ROOT
            / "native_cpu_controlled_resident_worker_campaign_20260727.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_native_cpu_controlled_resident_worker_campaign(
        allow_launch=bool(args.allow_launch),
        tag=str(args.tag),
        half_nodes=_parse_nodes(args.half_nodes),
        full_nodes=_parse_nodes(args.full_nodes),
        replicates=int(args.replicates),
        replicate_offset=int(args.replicate_offset),
        max_parallel=int(args.max_parallel),
        half_profiles=_parse_profiles(args.half_profiles),
        full_profiles=_parse_profiles(args.full_profiles),
    )
    _atomic_write_json(args.output, result)
    print(args.output)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "planned_count",
                    "launched_count",
                    "accepted_count",
                    "failed_count",
                    "coverage_ready",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if not args.allow_launch or result.get("coverage_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
