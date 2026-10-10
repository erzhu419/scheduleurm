"""ETA log-tail snapshot helpers used outside the scheduler state lock."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import PurePosixPath
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class EtaTailOutputDeps:
    node_is_windows: Callable[[str], bool]
    ps_quote: Callable[[str], str]
    windows_tail_ps: Callable[..., str]
    windows_path_for_task: Callable[[dict, str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    run_on: Callable[..., tuple[int, str, str]]
    remote_path_for_node: Callable[[str, str], str]
    timeout_s: int = 10
    remote_python: Callable[[str], str] = lambda node: "python3"


@dataclass(frozen=True)
class EtaTailSnapshotDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    eta_refresh_due: Callable[[dict], bool]
    eta_tail_targets: Callable[[dict], tuple[dict, list]]
    eta_tail_outputs_by_node: Callable[[dict], dict]
    notify: Callable[..., Any]


def _protocol_paths(task: dict, node: str,
                    remote_path_for_node: Callable[[str, str], str]) -> list[str]:
    tokens = shlex.split(task.get("cmd") or "")
    if any(token in ("--evaluate", "--eval", "evaluate", "eval") or
           token in ("--phase=evaluate", "--phase=eval") for token in tokens):
        return []

    def resolve(value: str) -> PurePosixPath:
        path = PurePosixPath(value)
        return path if path.is_absolute() else PurePosixPath(task.get("cwd") or ".") / path

    paths = []
    for index, token in enumerate(tokens):
        name, _, value = token.partition("=")
        if name not in ("--out", "--output", "--output-dir", "--output_dir", "--run-dir", "--run_dir"):
            continue
        value = value or (tokens[index + 1] if index + 1 < len(tokens) else "")
        if value:
            output = resolve(value)
            if output.suffix == ".json":
                output = output.parent
            paths.extend((output / "results/protocol.json", output / "protocol.json"))
    script = next((resolve(token) for token in tokens if token.endswith(".py")), None)
    if script is not None:
        arguments = {}
        for index, token in enumerate(tokens):
            name, _, value = token.partition("=")
            if name in ("--phase", "--mode", "--objective", "--arm", "--seed"):
                arguments[name] = value or (tokens[index + 1] if index + 1 < len(tokens) else "")
        if arguments.get("--phase") == "train" and all(
                arguments.get(name) for name in ("--mode", "--objective", "--seed")):
            prefix = "full_smoke" if "--full-smoke" in tokens else "smoke" if "--smoke" in tokens else "runs"
            run = (script.parent / prefix / arguments["--mode"] / arguments["--objective"] /
                   (arguments.get("--arm", "*") + "_seed" + arguments["--seed"]))
            paths.extend((run / "results/protocol.json", run / "protocol.json"))
        paths.extend((script.with_name(script.stem + "_protocol.json"), script.parent / "protocol.json"))
    return list(dict.fromkeys(remote_path_for_node(node, str(path)) for path in paths))


def _bus_protocol_probe(task: dict, node: str,
                        remote_path_for_node: Callable[[str, str], str]) -> str | None:
    cmd = task.get("cmd") or ""
    if "bus_control_repair_v3" not in cmd and "bus_cross_city_native_v1" not in cmd:
        return None
    tokens = shlex.split(cmd)

    def argument(name: str, default=None):
        value = next((token.split("=", 1)[1] for token in tokens if token.startswith(name + "=")), None)
        if value is not None:
            return value
        if name in tokens and tokens.index(name) + 1 < len(tokens):
            return tokens[tokens.index(name) + 1]
        return default

    script = next((PurePosixPath(token) for token in tokens if token.endswith(".py")), None)
    if script is None:
        return None
    root = script.parent
    if not root.is_absolute():
        root = PurePosixPath(task.get("cwd") or ".") / root
    root = PurePosixPath(remote_path_for_node(node, str(root)))
    if script.parent.name == "bus_control_repair_v3":
        if argument("--phase") != "train" or (script.name == "node_run.py" and "cs" not in tokens):
            return None
        if "--smoke" in tokens:
            return "printf '===ETA_PROTOCOL===\\n{\"protocol\":{\"episodes\":1}}\\n===ETA_PROTOCOL_END===\\n'"
        output = root / "cs_runs" / str(argument("--objective")) / f"{argument('--arm')}_seed{argument('--seed')}"
        path = output / "results/protocol.json"
        return ("echo '===ETA_PROTOCOL==='; "
                f"head -c 16384 {shlex.quote(str(path))} 2>/dev/null; "
                "printf '\\n===ETA_PROTOCOL_END===\\n'")
    if script.parent.name != "bus_cross_city_native_v1":
        return None
    evaluation_only = script.name == "source_dense_adapted_eval.py"
    if script.name in ("source_dense_continuation.py", "source_dense_adapted_eval.py"):
        prefix = {"iql": "source_dense_iql_continuation", "real-td": "source_dense_real_only",
                  "model-td": "source_dense_continuation", "adapted": "source_dense_adapted_eval",
                  "target-adapted": "target_adapted_eval"}.get(argument("--campaign", "adapted" if evaluation_only else "model-td"))
    else:
        prefix = script.stem
    if prefix is None or argument("--seed") is None:
        return None
    # Sample only protocol/summary JSON in the existing per-node SSH batch.
    # File completion times include interpreter/SUMO startup omitted by summary elapsed_seconds.
    code = f"""import json, statistics, time
from pathlib import Path
root = Path({str(root)!r})
prefix = {prefix!r}
evaluation_only = {evaluation_only!r}
protocol = json.loads((root / (prefix + '_protocol.json')).read_text())
branch = {argument('--branch')!r}
admission = {'--admission' in tokens!r}
cases = protocol['evaluation']['cases'][:1] if admission else protocol['evaluation']['cases']
case_ids = [row['case'] for row in cases]
artifact = protocol.get('branches', {{}}).get(branch, {{}}).get('artifact_folder', prefix)
folder = root / artifact / ('admission' if admission else 'seed' + {argument('--seed')!r})
started = {float(task.get('started_at') or 0)!r}
samples = {{case: [] for case in case_ids}}
def complete_time(path):
    try:
        return path.stat().st_mtime if json.loads(path.read_text()).get('complete') else None
    except (OSError, ValueError):
        return None
references = [prefix] + sorted(path.name[:-len('_protocol.json')] for path in root.glob('*_protocol.json')
                                if path.name != prefix + '_protocol.json')
def evaluation_settings(protocol):
    # These fields drive the evaluator; descriptive/comparison prose does not.
    evaluation = protocol.get('evaluation', {{}})
    return {{key: evaluation.get(key) for key in ('episode_seconds', 'demand_scale', 'rule', 'rule_gain', 'cases')}}
for candidate in references:
    try:
        reference_protocol = json.loads((root / (candidate + '_protocol.json')).read_text())
    except (OSError, ValueError):
        continue
    if evaluation_settings(reference_protocol) != evaluation_settings(protocol):
        continue
    artifacts = [spec['artifact_folder'] for spec in reference_protocol.get('branches', {{}}).values()
                 if spec.get('artifact_folder')] or [candidate]
    for artifact in artifacts:
        for seed in reference_protocol.get('training_seeds', []):
            reference = root / artifact / ('seed%d' % seed)
            previous = complete_time(reference / ('export_summary.json' if candidate.endswith('adapted_eval') else 'train_summary.json'))
            for case in case_ids:
                finished = complete_time(reference / 'evaluation' / ('case%02d' % case) / 'summary.json')
                if previous is not None and finished is not None and finished > previous:
                    samples[case].append(finished - previous)
                previous = finished
    if all(samples.values()):
        break
train_end = complete_time(folder / ('export_summary.json' if evaluation_only else 'train_summary.json'))
training_finished = train_end is not None and train_end >= started
previous = train_end if training_finished else None
done, observed = [], []
for case in case_ids:
    finished = complete_time(folder / 'evaluation' / ('case%02d' % case) / 'summary.json')
    if training_finished and finished is not None and finished >= started:
        done.append(case)
        if previous is not None and finished > previous:
            observed.append(finished - previous)
        previous = finished
typical = statistics.median(observed) if observed else None
durations = [statistics.median(samples[case]) if samples[case] else typical
             for case in case_ids if case not in done]
remaining = None if any(value is None for value in durations) else sum(durations)
if remaining is not None and previous is not None and len(done) < len(case_ids):
    remaining = max(0, remaining - max(0, time.time() - previous))
    if remaining == 0:
        remaining = None
evaluation = dict(training_finished=training_finished, completed_cases=len(done),
                  total_cases=len(case_ids), remaining_seconds=remaining,
                  case_seconds=statistics.median([value for values in samples.values() for value in values] or observed or [0]))
print('===ETA_PROTOCOL===')
print(json.dumps(dict(protocol=dict(continuation_updates=0 if evaluation_only else (4 if admission else protocol['continuation_updates']),
                                  evaluation=protocol['evaluation']), eta_evaluation=evaluation)))
print('===ETA_PROTOCOL_END===')
"""
    python = remote_path_for_node(node, tokens[0])
    return f"{shlex.quote(python)} -c {shlex.quote(code)}"


_ETA_TAIL_READER = """import base64, glob, json, re, subprocess, sys
from collections import deque
for item in json.loads(base64.b64decode(sys.argv[1])):
    protocol = ''
    if item['protocol']:
        try:
            result = subprocess.run(item['protocol'], shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, timeout=8)
            if result.returncode == 0:
                protocol = result.stdout.decode('utf-8', errors='replace')
        except subprocess.TimeoutExpired:
            pass
    if not protocol:
        for path in item['protocol_paths']:
            try:
                if '*' in path:
                    # A wrapper's default arm is usable only when this seed has
                    # one saved run protocol; never pick an arbitrary sibling.
                    matches = glob.glob(path)
                    if len(matches) != 1:
                        continue
                    path = matches[0]
                with open(path, 'rb') as handle:
                    document = json.loads(handle.read(65536))
                budget = document.get('protocol', document)
                if not isinstance(budget, dict):
                    budget = document
                if not any(key in budget for key in ('source_event_budget', 'source_events',
                            'continuation_updates', 'offline_updates', 'total_offline_updates',
                            'episodes', 'cs_episodes', 'total_episodes', 'epochs', 'total_epochs', 'total_steps',
                            'max_steps', 'max_iters', 'total')):
                    continue
                protocol = '===ETA_PROTOCOL===\\n' + json.dumps(document) + '\\n===ETA_PROTOCOL_END===\\n'
                break
            except (OSError, ValueError, AttributeError):
                continue
    try:
        with open(item['path'], 'rb') as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - item['bytes']))
            tail = handle.read(item['bytes'])
            context = deque(maxlen=item['limit'])
            saved = deque(maxlen=100)
            if size:
                pattern = re.compile(item['pattern'].encode()) if item['pattern'] else None
                handle.seek(0)
                for line in handle:
                    if pattern and pattern.search(line):
                        context.append(line)
                    if b'Results saved to:' in line:
                        saved.append(line)
    except OSError:
        if not protocol:
            continue
        tail, context, saved = b'', [], []
    print('===ETA_LOG_' + item['id'] + '===')
    if protocol:
        print(protocol.rstrip())
    sys.stdout.flush()
    sys.stdout.buffer.write(b''.join(context) + tail + b'\\n' + b''.join(saved) + b'\\n')
    sys.stdout.buffer.flush()
"""


def eta_tail_outputs_by_node(by_node: dict, *, deps: EtaTailOutputDeps) -> dict:
    if not by_node:
        return {}

    def _probe(node: str):
        entries = by_node[node]
        if deps.node_is_windows(node):
            parts = []
            for task, log_path in entries:
                tid = task["id"]
                parts.append(f"Write-Output {deps.ps_quote(f'===ETA_LOG_{tid}===')}")
                parts.append(
                    deps.windows_tail_ps(
                        deps.windows_path_for_task(task, log_path),
                        max_bytes=4096,
                    )
                )
            try:
                rc, out, _ = deps.run_windows_ps(
                    node,
                    "\n".join(parts),
                    timeout=max(1, int(deps.timeout_s)),
                    check=False,
                )
                return out if rc == 0 else None
            except Exception:
                return None

        specs = []
        for task, log_path in entries:
            tid = task["id"]
            task_text = " ".join(
                str(task.get(k) or "") for k in ("project", "signature", "description", "cmd")
            ).lower()
            protocol_probe = _bus_protocol_probe(task, node, deps.remote_path_for_node)
            pattern, limit, max_bytes = "", 200, 16384
            if protocol_probe:
                pass
            elif "bus_action_coverage" in task_text or "bus_uncertainty_actions" in task_text:
                pattern, limit = r"^ACTION_RISK_START ", 1
            elif "freqduet" in task_text or "run_freqduet_" in task_text:
                pattern, limit, max_bytes = r"^(Shard jobs|SKIP |DONE )", None, 65536
            elif "bapr" in task_text:
                pattern, limit, max_bytes = r"^(TRAIN SUBPROCESS:|Training complete! Total time:|PAIR .*COMPLETE:)", 40, 65536
            elif "kg_op" in task_text or "benchmark_sota.py" in task_text:
                pattern, max_bytes = r"(\[kg-inner\]|\[sota\]|Iter |Iteration |Step |ETA |DONE|Results saved to:)", 65536
            specs.append({"id": tid, "path": log_path, "bytes": max_bytes,
                          "pattern": pattern, "limit": limit, "protocol": protocol_probe,
                          "protocol_paths": _protocol_paths(task, node, deps.remote_path_for_node)})
        outputs = []
        batches, batch, batch_bytes = [], [], 0
        for spec in specs:
            spec_bytes = len(json.dumps(spec).encode())
            if batch and (len(batch) >= 32 or batch_bytes + spec_bytes > 32768):
                batches.append(batch)
                batch, batch_bytes = [], 0
            batch.append(spec)
            batch_bytes += spec_bytes
        if batch:
            batches.append(batch)
        reader = base64.b64encode(_ETA_TAIL_READER.encode()).decode('ascii')
        bootstrap = f'import base64; exec(base64.b64decode("{reader}"))'
        for batch in batches:
            # Encoding prevents quote expansion across bash/sudo/ssh/su hops.
            payload = base64.b64encode(json.dumps(batch).encode()).decode('ascii')
            cmd = (f"{shlex.quote(deps.remote_python(node))} -c {shlex.quote(bootstrap)} {payload}")
            try:
                rc, out, _ = deps.run_on(node, cmd, timeout=max(1, int(deps.timeout_s)), check=False)
                if rc == 0:
                    outputs.append(out)
            except Exception:
                continue
        return "\n".join(outputs) if outputs else None

    nodes_list = list(by_node.keys())
    with ThreadPoolExecutor(max_workers=max(1, len(nodes_list))) as ex:
        return dict(zip(nodes_list, ex.map(_probe, nodes_list)))


def eta_tail_snapshot_outside_lock(
    purpose: str = "eta-tail",
    *,
    deps: EtaTailSnapshotDeps,
) -> tuple[dict, set]:
    try:
        with deps.state_lock(shared=True, purpose=f"{purpose}:snapshot"):
            snapshot = deps.load_state()
            if not deps.eta_refresh_due(snapshot):
                return {}, set()
            by_node, _pure_ewma = deps.eta_tail_targets(snapshot)
            running_ids = {
                str(t.get("id") or "")
                for t in snapshot.get("tasks", [])
                if t.get("status") == "running" and t.get("id")
            }
            if not running_ids:
                return {}, running_ids
    except Exception as e:
        deps.notify(
            "eta_tail_snapshot_error",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        return {}, set()
    try:
        return deps.eta_tail_outputs_by_node(by_node), running_ids
    except Exception as e:
        deps.notify(
            "eta_tail_error_outer",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        return {}, running_ids
