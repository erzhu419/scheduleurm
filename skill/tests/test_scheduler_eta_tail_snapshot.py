from __future__ import annotations

from contextlib import contextmanager
import base64
import json
import os
import subprocess
import shlex
import sys
import time

from skill.scheduler_eta.tail_snapshot import (
    EtaTailOutputDeps,
    EtaTailSnapshotDeps,
    _protocol_paths,
    _bus_protocol_probe,
    eta_tail_outputs_by_node,
    eta_tail_snapshot_outside_lock,
)


def test_eta_tail_snapshot_tails_after_releasing_state_lock():
    events = []
    lock_depth = {"value": 0}
    state = {"tasks": [{"id": "t1", "status": "running"}]}

    @contextmanager
    def state_lock(**kwargs):
        events.append(("lock_enter", kwargs))
        lock_depth["value"] += 1
        try:
            yield
        finally:
            lock_depth["value"] -= 1
            events.append(("lock_exit", None))

    def tail_outputs(by_node):
        events.append(("tail", lock_depth["value"], by_node))
        return {"node001": "tail"}

    out, ids = eta_tail_snapshot_outside_lock(
        "unit",
        deps=EtaTailSnapshotDeps(
            state_lock=state_lock,
            load_state=lambda: state,
            eta_refresh_due=lambda snapshot: True,
            eta_tail_targets=lambda snapshot: ({"node001": [({"id": "t1"}, "/tmp/t.log")]}, []),
            eta_tail_outputs_by_node=tail_outputs,
            notify=lambda *args, **kwargs: None,
        ),
    )

    assert out == {"node001": "tail"}
    assert ids == {"t1"}
    assert events[0] == ("lock_enter", {"shared": True, "purpose": "unit:snapshot"})
    assert events[1] == ("lock_exit", None)
    assert events[2][0:2] == ("tail", 0)


def test_mcwm_probe_reads_this_runs_protocol_and_retains_multiple_json_rows(tmp_path):
    output = tmp_path / "run with spaces"
    (output / "results").mkdir(parents=True)
    (output / "results/protocol.json").write_text(json.dumps({
        "protocol": {"source_event_budget": 25000, "warmup_events": 5000}}))
    log = tmp_path / "train.log"
    log.write_text("\n".join(json.dumps({
        "continuation_updates": index * 1000, "elapsed_seconds": index * 100,
        "metrics": {"padding": "x" * 1500}}) for index in range(1, 9)))
    task = {"id": "t1", "cmd": f"python /experiments/mcwm_explore_v5/run.py --out '{output}'"}
    commands = []

    def run_on(node, cmd, **kwargs):
        commands.append(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr

    deps = EtaTailOutputDeps(
        node_is_windows=lambda node: False, ps_quote=lambda value: value,
        windows_tail_ps=lambda *args, **kwargs: "", windows_path_for_task=lambda task, path: path,
        run_windows_ps=lambda *args, **kwargs: (0, "", ""), run_on=run_on,
        remote_path_for_node=lambda node, path: path)
    text = eta_tail_outputs_by_node({"n1": [(task, str(log))]}, deps=deps)["n1"]
    assert "===ETA_PROTOCOL===" in text
    assert '"source_event_budget": 25000' in text
    assert text.count('"continuation_updates"') == 8
    assert len(commands) == 1
    from skill import eta_tracker
    assert eta_tracker.runtime_projection(text, elapsed_s=800)["total_units"] == 20000


def test_protocol_discovery_is_not_tied_to_project_or_entrypoint():
    assert _protocol_paths({"cmd": "python /experiments/mcwm_explore_v5/run.py --evaluate --out /run"},
                           "n1", lambda node, path: path) == []
    paths = _protocol_paths({"cmd": "python /experiments/new_family/node_run.py --out /run"},
                            "n1", lambda node, path: path)
    assert paths == ['/run/results/protocol.json', '/run/protocol.json',
                     '/experiments/new_family/node_run_protocol.json', '/experiments/new_family/protocol.json']


def test_mcwm_relative_output_is_resolved_under_remote_cwd():
    paths = _protocol_paths({"cwd": "/work dir", "cmd": "python mcwm_v5/run.py --out=runs/seed1"},
                            "n1", lambda node, path: path)
    assert paths[0] == '/work dir/runs/seed1/results/protocol.json'


def test_mcwm_probe_uses_existing_node_workspace_mapping():
    paths = _protocol_paths(
        {"cmd": "python /home/user/mine_code/Thesis/mcwm_v5/run.py --out /home/user/mine_code/Thesis/runs/seed1"},
        "node007", lambda node, path: path.replace("/home/user/mine_code", "/home/researcher/scheduleurm_work"))
    assert "/home/researcher/scheduleurm_work/Thesis/runs/seed1/results/protocol.json" in paths
    assert all("/home/user/mine_code" not in path for path in paths)


def test_new_wrapper_budget_reaches_parser_in_same_batch_without_subprocess(tmp_path):
    output = tmp_path / 'runs/seed1'
    (output / 'results').mkdir(parents=True)
    (output / 'results/protocol.json').write_text(json.dumps({
        'protocol': {'source_event_budget': 25000, 'warmup_events': 5000}}))
    log = tmp_path / 'train.log'
    log.write_text('\n'.join(json.dumps({'continuation_updates': i, 'elapsed_seconds': i * .1})
                             for i in (1000, 2000, 3000)))
    commands = []

    def run_on(node, cmd, **kwargs):
        commands.append(cmd)
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
        return result.returncode, result.stdout, result.stderr

    task = {'id': 'wrapper', 'cwd': str(tmp_path), 'cmd': 'python new_version/node_run.py --out runs/seed1'}
    output = eta_tail_outputs_by_node({'n1': [(task, str(log))]}, deps=_local_tail_deps(run_on))['n1']
    from skill import eta_tracker
    assert eta_tracker.runtime_projection(output, elapsed_s=1000)['eta_s'] == 1700
    assert len(commands) == 1
    assert json.loads(base64.b64decode(shlex.split(commands[0])[-1]))[0]['protocol'] is None


def test_script_protocol_is_fallback_and_command_total_overrides_it(tmp_path):
    (tmp_path / 'new_train_protocol.json').write_text(json.dumps({'epochs': 100}))
    log = tmp_path / 'train.log'
    log.write_text('\n'.join(json.dumps({'epoch': i, 'elapsed_seconds': i * 5}) for i in (10, 11, 12)))

    def run_on(node, cmd, **kwargs):
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
        return result.returncode, result.stdout, result.stderr

    task = {'id': 'new', 'cwd': str(tmp_path), 'cmd': 'python new_train.py --epochs 120'}
    output = eta_tail_outputs_by_node({'n1': [(task, str(log))]}, deps=_local_tail_deps(run_on))['n1']
    from skill import eta_tracker
    assert eta_tracker.runtime_projection(output, elapsed_s=100, cmd=task['cmd'])['eta_s'] == 540


def test_thesis_probe_uses_this_runs_saved_episode_budget(tmp_path):
    root = tmp_path / "bus_control_repair_v3"
    output = root / "cs_runs/headway/smooth_hard_seed64001/results"
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps({"protocol": {"episodes": 100}}))
    task = {"cmd": f"python {root}/node_run.py cs --phase train --objective headway --arm smooth_hard --seed 64001"}
    command = _bus_protocol_probe(task, "n1", lambda node, path: path)
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
    assert '"episodes": 100' in result.stdout
    assert _bus_protocol_probe({"cmd": task['cmd'].replace('--phase train', '--phase evaluate')},
                               "n1", lambda node, path: path) is None


def test_default_arm_wrapper_uses_saved_run_budget_not_descriptive_protocol(tmp_path):
    root = tmp_path / 'csbapr_centering_v7'
    output = root / 'runs/dual/passenger/symbolic_no_center_seed64002'
    output.mkdir(parents=True)
    (root / 'protocol.json').write_text(json.dumps({'arms': ['symbolic_no_center']}))
    (output / 'protocol.json').write_text(json.dumps({'protocol': {'cs_episodes': 100}}))
    log = tmp_path / 'train.log'
    log.write_text('\n'.join(repr({'episode': i, 'wall_seconds': stamp})
                              for i, stamp in ((95, 6900), (96, 6970), (97, 7040))))
    task = {'id': 'centering', 'cmd':
            f'python {root}/node_run.py --phase train --mode dual --objective passenger --seed 64002'}
    commands = []

    def run_on(node, cmd, **kwargs):
        commands.append(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.returncode, result.stdout, result.stderr

    text = eta_tail_outputs_by_node({'n1': [(task, str(log))]}, deps=_local_tail_deps(run_on))['n1']
    from skill import eta_tracker
    projection = eta_tracker.runtime_projection(text, elapsed_s=7100, cmd=task['cmd'])
    assert projection['source'] == 'progress_window'
    assert projection['total_units'] == 100
    assert projection['current'] == 97
    assert projection['eta_s'] == 210
    assert len(commands) == 1
    paths = _protocol_paths(dict(task, cmd=task['cmd'].replace('--phase train', '--phase evaluate')),
                            'n1', lambda node, path: path)
    assert not paths
    assert not _protocol_paths(dict(task, cmd=task['cmd'].replace('--phase train', '--phase=evaluate')),
                               'n1', lambda node, path: path)


def test_run_protocol_precedes_script_defaults_and_ambiguous_arms_are_not_guessed(tmp_path):
    root = tmp_path / 'new_bus_wrapper'
    for arm in ('a', 'b'):
        output = root / f'smoke/holding/headway/{arm}_seed1'
        output.mkdir(parents=True)
        (output / 'protocol.json').write_text(json.dumps({'protocol': {'episodes': 1}}))
    (root / 'protocol.json').write_text(json.dumps({'episodes': 100}))
    log = tmp_path / 'train.log'
    log.write_text(repr({'episode': 1, 'wall_seconds': 70}))
    task = {'id': 'smoke', 'cmd':
            f'python {root}/node_run.py --phase=train --mode=holding --objective=headway --seed=1 --smoke'}

    def run_on(node, cmd, **kwargs):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.returncode, result.stdout, result.stderr

    deps = _local_tail_deps(run_on)
    from skill import eta_tracker
    text = eta_tail_outputs_by_node({'n1': [(task, str(log))]}, deps=deps)['n1']
    assert eta_tracker._eta_protocol_budget(eta_tracker._eta_protocol_document(text))['episodes'] == 100
    task['cmd'] += ' --arm=a'
    text = eta_tail_outputs_by_node({'n1': [(task, str(log))]}, deps=deps)['n1']
    assert eta_tracker.runtime_projection(text, elapsed_s=80, cmd=task['cmd'])['total_units'] == 1


def test_cfcmt_probe_counts_cases_and_uses_full_case_walltimes(tmp_path):
    root = tmp_path / "bus_cross_city_native_v1"
    root.mkdir()
    prefix = "source_dense_iql_continuation"
    (root / (prefix + '_protocol.json')).write_text(json.dumps({
        "continuation_updates": 20000, "training_seeds": [75101, 75102],
        "evaluation": {"cases": [{"case": case} for case in range(3)]}}))
    now = time.time()

    def summary(folder, relative, stamp):
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"complete": True, "elapsed_seconds": 48}))
        os.utime(path, (stamp, stamp))

    previous = root / prefix / "seed75101"
    summary(previous, 'train_summary.json', now - 1000)
    for case in range(3):
        summary(previous, f'evaluation/case{case:02d}/summary.json', now - 1000 + 60 * (case + 1))
    current = root / prefix / "seed75102"
    summary(current, 'train_summary.json', now - 90)
    summary(current, 'evaluation/case00/summary.json', now - 30)
    task = {"started_at": now - 2000,
            "cmd": f"{sys.executable} {root}/source_dense_continuation.py --campaign iql --seed 75102"}
    command = _bus_protocol_probe(task, "n1", lambda node, path: path)
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
    document = json.loads(result.stdout.split('\n')[1])
    meta = document['eta_evaluation']
    assert meta['training_finished'] is True
    assert meta['completed_cases'] == 1
    assert meta['total_cases'] == 3
    assert meta['case_seconds'] == 60
    assert 85 < meta['remaining_seconds'] <= 90
    from skill import eta_tracker
    projection = eta_tracker.runtime_projection(result.stdout, elapsed_s=1600, cmd=task['cmd'])
    assert 85 <= projection['eta_s'] <= 90
    # A retry must not adopt stale artifacts from its previous attempt.
    task['started_at'] = now - 20
    result = subprocess.run(_bus_protocol_probe(task, 'n1', lambda node, path: path),
                            shell=True, text=True, capture_output=True, check=True)
    meta = json.loads(result.stdout.split('\n')[1])['eta_evaluation']
    assert meta['training_finished'] is False
    assert meta['completed_cases'] == 0
    assert meta['remaining_seconds'] == 180
    # A first IQL cohort can use completed real-TD cases with the same fixed evaluation.
    alternate = root / 'source_dense_real_only'
    alternate.mkdir()
    previous.rename(alternate / 'seed75101')
    alternate_protocol = root / 'source_dense_real_only_protocol.json'
    alternate_protocol.write_text((root / (prefix + '_protocol.json')).read_text())
    result = subprocess.run(_bus_protocol_probe(task, 'n1', lambda node, path: path),
                            shell=True, text=True, capture_output=True, check=True)
    meta = json.loads(result.stdout.split('\n')[1])['eta_evaluation']
    assert meta['remaining_seconds'] == 180
    # Different evaluation horizons are not interchangeable history.
    document = json.loads(alternate_protocol.read_text())
    document['evaluation']['episode_seconds'] = 12000
    alternate_protocol.write_text(json.dumps(document))
    result = subprocess.run(_bus_protocol_probe(task, 'n1', lambda node, path: path),
                            shell=True, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout.split('\n')[1])['eta_evaluation']['remaining_seconds'] is None


def _local_tail_deps(run_on):
    return EtaTailOutputDeps(
        node_is_windows=lambda node: False, ps_quote=lambda value: value,
        windows_tail_ps=lambda *args, **kwargs: '', windows_path_for_task=lambda task, path: path,
        run_windows_ps=lambda *args, **kwargs: (0, '', ''), run_on=run_on,
        remote_path_for_node=lambda node, path: path, remote_python=lambda node: sys.executable)


def test_large_batch_uses_one_reader_per_32_tasks_and_keeps_partial_success(tmp_path):
    entries, commands = [], []
    for i in range(65):
        path = tmp_path / f't{i}.log'
        path.write_text('')
        entries.append(({'id': f't{i}', 'cmd': 'python new_project.py'}, str(path)))

    def run_on(node, cmd, **kwargs):
        commands.append(cmd)
        if len(commands) == 2:
            return 1, '', 'timed out'
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
        return result.returncode, result.stdout, result.stderr

    output = eta_tail_outputs_by_node({'n1': entries}, deps=_local_tail_deps(run_on))['n1']
    assert len(commands) == 3
    assert '===ETA_LOG_t0===' in output
    assert '===ETA_LOG_t31===' in output
    assert '===ETA_LOG_t32===' not in output
    assert '===ETA_LOG_t64===' in output
    assert all('grep ' not in command and 'tail -c ' not in command for command in commands)


def test_reader_preserves_action_start_and_freqduet_done_context(tmp_path):
    action = tmp_path / 'action.log'
    action.write_text('ACTION_RISK_START branches=100/512\n' + 'padding\n' * 10000
                      + 'ACTION_RISK_BRANCH branches=130/512\n')
    freq = tmp_path / 'freq.log'
    freq.write_text('Shard jobs [0, 300) of 300\n' + ''.join(f'DONE cfg{i}\n' for i in range(250))
                    + 'padding\n' * 10000)

    def run_on(node, cmd, **kwargs):
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
        return result.returncode, result.stdout, result.stderr

    output = eta_tail_outputs_by_node({'n1': [
        ({'id': 'action', 'cmd': 'python bus_action_coverage.py'}, str(action)),
        ({'id': 'freq', 'cmd': 'python run_freqduet.py'}, str(freq))]}, deps=_local_tail_deps(run_on))['n1']
    assert 'ACTION_RISK_START branches=100/512' in output
    assert 'ACTION_RISK_BRANCH branches=130/512' in output
    assert 'Shard jobs [0, 300) of 300' in output
    assert output.count('DONE cfg') == 250


def test_cfcmt_eval_only_uses_export_marker_and_zero_training_budget(tmp_path):
    root = tmp_path / 'bus_cross_city_native_v1'
    folder = root / 'target_adapted_eval/seed75101'
    folder.mkdir(parents=True)
    (root / 'target_adapted_eval_protocol.json').write_text(json.dumps({
        'continuation_updates': 0, 'training_seeds': [75101],
        'evaluation': {'cases': [{'case': case} for case in range(3)]}}))
    now = time.time()
    export = folder / 'export_summary.json'
    export.write_text(json.dumps({'complete': True}))
    os.utime(export, (now - 100, now - 100))
    for case in range(3):
        summary = folder / f'evaluation/case{case:02d}/summary.json'
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({'complete': True}))
        os.utime(summary, (now - 100 + 30 * (case + 1),) * 2)
    task = {'started_at': now - 1000, 'cmd': f'{sys.executable} {root}/source_dense_adapted_eval.py --campaign target-adapted --seed 75101'}
    result = subprocess.run(_bus_protocol_probe(task, 'n1', lambda node, path: path),
                            shell=True, text=True, capture_output=True, check=True)
    from skill import eta_tracker
    projection = eta_tracker.runtime_projection(result.stdout, elapsed_s=100)
    assert projection['eta_s'] == 0
    assert projection['current'] == projection['total_units'] == 3


def test_new_cfcmt_script_uses_branch_artifacts_and_same_evaluation_history(tmp_path):
    root = tmp_path / 'bus_cross_city_native_v1'
    root.mkdir()
    evaluation = {'episode_seconds': 24000, 'demand_scale': 1.0, 'rule': 'forward_proportional',
                  'rule_gain': .6, 'cases': [{'case': case} for case in range(3)]}
    protocol = {'continuation_updates': 20000, 'training_seeds': [75101],
                'branches': {'new_arm': {'artifact_folder': 'new_train/new_arm'}},
                'evaluation': dict(evaluation, interpretation='new method')}
    (root / 'new_train_protocol.json').write_text(json.dumps(protocol))
    (root / 'old_train_protocol.json').write_text(json.dumps({
        'training_seeds': [75101], 'branches': {'old': {'artifact_folder': 'old_train/old'}},
        'evaluation': dict(evaluation, interpretation='old method')}))
    now = time.time()
    for index, relative in enumerate(('train_summary.json', *[
            f'evaluation/case{case:02d}/summary.json' for case in range(3)])):
        path = root / 'old_train/old/seed75101' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'complete': True}))
        os.utime(path, (now - 1000 + index * 60,) * 2)
    task = {'started_at': now - 100,
            'cmd': f'{sys.executable} {root}/new_train.py train --branch new_arm --seed 75101'}
    result = subprocess.run(_bus_protocol_probe(task, 'n1', lambda node, path: path),
                            shell=True, text=True, capture_output=True, check=True)
    meta = json.loads(result.stdout.split('\n')[1])['eta_evaluation']
    assert meta['remaining_seconds'] == 180
    assert meta['training_finished'] is False
    from skill import eta_tracker
    rows = '\n'.join(json.dumps({'phase': 'new_arm', 'phase_updates': i,
                                 'completed_updates': i + 55000, 'elapsed_seconds': i * .1})
                     for i in (1000, 2000, 3000))
    assert eta_tracker.runtime_projection(result.stdout + rows, elapsed_s=400)['eta_s'] == 1880


def test_evaluation_batches_fit_real_multi_hop_ssh_argument_limit():
    from skill.tests.test_scheduler_remote_exec import _deps
    from skill.scheduler_remote.exec import remote_bash_command_for_node

    batches = []
    def run_on(node, cmd, **kwargs):
        wrapped = remote_bash_command_for_node('hpc', cmd, deps=_deps(), timeout=10)
        assert len(wrapped.encode()) < 131072
        batches.append(json.loads(base64.b64decode(shlex.split(cmd)[-1])))
        return 0, '', ''

    entries = [({'id': f't{i}', 'started_at': 1000,
                 'cmd': f'python /work/bus_cross_city_native_v1/new_train.py train --branch arm --seed {i}'},
                f'/work/t{i}.log') for i in range(32)]
    eta_tail_outputs_by_node({'hpc': entries}, deps=_local_tail_deps(run_on))
    assert len(batches) > 1
    assert [item['id'] for batch in batches for item in batch] == [f't{i}' for i in range(32)]


def test_offline_protocol_reaches_parser_and_tui_without_project_adapter(tmp_path):
    from skill import eta_tracker, tui

    (tmp_path / 'protocol.json').write_text(json.dumps({
        'offline_updates': 50000, 'source_events': 25000, 'warmup_events': 5000,
        'model_updates': 1000}))
    log = tmp_path / 'initialize.log'
    log.write_text('\n'.join(json.dumps({'offline_updates': current, 'wall_seconds': stamp})
                             for current, stamp in ((37000, 5079.263), (38000, 5218.000), (39000, 5354.257))))
    task = {'id': 'init', 'status': 'running', 'cwd': str(tmp_path),
            'cmd': 'python arbitrary_wrapper.py --phase initialize'}

    def run_on(node, cmd, **kwargs):
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
        return result.returncode, result.stdout, result.stderr

    output = eta_tail_outputs_by_node({'n1': [(task, str(log))]}, deps=_local_tail_deps(run_on))['n1']
    projection = eta_tracker.runtime_projection(output, elapsed_s=5400, cmd=task['cmd'])
    assert projection['total_units'] == 50000
    assert projection['eta_s'] == 1512
    task.update(eta_seconds=projection['eta_s'], eta_source=projection['source'],
                runtime_current_unit=projection['current'], runtime_total_units=projection['total_units'])
    assert tui._fmt_eta(task, {}) == '~25.2m live 39000/50000'
    task.update(eta_seconds=1800, eta_source='runtime_history_overrun')
    assert tui._fmt_eta(task, {}) == '? hist'
