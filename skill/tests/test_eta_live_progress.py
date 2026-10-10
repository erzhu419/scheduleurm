from __future__ import annotations

import json
import pytest

from skill import eta_tracker as tracker
from skill.scheduler_eta.state import eta_source_tag


def _training_tail(points, total=20000):
    return f"SCHEDULEURM_ETA_TOTAL_UNITS={total}\n" + "\n".join(
        json.dumps({"source_events": current + 5000, "continuation_updates": current,
                    "elapsed_seconds": stamp, "metrics": {"train/update": current + 50000}})
        for current, stamp in points
    )


def test_branch_progress_overrides_unrelated_runtime_history():
    tail = "ACTION_RISK_START seed=8400 branches=0/512\nACTION_RISK_BRANCH branches=38/512"
    assert tracker.parse_progress(tail) == (38, 512)
    eta = tracker.compute_eta_seconds(tail, elapsed_s=300, fallback_ewma_s=998)
    assert eta == int((512 - 38) * 300 / 38)
    projection = tracker.runtime_projection(tail, elapsed_s=300)
    assert projection["eta_s"] == eta
    assert eta_source_tag(projection["source"]) == "live"


def test_restored_branches_do_not_count_as_work_completed_this_run():
    tail = "ACTION_RISK_START branches=100/512\nACTION_RISK_BRANCH branches=130/512"
    projection = tracker.runtime_projection(tail, elapsed_s=300)
    assert projection["current"] == 130
    assert projection["unit_s"] == 10
    assert tracker.compute_eta_seconds(tail, elapsed_s=300) == 3820


def test_branch_warmup_does_not_use_restored_progress_as_observed_rate():
    tail = "ACTION_RISK_START branches=100/512\nACTION_RISK_BRANCH branches=102/512"
    assert tracker.runtime_projection(tail, elapsed_s=300) is None
    assert tracker.compute_eta_seconds(tail, elapsed_s=300, fallback_ewma_s=1000) == 700


def test_json_updates_use_recent_elapsed_deltas_and_protocol_total():
    tail = _training_tail([(1000, 391), (2000, 491), (3000, 591)])
    projection = tracker.runtime_projection(tail, elapsed_s=700)
    assert projection["current"] == 3000
    assert projection["total_units"] == 20000
    assert projection["unit_s"] == .1
    assert projection["eta_s"] == 1700
    assert projection["total_s"] == 2400
    assert tracker.compute_eta_seconds(tail, elapsed_s=700, fallback_ewma_s=4000) == 1700
    assert eta_source_tag(projection["source"]) == "live"


def test_json_protocol_is_parsed_locally_without_remote_python():
    protocol = json.dumps({"protocol": {"source_event_budget": 25000, "warmup_events": 5000}})
    tail = "===ETA_PROTOCOL===\n" + protocol + "\n===ETA_PROTOCOL_END===\n"
    tail += _training_tail([(1000, 391), (2000, 491), (3000, 591)]).split("\n", 1)[1]
    projection = tracker.runtime_projection(tail, elapsed_s=700)
    assert projection["total_units"] == 20000
    assert projection["eta_s"] == 1700


def test_json_rate_reflects_collection_or_refit_pause():
    tail = _training_tail([(8000, 1000), (9000, 1100), (10000, 1400)])
    assert tracker.compute_eta_seconds(tail, elapsed_s=1450) == 2000


def test_json_without_total_or_enough_points_keeps_history_fallback():
    tail = _training_tail([(1000, 391), (2000, 491)])
    assert tracker.runtime_projection(tail, elapsed_s=600) is None
    assert tracker.compute_eta_seconds(tail, elapsed_s=600, fallback_ewma_s=3000) == 2400
    tail = _training_tail([(1000, 391), (2000, 491), (3000, 591)]).split("\n", 1)[1]
    assert tracker.runtime_projection(tail, elapsed_s=700) is None


def test_json_truncated_tail_and_reset_do_not_mix_training_runs():
    tail = "truncated metrics}\n" + _training_tail(
        [(10000, 1000), (1000, 100), (2000, 200), (3000, 300)])
    projection = tracker.runtime_projection(tail, elapsed_s=310)
    assert projection["eta_s"] == 1700
    assert projection["window_points"] == 3


def _protocol_tail(protocol, rows, evaluation=None):
    document = {"protocol": protocol}
    if evaluation is not None:
        document["eta_evaluation"] = evaluation
    return "===ETA_PROTOCOL===\n" + json.dumps(document) + "\n===ETA_PROTOCOL_END===\n" + "\n".join(rows)


def test_thesis_dict_reprs_use_episode_timing_not_nested_numpy_metrics():
    rows = [f"{{'episode': {i}, 'total_updates': {i * 460}, 'elapsed_seconds': {i * 180}, "
            "'metrics': {'belief_entropy': np.float64(2.98)}}" for i in (31, 32, 33)]
    tail = _protocol_tail({"episodes": 100}, rows)
    projection = tracker.runtime_projection(tail, elapsed_s=6000)
    assert projection["current"] == 33
    assert projection["total_units"] == 100
    assert projection["unit_s"] == 180
    assert projection["eta_s"] == 12060
    assert tracker.compute_eta_seconds(tail, elapsed_s=6000) == 12060


def test_saved_cs_episode_budget_matches_episode_counter_not_update_count():
    rows = [repr({'episode': i, 'updates': 480, 'wall_seconds': i * 70}) for i in (95, 96, 97)]
    tail = _protocol_tail({'cs_episodes': 100}, rows)
    projection = tracker.runtime_projection(tail, elapsed_s=6900)
    assert projection['current'] == 97
    assert projection['total_units'] == 100
    assert projection['unit_s'] == 70
    assert tracker.compute_eta_seconds(tail, elapsed_s=6900) == 210


def test_thesis_current_wrapper_reads_its_protocol_without_collector_restart(tmp_path):
    root = tmp_path / "bus_control_repair_v3"
    root.mkdir()
    (root / "protocol.json").write_text(json.dumps({"cs_episodes": 100}))
    cmd = f"python {root}/node_run.py cs --phase train"
    tail = "\n".join(repr({"episode": i, "elapsed_seconds": i * 100}) for i in (5, 6, 7))
    assert tracker.compute_eta_seconds(tail, elapsed_s=800, cmd=cmd) == 9300
    assert tracker.runtime_projection(tail, elapsed_s=800, cmd=cmd)["total_units"] == 100
    assert tracker.runtime_projection(tail, elapsed_s=800, cmd=cmd.replace('cs --phase train', 'mc --phase train')) is None
    assert tracker.runtime_projection(tail, elapsed_s=800, cmd=cmd.replace('--phase train', '--phase evaluate')) is None


@pytest.mark.parametrize('total', [100, 120])
def test_resac_episode_progress_uses_logged_budget_and_rate_not_inner_steps(total):
    tail = '\n'.join(
        f'Episode {i}/{total} | reward=-1362436.8 | steps={i * 4971} | '
        f'replay={i * 4971} | mode=normal | switches=0 | {rate}s/episode'
        for i, rate in ((37, 107.7), (38, 106.9), (39, 109.7))
    )
    assert tracker.parse_progress(tail) == (39, total)
    projection = tracker.runtime_projection(tail, elapsed_s=4500)
    assert projection['source'] == 'seconds_per_unit'
    assert projection['unit_s'] == 109.7
    assert projection['eta_s'] == int((total - 39) * 109.7)
    assert tracker.compute_eta_seconds(tail, elapsed_s=4500, fallback_ewma_s=5000) == projection['eta_s']


@pytest.mark.parametrize('kind,points', [
    ('cs_native', ((79, 4433.0157), (80, 4489.9384), (81, 4545.3476))),
    ('cs_symbolic', ((33, 3346.6046), (34, 3453.0166), (35, 3558.4783))),
])
@pytest.mark.parametrize('total', [100, 120])
def test_thesis_legacy_dict_episode_eta_reads_real_protocol(tmp_path, kind, points, total):
    root = tmp_path / 'bus_legacy_dual_v1'
    root.mkdir()
    (root / 'protocol.json').write_text(json.dumps({'episodes': total}))
    cmd = f'python {root}/node_run.py --kind {kind} --phase train --seed 64001'
    tail = '\n'.join(
        f"{{'native_reward': -9876., 'metrics': {{'belief_entropy': np.float64(2.73), 'L_eff': inf}}, "
        f"'episode': {i}, 'horizon_seconds': 18000, 'elapsed_seconds': {stamp}}}"
        for i, stamp in points
    )
    projection = tracker.runtime_projection(tail, elapsed_s=5000, cmd=cmd)
    unit_s = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
    assert projection['source'] == 'progress_window'
    assert projection['current'] == points[-1][0]
    assert projection['total_units'] == total
    assert projection['unit_s'] == pytest.approx(unit_s)
    assert projection['eta_s'] == int((total - points[-1][0]) * unit_s)
    assert tracker.compute_eta_seconds(tail, elapsed_s=5000, cmd=cmd) == projection['eta_s']
    assert tracker.runtime_projection(tail, elapsed_s=5000, cmd=cmd.replace('--phase train', '--phase evaluate')) is None
    assert tracker._thesis_episode_protocol(cmd + ' --smoke') == {'episodes': 1}


def _cfcmt_tail(points, **evaluation):
    rows = [json.dumps({"phase": "full_source_iql_continuation", "phase_updates": i,
                        "completed_updates": 55000 + i, "elapsed_seconds": stamp})
            for i, stamp in points]
    return _protocol_tail({"continuation_updates": 20000, "evaluation": {"cases": [0, 1, 2]}},
                          rows, evaluation or None)


def test_cfcmt_continuation_excludes_checkpoint_updates_and_includes_evaluation():
    tail = _cfcmt_tail([(1000, 80), (2000, 150), (3000, 220)],
                       remaining_seconds=165, training_finished=False)
    projection = tracker.runtime_projection(tail, elapsed_s=230)
    assert projection["current"] == 3000
    assert projection["unit_s"] == .07
    assert projection["eta_s"] == 1355
    assert tracker.compute_eta_seconds(tail, elapsed_s=230) == 1355


def test_cfcmt_without_evaluation_timing_does_not_claim_whole_job_finished():
    tail = _cfcmt_tail([(18000, 1300), (19000, 1370), (20000, 1440)])
    assert tracker.runtime_projection(tail, elapsed_s=1450) is None
    assert tracker.compute_eta_seconds(tail, elapsed_s=1450, fallback_ewma_s=2000) == 550


def test_cfcmt_evaluation_survives_training_rows_leaving_the_tail():
    tail = _cfcmt_tail([], training_finished=True, remaining_seconds=95,
                       completed_cases=1, total_cases=3, case_seconds=55)
    projection = tracker.runtime_projection(tail, elapsed_s=1510)
    assert projection["eta_s"] == 95
    assert projection["current"] == 1
    assert projection["total_units"] == 3
    assert tracker.compute_eta_seconds(tail, elapsed_s=1510) == 95


def test_structured_window_does_not_combine_different_phases():
    tail = _training_tail([(1000, 100), (2000, 200), (3000, 300)])
    tail += '\n' + json.dumps({"phase": "new", "continuation_updates": 4000, "elapsed_seconds": 400})
    assert tracker.runtime_projection(tail, elapsed_s=420) is None


def _generic_tail(points=((10, 50), (11, 55), (12, 60)), **extra):
    return "\n".join("SCHEDULEURM_PROGRESS " + json.dumps({
        "current": current, "total": 100, "elapsed_seconds": stamp, **extra,
    }) for current, stamp in points)


@pytest.mark.parametrize('current,total', [
    ('epoch', 'epochs'), ('episode', 'total_episodes'), ('step', 'total_steps'),
    ('iteration', 'max_iters'), ('iter', 'max_iters'), ('completed', 'total'),
])
def test_new_projects_use_plain_json_progress_without_name_adapters(current, total):
    tail = '\n'.join(json.dumps({current: i, total: 100, 'elapsed_seconds': i * 5})
                     for i in (10, 11, 12))
    projection = tracker.runtime_projection(tail, elapsed_s=900, cmd='python new_project.py')
    assert projection['current'] == 12
    assert projection['total_units'] == 100
    assert projection['eta_s'] == 440
    assert tracker.compute_eta_seconds(tail, elapsed_s=900) == 440


def test_generic_dict_and_matching_command_total_are_supported():
    tail = '\n'.join(f"{{'epoch': {i}, 'elapsed_seconds': {i * 5}, 'metrics': np.float64(4)}}"
                     for i in (10, 11, 12))
    assert tracker.compute_eta_seconds(tail, elapsed_s=900, cmd='python new.py --n-epochs 100') == 440
    assert tracker.runtime_projection(tail, elapsed_s=900, cmd='python new.py --max-steps 100') is None


def test_generic_restored_counts_do_not_become_this_runs_rate():
    tail = _generic_tail(points=((1010, 30), (1020, 60), (1030, 90)), total=2000)
    projection = tracker.runtime_projection(tail, elapsed_s=10000)
    assert projection['eta_s'] == 2910
    assert projection['unit_s'] == 3


def test_contract_multi_phase_requires_future_timing_even_with_inner_tqdm():
    tail = _generic_tail(phase='train', phase_index=1, phase_count=2)
    tail += '\n100/100 [00:10<00:00, 10.00it/s]'
    assert tracker.runtime_projection(tail, elapsed_s=300) is None
    assert tracker.compute_eta_seconds(tail, elapsed_s=300, fallback_ewma_s=1000) == 700
    assert tracker.runtime_projection_from_log(tail) is None
    tail = _generic_tail(phase='train', phase_index=1, phase_count=2, remaining_phases_seconds=60)
    assert tracker.runtime_projection(tail, elapsed_s=300)['eta_s'] == 500


def test_generic_window_resets_on_phase_changes():
    tail = _generic_tail(phase='train')
    tail += '\n' + _generic_tail(points=((1, 10),), phase='evaluate')
    assert tracker.runtime_projection(tail, elapsed_s=100) is None


def test_contract_native_eta_has_priority_but_plain_json_keeps_tqdm_priority():
    bar = '50/100 [00:10<00:12, 4.00it/s]'
    tail = _generic_tail(remaining_seconds=17, phase_index=1, phase_count=2) + '\n' + bar
    assert tracker.compute_eta_seconds(tail, elapsed_s=900) == 17
    assert tracker.runtime_projection(tail, elapsed_s=900)['eta_s'] == 17
    plain = _generic_tail().replace('SCHEDULEURM_PROGRESS ', '') + '\n' + bar
    assert tracker.compute_eta_seconds(plain, elapsed_s=900) == 12
    assert tracker.runtime_projection(plain, elapsed_s=900)['source'] == 'tqdm'


def test_completed_contract_is_eta_zero_not_a_scheduler_terminal_transition():
    projection = tracker.runtime_projection(_generic_tail(points=((100, 500),)), elapsed_s=550)
    assert projection['eta_s'] == 0
    assert projection['total_s'] == 550
    assert projection['current'] == projection['total_units'] == 100


def test_generic_local_profile_seeds_first_queued_run():
    projection = tracker.runtime_projection_from_log(_generic_tail())
    assert projection['source'] == 'local_test_progress'
    assert projection['total_s'] == 500
    assert tracker.runtime_projection_from_log(_generic_tail(), observed_duration_s=100)['total_s'] == 540


def test_silent_job_and_incomplete_contract_do_not_invent_progress():
    tail = '{"elapsed_seconds": 500, "target_calls": 5120}\nDONE'
    assert tracker.runtime_projection(tail, elapsed_s=600) is None
    assert tracker.compute_eta_seconds(tail, elapsed_s=600, fallback_ewma_s=1000) == 400
    assert tracker.runtime_projection(_generic_tail(points=((10, 50), (11, 55))), elapsed_s=100) is None


@pytest.mark.parametrize('counter,budget', [('epoch', 'epochs'), ('step', 'total_steps'),
                                          ('iteration', 'max_iters'), ('current', 'total')])
def test_generic_saved_protocol_supplies_matching_counter_budget(counter, budget):
    rows = [json.dumps({counter: i, 'elapsed_seconds': i * 5}) for i in (10, 11, 12)]
    tail = _protocol_tail({budget: 100}, rows)
    assert tracker.runtime_projection(tail, elapsed_s=1000)['eta_s'] == 440
    assert tracker.compute_eta_seconds(tail, elapsed_s=1000) == 440


def test_generic_protocol_does_not_confuse_units_or_hide_future_evaluation():
    rows = [json.dumps({'epoch': i, 'elapsed_seconds': i * 5}) for i in (10, 11, 12)]
    assert tracker.runtime_projection(_protocol_tail({'total_steps': 100}, rows), elapsed_s=1000) is None
    tail = _protocol_tail({'epochs': 100, 'evaluation': {'cases': [0, 1, 2]}}, rows)
    assert tracker.runtime_projection(tail, elapsed_s=1000) is None
    assert tracker.compute_eta_seconds(tail, elapsed_s=1000, fallback_ewma_s=2000) == 1000


def test_explicit_episode_total_precedes_shared_protocol_default():
    rows = [json.dumps({'episode': i, 'elapsed_seconds': i * 5}) for i in (10, 11, 12)]
    tail = _protocol_tail({'episodes': 100}, rows)
    assert tracker.compute_eta_seconds(tail, elapsed_s=1000, cmd='python new.py --episodes 120') == 540
    rows = [json.dumps({'episode': i, 'episodes': 120, 'elapsed_seconds': i * 5}) for i in (10, 11, 12)]
    assert tracker.runtime_projection(_protocol_tail({'episodes': 100}, rows), elapsed_s=1000)['eta_s'] == 540


@pytest.mark.parametrize('as_repr', [False, True])
def test_offline_initialization_uses_protocol_total_and_wall_seconds(as_repr):
    rows = []
    for current, stamp in ((37000, 5079.263), (38000, 5218.000), (39000, 5354.257)):
        row = {'offline_updates': current, 'wall_seconds': stamp,
               'metrics': {'train/update': 999999, 'model_updates': 1000}}
        rows.append(repr(row) if as_repr else json.dumps(row))
    tail = _protocol_tail({'offline_updates': 50000, 'source_events': 25000, 'warmup_events': 5000}, rows)
    projection = tracker.runtime_projection(tail, elapsed_s=5400, cmd='python new_wrapper.py --phase initialize')
    assert projection['total_units'] == 50000
    assert projection['current'] == 39000
    assert projection['unit_s'] == pytest.approx((5354.257 - 5079.263) / 2000)
    assert projection['eta_s'] == 1512
    assert tracker.compute_eta_seconds(tail, elapsed_s=5400, fallback_ewma_s=1000) == 1512


def test_offline_budget_override_and_completion_do_not_use_current_as_total():
    rows = [json.dumps({'offline_updates': i, 'wall_seconds': i * .1}) for i in (1000, 2000, 3000)]
    tail = _protocol_tail({'offline_updates': 50000}, rows)
    projection = tracker.runtime_projection(tail, elapsed_s=500, cmd='python train.py --offline-updates 10000')
    assert projection['current'] == 3000
    assert projection['total_units'] == 10000
    assert projection['eta_s'] == 700
    assert tracker.runtime_projection('\n'.join(rows), elapsed_s=500) is None
    tail = _protocol_tail({'offline_updates': 3000}, rows[-1:])
    assert tracker.runtime_projection(tail, elapsed_s=500)['eta_s'] == 0


def test_online_updates_use_event_budget_minus_warmup_not_offline_budget():
    rows = [json.dumps({'source_events': i + 5000, 'updates': i, 'wall_seconds': i * .2})
            for i in (1000, 2000, 3000)]
    tail = _protocol_tail({'offline_updates': 50000, 'source_events': 25000, 'warmup_events': 5000}, rows)
    projection = tracker.runtime_projection(tail, elapsed_s=900)
    assert projection['current'] == 3000
    assert projection['total_units'] == 20000
    assert projection['eta_s'] == 3400
