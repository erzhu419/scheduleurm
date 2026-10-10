"""Real measured-service fixtures for clock equivalence and hidden future releases."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundCompetitionDecision,
)
from algorithm.lane_refill_simple_controls import select_single_lane_control
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.empirical_service_lane_replay import (
    EmpiricalServiceTape, MeanService, simulate_empirical_service_trajectory,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import (
    simulate_observed_rate_contextual_trajectory,
)


ARTIFACTS = Path(__file__).resolve().parents[1] / "md/experiment_artifacts"


class EmpiricalServiceLaneReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = json.loads((ARTIFACTS / "aligned_hardware_BG_empirical_service_samples_20261004.json").read_text())
        cls.catalogues = {row["node"]: row for row in cls.bank["catalogues"]}
        cls.models = load_models(cls.bank["mean_model_snapshots"])
        cls.original = json.loads((ARTIFACTS /
            "aligned_hardware_background_competition_continuous_confirmation_20260930_protocol.json").read_text())

    def planner(self, model):
        return BackgroundCompetitionContinuousLaneRollout(model.options, "empirical-service-test", 8, 128)

    def test_mean_execution_matches_original_clock_on_four_real_catalogues(self):
        native_models = load_models(self.original["models"])
        self.assertEqual({model.node for model in self.models}, {model.node for model in native_models})
        for source, models in (("native", native_models), ("empirical_mean", self.models)):
            for model in models:
                fixture = next(tape for tape in self.original["tapes"]
                    if tape["node"] == model.node and tape["load"] == .8 and tape["seed"] == 100001)
                # A literal short prefix of the registered real-rate arrival tape.
                births = tuple(tuple(birth) for birth in fixture["arrivals"] if birth[0] < 40)
                self.assertTrue(births)
                for policy in ("MW", "BG"):
                    with self.subTest(catalogue=source, node=model.node, policy=policy):
                        before, after = self.planner(model), self.planner(model)
                        old_selector = before.maxweight if policy == "MW" else before
                        new_selector = after.maxweight if policy == "MW" else after
                        old = simulate_observed_rate_contextual_trajectory(
                            model, births, 40, 80, old_selector, policy=policy)
                        new = simulate_empirical_service_trajectory(
                            model, births, 40, 80, new_selector, MeanService(), policy=policy)
                        for key in ("jobs", "snapshots", "lane_events", "rollout_work", "decision_count",
                                    "frame_count", "total_decision_delay_s", "occupied_lane_time_s",
                                    "scheduled_idle_lane_time_s", "barrier_idle_lane_time_s", "waiting_area_s",
                                    "last_frame_end"):
                            self.assertEqual(new[key], old[key], key)
                        self.assertEqual(len(new["frames"]), len(old["frames"]))
                        for actual, expected in zip(new["frames"], old["frames"], strict=True):
                            self.assertEqual({key: actual[key] for key in expected}, expected)
                        self.assertEqual(after.rng.getstate(), before.rng.getstate())

    def test_running_lane_real_duration_cannot_change_same_prefix_decision(self):
        model = next(model for model in self.models if model.node == "gpu_b")
        catalogue = self.catalogues[model.node]
        queue = (2, 2, 2)
        births = tuple((5., coordinate, index) for coordinate, count in enumerate(queue) for index in range(count))
        first = self.planner(model).maxweight(model.options, queue, (5., 5.), 5., 0,
            tuple((0.,) * count for count in queue), queue)
        index, = first.selection.option_indices
        option = model.options[index]
        self.assertFalse(option.loaded)
        self.assertEqual(sum(option.requirements), 1)
        action = next(action for action in catalogue["actions"] if action["action_id"] == option.action_id)
        records = action["training"][:2]
        self.assertEqual([record["empty_wave"] for record in records], [1, 2])
        self.assertNotEqual(records[0]["duration"], records[1]["duration"])
        cutoff = 5 + min(record["duration"] for record in records) / 2
        executions, visible, decisions = [], [], []

        for record in records:
            planner, calls, selected = self.planner(model), [], []

            def selector(*args):
                decision = planner.maxweight(*args) if not any(args[1]) or not calls else planner(*args)
                if any(args[1]):
                    calls.append(args[1:])
                    selected.append(asdict(decision))
                return decision

            class FirstMeasuredService(MeanService):
                def __init__(self):
                    self.used = False

                def draw(self, selected_option):
                    if not self.used:
                        self.used = True
                        self.asserted_action = selected_option.action_id
                        return dict(record)
                    return super().draw(selected_option)

            service = FirstMeasuredService()
            result = simulate_empirical_service_trajectory(
                model, births, 5, cutoff, selector, service, policy="prefix-check")
            self.assertEqual(service.asserted_action, option.action_id)
            self.assertEqual(len(calls), 2)
            visible.append(calls[1])
            decisions.append(selected[1])
            executions.append(next(frame for frame in result["frames"]
                if frame["start"] == 5 and frame["lane_ids"] == (1,)))
        self.assertEqual(visible[0], visible[1])
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(executions[0]["available_before"], (5 + option.duration, 5.))
        self.assertNotEqual(executions[0]["execution_available_before"], executions[1]["execution_available_before"])

    def test_nth_action_service_repeats_despite_other_action_draws(self):
        for model in self.models:
            samples = {action["action_id"]: action["training"]
                for action in self.catalogues[model.node]["actions"]}
            for index, option in enumerate(model.options):
                with self.subTest(node=model.node, action=option.action_id):
                    first = EmpiricalServiceTape(samples, "paired-physical-service")
                    second = EmpiricalServiceTape(samples, "paired-physical-service")
                    other = model.options[(index + 1) % len(model.options)]
                    for occurrence in range(7):
                        reference = first.draw(option)
                        for _ in range(occurrence + 1):
                            second.draw(other)
                        reproduced = second.draw(option)
                        self.assertEqual(reproduced, reference)
                        self.assertEqual(reproduced["action_occurrence"], occurrence)

    def test_loaded_service_duration_and_offsets_are_one_measured_composite_record(self):
        for model in self.models:
            samples = {action["action_id"]: action["training"]
                for action in self.catalogues[model.node]["actions"]}
            service = EmpiricalServiceTape(samples, "joint-loaded-service")
            for option in model.options:
                if not option.loaded:
                    continue
                with self.subTest(node=model.node, action=option.action_id):
                    for occurrence in range(20):
                        actual = service.draw(option)
                        expected = samples[option.action_id][actual["sample_index"]]
                        self.assertEqual({key: actual[key] for key in expected}, expected)
                        self.assertEqual(actual["action_occurrence"], occurrence)
                    record = samples[option.action_id][1]

                    class FixedMeasuredService:
                        def draw(self, selected_option):
                            return record

                    # One completed frame executes this real joint record, not coordinatewise draws.
                    single = replace(model, lanes=1, options=(option,))
                    queue = option.requirements
                    births = tuple((0., coordinate, slot)
                        for coordinate, count in enumerate(queue) for slot in range(count))
                    def selector(options, queues, *_context):
                        return BackgroundCompetitionDecision(
                            selection=select_single_lane_control(options, queues, 1, 0),
                            eligible=False, candidate_count=0, rollout_calls=0, rollout_option_probes=0,
                            root_option_probes=len(options), baseline_predicted_completion_s=None,
                            chosen_predicted_completion_s=None, changed=False, queue_limit=sum(queues))

                    result = simulate_empirical_service_trajectory(
                        single, births, 0, 1, selector, FixedMeasuredService(), policy="loaded-record")
                    frame, = result["frames"]
                    self.assertEqual(frame["execution_duration"], record["duration"])
                    self.assertEqual(frame["execution_offsets"], tuple(tuple(group) for group in record["completion_offsets"]))
                    self.assertEqual(result["lane_events"][0]["local_end"], record["duration"])
                    for coordinate, group in enumerate(record["completion_offsets"]):
                        actual = [job["completion"] for job in result["jobs"] if job["coordinate"] == coordinate]
                        self.assertEqual(actual, sorted(group))


if __name__ == "__main__":
    unittest.main()
