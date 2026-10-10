"""Tests protect cutoff diagnosis, not a second simulation of the experiment."""
from copy import deepcopy
import unittest

from algorithm.experiments.bg_empirical_cutoff_diagnostic import analyse_pair


MODEL = dict(workloads=["LLM"], options=[dict(action_id="llm:p1", requirements=[1], loaded=False)])


def result(specifications):
    jobs, frames = [], []
    for index, (arrival, start, completion) in enumerate(specifications):
        jobs.append(dict(coordinate=0, index=index, arrival=arrival, start=start, completion=completion))
        if start is not None:
            frames.append(dict(start=start, option_indices=[0], lane_ids=[index % 4], duration=10.,
                execution_duration=completion - start, execution_offsets=[[completion - start]],
                rollout=dict(changed=False, eligible=True, planning_overflow=False,
                             exact_root_scores=dict(baseline_root=0, actual_root=0))))
    return dict(jobs=jobs, frames=frames, observation_end=1200.)


class CutoffDiagnosticTest(unittest.TestCase):
    def test_inclusive_birth_completion_and_dispatch_boundaries(self):
        saved = result([(299., 899., 901.), (300., 899., 900.),
                        (900., 900., 901.), (900.1, None, None)])
        analysis = analyse_pair(saved, saved, MODEL)
        arm = analysis["arms"]["BG"]
        cutoff = arm["cutoff"]["total"]
        self.assertEqual(arm["cohort_jobs"], 2)
        self.assertEqual((cutoff["completed"], cutoff["queued"], cutoff["inflight"]), (1, 0, 1))
        self.assertEqual(arm["snapshots"][0]["total"]["arrived"], 1)
        self.assertEqual(arm["middle_area_job_s"]["total"], dict(queued=599., inflight=1., unfinished=600.))
        self.assertEqual([frame["frame_index"] for frame in arm["active_frames"]], [0, 2])
        self.assertEqual(arm["active_frames"][0]["cohort_snapshot"]["total"]["arrived"], 0)
        self.assertEqual(arm["active_frames"][1]["decision"]["baseline_root_action_id"], "llm:p1")
        self.assertEqual(arm["snapshots"][-1]["total"]["arrived"], 2)

    def test_paired_cancellation_and_shared_queue_to_inflight_transition(self):
        mw = result([(880., 890., 905.), (881., 891., 895.), (882., 905., 910.)])
        bg = result([(880., 890., 895.), (881., 891., 905.), (882., 895., 910.)])
        paired = analyse_pair(mw, bg, MODEL)["paired"]
        self.assertEqual(paired["cutoff_delta"], dict(completed=0, queued=-1, inflight=1, unfinished=0))
        self.assertEqual(len(paired["bg_only_unfinished"]), 1)
        self.assertEqual(len(paired["mw_only_unfinished"]), 1)
        self.assertEqual(paired["shared_unfinished_jobs"], 1)
        transition = paired["state_transitions"]["total"]["mw_rows_bg_columns"]
        self.assertEqual((transition["inflight"]["completed"], transition["completed"]["inflight"],
                          transition["queued"]["inflight"]), (1, 1, 1))
        self.assertEqual(paired["flow_decomposition"]["total"]["delta_sums"],
                         dict(flow_s=0., waiting_s=-10., execution_s=10.))
        detail = paired["bg_only_unfinished"][0]
        self.assertEqual(detail["arms"]["BG"]["dispatch"]["frame_index"], 1)
        self.assertEqual(detail["arms"]["BG"]["dispatch"]["actual_job_offset"], 14.)

    def test_restricted_time_decomposition_with_unstarted_and_late_completion(self):
        mw = result([(800., 1100., 1250.), (850., None, None)])
        bg = result([(800., 1000., 1300.), (850., 1190., 1210.)])
        analysis = analyse_pair(mw, bg, MODEL)
        paired = analysis["paired"]
        self.assertEqual(paired["flow_decomposition"]["total"]["delta_sums"],
                         dict(flow_s=0., waiting_s=-110., execution_s=110.))
        self.assertEqual(paired["full_area_delta_job_s"], dict(queued=-110., inflight=110., unfinished=0.))
        self.assertEqual(analysis["arms"]["maxweight"]["cutoff"]["total"]["remaining_unknown_jobs"], 1)
        self.assertEqual(analysis["arms"]["BG"]["cutoff"]["total"]["known_remaining_wait_s"], 390.)
        broken = deepcopy(bg)
        broken["jobs"][0]["completion"] += 1
        with self.assertRaisesRegex(ValueError, "does not match frame"):
            analyse_pair(mw, broken, MODEL)

    def test_pairing_error_and_per_workload_areas(self):
        mw = result([(800., 850., 920.), (850., 880., 890.)])
        bg = result([(800., 860., 910.), (850., 890., 900.)])
        model = dict(workloads=["LLM", "CNN"], options=[
            dict(action_id="llm:p1", requirements=[1, 0], loaded=False),
            dict(action_id="cnn:p1", requirements=[0, 1], loaded=False)])
        for arm in (mw, bg):
            arm["jobs"][1]["coordinate"] = 1
            arm["frames"][0]["execution_offsets"].append([])
            arm["frames"][1]["option_indices"] = [1]
            arm["frames"][1]["execution_offsets"].insert(0, [])
        analysis = analyse_pair(mw, bg, model)
        flow = analysis["paired"]["flow_decomposition"]
        self.assertEqual(flow["total"]["delta_sums"]["flow_s"], 0.)
        self.assertEqual([row["delta_sums"]["flow_s"] for row in flow["per_workload"]], [-10., 10.])
        bg["jobs"][0]["arrival"] += 1
        with self.assertRaisesRegex(ValueError, "identities and arrivals differ"):
            analyse_pair(mw, bg, model)


if __name__ == "__main__":
    unittest.main()
