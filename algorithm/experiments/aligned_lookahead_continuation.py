"""Cached diagnostic decisions and single-action replacement on a shared tape.

PrefixReplay is a selector callback for the frozen simulator, not a simulator.
DecisionPairs accounts for actual diagnostic searches separately from native
per-frame selection attempts, which remain available to the simulator.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Any, Callable, Sequence

from algorithm.budgeted_frame_candidates import BudgetedFrameSelection, select_budgeted_frame
from algorithm.lookahead_frame_candidates import LookaheadFrameSelection, select_lookahead_frame


_EXTRA_COUNTERS = (
    "local_score_evaluations", "cost_candidate_evaluations", "completion_offset_reads",
    "next_queue_coordinate_evaluations", "terminal_value_coordinate_evaluations",
)


class DecisionPairs:
    """Memoize the frozen decisions by visible queue, charging only new searches."""

    def __init__(self, model, arrival_rates, service_rates, budget=26):
        self.model = model
        self.arrival_rates = tuple(arrival_rates)
        self.service_rates = tuple(service_rates)
        self.budget = budget
        self._support: dict[tuple[int, ...], BudgetedFrameSelection] = {}
        self._lookahead: dict[tuple[int, ...], LookaheadFrameSelection] = {}
        self._counts = dict.fromkeys((
            "calls", "support_calls", "lookahead_calls", "candidate_attempts",
            "option_probes", "threshold_evaluations", *_EXTRA_COUNTERS,
        ), 0)

    def _record(self, selected: BudgetedFrameSelection, branch: str) -> None:
        self._counts["calls"] += 1
        self._counts[f"{branch}_calls"] += 1
        self._counts["candidate_attempts"] += selected.attempts
        self._counts["option_probes"] += selected.bound_option_probes
        self._counts["threshold_evaluations"] += selected.bound_threshold_evaluations
        for name in _EXTRA_COUNTERS:
            self._counts[name] += getattr(selected, name, 0)

    def support(self, queues: Sequence[int]) -> BudgetedFrameSelection:
        """Compute only v4 when this queue is first encountered."""
        queue = tuple(queues)
        if queue not in self._support:
            selected = select_budgeted_frame(
                self.model.options, queue, self.model.lanes, self.budget,
            )
            self._support[queue] = selected
            self._record(selected, "support")
        return self._support[queue]

    def get(self, queues: Sequence[int]) -> tuple[BudgetedFrameSelection, LookaheadFrameSelection]:
        queue = tuple(queues)
        support = self.support(queue)
        if queue not in self._lookahead:
            selected = select_lookahead_frame(
                self.model.options, queue, self.model.lanes, self.budget,
                arrival_rates=self.arrival_rates,
                terminal_service_rates=self.service_rates,
                terminal_weight=1.0, max_relative_support_loss=0.01,
            )
            self._lookahead[queue] = selected
            self._record(selected, "lookahead")
        return support, self._lookahead[queue]

    def accounting(self) -> dict[str, int]:
        return dict(self._counts)


class PrefixReplay:
    """Replay a saved prefix, replace one action, then use a single continuation."""

    def __init__(self, prefix_actions, switch_index: int, first_action,
                 continuation: Callable[[tuple[int, ...]], BudgetedFrameSelection]):
        if not 0 <= switch_index < len(prefix_actions):
            raise ValueError("switch_index must identify a saved prefix action")
        self.prefix_actions = prefix_actions
        self.switch_index = switch_index
        self.first_action = first_action
        self.continuation = continuation
        self.frames: list[dict[str, Any]] = []
        self.now = 0.0
        self.frame_index = 0

    def __call__(self, options, queues, lanes, budget):
        queue = tuple(queues)
        if self.frame_index <= self.switch_index:
            expected_queue, original = self.prefix_actions[self.frame_index]
            if queue != tuple(expected_queue):
                raise ValueError(f"replayed queue differs at frame {self.frame_index}")
            selected = self.first_action if self.frame_index == self.switch_index else original
        else:
            selected = self.continuation(queue)
        self.frames.append({
            "start_s": self.now, "duration_s": selected.duration,
            "queue_before": queue, "departures": selected.service,
            "option_indices": selected.option_indices,
        })
        self.now += selected.duration
        self.frame_index += 1
        return selected


def _completion_offsets(model, selected) -> float:
    return sum(
        offset for index in selected.option_indices if index is not None
        for offsets in model.options[index].completion_offsets for offset in offsets
    )


def expected_parts(model, queues, selected, arrival_rates, service_rates) -> dict[str, float]:
    """Return the registered current-area and fluid-terminal expectations."""
    duration = selected.duration
    current_area = ((sum(queues) - sum(selected.service)) * duration
                    + _completion_offsets(model, selected)
                    + sum(arrival_rates) * duration**2 / 2.0)
    terminal_area = sum(
        ((q - served + rate * duration)**2 + rate * duration) / (2.0 * service_rate)
        for q, served, rate, service_rate in zip(
            queues, selected.service, arrival_rates, service_rates,
        )
    )
    return {"current_area": current_area, "terminal_area": terminal_area,
            "total": current_area + terminal_area}


def realized_current_area(model, queues, selected, start_s, arrival_times,
                          arrival_time_prefix_sums) -> float:
    """Measure current-frame area using sorted tape times and [0, cumsum(times)]."""
    end = start_s + selected.duration
    left = bisect_right(arrival_times, start_s)
    right = bisect_right(arrival_times, end)
    arrival_barrier = ((right - left) * end
                       - (arrival_time_prefix_sums[right] - arrival_time_prefix_sums[left]))
    return ((sum(queues) - sum(selected.service)) * selected.duration
            + _completion_offsets(model, selected) + arrival_barrier)
