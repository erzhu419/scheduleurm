"""Audit actual routed replay calls without constructing extra native ladders."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import (
    COUNTERS, _metrics, _record, check_selection, reference_key,
)


def selectors(protocol):
    """Bind the unchanged whole-controller budgets to the replay interface."""
    from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
    from algorithm.direct_probe_own_tight_native_frame_candidates import select_direct_probe_own_tight_native_frame
    from algorithm.small_dimension_exact_native_frame_candidates import select_small_dimension_exact_native_frame

    def bind(select):
        def invoke(options, queue, lanes, budget):
            if budget != protocol["lp_node_budget"]:
                raise ValueError("replay whole-controller budget differs from registration")
            return select(options, queue, lanes, protocol["initial_lp_budget"],
                          protocol["lp_node_budget"], protocol["integer_state_budget"])
        return invoke

    return {"candidate": bind(select_small_dimension_exact_native_frame),
            "direct": bind(select_direct_probe_own_tight_native_frame),
            "full_reference": select_canonical_count_space_bnb_frame}


class ReplayReferenceAudit:
    """Share only offline references; observed policy solves remain policy costs."""

    def __init__(self, model, protocol):
        self.model, self.protocol = model, protocol
        self.by_queue = {}
        self.lookup_calls = self.cache_hits = self.observed_policy_calls = 0
        self.metrics = _metrics()
        self.oracle_calls = 0

    def _store(self, queue, selected):
        from algorithm.experiments.integer_capacity_canonical_oracle import (
            canonical_at_certified_score, certified_score_from_selection,
        )
        from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import oracle_key
        from algorithm.experiments.score_face_completion_confirmation_summary import _reference

        queue = tuple(queue)
        payload, expected = asdict(selected), reference_key(selected, queue, len(self.model.options))
        oracle = None
        if selected.score_exact and not selected.canonical_exact:
            score = certified_score_from_selection(self.model.options, queue, self.model.lanes, payload)
            oracle = canonical_at_certified_score(self.model.options, queue, self.model.lanes, score,
                state_visit_budget=self.protocol["oracle_state_budget"], maximum_duration=selected.duration)
            self.oracle_calls += 1
            expected = oracle_key(oracle, len(self.model.options))
        record = {"queue": queue, "reference": payload, "oracle": oracle,
            "reference_complete": reference_key(selected, queue, len(self.model.options)) is not None,
            "effective_reference_complete": expected is not None,
            "effective_reference_source": ("full_K4096" if oracle is None and expected is not None else
                                           "independent_score_face_oracle" if expected is not None else None)}
        checked, errors = _reference(record, self.protocol, len(self.model.options))
        checks = check_selection(selected, self.model.options, queue, self.model.lanes,
                                 self.protocol["reference_budget"], checked)
        errors.extend("reference: " + key for key, failed in checks.items() if failed)
        self.by_queue[queue] = {**record, "expected": checked, "errors": errors}
        return self.by_queue[queue]

    def observe(self, queue, selected):
        if selected.lp_node_budget != self.protocol["reference_budget"]:
            raise ValueError("reference-policy call must use its registered K4096 cap")
        self.observed_policy_calls += 1
        queue = tuple(queue)
        if queue not in self.by_queue:
            return self._store(queue, selected)
        return self.by_queue[queue]

    def lookup(self, queue):
        from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame

        queue = tuple(queue)
        self.lookup_calls += 1
        if queue in self.by_queue:
            self.cache_hits += 1
            return self.by_queue[queue]
        started = perf_counter()
        selected = select_canonical_count_space_bnb_frame(
            self.model.options, queue, self.model.lanes, self.protocol["reference_budget"])
        elapsed = perf_counter() - started
        record = self._store(queue, selected)
        _record(self.metrics, selected, self.model, queue, self.model.lanes,
                self.protocol["reference_budget"], record["expected"], elapsed_s=elapsed)
        return record

    def summary(self):
        from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS

        records = list(self.by_queue.values())
        oracles = [record["oracle"] for record in records if record["oracle"] is not None]
        return {"calls": self.metrics["calls"], "actual_calls": self.metrics["calls"],
            "lookup_calls": self.lookup_calls, "cache_hits": self.cache_hits,
            "observed_policy_calls": self.observed_policy_calls, "oracle_calls": self.oracle_calls,
            "cache_size": len(records), "failures": sum(bool(record["errors"]) for record in records),
            "incomplete_count": sum(record["expected"] is None for record in records),
            "operational_elapsed_s": self.metrics["operational_elapsed_s"],
            "work": {"search_work": dict(self.metrics["search_work"]),
                     "oracle": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}}}


def _materialized_stages(result):
    stages = [("root_lp", result["root_lp"])]
    if result["settled_lp"] is not None:
        stages.append(("settled_lp", result["settled_lp"]))
    if result["initial_lp"]["selection"]["lp_node_budget"] != result["root_lp_budget"]:
        stages.append(("initial_lp", result["initial_lp"]))
    if result["resumed"]:
        stages.append(("final_lp", result["final_lp"]))
    return stages


def _controller_errors(result, queue, protocol, model, lanes, reference):
    """Check only returned native prefixes and the one actually paid probe."""
    from algorithm.experiments import small_dimension_exact_native_development_summary as exact
    from algorithm.experiments import direct_probe_own_tight_native_development_summary as direct
    from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
    from algorithm.experiments.settled_root_dual_face_checked_development_summary import audit_settled_dual_face
    from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as audit_probe, _same
    from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof
    from algorithm.experiments.affine_dual_fresh_confirmation_summary import audit_retained_completion
    from algorithm.experiments.early_root_canonical_development_summary import _catalogue, ROUTING_COUNTERS

    state = {"queue": queue, "reference": reference["reference"]}
    expected, errors = reference["expected"], []
    root, settled, initial, final, probe = (result[name] for name in (
        "root_lp", "settled_lp", "initial_lp", "final_lp", "probe"))
    roots, stages = result["root_lp_budget"], _materialized_stages(result)
    groups, thresholds = _catalogue(model)
    if roots != len(thresholds) or not _same(root["thresholds"], thresholds):
        errors.append("raw root cap or duration strata differ from the original catalogue")
    for name in ("initial_lp_budget", "lp_node_budget", "integer_state_budget"):
        if result[name] != protocol[name]:
            errors.append("whole-controller " + name + " differs from registration")
    auditor = exact if "small_dimension_lp_work" in final else direct
    previous = None
    for name, stage in stages:
        cap = stage["selection"]["lp_node_budget"]
        if cap not in (roots, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
            errors.append(name + " has an unregistered actual native cap")
        errors.extend(name + ": " + error for error in audit_reserved_stage(stage, cap, roots))
        errors.extend(name + ": " + error for error in auditor._stage_errors(
            stage, previous, model, queue, lanes, cap, expected, state))
        previous = stage
    preparation = result["routing_preparation"]
    if (set(preparation) != set(ROUTING_COUNTERS) or any(type(value) is not int or value < 0 for value in preparation.values())
            or any(preparation[name] for name in ROUTING_COUNTERS[:8])
            or preparation["root_snapshot_calls"] != 1 or preparation["root_snapshot_retained_entries_read"] != 0
            or not root["selection"]["frontier_regions_remaining"] <= preparation["root_snapshot_heap_entries_read"] <= root["root_pruning"]["roots_solved"]):
        errors.append("whole controller must retain its one genuine raw-root snapshot fee")
    if root["preprocessing"]["requirement_group_count"] != groups:
        errors.append("native grouping differs from the original catalogue")
    errors.extend(auditor._phase_errors(result, state))
    fields = [auditor.OWN_FIELDS]
    if auditor is exact:
        fields.append(exact.SMALL_FIELDS)
    for work_field, evidence_field in fields:
        copies = 0
        for name, stage in stages:
            copies += len(stage[evidence_field])
            if stage[work_field]["event_snapshot_entries_copied"] != copies:
                errors.append(name + " cumulative evidence tuple copy fee differs from actual constructed prefixes")
        if final[work_field]["event_snapshot_entries_copied"] != copies:
            errors.append("aliased final prefix must retain the last actual evidence-copy fee")
    if not _same(final, stages[-1][1]):
        errors.append("final prefix differs from the last actually constructed prefix")
    if settled is not None and not settled["selection"]["canonical_exact"]:
        errors.extend(audit_settled_dual_face(result, model, queue, lanes))
    elif result["dual_face_evidence"] is not None or any(result["dual_face_work"].values()):
        errors.append("untriggered outer dual face must export no evidence or fees")
    if probe is not None:
        baseline = {"root": settled, "initial": initial, "resumed": final, "frontier_late": final}.get(result["probe_stage"])
        if baseline is None:
            errors.append("integer probe lacks its actual native baseline")
        else:
            cap = baseline["selection"]["lp_node_budget"]
            _, extra, _ = audit_probe(probe, state, {**protocol, "lp_budgets": {"candidate": cap}},
                                     model, lanes, "candidate", expected)
            errors.extend("probe: " + error for error in extra)
            if any(not _same(probe[field], baseline[source]) for field, source in (
                    ("baseline_selection", "selection"), ("preprocessing", "preprocessing"),
                    ("capacity_propagation", "capacity_propagation"))):
                errors.append("probe must retain its actual paid native baseline")
            if result["probe_stage"] == "frontier_late":
                if result["frontier_evidence"] is None:
                    errors.append("late frontier probe lacks its returned frontier evidence")
                else:
                    errors.extend(audit_axis_capacity_proof(probe, result["frontier_evidence"], queue, lanes))
                    errors.extend(audit_retained_completion(probe, model, queue, lanes))
    elif result["probe_stage"] != "none":
        errors.append("probe stage must describe an actual paid probe")
    if result["frontier_evidence"] is not None and result["probe_stage"] != "frontier_late":
        errors.append("frontier evidence requires its actual late probe")
    selected = result["selection"]
    if root["selection"]["canonical_exact"]:
        status, source = "LP_ROOT_COMPLETE", root["selection"]
        if result["resumed"] or probe is not None or not _same(initial, root):
            errors.append("closed raw roots must return without a probe or continuation")
    elif settled is not None and settled["selection"]["canonical_exact"]:
        status, source = "LP_ROOT_SETTLE_COMPLETE", settled["selection"]
        if result["resumed"] or probe is not None or not _same(initial, settled):
            errors.append("closed settlement must return without a probe or continuation")
    elif probe is not None and probe["selection"]["canonical_exact"]:
        status, source = ("FRONTIER_PROBE_COMPLETE" if result["probe_stage"] == "frontier_late" else
                          "INTEGER_PROBE_COMPLETE"), probe["selection"]
    elif result["probe_stage"] == "frontier_late":
        status, source = "FRONTIER_PROBE_INCOMPLETE", probe["selection"]
    elif result["resumed"]:
        status, source = ("LP_RESUME_COMPLETE" if final["selection"]["canonical_exact"] else
                          "LP_RESUME_INCOMPLETE"), final["selection"]
    else:
        status, source = "LP_INITIAL_COMPLETE", initial["selection"]
    if result["status"] != status or not _same(selected, source):
        errors.append("whole-controller status or returned action differs from its paid completion path")
    if result["resumed"] and final["selection"]["lp_node_budget"] != protocol["lp_node_budget"]:
        errors.append("resumed controller must retain its actual registered final LP cap")
    return errors


def audit_trajectory(model, trajectory, selections, reference_audit, protocol):
    """Attach offline correctness and whole-call costs to every paid decision."""
    from algorithm.experiments.small_dimension_exact_native_development_summary import _work

    if len(trajectory["frames"]) != len(selections):
        raise ValueError("each paid replay decision needs its original whole selection")
    before = reference_audit.summary()
    metrics, diagnostics, records = _metrics(), [], []
    metrics["reference_incomplete_count"] = 0
    policy = trajectory["policy"]
    for number, (frame, result) in enumerate(zip(trajectory["frames"], selections, strict=True)):
        selected = getattr(result, "selection", result)
        queue = tuple(frame["queue"])
        reference = (reference_audit.observe(queue, selected) if policy == "full_reference"
                     else reference_audit.lookup(queue))
        payload = asdict(result)
        budget = (protocol["reference_budget"] if policy == "full_reference" else
                  payload["final_lp"]["selection"]["lp_node_budget"])
        errors = list(reference["errors"])
        if policy != "full_reference":
            errors.extend(_controller_errors(payload, queue, protocol, model.snapshot(), model.lanes, reference))
        previous_failures = metrics["failures"]
        _record(metrics, selected, model, queue, model.lanes, budget, reference["expected"],
                None if policy == "full_reference" else payload["final_lp"]["preprocessing"],
                elapsed_s=frame["decision_delay_s"])
        if errors and metrics["failures"] == previous_failures:
            metrics["failures"] += 1
        if errors and metrics["first_failure"] is None:
            metrics["first_failure"] = {"frame": number, "queue": queue, "errors": errors}
        metrics["reference_incomplete_count"] += reference["expected"] is None
        probe = None if policy == "full_reference" else payload["probe"]
        diagnostics.append({"frame": number, "status": "FULL_REFERENCE" if policy == "full_reference" else payload["status"],
            "probe_stage": "none" if policy == "full_reference" else payload["probe_stage"],
            "resumed": False if policy == "full_reference" else payload["resumed"],
            "root_lp_budget": None if policy == "full_reference" else payload["root_lp_budget"],
            "actual_lp_budget": budget, "lp_calls": selected.lp_calls,
            "shared_integer_states": 0 if probe is None else
                probe["score_proof"]["states_visited"] + probe["completion"]["states_visited"],
            "canonical_exact": selected.canonical_exact, "reference_complete": reference["expected"] is not None,
            "errors": errors})
        records.append(payload)
    metrics["whole_work"] = ({"calls": len(records), "search_work": dict(metrics["search_work"])} if policy == "full_reference"
                             else _work(records, policy))
    after = reference_audit.summary()
    audit = {key: after[key] - before[key] for key in (
        "calls", "actual_calls", "lookup_calls", "cache_hits", "observed_policy_calls", "oracle_calls", "failures",
        "operational_elapsed_s")}
    audit.update(incomplete_count=metrics["reference_incomplete_count"], cache_size=after["cache_size"],
        work={family: {key: value - before["work"][family][key] for key, value in counters.items()}
              for family, counters in after["work"].items()})
    trajectory.update(selector_metrics=metrics, decision_diagnostics=diagnostics, reference_audit=audit)
    return trajectory
