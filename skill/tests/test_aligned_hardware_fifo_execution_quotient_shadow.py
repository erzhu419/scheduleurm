"""Pre-run contracts for the registered FIFO-execution quotient shadow."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path
import shutil

import pytest

import algorithm.experiments.aligned_hardware_fifo_execution_quotient_shadow as shadow
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.fifo_execution_quotient_frame_candidates import (
    fifo_execution_signature,
    select_fifo_execution_quotient_frame,
)


def _forbid_state_execution(patcher: pytest.MonkeyPatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            pytest.fail(f"pre-run contract unexpectedly called {name}")

        return fail

    patcher.setattr(
        shadow,
        "select_fifo_execution_quotient_frame",
        forbidden("FIFO-execution selector"),
    )
    patcher.setattr(
        shadow,
        "_evaluate_state",
        forbidden("registered-state evaluation"),
    )
    return calls


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    """Register once while recording every source file opened by read_text."""
    directory = tmp_path_factory.mktemp("fifo_execution_quotient_registration")
    output = directory / "shadow.json"
    patcher = pytest.MonkeyPatch()
    calls = _forbid_state_execution(patcher)
    reads = []
    original_read_text = Path.read_text

    def tracked_read_text(path, *args, **kwargs):
        reads.append(Path(path))
        return original_read_text(path, *args, **kwargs)

    patcher.setattr(Path, "read_text", tracked_read_text)
    try:
        protocol = shadow.register(output)
    finally:
        patcher.undo()
    assert calls == []
    return output, protocol, reads


def _copy_registered_bundle(source_output: Path, destination: Path) -> Path:
    target_output = destination / source_output.name
    for source, target in zip(
        shadow._paths(source_output)[1:],
        shadow._paths(target_output)[1:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    return target_output


def test_registration_freezes_100_to_71_without_selector_or_roster_copy(
    registered_bundle,
):
    output, protocol, reads = registered_bundle
    result_path, protocol_path, sources_path = shadow._paths(output)
    copied_roster = output.with_name(output.stem + "_registered_states.json")
    model = load_model()
    raw_actions = tuple(
        product((None, *range(len(model.options))), repeat=model.lanes)
    )
    signatures = {
        fifo_execution_signature(
            model.options, raw, model.lanes, len(model.workloads)
        )
        for raw in raw_actions
    }

    assert result_path == output and not result_path.exists()
    assert protocol_path.exists() and sources_path.exists()
    assert not copied_roster.exists()
    assert "states" not in protocol
    assert protocol["registered_state_roster"] == shadow.REGISTERED_STATES.name
    assert protocol["state_count"] == shadow.STATE_COUNT == 5003
    assert len(protocol["source_population_cells"]) == 18
    assert len(raw_actions) == shadow.ORDERED_ACTION_COUNT == 100
    assert len(signatures) == shadow.FIFO_SIGNATURE_COUNT == 71
    assert protocol["fifo_execution_structure"] == {
        "option_count": 9,
        "lanes": 2,
        "represented_raw_action_count": 100,
        "represented_fifo_execution_signature_count": 71,
        "small_checkpoint_budget": 26,
        "exact_fifo_execution_signature_budget": 71,
    }
    assert protocol["signature_schema"] == shadow.SIGNATURE_SCHEMA
    assert protocol["evidence_partition"] == list(shadow.EVIDENCE_PARTITION)

    read_names = [path.name for path in reads]
    outcome_name = shadow.KNOWN_TRANSITION_PROTOCOL.name.replace(
        "_protocol.json", ".json"
    )
    assert shadow.KNOWN_TRANSITION_PROTOCOL.name in read_names
    assert outcome_name not in read_names
    assert outcome_name not in Path(shadow.__file__).read_text()


def test_registered_known_action_and_signature_roster_is_exact(registered_bundle):
    _, protocol, _ = registered_bundle
    model = load_model()
    source = json.loads(shadow.KNOWN_TRANSITION_PROTOCOL.read_text())
    pairs = protocol["known_protocol_action_pairs"]

    assert len(pairs) == 16
    cases = source["case_roster"]
    assert [row["state_index"] for row in cases] == sorted(
        {row["state_index"] for row in cases}
    )
    for case, pair in zip(cases, pairs, strict=True):
        baseline, _ = shadow._action_signatures(
            model, case["legacy_option_indices"]
        )
        variant, _ = shadow._action_signatures(
            model, case["variant_option_indices"]
        )
        assert pair == {
            "pair_id": f"quotient_case_{case['state_index']}",
            "source_kind": "registered_quotient_transition_case",
            "state_index": case["state_index"],
            "queue": case["queue"],
            "baseline": baseline,
            "variant": variant,
        }

    fixture_source = source["fifo_pairing_fixture"]
    fixture = protocol["known_fifo_pairing_fixture"]
    assert fixture["queue"] == fixture_source["queue"]
    assert fixture["forward"]["raw"] == fixture_source[
        "forward_raw_option_indices"
    ]
    assert fixture["reverse"]["raw"] == fixture_source[
        "reverse_raw_option_indices"
    ]
    assert fixture["raw_tuples_are_reversals"]
    assert fixture["physical_signatures_equal"]
    assert fixture["fifo_execution_signatures_differ"]
    assert fixture["forward"]["physical"] == fixture["reverse"]["physical"]
    assert (
        fixture["forward"]["fifo_execution"]
        != fixture["reverse"]["fifo_execution"]
    )


@pytest.mark.parametrize(
    ("raw_changed", "fifo_changed", "physical_changed", "known_fifo", "expected"),
    (
        (False, False, False, False, "RAW_UNCHANGED"),
        (True, False, False, False, "RAW_CHANGED_FIFO_EQUIVALENT"),
        (
            True,
            True,
            False,
            True,
            "FIFO_CHANGED_PHYSICAL_EQUIVALENT_KNOWN_PROTOCOL",
        ),
        (
            True,
            True,
            False,
            False,
            "FIFO_CHANGED_PHYSICAL_EQUIVALENT_NOVEL",
        ),
        (True, True, True, True, "PHYSICAL_CHANGED_KNOWN_PROTOCOL"),
        (True, True, True, False, "PHYSICAL_CHANGED_NOVEL"),
    ),
)
def test_evidence_partition_has_six_exclusive_branches(
    raw_changed, fifo_changed, physical_changed, known_fifo, expected,
):
    assert shadow._evidence_partition(
        raw_changed, fifo_changed, physical_changed, known_fifo
    ) == expected


def _decision_checks(*, execution=True, support=True, work=True, nonworse=True):
    return {
        "execution_passed": execution,
        "pointwise_support_nonworse": nonworse,
        "at_least_one_primary_load_strictly_improves_support": support,
        "at_least_one_primary_load_strictly_improves_certificate_or_attempts": work,
    }


def _decision_evaluations(*, fifo_changed):
    return [{
        "fifo_execution_signature_changed_from_legacy_k26": fifo_changed,
    }]


def test_decision_has_invalid_advance_efficiency_and_close_branches():
    assert shadow._decision(
        _decision_checks(execution=False),
        _decision_evaluations(fifo_changed=True),
    ) == "INVALID_EXECUTION"
    assert shadow._decision(
        _decision_checks(),
        _decision_evaluations(fifo_changed=True),
    ) == "ADVANCE_TO_EVIDENCE_REUSE_OR_AUTHORED_TRANSITION"
    assert shadow._decision(
        _decision_checks(support=False),
        _decision_evaluations(fifo_changed=False),
    ) == "EFFICIENCY_ONLY_STOP"
    assert shadow._decision(
        _decision_checks(nonworse=False),
        _decision_evaluations(fifo_changed=True),
    ) == "CLOSE_WITHOUT_TRANSITION_SCREEN"


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def test_synthetic_selector_work_identities_hold_and_detect_misaccounting():
    options = (
        _Option((1, 0), 1.0, ((0.5,), ())),
        _Option((1, 0), 1.0, ((0.5,), ())),
        _Option((1, 2), 1.0, ((0.25,), (0.5, 1.0))),
    )
    selected = select_fifo_execution_quotient_frame(
        options, (2, 1), lanes=2, budget=5
    )

    assert selected.cache_hits > 0
    assert shadow._work_identities(selected, len(options), 2)
    assert not shadow._work_identities(
        replace(selected, paid_search_pops=selected.paid_search_pops + 1),
        len(options),
        2,
    )


@pytest.mark.parametrize("tamper", ("protocol", "source"))
def test_tampering_fails_before_selector_or_state_evaluation(
    registered_bundle, tmp_path, monkeypatch, tamper,
):
    source_output, _, _ = registered_bundle
    output = _copy_registered_bundle(source_output, tmp_path)
    _, protocol_path, sources_path = shadow._paths(output)
    if tamper == "protocol":
        payload = json.loads(protocol_path.read_text())
        payload["exact_fifo_budget"] += 1
        protocol_path.write_text(json.dumps(payload))
        message = "registered FIFO-execution roster or constants changed"
    else:
        payload = json.loads(sources_path.read_text())
        payload[shadow.NEW_SOURCES[0]] += "\n# tampered\n"
        sources_path.write_text(json.dumps(payload))
        message = "registered FIFO-execution sources changed"

    calls = _forbid_state_execution(monkeypatch)
    with pytest.raises(ValueError, match=message):
        shadow.run_registered(output)
    assert calls == []
    assert not output.exists()


def test_write_new_and_registration_refuse_overwrite(registered_bundle, tmp_path):
    path = tmp_path / "sentinel.json"
    path.write_text('{"preserve": true}\n')
    with pytest.raises(FileExistsError):
        shadow._write_new(path, {"preserve": False})
    assert json.loads(path.read_text()) == {"preserve": True}

    output, _, _ = registered_bundle
    with pytest.raises(
        ValueError, match="FIFO-execution-quotient shadow artifacts already exist"
    ):
        shadow.register(output)
