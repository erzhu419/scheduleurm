from __future__ import annotations

from skill.scheduler_node import inventory as scheduler_nodes


def test_node_canonicalization_module_deduplicates_aliases():
    assert scheduler_nodes.canonical_node_name("node007-direct") == "node007"
    assert scheduler_nodes.canonicalize_node_list(
        ["node007-direct", "node007", "node001", ""]
    ) == ["node007", "node001"]


def test_explicit_allowed_nodes_are_not_expanded_to_node007(sch):
    task = {
        "id": "t-smoke",
        "status": "queued",
        "allowed_nodes": ["local", "jtl110gpu", "node007"],
        "allowed_nodes_user_explicit": True,
        "allowed_nodes_submitted": ["local", "jtl110gpu"],
        "allowed_nodes_expanded_reason": (
            "legacy GPU allowed_nodes omitted node007 after node007 unification"
        ),
    }
    state = {"tasks": [task]}

    assert sch._canonicalize_state_node_names(state) is True

    assert task["allowed_nodes"] == ["local", "jtl110gpu"]
    assert task["allowed_nodes_submitted"] == ["local", "jtl110gpu"]
    assert "allowed_nodes_expanded_reason" not in task


def test_explicit_node007_alias_is_canonicalized_not_duplicated(sch):
    task = {
        "id": "t-alias",
        "status": "queued",
        "allowed_nodes": ["node007-direct", "node007"],
        "allowed_nodes_user_explicit": True,
        "allowed_nodes_submitted": ["node007-direct"],
    }
    state = {"tasks": [task]}

    assert sch._canonicalize_state_node_names(state) is True

    assert task["allowed_nodes"] == ["node007"]
    assert task["allowed_nodes_submitted"] == ["node007"]


def test_state_node_fields_and_locations_are_canonicalized(sch):
    task = {
        "id": "t-locs",
        "status": "queued",
        "node": "node007-direct",
        "preferred_node": "node007-direct",
        "resume_preferred_nodes": ["node007-direct", "node007"],
        "blocked_nodes": ["node007-direct", "node001"],
        "resume_locations": [
            {"node": "node007-direct", "path": "/tmp/a"},
            {"node": "node002", "path": "/tmp/b"},
        ],
        "checkpoint_locations": [
            {"node": "node007-direct", "path": "/tmp/c"},
            "not-a-location",
        ],
    }
    state = {"tasks": [task]}

    assert sch._canonicalize_state_node_names(state) is True

    assert task["node"] == "node007"
    assert task["preferred_node"] == "node007"
    assert task["resume_preferred_nodes"] == ["node007"]
    assert task["blocked_nodes"] == ["node007", "node001"]
    assert task["resume_locations"][0]["node"] == "node007"
    assert task["resume_locations"][1]["node"] == "node002"
    assert task["checkpoint_locations"][0]["node"] == "node007"
