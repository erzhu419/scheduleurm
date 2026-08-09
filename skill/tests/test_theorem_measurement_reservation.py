from __future__ import annotations

import json

import pytest

from algorithm.experiments.theorem_measurement_reservation import (
    MeasurementReservation,
    ReservationConflict,
    acquire_reservation,
    fold_measurement_reservations_into_probe,
    list_active_reservations,
    release_reservation,
    renew_reservation,
)


def test_reservation_annotates_probe_without_fabricating_resource_pressure(tmp_path):
    path = tmp_path / "reservations.json"
    record = acquire_reservation(
        node="node007",
        purpose="loaded-state theorem wave",
        ttl_s=60,
        path=path,
    )
    nodes = [{
        "name": "node007",
        "alive": True,
        "free_cpu": 48,
        "free_ram_mb": 90000,
        "gpus": [{"idx": 0, "used_mb": 17, "free_mb": 10983, "total_mb": 11000}],
    }]
    folded = fold_measurement_reservations_into_probe(nodes, path=path)
    node = folded[0]
    gpu = node["gpus"][0]
    assert node["free_cpu"] == 48
    assert node["free_ram_mb"] == 90000
    assert gpu["used_mb"] == 17
    assert gpu["free_mb"] == 10983
    assert gpu["measurement_reservation_active"] is True
    assert node["measurement_reservation"]["reservation_id"] == record["reservation_id"]
    assert release_reservation(record["reservation_id"], path=path)
    assert list_active_reservations(path=path) == []


def test_live_reservation_conflicts_and_renews(tmp_path):
    path = tmp_path / "reservations.json"
    record = acquire_reservation(node="jtl110gpu", purpose="wave", ttl_s=30, path=path)
    with pytest.raises(ReservationConflict):
        acquire_reservation(node="jtl110gpu", purpose="other", ttl_s=30, path=path)
    renewed = renew_reservation(record["reservation_id"], ttl_s=90, path=path)
    assert renewed["expires_at"] > record["expires_at"]


def test_expired_or_dead_holder_is_ignored(tmp_path):
    path = tmp_path / "reservations.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "reservations": [{
            "reservation_id": "old",
            "node": "jtl311linux",
            "purpose": "old wave",
            "holder_host": "",
            "holder_pid": 99999999,
            "holder_start_ticks": 1,
            "expires_at": 10**20,
        }],
    }))
    assert list_active_reservations(path=path) == []
    fresh = acquire_reservation(node="jtl311linux", purpose="fresh wave", path=path)
    assert fresh["reservation_id"] != "old"


def test_context_manager_releases_on_exception(tmp_path):
    path = tmp_path / "reservations.json"
    with pytest.raises(RuntimeError):
        with MeasurementReservation(node="jtl110gpu2", purpose="wave", path=path):
            raise RuntimeError("controlled failure")
    assert list_active_reservations(path=path) == []
