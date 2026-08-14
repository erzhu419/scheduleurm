from __future__ import annotations

import json

from skill.scheduler_probe import cache as probe_cache


def test_save_node_probe_cache_skips_empty_and_writes_timestamped_states(tmp_path):
    cache_file = tmp_path / "node_probe_cache.json"

    probe_cache.save_node_probe_cache(cache_file, {}, now_fn=lambda: 10.0)
    assert not cache_file.exists()

    probe_cache.save_node_probe_cache(
        cache_file,
        {"node001": {"alive": True}},
        now_fn=lambda: 123.0,
    )
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data == {"ts": 123.0, "states": {"node001": {"alive": True}}}


def test_load_node_probe_cache_adds_fallback_metadata_and_skips_bad_rows(tmp_path):
    cache_file = tmp_path / "node_probe_cache.json"
    cache_file.write_text(
        json.dumps({
            "ts": 100.25,
            "states": {
                "node001": {"alive": True, "ram_mb": 1024},
                "node002": "bad",
            },
        }),
        encoding="utf-8",
    )

    states = probe_cache.load_node_probe_cache(
        cache_file,
        default_ttl_s=60,
        now_fn=lambda: 120.75,
    )

    assert set(states) == {"node001"}
    assert states["node001"]["alive"] is True
    assert states["node001"]["probe_fallback"] == "stale_node_probe_cache"
    assert states["node001"]["probe_cache_age_s"] == 20


def test_load_node_probe_cache_respects_ttl_override_future_and_disabled(tmp_path):
    cache_file = tmp_path / "node_probe_cache.json"
    cache_file.write_text(
        json.dumps({"ts": 100.0, "states": {"node001": {"alive": True}}}),
        encoding="utf-8",
    )

    assert probe_cache.load_node_probe_cache(
        cache_file,
        default_ttl_s=10,
        now_fn=lambda: 111.0,
    ) == {}
    assert probe_cache.load_node_probe_cache(
        cache_file,
        default_ttl_s=10,
        max_age_s=20,
        now_fn=lambda: 111.0,
    )["node001"]["probe_cache_age_s"] == 11
    assert probe_cache.load_node_probe_cache(
        cache_file,
        default_ttl_s=0,
        now_fn=lambda: 101.0,
    ) == {}
    assert probe_cache.load_node_probe_cache(
        cache_file,
        default_ttl_s=20,
        now_fn=lambda: 99.0,
    ) == {}


def test_load_node_probe_cache_returns_empty_for_corrupt_or_non_dict_states(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    assert probe_cache.load_node_probe_cache(
        corrupt,
        default_ttl_s=20,
        now_fn=lambda: 100.0,
    ) == {}

    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text(json.dumps({"ts": 100.0, "states": ["node001"]}), encoding="utf-8")
    assert probe_cache.load_node_probe_cache(
        non_dict,
        default_ttl_s=20,
        now_fn=lambda: 101.0,
    ) == {}


def test_scheduler_node_probe_cache_wrappers_use_configured_state_file(monkeypatch, sch):
    monkeypatch.setattr(sch, "NODE_PROBE_CACHE_TTL_S", 60, raising=False)

    sch._save_node_probe_cache({"node001": {"alive": True}})
    states = sch._load_node_probe_cache()

    assert states["node001"]["alive"] is True
    assert states["node001"]["probe_fallback"] == "stale_node_probe_cache"
    assert states["node001"]["probe_cache_age_s"] >= 0
