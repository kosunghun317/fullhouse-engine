"""Unit tests for mock-vs-mock league evaluation helpers."""

from __future__ import annotations

import json

from tools.evaluate_mock_league import build_specs, portal_pool, summarize_bot_results


def test_build_specs_supports_heads_up_and_sixmax():
    pool = {f"bot_{idx}": f"path_{idx}" for idx in range(7)}

    sixmax = build_specs(pool, "sixmax", table_size=6, max_tables=3)
    heads_up = build_specs(pool, "heads-up", max_pairs=4)

    assert len(sixmax) == 3
    assert all(len(spec["bots"]) == 6 for spec in sixmax)
    assert len(heads_up) == 4
    assert all(len(spec["bots"]) == 2 for spec in heads_up)


def test_portal_pool_loads_manifest_profiles(tmp_path):
    bot_dir = tmp_path / "bot"
    bot_dir.mkdir()
    manifest = {
        "profiles": [
            {"profile": "tight_overfolder", "bot_id": "portal_tight", "path": str(bot_dir)},
            {"profile": "sticky_station", "bot_id": "portal_station", "path": str(bot_dir)},
        ]
    }
    mocks_dir = tmp_path / "mocks"
    mocks_dir.mkdir()
    (mocks_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert portal_pool(mocks_dir, ["sticky_station"]) == {"portal_station": str(bot_dir)}


def test_summarize_bot_results_ranks_by_risk_adjusted_score():
    results = [
        {
            "chip_delta": {"steady": 1000, "swingy": -10000},
            "final_stacks": {"steady": 11000, "swingy": 0},
            "bot_errors": {"steady": [], "swingy": []},
        },
        {
            "chip_delta": {"steady": 1500, "swingy": 9000},
            "final_stacks": {"steady": 11500, "swingy": 19000},
            "bot_errors": {"steady": [], "swingy": []},
        },
    ]

    ranking = summarize_bot_results(results)

    assert ranking[0]["bot_id"] == "steady"
    assert ranking[1]["bust_count"] == 1
