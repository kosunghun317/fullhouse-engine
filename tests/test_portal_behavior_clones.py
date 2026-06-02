import importlib.util
import json

from tools.portal.behavior_clone import build_behavior_clone_summary, materialize_behavior_clones
from tools.portal.mock_generation import build_match_specs


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _portal_fixture(tmp_path):
    directory = tmp_path / "portal"
    matches = directory / "matches"
    matches.mkdir(parents=True)
    _write_json(
        directory / "selected_leaderboard.json",
        [
            {"bot_id": "bot-a", "rank": 1, "cumulative_delta": 1000},
            {"bot_id": "bot-b", "rank": 80, "cumulative_delta": -1000},
        ],
    )
    _write_json(
        directory / "bots.json",
        [
            {"id": "bot-a", "bot_name": "Alpha"},
            {"id": "bot-b", "bot_name": "Bravo"},
        ],
    )
    _write_json(
        directory / "strategy_report.json",
        {
            "bots": [
                {
                    "bot_id": "bot-a",
                    "hands_alive": 250,
                    "vpip": 0.30,
                    "pfr": 0.26,
                    "raise_rate": 0.24,
                    "call_rate": 0.12,
                    "pressure_fold_rate": 0.78,
                    "showdown_rate": 0.08,
                }
            ]
        },
    )
    _write_json(
        matches / "match-1.json",
        {
            "bots": [
                {"seat": 0, "bot_id": "bot-a"},
                {"seat": 1, "bot_id": "bot-b"},
            ],
            "hands": [
                {
                    "hand_num": 1,
                    "action_log": [
                        {"seat": 0, "action": "small_blind", "amount": 50},
                        {"seat": 1, "action": "big_blind", "amount": 100},
                        {"seat": 0, "action": "call", "amount": 50},
                        {"seat": 1, "action": "check", "amount": 0},
                        {"seat": 1, "action": "raise", "amount": 300},
                        {"seat": 0, "action": "fold", "amount": 0},
                    ],
                    "hand_winners": [{"bot_id": "bot-b", "amount": 500}],
                }
            ],
        },
    )
    return directory


def test_builds_behavior_clone_summary_from_ranked_replays(tmp_path):
    directory = _portal_fixture(tmp_path)

    summary = build_behavior_clone_summary(
        directory,
        rank_lte=64,
        group_by="profile",
        min_actions=1,
        min_cell_actions=1,
        smoothing=0.0,
    )

    assert summary["selected_bot_count"] == 1
    assert summary["policy_count"] == 1
    policy = summary["policies"][0]
    assert policy["group"] == "pressure_overfolder"
    assert policy["action_count"] == 2
    assert policy["bot_names"] == ["Alpha"]
    assert policy["fallback"]["action_probs"]["call"] == 0.5
    assert policy["fallback"]["action_probs"]["fold"] == 0.5
    assert "preflop|facing|hu|medium|low|r0" in policy["cells"]
    assert "flop|facing|hu|expensive|low|r1" in policy["cells"]


def test_materializes_behavior_clone_variants_as_portal_suites(tmp_path):
    directory = _portal_fixture(tmp_path)
    summary = build_behavior_clone_summary(
        directory,
        rank_lte=64,
        group_by="profile",
        min_actions=1,
        min_cell_actions=1,
    )
    template = tmp_path / "bot.py"
    template.write_text("def decide(state):\n    return {'action': 'check'}\n", encoding="utf-8")

    manifest = materialize_behavior_clones(
        summary,
        tmp_path / "clones",
        template=template,
        variants_per_policy=2,
        variant_seed=11,
    )

    assert manifest["source_policy_count"] == 1
    assert manifest["policy_count"] == 2
    assert manifest["profile_count"] == 2
    assert (tmp_path / "clones" / "pressure_overfolder_v01" / "bot.py").exists()
    written = json.loads(
        (tmp_path / "clones" / "pressure_overfolder_v01" / "data" / "policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["variant"]["name"] == "v01"

    specs = build_match_specs(manifest, candidate="candidate.py", mode="heads-up")
    assert len(specs) == 2
    assert set(specs[0]["bots"]) == {"candidate", manifest["profiles"][0]["bot_id"]}


def test_behavior_clone_bot_uses_policy_and_returns_legal_actions(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_json(
        data_dir / "policy.json",
        {
            "policy_type": "portal_behavior_clone_v1",
            "group": "test_clone",
            "fallback": {
                "action_probs": {"fold": 0.0, "check": 0.0, "call": 0.0, "raise": 1.0, "all_in": 0.0},
                "raise_to_pot": {"count": 3, "mean": 0.8, "p50": 0.8},
            },
            "cells": {
                "flop|free|hu|free|none|r0": {
                    "action_probs": {"fold": 1.0, "check": 0.0, "call": 0.0, "raise": 0.0, "all_in": 0.0},
                    "raise_to_pot": None,
                }
            },
        },
    )
    monkeypatch.setenv("BOT_DATA_DIR", str(data_dir))
    bot_path = "bots/mock_competitors/portal_behavior_clone/bot.py"
    spec = importlib.util.spec_from_file_location("portal_behavior_clone_test_bot", bot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    facing_state = {
        "type": "action_request",
        "hand_id": "h1",
        "street": "preflop",
        "seat_to_act": 0,
        "pot": 150,
        "community_cards": [],
        "current_bet": 100,
        "min_raise_to": 200,
        "amount_owed": 100,
        "can_check": False,
        "your_cards": ["As", "Ah"],
        "your_stack": 9900,
        "your_bet_this_street": 0,
        "players": [
            {"seat": 0, "state": "active", "is_folded": False},
            {"seat": 1, "state": "active", "is_folded": False},
        ],
        "action_log": [
            {"seat": 0, "action": "small_blind", "amount": 50},
            {"seat": 1, "action": "big_blind", "amount": 100},
        ],
    }
    action = module.decide(facing_state)
    assert action["action"] == "raise"
    assert action["amount"] >= 200

    free_state = dict(facing_state)
    free_state.update(
        {
            "hand_id": "h2",
            "street": "flop",
            "pot": 300,
            "community_cards": ["2s", "7d", "Th"],
            "current_bet": 0,
            "min_raise_to": 100,
            "amount_owed": 0,
            "can_check": True,
            "your_cards": ["3c", "8h"],
            "action_log": [],
        }
    )
    assert module.decide(free_state) == {"action": "check"}
