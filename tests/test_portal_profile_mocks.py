import json

from tools.evaluate_heuristic import register_portal_profile_suites, SUITES
from tools.portal.mock_generation import build_match_specs, materialize_profile_mocks


def _summary():
    return {
        "profiles": [
            {
                "profile": "tight_overfolder",
                "bot_count": 10,
                "top_official_rank": 3,
                "mock_config": {"profile": "tight_overfolder", "vpip": 0.17, "pfr": 0.13},
            },
            {
                "profile": "sticky_station",
                "bot_count": 7,
                "top_official_rank": 4,
                "mock_config": {"profile": "sticky_station", "vpip": 0.32, "pfr": 0.20},
            },
        ]
    }


def test_materializes_profile_mocks(tmp_path):
    template = tmp_path / "bot.py"
    template.write_text("def decide(state):\n    return {'action': 'check'}\n", encoding="utf-8")

    manifest = materialize_profile_mocks(_summary(), tmp_path / "mocks", template=template)

    assert manifest["profile_count"] == 2
    bot_dir = tmp_path / "mocks" / "tight_overfolder"
    assert (bot_dir / "bot.py").exists()
    profile = json.loads((bot_dir / "data" / "profile.json").read_text(encoding="utf-8"))
    assert profile["mock_config"]["profile"] == "tight_overfolder"
    assert json.loads((tmp_path / "mocks" / "manifest.json").read_text(encoding="utf-8"))["profile_count"] == 2


def test_builds_sixmax_and_heads_up_match_specs(tmp_path):
    manifest = {
        "profiles": [
            {"profile": f"profile_{idx}", "bot_id": f"portal_{idx}", "path": str(tmp_path / str(idx))}
            for idx in range(6)
        ]
    }

    sixmax = build_match_specs(manifest, candidate="candidate.py", mode="sixmax")
    heads_up = build_match_specs(manifest, candidate="candidate.py", mode="heads-up")

    assert len(sixmax) == 2
    assert len(sixmax[0]["bots"]) == 6
    assert len(heads_up) == 6
    assert set(heads_up[0]["bots"]) == {"candidate", "portal_0"}


def test_registers_profile_mocks_as_heuristic_suites(tmp_path):
    template = tmp_path / "bot.py"
    template.write_text("def decide(state):\n    return {'action': 'check'}\n", encoding="utf-8")
    manifest = materialize_profile_mocks(_summary(), tmp_path / "mocks", template=template)

    names = register_portal_profile_suites(tmp_path / "mocks", hands=33)

    assert names == ["portal_profiles_1"]
    assert SUITES["portal_profiles_1"]["hands"] == 33
    assert SUITES["portal_profiles_1"]["bots"]["heuristic"] == "bots/heuristic/bot.py"
    assert set(SUITES["portal_profiles_1"]["bots"]) == {
        "heuristic",
        manifest["profiles"][0]["bot_id"],
        manifest["profiles"][1]["bot_id"],
    }
