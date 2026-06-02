import json

from tools.portal.profiles import assign_profile, build_profiles


def test_assigns_distinct_portal_profiles():
    base = {"hands_alive": 1000, "vpip": 0.24, "pfr": 0.16}

    assert assign_profile({**base, "all_in_rate": 0.02}) == "large_size_jammer"
    assert assign_profile({
        **base,
        "vpip": 0.34,
        "pfr": 0.12,
        "call_rate": 0.19,
        "pressure_fold_rate": 0.78,
    }) == "loose_passive_overfolder"
    assert assign_profile({
        **base,
        "vpip": 0.31,
        "pfr": 0.26,
        "raise_rate": 0.25,
        "pressure_fold_rate": 0.76,
        "showdown_rate": 0.08,
    }) == "pressure_overfolder"
    assert assign_profile({
        **base,
        "vpip": 0.18,
        "pfr": 0.12,
        "pressure_fold_rate": 0.82,
    }) == "tight_overfolder"
    assert assign_profile({**base, "call_rate": 0.22}) == "sticky_station"


def test_build_profiles_from_existing_report_rows(tmp_path):
    report = {
        "directory": "sample",
        "bot_count": 3,
        "errors": {},
        "bots": [
            {
                "bot_id": "jam",
                "bot_name": "Jammer",
                "hands_alive": 1200,
                "official_rank": 1,
                "official_delta": 1000,
                "all_in_rate": 0.03,
                "vpip": 0.28,
                "pfr": 0.2,
                "raise_rate": 0.2,
                "call_rate": 0.1,
                "pressure_fold_rate": 0.5,
                "showdown_rate": 0.07,
                "avg_raise_to_pot": 6.0,
            },
            {
                "bot_id": "station",
                "bot_name": "Station",
                "hands_alive": 900,
                "official_rank": 2,
                "official_delta": 500,
                "vpip": 0.29,
                "pfr": 0.1,
                "raise_rate": 0.08,
                "call_rate": 0.24,
                "pressure_fold_rate": 0.4,
                "pressure_call_rate": 0.36,
                "showdown_rate": 0.16,
            },
            {
                "bot_id": "thin",
                "bot_name": "TooFewHands",
                "hands_alive": 10,
                "vpip": 0.6,
            },
        ],
    }
    report_path = tmp_path / "strategy_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    summary = build_profiles(tmp_path, report_path, min_hands=100)

    assert summary["input_bot_count"] == 3
    assert summary["profiled_bot_count"] == 2
    profiles = {row["profile"]: row for row in summary["profiles"]}
    assert set(profiles) == {"large_size_jammer", "sticky_station"}
    assert profiles["large_size_jammer"]["mock_config"]["profile"] == "large_size_jammer"
    assert summary["bots"][0]["bot_name"] == "Jammer"
