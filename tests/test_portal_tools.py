from collections import Counter, defaultdict

from tools.portal.analysis import analyze_match_payload, row_from_counts
from tools.portal.client import build_url, normalize_page_size
from tools.portal.download import (
    select_bot_ids,
    select_leaderboard_tournament,
    select_source_matches,
    source_tournament_ids,
)


def test_build_url_preserves_postgrest_in_filter():
    url = build_url(
        "hands",
        {
            "select": "id,match_id,hand_winners(bot_id,amount)",
            "match_id": "in.(m1,m2)",
            "limit": 1000,
        },
        supabase_url="https://example.supabase.co",
    )

    assert url.startswith("https://example.supabase.co/rest/v1/hands?")
    assert "hand_winners(bot_id,amount)" in url
    assert "match_id=in.(m1,m2)" in url
    assert normalize_page_size(5000) == 1000


def test_selects_ranked_bots_and_complete_source_matches():
    tournaments = [
        {"id": "official", "name": "Fullhouse 2026 Qualifier"},
        {"id": "source-a", "name": "INTERNAL-Qualifier-A"},
        {"id": "demo", "name": "Demo"},
    ]
    leaderboard = [
        {"tournament_id": "official", "bot_id": "bot-a", "rank": 1},
        {"tournament_id": "official", "bot_id": "bot-b", "rank": 2},
        {"tournament_id": "source-a", "bot_id": "bot-c", "rank": 1},
    ]
    bots = [
        {"id": "bot-a", "bot_name": "Alpha"},
        {"id": "bot-b", "bot_name": "Beta"},
        {"id": "bot-c", "bot_name": "Gamma"},
    ]
    matches = [
        {"id": "m1", "tournament_id": "source-a", "status": "complete", "round": 1, "table_index": 0},
        {"id": "m2", "tournament_id": "source-a", "status": "failed", "round": 1, "table_index": 1},
        {"id": "m3", "tournament_id": "demo", "status": "complete", "round": 1, "table_index": 2},
    ]
    match_bots = [
        {"match_id": "m1", "bot_id": "bot-a"},
        {"match_id": "m1", "bot_id": "bot-c"},
        {"match_id": "m2", "bot_id": "bot-b"},
        {"match_id": "m3", "bot_id": "bot-a"},
    ]

    official = select_leaderboard_tournament(tournaments, "Fullhouse 2026 Qualifier")
    sources = source_tournament_ids(tournaments, official["id"], "Qualifier")
    selected_bots = select_bot_ids(leaderboard, bots, official["id"], None, 1, [], [])
    selected_matches = select_source_matches(matches, match_bots, sources, selected_bots, [])
    all_matches = select_source_matches(matches, match_bots, sources, set(), [], all_complete_source_matches=True)

    assert selected_bots == {"bot-a"}
    assert [m["id"] for m in selected_matches] == ["m1"]
    assert [m["id"] for m in all_matches] == ["m1"]


def test_analyzes_fixed_seat_action_log():
    payload = {
        "bots": [
            {"seat": 0, "bot_id": "aggressor", "chip_delta": 450},
            {"seat": 1, "bot_id": "folder", "chip_delta": -450},
        ],
        "hands": [
            {
                "hand_num": 0,
                "action_log": [
                    {"seat": 0, "action": "small_blind", "amount": 50},
                    {"seat": 1, "action": "big_blind", "amount": 100},
                    {"seat": 0, "action": "raise", "amount": 300},
                    {"seat": 1, "action": "fold", "amount": 0},
                ],
                "revealed_cards": {},
                "hand_winners": [{"bot_id": "aggressor", "amount": 400}],
            }
        ],
    }
    stats = defaultdict(Counter)
    match_results = defaultdict(Counter)

    analyze_match_payload(payload, stats, match_results)
    row = row_from_counts(
        "aggressor",
        stats["aggressor"],
        {"aggressor": {"bot_name": "Aggressor"}},
        {"aggressor": {"rank": 1, "cumulative_delta": 450, "matches_played": 1}},
        match_results,
    )

    assert dict(stats.get("_errors", {})) == {}
    assert stats["aggressor"]["action_raise"] == 1
    assert stats["folder"]["pressure_fold"] == 1
    assert row["vpip"] == 1.0
    assert row["pfr"] == 1.0
    assert row["sample_delta"] == 450
