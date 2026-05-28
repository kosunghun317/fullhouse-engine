import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOT_PATH = ROOT / "bots" / "heuristic" / "bot.py"


def load_bot():
    spec = importlib.util.spec_from_file_location("heuristic_bot_under_test", BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OPPONENTS.clear()
    module.SEEN_ACTIONS.clear()
    module.EQUITY_CACHE.clear()
    return module


def add_stats(bot, bot_id, *, actions, raises=0, calls=0, folds=0, checks=0, all_ins=0):
    stats = bot._default_stats()
    stats.update({
        "actions": actions,
        "raises": raises,
        "calls": calls,
        "folds": folds,
        "checks": checks,
        "all_ins": all_ins,
        "raise_total": raises * 300,
        "raise_count": raises,
    })
    bot.OPPONENTS[bot_id] = stats


def base_state(can_check=True):
    return {
        "type": "action_request",
        "hand_id": "test-hand",
        "street": "flop",
        "seat_to_act": 0,
        "pot": 1000,
        "community_cards": ["Ah", "7d", "2c"],
        "current_bet": 0 if can_check else 700,
        "min_raise_to": 1400,
        "amount_owed": 0 if can_check else 700,
        "can_check": can_check,
        "your_cards": ["Ks", "Kh"],
        "your_stack": 9000,
        "your_bet_this_street": 0,
        "players": [
            {"seat": 0, "bot_id": "hero", "stack": 9000, "is_folded": False},
            {"seat": 1, "bot_id": "station", "stack": 9000, "is_folded": False},
            {"seat": 2, "bot_id": "nit", "stack": 9000, "is_folded": False},
        ],
        "action_log": [],
    }


def test_betting_profile_prioritizes_visible_station_in_mixed_table():
    bot = load_bot()
    add_stats(bot, "station", actions=40, raises=1, calls=25, folds=4, checks=10)
    add_stats(bot, "nit", actions=40, raises=2, calls=4, folds=25, checks=9)

    assert bot._profile_for("station") == "station"
    assert bot._profile_for("nit") == "nit"
    assert bot._betting_profile(base_state(can_check=True)) == "station"


def test_facing_bet_uses_last_aggressor_profile_instead_of_table_profile():
    bot = load_bot()
    add_stats(bot, "station", actions=40, raises=1, calls=25, folds=4, checks=10)
    add_stats(bot, "nit", actions=40, raises=2, calls=4, folds=25, checks=9)
    state = base_state(can_check=False)
    state["action_log"] = [{"seat": 2, "action": "raise", "amount": 700}]

    assert bot._decision_profile(state) == "nit"
