"""Heuristic population-exploit bot for Fullhouse.

One-file submission by design. The policy is intentionally interpretable:
preflop hand classes, bounded Monte Carlo equity, public-action opponent
modeling, and a single action sanitizer at the edge.
"""

import os
import random
import time

import eval7
import numpy as np

BOT_NAME = "Heuristic Exploit"
DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

_seed = os.environ.get("HEURISTIC_RNG_SEED")
if _seed is not None and _seed != "":
    try:
        random.seed(int(_seed))
    except ValueError:
        random.seed(_seed)


def _float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _int_env(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


# ---------------------------------------------------------------------------
# Constants and tunables
# ---------------------------------------------------------------------------

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {rank: i + 2 for i, rank in enumerate(RANK_ORDER)}
SUITS = "shdc"
FULL_DECK = [eval7.Card(rank + suit) for rank in RANK_ORDER for suit in SUITS]

BIG_BLIND = 100
DECIDE_BUDGET_S = 1.65
EQUITY_BUDGET_S = 0.20
EQUITY_CACHE_MAX = 4096
SEEN_ACTIONS_MAX = 1200

CALL_MARGIN_BASE = _float_env("HEURISTIC_CALL_MARGIN_BASE", 0.075)
CALL_MARGIN_MULTIWAY = _float_env("HEURISTIC_CALL_MARGIN_MULTIWAY", 0.035)
RISK_REQ_LOW = _float_env("HEURISTIC_RISK_REQ_LOW", 0.76)
RISK_REQ_MID = _float_env("HEURISTIC_RISK_REQ_MID", 0.86)
RISK_REQ_HIGH = _float_env("HEURISTIC_RISK_REQ_HIGH", 0.92)
RISK_CUTOFF_LOW = _float_env("HEURISTIC_RISK_CUTOFF_LOW", 0.28)
RISK_CUTOFF_MID = _float_env("HEURISTIC_RISK_CUTOFF_MID", 0.45)
RISK_CUTOFF_HIGH = _float_env("HEURISTIC_RISK_CUTOFF_HIGH", 0.70)
VALUE_THRESHOLD_BASE = _float_env("HEURISTIC_VALUE_THRESHOLD_BASE", 0.66)
THIN_VALUE_BASE = _float_env("HEURISTIC_THIN_VALUE_BASE", 0.59)
DRY_BLUFF_PROB = _float_env("HEURISTIC_DRY_BLUFF_PROB", 0.45)
WET_BLUFF_PROB = _float_env("HEURISTIC_WET_BLUFF_PROB", 0.25)
NORMAL_VALUE_FRACTION = _float_env("HEURISTIC_NORMAL_VALUE_FRACTION", 0.52)
PRESSURE_VALUE_FRACTION = _float_env("HEURISTIC_PRESSURE_VALUE_FRACTION", 0.75)
THIN_VALUE_FRACTION = _float_env("HEURISTIC_THIN_VALUE_FRACTION", 0.45)
DRY_BLUFF_FRACTION = _float_env("HEURISTIC_DRY_BLUFF_FRACTION", 0.42)
WET_SEMI_BLUFF_FRACTION = _float_env("HEURISTIC_WET_SEMI_BLUFF_FRACTION", 0.55)
HIGH_EQUITY_RAISE_FRACTION = _float_env("HEURISTIC_HIGH_EQUITY_RAISE_FRACTION", 0.85)
SPR_LOW = _float_env("HEURISTIC_SPR_LOW", 1.8)
SPR_HIGH = _float_env("HEURISTIC_SPR_HIGH", 6.0)
SPR_LOW_VALUE_DISCOUNT = _float_env("HEURISTIC_SPR_LOW_VALUE_DISCOUNT", 0.035)
SPR_LOW_THIN_VALUE_DISCOUNT = _float_env("HEURISTIC_SPR_LOW_THIN_VALUE_DISCOUNT", 0.025)
SPR_LOW_CALL_MARGIN_DISCOUNT = _float_env("HEURISTIC_SPR_LOW_CALL_MARGIN_DISCOUNT", 0.025)
SPR_HIGH_CALL_MARGIN_BONUS = _float_env("HEURISTIC_SPR_HIGH_CALL_MARGIN_BONUS", 0.025)
SPR_LOW_RAISE_FRACTION_BONUS = _float_env("HEURISTIC_SPR_LOW_RAISE_FRACTION_BONUS", 0.08)
SPR_COMMIT_EQUITY = _float_env("HEURISTIC_SPR_COMMIT_EQUITY", 0.74)
OFF_BUCKET_SIZING_PROB = _float_env("HEURISTIC_OFF_BUCKET_SIZING_PROB", 0.12)
OFF_BUCKET_BLUFF_MIN = _float_env("HEURISTIC_OFF_BUCKET_BLUFF_MIN", 0.56)
OFF_BUCKET_VALUE_STATION_BONUS = _float_env("HEURISTIC_OFF_BUCKET_VALUE_STATION_BONUS", 0.08)
OFF_BUCKET_VALUE_TIGHT_MAX = _float_env("HEURISTIC_OFF_BUCKET_VALUE_TIGHT_MAX", 0.49)
PREFLOP_PREMIUM_OPEN_BB = _float_env("HEURISTIC_PREFLOP_PREMIUM_OPEN_BB", 3.4)
PREFLOP_OPEN_BB = _float_env("HEURISTIC_PREFLOP_OPEN_BB", 3.0)
PREFLOP_HU_MANIAC_OPEN_BB = _float_env("HEURISTIC_PREFLOP_HU_MANIAC_OPEN_BB", 2.8)
PREFLOP_RERAISE_MIN_BB = _float_env("HEURISTIC_PREFLOP_RERAISE_MIN_BB", 7.0)
PREFLOP_RERAISE_MULT = _float_env("HEURISTIC_PREFLOP_RERAISE_MULT", 2.2)
PREFLOP_PREMIUM_OPEN_SCORE = _float_env("HEURISTIC_PREFLOP_PREMIUM_OPEN_SCORE", 88)
PREFLOP_OPEN_SCORE = _float_env("HEURISTIC_PREFLOP_OPEN_SCORE", 62)
PREFLOP_LATE_PLAY_SCORE = _float_env("HEURISTIC_PREFLOP_LATE_PLAY_SCORE", 48)
PREFLOP_CHEAP_CALL_SCORE = _float_env("HEURISTIC_PREFLOP_CHEAP_CALL_SCORE", 48)
PREFLOP_HU_MANIAC_OPEN_SCORE = _float_env("HEURISTIC_PREFLOP_HU_MANIAC_OPEN_SCORE", 52)
PREFLOP_HU_MANIAC_CALL_STRONG = _float_env("HEURISTIC_PREFLOP_HU_MANIAC_CALL_STRONG", 64)
PREFLOP_HU_MANIAC_CALL_MEDIUM = _float_env("HEURISTIC_PREFLOP_HU_MANIAC_CALL_MEDIUM", 54)
PREFLOP_PREMIUM_CALL_SCORE = _float_env("HEURISTIC_PREFLOP_PREMIUM_CALL_SCORE", 82)
PREFLOP_LATE_RAISE_CALL_SCORE = _float_env("HEURISTIC_PREFLOP_LATE_RAISE_CALL_SCORE", 56)
EQUITY_ADJ_NIT = _float_env("HEURISTIC_EQUITY_ADJ_NIT", -0.045)
EQUITY_ADJ_ABC = _float_env("HEURISTIC_EQUITY_ADJ_ABC", -0.030)
EQUITY_ADJ_MANIAC = _float_env("HEURISTIC_EQUITY_ADJ_MANIAC", 0.025)
EQUITY_ADJ_LARGE_BET = _float_env("HEURISTIC_EQUITY_ADJ_LARGE_BET", -0.030)
EQUITY_ADJ_MULTIWAY = _float_env("HEURISTIC_EQUITY_ADJ_MULTIWAY", -0.012)
FLOP_SAMPLES = _int_env("HEURISTIC_FLOP_SAMPLES", 520)
TURN_SAMPLES = _int_env("HEURISTIC_TURN_SAMPLES", 700)
RIVER_SAMPLES = _int_env("HEURISTIC_RIVER_SAMPLES", 900)
MULTIWAY_SAMPLE_FACTOR = _float_env("HEURISTIC_MULTIWAY_SAMPLE_FACTOR", 0.75)
PROFILE_MANIAC_RAISE_RATE = _float_env("HEURISTIC_PROFILE_MANIAC_RAISE_RATE", 0.33)
PROFILE_MANIAC_ALL_IN_RATE = _float_env("HEURISTIC_PROFILE_MANIAC_ALL_IN_RATE", 0.10)
PROFILE_MANIAC_AVG_RAISE_BB = _float_env("HEURISTIC_PROFILE_MANIAC_AVG_RAISE_BB", 8.0)
PROFILE_STATION_CALL_RATE = _float_env("HEURISTIC_PROFILE_STATION_CALL_RATE", 0.42)
PROFILE_STATION_FOLD_RATE = _float_env("HEURISTIC_PROFILE_STATION_FOLD_RATE", 0.25)
PROFILE_NIT_FOLD_RATE = _float_env("HEURISTIC_PROFILE_NIT_FOLD_RATE", 0.42)
PROFILE_NIT_RAISE_RATE = _float_env("HEURISTIC_PROFILE_NIT_RAISE_RATE", 0.18)
PROFILE_PRESSURE_NIT_FOLD_RATE = _float_env("HEURISTIC_PROFILE_PRESSURE_NIT_FOLD_RATE", 0.62)
PROFILE_PRESSURE_STATION_FOLD_RATE = _float_env("HEURISTIC_PROFILE_PRESSURE_STATION_FOLD_RATE", 0.30)


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _load_tables():
    try:
        return np.load(os.path.join(DATA_DIR, "tables.npz"), allow_pickle=False)
    except Exception:
        return None


TABLES = _load_tables()

ULTRA_PREMIUM_CLASSES = {"AA", "KK"}
PREMIUM_CLASSES = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
STRONG_CLASSES = {
    "TT", "99", "88", "AQs", "AQo", "AJs", "AJo", "ATs", "KQs", "KQo",
    "KJs", "QJs",
}
SPECULATIVE_CLASSES = {
    "77", "66", "55", "44", "33", "22",
    "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KTs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s",
}


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

OPPONENTS = {}
SEEN_ACTIONS = set()
EQUITY_CACHE = {}


# ---------------------------------------------------------------------------
# Card and hand helpers
# ---------------------------------------------------------------------------

def _rank(card):
    return card[0]


def _rank_value(card):
    return RANK_VALUE.get(_rank(card), 0)


def _hand_class(cards):
    if len(cards) < 2:
        return "72o"
    c1, c2 = cards[0], cards[1]
    r1, r2 = _rank(c1), _rank(c2)
    if RANK_VALUE[r2] > RANK_VALUE[r1]:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1 + r2
    suffix = "s" if c1[1] == c2[1] else "o"
    return r1 + r2 + suffix


def _is_pair(cards):
    return len(cards) >= 2 and _rank(cards[0]) == _rank(cards[1])


def _is_suited(cards):
    return len(cards) >= 2 and cards[0][1] == cards[1][1]


def _connectedness(cards):
    if len(cards) < 2:
        return 12
    return abs(_rank_value(cards[0]) - _rank_value(cards[1]))


def _preflop_score_formula(cards):
    cls = _hand_class(cards)
    high = max((_rank_value(c) for c in cards), default=2)
    low = min((_rank_value(c) for c in cards), default=2)

    if cls in PREMIUM_CLASSES:
        return 92
    if cls in STRONG_CLASSES:
        return 76
    if cls in SPECULATIVE_CLASSES:
        base = 55
    else:
        base = high * 4 + low * 1.5

    if _is_pair(cards):
        base += 18 + high
    if _is_suited(cards):
        base += 8
    gap = _connectedness(cards)
    if gap == 1:
        base += 8
    elif gap == 2:
        base += 4
    elif gap >= 5:
        base -= 8
    if high == 14:
        base += 7
    if high >= 11 and low >= 9:
        base += 8
    return min(90, max(8, int(base)))


def _representative_cards(hand_class):
    if len(hand_class) == 2:
        return [hand_class[0] + "s", hand_class[1] + "h"]
    high, low, suitedness = hand_class[0], hand_class[1], hand_class[2]
    if suitedness == "s":
        return [high + "s", low + "s"]
    return [high + "s", low + "h"]


PREFLOP_SCORE_TABLE = {
    "AA": 100,
    "AKs": 68,
    "AKo": 65,
    "AQs": 66,
    "AQo": 63,
    "AJs": 65,
    "AJo": 62,
    "ATs": 63,
    "ATo": 60,
    "A9s": 61,
    "A9o": 56,
    "A8s": 58,
    "A8o": 56,
    "A7s": 56,
    "A7o": 53,
    "A6s": 55,
    "A6o": 52,
    "A5s": 56,
    "A5o": 50,
    "A4s": 55,
    "A4o": 49,
    "A3s": 54,
    "A3o": 48,
    "A2s": 52,
    "A2o": 48,
    "KK": 95,
    "KQs": 62,
    "KQo": 58,
    "KJs": 60,
    "KJo": 57,
    "KTs": 59,
    "KTo": 56,
    "K9s": 56,
    "K9o": 52,
    "K8s": 53,
    "K8o": 49,
    "K7s": 52,
    "K7o": 48,
    "K6s": 51,
    "K6o": 46,
    "K5s": 48,
    "K5o": 44,
    "K4s": 47,
    "K4o": 44,
    "K3s": 46,
    "K3o": 40,
    "K2s": 44,
    "K2o": 39,
    "QQ": 90,
    "QJs": 56,
    "QJo": 52,
    "QTs": 55,
    "QTo": 51,
    "Q9s": 52,
    "Q9o": 48,
    "Q8s": 48,
    "Q8o": 45,
    "Q7s": 46,
    "Q7o": 42,
    "Q6s": 45,
    "Q6o": 40,
    "Q5s": 44,
    "Q5o": 37,
    "Q4s": 42,
    "Q4o": 38,
    "Q3s": 41,
    "Q3o": 35,
    "Q2s": 39,
    "Q2o": 34,
    "JJ": 85,
    "JTs": 51,
    "JTo": 48,
    "J9s": 48,
    "J9o": 44,
    "J8s": 45,
    "J8o": 41,
    "J7s": 43,
    "J7o": 37,
    "J6s": 40,
    "J6o": 35,
    "J5s": 38,
    "J5o": 35,
    "J4s": 36,
    "J4o": 31,
    "J3s": 36,
    "J3o": 30,
    "J2s": 33,
    "J2o": 28,
    "TT": 81,
    "T9s": 46,
    "T9o": 42,
    "T8s": 42,
    "T8o": 37,
    "T7s": 40,
    "T7o": 35,
    "T6s": 37,
    "T6o": 32,
    "T5s": 34,
    "T5o": 29,
    "T4s": 33,
    "T4o": 27,
    "T3s": 31,
    "T3o": 26,
    "T2s": 29,
    "T2o": 23,
    "99": 76,
    "98s": 40,
    "98o": 35,
    "97s": 37,
    "97o": 32,
    "96s": 33,
    "96o": 28,
    "95s": 31,
    "95o": 26,
    "94s": 28,
    "94o": 22,
    "93s": 26,
    "93o": 21,
    "92s": 25,
    "92o": 20,
    "88": 72,
    "87s": 35,
    "87o": 29,
    "86s": 31,
    "86o": 27,
    "85s": 29,
    "85o": 23,
    "84s": 26,
    "84o": 21,
    "83s": 23,
    "83o": 17,
    "82s": 22,
    "82o": 16,
    "77": 66,
    "76s": 31,
    "76o": 25,
    "75s": 29,
    "75o": 21,
    "74s": 25,
    "74o": 19,
    "73s": 21,
    "73o": 14,
    "72s": 17,
    "72o": 12,
    "66": 61,
    "65s": 26,
    "65o": 21,
    "64s": 23,
    "64o": 18,
    "63s": 20,
    "63o": 14,
    "62s": 18,
    "62o": 11,
    "55": 56,
    "54s": 24,
    "54o": 18,
    "53s": 21,
    "53o": 15,
    "52s": 18,
    "52o": 12,
    "44": 51,
    "43s": 19,
    "43o": 13,
    "42s": 17,
    "42o": 10,
    "33": 44,
    "32s": 14,
    "32o": 8,
    "22": 38,
}


def _preflop_score(cards):
    cls = _hand_class(cards)
    if TABLES is not None:
        try:
            classes = TABLES["preflop_classes"]
            scores = TABLES["preflop_scores"]
            for index, hand_class in enumerate(classes):
                if str(hand_class) == cls:
                    return int(scores[index])
        except Exception:
            pass
    return PREFLOP_SCORE_TABLE.get(cls, _preflop_score_formula(cards))


# ---------------------------------------------------------------------------
# State parsing helpers
# ---------------------------------------------------------------------------

def _players_in_hand(state):
    return [p for p in state.get("players", []) if not p.get("is_folded")]


def _active_opponent_count(state):
    hero_seat = state.get("seat_to_act")
    count = 0
    for player in _players_in_hand(state):
        if player.get("seat") != hero_seat:
            count += 1
    return max(1, count)


def _position_bucket(state):
    """Approximate position from blind logs and seat order.

    Fullhouse uses fixed seats within each hand. The dealer is the seat before
    the small blind for 3+ players, and the small blind is dealer heads-up.
    """
    players = state.get("players", [])
    n = max(1, len(players))
    seat = int(state.get("seat_to_act", 0))
    if n <= 2:
        return "late" if state.get("street") != "preflop" else "early"

    sb = None
    for action in state.get("action_log", []):
        if action.get("action") == "small_blind":
            sb = action.get("seat")
            break
    if sb is None:
        return "late" if seat >= n - 2 else "middle" if seat >= 2 else "early"

    dealer = (int(sb) - 1) % n
    rel = (seat - dealer) % n
    if rel in (0, n - 1):
        return "late"
    if rel >= max(1, n - 3):
        return "middle"
    return "early"


def _pot_odds(state):
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(0, int(state.get("pot", 0) or 0))
    if owed <= 0:
        return 0.0
    return owed / max(1, pot + owed)


def _facing_raise_preflop(state):
    if state.get("street") != "preflop":
        return False
    current_bet = int(state.get("current_bet", 0) or 0)
    return current_bet > BIG_BLIND


def _effective_stack(state):
    invested = int(state.get("your_bet_this_street", 0) or 0)
    return int(state.get("your_stack", 0) or 0) + invested


def _stack_to_pot_ratio(state):
    return _effective_stack(state) / max(1, int(state.get("pot", 0) or 0))


# ---------------------------------------------------------------------------
# Opponent model
# ---------------------------------------------------------------------------

def _default_stats():
    return {
        "actions": 0,
        "raises": 0,
        "calls": 0,
        "folds": 0,
        "checks": 0,
        "all_ins": 0,
        "raise_total": 0,
        "raise_count": 0,
        "pressure_events": 0,
        "pressure_folds": 0,
        "pressure_calls": 0,
    }


def _remember_action(action, previous_action=None):
    key = (
        action.get("hand_num"),
        action.get("seat"),
        action.get("bot_id"),
        action.get("action"),
        action.get("amount"),
    )
    if key in SEEN_ACTIONS:
        return
    if len(SEEN_ACTIONS) > SEEN_ACTIONS_MAX:
        SEEN_ACTIONS.clear()
    SEEN_ACTIONS.add(key)

    bot_id = action.get("bot_id")
    if not bot_id:
        return
    act = action.get("action")
    if act in ("small_blind", "big_blind", None):
        return

    stats = OPPONENTS.setdefault(bot_id, _default_stats())
    stats["actions"] += 1
    if act == "raise":
        stats["raises"] += 1
        amount = int(action.get("amount") or 0)
        stats["raise_total"] += amount
        stats["raise_count"] += 1
    elif act == "call":
        stats["calls"] += 1
    elif act == "fold":
        stats["folds"] += 1
    elif act == "check":
        stats["checks"] += 1
    elif act == "all_in":
        stats["all_ins"] += 1

    if (
        previous_action
        and previous_action.get("hand_num") == action.get("hand_num")
        and previous_action.get("bot_id") != bot_id
        and previous_action.get("action") in ("raise", "all_in")
        and act in ("fold", "call")
    ):
        stats["pressure_events"] += 1
        if act == "fold":
            stats["pressure_folds"] += 1
        elif act == "call":
            stats["pressure_calls"] += 1


def _update_memory(state):
    previous = None
    for action in state.get("match_action_log", []):
        _remember_action(action, previous)
        previous = action


def _rate(stats, key, prior=1.0, mass=5.0):
    return (stats.get(key, 0) + prior) / max(1.0, stats.get("actions", 0) + mass)


def _profile_for(bot_id):
    stats = OPPONENTS.get(bot_id)
    if not stats or stats.get("actions", 0) < 10:
        return "unknown"

    raise_rate = _rate(stats, "raises")
    call_rate = _rate(stats, "calls")
    fold_rate = _rate(stats, "folds")
    all_in_rate = _rate(stats, "all_ins", prior=0.2, mass=8.0)
    pressure_events = stats.get("pressure_events", 0)
    pressure_fold_rate = (
        (stats.get("pressure_folds", 0) + 1) / (pressure_events + 3)
        if pressure_events >= 4 else None
    )
    avg_raise_bb = stats.get("raise_total", 0) / max(1, stats.get("raise_count", 0)) / BIG_BLIND

    if (
        raise_rate > PROFILE_MANIAC_RAISE_RATE
        or all_in_rate > PROFILE_MANIAC_ALL_IN_RATE
        or avg_raise_bb > PROFILE_MANIAC_AVG_RAISE_BB
    ):
        return "maniac"
    if call_rate > PROFILE_STATION_CALL_RATE and fold_rate < PROFILE_STATION_FOLD_RATE:
        return "station"
    if (
        pressure_fold_rate is not None
        and pressure_fold_rate > PROFILE_PRESSURE_NIT_FOLD_RATE
        and raise_rate < 0.24
    ):
        return "nit"
    if fold_rate > PROFILE_NIT_FOLD_RATE and raise_rate < PROFILE_NIT_RAISE_RATE:
        return "nit"
    if (
        pressure_fold_rate is not None
        and pressure_fold_rate < PROFILE_PRESSURE_STATION_FOLD_RATE
        and call_rate > 0.34
    ):
        return "station"
    if raise_rate < 0.20 and call_rate < 0.34:
        return "abc"
    return "unknown"


def _table_profile(state):
    profiles = []
    hero_seat = state.get("seat_to_act")
    for player in state.get("players", []):
        if player.get("seat") == hero_seat or player.get("is_folded"):
            continue
        profiles.append(_profile_for(player.get("bot_id")))
    if not profiles:
        return "unknown"
    for preferred in ("maniac", "station", "nit", "abc"):
        if profiles.count(preferred) >= max(1, len(profiles) // 2):
            return preferred
    return "mixed"


def _fold_pressure(state):
    """Estimate how likely a table is to fold to a bet."""
    profile = _table_profile(state)
    hero_seat = state.get("seat_to_act")
    pressure_rates = []
    for player in state.get("players", []):
        if player.get("seat") == hero_seat or player.get("is_folded"):
            continue
        stats = OPPONENTS.get(player.get("bot_id"))
        if stats and stats.get("pressure_events", 0) >= 4:
            rate = (stats.get("pressure_folds", 0) + 1) / (stats.get("pressure_events", 0) + 3)
            pressure_rates.append(rate)
    if pressure_rates:
        observed = sum(pressure_rates) / len(pressure_rates)
        return _clamp(observed, 0.15, 0.82)
    if profile == "nit":
        return 0.72
    if profile == "abc":
        return 0.58
    if profile == "station":
        return 0.22
    if profile == "maniac":
        return 0.30
    return 0.45


# ---------------------------------------------------------------------------
# Equity engine
# ---------------------------------------------------------------------------

def _equity_cache_key(state, opponents):
    cards = tuple(sorted(state.get("your_cards", [])))
    board = tuple(state.get("community_cards", []))
    return cards, board, int(opponents)


def _monte_carlo_equity(state, opponents, samples, budget_s):
    key = _equity_cache_key(state, opponents)
    cached = EQUITY_CACHE.get(key)
    if cached is not None:
        return cached

    start = time.perf_counter()
    try:
        hero = [eval7.Card(card) for card in state.get("your_cards", [])]
        board = [eval7.Card(card) for card in state.get("community_cards", [])]
    except Exception:
        return 0.0

    dead = set(hero + board)
    deck = [card for card in FULL_DECK if card not in dead]
    needed_board = max(0, 5 - len(board))
    opponents = max(1, min(5, int(opponents)))
    need_cards = opponents * 2 + needed_board
    if len(deck) < need_cards:
        return 0.0

    wins = 0.0
    trials = 0
    for _ in range(max(1, samples)):
        if time.perf_counter() - start > budget_s:
            break
        random.shuffle(deck)
        idx = 0
        opp_hands = []
        for _opp in range(opponents):
            opp_hands.append([deck[idx], deck[idx + 1]])
            idx += 2
        runout = list(board)
        for _card in range(needed_board):
            runout.append(deck[idx])
            idx += 1

        hero_score = eval7.evaluate(hero + runout)
        opp_scores = [eval7.evaluate(hand + runout) for hand in opp_hands]
        best = max([hero_score] + opp_scores)
        if hero_score == best:
            tied = 1 + sum(1 for score in opp_scores if score == best)
            wins += 1.0 / tied
        trials += 1

    equity = wins / trials if trials else 0.0
    if len(EQUITY_CACHE) > EQUITY_CACHE_MAX:
        EQUITY_CACHE.clear()
    EQUITY_CACHE[key] = equity
    return equity


def _estimate_equity(state, started_at):
    opponents = _active_opponent_count(state)
    board_len = len(state.get("community_cards", []))
    if board_len == 0:
        pre = _preflop_score(state.get("your_cards", [])) / 100.0
        multiway_penalty = 0.055 * max(0, opponents - 1)
        return max(0.05, min(0.86, pre - multiway_penalty))

    remaining_budget = max(0.03, DECIDE_BUDGET_S - (time.perf_counter() - started_at) - 0.05)
    budget = min(EQUITY_BUDGET_S, remaining_budget)
    if board_len >= 5:
        samples = RIVER_SAMPLES
    elif board_len == 4:
        samples = TURN_SAMPLES
    else:
        samples = FLOP_SAMPLES
    if opponents >= 4:
        samples = int(samples * MULTIWAY_SAMPLE_FACTOR)
    return _monte_carlo_equity(state, opponents, samples, budget)


# ---------------------------------------------------------------------------
# Board texture
# ---------------------------------------------------------------------------

def _board_texture(state):
    board = state.get("community_cards", [])
    if len(board) < 3:
        return "none"
    suits = {}
    ranks = []
    for card in board:
        suits[card[1]] = suits.get(card[1], 0) + 1
        ranks.append(_rank_value(card))
    suited = max(suits.values()) if suits else 0
    unique = sorted(set(ranks))
    connected = 0
    for i in range(1, len(unique)):
        if unique[i] - unique[i - 1] <= 2:
            connected += 1
    paired = len(unique) < len(ranks)
    if suited >= 3 or connected >= 2:
        return "wet"
    if paired or (suited <= 2 and connected == 0):
        return "dry"
    return "medium"


def _straight_draw(ranks):
    vals = set(ranks)
    if 14 in vals:
        vals.add(1)
    for start in range(1, 11):
        window = {start, start + 1, start + 2, start + 3, start + 4}
        if len(window & vals) >= 4:
            return True
    return False


def _hand_features(state):
    cards = state.get("your_cards", []) + state.get("community_cards", [])
    if len(cards) < 5:
        return {"made_rank": 0, "hand_type": "unknown", "flush_draw": False, "straight_draw": False}
    try:
        eval_cards = [eval7.Card(card) for card in cards]
        score = eval7.evaluate(eval_cards)
        hand_type = str(eval7.handtype(score)).lower()
    except Exception:
        hand_type = "unknown"

    made_rank = 0
    if "straight flush" in hand_type:
        made_rank = 8
    elif "quads" in hand_type or "four" in hand_type:
        made_rank = 7
    elif "full house" in hand_type:
        made_rank = 6
    elif "flush" in hand_type:
        made_rank = 5
    elif "straight" in hand_type:
        made_rank = 4
    elif "trips" in hand_type or "three" in hand_type:
        made_rank = 3
    elif "two pair" in hand_type:
        made_rank = 2
    elif "pair" in hand_type:
        made_rank = 1

    suits = {}
    ranks = []
    for card in cards:
        suits[card[1]] = suits.get(card[1], 0) + 1
        ranks.append(_rank_value(card))
    board_len = len(state.get("community_cards", []))
    flush_draw = board_len < 5 and max(suits.values() or [0]) >= 4 and made_rank < 5
    straight_draw = board_len < 5 and _straight_draw(ranks) and made_rank < 4
    return {
        "made_rank": made_rank,
        "hand_type": hand_type,
        "flush_draw": flush_draw,
        "straight_draw": straight_draw,
    }


# ---------------------------------------------------------------------------
# Bet sizing and action sanitizer
# ---------------------------------------------------------------------------

def _raise_to_fraction(state, fraction):
    pot = int(state.get("pot", 0) or 0)
    current = int(state.get("your_bet_this_street", 0) or 0)
    stack = int(state.get("your_stack", 0) or 0)
    target = int(state.get("current_bet", 0) or 0) + max(BIG_BLIND, int(pot * fraction))
    target = max(target, int(state.get("min_raise_to", 0) or 0))
    return min(target, current + stack)


def _raise_to_preflop(state, big_blinds):
    current_bet = int(state.get("current_bet", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    stack = int(state.get("your_stack", 0) or 0)
    target = max(
        int(state.get("min_raise_to", 0) or 0),
        int(BIG_BLIND * big_blinds),
        int(current_bet * 2.4),
    )
    return min(target, invested + stack)


def _sizing_fraction(state, fraction, purpose, profile, texture, equity):
    shaped = float(fraction)
    spr = _stack_to_pot_ratio(state)

    if purpose == "value" and spr <= SPR_LOW and equity >= SPR_COMMIT_EQUITY:
        shaped += SPR_LOW_RAISE_FRACTION_BONUS

    if OFF_BUCKET_SIZING_PROB > 0 and random.random() < OFF_BUCKET_SIZING_PROB:
        if purpose in ("bluff", "semi_bluff") and profile in ("nit", "abc", "mixed", "unknown"):
            shaped = max(shaped, OFF_BUCKET_BLUFF_MIN)
        elif purpose == "value" and profile == "station":
            shaped += OFF_BUCKET_VALUE_STATION_BONUS
        elif purpose in ("value", "thin_value") and profile in ("nit", "abc"):
            shaped = min(shaped, OFF_BUCKET_VALUE_TIGHT_MAX)
        elif purpose == "thin_value":
            shaped = min(shaped, 0.49)

    if texture == "wet" and purpose == "value":
        shaped = max(shaped, fraction)
    return _clamp(shaped, 0.20, 1.25)


def _sanitize_action(state, intent):
    if not isinstance(intent, dict):
        intent = {}
    action = str(intent.get("action", "")).lower()
    owed = int(state.get("amount_owed", 0) or 0)
    can_check = bool(state.get("can_check", owed == 0))
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)

    if stack <= 0:
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "check":
        return {"action": "check"} if can_check else {"action": "call"}
    if action == "call":
        return {"action": "check"} if can_check else {"action": "call"}
    if action == "fold":
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "all_in":
        return {"action": "all_in"}
    if action == "raise":
        try:
            amount = int(intent.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0
        min_raise = int(state.get("min_raise_to", 0) or 0)
        max_total = invested + stack
        if max_total <= min_raise:
            return {"action": "all_in"} if max_total > owed else ({"action": "call"} if not can_check else {"action": "check"})
        amount = max(min_raise, min(amount, max_total))
        return {"action": "raise", "amount": amount}

    if can_check:
        return {"action": "check"}
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= max(BIG_BLIND, int(pot * 0.08)):
        return {"action": "call"}
    return {"action": "fold"}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _preflop_policy(state):
    cards = state.get("your_cards", [])
    cls = _hand_class(cards)
    score = _preflop_score(cards)
    position = _position_bucket(state)
    profile = _table_profile(state)
    owed = int(state.get("amount_owed", 0) or 0)
    stack_total = _effective_stack(state)
    pot = max(1, int(state.get("pot", 0) or 0))
    facing_raise = _facing_raise_preflop(state)
    odds = _pot_odds(state)
    heads_up = len(state.get("players", [])) <= 2

    late_bonus = 8 if position == "late" else 3 if position == "middle" else 0
    if profile == "nit":
        late_bonus += 5
    if profile == "maniac" and facing_raise:
        late_bonus -= 8

    adjusted = score + late_bonus
    shallow = stack_total <= 18 * BIG_BLIND

    if facing_raise:
        current_bet = int(state.get("current_bet", 0) or 0)
        risk = owed / max(1, stack_total)
        if heads_up and profile == "maniac":
            if adjusted >= PREFLOP_HU_MANIAC_CALL_STRONG and risk <= 0.24:
                return {"action": "call"}
            if adjusted >= PREFLOP_HU_MANIAC_CALL_MEDIUM and odds <= 0.30 and risk <= 0.13:
                return {"action": "call"}
        if cls in ULTRA_PREMIUM_CLASSES:
            if current_bet <= 9 * BIG_BLIND and risk <= 0.22:
                return {
                    "action": "raise",
                    "amount": _raise_to_preflop(
                        state,
                        max(PREFLOP_RERAISE_MIN_BB, current_bet / BIG_BLIND * PREFLOP_RERAISE_MULT),
                    ),
                }
            if risk <= 0.55 or shallow:
                return {"action": "call"}
            return {"action": "fold"}
        if cls in PREMIUM_CLASSES or adjusted >= PREFLOP_PREMIUM_CALL_SCORE:
            if risk <= 0.24 and owed <= pot * 0.48:
                return {"action": "call"}
            return {"action": "fold"}
        if adjusted >= PREFLOP_LATE_RAISE_CALL_SCORE and odds < 0.15 and position == "late":
            return {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if cls in PREMIUM_CLASSES or adjusted >= PREFLOP_PREMIUM_OPEN_SCORE:
        return {"action": "raise", "amount": _raise_to_preflop(state, PREFLOP_PREMIUM_OPEN_BB)}

    if heads_up and profile == "maniac" and adjusted >= PREFLOP_HU_MANIAC_OPEN_SCORE:
        return {"action": "raise", "amount": _raise_to_preflop(state, PREFLOP_HU_MANIAC_OPEN_BB)}

    if cls in STRONG_CLASSES or adjusted >= PREFLOP_OPEN_SCORE:
        return {"action": "raise", "amount": _raise_to_preflop(state, PREFLOP_OPEN_BB)}

    if (adjusted >= PREFLOP_LATE_PLAY_SCORE or cls in SPECULATIVE_CLASSES) and position != "early":
        if state.get("can_check"):
            if profile == "nit" and random.random() < 0.45:
                return {"action": "raise", "amount": _raise_to_fraction(state, DRY_BLUFF_FRACTION)}
            return {"action": "check"}
        if owed <= max(BIG_BLIND, int(pot * 0.16)):
            return {"action": "call"}

    if state.get("can_check"):
        return {"action": "check"}
    if owed <= max(BIG_BLIND, int(pot * 0.08)) and adjusted >= PREFLOP_CHEAP_CALL_SCORE:
        return {"action": "call"}
    return {"action": "fold"}


def _call_margin(state, profile):
    opponents = _active_opponent_count(state)
    margin = CALL_MARGIN_BASE + CALL_MARGIN_MULTIWAY * max(0, opponents - 1)
    if _position_bucket(state) == "early":
        margin += 0.025
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed > pot * 0.70:
        margin += 0.04
    if profile in ("nit", "abc"):
        margin += 0.03
    elif profile == "maniac":
        margin -= 0.055
    elif profile == "station":
        margin -= 0.015
    if _is_heads_up_stack_leader(state) and profile == "maniac":
        margin += 0.04
    spr = _stack_to_pot_ratio(state)
    if spr <= SPR_LOW:
        margin -= SPR_LOW_CALL_MARGIN_DISCOUNT
    elif spr >= SPR_HIGH and owed > pot * 0.35:
        margin += SPR_HIGH_CALL_MARGIN_BONUS
    return max(0.015, margin)


def _is_heads_up_stack_leader(state):
    players = state.get("players", [])
    if len(players) != 2:
        return False
    hero_seat = state.get("seat_to_act")
    hero_stack = int(state.get("your_stack", 0) or 0)
    opp_stack = 0
    for player in players:
        if player.get("seat") != hero_seat:
            opp_stack = int(player.get("stack", 0) or 0)
            break
    return hero_stack > max(1, int(opp_stack * 1.25))


def _passes_risk_guard(state, equity, profile):
    owed = int(state.get("amount_owed", 0) or 0)
    if owed <= 0:
        return True
    stack_total = max(1, _effective_stack(state))
    risk = owed / stack_total
    opponents = _active_opponent_count(state)
    if _stack_to_pot_ratio(state) <= SPR_LOW and equity >= SPR_COMMIT_EQUITY and profile != "nit":
        return True
    required = 0.0
    if risk >= RISK_CUTOFF_HIGH:
        required = RISK_REQ_HIGH
    elif risk >= RISK_CUTOFF_MID:
        required = RISK_REQ_MID
    elif risk >= RISK_CUTOFF_LOW:
        required = RISK_REQ_LOW
    if opponents >= 3:
        required += 0.04
    if profile == "maniac":
        required -= 0.02
    if profile == "maniac" and _is_heads_up_stack_leader(state) and risk >= 0.20:
        required += 0.08
    if required <= 0:
        return True
    return equity >= required


def _adjust_equity_for_context(state, raw_equity, profile):
    adjusted = raw_equity
    if not state.get("can_check"):
        if profile == "nit":
            adjusted += EQUITY_ADJ_NIT
        elif profile == "abc":
            adjusted += EQUITY_ADJ_ABC
        elif profile == "maniac":
            adjusted += EQUITY_ADJ_MANIAC
        current_bet = int(state.get("current_bet", 0) or 0)
        pot = max(1, int(state.get("pot", 0) or 0))
        if current_bet > pot * 0.70 and profile != "maniac":
            adjusted += EQUITY_ADJ_LARGE_BET
    adjusted += EQUITY_ADJ_MULTIWAY * max(0, _active_opponent_count(state) - 2)
    return _clamp(adjusted, 0.02, 0.98)


def _postflop_policy(state, equity):
    profile = _table_profile(state)
    equity = _adjust_equity_for_context(state, equity, profile)
    texture = _board_texture(state)
    features = _hand_features(state)
    odds = _pot_odds(state)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    can_check = bool(state.get("can_check"))
    opponents = _active_opponent_count(state)
    fold_pressure = _fold_pressure(state)
    spr = _stack_to_pot_ratio(state)

    value_threshold = VALUE_THRESHOLD_BASE + 0.055 * max(0, opponents - 1)
    thin_value = THIN_VALUE_BASE + 0.045 * max(0, opponents - 1)
    if spr <= SPR_LOW:
        value_threshold -= SPR_LOW_VALUE_DISCOUNT
        thin_value -= SPR_LOW_THIN_VALUE_DISCOUNT
    if profile == "station":
        value_threshold -= 0.05
        thin_value -= 0.045
    if profile == "maniac":
        value_threshold -= 0.03
    if texture == "wet":
        value_threshold += 0.025
    if features["made_rank"] >= 4:
        value_threshold -= 0.035
        thin_value -= 0.020

    if can_check:
        if equity >= value_threshold:
            frac = PRESSURE_VALUE_FRACTION if profile in ("station", "maniac") or texture == "wet" else NORMAL_VALUE_FRACTION
            frac = _sizing_fraction(state, frac, "value", profile, texture, equity)
            return {"action": "raise", "amount": _raise_to_fraction(state, frac)}
        if equity >= thin_value and profile in ("station", "maniac"):
            frac = _sizing_fraction(state, THIN_VALUE_FRACTION, "thin_value", profile, texture, equity)
            return {"action": "raise", "amount": _raise_to_fraction(state, frac)}
        has_draw = features["flush_draw"] or features["straight_draw"]
        if equity >= 0.42 and texture != "wet" and fold_pressure >= 0.55 and random.random() < DRY_BLUFF_PROB:
            frac = _sizing_fraction(state, DRY_BLUFF_FRACTION, "bluff", profile, texture, equity)
            return {"action": "raise", "amount": _raise_to_fraction(state, frac)}
        if has_draw and equity >= 0.30 and fold_pressure >= 0.52 and profile != "station" and random.random() < WET_BLUFF_PROB:
            frac = _sizing_fraction(state, WET_SEMI_BLUFF_FRACTION, "semi_bluff", profile, texture, equity)
            return {"action": "raise", "amount": _raise_to_fraction(state, frac)}
        return {"action": "check"}

    margin = _call_margin(state, profile)
    if not _passes_risk_guard(state, equity, profile):
        return {"action": "fold"}
    if equity >= odds + margin:
        if equity >= max(0.88, value_threshold + 0.16) and owed < pot * 0.20:
            frac = _sizing_fraction(state, HIGH_EQUITY_RAISE_FRACTION, "value", profile, texture, equity)
            return {"action": "raise", "amount": _raise_to_fraction(state, frac)}
        return {"action": "call"}

    if equity >= odds + 0.015 and profile == "maniac":
        return {"action": "call"}
    if owed <= max(BIG_BLIND, int(pot * 0.08)) and equity >= 0.18:
        return {"action": "call"}
    return {"action": "fold"}


def _fallback(state):
    if state.get("can_check", False):
        return {"action": "check"}
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= max(BIG_BLIND, int(pot * 0.08)):
        return {"action": "call"}
    return {"action": "fold"}


def decide(game_state):
    started_at = time.perf_counter()
    try:
        if game_state.get("type") == "warmup":
            return {"action": "check"}

        _update_memory(game_state)

        if game_state.get("street") == "preflop":
            intent = _preflop_policy(game_state)
        else:
            equity = _estimate_equity(game_state, started_at)
            intent = _postflop_policy(game_state, equity)

        return _sanitize_action(game_state, intent)
    except Exception:
        return _sanitize_action(game_state, _fallback(game_state))
