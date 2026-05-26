"""Heuristic population-exploit bot for Fullhouse.

One-file submission by design. The policy is intentionally interpretable:
preflop hand classes, bounded Monte Carlo equity, public-action opponent
modeling, and a single action sanitizer at the edge.
"""

import os
import random
import time

import eval7

BOT_NAME = "Heuristic Exploit"

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
VALUE_THRESHOLD_BASE = _float_env("HEURISTIC_VALUE_THRESHOLD_BASE", 0.66)
THIN_VALUE_BASE = _float_env("HEURISTIC_THIN_VALUE_BASE", 0.59)
DRY_BLUFF_PROB = _float_env("HEURISTIC_DRY_BLUFF_PROB", 0.45)
WET_BLUFF_PROB = _float_env("HEURISTIC_WET_BLUFF_PROB", 0.25)

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


def _preflop_score(cards):
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
    }


def _remember_action(action):
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


def _update_memory(state):
    for action in state.get("match_action_log", []):
        _remember_action(action)


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

    if raise_rate > 0.33 or all_in_rate > 0.10:
        return "maniac"
    if call_rate > 0.42 and fold_rate < 0.25:
        return "station"
    if fold_rate > 0.42 and raise_rate < 0.18:
        return "nit"
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
        samples = 900
    elif board_len == 4:
        samples = 700
    else:
        samples = 520
    if opponents >= 4:
        samples = int(samples * 0.75)
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
            if adjusted >= 64 and risk <= 0.24:
                return {"action": "call"}
            if adjusted >= 54 and odds <= 0.30 and risk <= 0.13:
                return {"action": "call"}
        if cls in ULTRA_PREMIUM_CLASSES:
            if current_bet <= 9 * BIG_BLIND and risk <= 0.22:
                return {"action": "raise", "amount": _raise_to_preflop(state, max(7.0, current_bet / BIG_BLIND * 2.2))}
            if risk <= 0.55 or shallow:
                return {"action": "call"}
            return {"action": "fold"}
        if cls in PREMIUM_CLASSES or adjusted >= 82:
            if risk <= 0.24 and owed <= pot * 0.48:
                return {"action": "call"}
            return {"action": "fold"}
        if adjusted >= 72 and odds < 0.15 and position == "late":
            return {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if cls in PREMIUM_CLASSES or adjusted >= 88:
        return {"action": "raise", "amount": _raise_to_preflop(state, 3.4)}

    if heads_up and profile == "maniac" and adjusted >= 58:
        return {"action": "raise", "amount": _raise_to_preflop(state, 2.8)}

    if adjusted >= 72:
        return {"action": "raise", "amount": _raise_to_preflop(state, 3.0)}

    if adjusted >= 62 and position != "early":
        if state.get("can_check"):
            if profile == "nit" and random.random() < 0.45:
                return {"action": "raise", "amount": _raise_to_fraction(state, 0.42)}
            return {"action": "check"}
        if owed <= max(BIG_BLIND, int(pot * 0.16)):
            return {"action": "call"}

    if state.get("can_check"):
        return {"action": "check"}
    if owed <= max(BIG_BLIND, int(pot * 0.08)) and adjusted >= 58:
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
    return max(0.015, margin)


def _passes_risk_guard(state, equity, profile):
    owed = int(state.get("amount_owed", 0) or 0)
    if owed <= 0:
        return True
    stack_total = max(1, _effective_stack(state))
    risk = owed / stack_total
    opponents = _active_opponent_count(state)
    required = 0.0
    if risk >= 0.70:
        required = RISK_REQ_HIGH
    elif risk >= 0.45:
        required = RISK_REQ_MID
    elif risk >= 0.28:
        required = RISK_REQ_LOW
    if opponents >= 3:
        required += 0.04
    if profile == "maniac":
        required -= 0.02
    if required <= 0:
        return True
    return equity >= required


def _postflop_policy(state, equity):
    profile = _table_profile(state)
    texture = _board_texture(state)
    odds = _pot_odds(state)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    can_check = bool(state.get("can_check"))
    opponents = _active_opponent_count(state)
    fold_pressure = _fold_pressure(state)

    value_threshold = VALUE_THRESHOLD_BASE + 0.055 * max(0, opponents - 1)
    thin_value = THIN_VALUE_BASE + 0.045 * max(0, opponents - 1)
    if profile == "station":
        value_threshold -= 0.05
        thin_value -= 0.045
    if profile == "maniac":
        value_threshold -= 0.03
    if texture == "wet":
        value_threshold += 0.025

    if can_check:
        if equity >= value_threshold:
            frac = 0.75 if profile in ("station", "maniac") or texture == "wet" else 0.52
            return {"action": "raise", "amount": _raise_to_fraction(state, frac)}
        if equity >= thin_value and profile in ("station", "maniac"):
            return {"action": "raise", "amount": _raise_to_fraction(state, 0.45)}
        if equity >= 0.42 and texture != "wet" and fold_pressure >= 0.55 and random.random() < DRY_BLUFF_PROB:
            return {"action": "raise", "amount": _raise_to_fraction(state, 0.42)}
        if equity >= 0.34 and texture == "wet" and fold_pressure >= 0.62 and random.random() < WET_BLUFF_PROB:
            return {"action": "raise", "amount": _raise_to_fraction(state, 0.55)}
        return {"action": "check"}

    margin = _call_margin(state, profile)
    if not _passes_risk_guard(state, equity, profile):
        return {"action": "fold"}
    if equity >= odds + margin:
        if equity >= max(0.88, value_threshold + 0.16) and owed < pot * 0.20:
            return {"action": "raise", "amount": _raise_to_fraction(state, 0.85)}
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
