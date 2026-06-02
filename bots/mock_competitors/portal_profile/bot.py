"""Replay-derived portal profile mock competitor."""

import json
import math
import os
import random

try:
    import eval7
except Exception:
    eval7 = None


DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

DEFAULT_PROFILE = {
    "profile": "unknown_mixed",
    "vpip": 0.26,
    "pfr": 0.18,
    "raise_rate": 0.19,
    "call_rate": 0.12,
    "pressure_fold_rate": 0.73,
    "showdown_rate": 0.07,
    "all_in_rate": 0.006,
    "avg_raise_to_pot": 1.7,
    "aggression_bias": 0.25,
    "bluff_bias": 0.20,
    "value_bias": 0.12,
}


def _load_profile():
    path = os.path.join(DATA_DIR, "profile.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("mock_config"), dict):
            data = data["mock_config"]
        if not isinstance(data, dict):
            return dict(DEFAULT_PROFILE)
        merged = dict(DEFAULT_PROFILE)
        merged.update({key: data[key] for key in DEFAULT_PROFILE if key in data})
        return merged
    except Exception:
        return dict(DEFAULT_PROFILE)


PROFILE = _load_profile()
BOT_NAME = "PortalProfile_" + str(PROFILE.get("profile", "unknown_mixed"))
BOT_AVATAR = "robot_2"

RANK_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


def _clamp(value, lo, hi):
    return min(hi, max(lo, value))


def _num(state, key, default=0):
    value = state.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return default


def _state_seed(state, salt):
    text = (
        str(state.get("hand_id", ""))
        + "|"
        + str(state.get("street", ""))
        + "|"
        + str(state.get("seat_to_act", ""))
        + "|"
        + str(len(state.get("action_log") or []))
        + "|"
        + "".join(state.get("your_cards") or [])
        + "|"
        + salt
    )
    acc = 2166136261
    for ch in text:
        acc = ((acc ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return acc


def _roll(state, salt):
    return (_state_seed(state, salt) % 1_000_000) / 1_000_000.0


def _active_opponents(state):
    seat = state.get("seat_to_act")
    count = 0
    for player in state.get("players") or []:
        if player.get("seat") == seat:
            continue
        if player.get("state") in {"active", "all_in"} and not player.get("is_folded"):
            count += 1
    return max(1, count)


def _preflop_score(cards):
    if len(cards) < 2:
        return 0.30
    r1 = RANK_VALUE.get(cards[0][0], 2)
    r2 = RANK_VALUE.get(cards[1][0], 2)
    high = max(r1, r2)
    low = min(r1, r2)
    suited = len(cards[0]) > 1 and len(cards[1]) > 1 and cards[0][1] == cards[1][1]
    gap = high - low
    if r1 == r2:
        return _clamp(0.50 + high / 28.0, 0.0, 1.0)
    score = 0.18 + high / 22.0 + low / 70.0
    if suited:
        score += 0.06
    if gap <= 1:
        score += 0.05
    elif gap >= 5:
        score -= 0.06
    if high >= 13 and low >= 10:
        score += 0.08
    return _clamp(score, 0.05, 0.92)


def _board_texture_bonus(cards, board):
    if not board:
        return 0.0
    ranks = [RANK_VALUE.get(c[0], 2) for c in cards + board]
    suits = [c[1] for c in cards + board if len(c) > 1]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}
    made = 0.0
    if max(rank_counts.values()) >= 3:
        made += 0.22
    elif any(count == 2 for count in rank_counts.values()):
        made += 0.10
    if max(suit_counts.values()) >= 4:
        made += 0.12
    unique = sorted(set(ranks))
    run = 1
    best_run = 1
    for idx in range(1, len(unique)):
        if unique[idx] == unique[idx - 1] + 1:
            run += 1
        else:
            run = 1
        best_run = max(best_run, run)
    if best_run >= 4:
        made += 0.10
    return _clamp(made, 0.0, 0.35)


def _eval7_equity(state, samples=72):
    if eval7 is None:
        return None
    try:
        hole = [eval7.Card(card) for card in state.get("your_cards") or []]
        board = [eval7.Card(card) for card in state.get("community_cards") or []]
        if len(hole) != 2 or len(board) > 5:
            return None
        known = set(str(card) for card in hole + board)
        deck = [card for card in eval7.Deck().cards if str(card) not in known]
        need_board = 5 - len(board)
        if need_board < 0 or len(deck) < need_board + 2:
            return None
        rng = random.Random(_state_seed(state, "eval7"))
        wins = 0.0
        for _ in range(samples):
            draw = rng.sample(deck, need_board + 2)
            full_board = board + draw[:need_board]
            villain = draw[need_board:]
            hero_value = eval7.evaluate(hole + full_board)
            villain_value = eval7.evaluate(villain + full_board)
            if hero_value > villain_value:
                wins += 1.0
            elif hero_value == villain_value:
                wins += 0.5
        equity = wins / max(1, samples)
        opponents = _active_opponents(state)
        return equity ** max(1.0, opponents * 0.62)
    except Exception:
        return None


def _strength(state):
    cards = list(state.get("your_cards") or [])
    preflop = _preflop_score(cards)
    if state.get("street") == "preflop":
        equity = _eval7_equity(state, 48)
        if equity is None:
            return preflop
        return _clamp(preflop * 0.70 + equity * 0.30, 0.0, 1.0)
    equity = _eval7_equity(state, 72)
    if equity is None:
        return _clamp(preflop * 0.62 + _board_texture_bonus(cards, state.get("community_cards") or []), 0.0, 1.0)
    return _clamp(equity, 0.0, 1.0)


def _max_total_bet(state):
    return int(_num(state, "your_stack", 0) + _num(state, "your_bet_this_street", 0))


def _sanitize(state, action, amount=None):
    can_check = bool(state.get("can_check"))
    max_total = _max_total_bet(state)
    min_raise = int(_num(state, "min_raise_to", 0))
    if action == "check":
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "fold":
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "call":
        return {"action": "check"} if can_check else {"action": "call"}
    if action == "all_in":
        return {"action": "all_in"} if max_total > 0 else ({"action": "check"} if can_check else {"action": "fold"})
    if action == "raise":
        if max_total <= min_raise or min_raise <= 0:
            return {"action": "call"} if not can_check else {"action": "check"}
        target = int(amount if amount is not None else min_raise)
        target = max(min_raise, min(max_total, target))
        if target >= max_total:
            return {"action": "all_in"}
        return {"action": "raise", "amount": target}
    return {"action": "check"} if can_check else {"action": "fold"}


def _raise_amount(state, strength, pressure=False):
    pot = max(1, int(_num(state, "pot", 1)))
    owed = max(0, int(_num(state, "amount_owed", 0)))
    current = max(0, int(_num(state, "your_bet_this_street", 0)))
    base = float(PROFILE.get("avg_raise_to_pot", 1.5))
    aggression = float(PROFILE.get("aggression_bias", 0.2))
    profile = str(PROFILE.get("profile", "unknown_mixed"))
    if profile == "large_size_jammer":
        base += 0.55
    elif profile == "tight_overfolder":
        base -= 0.25
    elif profile == "sticky_station":
        base -= 0.20
    if pressure:
        base += 0.20
    frac = _clamp(base * (0.72 + strength * 0.36 + aggression * 0.12), 0.45, 2.40)
    target = current + owed + int((pot + owed) * frac)
    return target


def _should_jam(state, strength):
    all_in_rate = float(PROFILE.get("all_in_rate", 0.004))
    if strength >= 0.88 and _roll(state, "jam-value") < min(0.18, all_in_rate * 8.0):
        return True
    if strength >= 0.62 and str(PROFILE.get("profile")) == "large_size_jammer":
        return _roll(state, "jam-profile") < min(0.12, all_in_rate * 5.0)
    return False


def _free_action(state, strength):
    street = state.get("street")
    pfr = float(PROFILE.get("pfr", 0.18))
    raise_rate = float(PROFILE.get("raise_rate", 0.18))
    bluff = float(PROFILE.get("bluff_bias", 0.15))
    value_bias = float(PROFILE.get("value_bias", 0.10))
    if street == "preflop":
        threshold = _clamp(0.82 - pfr * 0.70, 0.48, 0.86)
        if strength >= threshold:
            if _should_jam(state, strength):
                return _sanitize(state, "all_in")
            return _sanitize(state, "raise", _raise_amount(state, strength))
        return _sanitize(state, "check")
    value_threshold = _clamp(0.70 - value_bias * 0.20, 0.50, 0.80)
    bluff_threshold = _clamp(0.42 - bluff * 0.16, 0.30, 0.56)
    bluff_roll = _roll(state, "free-bluff")
    if strength >= value_threshold:
        return _sanitize(state, "raise", _raise_amount(state, strength))
    if strength >= bluff_threshold and bluff_roll < _clamp(raise_rate + bluff * 0.30, 0.02, 0.36):
        return _sanitize(state, "raise", _raise_amount(state, strength, pressure=True))
    return _sanitize(state, "check")


def _pressure_action(state, strength):
    owed = max(0, float(_num(state, "amount_owed", 0)))
    pot = max(1.0, float(_num(state, "pot", 1)))
    pot_odds = owed / max(1.0, pot + owed)
    pressure_fold = float(PROFILE.get("pressure_fold_rate", 0.70))
    call_rate = float(PROFILE.get("call_rate", 0.12))
    value_bias = float(PROFILE.get("value_bias", 0.10))
    raise_rate = float(PROFILE.get("raise_rate", 0.18))
    bluff = float(PROFILE.get("bluff_bias", 0.15))
    call_margin = 0.07 + pressure_fold * 0.13 - call_rate * 0.10 - value_bias * 0.08
    call_threshold = _clamp(pot_odds + call_margin, 0.17, 0.72)
    strong_raise = _clamp(0.77 - value_bias * 0.16, 0.55, 0.82)
    if _should_jam(state, strength):
        return _sanitize(state, "all_in")
    if strength >= strong_raise and _roll(state, "pressure-value") < _clamp(raise_rate + 0.20, 0.08, 0.48):
        return _sanitize(state, "raise", _raise_amount(state, strength, pressure=True))
    if strength >= call_threshold:
        semi_raise = strength >= _clamp(0.46 - bluff * 0.12, 0.32, 0.58)
        if semi_raise and _roll(state, "pressure-bluff") < _clamp(bluff * 0.24, 0.0, 0.16):
            return _sanitize(state, "raise", _raise_amount(state, strength, pressure=True))
        return _sanitize(state, "call")
    if state.get("can_check"):
        return _sanitize(state, "check")
    return _sanitize(state, "fold")


def decide(state):
    if state.get("type") == "warmup":
        return {"action": "check"}
    strength = _strength(state)
    if bool(state.get("can_check")) or int(_num(state, "amount_owed", 0)) <= 0:
        return _free_action(state, strength)
    return _pressure_action(state, strength)
