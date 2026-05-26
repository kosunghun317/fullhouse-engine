"""Shared local-only helpers for mock competitor families.

These helpers are intentionally outside the submitted heuristic bot. Variant
bot.py files import them to keep benchmark opponents small and consistent.
"""

import os
import random

import eval7
import numpy as np


RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANK_ORDER)}
FULL_DECK = [eval7.Card(rank + suit) for rank in RANK_ORDER for suit in "shdc"]
DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
_POLICY_CACHE = None


def rank_value(card):
    return RANK_VALUE.get(card[0], 2)


def active_opponents(state):
    seat = state.get("seat_to_act")
    return max(1, sum(
        1 for player in state.get("players", [])
        if player.get("seat") != seat and not player.get("is_folded")
    ))


def pot_odds(state):
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= 0:
        return 0.0
    return owed / max(1, pot + owed)


def hand_score(cards):
    if len(cards) < 2:
        return 0.2
    values = sorted((rank_value(card) for card in cards), reverse=True)
    pair = values[0] == values[1]
    suited = cards[0][1] == cards[1][1]
    gap = values[0] - values[1]
    score = values[0] / 14 * 0.45 + values[1] / 14 * 0.25
    if pair:
        score += 0.28 + values[0] / 100
    if suited:
        score += 0.06
    if gap <= 1:
        score += 0.04
    if values[0] == 14:
        score += 0.08
    return max(0.04, min(0.96, score))


def board_texture(state):
    board = state.get("community_cards", [])
    if len(board) < 3:
        return "none"
    suits = {}
    ranks = []
    for card in board:
        suits[card[1]] = suits.get(card[1], 0) + 1
        ranks.append(rank_value(card))
    wet = max(suits.values(), default=0) >= 3
    if len(ranks) >= 3 and max(ranks) - min(ranks) <= 5:
        wet = True
    paired = len(set(ranks)) < len(ranks)
    if wet:
        return "wet"
    if paired:
        return "paired"
    return "dry"


def raise_to_fraction(state, frac):
    pot = max(1, int(state.get("pot", 0) or 0))
    current = int(state.get("current_bet", 0) or 0)
    stack_total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    amount = max(int(state.get("min_raise_to", 0) or 0), current + int(pot * frac))
    return {"action": "raise", "amount": min(stack_total, amount)}


def equity_estimate(state, config):
    board = state.get("community_cards", [])
    preflop_bias = float(config.get("preflop_bias", 0.0))
    opponent_penalty = float(config.get("opponent_penalty", 0.055))
    if not board:
        return max(0.03, min(0.94, hand_score(state.get("your_cards", [])) + preflop_bias - opponent_penalty * (active_opponents(state) - 1)))

    try:
        hero = [eval7.Card(card) for card in state.get("your_cards", [])]
        community = [eval7.Card(card) for card in board]
    except Exception:
        return 0.0
    dead = set(hero + community)
    deck = [card for card in FULL_DECK if card not in dead]
    opp_count = min(5, active_opponents(state))
    needed = opp_count * 2 + max(0, 5 - len(community))
    if len(deck) < needed:
        return 0.0

    samples = int(config.get("samples", 140))
    wins = 0.0
    trials = 0
    for _ in range(max(1, samples)):
        random.shuffle(deck)
        idx = 0
        opp_hands = []
        for _opp in range(opp_count):
            opp_hands.append([deck[idx], deck[idx + 1]])
            idx += 2
        runout = list(community)
        while len(runout) < 5:
            runout.append(deck[idx])
            idx += 1
        hero_score = eval7.evaluate(hero + runout)
        opp_scores = [eval7.evaluate(hand + runout) for hand in opp_hands]
        best = max([hero_score] + opp_scores)
        if hero_score == best:
            wins += 1.0 / (1 + sum(1 for score in opp_scores if score == best))
        trials += 1
    return max(0.0, min(1.0, wins / max(1, trials) + float(config.get("postflop_bias", 0.0))))


def equity_decide(state, config):
    eq = equity_estimate(state, config)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    odds = pot_odds(state)
    if state.get("can_check"):
        if eq > float(config.get("value_threshold", 0.70)):
            return raise_to_fraction(state, float(config.get("value_frac", 0.62)))
        if state.get("street") != "preflop" and eq > float(config.get("semi_threshold", 0.48)) and random.random() < float(config.get("semi_prob", 0.35)):
            return raise_to_fraction(state, float(config.get("semi_frac", 0.45)))
        return {"action": "check"}
    if eq > odds + float(config.get("call_margin", 0.08)):
        return {"action": "call"}
    if eq > float(config.get("raise_threshold", 0.82)) and owed < pot * float(config.get("raise_max_owed_pot", 0.20)):
        return raise_to_fraction(state, float(config.get("raise_frac", 0.85)))
    return {"action": "fold"}


def bucket_name(state):
    cards = state.get("your_cards", [])
    values = sorted((rank_value(card) for card in cards), reverse=True) if len(cards) >= 2 else [2, 2]
    pair = len(cards) >= 2 and cards[0][0] == cards[1][0]
    suited = len(cards) >= 2 and cards[0][1] == cards[1][1]
    texture = board_texture(state)
    if pair and values[0] >= 10:
        return "premium"
    if values[0] == 14 and values[1] >= 10:
        return "premium"
    if pair or suited or values[0] >= 12:
        return "medium_wet" if texture == "wet" else "medium_dry"
    return "trash_wet" if texture == "wet" else "trash_dry"


def bucket_decide(state, config):
    bucket = bucket_name(state)
    mode = config.get("mode", "halfpot")
    if state.get("street") == "preflop":
        if bucket == "premium":
            return raise_to_fraction(state, float(config.get("premium_preflop", 0.80)))
        if bucket.startswith("medium") and state.get("can_check") and random.random() < float(config.get("medium_open_prob", 0.65)):
            return raise_to_fraction(state, float(config.get("medium_preflop", 0.40)))
        if bucket.startswith("medium") and int(state.get("amount_owed", 0) or 0) <= int(max(100, state.get("pot", 0) * float(config.get("preflop_call_pot", 0.16)))):
            return {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if state.get("can_check"):
        if bucket == "premium":
            frac = float(config.get("premium_frac", 0.75))
            if mode == "mixed" and random.random() < 0.35:
                frac = float(config.get("small_frac", 0.33))
            return raise_to_fraction(state, frac)
        if bucket == "medium_dry" and random.random() < float(config.get("medium_bet_prob", 0.55)):
            return raise_to_fraction(state, float(config.get("medium_frac", 0.50)))
        if bucket.startswith("trash") and random.random() < float(config.get("trash_bluff_prob", 0.15)):
            return raise_to_fraction(state, float(config.get("trash_frac", 0.67)))
        return {"action": "check"}

    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    threshold = float(config.get("premium_call", 0.34)) if bucket == "premium" else float(config.get("medium_call", 0.24)) if bucket.startswith("medium") else float(config.get("trash_call", 0.10))
    if owed <= pot * threshold:
        return {"action": "call"}
    return {"action": "fold"}


def _policy_data():
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    path = os.path.join(DATA_DIR, "policy.npz")
    try:
        data = np.load(path, allow_pickle=False)
        _POLICY_CACHE = {
            "w1": data["w1"].astype(float),
            "b1": data["b1"].astype(float),
            "w2": data["w2"].astype(float),
            "b2": data["b2"].astype(float),
            "mean": data["mean"].astype(float),
            "scale": data["scale"].astype(float),
        }
    except Exception:
        _POLICY_CACHE = None
    return _POLICY_CACHE


def policy_features(state):
    cards = state.get("your_cards", [])
    high = 0.0
    pair = 0.0
    suited = 0.0
    if len(cards) >= 2:
        vals = [rank_value(card) for card in cards]
        high = max(vals) / 14
        pair = 1.0 if vals[0] == vals[1] else 0.0
        suited = 1.0 if cards[0][1] == cards[1][1] else 0.0
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    stack = max(1, int(state.get("your_stack", 0) or 0))
    active = sum(1 for player in state.get("players", []) if not player.get("is_folded"))
    street = state.get("street")
    return np.array([
        1.0,
        owed / max(1, pot + owed),
        min(1.0, pot / 20000),
        high,
        pair,
        min(1.0, owed / stack),
        suited,
        min(1.0, active / 6),
        1.0 if street in ("turn", "river") else 0.0,
        1.0 if state.get("can_check") else 0.0,
    ], dtype=float)


def policy_probs(state, config):
    data = _policy_data()
    x = policy_features(state)
    if data is None:
        weights = np.array([0.65, -0.45, -0.30, 0.25, 0.20, -0.55, 0.35, 0.12, -0.08, 0.16], dtype=float)
        score = float(weights @ x)
        logits = np.array([0.25 - score, score - 0.55, score - 0.92], dtype=float)
    else:
        z = (x - data["mean"]) / data["scale"]
        hidden = np.tanh(z @ data["w1"] + data["b1"])
        logits = hidden @ data["w2"] + data["b2"]
    logits = logits + np.array([
        float(config.get("fold_bias", 0.0)),
        float(config.get("call_bias", 0.0)),
        float(config.get("raise_bias", 0.0)),
    ])
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / max(1e-9, float(np.sum(exp)))


def policy_decide(state, config):
    fold_p, call_p, raise_p = [float(value) for value in policy_probs(state, config)]
    if state.get("can_check"):
        if raise_p > float(config.get("bet_threshold", 0.62)):
            return raise_to_fraction(state, float(config.get("large_frac", 0.80)))
        if raise_p > float(config.get("small_bet_threshold", 0.42)):
            return raise_to_fraction(state, float(config.get("small_frac", 0.50)))
        return {"action": "check"}
    if raise_p > float(config.get("raise_threshold", 0.72)):
        return raise_to_fraction(state, float(config.get("raise_frac", 1.0)))
    if call_p >= fold_p:
        return {"action": "call"}
    return {"action": "fold"}


def pressure_response_rate(state):
    hero_seat = state.get("seat_to_act")
    log = state.get("match_action_log", [])
    raises = 0
    folds_to_raise = 0
    previous = None
    for action in log[-80:]:
        if (
            previous
            and previous.get("action") in ("raise", "all_in")
            and previous.get("seat") == hero_seat
            and action.get("action") in ("fold", "call")
        ):
            raises += 1
            if action.get("action") == "fold":
                folds_to_raise += 1
        previous = action
    return (folds_to_raise + 1) / (raises + 3) if raises else 0.45


def anti_heuristic_decide(state, config):
    eq = equity_estimate(state, config)
    fold_rate = pressure_response_rate(state)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    odds = pot_odds(state)
    if state.get("can_check"):
        if eq > 0.70:
            return raise_to_fraction(state, 0.90)
        if fold_rate > 0.45 and eq > 0.36 and state.get("street") != "preflop":
            return raise_to_fraction(state, 0.78)
        if state.get("street") == "preflop" and hand_score(state.get("your_cards", [])) > 0.54:
            return raise_to_fraction(state, 0.55)
        return {"action": "check"}
    if owed > pot * 0.65 and eq < odds + 0.18:
        return {"action": "fold"}
    if eq > odds + 0.03:
        return {"action": "call"}
    return {"action": "fold"}


def pressure_heads_up_decide(state, config):
    eq = equity_estimate(state, config)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    odds = pot_odds(state)
    if state.get("can_check"):
        if state.get("street") == "preflop" and hand_score(state.get("your_cards", [])) > 0.48:
            return raise_to_fraction(state, 0.65)
        if eq > 0.62 or random.random() < float(config.get("pressure_prob", 0.42)):
            return raise_to_fraction(state, float(config.get("pressure_frac", 0.85)))
        return {"action": "check"}
    if eq > odds + float(config.get("call_margin", 0.01)):
        return {"action": "call"}
    if owed <= pot * 0.18 and random.random() < 0.50:
        return {"action": "call"}
    return {"action": "fold"}
