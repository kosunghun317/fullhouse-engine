"""Standalone threshold-aware rollout submission candidate.

This keeps the tuned rollout equity sampler but changes bet sizing around a
mechanical pot-odds opponent model: value sizes sit below estimated call
thresholds, bluffs sit just above fold thresholds, and large calls are treated
as under-bluffed until the equity margin is clear.
"""

import random
import time

import eval7


RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = tuple(eval7.Card(rank + suit) for rank in RANKS for suit in SUITS)

PARAMS = {
    "samples": 1024.0,
    "f_min_equity": 0.38600570637776854,
    "f_center": 0.5536016465664075,
    "f_scale": 0.10075311307590826,
    "f_min_raise": 0.35886009720451495,
    "f_max_raise": 0.9085659785544977,
    "g_call_edge_base": 0.06366738517571167,
    "g_call_edge_active": 0.03899529132884684,
    "g_pressure_discount": 0.024391452794899727,
    "g_thin_call_edge": 0.01937116600090552,
    "g_raise_edge": 0.22216495168990685,
    "g_raise_equity": 0.9198701106956259,
    "g_raise_max_owed_pot": 0.20878296621524373,
    "g_min_raise": 0.9369258774068039,
}

TIME_BUDGET_SECONDS = 1.50
MIN_ROLLOUT_SAMPLES = 128
TIME_CHECK_INTERVAL = 32
BIG_BLIND = 100
STANDARD_FRACTIONS = (0.27, 0.41, 0.63, 0.88, 1.27, 1.75, 2.60)


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _sigmoid(value):
    value = _clip(float(value), -60.0, 60.0)
    return 1.0 / (1.0 + pow(2.718281828459045, -value))


def _rank_value(card):
    text = str(card)
    return RANKS.index(text[0]) + 2 if text and text[0] in RANKS else 2


def _active_players(state):
    players = list(state.get("players", []) or [])
    if not players:
        return 2
    return max(2, sum(1 for player in players if not player.get("is_folded", False)))


def _state_rng(state, salt="threshold_rollout"):
    action_log = state.get("action_log", []) if isinstance(state, dict) else []
    parts = [
        salt,
        str(state.get("hand_id", "")),
        str(state.get("seat_to_act", "")),
        str(len(action_log)),
        str(state.get("street", "")),
        str(state.get("pot", "")),
        str(state.get("amount_owed", "")),
        ",".join(str(card) for card in (state.get("your_cards", []) or [])),
        ",".join(str(card) for card in (state.get("community_cards", []) or [])),
    ]
    seed = 0
    for ch in "|".join(parts):
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(seed)


def _preflop_strength(cards):
    cards = list(cards or [])
    if len(cards) < 2:
        return 0.15
    values = sorted((_rank_value(card) for card in cards[:2]), reverse=True)
    pair = values[0] == values[1]
    suited = len(cards[0]) > 1 and len(cards[1]) > 1 and cards[0][1] == cards[1][1]
    score = 0.42 * values[0] / 14.0 + 0.20 * values[1] / 14.0
    if pair:
        score += 0.30 + values[0] / 120.0
    if suited:
        score += 0.06
    if values[0] == 14:
        score += 0.06
    return _clip(score, 0.03, 0.96)


def _rollout_equity(state, samples, rng):
    board = list(state.get("community_cards", []) or [])
    cards = list(state.get("your_cards", []) or [])
    try:
        hero = [eval7.Card(card) for card in cards[:2]]
        community = [eval7.Card(card) for card in board[:5]]
    except Exception:
        active = _active_players(state)
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, active - 2))

    if len(hero) < 2:
        active = _active_players(state)
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, active - 2))

    active = _active_players(state)
    opponents = max(1, min(5, active - 1))
    dead = set(hero + community)
    deck = [card for card in FULL_DECK if card not in dead]
    board_needed = max(0, 5 - len(community))
    need = opponents * 2 + board_needed
    if len(deck) < need:
        return 0.0

    target = max(1, int(samples))
    deadline = time.perf_counter() + TIME_BUDGET_SECONDS
    wins = 0.0
    completed = 0
    for index in range(target):
        if (
            index >= MIN_ROLLOUT_SAMPLES
            and index % TIME_CHECK_INTERVAL == 0
            and time.perf_counter() >= deadline
        ):
            break
        draw = rng.sample(deck, need)
        offset = 0
        opp_hands = []
        for _ in range(opponents):
            opp_hands.append([draw[offset], draw[offset + 1]])
            offset += 2
        runout = list(community)
        if board_needed:
            runout.extend(draw[offset:offset + board_needed])

        hero_score = eval7.evaluate(hero + runout)
        best_score = hero_score
        tied = 1
        for opp_hand in opp_hands:
            score = eval7.evaluate(opp_hand + runout)
            if score > best_score:
                best_score = score
                tied = 1
            elif score == best_score:
                tied += 1
        if hero_score == best_score:
            wins += 1.0 / tied
        completed += 1

    return wins / max(1, completed)


def _threshold_from_fraction(fraction):
    fraction = max(0.01, float(fraction))
    return fraction / (1.0 + 2.0 * fraction)


def _fraction_from_threshold(q):
    q = _clip(float(q), 0.03, 0.485)
    return q / max(0.05, 1.0 - 2.0 * q)


def _board_drawiness(state):
    board = list(state.get("community_cards", []) or [])
    cards = list(state.get("your_cards", []) or [])
    all_cards = board + cards
    if len(board) < 3:
        return 0.10
    suits = {}
    ranks = set()
    for card in all_cards:
        text = str(card)
        if len(text) > 1:
            suits[text[1]] = suits.get(text[1], 0) + 1
        rank = _rank_value(text)
        ranks.add(rank)
        if rank == 14:
            ranks.add(1)
    flush_draw = 0.20 if max(suits.values(), default=0) >= 4 and len(board) < 5 else 0.0
    straight_draw = 0.0
    for start in range(1, 11):
        if len(ranks & {start, start + 1, start + 2, start + 3, start + 4}) >= 4:
            straight_draw = 0.16
            break
    paired = 0.08 if len({_rank_value(card) for card in board}) < len(board) else 0.0
    return _clip(flush_draw + straight_draw + paired, 0.0, 0.36)


def _opponent_equity_points(equity, state):
    active = _active_players(state)
    drawiness = _board_drawiness(state)
    base = _clip(1.0 - float(equity) + 0.35 * drawiness + 0.025 * max(0, active - 2), 0.08, 0.46)
    points = [
        base - 0.10,
        base - 0.05,
        base,
        base + 0.05 + 0.03 * drawiness,
        base + 0.10 + 0.04 * drawiness,
    ]
    return [_clip(point, 0.06, 0.485) for point in points]


def _candidate_fractions(equity, state, mode):
    values = list(STANDARD_FRACTIONS)
    for q in _opponent_equity_points(equity, state):
        indifferent = _fraction_from_threshold(q)
        if mode == "value":
            values.append(indifferent * 0.86)
            values.append(indifferent * 0.94)
        else:
            values.append(indifferent * 1.08)
            values.append(indifferent * 1.18)
    cap = 3.20 if float(equity) >= 0.86 else 1.75
    cleaned = sorted({_clip(round(value, 3), 0.18, cap) for value in values})
    return cleaned


def _call_probability_for_fraction(fraction, equity, state):
    threshold = _threshold_from_fraction(fraction)
    active = _active_players(state)
    bias = 0.010 + 0.010 * max(0, active - 2)
    temperature = 0.038 + 0.020 * _board_drawiness(state)
    probs = []
    for q in _opponent_equity_points(equity, state):
        probs.append(_sigmoid((q - threshold - bias) / max(0.012, temperature)))
    return sum(probs) / max(1, len(probs))


def _check_ev(equity, pot):
    return float(equity) * float(pot)


def _bet_ev(fraction, equity, state):
    pot = max(1.0, float(state.get("pot", 0) or 0))
    bet = pot * float(fraction)
    call_prob = _call_probability_for_fraction(fraction, equity, state)
    fold_prob = 1.0 - call_prob
    caller_penalty = 0.06 + 0.10 * call_prob + 0.04 * max(0, _active_players(state) - 2)
    equity_called = _clip(float(equity) - caller_penalty, 0.02, 0.98)
    ev = fold_prob * pot + call_prob * (equity_called * (pot + 2.0 * bet) - bet)
    return ev, call_prob, equity_called


def _best_bet_fraction(equity, state):
    pot = max(1.0, float(state.get("pot", 0) or 0))
    check_ev = _check_ev(equity, pot)
    drawiness = _board_drawiness(state)

    if equity >= 0.64:
        best = (check_ev, 0.0, 0.0)
        for fraction in _candidate_fractions(equity, state, "value"):
            ev, call_prob, equity_called = _bet_ev(fraction, equity, state)
            if equity < 0.78 and fraction > 1.05:
                ev -= pot * 0.08
            if equity_called < 0.52:
                ev -= pot * 0.18
            if ev > best[0] and call_prob > 0.10:
                best = (ev, fraction, call_prob)
        if best[1] > 0 and best[0] > check_ev + pot * 0.03:
            return best[1]
        if equity >= 0.82:
            return _clip(_fraction_from_threshold(max(_opponent_equity_points(equity, state))) * 0.90, 0.41, 1.75)
        return 0.0

    if equity <= 0.42 + 0.10 * drawiness:
        best = (-10**9, 0.0, 1.0)
        for fraction in _candidate_fractions(equity, state, "bluff"):
            if fraction > 1.27 and equity < 0.30:
                continue
            call_prob = _call_probability_for_fraction(fraction, equity, state)
            fold_prob = 1.0 - call_prob
            break_even = fraction / (1.0 + fraction)
            semi_equity = max(0.0, equity - 0.08)
            ev = fold_prob * pot - call_prob * fraction * pot + call_prob * semi_equity * (pot + 2 * fraction * pot)
            if fold_prob > break_even + 0.06 and ev > best[0]:
                best = (ev, fraction, call_prob)
        if best[1] > 0:
            return best[1]
    return 0.0


def _sanitize_action(state, action):
    act = str(action.get("action", "")).lower()
    can_check = bool(state.get("can_check", False))
    stack = max(0, int(state.get("your_stack", 0) or 0))
    invested = max(0, int(state.get("your_bet_this_street", 0) or 0))
    owed = max(0, int(state.get("amount_owed", 0) or 0))

    if stack <= 0:
        return {"action": "check"} if can_check else {"action": "fold"}
    if act == "fold":
        return {"action": "check"} if can_check else {"action": "fold"}
    if act == "check":
        return {"action": "check"} if can_check else {"action": "call"}
    if act == "call":
        return {"action": "check"} if can_check else {"action": "call"}
    if act == "all_in":
        return {"action": "all_in"}
    if act == "raise":
        amount = int(action.get("amount", 0) or 0)
        min_raise = max(0, int(state.get("min_raise_to", 0) or 0))
        max_total = invested + stack
        if max_total <= min_raise:
            if can_check:
                return {"action": "check"}
            return {"action": "all_in"} if stack <= owed else {"action": "call"}
        return {"action": "raise", "amount": max(min_raise, min(amount, max_total))}
    return {"action": "check"} if can_check else {"action": "fold"}


def _raise_to_fraction(state, fraction):
    pot = max(1, int(state.get("pot", 0) or 0))
    current_bet = max(0, int(state.get("current_bet", 0) or 0))
    min_raise = max(0, int(state.get("min_raise_to", 0) or 0))
    amount = max(min_raise, current_bet + int(pot * float(fraction)))
    return _sanitize_action(state, {"action": "raise", "amount": amount})


def _passive_action(state):
    return _sanitize_action(state, {"action": "check"} if state.get("can_check") else {"action": "call"})


def _facing_bet_fraction(equity, state):
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    active = _active_players(state)
    odds = owed / max(1.0, float(pot + owed))
    owed_pot = owed / max(1.0, float(pot))
    edge = float(equity) - odds
    call_edge = PARAMS["g_call_edge_base"] + PARAMS["g_call_edge_active"] * max(0, active - 2)
    call_edge -= PARAMS["g_pressure_discount"]
    if owed_pot >= 0.75:
        call_edge += 0.045
    elif owed_pot >= 0.45:
        call_edge += 0.025
    if edge < call_edge:
        if edge > PARAMS["g_thin_call_edge"] and owed_pot <= 0.45:
            return 0.0
        return -1.0
    if equity >= PARAMS["g_raise_equity"] and edge >= PARAMS["g_raise_edge"] and owed_pot <= PARAMS["g_raise_max_owed_pot"]:
        return PARAMS["g_min_raise"]
    return 0.0


def decide(state):
    if not isinstance(state, dict) or state.get("type") == "warmup":
        return {"action": "check"}

    equity = _rollout_equity(state, samples=int(PARAMS["samples"]), rng=_state_rng(state))
    if bool(state.get("can_check", False)):
        fraction = _best_bet_fraction(equity, state)
        if fraction > 0.0:
            return _raise_to_fraction(state, fraction)
        return _passive_action(state)

    fraction = _facing_bet_fraction(equity, state)
    if fraction > 0.0:
        return _raise_to_fraction(state, fraction)
    if fraction == 0.0:
        return _passive_action(state)
    return _sanitize_action(state, {"action": "fold"})
