"""Standalone adaptive threshold-rollout submission candidate.

This bot keeps the tuned 1024-sample rollout equity core, then applies a
threshold-machine exploit only when the observed action history makes that
model plausible. It has no repo-local imports and no data-file dependency.
"""

import random
import time

import eval7


RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = tuple(eval7.Card(rank + suit) for rank in RANKS for suit in SUITS)

PARAMS = {
    "samples": 1024.0,
    "deep_samples": 1024.0,
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
    "g_raise_center": 0.9198701106956259,
    "g_raise_scale": 0.041550034474822965,
    "g_min_raise": 0.9369258774068039,
    "g_max_raise": 0.9369258774068039,
}

TIME_BUDGET_SECONDS = 1.50
MIN_ROLLOUT_SAMPLES = 128
TIME_CHECK_INTERVAL = 32
BIG_BLIND = 100
OFF_TREE_FRACTIONS = (0.27, 0.41, 0.63, 0.88, 1.27, 1.75, 2.60)


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
    active = 0
    for player in players:
        if not player.get("is_folded", False):
            active += 1
    return max(2, active)


def _opponents(state):
    return max(1, _active_players(state) - 1)


def _hero_identity(state):
    seat = state.get("seat_to_act")
    players = list(state.get("players", []) or [])
    for player in players:
        if player.get("seat") == seat:
            return seat, player.get("bot_id")
    return seat, None


def _state_rng(state, salt="adaptive_threshold_rollout"):
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
    cards = [str(card) for card in list(cards or [])]
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
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, _active_players(state) - 2))

    if len(hero) < 2:
        return max(0.03, _preflop_strength(cards) - 0.055 * max(0, _active_players(state) - 2))

    opponents = max(1, min(5, _active_players(state) - 1))
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


def _baseline_open_fraction(equity):
    if equity < PARAMS["f_min_equity"]:
        return 0.0
    weight = _sigmoid((float(equity) - PARAMS["f_center"]) / PARAMS["f_scale"])
    return PARAMS["f_min_raise"] + weight * (PARAMS["f_max_raise"] - PARAMS["f_min_raise"])


def _baseline_facing_fraction(equity, pot_odds, active_players, owed_pot_ratio):
    edge = float(equity) - float(pot_odds)
    active_extra = max(0, int(active_players) - 2)
    call_edge = PARAMS["g_call_edge_base"] + PARAMS["g_call_edge_active"] * active_extra
    call_edge -= PARAMS["g_pressure_discount"]
    if edge < call_edge:
        if edge > PARAMS["g_thin_call_edge"]:
            return 0.0
        return -1.0
    if (
        edge < PARAMS["g_raise_edge"]
        or equity < PARAMS["g_raise_equity"]
        or owed_pot_ratio > PARAMS["g_raise_max_owed_pot"]
    ):
        return 0.0
    weight = _sigmoid((equity - PARAMS["g_raise_center"]) / PARAMS["g_raise_scale"])
    return PARAMS["g_min_raise"] + weight * (PARAMS["g_max_raise"] - PARAMS["g_min_raise"])


def _board_drawiness(state):
    board = list(state.get("community_cards", []) or [])
    cards = list(state.get("your_cards", []) or [])
    if len(board) < 3:
        return 0.08
    suits = {}
    ranks = set()
    for card in board + cards:
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
    paired = 0.07 if len({_rank_value(card) for card in board}) < len(board) else 0.0
    high = 0.04 if any(_rank_value(card) >= 12 for card in board) else 0.0
    return _clip(flush_draw + straight_draw + paired + high, 0.0, 0.40)


def _threshold_from_fraction(fraction):
    fraction = max(0.01, float(fraction))
    return fraction / (1.0 + 2.0 * fraction)


def _fraction_from_threshold(q):
    q = _clip(float(q), 0.025, 0.485)
    return q / max(0.05, 1.0 - 2.0 * q)


def _mc_buffer(q, samples):
    q = _clip(float(q), 0.01, 0.99)
    sigma = pow(max(0.0, q * (1.0 - q)) / max(1.0, float(samples)), 0.5)
    return _clip(2.5 * sigma + 0.010, 0.018, 0.055)


def _opponent_equity_points(equity, state):
    active_extra = max(0, _active_players(state) - 2)
    drawiness = _board_drawiness(state)
    street = str(state.get("street", ""))
    street_bonus = 0.02 if street == "flop" else 0.00
    if street == "turn":
        street_bonus = -0.01
    elif street == "river":
        street_bonus = -0.025
    center = _clip(
        1.0 - float(equity) + 0.30 * drawiness + 0.025 * active_extra + street_bonus,
        0.07,
        0.47,
    )
    width = 0.050 + 0.040 * drawiness + 0.010 * active_extra
    return [
        _clip(center - 1.4 * width, 0.04, 0.485),
        _clip(center - 0.7 * width, 0.04, 0.485),
        _clip(center, 0.04, 0.485),
        _clip(center + 0.7 * width, 0.04, 0.485),
        _clip(center + 1.4 * width, 0.04, 0.485),
    ]


def _scan_pressure_responses(entries, hero_seat, hero_id):
    observations = 0.0
    score = 0.0
    last_hero_pressure = False
    last_pressure_amount = 0.0
    for entry in entries:
        action = str(entry.get("action", "")).lower()
        seat = entry.get("seat")
        bot_id = entry.get("bot_id")
        is_hero = (hero_id is not None and bot_id == hero_id) or seat == hero_seat
        amount = float(entry.get("amount") or 0)

        if is_hero:
            last_hero_pressure = action in {"raise", "all_in"}
            last_pressure_amount = amount if last_hero_pressure else 0.0
            continue

        if last_hero_pressure:
            if action == "fold":
                observations += 1.0
                score += 0.80 + min(0.25, last_pressure_amount / 40000.0)
                last_hero_pressure = False
            elif action == "call":
                observations += 1.0
                score += 0.35
                last_hero_pressure = False
            elif action in {"raise", "all_in"}:
                observations += 1.0
                score -= 0.70
                last_hero_pressure = False
            elif action == "check":
                observations += 0.25
                score -= 0.05
        elif action in {"raise", "all_in"}:
            observations += 0.30
            score -= 0.10
    return observations, score


def _mechanical_weight(state):
    hero_seat, hero_id = _hero_identity(state)
    observations = 0.0
    score = 0.0

    hand_entries = list(state.get("action_log", []) or [])[-80:]
    obs, val = _scan_pressure_responses(hand_entries, hero_seat, hero_id)
    observations += obs
    score += val

    match_entries = list(state.get("match_action_log", []) or [])[-160:]
    obs, val = _scan_pressure_responses(match_entries, hero_seat, hero_id)
    observations += obs
    score += val

    if observations < 2.0:
        base = 0.44
    else:
        base = 0.40 + 0.42 * _clip(score / max(1.0, observations), -1.0, 1.0)

    if _active_players(state) == 2:
        base += 0.06
    else:
        base -= 0.04 + 0.015 * max(0, _active_players(state) - 3)
    if str(state.get("street", "")) == "river":
        base += 0.05
    return _clip(base, 0.18, 0.86)


def _generic_call_probability(fraction, equity, state):
    active_extra = max(0, _active_players(state) - 2)
    drawiness = _board_drawiness(state)
    mean_q = sum(_opponent_equity_points(equity, state)) / 5.0
    size_pressure = 0.17 * float(fraction) + 0.025 * active_extra
    stickiness = 0.19 + 0.10 * drawiness
    return _sigmoid((mean_q - size_pressure - 0.12) / max(0.08, stickiness))


def _mechanical_call_probability(fraction, equity, state, weight):
    threshold = _threshold_from_fraction(fraction)
    active_extra = max(0, _active_players(state) - 2)
    bias = 0.010 + 0.012 * active_extra
    temperature = 0.030 + 0.035 * (1.0 - weight) + 0.020 * _board_drawiness(state)
    threshold_probs = []
    for q in _opponent_equity_points(equity, state):
        threshold_probs.append(_sigmoid((q - threshold - bias) / max(0.015, temperature)))
    threshold_call = sum(threshold_probs) / max(1, len(threshold_probs))
    generic_call = _generic_call_probability(fraction, equity, state)
    individual = weight * threshold_call + (1.0 - weight) * generic_call
    return _clip(1.0 - pow(1.0 - _clip(individual, 0.0, 1.0), _opponents(state)), 0.0, 1.0)


def _effective_fraction_cap(state):
    pot = max(1.0, float(state.get("pot", 0) or 0))
    stack = max(0.0, float(state.get("your_stack", 0) or 0))
    players = list(state.get("players", []) or [])
    hero_seat, _ = _hero_identity(state)
    opp_stacks = [
        float(player.get("stack", 0) or 0)
        for player in players
        if player.get("seat") != hero_seat and not player.get("is_folded", False)
    ]
    effective = min([stack] + opp_stacks) if opp_stacks else stack
    return _clip(effective / pot, 0.08, 3.20)


def _candidate_fractions(equity, state, mode):
    cap = _effective_fraction_cap(state)
    values = list(OFF_TREE_FRACTIONS)
    base = _baseline_open_fraction(equity)
    if base > 0.0:
        values.extend([base * 0.85, base, base * 1.15])

    samples = PARAMS["samples"]
    for q in _opponent_equity_points(equity, state):
        buffer = _mc_buffer(q, samples)
        if mode == "value":
            target = q - buffer
            values.append(_fraction_from_threshold(target) * 0.92)
            values.append(_fraction_from_threshold(target) * 0.98)
        else:
            target = q + buffer
            values.append(_fraction_from_threshold(target) * 1.03)
            values.append(_fraction_from_threshold(target) * 1.10)

    cleaned = []
    for value in values:
        value = _clip(float(value), 0.16, cap)
        if value >= 0.12:
            cleaned.append(round(value, 3))
    return sorted(set(cleaned))


def _check_ev(equity, pot):
    return float(equity) * float(pot)


def _bet_ev(fraction, equity, state, weight):
    pot = max(1.0, float(state.get("pot", 0) or 0))
    bet = pot * float(fraction)
    call_prob = _mechanical_call_probability(fraction, equity, state, weight)
    fold_prob = 1.0 - call_prob
    active_extra = max(0, _active_players(state) - 2)
    caller_penalty = (
        0.045
        + 0.105 * call_prob
        + 0.025 * active_extra
        + 0.030 * max(0.0, float(fraction) - 1.0)
    )
    equity_called = _clip(float(equity) - caller_penalty, 0.02, 0.98)
    ev = fold_prob * pot + call_prob * (equity_called * (pot + 2.0 * bet) - bet)
    return ev, call_prob, equity_called


def _best_threshold_bet_fraction(equity, state, weight):
    pot = max(1.0, float(state.get("pot", 0) or 0))
    check_ev = _check_ev(equity, pot)
    baseline = _baseline_open_fraction(equity)
    best_fraction = baseline
    best_ev = check_ev
    if baseline > 0.0:
        best_ev = max(best_ev, _bet_ev(baseline, equity, state, weight)[0])

    drawiness = _board_drawiness(state)
    value_mode = equity >= 0.62
    bluff_mode = equity <= 0.39 + 0.11 * drawiness and weight >= 0.52
    if not value_mode and not bluff_mode:
        return baseline

    mode = "value" if value_mode else "bluff"
    for fraction in _candidate_fractions(equity, state, mode):
        if equity < 0.75 and fraction > 1.20:
            continue
        ev, call_prob, equity_called = _bet_ev(fraction, equity, state, weight)
        if value_mode:
            if equity_called < 0.505 or call_prob < 0.08:
                continue
            if fraction > 1.50 and equity < 0.84:
                ev -= pot * (0.08 + 0.10 * (1.0 - weight))
            hurdle = pot * (0.020 + 0.070 * (1.0 - weight))
            if ev > best_ev + hurdle:
                best_ev = ev
                best_fraction = fraction
        else:
            fold_prob = 1.0 - call_prob
            break_even = fraction / (1.0 + fraction)
            margin = 0.035 + 0.100 * (1.0 - weight) + 0.025 * max(0, _active_players(state) - 2)
            semi_equity = max(0.0, equity - 0.075)
            bluff_ev = fold_prob * pot - call_prob * fraction * pot
            bluff_ev += call_prob * semi_equity * (pot + 2.0 * fraction * pot)
            if fold_prob > break_even + margin and bluff_ev > max(best_ev, check_ev) + pot * 0.015:
                best_ev = bluff_ev
                best_fraction = fraction

    if best_fraction > 0.0 and best_ev > check_ev + pot * 0.010:
        return best_fraction
    return 0.0


def _facing_threshold_fraction(equity, state, weight):
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    if owed <= 0:
        return 0.0

    active = _active_players(state)
    odds = owed / max(1.0, float(pot + owed))
    owed_pot = owed / max(1.0, float(pot))
    baseline = _baseline_facing_fraction(equity, odds, active, owed_pot)

    active_extra = max(0, active - 2)
    edge = float(equity) - odds
    large_bet_tax = 0.035 if owed_pot >= 0.45 else 0.0
    if owed_pot >= 0.85:
        large_bet_tax = 0.070
    river_tax = 0.035 if str(state.get("street", "")) == "river" else 0.0
    call_edge = (
        PARAMS["g_call_edge_base"]
        + PARAMS["g_call_edge_active"] * active_extra
        - PARAMS["g_pressure_discount"]
        + weight * (large_bet_tax + river_tax)
    )

    if edge < call_edge:
        if baseline == 0.0 and owed_pot <= 0.33 and edge > PARAMS["g_thin_call_edge"] + 0.010:
            return 0.0
        return -1.0

    if equity >= 0.90 and edge >= PARAMS["g_raise_edge"] and owed_pot <= 0.30:
        target_q = max(_opponent_equity_points(equity, state))
        fraction = _fraction_from_threshold(max(0.08, target_q - _mc_buffer(target_q, PARAMS["samples"])))
        return _clip(fraction * 0.92, PARAMS["g_min_raise"], _effective_fraction_cap(state))

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


def decide(state):
    if not isinstance(state, dict) or state.get("type") == "warmup":
        return {"action": "check"}

    equity = _rollout_equity(state, samples=int(PARAMS["samples"]), rng=_state_rng(state))
    weight = _mechanical_weight(state)

    if bool(state.get("can_check", False)):
        fraction = _best_threshold_bet_fraction(equity, state, weight)
        if fraction > 0.0:
            return _raise_to_fraction(state, fraction)
        return _passive_action(state)

    fraction = _facing_threshold_fraction(equity, state, weight)
    if fraction > 0.0:
        return _raise_to_fraction(state, fraction)
    if fraction == 0.0:
        return _passive_action(state)
    return _sanitize_action(state, {"action": "fold"})
