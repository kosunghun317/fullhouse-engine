"""Mock competitor: runtime-light eval7 equity bot.

Imitates teams that ship a preflop chart plus Monte Carlo equity and pot odds.
"""

import random

import eval7


RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANK_ORDER)}
FULL_DECK = [eval7.Card(rank + suit) for rank in RANK_ORDER for suit in "shdc"]


def _hand_score(cards):
    if len(cards) < 2:
        return 0.2
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
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
    return max(0.05, min(0.92, score))


def _opponents(state):
    seat = state.get("seat_to_act")
    return max(1, sum(1 for p in state.get("players", []) if p.get("seat") != seat and not p.get("is_folded")))


def _equity(state, samples=160):
    board = state.get("community_cards", [])
    if not board:
        return max(0.04, _hand_score(state.get("your_cards", [])) - 0.055 * (_opponents(state) - 1))
    try:
        hero = [eval7.Card(card) for card in state.get("your_cards", [])]
        community = [eval7.Card(card) for card in board]
    except Exception:
        return 0.0
    dead = set(hero + community)
    deck = [card for card in FULL_DECK if card not in dead]
    opp_count = min(5, _opponents(state))
    needed = opp_count * 2 + max(0, 5 - len(community))
    if len(deck) < needed:
        return 0.0
    wins = 0.0
    trials = 0
    for _ in range(samples):
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
    return wins / max(1, trials)


def _raise_to_fraction(state, frac):
    pot = max(1, int(state.get("pot", 0) or 0))
    current = int(state.get("current_bet", 0) or 0)
    stack_total = int(state.get("your_stack", 0) or 0) + int(state.get("your_bet_this_street", 0) or 0)
    amount = min(stack_total, max(int(state.get("min_raise_to", 0) or 0), current + int(pot * frac)))
    return {"action": "raise", "amount": amount}


def decide(state):
    eq = _equity(state)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(1, int(state.get("pot", 0) or 0))
    odds = owed / max(1, pot + owed)
    if state.get("can_check"):
        if eq > 0.70:
            return _raise_to_fraction(state, 0.62)
        if eq > 0.48 and state.get("street") != "preflop" and random.random() < 0.35:
            return _raise_to_fraction(state, 0.45)
        return {"action": "check"}
    if eq > odds + 0.08:
        return {"action": "call"}
    if eq > 0.82 and owed < pot * 0.20:
        return _raise_to_fraction(state, 0.85)
    return {"action": "fold"}
