"""Generate an explicit 169-class preflop score table.

This is a development tool. It is intentionally outside the submitted bot path.
Scores are normalized from deterministic heads-up all-in equity samples against
one random hand. The bot then adjusts those scores by position, table profile,
and pot state.
"""

import argparse
import json
import random

import eval7


RANK_ORDER = "23456789TJQKA"
RANKS_DESC = list(reversed(RANK_ORDER))
SUITS = "shdc"
FULL_DECK = [eval7.Card(rank + suit) for rank in RANK_ORDER for suit in SUITS]


def hand_classes():
    for i, high in enumerate(RANKS_DESC):
        for j, low in enumerate(RANKS_DESC):
            if i == j:
                yield high + low
            elif i < j:
                yield high + low + "s"
                yield high + low + "o"


def representative_cards(hand_class):
    if len(hand_class) == 2:
        return [eval7.Card(hand_class[0] + "s"), eval7.Card(hand_class[1] + "h")]
    high, low, suitedness = hand_class[0], hand_class[1], hand_class[2]
    if suitedness == "s":
        return [eval7.Card(high + "s"), eval7.Card(low + "s")]
    return [eval7.Card(high + "s"), eval7.Card(low + "h")]


def estimate_equity(hero, iterations, rng):
    dead = set(hero)
    deck = [card for card in FULL_DECK if card not in dead]
    wins = 0.0
    for _ in range(iterations):
        sample = rng.sample(deck, 7)
        board = sample[:5]
        villain = sample[5:]
        hero_score = eval7.evaluate(hero + board)
        villain_score = eval7.evaluate(villain + board)
        if hero_score > villain_score:
            wins += 1.0
        elif hero_score == villain_score:
            wins += 0.5
    return wins / iterations


def build_table(iterations, seed):
    rng = random.Random(seed)
    equities = {}
    for hand_class in hand_classes():
        equities[hand_class] = estimate_equity(representative_cards(hand_class), iterations, rng)

    lo = min(equities.values())
    hi = max(equities.values())
    scores = {
        hand_class: int(round(8 + (equity - lo) / (hi - lo) * 92))
        for hand_class, equity in equities.items()
    }
    return equities, scores


def format_python(scores):
    lines = ["PREFLOP_SCORE_TABLE = {"]
    for hand_class in hand_classes():
        lines.append(f'    "{hand_class}": {scores[hand_class]},')
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a sampled 169-class preflop score table")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    equities, scores = build_table(args.iterations, args.seed)
    if args.json:
        print(json.dumps({
            "iterations": args.iterations,
            "seed": args.seed,
            "equities": {k: round(v, 5) for k, v in equities.items()},
            "scores": scores,
        }, indent=2, sort_keys=True))
        return
    print(format_python(scores))


if __name__ == "__main__":
    main()
