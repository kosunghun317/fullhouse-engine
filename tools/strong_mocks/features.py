"""Feature extraction and abstraction for benchmark-only strong mock bots."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANK_ORDER)}
STREETS = ("preflop", "flop", "turn", "river")

FEATURE_NAMES = (
    "bias",
    "is_preflop",
    "is_flop",
    "is_turn",
    "is_river",
    "owed_ratio",
    "pot_norm",
    "stack_norm",
    "spr_norm",
    "can_check",
    "active_players_norm",
    "position_norm",
    "hole_high",
    "hole_low",
    "hole_pair",
    "hole_suited",
    "hole_gap_norm",
    "preflop_strength",
    "board_wet",
    "board_paired",
    "board_monotone",
    "board_ace_or_king",
    "made_pairish",
    "flush_draw",
    "straight_draw",
    "last_raise",
    "last_call",
    "last_check",
    "last_fold",
    "raises_this_street_norm",
    "allins_this_hand_norm",
    "hero_stack_share",
)


def rank_value(card: str) -> int:
    return RANK_VALUE.get(str(card)[0], 2)


def _street_one_hot(street: str) -> list[float]:
    return [1.0 if street == item else 0.0 for item in STREETS]


def _active_players(state: dict) -> int:
    return sum(1 for player in state.get("players", []) if not player.get("is_folded"))


def _position_norm(state: dict) -> float:
    players = state.get("players", [])
    n = max(1, len(players))
    seat = int(state.get("seat_to_act", 0) or 0)
    if n <= 2:
        return 0.5 if state.get("street") == "preflop" else 1.0

    small_blind = None
    for action in state.get("action_log", []):
        if action.get("action") == "small_blind":
            small_blind = int(action.get("seat", 0) or 0)
            break
    if small_blind is None:
        return seat / max(1, n - 1)

    dealer = (small_blind - 1) % n
    relative = (seat - dealer) % n
    return relative / max(1, n - 1)


def hole_features(cards: Iterable[str]) -> tuple[float, float, float, float, float, float]:
    cards = list(cards or [])
    if len(cards) < 2:
        return 0.2, 0.1, 0.0, 0.0, 1.0, 0.12
    values = sorted((rank_value(card) for card in cards[:2]), reverse=True)
    high, low = values
    pair = float(high == low)
    suited = float(cards[0][1] == cards[1][1]) if len(cards[0]) > 1 and len(cards[1]) > 1 else 0.0
    gap = abs(high - low)
    strength = 0.40 * (high / 14.0) + 0.22 * (low / 14.0)
    if pair:
        strength += 0.30 + high / 120.0
    if suited:
        strength += 0.06
    if gap <= 1:
        strength += 0.05
    elif gap >= 5:
        strength -= 0.08
    if high == 14:
        strength += 0.06
    return (
        high / 14.0,
        low / 14.0,
        pair,
        suited,
        min(1.0, gap / 12.0),
        float(max(0.02, min(0.98, strength))),
    )


def board_features(board: Iterable[str]) -> tuple[float, float, float, float]:
    board = list(board or [])
    if len(board) < 3:
        return 0.0, 0.0, 0.0, 0.0
    ranks = [rank_value(card) for card in board]
    suits = {}
    for card in board:
        suits[card[1]] = suits.get(card[1], 0) + 1
    paired = float(len(set(ranks)) < len(ranks))
    monotone = float(max(suits.values(), default=0) >= 3)
    ace_or_king = float(any(rank >= 13 for rank in ranks))
    unique = sorted(set(ranks))
    connected_edges = sum(1 for i in range(1, len(unique)) if unique[i] - unique[i - 1] <= 2)
    wet = float(monotone or connected_edges >= 2)
    return wet, paired, monotone, ace_or_king


def draw_features(cards: Iterable[str], board: Iterable[str]) -> tuple[float, float, float]:
    cards = list(cards or [])
    board = list(board or [])
    all_cards = cards + board
    ranks = [rank_value(card) for card in all_cards]
    rank_counts = {}
    suits = {}
    for card in all_cards:
        rank_counts[rank_value(card)] = rank_counts.get(rank_value(card), 0) + 1
        if len(card) > 1:
            suits[card[1]] = suits.get(card[1], 0) + 1
    pairish = float(any(count >= 2 for count in rank_counts.values()))
    flush_draw = float(len(board) < 5 and max(suits.values(), default=0) >= 4)

    vals = set(ranks)
    if 14 in vals:
        vals.add(1)
    straight_draw = 0.0
    for start in range(1, 11):
        window = {start, start + 1, start + 2, start + 3, start + 4}
        if len(vals & window) >= 4:
            straight_draw = 1.0
            break
    return pairish, flush_draw, straight_draw


def recent_action_features(state: dict) -> tuple[float, float, float, float, float, float]:
    log = state.get("action_log", [])
    last = log[-1].get("action") if log else None
    raises = sum(1 for action in log if action.get("action") == "raise")
    allins = sum(1 for action in log if action.get("action") == "all_in")
    return (
        float(last in ("raise", "all_in")),
        float(last == "call"),
        float(last == "check"),
        float(last == "fold"),
        min(1.0, raises / 4.0),
        min(1.0, allins / 3.0),
    )


def extract_features(state: dict) -> np.ndarray:
    street = state.get("street", "preflop")
    owed = max(0, int(state.get("amount_owed", 0) or 0))
    pot = max(1, int(state.get("pot", 0) or 0))
    stack = max(1, int(state.get("your_stack", 0) or 0))
    active = max(1, _active_players(state))
    total_chips = stack + sum(max(0, int(p.get("stack", 0) or 0)) for p in state.get("players", []))

    high, low, pair, suited, gap, strength = hole_features(state.get("your_cards", []))
    wet, paired, monotone, ace_or_king = board_features(state.get("community_cards", []))
    made_pairish, flush_draw, straight_draw = draw_features(
        state.get("your_cards", []),
        state.get("community_cards", []),
    )
    last_raise, last_call, last_check, last_fold, raises_norm, allins_norm = recent_action_features(state)

    values = [
        1.0,
        *_street_one_hot(street),
        owed / max(1, pot + owed),
        min(1.0, math.log1p(pot) / math.log1p(60000)),
        min(1.0, stack / 30000.0),
        min(1.0, stack / pot / 20.0),
        float(bool(state.get("can_check"))),
        min(1.0, active / 6.0),
        _position_norm(state),
        high,
        low,
        pair,
        suited,
        gap,
        strength,
        wet,
        paired,
        monotone,
        ace_or_king,
        made_pairish,
        flush_draw,
        straight_draw,
        last_raise,
        last_call,
        last_check,
        last_fold,
        raises_norm,
        allins_norm,
        min(1.0, stack / max(1, total_chips)),
    ]
    return np.asarray(values, dtype=np.float32)


def bucket_id(state: dict, bucket_count: int = 4096) -> int:
    features = extract_features(state)
    street = int(np.argmax(features[1:5]))
    strength = int(min(7, max(0, features[17] * 8)))
    wet = int(features[18] > 0.5)
    paired = int(features[19] > 0.5)
    owed = int(min(3, max(0, features[5] * 4)))
    active = int(min(5, max(0, features[10] * 6)))
    position = int(min(3, max(0, features[11] * 4)))
    raises = int(min(3, max(0, features[29] * 4)))
    raw = (((((((street * 8 + strength) * 2 + wet) * 2 + paired) * 4 + owed) * 6 + active) * 4 + position) * 4 + raises)
    return int(raw % bucket_count)
